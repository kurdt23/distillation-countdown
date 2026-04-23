from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import infer_task_fields, read_records
from countdown_distill.expressions import (
    ValidationResult,
    extract_expression_candidate,
    extract_expression_candidates,
    normalize_symbols,
    validate_expression,
    validation_quality_key,
)
from countdown_distill.prompting import build_messages, build_repair_messages, render_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and write a Kaggle submission CSV.")
    parser.add_argument("--model-path", type=Path, required=True, help="Adapter dir or merged model dir.")
    parser.add_argument("--input-path", type=Path, required=True, help="CSV / Parquet / JSONL with test or validation rows.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--base-model-id", default="google/gemma-3-1b-it", help="Required when model-path points to a LoRA adapter.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of candidates to generate per prompt.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for deterministic decoding.")
    parser.add_argument("--repair-attempts", type=int, default=0, help="How many model-only repair rounds to run for invalid outputs.")
    parser.add_argument("--repair-temperature", type=float, default=0.4)
    parser.add_argument("--repair-top-p", type=float, default=0.95)
    parser.add_argument("--repair-num-candidates", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep extracted expressions even when local validation fails.")
    parser.add_argument(
        "--allow-empty-equation",
        action="store_true",
        help="Allow empty equations for invalid rows. Off by default because Kaggle rejects null values.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on how many rows to process.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N batches.")
    parser.add_argument("--debug-jsonl", type=Path, default=None, help="Optional debug dump with candidates and chosen outputs.")
    parser.add_argument(
        "--restrict-output-charset",
        action="store_true",
        help="Restrict generation to characters that can appear in arithmetic expressions.",
    )
    return parser.parse_args()


class AllowedExpressionLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]) -> None:
        self.allowed_token_ids = sorted(set(allowed_token_ids))
        self._mask_cache: dict[tuple[str, int], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cache_key = (str(scores.device), scores.shape[-1])
        mask = self._mask_cache.get(cache_key)
        if mask is None:
            mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device, dtype=scores.dtype)
            mask[self.allowed_token_ids] = 0
            self._mask_cache[cache_key] = mask
        return scores + mask


def build_allowed_token_ids(tokenizer: Any) -> list[int]:
    allowed_chars = set("0123456789+-*/() \n\t")
    allowed_token_ids: list[int] = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)

    for token_id in range(vocab_size):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
        piece = normalize_symbols(piece)
        if piece and all(char in allowed_chars for char in piece):
            allowed_token_ids.append(token_id)

    for special_token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_token_id is not None:
            allowed_token_ids.append(int(special_token_id))

    return sorted(set(allowed_token_ids))


def build_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    if platform.system() == "Windows":
        raise RuntimeError("bitsandbytes 4-bit inference is usually unreliable on native Windows. Use WSL2/Linux or drop --load-in-4bit.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    quantization_config = build_quantization_config(args.load_in_4bit)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    adapter_config_path = args.model_path / "adapter_config.json"
    if adapter_config_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path if (args.model_path / "tokenizer_config.json").exists() else args.base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def reason_to_issue_text(result: ValidationResult) -> str:
    mapping = {
        "empty_expression": "the answer was empty",
        "must_use_all_numbers": "not all provided numbers were used exactly once",
        "wrong_target": "the expression does not evaluate to the target",
        "used_unavailable_number": "the expression uses numbers outside the provided multiset",
    }
    return mapping.get(result.reason or "", result.reason or "the expression is invalid")


def generate_text_groups(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_candidates: int,
    num_beams: int,
    allowed_token_ids: list[int] | None = None,
) -> list[list[str]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    if hasattr(model, "device"):
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    do_sample = temperature > 0
    effective_num_candidates = max(1, num_candidates)
    effective_num_beams = max(num_beams, effective_num_candidates if not do_sample else 1)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": effective_num_candidates,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["num_beams"] = effective_num_beams
    if allowed_token_ids is not None:
        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [AllowedExpressionLogitsProcessor(allowed_token_ids)]
        )

    with torch.no_grad():
        outputs = model.generate(**encoded, **generation_kwargs)

    generated_tokens = outputs[:, encoded["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    grouped = []
    for index in range(len(prompts)):
        start = index * effective_num_candidates
        stop = start + effective_num_candidates
        grouped.append(texts[start:stop])
    return grouped


def validate_candidates(candidate_texts: list[str], nums: list[int], target: int) -> list[dict[str, Any]]:
    validated = []
    for index, generated_text in enumerate(candidate_texts):
        extracted_candidates = extract_expression_candidates(generated_text)
        if not extracted_candidates:
            extracted_candidates = [extract_expression_candidate(generated_text)]

        best_extracted = ""
        best_result = validate_expression("", nums, target)
        for extracted in extracted_candidates:
            result = validate_expression(extracted, nums, target)
            if validation_quality_key(result, nums, target) > validation_quality_key(best_result, nums, target):
                best_extracted = extracted
                best_result = result

        validated.append(
            {
                "index": index,
                "generated_text": generated_text,
                "extracted": best_extracted,
                "validation": best_result,
                "all_extracted": extracted_candidates,
            }
        )
    return validated


def choose_best_candidate(candidates: list[dict[str, Any]], nums: list[int], target: int) -> dict[str, Any] | None:
    valid_candidates = [candidate for candidate in candidates if candidate["validation"].is_valid]
    if valid_candidates:
        frequency = Counter(candidate["validation"].normalized_expression for candidate in valid_candidates)
        return max(
            valid_candidates,
            key=lambda candidate: (
                frequency[candidate["validation"].normalized_expression],
                validation_quality_key(candidate["validation"], nums, target),
                -candidate["index"],
            ),
        )

    if candidates:
        return max(
            candidates,
            key=lambda candidate: (
                validation_quality_key(candidate["validation"], nums, target),
                -candidate["index"],
            ),
        )
    return None


def build_fallback_equation(nums: list[int]) -> str:
    if not nums:
        return "0"
    if len(nums) == 1:
        return str(nums[0])
    return " + ".join(str(num) for num in nums)


def main() -> None:
    args = parse_args()
    records = read_records(args.input_path)
    if args.limit > 0:
        records = records[: min(args.limit, len(records))]
    model, tokenizer = load_model_and_tokenizer(args)
    allowed_token_ids = build_allowed_token_ids(tokenizer) if args.restrict_output_charset else None

    rows: list[dict[str, Any]] = []
    valid_predictions = 0
    debug_rows: list[dict[str, Any]] = []
    total_batches = (len(records) + args.batch_size - 1) // args.batch_size if records else 0

    for batch_index, start in enumerate(range(0, len(records), args.batch_size), start=1):
        batch = records[start : start + args.batch_size]
        prompts = []
        task_fields = []
        ids = []

        for index, record in enumerate(batch):
            row_id = record.get("id", start + index)
            target, nums = infer_task_fields(record)
            ids.append(row_id)
            task_fields.append((target, nums))
            prompts.append(render_chat_prompt(tokenizer, build_messages(nums, target), add_generation_prompt=True))

        grouped_texts = generate_text_groups(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_candidates=args.num_candidates,
            num_beams=args.num_beams,
            allowed_token_ids=allowed_token_ids,
        )

        for row_id, (target, nums), generated_texts in zip(ids, task_fields, grouped_texts):
            candidates = validate_candidates(generated_texts, nums, target)
            chosen = choose_best_candidate(candidates, nums, target)

            repair_trace: list[dict[str, Any]] = []
            for _ in range(args.repair_attempts):
                if chosen is not None and chosen["validation"].is_valid:
                    break
                previous_expression = ""
                previous_issue = "the expression is invalid"
                if chosen is not None:
                    previous_expression = chosen["extracted"]
                    previous_issue = reason_to_issue_text(chosen["validation"])

                repair_prompt = render_chat_prompt(
                    tokenizer,
                    build_repair_messages(nums, target, previous_expression, previous_issue),
                    add_generation_prompt=True,
                )
                repair_groups = generate_text_groups(
                    model,
                    tokenizer,
                    [repair_prompt],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.repair_temperature,
                    top_p=args.repair_top_p,
                    num_candidates=args.repair_num_candidates,
                    num_beams=1,
                    allowed_token_ids=allowed_token_ids,
                )
                repair_candidates = validate_candidates(repair_groups[0], nums, target)
                repair_chosen = choose_best_candidate(repair_candidates, nums, target)
                repair_trace.append(
                    {
                        "previous_expression": previous_expression,
                        "previous_issue": previous_issue,
                        "candidates": [
                            {
                                "generated_text": candidate["generated_text"],
                                "extracted": candidate["extracted"],
                                "all_extracted": candidate.get("all_extracted", []),
                                "is_valid": candidate["validation"].is_valid,
                                "reason": candidate["validation"].reason,
                                "normalized_expression": candidate["validation"].normalized_expression,
                            }
                            for candidate in repair_candidates
                        ],
                    }
                )
                if repair_chosen is not None:
                    chosen = repair_chosen

            validation = chosen["validation"] if chosen is not None else validate_expression("", nums, target)
            equation = validation.normalized_expression if validation.is_valid else (chosen["extracted"] if chosen is not None else "")
            if validation.is_valid:
                valid_predictions += 1
            if not validation.is_valid and not args.keep_invalid:
                equation = ""
            if not equation and not validation.is_valid and not args.allow_empty_equation:
                equation = build_fallback_equation(nums)
            rows.append({"id": row_id, "equation": equation})
            if args.debug_jsonl is not None:
                debug_rows.append(
                    {
                        "id": row_id,
                        "target": target,
                        "nums": nums,
                        "chosen_equation": equation,
                        "chosen_valid": validation.is_valid,
                        "chosen_reason": validation.reason,
                        "initial_candidates": [
                            {
                                "generated_text": candidate["generated_text"],
                                "extracted": candidate["extracted"],
                                "all_extracted": candidate.get("all_extracted", []),
                                "is_valid": candidate["validation"].is_valid,
                                "reason": candidate["validation"].reason,
                                "normalized_expression": candidate["validation"].normalized_expression,
                            }
                            for candidate in candidates
                        ],
                        "repair_trace": repair_trace,
                    }
                )

        if args.progress_every > 0 and (batch_index % args.progress_every == 0 or batch_index == total_batches):
            processed_rows = min(start + len(batch), len(records))
            print(
                f"Processed batch {batch_index}/{total_batches} "
                f"({processed_rows}/{len(records)} rows), locally valid so far: {valid_predictions}"
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    if args.debug_jsonl is not None:
        args.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.debug_jsonl.open("w", encoding="utf-8") as handle:
            for row in debug_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Rows: {len(rows)}")
    print(f"Locally valid predictions: {valid_predictions}")
    print(f"Wrote {args.output_csv}")
    if args.debug_jsonl is not None:
        print(f"Wrote {args.debug_jsonl}")


if __name__ == "__main__":
    main()
