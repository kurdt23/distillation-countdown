from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import ensure_parent_dir
from countdown_distill.expressions import extract_expression_candidates, validate_expression, validation_quality_key
from countdown_distill.prompting import build_messages, build_training_messages, render_chat_prompt
from countdown_distill.solver import solve_countdown


DATASET_ID = "HuggingFaceTB/Countdown-Task-GOLD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional teacher traces for Countdown tasks.")
    parser.add_argument("--backend", choices=["openai", "transformers"], required=True)
    parser.add_argument("--model-id", required=True, help="Teacher model id.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the source split before applying offset/limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=1, help="How many teacher samples to draw per prompt.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for deterministic decoding.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N processed rows.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repair-with-solver", action="store_true")
    return parser.parse_args()


def build_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    if platform.system() == "Windows":
        raise RuntimeError("bitsandbytes 4-bit loading is usually unreliable on native Windows. Use WSL2/Linux or drop --load-in-4bit.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
def build_output_record(row_index: int, nums: list[int], target: int, raw_texts: list[str], args: argparse.Namespace) -> dict[str, Any]:
    best_raw_text = raw_texts[0] if raw_texts else ""
    best_extracted = ""
    best_validation = validate_expression("", nums, target)
    candidate_rows: list[dict[str, Any]] = []

    for raw_text in raw_texts:
        extracted_candidates = extract_expression_candidates(raw_text)
        if not extracted_candidates:
            extracted_candidates = [""]

        local_best_extracted = ""
        local_best_validation = validate_expression("", nums, target)
        for extracted in extracted_candidates:
            validated = validate_expression(extracted, nums, target)
            if validation_quality_key(validated, nums, target) > validation_quality_key(local_best_validation, nums, target):
                local_best_extracted = extracted
                local_best_validation = validated

        candidate_rows.append(
            {
                "raw_response": raw_text,
                "candidate_expressions": extracted_candidates,
                "chosen_expression": local_best_extracted,
                "is_valid": local_best_validation.is_valid,
                "reason": local_best_validation.reason,
                "normalized_expression": local_best_validation.normalized_expression,
            }
        )
        if validation_quality_key(local_best_validation, nums, target) > validation_quality_key(best_validation, nums, target):
            best_raw_text = raw_text
            best_extracted = local_best_extracted
            best_validation = local_best_validation

    label_expression = best_validation.normalized_expression if best_validation.is_valid else ""
    repaired = False
    if not label_expression and args.repair_with_solver:
        solver_expression = solve_countdown(nums, target)
        solver_validated = validate_expression(solver_expression, nums, target)
        if solver_validated.is_valid:
            label_expression = solver_validated.normalized_expression
            repaired = True

    record = {
        "example_id": f"{args.dataset_config}:{args.offset + row_index}",
        "source": f"teacher::{args.model_id}",
        "teacher_model": args.model_id,
        "target": int(target),
        "nums": [int(num) for num in nums],
        "raw_response": best_raw_text,
        "raw_responses": raw_texts,
        "extracted_expression": best_extracted,
        "label_expression": label_expression,
        "valid": bool(label_expression),
        "repaired_with_solver": repaired,
        "candidates": candidate_rows,
    }
    if label_expression:
        record["messages"] = build_training_messages(nums, target, label_expression)
    return record


def generate_openai_examples(dataset: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Environment variable {args.api_key_env} is not set.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    generated: list[dict[str, Any]] = []

    for row_index, row in enumerate(dataset):
        nums = [int(num) for num in row["nums"]]
        target = int(row["target"])
        messages = build_messages(nums, target)
        response = client.chat.completions.create(
            model=args.model_id,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            n=max(1, args.num_candidates),
        )
        raw_texts = [(choice.message.content or "") for choice in response.choices]
        generated.append(build_output_record(row_index, nums, target, raw_texts, args))
        if args.progress_every > 0 and ((row_index + 1) % args.progress_every == 0 or row_index + 1 == len(dataset)):
            valid_count = sum(1 for record in generated if record["valid"])
            print(f"Processed {row_index + 1}/{len(dataset)} rows, valid so far: {valid_count}")

    return generated


def generate_transformers_examples(dataset: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = build_quantization_config(args.load_in_4bit)
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    generated: list[dict[str, Any]] = []
    rows = list(dataset)
    do_sample = args.temperature > 0
    effective_num_candidates = max(1, args.num_candidates)
    effective_num_beams = max(args.num_beams, effective_num_candidates if not do_sample else 1)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [render_chat_prompt(tokenizer, build_messages(row["nums"], row["target"]), add_generation_prompt=True) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        if hasattr(model, "device"):
            encoded = {key: value.to(model.device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "num_return_sequences": effective_num_candidates,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(args.temperature, 1e-5)
            generation_kwargs["top_p"] = args.top_p
        else:
            generation_kwargs["num_beams"] = effective_num_beams

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)

        generated_tokens = outputs[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        grouped_texts: list[list[str]] = []
        for batch_index in range(len(batch)):
            chunk_start = batch_index * effective_num_candidates
            chunk_end = chunk_start + effective_num_candidates
            grouped_texts.append(texts[chunk_start:chunk_end])

        for relative_index, (row, raw_texts) in enumerate(zip(batch, grouped_texts)):
            global_index = start + relative_index
            nums = [int(num) for num in row["nums"]]
            target = int(row["target"])
            generated.append(build_output_record(global_index, nums, target, raw_texts, args))

        if args.progress_every > 0:
            processed = min(start + len(batch), len(rows))
            if processed % args.progress_every == 0 or processed == len(rows):
                valid_count = sum(1 for record in generated if record["valid"])
                print(f"Processed {processed}/{len(rows)} rows, valid so far: {valid_count}")

    return generated


def main() -> None:
    args = parse_args()

    dataset = load_dataset(DATASET_ID, args.dataset_config, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.offset > 0:
        dataset = dataset.select(range(args.offset, len(dataset)))
    dataset = dataset.select(range(min(args.limit, len(dataset))))

    if args.backend == "openai":
        generated = generate_openai_examples(dataset, args)
    else:
        generated = generate_transformers_examples(dataset, args)

    ensure_parent_dir(args.output_jsonl)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid_count = sum(1 for row in generated if row["valid"])
    print(json.dumps({"rows": len(generated), "valid_rows": valid_count}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
