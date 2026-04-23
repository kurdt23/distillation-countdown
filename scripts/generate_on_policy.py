from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DATASET_ID = "HuggingFaceTB/Countdown-Task-GOLD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate on-policy student samples, then build verifier-filtered SFT examples "
            "and DPO correct/incorrect pairs."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Student adapter dir or merged model dir.")
    parser.add_argument("--base-model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--source-jsonl", type=Path, default=None, help="Optional prepared train JSONL with target/nums.")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--allow-test-source", action="store_true")
    parser.add_argument("--output-sft-train-jsonl", type=Path, required=True)
    parser.add_argument("--output-sft-val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dpo-train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dpo-val-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--restrict-output-charset", action="store_true")
    parser.add_argument("--solver-chosen-for-dpo", action="store_true", help="Use solver as chosen when student produced only rejected candidates.")
    parser.add_argument("--include-solver-chosen-in-sft", action="store_true")
    parser.add_argument("--max-dpo-pairs-per-task", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def build_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    import torch
    from transformers import BitsAndBytesConfig

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
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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


def read_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from countdown_distill.data import iter_jsonl

    if args.source_jsonl is not None:
        rows = list(iter_jsonl(args.source_jsonl))
    else:
        if args.split.startswith("test") and not args.allow_test_source:
            raise ValueError("Refusing to sample from a test split without --allow-test-source.")
        rows = list(load_dataset(DATASET_ID, args.dataset_config, split=args.split))

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]

    seen: set[tuple[int, tuple[int, ...]]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        target = int(row["target"])
        nums = [int(num) for num in row["nums"]]
        key = (target, tuple(nums))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"target": target, "nums": nums})
    return deduped


def build_fallback_rejected(nums: list[int], target: int) -> str:
    from countdown_distill.expressions import validate_expression

    if len(nums) <= 1:
        return str(nums[0]) if nums else "0"

    expression = " + ".join(str(num) for num in nums)
    validation = validate_expression(expression, nums, target)
    if not validation.is_valid:
        return expression

    expression = " - ".join(str(num) for num in nums)
    validation = validate_expression(expression, nums, target)
    if not validation.is_valid:
        return expression

    return str(nums[0])


def select_valid_expression(results: list[ValidationResult], nums: list[int], target: int) -> str | None:
    from countdown_distill.expressions import validation_quality_key

    valid_results = [result for result in results if result.is_valid]
    if not valid_results:
        return None
    frequency = Counter(result.normalized_expression for result in valid_results)
    best = max(
        valid_results,
        key=lambda result: (
            frequency[result.normalized_expression],
            validation_quality_key(result, nums, target),
        ),
    )
    return best.normalized_expression


def select_rejected_expressions(results: list[ValidationResult], nums: list[int], target: int, max_items: int) -> list[str]:
    from countdown_distill.expressions import validation_quality_key

    rejected_results = [
        result
        for result in results
        if not result.is_valid and result.normalized_expression
    ]
    rejected_results.sort(
        key=lambda result: validation_quality_key(result, nums, target),
        reverse=True,
    )

    rejected: list[str] = []
    seen: set[str] = set()
    for result in rejected_results:
        if result.normalized_expression in seen:
            continue
        seen.add(result.normalized_expression)
        rejected.append(result.normalized_expression)
        if len(rejected) >= max_items:
            break
    return rejected


def split_rows(rows: list[dict[str, Any]], val_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if val_size <= 0:
        return shuffled, []
    effective_val_size = min(val_size, max(0, len(shuffled) - 1))
    return shuffled[effective_val_size:], shuffled[:effective_val_size]


def main() -> None:
    args = parse_args()
    from countdown_distill.data import write_jsonl
    from countdown_distill.expressions import extract_expression_candidates, validate_expression
    from countdown_distill.prompting import build_messages, build_training_messages, render_chat_prompt
    from countdown_distill.solver import solve_countdown
    from predict import build_allowed_token_ids, generate_text_groups

    source_rows = read_source_rows(args)
    model, tokenizer = load_model_and_tokenizer(args)
    allowed_token_ids = build_allowed_token_ids(tokenizer) if args.restrict_output_charset else None

    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    total_batches = (len(source_rows) + args.batch_size - 1) // args.batch_size if source_rows else 0

    for batch_index, start in enumerate(range(0, len(source_rows), args.batch_size), start=1):
        batch = source_rows[start : start + args.batch_size]
        prompts = [
            render_chat_prompt(tokenizer, build_messages(row["nums"], row["target"]), add_generation_prompt=True)
            for row in batch
        ]
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

        for relative_index, (row, raw_texts, prompt_text) in enumerate(zip(batch, grouped_texts, prompts)):
            global_index = start + relative_index
            nums = row["nums"]
            target = row["target"]
            validation_results: list[ValidationResult] = []
            raw_candidate_count = 0

            for raw_text in raw_texts:
                extracted_candidates = extract_expression_candidates(raw_text)
                if not extracted_candidates:
                    extracted_candidates = [raw_text.strip()]
                for extracted in extracted_candidates:
                    raw_candidate_count += 1
                    validation_results.append(validate_expression(extracted, nums, target))

            chosen_expression = select_valid_expression(validation_results, nums, target)
            chosen_source = "student"
            rejected_expressions = select_rejected_expressions(
                validation_results,
                nums,
                target,
                max(args.max_dpo_pairs_per_task, 1),
            )

            if chosen_expression is None and args.solver_chosen_for_dpo:
                solver_expression = solve_countdown(nums, target)
                solver_validation = validate_expression(solver_expression, nums, target)
                if solver_validation.is_valid:
                    chosen_expression = solver_validation.normalized_expression
                    chosen_source = "solver"

            if not rejected_expressions:
                fallback_rejected = build_fallback_rejected(nums, target)
                fallback_validation = validate_expression(fallback_rejected, nums, target)
                if not fallback_validation.is_valid:
                    rejected_expressions = [fallback_validation.normalized_expression or fallback_rejected]

            if chosen_expression is not None and (chosen_source == "student" or args.include_solver_chosen_in_sft):
                sft_rows.append(
                    {
                        "example_id": f"onpolicy_sft:{global_index}",
                        "source": f"onpolicy::{chosen_source}",
                        "target": target,
                        "nums": nums,
                        "label_expression": chosen_expression,
                        "messages": build_training_messages(nums, target, chosen_expression),
                    }
                )
                stats[f"sft::{chosen_source}"] += 1

            if chosen_expression is not None and rejected_expressions:
                for pair_index, rejected_expression in enumerate(rejected_expressions[: args.max_dpo_pairs_per_task]):
                    dpo_rows.append(
                        {
                            "example_id": f"onpolicy_dpo:{global_index}:{pair_index}",
                            "source": f"onpolicy::{chosen_source}",
                            "target": target,
                            "nums": nums,
                            "prompt": prompt_text,
                            "chosen": chosen_expression,
                            "rejected": rejected_expression,
                        }
                    )
                    stats[f"dpo::{chosen_source}"] += 1

            stats["tasks"] += 1
            stats["raw_generations"] += len(raw_texts)
            stats["raw_candidates"] += raw_candidate_count
            if any(result.is_valid for result in validation_results):
                stats["tasks_with_student_valid"] += 1
            if rejected_expressions:
                stats["tasks_with_rejected"] += 1

        if args.progress_every > 0 and (batch_index % args.progress_every == 0 or batch_index == total_batches):
            print(
                f"Processed batch {batch_index}/{total_batches} "
                f"({min(start + len(batch), len(source_rows))}/{len(source_rows)} tasks), "
                f"SFT rows: {len(sft_rows)}, DPO rows: {len(dpo_rows)}"
            )

    sft_train, sft_val = split_rows(sft_rows, args.val_size, args.seed)
    dpo_train, dpo_val = split_rows(dpo_rows, args.val_size, args.seed)

    write_jsonl(args.output_sft_train_jsonl, sft_train)
    write_jsonl(args.output_sft_val_jsonl, sft_val)
    write_jsonl(args.output_dpo_train_jsonl, dpo_train)
    write_jsonl(args.output_dpo_val_jsonl, dpo_val)

    summary = {
        "source_tasks": len(source_rows),
        "sft_train_rows": len(sft_train),
        "sft_val_rows": len(sft_val),
        "dpo_train_rows": len(dpo_train),
        "dpo_val_rows": len(dpo_val),
        "stats": dict(sorted(stats.items())),
    }
    stats_path = args.stats_json or args.output_dpo_train_jsonl.parent / "onpolicy_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_sft_train_jsonl}")
    print(f"Wrote {args.output_sft_val_jsonl}")
    print(f"Wrote {args.output_dpo_train_jsonl}")
    print(f"Wrote {args.output_dpo_val_jsonl}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
