from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import iter_jsonl, write_jsonl
from countdown_distill.expressions import validate_expression
from countdown_distill.prompting import build_training_messages
from countdown_distill.solver import solve_countdown


DATASET_ID = "HuggingFaceTB/Countdown-Task-GOLD"
DEFAULT_VERIFIED_CONFIGS = [
    "verified_Qwen2.5-7B-Instruct",
    "verified_Qwen3-4B-Instruct-2507",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an SFT dataset for the Countdown distillation challenge.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where train/val JSONL files will be written.")
    parser.add_argument(
        "--dedupe-key",
        choices=["task", "task_label"],
        default="task",
        help=(
            "How to deduplicate examples. "
            "'task' keeps one label per (target, nums). "
            "'task_label' keeps multiple distinct labels for the same task."
        ),
    )
    parser.add_argument(
        "--verified-config",
        dest="verified_configs",
        action="append",
        default=[],
        help="Verified HF config to include. Can be specified multiple times.",
    )
    parser.add_argument(
        "--extra-jsonl",
        action="append",
        default=[],
        help="Optional generated teacher JSONL to merge into the training set.",
    )
    parser.add_argument(
        "--solver-source",
        action="append",
        default=[],
        help="Dataset source to solve exactly, in config:split format, e.g. all:train or test:test.",
    )
    parser.add_argument(
        "--allow-test-sources",
        action="store_true",
        help="Allow solver sources from the HF test split. Off by default because this is unsafe for competition training.",
    )
    parser.add_argument("--canonicalize-with-solver", action="store_true", help="Replace teacher labels with shorter exact solver outputs when possible.")
    parser.add_argument("--repair-with-solver", action="store_true", help="Repair invalid verified / generated labels with the exact solver.")
    parser.add_argument("--augment-all-limit", type=int, default=0, help="How many examples from the HF 'all' config to solve exactly and add.")
    parser.add_argument("--solver-limit-per-source", type=int, default=0, help="Optional cap per solver source.")
    parser.add_argument("--shuffle-solver-sources", action="store_true", help="Shuffle each solver source before truncation.")
    parser.add_argument("--limit-per-config", type=int, default=0, help="Optional cap per verified config.")
    parser.add_argument("--val-size", type=int, default=1500, help="Validation set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def label_priority(source: str) -> tuple[int, int, str]:
    source_rank = 2
    if source.startswith("solver::") or source == "solver_all" or "+solver_" in source:
        source_rank = 0
    elif "verified" in source:
        source_rank = 1
    return (source_rank, 0, source)


def should_replace(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    existing_key = (len(existing["label_expression"]),) + label_priority(existing["source"])
    candidate_key = (len(candidate["label_expression"]),) + label_priority(candidate["source"])
    return candidate_key < existing_key


def build_merge_key(target: int, nums: list[int], label_expression: str, dedupe_key: str) -> tuple[Any, ...]:
    base_key: tuple[Any, ...] = (int(target), tuple(int(num) for num in nums))
    if dedupe_key == "task_label":
        return base_key + (label_expression,)
    return base_key


def make_example(
    *,
    example_id: str,
    nums: list[int],
    target: int,
    label_expression: str,
    source: str,
    teacher_raw: str | None = None,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "target": int(target),
        "nums": [int(num) for num in nums],
        "source": source,
        "teacher_raw": teacher_raw,
        "label_expression": label_expression,
        "messages": build_training_messages(nums, target, label_expression),
    }


def choose_label(
    *,
    nums: list[int],
    target: int,
    teacher_expression: str,
    source: str,
    canonicalize_with_solver: bool,
    repair_with_solver: bool,
) -> tuple[str | None, str]:
    validated_teacher = validate_expression(teacher_expression, nums, target)
    solver_expression: str | None = None

    if canonicalize_with_solver or (repair_with_solver and not validated_teacher.is_valid):
        solver_expression = solve_countdown(nums, target)

    if validated_teacher.is_valid:
        final_expression = validated_teacher.normalized_expression
        final_source = source
        if solver_expression:
            candidate = validate_expression(solver_expression, nums, target)
            if candidate.is_valid and len(candidate.normalized_expression) < len(final_expression):
                final_expression = candidate.normalized_expression
                final_source = f"{source}+solver_canonical"
        return final_expression, final_source

    if repair_with_solver and solver_expression:
        candidate = validate_expression(solver_expression, nums, target)
        if candidate.is_valid:
            return candidate.normalized_expression, f"{source}+solver_repair"

    return None, source


def load_verified_examples(args: argparse.Namespace, stats: Counter[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    configs = args.verified_configs or DEFAULT_VERIFIED_CONFIGS
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}

    for config_name in configs:
        dataset = load_dataset(DATASET_ID, config_name, split="train")
        if args.limit_per_config > 0:
            dataset = dataset.select(range(min(args.limit_per_config, len(dataset))))

        for row_index, row in enumerate(dataset):
            nums = [int(num) for num in row["nums"]]
            target = int(row["target"])
            teacher_raw = row["messages"][-1]["content"]
            label_expression, final_source = choose_label(
                nums=nums,
                target=target,
                teacher_expression=teacher_raw,
                source=config_name,
                canonicalize_with_solver=args.canonicalize_with_solver,
                repair_with_solver=args.repair_with_solver,
            )
            if not label_expression:
                stats["skipped_invalid_verified"] += 1
                continue

            key = build_merge_key(target, nums, label_expression, args.dedupe_key)
            candidate = make_example(
                example_id=f"{config_name}:{row_index}",
                nums=nums,
                target=target,
                label_expression=label_expression,
                source=final_source,
                teacher_raw=teacher_raw,
            )
            if key not in merged or should_replace(merged[key], candidate):
                merged[key] = candidate
            stats[f"source::{final_source}"] += 1

    return merged


def load_extra_teacher_jsonl(args: argparse.Namespace, merged: dict[tuple[Any, ...], dict[str, Any]], stats: Counter[str]) -> None:
    for file_name in args.extra_jsonl:
        path = Path(file_name)
        for row_index, record in enumerate(iter_jsonl(path)):
            nums = [int(num) for num in record["nums"]]
            target = int(record["target"])
            teacher_raw = record.get("raw_response") or record.get("teacher_raw") or record.get("label_expression", "")
            teacher_expression = record.get("label_expression") or teacher_raw
            label_expression, final_source = choose_label(
                nums=nums,
                target=target,
                teacher_expression=teacher_expression,
                source=f"{path.stem}",
                canonicalize_with_solver=args.canonicalize_with_solver,
                repair_with_solver=args.repair_with_solver,
            )
            if not label_expression:
                stats["skipped_invalid_extra_jsonl"] += 1
                continue

            key = build_merge_key(target, nums, label_expression, args.dedupe_key)
            candidate = make_example(
                example_id=f"{path.stem}:{row_index}",
                nums=nums,
                target=target,
                label_expression=label_expression,
                source=final_source,
                teacher_raw=teacher_raw,
            )
            if key not in merged or should_replace(merged[key], candidate):
                merged[key] = candidate
            stats[f"source::{final_source}"] += 1


def augment_with_exact_solver(args: argparse.Namespace, merged: dict[tuple[Any, ...], dict[str, Any]], stats: Counter[str]) -> None:
    if args.augment_all_limit <= 0:
        return

    dataset = load_dataset(DATASET_ID, "all", split="train").shuffle(seed=args.seed)
    dataset = dataset.select(range(min(args.augment_all_limit, len(dataset))))

    for row_index, row in enumerate(dataset):
        nums = [int(num) for num in row["nums"]]
        target = int(row["target"])
        label_expression = solve_countdown(nums, target)
        if not label_expression:
            stats["skipped_unsolved_solver_all"] += 1
            continue
        key = build_merge_key(target, nums, label_expression, args.dedupe_key)
        candidate = make_example(
            example_id=f"solver_all:{row_index}",
            nums=nums,
            target=target,
            label_expression=label_expression,
            source="solver_all",
        )
        if key not in merged or should_replace(merged[key], candidate):
            merged[key] = candidate
        stats["source::solver_all"] += 1


def parse_solver_source(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        if spec == "test":
            return spec, "test"
        return spec, "train"
    config_name, split_name = spec.split(":", 1)
    config_name = config_name.strip()
    split_name = split_name.strip()
    if not config_name or not split_name:
        raise ValueError(f"Invalid solver source spec: {spec!r}")
    return config_name, split_name


def load_solver_sources(args: argparse.Namespace, merged: dict[tuple[Any, ...], dict[str, Any]], stats: Counter[str]) -> None:
    for solver_spec in args.solver_source:
        config_name, split_name = parse_solver_source(solver_spec)
        if config_name == "test" and not args.allow_test_sources:
            raise ValueError(
                "Refusing to use solver source 'test:*' without --allow-test-sources. "
                "For competition-safe training, keep test sources out of the training set."
            )
        dataset = load_dataset(DATASET_ID, config_name, split=split_name)
        if args.shuffle_solver_sources:
            dataset = dataset.shuffle(seed=args.seed)
        if args.solver_limit_per_source > 0:
            dataset = dataset.select(range(min(args.solver_limit_per_source, len(dataset))))

        source_name = f"solver::{config_name}:{split_name}"
        for row_index, row in enumerate(dataset):
            nums = [int(num) for num in row["nums"]]
            target = int(row["target"])
            label_expression = solve_countdown(nums, target)
            if not label_expression:
                stats[f"skipped_unsolved::{source_name}"] += 1
                continue

            key = build_merge_key(target, nums, label_expression, args.dedupe_key)
            candidate = make_example(
                example_id=f"{source_name}:{row_index}",
                nums=nums,
                target=target,
                label_expression=label_expression,
                source=source_name,
            )
            if key not in merged or should_replace(merged[key], candidate):
                merged[key] = candidate
            stats[f"source::{source_name}"] += 1


def train_val_split(examples: list[dict[str, Any]], val_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0:
        return examples, []
    if val_size >= len(examples):
        raise ValueError(f"val_size={val_size} must be smaller than the number of examples={len(examples)}")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_examples = shuffled[:val_size]
    train_examples = shuffled[val_size:]
    return train_examples, val_examples


def main() -> None:
    args = parse_args()
    stats: Counter[str] = Counter()

    merged = load_verified_examples(args, stats)
    load_extra_teacher_jsonl(args, merged, stats)
    augment_with_exact_solver(args, merged, stats)
    load_solver_sources(args, merged, stats)

    examples = list(merged.values())
    train_examples, val_examples = train_val_split(examples, args.val_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    stats_path = args.output_dir / "stats.json"

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    summary = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "total_examples": len(examples),
        "dedupe_key": args.dedupe_key,
        "stats": dict(sorted(stats.items())),
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
