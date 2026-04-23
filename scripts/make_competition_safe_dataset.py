from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter unsafe training sources out of a prepared dataset directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with train.jsonl / val.jsonl / stats.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where the filtered dataset copy will be written.")
    parser.add_argument(
        "--exclude-source-prefix",
        action="append",
        default=["solver::test:test"],
        help="Drop rows whose source starts with this prefix. Can be specified multiple times.",
    )
    return parser.parse_args()


def keep_row(row: dict[str, Any], excluded_prefixes: list[str]) -> bool:
    source = str(row.get("source", ""))
    return not any(source.startswith(prefix) for prefix in excluded_prefixes)


def filter_split(input_path: Path, output_path: Path, excluded_prefixes: list[str]) -> tuple[list[dict[str, Any]], int]:
    kept_rows: list[dict[str, Any]] = []
    filtered_count = 0
    for row in iter_jsonl(input_path):
        if keep_row(row, excluded_prefixes):
            kept_rows.append(row)
        else:
            filtered_count += 1
    write_jsonl(output_path, kept_rows)
    return kept_rows, filtered_count


def main() -> None:
    args = parse_args()
    train_path = args.input_dir / "train.jsonl"
    val_path = args.input_dir / "val.jsonl"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected {train_path} and {val_path} to exist.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_filtered = filter_split(train_path, args.output_dir / "train.jsonl", args.exclude_source_prefix)
    val_rows, val_filtered = filter_split(val_path, args.output_dir / "val.jsonl", args.exclude_source_prefix)

    source_counts = Counter[str]()
    for row in train_rows + val_rows:
        source_counts[str(row.get("source", "unknown"))] += 1

    summary = {
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "total_examples": len(train_rows) + len(val_rows),
        "filtered_out": {
            "train_examples": train_filtered,
            "val_examples": val_filtered,
            "total_examples": train_filtered + val_filtered,
            "excluded_source_prefixes": args.exclude_source_prefix,
        },
        "stats": dict(sorted(source_counts.items())),
    }
    (args.output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
