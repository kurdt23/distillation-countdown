from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import infer_task_fields, read_records
from countdown_distill.expressions import validate_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate submission accuracy against a reference file.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions = pd.read_csv(args.predictions)
    prediction_map = {row["id"]: row["equation"] for _, row in predictions.iterrows()}

    reference_rows = read_records(args.reference)
    total = 0
    correct = 0
    valid = 0

    for index, record in enumerate(reference_rows):
        row_id = record.get("id", index)
        equation = prediction_map.get(row_id, "")
        target, nums = infer_task_fields(record)
        result = validate_expression(equation, nums, target)
        total += 1
        if result.is_valid:
            valid += 1
            correct += 1

    summary = {
        "total": total,
        "valid": valid,
        "accuracy": correct / total if total else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
