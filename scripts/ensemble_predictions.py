from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.data import infer_task_fields, read_records
from countdown_distill.expressions import validate_expression, validation_quality_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble multiple prediction CSV files using local expression validation."
    )
    parser.add_argument("--reference", type=Path, required=True, help="CSV / Parquet / JSONL with target and nums.")
    parser.add_argument("--input-csv", dest="input_csvs", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--debug-jsonl", type=Path, default=None)
    return parser.parse_args()


def load_prediction_map(path: Path) -> dict[Any, str]:
    frame = pd.read_csv(path)
    result: dict[Any, str] = {}
    for _, row in frame.iterrows():
        equation = row["equation"] if isinstance(row["equation"], str) else ""
        result[row["id"]] = equation
    return result


def choose_candidate(candidates: list[dict[str, Any]], nums: list[int], target: int) -> dict[str, Any]:
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

    return max(
        candidates,
        key=lambda candidate: (
            validation_quality_key(candidate["validation"], nums, target),
            -candidate["index"],
        ),
    )


def main() -> None:
    args = parse_args()
    prediction_maps = [load_prediction_map(path) for path in args.input_csvs]
    reference_rows = read_records(args.reference)

    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    any_valid_count = 0
    all_valid_count = 0

    for index, record in enumerate(reference_rows):
        row_id = record.get("id", index)
        target, nums = infer_task_fields(record)

        candidates: list[dict[str, Any]] = []
        for source_index, prediction_map in enumerate(prediction_maps):
            equation = prediction_map.get(row_id, "")
            validation = validate_expression(equation, nums, target)
            candidates.append(
                {
                    "index": source_index,
                    "source_csv": str(args.input_csvs[source_index]),
                    "equation": equation,
                    "validation": validation,
                }
            )

        if any(candidate["validation"].is_valid for candidate in candidates):
            any_valid_count += 1
        if all(candidate["validation"].is_valid for candidate in candidates):
            all_valid_count += 1

        chosen = choose_candidate(candidates, nums, target)
        chosen_equation = chosen["validation"].normalized_expression if chosen["validation"].is_valid else chosen["equation"]
        rows.append({"id": row_id, "equation": chosen_equation})

        if args.debug_jsonl is not None:
            debug_rows.append(
                {
                    "id": row_id,
                    "target": target,
                    "nums": nums,
                    "chosen_equation": chosen_equation,
                    "chosen_valid": chosen["validation"].is_valid,
                    "candidates": [
                        {
                            "source_csv": candidate["source_csv"],
                            "equation": candidate["equation"],
                            "is_valid": candidate["validation"].is_valid,
                            "reason": candidate["validation"].reason,
                            "normalized_expression": candidate["validation"].normalized_expression,
                        }
                        for candidate in candidates
                    ],
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    if args.debug_jsonl is not None:
        args.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.debug_jsonl.open("w", encoding="utf-8") as handle:
            for row in debug_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "rows_with_any_valid_candidate": any_valid_count,
                "rows_with_all_candidates_valid": all_valid_count,
                "output_csv": str(args.output_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
