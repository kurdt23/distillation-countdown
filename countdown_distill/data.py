from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import numpy as np


PROMPT_RE = re.compile(
    r"Using the numbers\s*\[(?P<nums>[^\]]+)\].*?equals\s*(?P<target>-?\d+)",
    re.IGNORECASE | re.DOTALL,
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_nums(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist()]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [int(item) for item in parsed]
    raise ValueError(f"Cannot parse nums from {value!r}")


def extract_prompt_text(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if isinstance(prompt_field, list):
        for item in reversed(prompt_field):
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content", ""))
        joined = []
        for item in prompt_field:
            if isinstance(item, dict):
                joined.append(str(item.get("content", "")))
        return "\n".join(joined)
    return str(prompt_field)


def infer_task_fields(record: dict[str, Any]) -> tuple[int, list[int]]:
    if "target" in record and "nums" in record:
        return int(record["target"]), parse_nums(record["nums"])

    prompt_field = record.get("prompt")
    if prompt_field is None:
        raise KeyError("Record does not contain target/nums or a prompt field.")

    prompt_text = extract_prompt_text(prompt_field)
    match = PROMPT_RE.search(prompt_text)
    if not match:
        raise ValueError("Could not extract target and nums from prompt.")

    nums = [int(piece.strip()) for piece in match.group("nums").split(",") if piece.strip()]
    target = int(match.group("target"))
    return target, nums


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        return list(iter_jsonl(input_path))

    if suffix == ".csv":
        frame = pd.read_csv(input_path)
        return frame.to_dict(orient="records")

    if suffix == ".parquet":
        frame = pd.read_parquet(input_path)
        return frame.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {input_path}")
