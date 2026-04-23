from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


ALLOWED_CHARS_RE = re.compile(r"^[0-9+\-*/()\s]+$")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)
PROMPT_PATTERN_RE = re.compile(r"[0-9+\-*/()=\s]+")


class ExpressionError(ValueError):
    """Raised when a candidate expression cannot be parsed safely."""


_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class ValidationResult:
    expression: str
    normalized_expression: str
    is_valid: bool
    reason: str | None
    value: Fraction | None
    used_numbers: tuple[int, ...]


def normalize_symbols(text: str) -> str:
    replacements = {
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00d7": "*",
        "\u2715": "*",
        "\u00f7": "/",
        "\u2044": "/",
        "\u2215": "/",
        "\u00a0": " ",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def strip_outer_parentheses(expression: str) -> str:
    candidate = expression.strip()
    while candidate.startswith("(") and candidate.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(candidate):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and index != len(candidate) - 1:
                balanced = False
                break
        if not balanced:
            break
        candidate = candidate[1:-1].strip()
    return candidate


def _normalize_generation_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = normalize_symbols(text).strip()

    answer_match = ANSWER_TAG_RE.search(normalized)
    if answer_match:
        normalized = answer_match.group(1).strip()

    code_match = CODE_BLOCK_RE.search(normalized)
    if code_match:
        normalized = code_match.group(1).strip()

    normalized = re.sub(r"</?think>", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"</?answer>", " ", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\r", "\n")
    return normalized


def _clean_candidate_fragment(fragment: str) -> str:
    normalized = re.sub(r"^(answer|equation|expression|final answer)\s*:\s*", "", fragment, flags=re.IGNORECASE)
    normalized = normalized.strip("` ")

    if "=" in normalized:
        normalized = normalized.split("=", 1)[0].strip()

    normalized = normalized.strip().rstrip(".,;:")
    normalized = re.sub(r"\s+", " ", normalized)
    return strip_outer_parentheses(normalized) if normalized else ""


def extract_expression_candidates(text: str | None) -> list[str]:
    normalized = _normalize_generation_text(text)
    if not normalized:
        return []

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    raw_candidates: list[str] = []

    for line in lines:
        if any(op in line for op in "+-*/=") and any(ch.isdigit() for ch in line):
            raw_candidates.append(line)
        segments = [segment.strip() for segment in PROMPT_PATTERN_RE.findall(line)]
        for segment in segments:
            if any(ch.isdigit() for ch in segment) and any(op in segment for op in "+-*/"):
                raw_candidates.append(segment)

    if not raw_candidates:
        raw_candidates.append(normalized)

    seen: set[str] = set()
    candidates: list[str] = []
    for raw_candidate in raw_candidates:
        candidate = _clean_candidate_fragment(raw_candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def extract_expression_candidate(text: str | None) -> str:
    candidates = extract_expression_candidates(text)
    return candidates[0] if candidates else ""


def _evaluate_ast(node: ast.AST, used_numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body, used_numbers)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExpressionError("Only integer literals are allowed.")
        if value < 0:
            raise ExpressionError("Direct negative literals are not allowed.")
        used_numbers.append(value)
        return Fraction(value, 1)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            raise ExpressionError("Direct negative literals are not allowed.")
        return -_evaluate_ast(node.operand, used_numbers)

    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left, used_numbers)
        right = _evaluate_ast(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ExpressionError("Division by zero.")
            return left / right
        raise ExpressionError("Unsupported operator.")

    raise ExpressionError(f"Unsupported syntax node: {type(node).__name__}")


def evaluate_expression(expression: str) -> tuple[Fraction, tuple[int, ...]]:
    normalized = extract_expression_candidate(expression)
    if not normalized:
        raise ExpressionError("Empty expression.")
    if not ALLOWED_CHARS_RE.fullmatch(normalized):
        raise ExpressionError("Expression contains unsupported characters.")

    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("Syntax error.") from exc

    used_numbers: list[int] = []
    value = _evaluate_ast(parsed, used_numbers)
    return value, tuple(used_numbers)


def validate_expression(
    expression: str | None,
    nums: Sequence[int],
    target: int | None = None,
    *,
    require_all_numbers: bool = True,
) -> ValidationResult:
    candidate = expression or ""
    normalized = extract_expression_candidate(candidate)
    if not normalized:
        return ValidationResult(candidate, normalized, False, "empty_expression", None, ())

    try:
        value, used_numbers = evaluate_expression(normalized)
    except ExpressionError as exc:
        return ValidationResult(candidate, normalized, False, str(exc), None, ())

    provided_counter = Counter(int(num) for num in nums)
    used_counter = Counter(used_numbers)
    for number, count in used_counter.items():
        if count > provided_counter.get(number, 0):
            return ValidationResult(candidate, normalized, False, "used_unavailable_number", value, used_numbers)

    if require_all_numbers and used_counter != provided_counter:
        return ValidationResult(candidate, normalized, False, "must_use_all_numbers", value, used_numbers)

    if target is not None and value != Fraction(int(target), 1):
        return ValidationResult(candidate, normalized, False, "wrong_target", value, used_numbers)

    return ValidationResult(candidate, normalized, True, None, value, used_numbers)


def validation_issue_rank(reason: str | None) -> int:
    reason_rank = {
        None: 5,
        "wrong_target": 4,
        "must_use_all_numbers": 3,
        "used_unavailable_number": 2,
        "Syntax error.": 1,
        "empty_expression": 0,
    }
    return reason_rank.get(reason, 1)


def validation_quality_key(
    result: ValidationResult,
    nums: Sequence[int],
    target: int | None = None,
) -> tuple[int, int, int, int, int, Fraction, int, int]:
    provided_counter = Counter(int(num) for num in nums)
    used_counter = Counter(int(num) for num in result.used_numbers)

    overlap = sum(min(used_counter[number], count) for number, count in provided_counter.items())
    overuse = sum(max(0, count - provided_counter.get(number, 0)) for number, count in used_counter.items())
    missing = sum(max(0, count - used_counter.get(number, 0)) for number, count in provided_counter.items())

    if target is not None and result.value is not None:
        distance = abs(result.value - Fraction(int(target), 1))
    elif result.value is not None:
        distance = Fraction(0, 1)
    else:
        distance = Fraction(10**9, 1)

    return (
        1 if result.is_valid else 0,
        1 if result.value is not None else 0,
        -overuse,
        overlap,
        -missing,
        -distance,
        validation_issue_rank(result.reason),
        -len(result.normalized_expression),
    )
