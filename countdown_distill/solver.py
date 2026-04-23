from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from countdown_distill.expressions import strip_outer_parentheses


def _better_expression(candidate: str, existing: str | None) -> bool:
    if existing is None:
        return True
    return (len(candidate), candidate) < (len(existing), existing)


def _store(best: dict[Fraction, str], value: Fraction, expression: str) -> None:
    current = best.get(value)
    if _better_expression(expression, current):
        best[value] = expression


def _commutative_expression(left: str, operator: str, right: str) -> str:
    ordered = sorted((left, right), key=lambda item: (len(item), item))
    return f"({ordered[0]} {operator} {ordered[1]})"


def enumerate_solutions(nums: Sequence[int]) -> dict[Fraction, str]:
    numbers = [int(num) for num in nums]
    if not numbers:
        return {}

    total_masks = 1 << len(numbers)
    dp: list[dict[Fraction, str]] = [dict() for _ in range(total_masks)]

    for index, number in enumerate(numbers):
        dp[1 << index][Fraction(number, 1)] = str(number)

    for mask in range(1, total_masks):
        if mask & (mask - 1) == 0:
            continue

        best: dict[Fraction, str] = {}
        submask = (mask - 1) & mask
        while submask:
            other = mask ^ submask
            if submask < other:
                left_map = dp[submask]
                right_map = dp[other]
                for left_value, left_expression in left_map.items():
                    for right_value, right_expression in right_map.items():
                        _store(best, left_value + right_value, _commutative_expression(left_expression, "+", right_expression))
                        _store(best, left_value * right_value, _commutative_expression(left_expression, "*", right_expression))
                        _store(best, left_value - right_value, f"({left_expression} - {right_expression})")
                        _store(best, right_value - left_value, f"({right_expression} - {left_expression})")
                        if right_value != 0:
                            _store(best, left_value / right_value, f"({left_expression} / {right_expression})")
                        if left_value != 0:
                            _store(best, right_value / left_value, f"({right_expression} / {left_expression})")
            submask = (submask - 1) & mask

        dp[mask] = best

    return dp[-1]


def solve_countdown(nums: Sequence[int], target: int) -> str | None:
    solutions = enumerate_solutions(nums)
    expression = solutions.get(Fraction(int(target), 1))
    if expression is None:
        return None
    return strip_outer_parentheses(expression)

