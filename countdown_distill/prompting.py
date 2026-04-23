from __future__ import annotations

from typing import Any, Sequence


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise Countdown arithmetic solver. "
    "Use each provided number exactly once. "
    "Allowed operators: +, -, *, /. "
    "Your entire reply must be exactly one arithmetic expression on one line. "
    "Allowed characters in the reply: digits, spaces, parentheses, +, -, *, /. "
    "Do not include words, bullets, explanation, an equals sign, tags, or extra text."
)


def format_numbers(nums: Sequence[int]) -> str:
    return "[" + ", ".join(str(num) for num in nums) + "]"


def build_user_prompt(nums: Sequence[int], target: int) -> str:
    return (
        f"Numbers: {format_numbers(nums)}\n"
        f"Target: {target}\n"
        "Rules: use each number exactly once; allowed operators are +, -, *, /.\n"
        "Reply with exactly one expression and nothing else.\n"
        "Example valid reply: (83 * (68 - 11) + 23) / 62"
    )


def build_messages(
    nums: Sequence[int],
    target: int,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(nums, target)},
    ]


def build_training_messages(
    nums: Sequence[int],
    target: int,
    label_expression: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = build_messages(nums, target, system_prompt=system_prompt)
    messages.append({"role": "assistant", "content": label_expression})
    return messages


def build_repair_messages(
    nums: Sequence[int],
    target: int,
    previous_expression: str,
    issue: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    user_prompt = (
        f"Numbers: {format_numbers(nums)}\n"
        f"Target: {target}\n"
        f"Previous answer: {previous_expression or '<empty>'}\n"
        f"Issue: {issue}\n"
        "Return a corrected expression only.\n"
        "Use each number exactly once and only operators +, -, *, /.\n"
        "Do not include an equals sign, explanation, or any words."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_chat_prompt(tokenizer: Any, messages: Sequence[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered = []
    for message in messages:
        role = message["role"].upper()
        rendered.append(f"{role}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)
