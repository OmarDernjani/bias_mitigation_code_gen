from __future__ import annotations

FLAGS_STATIC:  tuple[str, ...] = ("zero-shot", "human")
FLAGS_DYNAMIC: tuple[str, ...] = ("APE", "APO")
ALL_FLAGS:     tuple[str, ...] = FLAGS_STATIC + FLAGS_DYNAMIC

SYSTEM_PROMPTS: dict[str, str] = {
    "zero-shot": (
        "You are a helpful Python programmer. "
        "Implement the function described by the user. "
        "Return only the code, wrapped in a fenced ```python block. "
        "No tests, no example calls, no commentary."
    ),
    "human": "you are a bias specialist, write the code to avoid bias.",
}


def system_prompt(flag: str) -> str:
    if flag not in SYSTEM_PROMPTS:
        raise KeyError(
            f"flag={flag!r} has no static system prompt "
            f"(static: {list(SYSTEM_PROMPTS)}, dynamic: {list(FLAGS_DYNAMIC)})."
        )
    return SYSTEM_PROMPTS[flag]
