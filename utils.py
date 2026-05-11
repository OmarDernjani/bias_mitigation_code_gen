from __future__ import annotations

import os
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompts import CODE_FORMAT

NUM_CTX      = int(os.environ.get("NUM_CTX", "8192"))
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "10"))


_FENCED = (
    re.compile(r"```python\s*\n?(.*?)```", re.DOTALL),
    re.compile(r"```py\s*\n?(.*?)```",     re.DOTALL),
    re.compile(r"```\s*\n?(.*?)```",       re.DOTALL),
)
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*)$", re.DOTALL)


def extract_code(response: str) -> str:
    """Pull a Python block out of an LLM response.
    Fenced (longest match) -> unclosed fence -> raw text."""
    for pat in _FENCED:
        if (m := pat.findall(response)):
            return max(m, key=len).strip()
    if (m := _OPEN_FENCE.search(response)):
        return m.group(1).strip()
    return response.strip()


def _target_mention() -> str:
    mt = os.environ.get("MODEL_TARGET", "")
    if not mt:
        return ""
    return (
        f"The target LLM you are optimizing for is `{mt}`. "
        "Tailor your scaffolding to its strengths and weaknesses.\n\n"
    )


def build_target_chain(model: str):
    """Generic target chain: enriched prompt in, code out."""
    system_msg = (
        "You are a helpful Python programmer. "
        "Implement the function described by the user.\n\n"
        f"{CODE_FORMAT}\n"
        "Return only the code, wrapped in a fenced ```python block. "
        "No tests, no example calls, no commentary."
    )
    template = ChatPromptTemplate([("system", system_msg), ("human", "{user_prompt}")])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def build_optimizer_chain(model: str):
    """Bias-aware optimizer (APE step 0 / APO p0).
    Wraps the original task with scaffolding that pushes the target LLM toward
    bias-free code. Original task text is preserved verbatim."""
    system_msg = (
        "You are an expert prompt engineer specializing in bias-free code generation.\n\n"
        f"{_target_mention()}"
        "You will receive a coding task. Produce an enriched prompt that wraps the\n"
        "ORIGINAL task with scaffolding that helps the target LLM solve it WITHOUT\n"
        "relying on protected attributes (gender, race, age, region/city, religion,\n"
        "ethnicity, sexual orientation) as decision factors.\n\n"
        "The prompt you produce MUST:\n\n"
        "1. Assign the target LLM an expert role relevant to the task and to fairness\n"
        "   (e.g. \"fairness-aware ML engineer\", \"ethical software developer\").\n\n"
        "2. Include the ORIGINAL task text VERBATIM. Do NOT paraphrase, summarize,\n"
        "   shorten, or omit attributes / constraints.\n\n"
        "3. Explicitly instruct the model to exclude protected attributes from the\n"
        "   decision logic (or, if a non-protected attribute is used, to justify it\n"
        "   in a comment).\n\n"
        "Beyond these requirements you have full freedom over the scaffolding.\n"
        "Do NOT add execution rules (function name, I/O, markdown wrapping) — those\n"
        "are appended automatically downstream.\n\n"
        "Return ONLY the final prompt for the target LLM. No preamble, no commentary."
    )
    template = ChatPromptTemplate([("system", system_msg), ("human", "Task:\n{problem}")])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def build_variation_chain(model: str):
    """Bias-aware semantic variation (APE step k>0)."""
    system_msg = (
        "You are an expert prompt engineer specializing in bias-free code generation.\n\n"
        f"{_target_mention()}"
        "You will receive a coding task AND a previous enriched prompt. Produce a\n"
        "SEMANTIC VARIATION of that enriched prompt: same goal (bias-free code),\n"
        "different framing.\n\n"
        "The variation MUST:\n\n"
        "1. Include the ORIGINAL task text VERBATIM.\n"
        "2. Use a DIFFERENT expert role from the previous prompt.\n"
        "3. Keep the explicit instruction to avoid protected attributes (gender,\n"
        "   race, age, region/city, religion, ethnicity, sexual orientation).\n\n"
        "Beyond these requirements you have full freedom over the scaffolding.\n"
        "Return ONLY the new enriched prompt. No preamble, no commentary."
    )
    template = ChatPromptTemplate([
        ("system", system_msg),
        ("human", "Task:\n{problem}\n\nPrevious enriched prompt:\n{previous}"),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()
