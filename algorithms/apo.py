"""APO / ProTeGi (Pryzant et al. 2023) adapted to per-task bias mitigation.

Same beam-search structure as the paper, but the failure signal is
"this prompt produced code that exhibits bias on PROTECTED attributes"
instead of "this prompt produced code that fails on test cases". The
gradient chain critiques the prompt in light of biased attributes; the
edit chain rewrites it; the paraphrase chain expands the pool via Monte
Carlo paraphrase.

Carried over from the paper-faithful APPS implementation: minibatch ==
full dev pool; successor selection by exhaustive evaluation; no UCB /
successive halving; no elitism between iterations."""

from __future__ import annotations

import os
import random as _random
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..metamorphic import evaluate_code_bias, get_biased_attributes
from ..utils import NUM_CTX, extract_code

NUM_GRADIENTS   = int(os.getenv("APO_NUM_GRADIENTS",   2))  # m: critiques per beam member
NUM_EDITS       = int(os.getenv("APO_NUM_EDITS",       1))  # q: improved prompts per critique
NUM_PARAPHRASES = int(os.getenv("APO_NUM_PARAPHRASES", 1))  # r: paraphrases per gradient candidate
BEAM_WIDTH      = int(os.getenv("APO_BEAM_WIDTH",      2))
MAX_ITERS       = int(os.getenv("APO_MAX_ITERS",       4))
MAX_NO_IMPROVE  = int(os.getenv("APO_MAX_NO_IMPROVE",  2))


def _target_mention() -> str:
    mt = os.environ.get("MODEL_TARGET", "")
    return f"The target LLM is `{mt}`. " if mt else ""


# ── Optimizer chains ─────────────────────────────────────────────────────────
def _gradient_chain(model: str):
    """ProTeGi step 1 — textual gradients (critiques) of a biased prompt."""
    template = ChatPromptTemplate([
        ("system",
         "You are an expert in fairness-aware programming and prompt engineering. "
         f"{_target_mention()}"
         "I am writing a prompt that asks a target language model to solve a coding "
         "task. My current prompt led the model to produce code that exhibits bias on "
         "protected attributes (varying a single protected attribute changes the "
         "output). Your job is to identify weaknesses in the PROMPT (not the code) "
         "that could have caused the bias.\n\n"
         "Generate EXACTLY {m} DISTINCT critiques. Each critique should focus on a "
         "DIFFERENT weakness, e.g.: prompt does not mention fairness; prompt suggests "
         "using all available attributes including protected ones; prompt does not "
         "ask to justify feature selection; prompt does not assign a fairness-aware "
         "role; prompt allows protected attributes to influence the decision.\n\n"
         "Output format (mandatory, one critique per <critique> block):\n"
         "<critique>first weakness explained in 1-3 sentences</critique>\n"
         "<critique>second weakness explained in 1-3 sentences</critique>\n"
         "...\n"
         "Output NOTHING outside the <critique> blocks."),
        ("human",
         "Current prompt:\n{prompt}\n\n"
         "Generated code:\n{code}\n\n"
         "Biased protected attributes:\n{biases}\n\n"
         "Generate {m} distinct critiques."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _edit_chain(model: str):
    """ProTeGi step 2 — apply a textual gradient: write q improved prompts."""
    template = ChatPromptTemplate([
        ("system",
         "You are an expert prompt engineer for fair code generation. "
         f"{_target_mention()}"
         "Given a current prompt that led the target LLM to produce biased code, the "
         "list of biased protected attributes, and a critique that identifies a "
         "weakness in the prompt, write {q} DIFFERENT improved prompts that address "
         "the critique.\n\n"
         "Each improved prompt MUST:\n"
         "- Include the ORIGINAL task statement VERBATIM (do not omit attributes,\n"
         "  examples, or constraints)\n"
         "- Specifically address the critique with new scaffolding\n"
         "- Explicitly forbid using protected attributes (gender, race, age,\n"
         "  region/city, religion, ethnicity, sexual orientation) as decision factors\n"
         "- Be self-contained and ready to send to the target model\n\n"
         "Do NOT add execution rules (function name, I/O handling, markdown wrapping).\n\n"
         "Output format (mandatory, one prompt per <prompt> block):\n"
         "<prompt>first improved prompt</prompt>\n"
         "<prompt>second improved prompt</prompt>\n"
         "...\n"
         "Output NOTHING outside the <prompt> blocks."),
        ("human",
         "Current prompt:\n{prompt}\n\n"
         "Biased protected attributes:\n{biases}\n\n"
         "Critique:\n{critique}\n\n"
         "Write {q} improved prompts that address this critique."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _paraphrase_chain(model: str):
    """ProTeGi step 3 — Monte Carlo expansion via semantic paraphrase."""
    template = ChatPromptTemplate([
        ("system",
         f"You are an expert prompt engineer. {_target_mention()}"
         "Generate a SEMANTIC PARAPHRASE of a given prompt while preserving its "
         "instructional intent. The paraphrase MUST:\n"
         "- Include the original task statement VERBATIM (paraphrase only the\n"
         "  surrounding instructions)\n"
         "- Restructure or rephrase the role, hints, scaffolding, decomposition\n"
         "- Keep the explicit ban on protected attributes\n"
         "- Be meaningfully different from the input but with the same goal\n\n"
         "Do NOT add execution rules. Return ONLY the paraphrased prompt, no preamble."),
        ("human", "Original prompt:\n{prompt}\n\nGenerate a paraphrase."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _format_biases(biased: list[dict]) -> str:
    if not biased:
        return "None — the code is unbiased on all PROTECTED attributes."
    lines = []
    for i, b in enumerate(biased, 1):
        lines.append(f"{i}. attribute={b['attribute']!r}: "
                     f"{b['left_call']} differs from {b['right_call']}")
    return "\n".join(lines)


def _extract_tagged(text: str, tag: str, min_len: int = 10) -> list[str]:
    """<tag>...</tag> with fallback to numbered list, then to whole-text."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = [m.strip() for m in re.findall(pattern, text, re.DOTALL)
               if m.strip() and len(m.strip()) >= min_len]
    if matches:
        return matches
    items = re.split(r'\n\s*\d+[\.\)]\s+', text)
    items = [it.strip() for it in items if len(it.strip()) >= min_len]
    if len(items) > 1:
        return items
    return [text.strip()] if len(text.strip()) >= min_len else []


# ── Main loop ────────────────────────────────────────────────────────────────
def run_apo(
    question: str,
    global_pool: dict | None,
    rng_dev: _random.Random,
    target_chain,
    model_optimizer: str,
    optimizer_chain=None,
    num_gradients: int = NUM_GRADIENTS,
    num_edits: int = NUM_EDITS,
    num_paraphrases: int = NUM_PARAPHRASES,
    beam_width: int = BEAM_WIDTH,
    max_iters: int = MAX_ITERS,
    max_no_improve: int = MAX_NO_IMPROVE,
    k_dev: int = 2,
) -> dict:
    """Per-task APO/ProTeGi adapted to bias mitigation.

    Beam search of width `beam_width`. Each iteration:

      1. For every beam member with bias on dev:
         a. `num_gradients` distinct critiques (single LLM call).
         b. For each critique, `num_edits` improved prompts.
      2. MC expansion: `num_paraphrases` paraphrases per gradient candidate.
      3. Score the entire pool on dev.
      4. Top-`beam_width` becomes the next beam (no elitism — paper-faithful).
      5. Early stop on dev=1.0 or `max_no_improve` stagnant iters.

    Returns the best member of the final pool."""
    print(f"\n[APO] Avvio — beam={beam_width}, m={num_gradients}, q={num_edits}, "
          f"r={num_paraphrases}, max_iters={max_iters}, k_dev={k_dev}")

    grad_chain = _gradient_chain(model_optimizer)
    edit_chain = _edit_chain(model_optimizer)
    para_chain = _paraphrase_chain(model_optimizer)

    # ── Initial beam: enriched prompt via optimizer (or raw question fallback) ──
    if optimizer_chain is not None:
        p0 = optimizer_chain.invoke({"problem": question})
    else:
        p0 = question
    p0_code  = extract_code(target_chain.invoke({"user_prompt": p0}))
    p0_score = evaluate_code_bias(p0_code, global_pool, rng_dev, k=k_dev)
    beam: list[dict] = [{"prompt": p0, "code": p0_code, "dev_score": p0_score, "source": "init"}]
    best_so_far = p0_score
    no_improve  = 0
    iterations: list[dict] = []
    last_pool: list[dict] = list(beam)

    for t in range(1, max_iters + 1):
        print(f"  [APO] Iter {t}/{max_iters} — espansione beam ({len(beam)} membri) …")

        gradient_candidates: list[dict] = []
        for member in beam:
            if member["dev_score"] >= 1.0:
                continue  # already unbiased on the dev signal — no gradient

            biased   = get_biased_attributes(member["code"], global_pool, rng_dev)
            bias_str = _format_biases(biased)

            grad_resp = grad_chain.invoke({
                "prompt": member["prompt"],
                "code":   member["code"],
                "biases": bias_str,
                "m":      num_gradients,
            })
            gradients = _extract_tagged(grad_resp, "critique")[:num_gradients]
            if not gradients:
                continue

            for grad in gradients:
                edit_resp = edit_chain.invoke({
                    "prompt":   member["prompt"],
                    "biases":   bias_str,
                    "critique": grad,
                    "q":        num_edits,
                })
                improved = _extract_tagged(edit_resp, "prompt")[:num_edits]
                for imp in improved:
                    code  = extract_code(target_chain.invoke({"user_prompt": imp}))
                    score = evaluate_code_bias(code, global_pool, rng_dev, k=k_dev)
                    gradient_candidates.append({
                        "prompt":    imp,
                        "code":      code,
                        "dev_score": score,
                        "source":    "gradient",
                        "critique":  grad,
                    })

        # Step 3 — Monte Carlo paraphrase expansion
        paraphrase_candidates: list[dict] = []
        for cand in gradient_candidates:
            for _ in range(num_paraphrases):
                para  = para_chain.invoke({"prompt": cand["prompt"]})
                code  = extract_code(target_chain.invoke({"user_prompt": para}))
                score = evaluate_code_bias(code, global_pool, rng_dev, k=k_dev)
                paraphrase_candidates.append({
                    "prompt":    para,
                    "code":      code,
                    "dev_score": score,
                    "source":    "paraphrase",
                })

        full_pool = gradient_candidates + paraphrase_candidates
        if not full_pool:
            print("  [APO] Nessun candidato generato — stop.")
            break

        # Step 4 — successor selection (top-b on dev, no elitism)
        full_pool.sort(key=lambda c: c["dev_score"], reverse=True)
        beam      = full_pool[:beam_width]
        last_pool = full_pool
        best_dev  = beam[0]["dev_score"]

        n_grad = sum(1 for c in full_pool if c["source"] == "gradient")
        n_para = sum(1 for c in full_pool if c["source"] == "paraphrase")
        print(f"  [APO] Iter {t}: best_dev={best_dev:.3f}  pool={len(full_pool)} (grad={n_grad}, para={n_para})")

        iterations.append({
            "iter":         t,
            "best_dev":     best_dev,
            "pool_size":    len(full_pool),
            "n_gradient":   n_grad,
            "n_paraphrase": n_para,
            "beam":         [
                {"prompt": c["prompt"], "code": c["code"],
                 "dev_score": c["dev_score"], "source": c["source"]}
                for c in beam
            ],
        })

        if best_dev > best_so_far:
            best_so_far = best_dev
            no_improve  = 0
            print("  [APO] Miglioramento — reset contatore.")
        else:
            no_improve += 1
            print(f"  [APO] Nessun miglioramento ({no_improve}/{max_no_improve}).")
            if no_improve >= max_no_improve:
                print(f"  [APO] Convergenza dopo {t} iterazioni.")
                break

    best = max(last_pool, key=lambda c: c["dev_score"])
    print(f"[APO] Best: dev={best['dev_score']:.3f}")

    return {
        "best_prompt": best["prompt"],
        "best_code":   best["code"],
        "best_dev":    best["dev_score"],
        "iterations":  iterations,
    }
