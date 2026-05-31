"""APE (Zhou et al. 2022) adapted to per-task bias mitigation.

Each task in dataset.json is its own optimization instance. Candidates are
scored on a bias signal (`evaluate_code_bias`) computed via metamorphic
testing — there is no test-case correctness signal in this experiment."""

from __future__ import annotations

import os
import random as _random

from metamorphic import evaluate_code_bias
from utils import build_variation_chain, extract_code

DEFAULT_N_PROPOSALS = int(os.getenv("APE_N_PROPOSALS", 5))
DEFAULT_N_ITERS     = int(os.getenv("APE_N_ITERS", 2))
DEFAULT_N_KEEP      = int(os.getenv("APE_N_KEEP", 2))


def run_ape(
    question: str,
    global_pool: dict | None,
    rng_dev: _random.Random,
    target_chain,
    optimizer_chain,
    model_optimizer: str,
    n_proposals: int = DEFAULT_N_PROPOSALS,
    n_iters: int = DEFAULT_N_ITERS,
    n_keep: int = DEFAULT_N_KEEP,
    k_dev: int = 2,
) -> dict:
    """Per-task APE.

      1. Proposal: `n_proposals` enriched prompts via `optimizer_chain`.
      2. Score each candidate on the bias signal (mean over `k_dev` metamorphic
         runs).
      3. Iterative MC: keep top-`n_keep`, generate variations, re-score. Repeat
         for `n_iters`.

    Returns the best candidate of the final population."""
    print(f"\n[APE] Avvio — n_proposals={n_proposals}, n_iters={n_iters}, "
          f"n_keep={n_keep}, k_dev={k_dev}")

    variation_chain = build_variation_chain(model_optimizer)
    iterations: list[dict] = []

    # ── Round 0: initial proposals ──
    print(f"  [APE] Round 0: {n_proposals} candidati iniziali …")
    candidates: list[dict] = []
    for _ in range(n_proposals):
        enriched = optimizer_chain.invoke({"problem": question})
        code     = extract_code(target_chain.invoke({"user_prompt": enriched}))
        score    = evaluate_code_bias(code, global_pool, rng_dev, k=k_dev, task_prompt=question)
        candidates.append({"prompt": enriched, "code": code, "dev_score": score})
    iterations.append({
        "round":      0,
        "candidates": [
            {"prompt": c["prompt"], "code": c["code"], "dev_score": c["dev_score"]}
            for c in candidates
        ],
        "best_dev":   max(c["dev_score"] for c in candidates),
    })
    print(f"  [APE] Round 0: best_dev={iterations[-1]['best_dev']:.3f}")

    # ── Iterative MC refinement ──
    for it in range(1, n_iters + 1):
        candidates.sort(key=lambda c: c["dev_score"], reverse=True)
        top   = candidates[:n_keep]
        n_var = max(1, n_proposals // n_keep)
        print(f"  [APE] Round {it}: variazioni dei top-{n_keep} ({n_var} per kept) …")

        new_candidates: list[dict] = []
        for parent in top:
            for _ in range(n_var):
                variation = variation_chain.invoke({
                    "problem":  question,
                    "previous": parent["prompt"],
                })
                code  = extract_code(target_chain.invoke({"user_prompt": variation}))
                score = evaluate_code_bias(code, global_pool, rng_dev, k=k_dev, task_prompt=question)
                new_candidates.append({"prompt": variation, "code": code, "dev_score": score})

        # Elitism: top-k carry over.
        candidates = top + new_candidates
        iterations.append({
            "round":      it,
            "candidates": [
                {"prompt": c["prompt"], "code": c["code"], "dev_score": c["dev_score"]}
                for c in new_candidates
            ],
            "best_dev":   max(c["dev_score"] for c in candidates),
        })
        print(f"  [APE] Round {it}: best_dev={iterations[-1]['best_dev']:.3f}")

    best = max(candidates, key=lambda c: c["dev_score"])
    print(f"[APE] Best: dev={best['dev_score']:.3f}")

    return {
        "best_prompt": best["prompt"],
        "best_code":   best["code"],
        "best_dev":    best["dev_score"],
        "iterations":  iterations,
    }
