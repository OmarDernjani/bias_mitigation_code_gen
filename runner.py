from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Load .env BEFORE any module-level env reads in submodules (utils, metamorphic,
# algorithms). Idempotent — also called by package __init__ for import-time use.
from dotenv import load_dotenv
PKG_DIR = Path(__file__).parent
load_dotenv(PKG_DIR / ".env")

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from algorithms.ape import run_ape
from algorithms.apo import run_apo
from metamorphic import build_global_pool, compute_biask
from prompts import FLAGS_DYNAMIC, FLAGS_STATIC, system_prompt
from utils import build_optimizer_chain, build_target_chain

MODEL_TARGET     = os.getenv("MODEL_TARGET", "llama3.1:8b")
MODEL_OPTIMIZER  = os.getenv("MODEL_OPTIMIZER", "mistral-nemo")
NUM_CTX          = int(os.getenv("NUM_CTX", 8192))
K_TIMES          = int(os.getenv("CBS_K_TIMES", 3))
K_DEV            = int(os.getenv("CBS_K_DEV", 2))
N_PER_CATEGORY   = int(os.getenv("CBS_N_PER_CATEGORY", 5))  # campiona solo 5 prompt per categoria
FLAGS            = os.getenv("CBS_FLAGS", "zero-shot,human,APE,APO").split(",")
CATEGORIES       = [c.strip() for c in os.getenv("CBS_CATEGORIES", "").split(",") if c.strip()] or None
DATASET_PATH     = Path(os.getenv("CBS_DATASET", PKG_DIR / "dataset.json"))
OUTPUT_DIR       = Path(os.getenv("CBS_OUTPUT_DIR", "result"))
MAX_CONST_COMBOS = int(os.getenv("CBS_MAX_CONST_COMBINATIONS", 0)) or None
SEED             = int(os.getenv("CBS_SEED", 42))


_FENCED_PATTERNS = (
    re.compile(r"```python\s*\n?(.*?)```", re.DOTALL),
    re.compile(r"```py\s*\n?(.*?)```",     re.DOTALL),
    re.compile(r"```\s*\n?(.*?)```",       re.DOTALL),
)
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*)$", re.DOTALL)


def _extract_code(response: str) -> str:
    """Pull a Python block out of an LLM response.
    Fenced (longest match) -> unclosed-fence -> raw text."""
    for pat in _FENCED_PATTERNS:
        if (m := pat.findall(response)):
            return max(m, key=len).strip()
    if (m := _OPEN_FENCE.search(response)):
        return m.group(1).strip()
    return response.strip()


def _sample_balanced(data: list[dict], n_per_cat: int, rng: random.Random) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_cat[entry.get("category", "unknown")].append(entry)
    out: list[dict] = []
    for entries in by_cat.values():
        out.extend(rng.sample(entries, min(n_per_cat, len(entries))))
    return out



def _build_static_chain(flag: str):
    template = ChatPromptTemplate([
        ("system", system_prompt(flag)),
        ("human",  "{task}"),
    ])
    llm = ChatOllama(model=MODEL_TARGET, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _generate_static(flag: str, task: str) -> str:
    return _extract_code(_build_static_chain(flag).invoke({"task": task}))


DUMP_EVERY = int(os.getenv("CBS_DUMP_EVERY", 10))


def _dump(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _maybe_dump(results: list[dict], path: Path) -> None:
    """Throttled dump: rewrites the JSON only every DUMP_EVERY appends.
    Phase boundaries should still call _dump unconditionally for safety."""
    if len(results) % DUMP_EVERY == 0:
        _dump(results, path)


def _dump_best_prompts(results: list[dict], path: Path) -> None:
    """One row per (category, prompt, flag): the FINAL prompt sent to the target
    plus the bias outcome. For APE/APO selects the k with the highest dev score;
    for zero-shot/human the user-message is the task itself (the anti-bias
    scaffolding, if any, lives in the system message — same for every k).
    Drops the iteration arrays so the file is human-skimmable for slide prep."""
    flag_order = [f for f in FLAGS if f in FLAGS_STATIC or f in FLAGS_DYNAMIC]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in results:
        grouped[(r["category"], r["prompt"], r["flag"])].append(r)

    rows: list[dict] = []
    for (cat, task, flag), runs in grouped.items():
        if flag in FLAGS_DYNAMIC:
            winner       = max(runs, key=lambda r: r.get("best_dev_score") or 0.0)
            final_prompt = winner.get("best_prompt", "")
        else:
            winner       = runs[0]
            final_prompt = task  

        attrs_seen = sorted({
            a for r in runs for a in (r.get("biask_bias_per_attribute") or {})
        })
        rows.append({
            "category":        cat,
            "task_prompt":     task,
            "flag":            flag,
            "K_total":         len(runs),
            "K_biased":        sum(1 for r in runs if r.get("biask_is_biased")),
            "K_untestable":    sum(1 for r in runs if not r.get("biask_executed")),
            "winning_k":       winner.get("k"),
            "best_dev_score":  winner.get("best_dev_score"),
            "final_prompt":    final_prompt,
            "final_code":      winner.get("completion", ""),
            "bias_attrs_seen": attrs_seen,
        })

    rows.sort(key=lambda r: (
        r["category"],
        r["task_prompt"],
        flag_order.index(r["flag"]) if r["flag"] in flag_order else len(flag_order),
    ))
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _pass1_generate(sampled: list[dict], out_path: Path, rng: random.Random) -> list[dict]:
    """Phase A: static flags. Phase B: bootstrap global pool from phase-A
    completions. Phase C: dynamic flags using the bootstrap pool as the
    bias-signal global pool."""
    results: list[dict] = []
    static_flags  = [f for f in FLAGS if f in FLAGS_STATIC]
    dynamic_flags = [f for f in FLAGS if f in FLAGS_DYNAMIC]
    unknown       = [f for f in FLAGS if f not in FLAGS_STATIC and f not in FLAGS_DYNAMIC]
    if unknown:
        print(f"[pass1] WARNING: ignoro flag sconosciute {unknown}", file=sys.stderr)

    if static_flags:
        for entry in tqdm(sampled, desc="pass1 static"):
            for flag in static_flags:
                for k in range(K_TIMES):
                    try:
                        code = _generate_static(flag, entry["prompt"])
                    except Exception as e:
                        print(f"[gen-err static] {flag} k={k}: {e}", file=sys.stderr)
                        code = ""
                    results.append({
                        "prompt":     entry["prompt"],
                        "category":   entry["category"],
                        "flag":       flag,
                        "k":          k,
                        "completion": code,
                        "model":      MODEL_TARGET,
                    })
                    _maybe_dump(results, out_path)

    boot_pool: dict | None = None
    if dynamic_flags:
        boot_codes = [r["completion"] for r in results if r.get("completion")]
        boot_pool  = build_global_pool(boot_codes) if boot_codes else None
        if boot_pool:
            keys_preview = list(boot_pool)[:20]
            ellipsis     = " …" if len(boot_pool) > 20 else ""
            print(f"[pass1] bootstrap pool: {len(boot_pool)} keys → {keys_preview}{ellipsis}")
        else:
            print("[pass1] bootstrap pool vuoto — APE/APO useranno solo fallback values")

    if dynamic_flags:
        target_chain = build_target_chain(MODEL_TARGET)
        opt_chain    = build_optimizer_chain(MODEL_OPTIMIZER)
        for entry in tqdm(sampled, desc="pass1 dynamic"):
            for flag in dynamic_flags:
                for k in range(K_TIMES):
                    try:
                        if flag == "APE":
                            out = run_ape(
                                question=entry["prompt"],
                                global_pool=boot_pool,
                                rng_dev=rng,
                                target_chain=target_chain,
                                optimizer_chain=opt_chain,
                                model_optimizer=MODEL_OPTIMIZER,
                                k_dev=K_DEV,
                            )
                        else:  # APO
                            out = run_apo(
                                question=entry["prompt"],
                                global_pool=boot_pool,
                                rng_dev=rng,
                                target_chain=target_chain,
                                model_optimizer=MODEL_OPTIMIZER,
                                optimizer_chain=opt_chain,
                                k_dev=K_DEV,
                            )
                        completion  = out["best_code"]
                        best_prompt = out["best_prompt"]
                        best_dev    = out["best_dev"]
                        iterations  = out.get("iterations", [])
                    except Exception as e:
                        print(f"[gen-err dyn] {flag} k={k}: {e}", file=sys.stderr)
                        completion, best_prompt, best_dev, iterations = "", "", 0.0, []
                    results.append({
                        "prompt":         entry["prompt"],
                        "category":       entry["category"],
                        "flag":           flag,
                        "k":              k,
                        "completion":     completion,
                        "best_prompt":    best_prompt,
                        "best_dev_score": best_dev,
                        "iterations":     iterations,
                        "model":          MODEL_TARGET,
                    })
                    _maybe_dump(results, out_path)

    _dump(results, out_path)
    return results


def _pass2_biask(results: list[dict], out_path: Path, rng: random.Random) -> list[dict]:
    codes = [r["completion"] for r in results if r.get("completion")]
    pool  = build_global_pool(codes)
    print(f"[pass2] global pool keys: {list(pool)[:20]}{' …' if len(pool) > 20 else ''}")
    for i, r in enumerate(tqdm(results, desc="pass2 bias@k"), 1):
        r.update(compute_biask(r["completion"], pool, MAX_CONST_COMBOS, rng))
        if i % DUMP_EVERY == 0:
            _dump(results, out_path)
    _dump(results, out_path)
    return results


def run_experiment() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"cbs_results_{datetime.now():%Y%m%d_%H%M%S}.json"

    rng = random.Random(SEED)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if CATEGORIES:
        n_pre = len(dataset)
        dataset = [e for e in dataset if e.get("category") in CATEGORIES]
        print(f"[cbs] filtro categorie {CATEGORIES}: {n_pre} → {len(dataset)} prompt")
    sampled = _sample_balanced(dataset, N_PER_CATEGORY, rng)

    print(f"[cbs] target={MODEL_TARGET}  optimizer={MODEL_OPTIMIZER}")
    print(f"[cbs] flags={FLAGS}  k={K_TIMES}  n_per_cat={N_PER_CATEGORY}  entries={len(sampled)}")
    print(f"[cbs] dataset={DATASET_PATH}  output={out_path}")

    results = _pass1_generate(sampled, out_path, rng)
    best_path = out_path.with_name(out_path.stem.replace("cbs_results", "best_prompts") + ".json")
    _dump_best_prompts(results, best_path)
    print(f"[cbs] best prompts -> {best_path}")
    results = _pass2_biask(results, out_path, rng)
    print(f"[cbs] done: {len(results)} items -> {out_path}")
    return out_path


if __name__ == "__main__":
    run_experiment()
