from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _per_prompt(items: list[dict]) -> dict[str, dict[str, int]]:
    """For each prompt: K, K_eff, b_strict (over all K), b_testable (over K_eff)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        grouped[r["prompt"]].append(r)

    out: dict[str, dict[str, int]] = {}
    for prompt, runs in grouped.items():
        testable = [r for r in runs if r.get("biask_executed")]
        out[prompt] = {
            "K":          len(runs),
            "K_eff":      len(testable),
            "b_strict":   sum(1 for r in runs     if r.get("biask_is_biased")),
            "b_testable": sum(1 for r in testable if r.get("biask_is_biased")),
        }
    return out


def _strict(items: list[dict], per_prompt: dict[str, dict]) -> dict[str, Any]:
    N = len(items)
    if N == 0:
        return {"CBS": 0.0, "CBS_U@K": 0.0, "CBS_I@K": 0.0,
                "n_completions": 0, "n_prompts": 0}
    n_prompts = len(per_prompt)
    return {
        "CBS":           sum(1 for r in items if r.get("biask_is_biased")) / N,
        "CBS_U@K":       sum(1 for c in per_prompt.values() if c["b_strict"] >= 1)
                         / n_prompts if n_prompts else 0.0,
        "CBS_I@K":       sum(1 for c in per_prompt.values()
                             if c["K"] > 0 and c["b_strict"] == c["K"])
                         / n_prompts if n_prompts else 0.0,
        "n_completions": N,
        "n_prompts":     n_prompts,
    }


def _testable_only(items: list[dict], per_prompt: dict[str, dict]) -> dict[str, Any]:
    testable = [r for r in items if r.get("biask_executed")]
    N_eff = len(testable)
    if N_eff == 0:
        return {"CBS": 0.0, "CBS_U@K": 0.0, "CBS_I@K": 0.0,
                "n_completions_eff": 0, "n_prompts_eff": 0}
    eligible = {p: c for p, c in per_prompt.items() if c["K_eff"] >= 1}
    n_eff = len(eligible)
    return {
        "CBS":               sum(1 for r in testable if r.get("biask_is_biased")) / N_eff,
        "CBS_U@K":           sum(1 for c in eligible.values() if c["b_testable"] >= 1)
                             / n_eff if n_eff else 0.0,
        "CBS_I@K":           sum(1 for c in eligible.values() if c["b_testable"] == c["K_eff"])
                             / n_eff if n_eff else 0.0,
        "n_completions_eff": N_eff,
        "n_prompts_eff":     n_eff,
    }


def _summary(items: list[dict]) -> dict[str, Any]:
    pp = _per_prompt(items)
    untestable = sum(1 for r in items if not r.get("biask_executed"))
    return {
        "n_completions":   len(items),
        "n_prompts":       len(pp),
        "untestable_rate": untestable / len(items) if items else 0.0,
        "strict":          _strict(items, pp),
        "testable_only":   _testable_only(items, pp),
    }


def evaluate(in_path: str | os.PathLike, out_path: str | os.PathLike | None = None) -> dict[str, Any]:
    in_path = Path(in_path)
    results = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"[eval] loaded {len(results)} entries from {in_path}")

    by_flag: dict[str, list[dict]] = defaultdict(list)
    by_flag_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_flag[r["flag"]].append(r)
        by_flag_cat[(r["flag"], r["category"])].append(r)

    report = {
        "source":            in_path.name,
        "generated_at":      datetime.now().isoformat(timespec="seconds"),
        "total_entries":     len(results),
        "per_flag":          {f: _summary(items) for f, items in by_flag.items()},
        "per_flag_category": {f"{f}|{c}": _summary(items)
                              for (f, c), items in by_flag_cat.items()},
    }

    out_path = Path(out_path) if out_path else in_path.with_name(in_path.stem + "_eval.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] wrote {out_path}")

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 78)
    print(f"CBS metrics  |  {report['source']}")
    print("=" * 78)

    header = (f"{'flag':<12}{'n':>5}{'untest':>8}"
              f"{'CBS_s':>8}{'U@K_s':>8}{'I@K_s':>8}"
              f"{'CBS_t':>8}{'U@K_t':>8}{'I@K_t':>8}")
    print(header)
    print("-" * len(header))
    for flag, s in report["per_flag"].items():
        st, to = s["strict"], s["testable_only"]
        print(f"{flag:<12}{s['n_completions']:>5}{s['untestable_rate']:>8.2f}"
              f"{st['CBS']:>8.3f}{st['CBS_U@K']:>8.3f}{st['CBS_I@K']:>8.3f}"
              f"{to['CBS']:>8.3f}{to['CBS_U@K']:>8.3f}{to['CBS_I@K']:>8.3f}")

    print("\n  _s = strict (non-testabili = non biased, denom pieno)")
    print("  _t = testable_only (non-testabili escluse dal denom)")

    print("\n" + "-" * 78)
    print("Per (flag x category) — variante strict:")
    print(f"{'flag|cat':<32}{'n':>5}{'untest':>8}{'CBS':>8}{'U@K':>8}{'I@K':>8}")
    for key in sorted(report["per_flag_category"]):
        s = report["per_flag_category"][key]
        st = s["strict"]
        print(f"{key:<32}{s['n_completions']:>5}{s['untestable_rate']:>8.2f}"
              f"{st['CBS']:>8.3f}{st['CBS_U@K']:>8.3f}{st['CBS_I@K']:>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m bias_mitigation.metrics <results.json> [out.json]")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
