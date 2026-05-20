#!/usr/bin/env python
"""Final report writer — runs after PIPELINE_DONE.

Aggregates all results from outputs/{ac_baseline,ac_fixed,iql,awr,cql,td3_strict,phase2,extended200k}
into a single markdown report at outputs/FINAL_REPORT_20260521.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path("/home/coder/code/Evo-RL-quick")
SOURCES = [
    ("TD3+BC-unfixed", REPO / "outputs/ac_baseline_20260521/results.json"),
    ("TD3+BC-fixed", REPO / "outputs/ac_fixed_20260521/results.json"),
    ("IQL", REPO / "outputs/iql_20260521/results.json"),
    ("AWR", REPO / "outputs/awr_20260521/results.json"),
    ("CQL", REPO / "outputs/cql_20260521/results.json"),
    ("TD3-strict", REPO / "outputs/td3_strict_20260521/results.json"),
    ("Phase2", REPO / "outputs/phase2_20260521/results.json"),
    ("Extended200k", REPO / "outputs/extended200k_20260521/results.json"),
]


def composite_score(r: dict) -> float:
    return r["ref_mse"] * 100.0 - r.get("q_gap", 0.0) * 1.0


def main():
    all_results = []
    by_algo = {}
    for algo, path in SOURCES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"failed to read {path}: {e}", file=sys.stderr)
            continue
        rows = []
        for r in data:
            r = dict(r)
            r["_algo"] = algo
            r["_score"] = composite_score(r)
            rows.append(r)
            all_results.append(r)
        by_algo[algo] = rows

    lines = [
        "# RLT 2026-05-21 Overnight Pipeline Report",
        f"\n_Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}_",
        "",
        "## Pipeline stages",
        "1. TD3+BC unfixed (12 cells × 50k) — current code on fixed cache",
        "2. TD3+BC fixed (12 cells × 50k) — AC1+AC2+AC3+AC4 patches applied",
        "3. IQL (4 cells × 50k) — implicit Q-learning with expectile V",
        "4. AWR (4 cells × 50k) — advantage-weighted regression",
        "5. CQL (4 cells × 50k) — conservative Q-learning with OOD penalty",
        "6. TD3-strict (3 cells × 50k) — TD3+BC at β∈{0, 0.01, 0.05}",
        "7. Phase 2 (6 cells × 50k) — fine sweep around overall best",
        "8. Extended 200k — winner config trained 4× longer",
        "",
        "## Headline numbers — best per algorithm",
    ]

    # Best per algo
    for algo, rows in by_algo.items():
        if not rows:
            continue
        best = min(rows, key=lambda r: r["_score"])
        lines.append(f"- **{algo}**: `{best['name']}` ref_mse={best['ref_mse']:.5f} "
                     f"q_gap={best.get('q_gap', 0.0):+.5f} score={best['_score']:.4f}")

    lines += ["", "## Global top 10 by composite score (lower = better)", "",
              "| Rank | Algo | Name | ref_mse | q_gap | score |", "|---|---|---|---|---|---|"]
    sorted_all = sorted(all_results, key=lambda r: r["_score"])[:10]
    for i, r in enumerate(sorted_all, 1):
        lines.append(f"| {i} | {r['_algo']} | `{r['name']}` | {r['ref_mse']:.5f} | "
                     f"{r.get('q_gap', 0.0):+.5f} | {r['_score']:.4f} |")

    # A/B: unfixed vs fixed best
    if "TD3+BC-unfixed" in by_algo and "TD3+BC-fixed" in by_algo:
        u = min(by_algo["TD3+BC-unfixed"], key=lambda r: r["_score"])
        f = min(by_algo["TD3+BC-fixed"], key=lambda r: r["_score"])
        lines += [
            "",
            "## A/B: AC1–AC4 fix impact",
            f"- **Unfixed best**: `{u['name']}` ref_mse={u['ref_mse']:.5f} q_gap={u.get('q_gap',0):+.5f}",
            f"- **Fixed best**: `{f['name']}` ref_mse={f['ref_mse']:.5f} q_gap={f.get('q_gap',0):+.5f}",
            f"- Δ ref_mse: {(f['ref_mse'] - u['ref_mse'])*100:+.2f}% (negative = improvement)",
            f"- Δ q_gap: {(f.get('q_gap',0) - u.get('q_gap',0)):+.5f} (positive = improvement)",
        ]

    # Full per-algo tables
    lines += ["", "## Full per-algo tables", ""]
    for algo, rows in by_algo.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda r: r["_score"])
        lines += [f"### {algo}", ""]
        lines += ["| Rank | Name | ref_mse | q_gap | actor_loss | elapsed |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(sorted_rows, 1):
            lines.append(f"| {i} | `{r['name']}` | {r['ref_mse']:.5f} | "
                         f"{r.get('q_gap', 0.0):+.5f} | {r.get('final_actor_loss', 0):+.3f} | "
                         f"{r.get('elapsed_sec', 0):.0f}s |")
        lines.append("")

    # Boundary jump baseline
    bj = REPO / "outputs/boundary_jump/vla_only_20260521.json"
    if bj.exists():
        data = json.loads(bj.read_text())
        lines += ["", "## Chunk-boundary jump — VLA-only baseline (cotrained pi0.5, 30 ep)", ""]
        lines += ["| Metric | mean | p50 | p95 | max |", "|---|---|---|---|---|"]
        for k, v in data["summary"].items():
            lines.append(f"| {k} | {v['mean']:.3f} | {v['p50']:.3f} | "
                         f"{v['p95']:.3f} | {v['max']:.3f} |")

    # Write report
    out = REPO / "outputs/FINAL_REPORT_20260521.md"
    out.write_text("\n".join(lines))
    print(f"WROTE {out}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
