#!/usr/bin/env python
"""Multi-seed validation of top-K configs across all algos.

After PIPELINE_DONE, picks top-3 by composite score across all baseline
results, then trains each at 3 seeds × 50k steps to quantify run-to-run
variance. Outputs a CSV-like summary with mean ± std per config.

Run automatically by the post-pipeline orchestrator.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

REPO = Path("/home/coder/code/Evo-RL-quick")
CACHE = Path("/home/coder/share/cache/0520_critical_c_rlt_cotrain_fixed")
SRC = REPO / "src"
SCRIPTS = REPO / "scripts/rlt_training"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("multiseed")


def composite_score(r: dict) -> float:
    # Reject diverged runs: legit ref_mse < 1.0 and |q_gap| < 10.0
    ref_mse = r["ref_mse"]
    q_gap = r.get("q_gap", 0.0)
    if ref_mse > 1.0 or abs(q_gap) > 10.0 or ref_mse != ref_mse:
        return float("inf")
    return ref_mse * 100.0 - q_gap * 1.0


def collect_baseline_winners(k: int = 3) -> list[dict]:
    sources = [
        ("TD3+BC-unfixed", REPO / "outputs/ac_baseline_20260521/results.json"),
        ("TD3+BC-fixed", REPO / "outputs/ac_fixed_20260521/results.json"),
        ("IQL", REPO / "outputs/iql_20260521/results.json"),
        ("AWR", REPO / "outputs/awr_20260521/results.json"),
        ("CQL", REPO / "outputs/cql_20260521/results.json"),
        ("TD3-strict", REPO / "outputs/td3_strict_20260521/results.json"),
    ]
    all_rows = []
    for algo, p in sources:
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            r = dict(r); r["_algo"] = algo; r["_score"] = composite_score(r)
            all_rows.append(r)
    return sorted(all_rows, key=lambda r: r["_score"])[:k]


def run_seed(winner: dict, seed: int, results_file: Path):
    torch.manual_seed(seed)
    algo = winner["_algo"]
    cfg = dict(winner["config"])

    if algo in ("TD3+BC-fixed", "TD3+BC-unfixed", "TD3-strict"):
        from ac_baseline_sweep_20260521 import ExperimentConfig, run_experiment
        e = ExperimentConfig(**{k: v for k, v in cfg.items() if k in ExperimentConfig.__dataclass_fields__})
        e.name = f"{winner['name']}_seed{seed}"
        e.gradient_steps = 50_000
        run_experiment(e, str(CACHE), "cuda", results_file)
    elif algo == "IQL":
        from train_iql_baseline import IQLExperimentConfig, run_iql
        e = IQLExperimentConfig(**{k: v for k, v in cfg.items() if k in IQLExperimentConfig.__dataclass_fields__})
        e.name = f"{winner['name']}_seed{seed}"
        e.gradient_steps = 50_000
        run_iql(e, str(CACHE), "cuda", results_file)
    elif algo == "AWR":
        from train_awr_baseline import AWRExperimentConfig, run_awr
        e = AWRExperimentConfig(**{k: v for k, v in cfg.items() if k in AWRExperimentConfig.__dataclass_fields__})
        e.name = f"{winner['name']}_seed{seed}"
        e.gradient_steps = 50_000
        run_awr(e, str(CACHE), "cuda", results_file)
    elif algo == "CQL":
        from train_cql_baseline import CQLExperimentConfig, run_cql
        e = CQLExperimentConfig(**{k: v for k, v in cfg.items() if k in CQLExperimentConfig.__dataclass_fields__})
        e.name = f"{winner['name']}_seed{seed}"
        e.gradient_steps = 50_000
        run_cql(e, str(CACHE), "cuda", results_file)


def main():
    while not (REPO / "outputs/PIPELINE_DONE.txt").exists():
        logger.info("Waiting for PIPELINE_DONE...")
        time.sleep(120)

    top_k = collect_baseline_winners(k=3)
    logger.info("Top-3 winners for multi-seed validation:")
    for w in top_k:
        logger.info("  %s (%s) ref_mse=%.5f q_gap=%+.5f score=%.4f",
                    w["name"], w["_algo"], w["ref_mse"], w.get("q_gap", 0.0), w["_score"])

    out_dir = REPO / "outputs/multiseed_20260521"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"

    for w in top_k:
        for seed in [1001, 2002, 3003]:
            logger.info("Running %s seed=%d", w["name"], seed)
            run_seed(w, seed, results_file)

    # Compute mean / std per winner
    data = json.loads(results_file.read_text())
    by_name = {}
    for r in data:
        base = r["name"].rsplit("_seed", 1)[0]
        by_name.setdefault(base, []).append(r)

    summary_lines = ["# Multi-seed Validation Summary\n",
                     f"_Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}_\n",
                     "## Per-config aggregated stats (3 seeds)\n",
                     "| Config | ref_mse mean ± std | q_gap mean ± std | seeds |",
                     "|---|---|---|---|"]
    import statistics
    for base, rs in by_name.items():
        refs = [r["ref_mse"] for r in rs]
        gaps = [r.get("q_gap", 0.0) for r in rs]
        ref_mean = statistics.mean(refs); ref_std = statistics.pstdev(refs)
        gap_mean = statistics.mean(gaps); gap_std = statistics.pstdev(gaps)
        summary_lines.append(f"| `{base}` | {ref_mean:.5f} ± {ref_std:.5f} | "
                             f"{gap_mean:+.5f} ± {gap_std:.5f} | {len(rs)} |")

    (out_dir / "summary.md").write_text("\n".join(summary_lines))
    logger.info("Wrote multiseed summary")
    (REPO / "outputs/MULTISEED_DONE.txt").write_text("done")


if __name__ == "__main__":
    main()
