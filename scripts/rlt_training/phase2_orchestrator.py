#!/usr/bin/env python
"""Phase 2 orchestrator (runs after all algo baselines complete).

1. Loads all baseline results.json files (TD3+BC unfixed, TD3+BC fixed, IQL, AWR, TD3-strict).
2. Picks the absolute winner by ref_mse (tie-break: q_gap higher = better).
3. Launches a 6-cell phase2 grid around the winner's algo+config.
4. Loads phase2 results, picks phase2-winner.
5. Launches a 200k extended training on the phase2 winner.
6. Writes summary report.

Run as the final stage in the algos_orchestrator tmux session, after all base sweeps.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

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
logger = logging.getLogger("phase2")


def composite_score(r: dict) -> float:
    """Lower = better. Penalize ref_mse strongly, reward positive q_gap."""
    return r["ref_mse"] * 100.0 - r.get("q_gap", 0.0) * 1.0


def load_all_baselines() -> list[dict]:
    """Aggregate all baseline result dicts with an algo label and source path."""
    sources = [
        ("TD3+BC-unfixed", REPO / "outputs/ac_baseline_20260521/results.json"),
        ("TD3+BC-fixed", REPO / "outputs/ac_fixed_20260521/results.json"),
        ("IQL", REPO / "outputs/iql_20260521/results.json"),
        ("AWR", REPO / "outputs/awr_20260521/results.json"),
        ("TD3-strict", REPO / "outputs/td3_strict_20260521/results.json"),
    ]
    out: list[dict] = []
    for algo, path in sources:
        if not path.exists():
            logger.warning("missing baseline file %s — skipping", path)
            continue
        for r in json.loads(path.read_text()):
            r = dict(r)
            r["_algo"] = r.get("algo", algo)
            r["_source"] = str(path)
            r["_score"] = composite_score(r)
            out.append(r)
    return out


def pick_winner(results: list[dict]) -> dict:
    sorted_res = sorted(results, key=lambda r: r["_score"])
    if not sorted_res:
        raise RuntimeError("No baselines found")
    winner = sorted_res[0]
    logger.info("WINNER: %s (algo=%s) score=%.4f ref_mse=%.5f q_gap=%+.5f",
                winner["name"], winner["_algo"], winner["_score"],
                winner["ref_mse"], winner.get("q_gap", 0.0))
    logger.info("Top-5 by score:")
    for r in sorted_res[:5]:
        logger.info("  %-30s algo=%-15s score=%.4f ref_mse=%.5f q_gap=%+.5f",
                    r["name"], r["_algo"], r["_score"], r["ref_mse"], r.get("q_gap", 0.0))
    return winner


def build_phase2_grid_for_td3_bc(winner: dict) -> list:
    """6-cell grid around a TD3+BC winner: β, lr, ref_dropout fine."""
    from ac_baseline_sweep_20260521 import ExperimentConfig
    base_cfg = winner["config"]
    base = ExperimentConfig(**{k: v for k, v in base_cfg.items() if k in ExperimentConfig.__dataclass_fields__})
    exps = []
    # Beta fine
    base_beta = base.beta
    for b in [max(0.01, base_beta * 0.5), base_beta * 1.5, max(0.01, base_beta * 0.3)]:
        c = deepcopy(base)
        c.beta = round(b, 3)
        c.name = f"p2_b{c.beta}"
        exps.append(c)
    # LR
    for lr in [1e-4, 1e-3]:
        c = deepcopy(base)
        c.actor_lr = lr
        c.critic_lr = lr
        c.name = f"p2_lr{lr:g}"
        exps.append(c)
    # Ref dropout
    c = deepcopy(base)
    c.ref_dropout_p = 0.0 if base.ref_dropout_p > 0.4 else 0.5
    c.name = f"p2_rd{c.ref_dropout_p}"
    exps.append(c)
    return exps


def build_phase2_grid_for_iql(winner: dict) -> list:
    from train_iql_baseline import IQLExperimentConfig
    base_cfg = winner["config"]
    base = IQLExperimentConfig(**{k: v for k, v in base_cfg.items() if k in IQLExperimentConfig.__dataclass_fields__})
    exps = []
    for te in [max(0.5, base.expectile_tau - 0.1), min(0.95, base.expectile_tau + 0.1), 0.95]:
        c = deepcopy(base); c.expectile_tau = round(te, 2); c.name = f"p2_te{c.expectile_tau}"
        exps.append(c)
    for t in [base.awr_temp * 0.3, base.awr_temp * 3.0]:
        c = deepcopy(base); c.awr_temp = round(t, 2); c.name = f"p2_bawr{c.awr_temp}"
        exps.append(c)
    c = deepcopy(base); c.actor_lr = 1e-4; c.critic_lr = 1e-4; c.value_lr = 1e-4; c.name = "p2_lr1e-4"
    exps.append(c)
    return exps


def build_phase2_grid_for_awr(winner: dict) -> list:
    from train_awr_baseline import AWRExperimentConfig
    base_cfg = winner["config"]
    base = AWRExperimentConfig(**{k: v for k, v in base_cfg.items() if k in AWRExperimentConfig.__dataclass_fields__})
    exps = []
    for t in [base.awr_temp * 0.3, base.awr_temp * 0.5, base.awr_temp * 2.0, base.awr_temp * 5.0]:
        c = deepcopy(base); c.awr_temp = round(t, 2); c.name = f"p2_bawr{c.awr_temp}"
        exps.append(c)
    for lr in [1e-4, 1e-3]:
        c = deepcopy(base); c.actor_lr = lr; c.critic_lr = lr; c.name = f"p2_lr{lr:g}"
        exps.append(c)
    return exps


def run_phase2(winner: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"
    algo = winner["_algo"]
    logger.info("Phase 2: winner algo = %s", algo)

    if algo == "TD3+BC-fixed" or algo == "TD3+BC-unfixed" or algo == "TD3-strict":
        from ac_baseline_sweep_20260521 import run_experiment
        grid = build_phase2_grid_for_td3_bc(winner)
        for i, e in enumerate(grid):
            e.gradient_steps = 50_000
            logger.info("p2 cell %d/%d: %s", i+1, len(grid), e.name)
            run_experiment(e, str(CACHE), "cuda", results_file)
    elif algo == "IQL":
        from train_iql_baseline import run_iql
        grid = build_phase2_grid_for_iql(winner)
        for i, e in enumerate(grid):
            e.gradient_steps = 50_000
            logger.info("p2 cell %d/%d: %s", i+1, len(grid), e.name)
            run_iql(e, str(CACHE), "cuda", results_file)
    elif algo == "AWR":
        from train_awr_baseline import run_awr
        grid = build_phase2_grid_for_awr(winner)
        for i, e in enumerate(grid):
            e.gradient_steps = 50_000
            logger.info("p2 cell %d/%d: %s", i+1, len(grid), e.name)
            run_awr(e, str(CACHE), "cuda", results_file)
    else:
        raise ValueError(f"Unknown algo {algo}")

    return results_file


def pick_phase2_winner(p2_file: Path, baseline_winner: dict) -> dict:
    if not p2_file.exists():
        return baseline_winner
    p2_results = [dict(r, _algo=baseline_winner["_algo"], _source=str(p2_file)) for r in json.loads(p2_file.read_text())]
    for r in p2_results:
        r["_score"] = composite_score(r)
    # Compare phase2 best vs original winner
    p2_sorted = sorted(p2_results, key=lambda r: r["_score"])
    p2_best = p2_sorted[0]
    if p2_best["_score"] < baseline_winner["_score"]:
        logger.info("Phase 2 improved: %s (score %.4f → %.4f)",
                    p2_best["name"], baseline_winner["_score"], p2_best["_score"])
        return p2_best
    logger.info("Phase 2 didn't beat baseline; sticking with %s", baseline_winner["name"])
    return baseline_winner


def run_extended_200k(winner: dict, out_dir: Path):
    """Run 200k extended training on the winning config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"
    algo = winner["_algo"]
    logger.info("EXTENDED 200k on winner algo=%s name=%s", algo, winner["name"])

    if algo in ("TD3+BC-fixed", "TD3+BC-unfixed", "TD3-strict"):
        from ac_baseline_sweep_20260521 import ExperimentConfig, run_experiment
        cfg = winner["config"]
        e = ExperimentConfig(**{k: v for k, v in cfg.items() if k in ExperimentConfig.__dataclass_fields__})
        e.name = f"extended200k_{winner['name']}"
        e.gradient_steps = 200_000
        run_experiment(e, str(CACHE), "cuda", results_file)
    elif algo == "IQL":
        from train_iql_baseline import IQLExperimentConfig, run_iql
        cfg = winner["config"]
        e = IQLExperimentConfig(**{k: v for k, v in cfg.items() if k in IQLExperimentConfig.__dataclass_fields__})
        e.name = f"extended200k_{winner['name']}"
        e.gradient_steps = 200_000
        run_iql(e, str(CACHE), "cuda", results_file)
    elif algo == "AWR":
        from train_awr_baseline import AWRExperimentConfig, run_awr
        cfg = winner["config"]
        e = AWRExperimentConfig(**{k: v for k, v in cfg.items() if k in AWRExperimentConfig.__dataclass_fields__})
        e.name = f"extended200k_{winner['name']}"
        e.gradient_steps = 200_000
        run_awr(e, str(CACHE), "cuda", results_file)
    return results_file


def write_final_report(baseline_winner: dict, phase2_winner: dict, ext_file: Path):
    report = REPO / "outputs/PHASE2_REPORT.md"
    lines = [
        "# Phase 2 + Extended Training Report",
        f"\n_{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}_",
        "",
        "## Baseline winner",
        f"- name: `{baseline_winner['name']}`",
        f"- algo: **{baseline_winner['_algo']}**",
        f"- ref_mse: {baseline_winner['ref_mse']:.5f}",
        f"- q_gap: {baseline_winner.get('q_gap', 0.0):+.5f}",
        "",
        "## Phase 2 winner",
        f"- name: `{phase2_winner['name']}`",
        f"- algo: **{phase2_winner['_algo']}**",
        f"- ref_mse: {phase2_winner['ref_mse']:.5f}",
        f"- q_gap: {phase2_winner.get('q_gap', 0.0):+.5f}",
        f"- score (lower=better): {phase2_winner['_score']:.4f} vs baseline {baseline_winner['_score']:.4f}",
        "",
        "## Extended 200k result",
    ]
    if ext_file.exists():
        ext = json.loads(ext_file.read_text())
        if ext:
            r = ext[-1]
            lines += [
                f"- name: `{r['name']}`",
                f"- ref_mse: {r['ref_mse']:.5f}",
                f"- q_gap: {r.get('q_gap', 0.0):+.5f}",
                f"- elapsed: {r['elapsed_sec']:.0f}s",
            ]
    report.write_text("\n".join(lines))
    logger.info("Wrote %s", report)


def main():
    logger.info("Phase 2 orchestrator starting")
    while not (REPO / "outputs/ALL_ALGOS_DONE.txt").exists():
        logger.info("Waiting for ALL_ALGOS_DONE...")
        time.sleep(60)
    baselines = load_all_baselines()
    if not baselines:
        logger.error("No baseline results found")
        return
    logger.info("Loaded %d baseline rows", len(baselines))
    winner = pick_winner(baselines)

    p2_dir = REPO / "outputs/phase2_20260521"
    p2_results = run_phase2(winner, p2_dir)
    p2_winner = pick_phase2_winner(p2_results, winner)

    ext_dir = REPO / "outputs/extended200k_20260521"
    ext_file = run_extended_200k(p2_winner, ext_dir)
    write_final_report(winner, p2_winner, ext_file)
    (REPO / "outputs/PHASE2_DONE.txt").write_text("done")
    logger.info("PHASE2_DONE")


if __name__ == "__main__":
    main()
