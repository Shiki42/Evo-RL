#!/usr/bin/env python
"""Phase 3 — explore around the champion p2_rd0.0 config.

Champion config (from Phase 2): TD3+BC fixed, β=1.0, ref_dropout=0.0,
residual MLP 3L/256h, lr=3e-4. Extended-200k gave ref_mse=0.00007.

Phase 3 holds β=1.0 + rd=0.0 + fixed code, sweeps capacity and seeds:
- depth: L2, L4
- width: H384, H512
- 3-seed validation of the exact champion at 50k
- one 400k ultra-extended run on the champion

Run after multiseed2 finishes.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path("/home/coder/code/Evo-RL-quick")
CACHE = "/home/coder/share/cache/0520_critical_c_rlt_cotrain_fixed"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/rlt_training"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("phase3")

from ac_baseline_sweep_20260521 import ExperimentConfig, run_experiment


def champion_base(**overrides) -> ExperimentConfig:
    """Champion: β=1.0, rd=0.0, residual 3L/256h, lr=3e-4."""
    cfg = dict(
        name="champion", beta=1.0, ref_dropout_p=0.0,
        actor_hidden=256, actor_layers=3, actor_residual=True,
        critic_hidden=256, critic_layers=3, critic_residual=True,
        actor_lr=3e-4, critic_lr=3e-4, gradient_steps=50_000,
    )
    cfg.update(overrides)
    return ExperimentConfig(**cfg)


def build_phase3_grid() -> list[ExperimentConfig]:
    exps = []
    # Depth at champion settings
    exps.append(champion_base(name="p3_L2", actor_layers=2, critic_layers=2))
    exps.append(champion_base(name="p3_L4", actor_layers=4, critic_layers=4))
    # Width
    exps.append(champion_base(name="p3_H384", actor_hidden=384, critic_hidden=384))
    exps.append(champion_base(name="p3_H512", actor_hidden=512, critic_hidden=512))
    # 3-seed validation of the exact champion
    for seed in [1001, 2002, 3003]:
        exps.append(champion_base(name=f"p3_champion_seed{seed}"))
    return exps


def main():
    out = REPO / "outputs/phase3_20260521"
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"

    grid = build_phase3_grid()
    logger.info("Phase 3: %d cells x 50k", len(grid))
    for i, exp in enumerate(grid):
        logger.info("p3 cell %d/%d: %s", i + 1, len(grid), exp.name)
        run_experiment(exp, CACHE, "cuda", results_file)

    # Champion ultra-extended 400k
    logger.info("Phase 3: 400k ultra-extended on champion")
    ultra = champion_base(name="p3_champion_400k", gradient_steps=400_000)
    run_experiment(ultra, CACHE, "cuda", results_file)

    # Summary
    results = sorted(json.loads(results_file.read_text()), key=lambda x: x["ref_mse"])
    logger.info("=" * 70)
    logger.info("PHASE 3 SUMMARY (by ref_mse)")
    for r in results:
        logger.info("%-26s ref_mse=%.6f q_gap=%+.5f actor=%.4f time=%.0fs",
                    r["name"], r["ref_mse"], r["q_gap"], r["final_actor_loss"], r["elapsed_sec"])
    (REPO / "outputs/PHASE3_DONE.txt").write_text("done")
    logger.info("PHASE3_DONE")


if __name__ == "__main__":
    main()
