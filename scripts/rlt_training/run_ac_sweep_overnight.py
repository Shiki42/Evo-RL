#!/usr/bin/env python
"""AC hyperparameter mini-sweep on a freshly built RL transition cache.

Designed for the 2026-05-02 overnight pipeline: after picking the best RL token
ckpt and rebuilding a transition cache from it, this script sweeps a small
beta x architecture grid in serial subprocesses. Each cell runs
train_chunk_actor_critic.py for N gradient steps, then parses the final
'Eval: ...' log line for q_gap / ref_mse / expert_mse / td_error.

Past sweep history (docs/rlt/ac_training_findings_timeline.md):
- beta sweet spot: plateau in [0.1, 1.0]; pick 0.3 by default
- ref_dropout_p stays 0.5
- focus on ref_mse + q values (not ref_dropped_mse)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging

logger = configure_logging("ac_sweep")

EVAL_LINE_RE = re.compile(
    r"Eval:\s+expert_mse=([-\d.]+)\s+ref_mse=([-\d.]+)\s+q_policy=([-\d.]+)\s+q_expert=([-\d.]+)\s+q_gap=([-\d.]+)\s+td_err=([-\d.]+)"
)


@dataclass
class AcCell:
    name: str
    beta: float
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 64


def build_grid() -> list[AcCell]:
    cells: list[AcCell] = []
    # Plateau betas (cheap, 3 cells)
    for beta in [0.1, 0.3, 1.0]:
        cells.append(AcCell(name=f"b{beta}", beta=beta))
    return cells


def run_cell(args: argparse.Namespace, cell: AcCell) -> dict:
    out_dir = Path(args.output_root) / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        sys.executable, "-u",
        str(SCRIPT_ROOT / "train_chunk_actor_critic.py"),
        "--transition-cache-dir", args.transition_cache_dir,
        "--output-dir", str(out_dir),
        "--device", "cuda",
        "--gradient-steps", str(args.gradient_steps),
        "--batch-size", str(cell.batch_size),
        "--beta", str(cell.beta),
        "--actor-lr", str(cell.actor_lr),
        "--critic-lr", str(cell.critic_lr),
        "--log-every", "500",
        "--save-every", str(max(5000, args.gradient_steps)),
        "--eval-every", str(max(5000, args.gradient_steps // 6)),
    ]

    logger.info("[%s] launching: %s", cell.name, " ".join(cmd))
    start = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start

    eval_metrics = parse_eval(log_path)
    row = {
        "name": cell.name,
        "config": asdict(cell),
        "rc": rc,
        "elapsed_sec": elapsed,
        "eval": eval_metrics,
        "log_path": str(log_path),
        "ckpt_path": str(out_dir / "rl_checkpoint.pt"),
    }
    if eval_metrics:
        logger.info(
            "[%s] %s expert_mse=%.4f ref_mse=%.4f q_pol=%.4f q_gap=%.4f td=%.4f time=%.1fmin",
            cell.name, "OK" if rc == 0 else f"FAIL rc={rc}",
            eval_metrics["expert_mse"], eval_metrics["ref_mse"],
            eval_metrics["q_policy"], eval_metrics["q_gap"], eval_metrics["td_err"],
            elapsed / 60,
        )
    else:
        logger.warning("[%s] no eval line parsed; rc=%d log=%s", cell.name, rc, log_path)
    return row


def parse_eval(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text()
    matches = EVAL_LINE_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    return {
        "expert_mse": float(last[0]),
        "ref_mse": float(last[1]),
        "q_policy": float(last[2]),
        "q_expert": float(last[3]),
        "q_gap": float(last[4]),
        "td_err": float(last[5]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gradient-steps", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.json"

    grid = build_grid()
    logger.info("AC sweep: %d cells x %d gradient steps", len(grid), args.gradient_steps)

    sweep_start = time.time()
    for idx, cell in enumerate(grid):
        logger.info("=== Cell %d/%d: %s ===", idx + 1, len(grid), cell.name)
        row = run_cell(args, cell)
        results = json.loads(results_path.read_text()) if results_path.exists() else []
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2))

    logger.info("Sweep done in %.1fmin", (time.time() - sweep_start) / 60)
    results = json.loads(results_path.read_text())
    healthy = [r for r in results if r["eval"] is not None and abs(r["eval"]["q_gap"]) < 0.05]
    healthy.sort(key=lambda r: r["eval"]["ref_mse"])
    logger.info("\n=== AC SWEEP SUMMARY ===")
    for r in results:
        e = r["eval"]
        if e is None:
            logger.info("%-20s NO EVAL", r["name"])
            continue
        logger.info(
            "%-20s expert_mse=%.4f ref_mse=%.4f q_pol=%.4f q_gap=%+.4f td=%.4f",
            r["name"], e["expert_mse"], e["ref_mse"], e["q_policy"], e["q_gap"], e["td_err"],
        )


if __name__ == "__main__":
    main()
