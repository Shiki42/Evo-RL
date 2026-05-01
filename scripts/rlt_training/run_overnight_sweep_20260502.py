#!/usr/bin/env python
"""Overnight RL token sweep 2026-05-02.

All experiments default to no-pooling (token_pool_size=0) per user directive.
Tests:
  - A: image_only=True {1 tok, 4 tok}
  - B: camera subset {2 wrist, right_wrist only} x {1, 4 tok}
  - C: weighted MSE gamma=0.5 (full prefix) x {1, 4 tok}
  - D: image_only + gamma=0.5 x {1, 4 tok}
  - E (conditional): if 4 tok wins big, run 8 tok on best config

Each experiment runs train_rl_token.py as a subprocess, captures returncode,
parses losses.json, appends a row to results.json. Run sequentially on one GPU.

Usage:
  python scripts/rlt_training/run_overnight_sweep_20260502.py \
      --model-path /home/coder/share/policy_pi05_screw \
      --dataset-path /home/coder/share/dataset/0420_0423screw \
      --output-root outputs/rlt_overnight_20260502 \
      [--steps 5000] [--phase A,C,D,B] [--skip-variance]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging

logger = configure_logging("overnight_sweep")


@dataclass
class Experiment:
    name: str
    num_rl_tokens: int
    image_only: bool = False
    active_cameras: str | None = None  # comma-separated obs-key names
    norm_gamma: float = 0.0
    norm_stats_kind: str | None = None  # which variance file to use ("full" / "image_only")
    description: str = ""


def build_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # A: image_only
    exps += [
        Experiment("A1_imgonly_1tok",   1, image_only=True,  description="image-only 768 tok, 1 RL tok"),
        Experiment("A2_imgonly_4tok",   4, image_only=True,  description="image-only 768 tok, 4 RL tok"),
    ]

    # C: weighted MSE on full prefix
    exps += [
        Experiment("C1_wmse_full_1tok", 1, image_only=False, norm_gamma=0.5, norm_stats_kind="full",
                   description="full 968 tok, weighted MSE gamma=0.5, 1 RL tok"),
        Experiment("C2_wmse_full_4tok", 4, image_only=False, norm_gamma=0.5, norm_stats_kind="full",
                   description="full 968 tok, weighted MSE gamma=0.5, 4 RL tok"),
    ]

    # D: combined image_only + weighted MSE
    exps += [
        Experiment("D1_imgonly_wmse_1tok", 1, image_only=True, norm_gamma=0.5, norm_stats_kind="image_only",
                   description="image-only + weighted MSE gamma=0.5, 1 RL tok"),
        Experiment("D2_imgonly_wmse_4tok", 4, image_only=True, norm_gamma=0.5, norm_stats_kind="image_only",
                   description="image-only + weighted MSE gamma=0.5, 4 RL tok"),
    ]

    # B: camera subset (more aggressive)
    exps += [
        Experiment("B1_2wrist_1tok",       1, active_cameras="left_wrist,right_wrist",
                   description="2 wrist cameras only, 512 tok, 1 RL tok"),
        Experiment("B2_2wrist_4tok",       4, active_cameras="left_wrist,right_wrist",
                   description="2 wrist cameras only, 512 tok, 4 RL tok"),
        Experiment("B3_rwrist_1tok",       1, active_cameras="right_wrist",
                   description="right wrist only, 256 tok, 1 RL tok"),
        Experiment("B4_rwrist_4tok",       4, active_cameras="right_wrist",
                   description="right wrist only, 256 tok, 4 RL tok"),
    ]

    return exps


def run_variance(args, label: str, image_only: bool, active_cameras: str | None) -> Path:
    """Compute (and cache) per-dim std for the given token-selection config."""
    out_root = Path(args.output_root)
    stats_dir = out_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"token_stats_{label}.pt"
    if stats_path.exists():
        logger.info("[variance:%s] already exists at %s", label, stats_path)
        return stats_path

    cmd = [
        sys.executable, "-u",
        str(SCRIPT_ROOT / "compute_token_variance.py"),
        "--model-path", args.model_path,
        "--demo-dataset-path", args.dataset_path,
        "--output", str(stats_path),
        "--num-batches", str(args.variance_batches),
        "--batch-size", "2",
        "--device", "cuda",
        "--dtype", "float32",
        "--task-instruction", args.task_instruction,
    ]
    if image_only:
        cmd.append("--image-only")
    if active_cameras:
        cmd += ["--active-cameras", active_cameras]

    log_path = stats_dir / f"variance_{label}.log"
    logger.info("[variance:%s] running: %s", label, " ".join(cmd))
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        logger.error("[variance:%s] FAILED rc=%d, see %s", label, rc, log_path)
        raise RuntimeError(f"variance {label} failed")
    logger.info("[variance:%s] done -> %s", label, stats_path)
    return stats_path


def run_experiment(args, exp: Experiment, stats_paths: dict[str, Path]) -> dict:
    out_root = Path(args.output_root)
    exp_dir = out_root / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"

    cmd = [
        sys.executable, "-u",
        str(SCRIPT_ROOT / "train_rl_token.py"),
        "--model-path", args.model_path,
        "--demo-dataset-path", args.dataset_path,
        "--task-instruction", args.task_instruction,
        "--output-dir", str(exp_dir),
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--steps", str(args.steps),
        "--batch-size", "2",
        "--lr", "2e-4",
        "--save-every", str(max(1000, args.steps // 5)),
        "--num-rl-tokens", str(exp.num_rl_tokens),
        "--token-pool-size", "0",
    ]
    if exp.image_only:
        cmd.append("--image-only")
    if exp.active_cameras:
        cmd += ["--active-cameras", exp.active_cameras]
    if exp.norm_gamma > 0:
        if exp.norm_stats_kind not in stats_paths:
            raise RuntimeError(f"missing stats for {exp.name} (kind={exp.norm_stats_kind})")
        cmd += [
            "--norm-stats", str(stats_paths[exp.norm_stats_kind]),
            "--norm-gamma", str(exp.norm_gamma),
        ]

    logger.info("[%s] starting: %s", exp.name, " ".join(cmd))
    start = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start

    losses_path = exp_dir / "losses.json"
    final_loss = avg_last_100 = avg_last_500 = min_loss = float("nan")
    n_steps = 0
    if losses_path.exists():
        losses = json.loads(losses_path.read_text())
        n_steps = len(losses)
        if losses:
            final_loss = losses[-1]
            avg_last_100 = sum(losses[-100:]) / min(len(losses), 100)
            avg_last_500 = sum(losses[-500:]) / min(len(losses), 500)
            min_loss = min(losses)
    row = {
        "name": exp.name,
        "config": asdict(exp),
        "rc": rc,
        "elapsed_sec": elapsed,
        "n_steps": n_steps,
        "final_loss": final_loss,
        "avg_last_100": avg_last_100,
        "avg_last_500": avg_last_500,
        "min_loss": min_loss,
        "log_path": str(log_path),
        "ckpt_path": str(exp_dir / "demo_adapt_checkpoint.pt"),
    }
    status = "OK" if rc == 0 else f"FAIL rc={rc}"
    logger.info("[%s] %s avg100=%.4f avg500=%.4f min=%.4f time=%.1fmin",
                exp.name, status, avg_last_100, avg_last_500, min_loss, elapsed / 60)
    return row


def append_result(results_path: Path, row: dict) -> None:
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    results.append(row)
    results_path.write_text(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-instruction", default="screw")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--variance-batches", type=int, default=100)
    parser.add_argument("--phase", default="A,C,D,B",
                        help="Comma-separated phase letters to run (default A,C,D,B).")
    parser.add_argument("--skip-variance", action="store_true",
                        help="Skip variance computation (assumes stats already exist).")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip the first N experiments in the filtered list.")
    parser.add_argument("--max-runtime-hours", type=float, default=None,
                        help="Stop launching new experiments after this wall-clock budget.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.json"
    logger.info("Output root: %s", out_root)
    logger.info("Steps per experiment: %d", args.steps)

    # Compute variance stats if needed.
    stats_paths: dict[str, Path] = {}
    if not args.skip_variance:
        # We always need full stats for C1/C2 if Phase C is enabled, and image_only stats for D1/D2.
        phases = set(args.phase.split(","))
        if "C" in phases:
            stats_paths["full"] = run_variance(args, "full", image_only=False, active_cameras=None)
        if "D" in phases:
            stats_paths["image_only"] = run_variance(args, "image_only", image_only=True, active_cameras=None)
    else:
        # Recover existing paths.
        stats_dir = out_root / "stats"
        for kind in ("full", "image_only"):
            p = stats_dir / f"token_stats_{kind}.pt"
            if p.exists():
                stats_paths[kind] = p

    # Filter experiments by phase.
    phases = set(args.phase.split(","))
    all_exps = build_experiments()
    selected = [e for e in all_exps if e.name[0] in phases]
    logger.info("Selected %d experiments (phases=%s):", len(selected), sorted(phases))
    for e in selected:
        logger.info("  - %s: %s", e.name, e.description)

    sweep_start = time.time()
    for idx, exp in enumerate(selected):
        if idx < args.start_from:
            logger.info("[skip %d] %s", idx, exp.name)
            continue
        if args.max_runtime_hours is not None:
            elapsed_hr = (time.time() - sweep_start) / 3600
            if elapsed_hr >= args.max_runtime_hours:
                logger.warning("Hit runtime budget %.2fh, stopping at exp idx=%d", args.max_runtime_hours, idx)
                break
        logger.info("=== Experiment %d/%d: %s ===", idx + 1, len(selected), exp.name)
        row = run_experiment(args, exp, stats_paths)
        append_result(results_path, row)

    # Summary.
    if results_path.exists():
        results = json.loads(results_path.read_text())
        results_sorted = sorted([r for r in results if r["rc"] == 0], key=lambda r: r["avg_last_100"])
        logger.info("\n=== FINAL SUMMARY (sorted by avg_last_100) ===")
        for r in results_sorted:
            logger.info("%-30s avg100=%.4f avg500=%.4f min=%.4f time=%.1fmin",
                        r["name"], r["avg_last_100"], r["avg_last_500"], r["min_loss"], r["elapsed_sec"] / 60)


if __name__ == "__main__":
    main()
