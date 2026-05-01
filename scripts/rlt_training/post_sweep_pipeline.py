#!/usr/bin/env python
"""Post-sweep pipeline for the 2026-05-02 overnight run.

After run_overnight_sweep_20260502.py finishes, this:
  1. Picks the best RL token ckpt from results.json (rc==0, lowest avg_last_100)
  2. Builds a fresh chunk-transition cache from that ckpt with the matching
     image_only / active_cameras / num_rl_tokens config
  3. Runs run_ac_sweep_overnight.py on the new cache
  4. Writes a summary markdown to docs/rlt/

If --conditional-eight is set, before step 2 it also checks whether 4-token
configs significantly beat 1-token (>1.5x lower avg_last_100); if so, kicks off
an 8-token run on the best 4-token config first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging

logger = configure_logging("post_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-output", required=True, help="The --output-root used in run_overnight_sweep_20260502.py")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task-instruction", default="screw")
    parser.add_argument("--ac-gradient-steps", type=int, default=30000)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--conditional-eight", action="store_true",
                        help="If best 4-tok beats best 1-tok by >1.5x, run 8-tok on the same config first.")
    parser.add_argument("--summary-md", default="docs/rlt/rl_token_overnight_2026-05-02.md")
    parser.add_argument("--skip-ac", action="store_true")
    return parser.parse_args()


def pick_best(results: list[dict]) -> dict:
    ok = [r for r in results if r["rc"] == 0 and r["n_steps"] >= 100]
    if not ok:
        raise RuntimeError("No successful experiments in results.json")
    ok.sort(key=lambda r: r["avg_last_100"])
    return ok[0]


def maybe_run_eight(args: argparse.Namespace, results: list[dict]) -> dict | None:
    """If best 4-tok beats best 1-tok by >1.5x, train an 8-tok variant of best 4-tok config."""
    by_n: dict[int, list[dict]] = {}
    for r in results:
        if r["rc"] != 0:
            continue
        n = r["config"]["num_rl_tokens"]
        by_n.setdefault(n, []).append(r)
    if 1 not in by_n or 4 not in by_n:
        logger.info("[8tok-trigger] missing 1-tok or 4-tok results, skipping conditional 8-tok")
        return None

    best_1 = sorted(by_n[1], key=lambda r: r["avg_last_100"])[0]
    best_4 = sorted(by_n[4], key=lambda r: r["avg_last_100"])[0]
    ratio = best_1["avg_last_100"] / max(best_4["avg_last_100"], 1e-6)
    logger.info("[8tok-trigger] best 1tok=%.4f best 4tok=%.4f  ratio=%.2fx",
                best_1["avg_last_100"], best_4["avg_last_100"], ratio)
    if ratio < 1.5:
        logger.info("[8tok-trigger] ratio < 1.5x, skipping 8-tok")
        return None

    cfg = best_4["config"]
    name = f"E1_8tok_from_{cfg['name']}"
    out_dir = Path(args.sweep_output) / name
    if out_dir.exists():
        logger.info("[8tok-trigger] %s already exists, skipping launch", name)
        return None
    cmd = [
        sys.executable, "-u", str(SCRIPT_ROOT / "train_rl_token.py"),
        "--model-path", args.model_path,
        "--demo-dataset-path", args.dataset_path,
        "--task-instruction", args.task_instruction,
        "--output-dir", str(out_dir),
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--steps", "5000",
        "--batch-size", "2",
        "--lr", "2e-4",
        "--save-every", "1000",
        "--num-rl-tokens", "8",
        "--token-pool-size", "0",
    ]
    if cfg.get("image_only"):
        cmd.append("--image-only")
    if cfg.get("active_cameras"):
        cmd += ["--active-cameras", cfg["active_cameras"]]
    if cfg.get("norm_gamma", 0) > 0:
        kind = cfg.get("norm_stats_kind", "full")
        cmd += [
            "--norm-stats", str(Path(args.sweep_output) / "stats" / f"token_stats_{kind}.pt"),
            "--norm-gamma", str(cfg["norm_gamma"]),
        ]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logger.info("[8tok-trigger] launching: %s", " ".join(cmd))
    start = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start
    losses_path = out_dir / "losses.json"
    avg_last_100 = float("nan")
    if losses_path.exists():
        losses = json.loads(losses_path.read_text())
        if losses:
            avg_last_100 = sum(losses[-100:]) / min(len(losses), 100)
    row = {
        "name": name,
        "config": {**cfg, "name": name, "num_rl_tokens": 8},
        "rc": rc,
        "elapsed_sec": elapsed,
        "avg_last_100": avg_last_100,
        "log_path": str(log_path),
        "ckpt_path": str(out_dir / "demo_adapt_checkpoint.pt"),
    }
    results.append(row)
    Path(args.sweep_output, "results.json").write_text(json.dumps(results, indent=2))
    logger.info("[8tok-trigger] done rc=%d avg100=%.4f time=%.1fmin", rc, avg_last_100, elapsed / 60)
    return row


def build_cache(args: argparse.Namespace, best: dict) -> Path:
    cfg = best["config"]
    cache_dir = Path(args.sweep_output) / "cache_best"
    if (cache_dir / "chunk_transitions_train.pt").exists():
        logger.info("[cache] already exists at %s", cache_dir)
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u",
        str(SCRIPT_ROOT / "build_transition_cache.py"),
        "--demo-dataset-path", args.dataset_path,
        "--transition-cache-dir", str(cache_dir),
        "--model-path", args.model_path,
        "--rl-token-checkpoint", best["ckpt_path"],
        "--task-instruction", args.task_instruction,
        "--token-pool-size", "0",
        "--num-rl-tokens", str(cfg["num_rl_tokens"]),
        "--frame-stride", str(args.frame_stride),
        "--batch-size", str(args.cache_batch_size),
        "--dtype", "bfloat16",
        "--device", "cuda",
    ]
    if cfg.get("image_only"):
        cmd.append("--image-only")
    if cfg.get("active_cameras"):
        cmd += ["--active-cameras", cfg["active_cameras"]]

    log_path = cache_dir / "build.log"
    logger.info("[cache] launching: %s", " ".join(cmd))
    start = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start
    if rc != 0:
        raise RuntimeError(f"build_transition_cache failed rc={rc}, see {log_path}")
    logger.info("[cache] done rc=%d time=%.1fmin", rc, elapsed / 60)
    return cache_dir


def run_ac_sweep(args: argparse.Namespace, cache_dir: Path) -> Path:
    ac_out = Path(args.sweep_output) / "ac_sweep"
    ac_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_ROOT / "run_ac_sweep_overnight.py"),
        "--transition-cache-dir", str(cache_dir),
        "--output-root", str(ac_out),
        "--gradient-steps", str(args.ac_gradient_steps),
    ]
    log_path = ac_out / "orchestrator.log"
    logger.info("[ac] launching: %s", " ".join(cmd))
    start = time.time()
    with log_path.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start
    logger.info("[ac] done rc=%d time=%.1fmin (log=%s)", rc, elapsed / 60, log_path)
    return ac_out


def write_summary(args: argparse.Namespace, results: list[dict], best: dict, ac_results_path: Path | None) -> None:
    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# RL Token Overnight Sweep — 2026-05-02")
    lines.append("")
    lines.append("> 自动生成。来源：`outputs/rlt_overnight_20260502/`")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(f"- Best RL token config: **{best['name']}** avg_last_100={best['avg_last_100']:.4f}")
    cfg = best["config"]
    lines.append(f"  - image_only={cfg.get('image_only')}, active_cameras={cfg.get('active_cameras')}, "
                 f"num_rl_tokens={cfg['num_rl_tokens']}, norm_gamma={cfg.get('norm_gamma', 0)}")
    lines.append(f"  - ckpt: `{best['ckpt_path']}`")
    lines.append("")
    lines.append("## RL Token Sweep Results (sorted by avg_last_100)")
    lines.append("")
    lines.append("| name | num_rl_tokens | image_only | active_cameras | gamma | avg_last_100 | avg_last_500 | min_loss | rc | time(min) |")
    lines.append("|---|---:|:-:|---|---:|---:|---:|---:|---:|---:|")
    sorted_rows = sorted([r for r in results if r["rc"] == 0], key=lambda r: r["avg_last_100"])
    for r in sorted_rows:
        c = r["config"]
        lines.append(
            f"| {r['name']} | {c['num_rl_tokens']} | {c.get('image_only')} | "
            f"{c.get('active_cameras') or '-'} | {c.get('norm_gamma', 0)} | "
            f"{r['avg_last_100']:.4f} | {r['avg_last_500']:.4f} | {r['min_loss']:.4f} | "
            f"{r['rc']} | {r['elapsed_sec'] / 60:.1f} |"
        )
    failed = [r for r in results if r["rc"] != 0]
    if failed:
        lines.append("")
        lines.append("### Failed experiments")
        for r in failed:
            lines.append(f"- {r['name']}: rc={r['rc']} log={r['log_path']}")

    # Variance summary
    full_summary = Path(args.sweep_output) / "stats" / "token_stats_full.summary.json"
    if full_summary.exists():
        s = json.loads(full_summary.read_text())
        lines.append("")
        lines.append("## Per-dim variance (full prefix, n=%d tokens, D=%d)" % (s["n_tokens"], s["feature_dim"]))
        lines.append(f"- std: min={s['std_min']:.4f} median={s['std_median']:.4f} mean={s['std_mean']:.4f} max={s['std_max']:.4f}")
        lines.append(f"- max/median = {s['max_over_median_ratio']:.1f}x")
        lines.append(f"- top-10 dims explain {100*s['top10_var_share']:.1f}% of variance")
        lines.append(f"- top-50 dims explain {100*s['top50_var_share']:.1f}% of variance")
        lines.append(f"- top-200 dims explain {100*s['top200_var_share']:.1f}% of variance")
        lines.append("- top-5 dims:")
        for d in s["top20_dims"][:5]:
            lines.append(f"  - dim {d['dim']:4d}  std={d['std']:.4f}  mean={d['mean']:+.4f}")

    # AC sweep
    if ac_results_path is not None and ac_results_path.exists():
        ac_results = json.loads(ac_results_path.read_text())
        lines.append("")
        lines.append("## AC Sweep Results (on best-RL-token cache)")
        lines.append("")
        lines.append("| beta | expert_mse | ref_mse | q_policy | q_expert | q_gap | td_err | rc | time(min) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in ac_results:
            e = r.get("eval")
            if e is None:
                lines.append(f"| {r['config']['beta']} | NO EVAL | | | | | | {r['rc']} | {r['elapsed_sec']/60:.1f} |")
                continue
            lines.append(
                f"| {r['config']['beta']} | {e['expert_mse']:.4f} | {e['ref_mse']:.4f} | "
                f"{e['q_policy']:+.4f} | {e['q_expert']:+.4f} | {e['q_gap']:+.4f} | {e['td_err']:.4f} | "
                f"{r['rc']} | {r['elapsed_sec']/60:.1f} |"
            )

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/coder/share/Evo-RL-quick")
    lines.append("git fetch fork && git reset --hard fork/shuyuan/rlt-overnight-20260502")
    lines.append("python scripts/rlt_training/run_overnight_sweep_20260502.py \\")
    lines.append(f"  --model-path {args.model_path} \\")
    lines.append(f"  --dataset-path {args.dataset_path} \\")
    lines.append(f"  --output-root {args.sweep_output} --steps 5000")
    lines.append("```")

    summary_path.write_text("\n".join(lines))
    logger.info("Wrote summary to %s (%d bytes)", summary_path, len(summary_path.read_text()))


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_output)
    results_path = sweep_root / "results.json"
    if not results_path.exists():
        raise RuntimeError(f"missing {results_path}")

    results = json.loads(results_path.read_text())
    logger.info("Loaded %d results", len(results))

    if args.conditional_eight:
        new_row = maybe_run_eight(args, results)
        if new_row is not None:
            results = json.loads(results_path.read_text())  # refresh

    best = pick_best(results)
    logger.info("Best: %s avg_last_100=%.4f", best["name"], best["avg_last_100"])

    ac_results_path = None
    if not args.skip_ac:
        cache_dir = build_cache(args, best)
        ac_out = run_ac_sweep(args, cache_dir)
        ac_results_path = ac_out / "results.json"

    results = json.loads(results_path.read_text())
    write_summary(args, results, best, ac_results_path)


if __name__ == "__main__":
    main()
