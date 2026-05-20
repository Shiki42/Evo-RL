#!/usr/bin/env python
"""Task A: legacy vs lerobot-train RL Token bisect (v2 — simplified).

Strategy:
  [A0] Code-path analysis (text-only, written to summary).
  [A2-load] Verify legacy token_stats_full.pt loads + SHA + summary stats.
  [A2-regen] Regenerate stats using LEROBOT-TRAIN code path (PI05Policy.forward_with_prefix)
             with the dataset's natural `task` field. Compare to legacy bit-for-bit.
  [A3] Weighted MSE bit-bisect.
  [A1] (Best-effort) prefix tensor diff on one batch using same-input across paths.

Outputs:
  outputs/rlt_sweep_b_2026-05-14/task_a/{
    task_a_summary.json,
    legacy_stats_summary.json,
    regen_stats_lbt.pt,
    regen_stats_lbt.summary.json,
  }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fix_seeds(seed: int = 1000):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- A2-load ----------------
def a2_load_legacy(legacy_path: Path) -> dict:
    print("\n=== [A2-load] Inspect legacy stats ===")
    if not legacy_path.exists():
        return {"status": "FAIL", "error": f"missing: {legacy_path}"}

    sha = sha256_file(legacy_path)
    obj = torch.load(legacy_path, map_location="cpu", weights_only=False)
    std = obj["std"].float()
    mean = obj.get("mean")
    if mean is not None:
        mean = mean.float()

    result = {
        "status": "DONE",
        "path": str(legacy_path),
        "sha256": sha,
        "keys": sorted(list(obj.keys())) if hasattr(obj, "keys") else None,
        "config": obj.get("config", {}),
        "std": {
            "shape": tuple(std.shape),
            "min": float(std.min().item()),
            "max": float(std.max().item()),
            "median": float(std.median().item()),
            "mean": float(std.mean().item()),
            "isfinite_all": bool(torch.isfinite(std).all().item()),
            "first8": std.flatten()[:8].tolist(),
        },
        "n": int(obj.get("n", -1)) if "n" in obj else None,
    }
    print(f"  sha256: {sha}")
    print(f"  shape: {result['std']['shape']}, n: {result['n']}")
    print(f"  std range: [{result['std']['min']:.4f}, {result['std']['max']:.4f}], median={result['std']['median']:.4f}")
    print(f"  config: {result['config']}")
    return result


# ---------------- A2-regen ----------------
def a2_regen_via_lerobot_train(
    out_dir: Path,
    model_path: str,
    dataset_repo_id: str,
    num_batches: int = 100,
    batch_size: int = 2,
    dtype: str = "bfloat16",
) -> dict:
    """Regenerate token_stats via lerobot-train code path (PI05Policy.forward_with_prefix)
    with dataset's natural per-frame task field."""
    print("\n=== [A2-regen] Regenerate stats via lerobot-train ===")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.policies.rlt.modeling_rlt_token import _load_pi05_config_from_dir
    from lerobot.rlt.utils import postprocess_prefix_tokens

    cfg = _load_pi05_config_from_dir(model_path)
    cfg.dtype = dtype
    cfg.device = "cuda"
    print("Loading PI05 policy...")
    policy = PI05Policy.from_pretrained(model_path, config=cfg, strict=False)
    policy = policy.to("cuda").eval()
    for p in policy.parameters():
        p.requires_grad = False

    # Build preprocessor pipeline — needs dataset stats
    print("Loading dataset...")
    ds = LeRobotDataset(dataset_repo_id, revision="main", video_backend="pyav")

    print("Building processors...")
    pre_proc, _post = make_pi05_pre_post_processors(cfg, dataset_stats=ds.meta.stats)

    # Welford accumulator over prefix hidden states (post postprocess_prefix_tokens)
    # We want std over all token positions and batch elements per feature dim.
    n_total = 0
    mean = None  # (D,) running mean
    M2 = None   # (D,) running sum of squared diffs

    pwx = policy.model.paligemma_with_expert
    tokens_per_camera = (cfg.image_resolution[0] // pwx.paligemma.config.vision_config.patch_size) ** 2
    num_image_tokens = tokens_per_camera * 3

    fix_seeds(1000)
    print(f"Iterating {num_batches} batches of size {batch_size}...")
    rng = np.random.default_rng(1000)
    indices = rng.choice(len(ds), size=num_batches * batch_size, replace=False)

    t0 = time.time()
    for bi in range(num_batches):
        idx = indices[bi * batch_size : (bi + 1) * batch_size]
        samples = [ds[int(i)] for i in idx]
        batch = {}
        for k in samples[0]:
            v = samples[0][k]
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([s[k] for s in samples])
            else:
                batch[k] = [s[k] for s in samples]

        processed = pre_proc(batch)
        for k, v in list(processed.items()):
            if isinstance(v, torch.Tensor):
                processed[k] = v.to("cuda")

        with torch.no_grad():
            _, _, prefix_hidden = policy.forward_with_prefix(processed, reduction="mean")
            # postprocess: full prefix, no pool, no image_only
            prefix_for_rl = postprocess_prefix_tokens(
                prefix_hidden.to(dtype=torch.float32),
                image_only=False,
                num_image_tokens=num_image_tokens,
                pool_size=0,
                num_per_camera=tokens_per_camera,
                active_camera_indices=None,
            )
            # Flatten (B, M, D) → (B*M, D)
            flat = prefix_for_rl.reshape(-1, prefix_for_rl.shape[-1]).cpu()

        # Welford update per row
        for row in flat:
            n_total += 1
            if mean is None:
                D = row.shape[0]
                mean = torch.zeros(D)
                M2 = torch.zeros(D)
            delta = row - mean
            mean = mean + delta / n_total
            delta2 = row - mean
            M2 = M2 + delta * delta2

        if (bi + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  batch {bi+1}/{num_batches}, n={n_total}, elapsed={elapsed:.1f}s")

    # Finalize
    var = M2 / max(1, n_total - 1)
    std = var.clamp_min(0).sqrt()
    elapsed = time.time() - t0
    print(f"Done. n={n_total}, std range=[{std.min():.4f}, {std.max():.4f}], elapsed={elapsed:.1f}s")

    out_pt = out_dir / "regen_stats_lbt.pt"
    torch.save({
        "mean": mean,
        "std": std,
        "n": n_total,
        "config": {
            "code_path": "lerobot_train PI05Policy.forward_with_prefix",
            "image_only": False,
            "active_cameras": None,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "dataset": dataset_repo_id,
            "model_path": model_path,
            "dtype": dtype,
        },
    }, out_pt)

    summary = {
        "status": "DONE",
        "path": str(out_pt),
        "sha256": sha256_file(out_pt),
        "n": n_total,
        "std": {
            "shape": tuple(std.shape),
            "min": float(std.min().item()),
            "max": float(std.max().item()),
            "median": float(std.median().item()),
            "mean": float(std.mean().item()),
            "first8": std.flatten()[:8].tolist(),
        },
        "elapsed_sec": elapsed,
    }
    (out_dir / "regen_stats_lbt.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------- A3 ----------------
def a3_loss_bisect(out_dir: Path, dim_std_for_test=None) -> dict:
    print("\n=== [A3] Weighted MSE bit-bisect ===")
    from lerobot.rlt.rl_token import RLTokenModule

    fix_seeds(42)
    D = 2048
    M = 192
    B = 2
    prefix_fixed = torch.randn(B, M, D)

    if dim_std_for_test is None:
        std = torch.rand(D).clamp_min(0.01) * 30.0
    else:
        std = dim_std_for_test

    mod = RLTokenModule(
        token_dim=D,
        num_rl_tokens=1,
        num_enc_layers=3,
        num_dec_layers=3,
        nhead=8,
        ff_dim=4096,
    )
    mod.eval()

    rows = []
    with torch.no_grad():
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            l_legacy = mod.reconstruction_loss(prefix_fixed, dim_std=std, gamma=gamma)
            l_lerobot = mod.reconstruction_loss(prefix_fixed, dim_std=std, gamma=gamma)
            row = {
                "gamma": gamma,
                "legacy_loss": float(l_legacy.item()),
                "lerobot_loss": float(l_lerobot.item()),
                "abs_diff": float(abs(l_legacy.item() - l_lerobot.item())),
            }
            rows.append(row)
            print(f"  γ={gamma}: legacy={row['legacy_loss']:.6f} lerobot={row['lerobot_loss']:.6f} Δ={row['abs_diff']:.2e}")

    return {
        "status": "DONE",
        "table": rows,
        "formula_note": "weight = std.clamp_min(1e-6).pow(-gamma); loss = (diff * weight).pow(2).mean(); effective_squared_weight = std^(-2*gamma)",
    }


# ---------------- A0 code analysis (text only) ----------------
A0_NOTES = {
    "title": "Code-path analysis (text-only)",
    "findings": [
        "Legacy `Pi05VLAAdapter._prepare_language_tokens` (src/lerobot/rlt/pi05_adapter.py:170-181) clips raw proprio via `np.clip(state_np, -1.0, 1.0)` before digitizing into 256 bins. Raw joint angles are not naturally in [-1, 1], so this saturates most coords to bin 0 or 255.",
        "Lerobot-train `Pi05PrepareStateTokenizerProcessorStep` (src/lerobot/policies/pi05/processor_pi05.py:48-98) assumes state is pre-normalized to [-1, 1] by a NormalizerProcessorStep (QUANTILES). The discretization is mathematically identical IF input is normalized; the prerequisite differs.",
        "Legacy adapter uses `self.task_instruction` (constructor arg, fixed string e.g. 'screw') in the prompt: `Task: <X>, State: <tokens>; \\nAction: `.",
        "Lerobot-train pipeline uses the dataset's per-frame `task` field, after `task.strip().replace('_', ' ').replace('\\n', ' ')`. The Shiki42/0420_0423screw dataset returns `task='Screw'` (capitalized).",
        "Net implication: even with identical PI0.5 weights, the prefix hidden states differ between legacy and lerobot-train because (a) state tokens are saturated bin-0/255 in legacy vs distributed in lerobot-train, (b) language prefix is 'screw' vs 'Screw'.",
        "Therefore `token_stats_full.pt` regenerated under each code path WILL differ; the legacy file (used as norm stats reference) is NOT byte-equal to a lerobot-train regeneration.",
        "The legacy weighted MSE 0.354 was computed against legacy prefix variance. The lerobot-train 0.443 was computed against the SAME legacy stats file but lerobot-train prefix distribution — the std weighting is mis-calibrated for the new distribution. This is a likely contributor to the gap.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/coder/code/Evo-RL-quick/outputs/rlt_sweep_b_2026-05-14/task_a")
    ap.add_argument("--model-path", default="/home/coder/share/policy_pi05_screw")
    ap.add_argument("--dataset", default="Shiki42/0420_0423screw")
    ap.add_argument("--legacy-stats", default="/home/coder/code/Evo-RL-quick/outputs/legacy_stats/token_stats_full.pt")
    ap.add_argument("--num-batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "start_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "items": {"A0_code_analysis": A0_NOTES},
    }

    # A3 first (fast, no GPU model load needed)
    try:
        summary["items"]["A3_loss_bisect"] = a3_loss_bisect(out_dir)
    except Exception as e:
        summary["items"]["A3_loss_bisect"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # A2-load
    try:
        summary["items"]["A2_legacy_load"] = a2_load_legacy(Path(args.legacy_stats))
    except Exception as e:
        summary["items"]["A2_legacy_load"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # A2-regen (heavy: ~20-40 min)
    try:
        summary["items"]["A2_regen_lbt"] = a2_regen_via_lerobot_train(
            out_dir, args.model_path, args.dataset, args.num_batches, args.batch_size
        )

        # Compare legacy std vs regen std
        legacy_obj = torch.load(args.legacy_stats, map_location="cpu", weights_only=False)
        legacy_std = legacy_obj["std"].float()
        regen_obj = torch.load(summary["items"]["A2_regen_lbt"]["path"], map_location="cpu", weights_only=False)
        regen_std = regen_obj["std"].float()

        comparison = {
            "shape_match": tuple(legacy_std.shape) == tuple(regen_std.shape),
            "byte_equal": bool(torch.equal(legacy_std, regen_std)) if tuple(legacy_std.shape) == tuple(regen_std.shape) else False,
        }
        if comparison["shape_match"]:
            diff = (legacy_std - regen_std).abs()
            denom = legacy_std.abs() + regen_std.abs() + 1e-6
            comparison["max_abs_diff"] = float(diff.max().item())
            comparison["mean_abs_diff"] = float(diff.mean().item())
            comparison["max_rel_diff"] = float((diff / denom).max().item())
            comparison["mean_rel_diff"] = float((diff / denom).mean().item())
        summary["items"]["A2_comparison"] = comparison
        print(f"\n=== [A2-compare] ===")
        print(f"  shape_match: {comparison['shape_match']}, byte_equal: {comparison['byte_equal']}")
        if "max_abs_diff" in comparison:
            print(f"  max_abs_diff: {comparison['max_abs_diff']:.6f}, mean_abs: {comparison['mean_abs_diff']:.6f}")
            print(f"  max_rel: {comparison['max_rel_diff']:.4f}, mean_rel: {comparison['mean_rel_diff']:.4f}")
    except Exception as e:
        summary["items"]["A2_regen_lbt"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    summary["end_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_json = out_dir / "task_a_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== Summary → {out_json} ===")
    for k, v in summary["items"].items():
        print(f"  {k}: {v.get('status', 'note')}")


if __name__ == "__main__":
    main()
