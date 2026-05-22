"""Compute matched per-dim mean/std of pi0.5 prefix tokens for NEW SFT + NEW dataset.

Mirrors lerobot_train.py setup so stats EXACTLY match what training sees.
Output is a .pt with keys mean (D,), std (D,), n (int).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="/home/coder/share/policy_pi05_screw_c_mix_20k")
    p.add_argument("--repo-id", default="Shiki42/0420_0423screw_c_mix")
    p.add_argument("--output", required=True)
    p.add_argument("--num-batches", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.rlt.configuration_rlt_token import RLTokenPolicyConfig
    from lerobot.rlt.utils import postprocess_prefix_tokens

    policy_cfg = RLTokenPolicyConfig(
        vla_pretrained_path=args.sft,
        vla_dtype="bfloat16",
        rl_token_num_rl_tokens=1,
        rl_token_enc_layers=4,
        rl_token_dec_layers=4,
        rl_token_ff_dim=8192,
        rl_token_nhead=16,
        token_pool_size=0,
        image_only=False,
        norm_gamma=0.0,
        norm_stats_path=None,
        device="cuda",
    )

    ds_cfg = DatasetConfig(repo_id=args.repo_id, revision="main", video_backend="pyav")
    cfg = TrainPipelineConfig(
        dataset=ds_cfg,
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tolerance_s=1e-4,
        output_dir=Path("/tmp/stats_run_throwaway"),
        save_checkpoint=False,
    )

    print(f"[{time.strftime('%H:%M:%S')}] Building dataset...", flush=True)
    dataset = make_dataset(cfg)
    print(f"[{time.strftime('%H:%M:%S')}] Dataset built: {len(dataset)} samples", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building policy + loading pi0.5...", flush=True)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.to("cuda").eval()
    pi05 = policy._pi05
    print(f"[{time.strftime('%H:%M:%S')}] Policy built. num_image_tokens={policy._num_image_tokens}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building preprocessor...", flush=True)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=None,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    n = 0
    mean: torch.Tensor | None = None
    M2: torch.Tensor | None = None
    start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Welford over {args.num_batches} batches (bs={args.batch_size})...", flush=True)

    it = iter(dl)
    for i in range(args.num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {
            k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.no_grad():
            _, info, prefix_hidden = pi05.forward_with_prefix(batch, reduction="mean")
        prefix_for_rl = postprocess_prefix_tokens(
            prefix_hidden.to(dtype=torch.float32),
            image_only=False,
            num_image_tokens=policy._num_image_tokens,
            pool_size=0,
            num_per_camera=0,
            active_camera_indices=None,
        )
        x = prefix_for_rl.reshape(-1, prefix_for_rl.shape[-1])
        bn = x.shape[0]
        if mean is None:
            mean = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
            M2 = torch.zeros_like(mean)
        delta = x - mean
        n += bn
        mean = mean + delta.sum(dim=0) / n
        delta2 = x - mean
        M2 = M2 + (delta * delta2).sum(dim=0)
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (args.num_batches - (i + 1)) / max(rate, 1e-6)
            std_now = (M2 / max(n - 1, 1)).sqrt()
            print(
                f"[{time.strftime('%H:%M:%S')}] batch {i+1}/{args.num_batches} "
                f"n={n} std_mean={float(std_now.mean()):.4f} "
                f"std_max={float(std_now.max()):.2f} eta={eta:.0f}s",
                flush=True,
            )

    var = M2 / max(n - 1, 1)
    std = var.sqrt()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "n": n,
        "config": {
            "sft": args.sft,
            "repo_id": args.repo_id,
            "image_only": False,
            "token_pool_size": 0,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "via": "lerobot_train_match",
        },
    }
    torch.save(payload, out)

    sorted_std, _ = std.sort(descending=True)
    summary = {
        "n_tokens": n,
        "D": int(std.shape[0]),
        "std_mean": float(std.mean()),
        "std_median": float(std.median()),
        "std_top5pct_mean": float(sorted_std[: max(1, int(0.05 * std.shape[0]))].mean()),
        "std_max": float(std.max()),
        "std_min": float(std.min()),
        "mean_abs_mean": float(mean.abs().mean()),
    }
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump({"summary": summary, "config": payload["config"]}, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] Saved {out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
