#!/usr/bin/env python
"""Chunk-boundary jump metrics — VLA-only baseline.

For each chunk boundary in a demo dataset, records three jump-style metrics:

  raw_action0_to_obs_{l1,linf}   = | raw_chunk[0] - obs.state |   (normalized)
  sent_action0_to_obs_{l1,linf}  = | sent_chunk[0] - obs.state_raw | (robot frame)
  boundary_action_jump_{l1,linf} = | sent_chunk[0] - last_sent_chunk[-1] | (robot frame)

The baseline run uses the cotrained pi0.5 (no AC layer, no RLT logic) — the
"smooth ceiling" any AC-trained policy should approach but not exceed (jumps
should be ~same magnitude or smaller).

Run (after AC sweep finishes):
  cd /home/coder/code/Evo-RL-quick
  source /home/coder/venv-lerobot/bin/activate
  python scripts/rlt_training/eval_chunk_boundary_jump.py \
      --policy-hf-id Shiki42/0520_pi0.5screw_rlt_cotrain_c \
      --dataset-repo-id Shiki42/0420_0423screw_critical_c \
      --num-episodes 20 \
      --out outputs/boundary_jump_vla_only_20260521.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("boundary_jump")


def l1_linf(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    x = x.detach().to(torch.float32).cpu().flatten()
    y = y.detach().to(torch.float32).cpu().flatten()
    d = (x - y).abs()
    return float(d.sum().item()), float(d.max().item())


def run_episode(
    policy,
    preproc,
    postproc,
    dataset,
    ep_idx: int,
    chunk_length: int,
    task_instruction: str,
    device: torch.device,
) -> list[dict]:
    """Replay one episode chunk-by-chunk, recording metrics per boundary."""
    ep_meta = dataset.meta.episodes[ep_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    n_frames = ep_to - ep_from
    n_chunks = max(1, n_frames // chunk_length)

    rows: list[dict] = []
    last_sent_chunk = None
    last_raw_chunk = None

    for ci in range(n_chunks):
        t_global = ep_from + ci * chunk_length
        sample = dataset[t_global]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}
        batch["task"] = [task_instruction]
        pre = preproc(batch)

        with torch.no_grad():
            raw_chunk = policy.predict_action_chunk(pre)
        # (1, H, A). Truncate to first chunk_length entries.
        raw_chunk = raw_chunk[:, :chunk_length, :]
        A = raw_chunk.shape[-1]

        # Postprocess back to robot space: pipeline takes an output batch
        sent_chunk = postproc(raw_chunk[0].clone())  # PolicyAction = Tensor  # (C, A)

        obs_state_raw = sample["observation.state"][:A].detach().cpu()
        obs_state_norm = pre["observation.state"][0, :A].detach().cpu()
        raw_a0 = raw_chunk[0, 0].detach().cpu()
        sent_a0 = sent_chunk[0].detach().cpu() if hasattr(sent_chunk, "shape") else sent_chunk
        raw_l1, raw_linf = l1_linf(raw_a0, obs_state_norm)
        sent_l1, sent_linf = l1_linf(sent_a0, obs_state_raw)

        rec = {
            "episode": ep_idx,
            "chunk_idx": ci,
            "t_global": t_global,
            "raw_action0_to_obs_l1": raw_l1,
            "raw_action0_to_obs_linf": raw_linf,
            "sent_action0_to_obs_l1": sent_l1,
            "sent_action0_to_obs_linf": sent_linf,
        }
        if last_sent_chunk is not None:
            j_l1, j_linf = l1_linf(sent_chunk[0], last_sent_chunk[-1])
            rec["boundary_action_jump_l1"] = j_l1
            rec["boundary_action_jump_linf"] = j_linf
            jn_l1, jn_linf = l1_linf(raw_chunk[0, 0], last_raw_chunk[-1])
            rec["boundary_action_jump_norm_l1"] = jn_l1
            rec["boundary_action_jump_norm_linf"] = jn_linf
        rows.append(rec)
        last_sent_chunk = sent_chunk.detach().cpu()
        last_raw_chunk = raw_chunk[0].detach().cpu()

    return rows


def summarize(rows: list[dict]) -> dict:
    keys = [
        "raw_action0_to_obs_l1", "raw_action0_to_obs_linf",
        "sent_action0_to_obs_l1", "sent_action0_to_obs_linf",
        "boundary_action_jump_l1", "boundary_action_jump_linf",
        "boundary_action_jump_norm_l1", "boundary_action_jump_norm_linf",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        a = np.array(vals)
        out[k] = {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-hf-id", default="Shiki42/0520_pi0.5screw_rlt_cotrain_c")
    p.add_argument("--policy-path", default=None, help="local path overrides hf id (root, not /rlt)")
    p.add_argument("--dataset-repo-id", default="Shiki42/0420_0423screw_critical_c")
    p.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve.")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="outputs/boundary_jump_vla_only_20260521.json")
    args = p.parse_args()

    device = torch.device(args.device)

    # 1. Resolve local model path
    if args.policy_path:
        local = args.policy_path
    else:
        from huggingface_hub import snapshot_download
        local = snapshot_download(args.policy_hf_id, repo_type="model")
    logger.info("Policy root: %s", local)

    # 2. Load pi0.5 directly from cotrained root (the "VLA-only" baseline)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    pi_cfg = PreTrainedConfig.from_pretrained(local)
    policy = PI05Policy.from_pretrained(local, config=pi_cfg).to(device)
    policy.eval()

    # 3. Pre/post processors at root
    from lerobot.policies.factory import make_pre_post_processors
    preproc, postproc = make_pre_post_processors(pi_cfg, pretrained_path=local)

    # 4. Dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    delta_ts = {"action": [i / 30.0 for i in range(args.chunk_length)]}
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )
    total_ep = len(ds.meta.episodes)
    eps = list(range(min(args.num_episodes, total_ep)))
    logger.info("Dataset episodes: %d total; sweeping %d", total_ep, len(eps))

    # 5. Run
    all_rows: list[dict] = []
    t0 = time.time()
    for i, ep_idx in enumerate(eps):
        ep_rows = run_episode(
            policy, preproc, postproc, ds, ep_idx,
            args.chunk_length, args.task_instruction, device,
        )
        all_rows.extend(ep_rows)
        if (i + 1) % 5 == 0:
            logger.info("Episode %d/%d done (%d rows so far)", i + 1, len(eps), len(all_rows))

    elapsed = time.time() - t0
    summary = summarize(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "policy_hf_id": args.policy_hf_id,
        "policy_path": args.policy_path,
        "policy_local": local,
        "dataset": args.dataset_repo_id,
        "num_episodes": len(eps),
        "chunk_length": args.chunk_length,
        "elapsed_sec": elapsed,
        "summary": summary,
        "rows": all_rows,
    }, indent=2))
    logger.info("Wrote %s (%d rows, %.1fs)", out_path, len(all_rows), elapsed)

    logger.info("=" * 60)
    logger.info("SUMMARY (mean / p50 / p95 / max)")
    for k, v in summary.items():
        logger.info("  %-38s  mean=%.4f  p50=%.4f  p95=%.4f  max=%.4f",
                    k, v["mean"], v["p50"], v["p95"], v["max"])


if __name__ == "__main__":
    main()
