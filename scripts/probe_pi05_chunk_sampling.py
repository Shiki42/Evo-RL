#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_STATE


def make_batch(preprocessor, item: dict) -> dict:
    observation = {key: value for key, value in item.items() if str(key).startswith("observation.")}
    observation["task"] = item["task"]
    observation["robot_type"] = "bi_so_follower"
    return preprocessor(observation)


def denorm_action(postprocessor, action: torch.Tensor) -> np.ndarray:
    return postprocessor(action).detach().cpu().numpy().reshape(-1)


def profile_to_current(chunk_raw: np.ndarray, obs: np.ndarray) -> list[float]:
    return np.abs(chunk_raw - obs[None]).sum(axis=1).astype(float).tolist()


def profile_to_future(chunk_raw: np.ndarray, dataset: LeRobotDataset, row: int) -> list[float]:
    future_states = []
    base_episode = int(dataset[row]["episode_index"].item())
    for offset in range(min(chunk_raw.shape[0], len(dataset) - row)):
        item = dataset[row + offset]
        if int(item["episode_index"].item()) != base_episode:
            break
        future_states.append(item[OBS_STATE].numpy().astype(np.float32))
    future = np.stack(future_states)
    return np.abs(chunk_raw[: len(future)] - future).sum(axis=1).astype(float).tolist()


def profile_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    min_idx = int(arr.argmin())
    return {
        "first15": [round(float(value), 3) for value in arr[:15]],
        "min_idx": min_idx,
        "min": round(float(arr[min_idx]), 3),
        "last": round(float(arr[-1]), 3),
    }


def probe_row(policy, preprocessor, postprocessor, dataset: LeRobotDataset, row: int, seeds: list[int]) -> dict:
    item = dataset[row]
    batch = make_batch(preprocessor, item)
    obs = item[OBS_STATE].numpy().astype(np.float32)
    raw_action0 = []
    seed0_chunk = None

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            chunk = policy.predict_action_chunk(batch)
            if seed0_chunk is None:
                seed0_chunk = chunk.detach()
            raw_action0.append(denorm_action(postprocessor, chunk[:, 0]))
        noise = torch.zeros(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            device=policy.config.device,
        )
        zero_chunk = policy.predict_action_chunk(batch, noise=noise)

    action0 = np.stack(raw_action0)
    l1 = np.abs(action0 - obs[None]).sum(axis=1)
    pairwise_l1 = np.abs(action0[:, None, :] - action0[None, :, :]).sum(axis=2)
    upper = pairwise_l1[np.triu_indices_from(pairwise_l1, k=1)]
    upper_mean = float(upper.mean()) if upper.size else 0.0
    chunk_raw = np.stack(
        [denorm_action(postprocessor, seed0_chunk[:, step]) for step in range(seed0_chunk.shape[1])]
    )
    zero_action0 = denorm_action(postprocessor, zero_chunk[:, 0])

    return {
        "row": row,
        "frame_index": int(item["frame_index"].item()),
        "random_l1_mean": round(float(l1.mean()), 3),
        "random_l1_min": round(float(l1.min()), 3),
        "random_l1_max": round(float(l1.max()), 3),
        "seed_l1s": [round(float(value), 3) for value in l1],
        "pairwise_l1_mean": round(upper_mean, 3),
        "pairwise_l1_max": round(float(pairwise_l1.max()), 3),
        "zero_noise_l1": round(float(np.abs(zero_action0 - obs).sum()), 3),
        "seed0_chunk_to_current": profile_summary(profile_to_current(chunk_raw, obs)),
        "seed0_chunk_to_future_state": profile_summary(profile_to_future(chunk_raw, dataset, row)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=[0, 25, 50])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--repo-id", default="local/pi05")
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        download_videos=False,
        video_backend=args.video_backend,
    )
    policy = PI05Policy.from_pretrained(str(args.checkpoint), local_files_only=True)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    seeds = list(range(args.seeds))
    results = [probe_row(policy, preprocessor, postprocessor, dataset, row, seeds) for row in args.rows]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
