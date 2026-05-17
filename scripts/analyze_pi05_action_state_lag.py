#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def fixed_list_to_numpy(column, width: int) -> np.ndarray:
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    return values.reshape(-1, width).astype(np.float32, copy=False)


def summarize(values: np.ndarray) -> str:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n=0"
    return (
        f"n={values.size} mean={values.mean():.3f} p50={np.percentile(values, 50):.3f} "
        f"p95={np.percentile(values, 95):.3f} max={values.max():.3f}"
    )


def lag_distance(action: np.ndarray, state: np.ndarray, episode: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return np.abs(action - state).sum(axis=1)
    if lag > 0:
        mask = episode[lag:] == episode[:-lag]
        return np.abs(action[lag:][mask] - state[:-lag][mask]).sum(axis=1)
    mask = episode[:lag] == episode[-lag:]
    return np.abs(action[:lag][mask] - state[-lag:][mask]).sum(axis=1)


def analyze_root(root: Path, max_lag: int, chunk_size: int, width: int) -> None:
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    action_parts = []
    state_parts = []
    episode_parts = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["action", "observation.state", "episode_index"])
        action_parts.append(fixed_list_to_numpy(table["action"], width))
        state_parts.append(fixed_list_to_numpy(table["observation.state"], width))
        episode_parts.append(table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False))
    action = np.concatenate(action_parts, axis=0)
    state = np.concatenate(state_parts, axis=0)
    episode = np.concatenate(episode_parts, axis=0)

    _, counts = np.unique(episode, return_counts=True)
    print(f"root={root}")
    print(
        f"rows={len(action)} episodes={len(counts)} mean_episode_len={counts.mean():.1f} "
        f"min_episode_len={counts.min()} max_episode_len={counts.max()}"
    )

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        distances = lag_distance(action, state, episode, lag)
        rows.append((lag, distances.mean(), distances))
    best_lag, best_mean, _ = min(rows, key=lambda item: item[1])
    print(f"best_lag_action[t+lag]_vs_state[t]={best_lag} mean_l1={best_mean:.3f}")
    for lag, _, distances in rows:
        if lag == best_lag or lag % 2 == 0:
            print(f"lag={lag:3d} {summarize(distances)}")

    same_l1 = np.abs(action - state).sum(axis=1)
    same_linf = np.abs(action - state).max(axis=1)
    print(f"same_time_l1 {summarize(same_l1)}")
    print(f"same_time_linf {summarize(same_linf)}")

    same_episode = episode[1:] == episode[:-1]
    action_step = np.abs(action[1:][same_episode] - action[:-1][same_episode]).sum(axis=1)
    state_step = np.abs(state[1:][same_episode] - state[:-1][same_episode]).sum(axis=1)
    print(f"action_step_l1 {summarize(action_step)}")
    print(f"state_step_l1 {summarize(state_step)}")

    total = counts.sum()
    pad_cells = 0
    for horizon in range(chunk_size):
        padded = np.minimum(horizon, counts).sum() if horizon > 0 else 0
        pad_cells += padded
        if horizon in {0, 1, 5, 10, chunk_size // 2, chunk_size - 1}:
            print(f"horizon={horizon} pad_frame_fraction={padded / total:.6f}")
    print(f"chunk_pad_cell_fraction={pad_cells / (total * chunk_size):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--width", type=int, default=12)
    args = parser.parse_args()
    for root in args.roots:
        analyze_root(root, args.max_lag, args.chunk_size, args.width)


if __name__ == "__main__":
    main()
