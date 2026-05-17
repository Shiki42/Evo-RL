#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _nested(row: dict[str, Any], key: str, subkey: str | None = None) -> float:
    value = row.get(key)
    if subkey is not None:
        if value is None:
            return np.nan
        value = value.get(subkey)
    if value is None:
        return np.nan
    return float(value)


def _summary(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.3f} p50={np.percentile(arr, 50):.3f} "
        f"p95={np.percentile(arr, 95):.3f} max={arr.max():.3f}"
    )


def _print_metric(rows: list[dict[str, Any]], key: str, subkey: str | None = "l1") -> None:
    label = key if subkey is None else f"{key}.{subkey}"
    print(f"{label}: {_summary([_nested(row, key, subkey) for row in rows])}")


def _distance_profile(row: dict[str, Any], key: str) -> list[float]:
    values = row.get(key)
    if not values:
        return []
    return [float(value["l1"]) for value in values]


def _format_profile(values: list[float]) -> str:
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    min_idx = int(arr.argmin())
    first = ",".join(f"{value:.1f}" for value in arr[:10])
    return f"first10=[{first}] min_idx={min_idx} min={arr[min_idx]:.3f} last={arr[-1]:.3f}"


def _load_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "action_state_debug.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _print_dataset_action_jumps(root: Path) -> None:
    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        return
    import pyarrow.parquet as pq

    actions = []
    episodes = []
    frames = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["episode_index", "frame_index", "action"])
        data = table.to_pydict()
        actions.extend(data["action"])
        episodes.extend(data["episode_index"])
        frames.extend(data["frame_index"])
    if len(actions) < 2:
        return
    action_arr = np.asarray(actions, dtype=np.float32)
    episode_arr = np.asarray(episodes)
    frame_arr = np.asarray(frames)
    same_episode = episode_arr[1:] == episode_arr[:-1]
    step_l1 = np.abs(np.diff(action_arr, axis=0)).sum(axis=1)
    step_l1 = step_l1[same_episode]
    boundary_mask = (frame_arr[1:] % 25 == 0)[same_episode]
    print(f"dataset_action_step_l1_all: {_summary(step_l1.tolist())}")
    print(f"dataset_action_step_l1_boundary_mod25: {_summary(step_l1[boundary_mask].tolist())}")


def analyze(root: Path) -> None:
    rows = _load_rows(root)
    print(f"root={root}")
    print(f"debug_rows={len(rows)}")
    if not rows:
        return
    for key, subkey in [
        ("infer_ms", None),
        ("obs_to_before_send", "l1"),
        ("raw_action0_to_obs", "l1"),
        ("executed_action0_to_obs", "l1"),
        ("previous_overlap_action0_to_obs", "l1"),
        ("robot_action_to_send_to_obs", "l1"),
        ("sent_action_to_obs", "l1"),
        ("raw_action0_to_obs", "linf"),
        ("executed_action0_to_obs", "linf"),
    ]:
        _print_metric(rows, key, subkey)

    raw = [_nested(row, "raw_action0_to_obs", "l1") for row in rows]
    executed = [_nested(row, "executed_action0_to_obs", "l1") for row in rows]
    delta = [r - e for r, e in zip(raw, executed) if np.isfinite(r) and np.isfinite(e)]
    print(f"raw_minus_executed_l1: {_summary(delta)}")
    if any(row.get("raw_chunk_to_obs") for row in rows):
        print("chunk_profiles:")
        for row in rows:
            print(
                "frame={:4d} raw {} executed {}".format(
                    row["episode_frame_index"],
                    _format_profile(_distance_profile(row, "raw_chunk_to_obs")),
                    _format_profile(_distance_profile(row, "executed_chunk_to_obs")),
                )
            )
    print("rows:")
    for row in rows:
        print(
            f"frame={row['episode_frame_index']:4d} "
            f"overlap={row.get('overlap_len')} "
            f"w={row.get('chunk_overlap_ensemble_prev_weight')} "
            f"bridge={row.get('chunk_boundary_bridge_steps')} "
            f"infer={row.get('infer_ms', 0):.1f}ms "
            f"raw_l1={_nested(row, 'raw_action0_to_obs', 'l1'):.3f} "
            f"exec_l1={_nested(row, 'executed_action0_to_obs', 'l1'):.3f} "
            f"prev_l1={_nested(row, 'previous_overlap_action0_to_obs', 'l1'):.3f} "
            f"sent_l1={_nested(row, 'sent_action_to_obs', 'l1'):.3f}"
        )
    _print_dataset_action_jumps(root)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_pi05_chunk0_debug.py ROOT [ROOT ...]")
    for arg in sys.argv[1:]:
        analyze(Path(arg))


if __name__ == "__main__":
    main()
