"""Trim each episode of a LeRobot dataset to the gripper-release frame.

Detects the first frame where action[--gripper-dim] >= --threshold and keeps
frames [0, cross_t] inclusive; episodes listed in --exclude-episodes are dropped
entirely. Output dataset is built via
``lerobot.utils.critical_phase_extraction_fast.extract_critical_phase_dataset_direct``.

Example::

    PYTHONPATH=src python scripts/trim_dataset_by_gripper.py \\
        --src /home/zhaobo-4090-1/.roboclaw/workspace/embodied/datasets/271ep_sft_success_critical_phase_478ep_filtered \\
        --dst /home/zhaobo-4090-1/.roboclaw/workspace/embodied/datasets/271ep_sft_success_critical_phase_478ep_filtered_gripper_trimmed \\
        --threshold 10.0 --gripper-dim 5 \\
        --task "Insert the copper screw into the black sleeve" \\
        --exclude-episodes 90,124,142,176,218,236
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _parse_int_list(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def compute_cross_frames(
    data_parquet: Path,
    gripper_dim: int,
    threshold: float,
) -> dict[int, tuple[int, int]]:
    """Return {episode_index: (episode_length, cross_t)}.  cross_t = -1 if no crossing."""
    df = pq.read_table(
        str(data_parquet), columns=["episode_index", "frame_index", "action"]
    ).to_pandas()
    out: dict[int, tuple[int, int]] = {}
    for ep_idx, grp in df.groupby("episode_index"):
        traj = np.stack(grp["action"].values)[:, gripper_dim]
        idxs = np.where(traj >= threshold)[0]
        cross_t = int(idxs[0]) if len(idxs) else -1
        out[int(ep_idx)] = (len(traj), cross_t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Source LeRobot dataset directory.")
    parser.add_argument("--dst", type=Path, required=True, help="Output LeRobot dataset directory (must not exist).")
    parser.add_argument("--threshold", type=float, required=True, help="action[gripper_dim] >= threshold => gripper open.")
    parser.add_argument("--gripper-dim", type=int, required=True, help="Index of the gripper action dimension.")
    parser.add_argument("--task", type=str, required=True, help="Task description string written into the output dataset.")
    parser.add_argument("--exclude-episodes", type=str, default="", help="Comma-separated episode indices to drop entirely.")
    parser.add_argument("--source-repo-id", type=str, default="local/trim_src", help="Dummy repo id for source.")
    parser.add_argument("--output-repo-id", type=str, default="local/trim_out", help="Dummy repo id for output.")
    parser.add_argument("--vcodec", type=str, default="h264", help="Video codec for the new dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Compute intervals, print summary, do not build dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("trim")

    if not args.src.exists():
        log.error("Source dataset not found: %s", args.src)
        sys.exit(1)
    if args.dst.exists():
        log.error("Destination already exists, refusing to overwrite: %s", args.dst)
        sys.exit(1)

    data_parquet = args.src / "data" / "chunk-000" / "file-000.parquet"
    if not data_parquet.exists():
        log.error("Expected single-file parquet at %s", data_parquet)
        sys.exit(1)

    exclude = set(_parse_int_list(args.exclude_episodes))
    log.info("Excluding %d episode(s): %s", len(exclude), sorted(exclude))

    log.info("Scanning action dim=%d against threshold=%.3f ...", args.gripper_dim, args.threshold)
    cross_map = compute_cross_frames(data_parquet, args.gripper_dim, args.threshold)

    intervals: list[tuple[int, int, int]] = []
    n_no_cross = 0
    total_src_frames = 0
    total_kept_frames = 0
    kept_tail_cuts: list[int] = []
    for ep_idx in sorted(cross_map):
        N, cross_t = cross_map[ep_idx]
        total_src_frames += N
        if ep_idx in exclude:
            continue
        if cross_t < 0:
            n_no_cross += 1
            continue
        end_exclusive = cross_t + 1
        intervals.append((ep_idx, 0, end_exclusive))
        total_kept_frames += end_exclusive
        kept_tail_cuts.append(N - end_exclusive)

    log.info(
        "Source: %d episodes, %d frames.  Kept: %d episodes, %d frames (%.1f%%).",
        len(cross_map), total_src_frames, len(intervals), total_kept_frames,
        100.0 * total_kept_frames / max(1, total_src_frames),
    )
    if n_no_cross:
        log.warning("Episodes with no threshold crossing (dropped): %d", n_no_cross)
    if kept_tail_cuts:
        arr = np.asarray(kept_tail_cuts)
        log.info(
            "Tail frames trimmed per kept episode: mean=%.2f median=%.1f min=%d max=%d p99=%.1f",
            arr.mean(), float(np.median(arr)), int(arr.min()), int(arr.max()), float(np.percentile(arr, 99)),
        )

    if args.dry_run:
        log.info("--dry-run: not building output dataset.")
        return

    from lerobot.utils.critical_phase_extraction_fast import extract_critical_phase_dataset_direct

    log.info("Building output dataset at %s (%d segments) ...", args.dst, len(intervals))
    result = extract_critical_phase_dataset_direct(
        source_repo_id=args.source_repo_id,
        source_root=args.src,
        output_repo_id=args.output_repo_id,
        output_root=args.dst,
        intervals=intervals,
        task=args.task,
        vcodec=args.vcodec,
    )
    if result is None:
        log.error("Extraction returned None; output dataset not created.")
        sys.exit(2)
    log.info("Done. Output at %s", result)


if __name__ == "__main__":
    main()
