from __future__ import annotations

import json
from pathlib import Path

from scripts.rlt_online_cycle import (
    DEFAULT_BASE_AC_FILE,
    build_parser,
    discover_dataset_dirs,
    latest_ac_from_files,
    parse_result,
    sanitize_name,
)
from tests.rlt.helpers import make_test_algorithm


def _write_dataset(root: Path, name: str, episodes: int, frames: int) -> Path:
    ds = root / name
    meta = ds / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": episodes, "total_frames": frames}))
    return ds


def test_sanitize_name_makes_hf_safe_component() -> None:
    assert sanitize_name("ac_checkpoint/2026-05-28 10:20:30") == "ac_checkpoint_2026-05-28_10_20_30"


def test_latest_ac_prefers_newest_timestamp_checkpoint() -> None:
    files = [
        DEFAULT_BASE_AC_FILE,
        "ac_checkpoint/20260528_101010/rl_checkpoint.pt",
        "ac_checkpoint/20260528_121212/rl_checkpoint.pt",
        "ac_checkpoint/20260528_121212/metrics.json",
    ]
    ref = latest_ac_from_files(files)
    assert ref.version == "20260528_121212"
    assert ref.hf_path == "ac_checkpoint/20260528_121212/rl_checkpoint.pt"


def test_latest_ac_falls_back_to_base_checkpoint() -> None:
    ref = latest_ac_from_files([DEFAULT_BASE_AC_FILE])
    assert ref.version == "online_base_ac_0528"
    assert ref.hf_path == DEFAULT_BASE_AC_FILE


def test_discover_dataset_dirs_filters_empty_and_keeps_newest(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "eval_rlt_hil_wo_prefix_old", episodes=1, frames=10)
    _write_dataset(tmp_path, "eval_rlt_hil_wo_prefix_empty", episodes=0, frames=0)
    _write_dataset(tmp_path, "other", episodes=1, frames=10)
    newest = _write_dataset(tmp_path, "eval_rlt_hil_wo_prefix_new", episodes=2, frames=20)
    found = discover_dataset_dirs(tmp_path, ["eval_rlt_hil_wo_prefix_"], max_datasets=1)
    assert found == [newest]


def test_parse_result_uses_last_marker() -> None:
    stdout = 'noise\nRLT_ONLINE_RESULT {"a": 1}\nmore\nRLT_ONLINE_RESULT {"a": 2}\n'
    assert parse_result(stdout) == {"a": 2}


def test_algorithm_to_moves_target_actor_to_meta_device() -> None:
    algorithm, _ = make_test_algorithm()
    algorithm.to("meta")
    assert next(algorithm.target_actor.parameters()).device.type == "meta"


def test_cycle_does_not_truncate_training_episodes_by_default() -> None:
    args = build_parser().parse_args(["cycle"])
    assert args.train_max_episodes is None
