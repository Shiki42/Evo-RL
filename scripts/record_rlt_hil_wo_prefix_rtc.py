"""Record RLT HIL data in wo-prefix mode with RTC guidance enabled.

This is the RTC variant of ``record_rlt_hil_wo_prefix.py``. It keeps the same
HIL episode controls and dataset schema, but enables Real-Time Chunking in the
frozen pi0.5 reference path used by ``rlt_ac``.

Usage on the real-robot machine:

    cd /home/kye/evo-rl
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate evo-rl
    PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_rlt_hil_wo_prefix_rtc.py \
        --policy-path /home/kye/rlt_deploy/ac_std0p02_b0p3 \
        --vla-path /home/kye/.cache/huggingface/hub/models--Shiki42--0520_pi0.5screw_rlt_cotrain_c/... \
        --rl-token-path /home/kye/rlt_deploy/rlt \
        --num-episodes 5 \
        --vla-ref
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset.setup_helpers import (
    get_sorted_followers,
    get_sorted_leaders,
    load_setup_json,
    resolve_dataset_root,
)
from scripts.record_rlt_hil_wo_prefix import _build_camera_configs

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record RLT HIL wo-prefix data with RTC")
    p.add_argument(
        "--policy-path",
        required=True,
        help="ChunkACPolicy ckpt dir (config.json + model.safetensors + processor JSONs).",
    )
    p.add_argument(
        "--vla-path",
        default=None,
        help="Override AC config's vla_pretrained_path (pi0.5 SFT dir on this machine).",
    )
    p.add_argument(
        "--rl-token-path",
        default=None,
        help="Override AC config's rl_token_pretrained_path.",
    )
    p.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--setup-json", default=None)
    p.add_argument("--dataset-tag", default="rlt_hil_wo_prefix_rtc")
    p.add_argument("--vcodec", default="h264")
    p.add_argument(
        "--no-teleop",
        action="store_true",
        default=False,
        help="Skip leader arm teleop (disables human intervention).",
    )
    p.add_argument(
        "--double-tap-window-s",
        type=float,
        default=0.6,
        help="Window for second 'r' press to mark failure instead of success.",
    )
    p.add_argument(
        "--vla-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the AC actor sees the RTC-guided VLA reference chunk. "
            "Pass --no-vla-ref to feed a zeroed reference."
        ),
    )
    p.add_argument(
        "--rtc-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RTC guidance in the pi0.5 reference path.",
    )
    p.add_argument(
        "--intervention-action-blend-time-s",
        type=float,
        default=0.4,
        help="Seconds to blend follower commands from policy to teleop after SPACE handoff.",
    )
    p.add_argument("--rtc-execution-horizon", type=int, default=10)
    p.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    p.add_argument(
        "--rtc-prefix-attention-schedule",
        default="EXP",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
    )
    p.add_argument(
        "--rtc-action-queue-size-to-get-new-actions",
        type=int,
        default=None,
        help=(
            "Queue refill threshold in action steps. Default is chunk_length - 1, "
            "which requests a new chunk soon after the current one starts."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_policy_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"--policy.path={args.policy_path}"]
    if args.vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={args.vla_path}")
    if args.rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={args.rl_token_path}")
    return overrides


def _build_rtc_argv(args: argparse.Namespace) -> list[str]:
    rtc_argv = [
        f"--rlt.rtc_enabled={'true' if args.rtc_enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={args.rtc_execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={args.rtc_max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={args.rtc_prefix_attention_schedule}",
    ]
    if args.rtc_action_queue_size_to_get_new_actions is not None:
        rtc_argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{args.rtc_action_queue_size_to_get_new_actions}"
        )
    return rtc_argv


def _resolve_paths(setup: dict[str, Any], args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    now = datetime.now()
    date_folder = now.strftime("%m%d") + f"_{args.dataset_tag}"
    dataset_leaf = f"eval_rlt_hil_wo_prefix_rtc_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    log_file = day_dir / f"{dataset_leaf}.log"
    return f"local/{dataset_leaf}", dataset_root, day_dir, log_file


def _build_teleop_argv(args: argparse.Namespace, leaders: list[dict[str, Any]]) -> list[str]:
    if args.no_teleop or len(leaders) < 2:
        log.warning("Teleop disabled - human intervention not available")
        return []

    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        "--teleop.id=bimanual_leader",
    ]


def _stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    serial = Path(arm["calibration_dir"]).name
    src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Calibration staged: %s -> %s", src, dst)
        return
    log.warning("Calibration file not found: %s", src)


def _stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    for side, arm in [("left", followers[0]), ("right", followers[1])]:
        _stage_arm_calibration(arm, Path(cal_dir) / f"bimanual_{side}.json")


def _stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str]
) -> TemporaryDirectory | None:
    if not teleop_argv:
        return None

    leader_cal_dir = TemporaryDirectory(prefix="rlt-leader-cal-")
    for side, arm in [("left", leaders[0]), ("right", leaders[1])]:
        _stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"bimanual_leader_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def _build_record_argv(
    args: argparse.Namespace,
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
    teleop_argv: list[str],
    dataset_name: str,
    dataset_root: Path,
) -> list[str]:
    return [
        "record_rlt_hil_wo_prefix_rtc",
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
        *teleop_argv,
        *_build_policy_overrides(args),
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.fps={args.fps}",
        f"--dataset.vcodec={args.vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={args.num_episodes + 1}",
        "--rlt.enable=true",
        "--rlt.skip_prefix_recording=true",
        "--rlt.rl_phase_key_toggles_episode=true",
        "--rlt.start_in_teleop=true",
        f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
        f"--rlt.intervention_action_blend_time_s={args.intervention_action_blend_time_s}",
        *_build_rtc_argv(args),
        "--enable_episode_outcome_labeling=true",
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        "--play_sounds=true",
    ]


def _print_run_summary(
    args: argparse.Namespace, dataset_name: str, dataset_root: Path, log_file: Path
) -> None:
    print(f"\nDataset: {dataset_name} -> {dataset_root}")
    print(f"Log: {log_file}")
    print(f"Policy: {args.policy_path}")
    print(f"VLA reference to AC actor: {'ON' if args.vla_ref else 'OFF (zeroed)'}")
    print(
        "RTC: "
        f"enabled={args.rtc_enabled} horizon={args.rtc_execution_horizon} "
        f"guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions or 'chunk_length-1'}"
    )
    print(f"Intervention action blend: {args.intervention_action_blend_time_s:.2f}s")
    print(
        "RLT HIL wo-prefix RTC mode: teleop->r=start RL episode; "
        "in RL: r=end success, r+r within "
        f"{args.double_tap_window_s:.1f}s=end failure, SPACE=intervene (returns to RL)"
    )
    print()


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    setup = load_setup_json(args.setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if len(followers) < 2:
        log.error("Need at least 2 follower arms, got %d", len(followers))
        sys.exit(1)

    dataset_name, dataset_root, day_dir, log_file = _resolve_paths(setup, args)
    day_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=== record_rlt_hil_wo_prefix_rtc started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)
    if args.rtc_enabled and not args.vla_ref:
        raise ValueError("RTC RLT recording requires --vla-ref; --no-vla-ref hides the guided reference.")
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))
    teleop_argv = _build_teleop_argv(args, leaders)

    with TemporaryDirectory(prefix="rlt-hil-wo-prefix-rtc-") as cal_dir:
        _stage_follower_calibrations(followers, cal_dir)
        leader_cal_dir = _stage_leader_calibrations(leaders, teleop_argv)
        sys.argv = _build_record_argv(
            args,
            followers,
            left_cameras,
            right_cameras,
            cal_dir,
            teleop_argv,
            dataset_name,
            dataset_root,
        )

        log.info("Calling record() with %d argv entries", len(sys.argv))
        _print_run_summary(args, dataset_name, dataset_root, log_file)
        from lerobot.scripts.lerobot_rlt_record import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()
    log.info("=== record_rlt_hil_wo_prefix_rtc finished ===")


if __name__ == "__main__":
    main()
