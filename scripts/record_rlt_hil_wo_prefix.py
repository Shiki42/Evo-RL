"""Record RLT HIL data without the VLA prefix segment (single --policy.path entry).

Pure RL-only HIL recording for online replay-buffer collection.

Model loading: pass a single ``--policy-path=<AC_ckpt_dir>`` (a ChunkACPolicy
checkpoint saved by ``lerobot-train --policy.type=rlt_ac``). The AC ckpt dir
must contain ``config.json``, ``model.safetensors``,
``policy_preprocessor.json`` (+ normalizer safetensors), and
``policy_postprocessor.json`` (+ unnormalizer safetensors). The preprocessor
JSON is byte-identical to the SFT pi05 preprocessor by construction
(``make_rlt_ac_pre_post_processors`` loads from
``config.vla_pretrained_path``), so deploy normalization equals SFT.

The frozen pi0.5 weights are NOT in the AC ckpt — they live wherever
``ChunkACPolicyConfig.vla_pretrained_path`` points. Use
``--vla-path=<local-dir>`` to override that path if the AC config's recorded
path doesn't exist on the deploy machine. Same for the RL Token ckpt:
``--rl-token-path=<dir>`` overrides ``rl_token_pretrained_path``.

Recording flow (unchanged from the legacy script):
  * VLA never drives the robot. Between episodes and before the first ``r``
    of each episode the robot is in human-teleop mode (leader drives
    follower). Press ``r`` to start an episode in RL mode.
  * Episode boundary is the ``r`` key. First press starts RL phase; second
    press ends it. Single end-press = success; double-tap inside the window
    = failure. After end, robot returns to teleop for human reset.
  * SPACE toggles human intervention during the RL phase.

Usage (on zhaobo-4090-1):
    cd ~/code/hsy/Evo-RL
    conda activate evo-rl
    PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_rlt_hil_wo_prefix.py \\
        --policy-path /path/to/ac_ckpt/checkpoints/last/pretrained_model \\
        --vla-path /home/zhaobo-4090-1/models/pi05_screw_271ep_sft_fp32 \\
        --rl-token-path /home/zhaobo-4090-1/checkpoints/rl_token_last/pretrained_model \\
        --num-episodes 5
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
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record RLT HIL data without prefix segment")
    p.add_argument("--policy-path", required=True,
                   help="ChunkACPolicy ckpt dir (config.json + model.safetensors + processor JSONs).")
    p.add_argument("--vla-path", default=None,
                   help="Override AC config's vla_pretrained_path (pi0.5 SFT dir on this machine).")
    p.add_argument("--rl-token-path", default=None,
                   help="Override AC config's rl_token_pretrained_path.")
    p.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--setup-json", default=None)
    p.add_argument("--dataset-tag", default="rlt_hil_wo_prefix")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--no-teleop", action="store_true", default=False,
                   help="Skip leader arm teleop (disables human intervention).")
    p.add_argument("--double-tap-window-s", type=float, default=0.6,
                   help="Window for second 'r' press to mark failure instead of success.")
    p.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether the AC actor sees the VLA reference chunk. Pass "
                        "--no-vla-ref to feed a zeroed reference (mirrors training ref-dropout).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_camera_configs(cameras: list[dict]) -> tuple[dict, dict]:
    CAM_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
    LEFT_CAMS = {"left_wrist"}
    RIGHT_CAMS = {"right_wrist", "right_front"}
    left_cameras, right_cameras = {}, {}
    for cam in cameras:
        alias = cam["alias"]
        new_name = CAM_RENAME.get(alias, alias)
        cam_cfg = {
            "type": "opencv",
            "index_or_path": cam["port"],
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
        if cam.get("fourcc"):
            cam_cfg["fourcc"] = cam["fourcc"]
        if alias in LEFT_CAMS:
            left_cameras[new_name] = cam_cfg
        elif alias in RIGHT_CAMS:
            right_cameras[new_name] = cam_cfg
    return left_cameras, right_cameras


def _disconnect_preflight_device(device: Any) -> None:
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(device, arm_name, None)
        if arm is not None and arm.is_connected:
            arm.disconnect()

    if getattr(device, "is_connected", False):
        device.disconnect()


def _preflight_motor_connections(
    followers: list[dict],
    leaders: list[dict],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    log.info("Preflight checking follower motor connections before loading policy")
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id="bimanual",
            calibration_dir=Path(cal_dir),
            left_arm_config=SOFollowerConfig(
                port=followers[0]["port"],
                use_degrees=True,
            ),
            right_arm_config=SOFollowerConfig(
                port=followers[1]["port"],
                use_degrees=True,
            ),
        )
    )
    try:
        robot.connect(calibrate=True)
        log.info("Preflight follower motor check passed")
    finally:
        _disconnect_preflight_device(robot)

    if not leaders or leader_cal_dir is None:
        return

    log.info("Preflight checking leader motor connections before loading policy")
    teleop = BiSOLeader(
        BiSOLeaderConfig(
            id="bimanual_leader",
            calibration_dir=Path(leader_cal_dir),
            left_arm_config=SOLeaderConfig(
                port=leaders[0]["port"],
                use_degrees=True,
            ),
            right_arm_config=SOLeaderConfig(
                port=leaders[1]["port"],
                use_degrees=True,
            ),
        )
    )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        _disconnect_preflight_device(teleop)


def main():
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

    now = datetime.now()
    date_folder = now.strftime("%m%d") + f"_{args.dataset_tag}"
    time_tag = now.strftime("%H%M%S")
    dataset_leaf = f"eval_rlt_hil_wo_prefix_{time_tag}"
    day_dir = resolve_dataset_root(setup) / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"

    log_file = day_dir / f"{dataset_leaf}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=== record_rlt_hil_wo_prefix started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))

    teleop_argv: list[str] = []
    teleop_id = "bimanual_leader"
    if not args.no_teleop and len(leaders) >= 2:
        teleop_argv = [
            "--teleop.type=bi_so_leader",
            f"--teleop.left_arm_config.port={leaders[0]['port']}",
            "--teleop.left_arm_config.use_degrees=true",
            f"--teleop.right_arm_config.port={leaders[1]['port']}",
            "--teleop.right_arm_config.use_degrees=true",
            f"--teleop.id={teleop_id}",
        ]
        log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    else:
        log.warning("Teleop disabled — human intervention not available")

    with TemporaryDirectory(prefix="rlt-hil-wo-prefix-") as cal_dir:
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            dst = Path(cal_dir) / f"bimanual_{side}.json"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                log.warning("Calibration file not found: %s", src)

        leader_cal_dir = None
        if teleop_argv and len(leaders) >= 2:
            leader_cal_dir = TemporaryDirectory(prefix="rlt-leader-cal-")
            for side, arm in [("left", leaders[0]), ("right", leaders[1])]:
                serial = Path(arm["calibration_dir"]).name
                src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
                dst = Path(leader_cal_dir.name) / f"{teleop_id}_{side}.json"
                if src.exists():
                    shutil.copy2(src, dst)
                    log.info("Leader calibration staged: %s -> %s", src, dst)
                else:
                    log.warning("Leader calibration file not found: %s", src)
            teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")

        _preflight_motor_connections(
            followers,
            leaders if teleop_argv else [],
            cal_dir,
            leader_cal_dir.name if leader_cal_dir is not None else None,
        )

        policy_overrides: list[str] = [f"--policy.path={args.policy_path}"]
        if args.vla_path is not None:
            policy_overrides.append(f"--policy.vla_pretrained_path={args.vla_path}")
        if args.rl_token_path is not None:
            policy_overrides.append(f"--policy.rl_token_pretrained_path={args.rl_token_path}")

        sys.argv = [
            "record_rlt_hil_wo_prefix",
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
            *policy_overrides,
            f"--dataset.repo_id={dataset_name}",
            f"--dataset.root={dataset_root}",
            f"--dataset.single_task={args.task}",
            f"--dataset.num_episodes={args.num_episodes}",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.fps={args.fps}",
            f"--dataset.vcodec={args.vcodec}",
            "--dataset.push_to_hub=false",
            f"--dataset.video_encoding_batch_size={args.num_episodes + 1}",
            # HIL recording flow flags (no model fields — those are in the AC ckpt).
            "--rlt.enable=true",
            "--rlt.skip_prefix_recording=true",
            "--rlt.rl_phase_key_toggles_episode=true",
            "--rlt.start_in_teleop=true",
            f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
            "--enable_episode_outcome_labeling=true",
            "--intervention_state_machine_enabled=true",
            f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
            f"--vla_ref={'true' if args.vla_ref else 'false'}",
            "--play_sounds=true",
        ]

        log.info("Calling record() with %d argv entries", len(sys.argv))
        print(f"\nDataset: {dataset_name} -> {dataset_root}")
        print(f"Log: {log_file}")
        print(f"Policy: {args.policy_path}")
        print(f"VLA reference to AC actor: {'ON' if args.vla_ref else 'OFF (zeroed)'}")
        print(
            "RLT HIL wo-prefix mode (pure RL): teleop→r=start RL episode; "
            "in RL: r=end success, r+r within "
            f"{args.double_tap_window_s:.1f}s=end failure, SPACE=intervene (returns to RL)"
        )
        print()

        from lerobot.scripts.lerobot_rlt_record import record
        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()

    log.info("=== record_rlt_hil_wo_prefix finished ===")


if __name__ == "__main__":
    main()
