"""Record SFT VLA HIL data without the pre-r prefix segment.

This is the PI0.5/SFT comparison entrypoint for
``record_rlt_hil_wo_prefix.py``. It keeps the same human-in-the-loop episode
flow:

  * Before the first ``r`` press of each episode, the robot stays in teleop so
    the operator can reset or position the scene.
  * Press ``r`` to start the episode. From that point on, the policy action
    sent to the robot is the raw SFT VLA chunk, not the AC actor output.
  * The VLA chunk execution length is fixed to 25 actions per inference.
  * Press ``r`` again to end the episode as success; double-tap inside the
    window to mark failure. SPACE still toggles human intervention.

The script still loads a ChunkACPolicy checkpoint through ``--policy-path`` so
it can reuse the deploy-time PI0.5 path and the exact saved pre/postprocessor
pipeline. The AC actor is bypassed by forcing ``policy.phase_mode=always_vla``.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset.setup_helpers import (
    get_sorted_followers,
    get_sorted_leaders,
    load_setup_json,
    resolve_dataset_root,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record SFT VLA HIL data without prefix segment")
    p.add_argument(
        "--policy-path",
        required=True,
        help="ChunkACPolicy ckpt dir used for PI0.5 + pre/postprocessor loading.",
    )
    p.add_argument(
        "--vla-path",
        default=None,
        help="Override AC config's vla_pretrained_path (pi0.5 SFT dir on this machine).",
    )
    p.add_argument(
        "--rl-token-path",
        default=None,
        help="Override AC config's rl_token_pretrained_path. Loaded but not used for actions.",
    )
    p.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--chunk-exec-steps", type=int, default=25)
    p.add_argument("--setup-json", default=None)
    p.add_argument("--dataset-tag", default="sft_vla_hil_wo_prefix")
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
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_camera_configs(cameras: list[dict]) -> tuple[dict, dict]:
    camera_rename = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
    left_camera_aliases = {"left_wrist"}
    right_camera_aliases = {"right_wrist", "right_front"}
    left_cameras, right_cameras = {}, {}
    for cam in cameras:
        alias = cam["alias"]
        new_name = camera_rename.get(alias, alias)
        cam_cfg = {
            "type": "opencv",
            "index_or_path": cam["port"],
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
        if cam.get("fourcc"):
            cam_cfg["fourcc"] = cam["fourcc"]
        if alias in left_camera_aliases:
            left_cameras[new_name] = cam_cfg
        elif alias in right_camera_aliases:
            right_cameras[new_name] = cam_cfg
    return left_cameras, right_cameras


def _stage_arm_calibration(cal_dir: str, side: str, arm: dict) -> None:
    serial = Path(arm["calibration_dir"]).name
    src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
    dst = Path(cal_dir) / f"bimanual_{side}.json"
    if src.exists():
        shutil.copy2(src, dst)
    else:
        log.warning("Calibration file not found: %s", src)


def _stage_leader_calibrations(leaders: list[dict], teleop_id: str) -> TemporaryDirectory | None:
    if len(leaders) < 2:
        return None
    leader_cal_dir = TemporaryDirectory(prefix="sft-vla-leader-cal-")
    for side, arm in [("left", leaders[0]), ("right", leaders[1])]:
        serial = Path(arm["calibration_dir"]).name
        src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
        dst = Path(leader_cal_dir.name) / f"{teleop_id}_{side}.json"
        if src.exists():
            shutil.copy2(src, dst)
            log.info("Leader calibration staged: %s -> %s", src, dst)
        else:
            log.warning("Leader calibration file not found: %s", src)
    return leader_cal_dir


def _build_teleop_argv(args: argparse.Namespace, leaders: list[dict]) -> tuple[list[str], str]:
    teleop_id = "bimanual_leader"
    if args.no_teleop or len(leaders) < 2:
        log.warning("Teleop disabled; human intervention not available")
        return [], teleop_id
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={teleop_id}",
    ], teleop_id


def _build_policy_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [
        f"--policy.path={args.policy_path}",
        "--policy.phase_mode=always_vla",
        f"--policy.chunk_exec_steps={args.chunk_exec_steps}",
    ]
    if args.vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={args.vla_path}")
    if args.rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={args.rl_token_path}")
    return overrides


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

    now = datetime.now()
    date_folder = now.strftime("%m%d") + f"_{args.dataset_tag}"
    time_tag = now.strftime("%H%M%S")
    dataset_leaf = f"eval_sft_vla_hil_wo_prefix_{time_tag}"
    day_dir = resolve_dataset_root(setup) / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"

    log_file = day_dir / f"{dataset_leaf}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=== record_sft_vla_hil_wo_prefix started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))
    teleop_argv, teleop_id = _build_teleop_argv(args, leaders)

    with TemporaryDirectory(prefix="sft-vla-hil-wo-prefix-") as cal_dir:
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            _stage_arm_calibration(cal_dir, side, arm)

        leader_cal_dir = None
        if teleop_argv:
            leader_cal_dir = _stage_leader_calibrations(leaders, teleop_id)
            if leader_cal_dir is not None:
                teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")

        sys.argv = [
            "record_sft_vla_hil_wo_prefix",
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
            "--enable_episode_outcome_labeling=true",
            "--intervention_state_machine_enabled=true",
            f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
            "--play_sounds=true",
        ]

        log.info("Calling record() with %d argv entries", len(sys.argv))
        print(f"\nDataset: {dataset_name} -> {dataset_root}")
        print(f"Log: {log_file}")
        print(f"Policy: {args.policy_path}")
        print(f"Control source: SFT VLA passthrough, chunk_exec_steps={args.chunk_exec_steps}")
        print(
            "SFT VLA HIL wo-prefix mode: teleop -> r=start SFT VLA episode; "
            "in policy mode: r=end success, r+r within "
            f"{args.double_tap_window_s:.1f}s=end failure, SPACE=intervene"
        )
        print()

        from lerobot.scripts.lerobot_record import record
        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()

    log.info("=== record_sft_vla_hil_wo_prefix finished ===")


if __name__ == "__main__":
    main()
