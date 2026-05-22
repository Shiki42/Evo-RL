"""Run a pi0.5 VLA with Real-Time Chunking (RTC) on the bimanual SO101 robot.

Thin wrapper around ``examples/rtc/eval_with_real_robot.py``: it auto-builds the
``bi_so_follower`` robot args and stages follower calibration from the roboclaw
``setup.json``, so RTC can be launched with just ``--policy-path`` plus the
``--rtc-*`` knobs (no hand-written ``--robot.*`` args).

Live demo only — ``eval_with_real_robot.py`` runs the policy on the robot for
``--duration`` seconds with async chunking + latency tracking and does NOT save
a dataset. Adapted from ``zhaobo-4090-1:scripts/run_policy_rtc.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset.setup_helpers import get_sorted_followers, load_setup_json
from scripts.record_rlt_hil_wo_prefix import _build_camera_configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a VLA with RTC on the bimanual SO101 robot")
    p.add_argument("--policy-path", required=True,
                   help="pi0.5 (type=pi05) checkpoint dir: config.json + model.safetensors + processor JSONs.")
    p.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    p.add_argument("--duration", type=float, default=120.0, help="Demo duration in seconds.")
    p.add_argument("--fps", type=float, default=30.0, help="Action execution frequency (Hz).")
    p.add_argument("--setup-json", default=None)
    p.add_argument("--rtc-enabled", default="true", choices=["true", "false"],
                   help="false runs the same policy without RTC (baseline).")
    p.add_argument("--rtc-execution-horizon", type=int, default=10)
    p.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    p.add_argument("--rtc-prefix-attention-schedule", default="EXP",
                   choices=["EXP", "LINEAR", "ONES", "ZEROS"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    setup = load_setup_json(args.setup_json)
    followers = get_sorted_followers(setup)
    if len(followers) < 2:
        sys.exit(f"Need 2 follower arms, got {len(followers)}")
    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))

    eval_script = Path(__file__).resolve().parent.parent / "examples" / "rtc" / "eval_with_real_robot.py"
    if not eval_script.exists():
        sys.exit(f"RTC eval script not found: {eval_script}")

    with TemporaryDirectory(prefix="rtc-bimanual-cal-") as cal_dir:
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            if not src.exists():
                sys.exit(f"Calibration file not found: {src}")
            shutil.copy2(src, Path(cal_dir) / f"bimanual_{side}.json")

        sys.argv = [
            "eval_with_real_robot",
            f"--policy.path={args.policy_path}",
            "--policy.device=cuda",
            "--device=cuda",
            f"--rtc.enabled={args.rtc_enabled}",
            f"--rtc.execution_horizon={args.rtc_execution_horizon}",
            f"--rtc.max_guidance_weight={args.rtc_max_guidance_weight}",
            f"--rtc.prefix_attention_schedule={args.rtc_prefix_attention_schedule}",
            "--robot.type=bi_so_follower",
            "--robot.id=bimanual",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.left_arm_config.port={followers[0]['port']}",
            "--robot.left_arm_config.use_degrees=true",
            f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
            f"--robot.right_arm_config.port={followers[1]['port']}",
            "--robot.right_arm_config.use_degrees=true",
            f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
            f"--task={args.task}",
            f"--duration={args.duration}",
            f"--fps={args.fps}",
        ]

        print(f"RTC live demo (no dataset recorded) — policy: {args.policy_path}")
        print(f"RTC: enabled={args.rtc_enabled} horizon={args.rtc_execution_horizon} "
              f"guidance={args.rtc_max_guidance_weight} schedule={args.rtc_prefix_attention_schedule} "
              f"fps={args.fps} duration={args.duration}s")
        runpy.run_path(str(eval_script), run_name="__main__")


if __name__ == "__main__":
    main()
