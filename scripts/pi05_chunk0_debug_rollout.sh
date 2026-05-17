#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${WORKTREE:-/tmp/codex-pi05-n25-chunk0-20260518}
PYTHON_BIN=${PYTHON_BIN:-/home/kye/miniconda3/envs/evo-rl/bin/python}
POLICY_PATH=${POLICY_PATH:-/home/kye/.roboclaw/workspace/embodied/policies/pi05_abs_screw_new_old_cam_mix_60k_8gpu_bs8_20260427_2336}
RUN_ID=${RUN_ID:-eval_pi05_chunk0_n25_$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-/tmp/$RUN_ID}
LOG=${LOG:-/tmp/${RUN_ID}.log}
EPISODE_TIME_S=${EPISODE_TIME_S:-25}
N_ACTION_STEPS=${N_ACTION_STEPS:-25}
OVERLAP_PREV_WEIGHT=${OVERLAP_PREV_WEIGHT:-0.0}
BRIDGE_STEPS=${BRIDGE_STEPS:-0}
BRIDGE_TO_STATE=${BRIDGE_TO_STATE:-false}
TASK=${TASK:-Insert the copper screw into the black sleeve.}

cd "$WORKTREE"
export PYTHONPATH="$WORKTREE/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LEROBOT_ACTION_STATE_DEBUG_JSONL=${LEROBOT_ACTION_STATE_DEBUG_JSONL:-1}
export LEROBOT_ACTION_STATE_DEBUG_INCLUDE_CHUNKS=${LEROBOT_ACTION_STATE_DEBUG_INCLUDE_CHUNKS:-0}

set -x
set +e
"$PYTHON_BIN" -m lerobot.scripts.lerobot_record \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.calibration_dir=/home/kye/.roboclaw/workspace/embodied/calibration/bimanual_followers \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61032871-if00 \
  --robot.left_arm_config.use_degrees=true \
  '--robot.left_arm_config.cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61033526-if00 \
  --robot.right_arm_config.use_degrees=true \
  '--robot.right_arm_config.cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "front": {"type": "opencv", "index_or_path": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
  --policy.path="$POLICY_PATH" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.chunk_overlap_ensemble_prev_weight="$OVERLAP_PREV_WEIGHT" \
  --policy.chunk_boundary_bridge_steps="$BRIDGE_STEPS" \
  --policy.chunk_boundary_bridge_to_state="$BRIDGE_TO_STATE" \
  --dataset.repo_id="local/$RUN_ID" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false \
  --play_sounds=false 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
printf "\nRUN_ID=%s\nROOT=%s\nLOG=%s\nDEBUG_JSONL=%s\n" "$RUN_ID" "$ROOT" "$LOG" "$ROOT/action_state_debug.jsonl"
exit "$status"
