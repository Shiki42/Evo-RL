#!/bin/bash
# Watcher v2: more robust completion detection.
# Triggers post-pipeline on any of:
#   - results.json has >= EXPECTED entries
#   - run_overnight_sweep python process not seen for 90 consecutive seconds (crashed/finished)
#   - 10h timeout
set +e
echo START $(date)
START_EPOCH=$(date +%s)
MAX_WAIT=$((10*3600))
EXPECTED=10
RESULTS=/home/coder/share/Evo-RL-quick/outputs/rlt_overnight_20260502/results.json
NO_PROC_STREAK=0
while true; do
  if [ -f "$RESULTS" ]; then
    N=$(/home/coder/share/venv-lerobot/bin/python -c "import json; print(len(json.load(open(\"$RESULTS\"))))" 2>/dev/null || echo 0)
    if [ "$N" -ge "$EXPECTED" ]; then
      echo SWEEP_COMPLETE n=$N $(date)
      break
    fi
  else
    N=0
  fi
  if pgrep -f "run_overnight_sweep_20260502" >/dev/null; then
    NO_PROC_STREAK=0
  else
    NO_PROC_STREAK=$((NO_PROC_STREAK + 1))
  fi
  if [ $NO_PROC_STREAK -ge 3 ]; then
    echo SWEEP_PROCESS_GONE n=$N streak=$NO_PROC_STREAK $(date)
    break
  fi
  ELAPSED=$(($(date +%s) - START_EPOCH))
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo SWEEP_TIMEOUT n=$N $(date)
    break
  fi
  if [ $((ELAPSED % 600)) -lt 30 ]; then
    echo "  watcher: elapsed=${ELAPSED}s n=$N proc_streak=$NO_PROC_STREAK"
  fi
  sleep 30
done

cd /home/coder/share/Evo-RL-quick
exec /home/coder/share/venv-lerobot/bin/python -u scripts/rlt_training/post_sweep_pipeline.py \
  --sweep-output outputs/rlt_overnight_20260502 \
  --model-path /home/coder/share/policy_pi05_screw \
  --dataset-path /home/coder/share/dataset/0420_0423screw \
  --task-instruction screw \
  --conditional-eight \
  --ac-gradient-steps 30000 \
  2>&1 | tee outputs/rlt_overnight_20260502/post_pipeline.log
echo PIPELINE_DONE $(date)
