#!/bin/bash
# Watcher: waits for the overnight RL token sweep to finish, then runs post_sweep_pipeline.py.
# Triggers post-pipeline on any of:
#   - DONE_SWEEP marker in orchestrator.log (sweep finished cleanly)
#   - overnight tmux session vanished (sweep crashed)
#   - 10-hour wall-clock timeout
set +e
echo START $(date)
START_EPOCH=$(date +%s)
MAX_WAIT=$((10*3600))
ORCH_LOG=/home/coder/share/Evo-RL-quick/outputs/rlt_overnight_20260502/orchestrator.log
while true; do
  if [ -f "$ORCH_LOG" ] && grep -q DONE_SWEEP "$ORCH_LOG" 2>/dev/null; then
    echo SWEEP_DONE_NORMAL $(date)
    break
  fi
  if ! tmux has-session -t overnight 2>/dev/null; then
    echo SWEEP_TMUX_DEAD $(date)
    break
  fi
  ELAPSED=$(($(date +%s) - START_EPOCH))
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo SWEEP_TIMEOUT $(date)
    break
  fi
  sleep 60
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
