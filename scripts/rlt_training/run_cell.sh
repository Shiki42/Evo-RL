#!/usr/bin/env bash
# Run one RL Token sweep cell with two-phase gate: 2K probe → 15K extend (or kill).
# Usage: run_cell.sh <cell_name> <gamma> <enc_layers> <dec_layers> <lr> <stats_path>
set -uo pipefail

NAME="$1"
GAMMA="$2"
L_ENC="$3"
L_DEC="$4"
LR="$5"
STATS="$6"

REPO=/home/coder/code/Evo-RL-quick
VENV=/home/coder/venv-lerobot
SFT_MODEL=/home/coder/share/policy_pi05_screw
DATASET=Shiki42/0420_0423screw
OUT="$REPO/outputs/rlt_sweep_b_2026-05-14/cells/$NAME"
mkdir -p "$OUT"
STATUS_FILE="$OUT/STATUS.txt"
LOG_PROBE="$OUT/probe_2k.log"
LOG_EXTEND="$OUT/extend_15k.log"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "RUNNING_PROBE" > "$STATUS_FILE"
echo "[$(date -Is)] Cell $NAME γ=$GAMMA L=$L_ENC/$L_DEC lr=$LR stats=$STATS" | tee "$LOG_PROBE"

run_train() {
  local steps="$1" output_dir="$2" log="$3"
  "$VENV/bin/python" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET" --dataset.revision=main \
    --dataset.video_backend=pyav \
    --policy.type=rlt_token \
    --policy.vla_pretrained_path="$SFT_MODEL" \
    --policy.vla_dtype=bfloat16 \
    --policy.rl_token_num_rl_tokens=1 \
    --policy.rl_token_enc_layers="$L_ENC" --policy.rl_token_dec_layers="$L_DEC" \
    --policy.rl_token_ff_dim=4096 --policy.rl_token_nhead=8 \
    --policy.token_pool_size=0 --policy.image_only=false \
    --policy.norm_stats_path="$STATS" --policy.norm_gamma="$GAMMA" \
    --policy.push_to_hub=false \
    --batch_size=2 --steps="$steps" --num_workers=2 \
    --log_freq=50 --save_freq=$((steps / 2)) --save_checkpoint=true \
    --output_dir="$output_dir" --wandb.enable=false \
    --use_policy_training_preset=false \
    --optimizer.type=adamw --optimizer.lr="$LR" \
    --optimizer.weight_decay=0.0 --optimizer.grad_clip_norm=1.0 \
    --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr="$LR" \
    --scheduler.decay_lr=5e-6 \
    --scheduler.num_warmup_steps=200 --scheduler.num_decay_steps="$steps" \
    --seed=1000 \
    >> "$log" 2>&1
}

# ----- Probe: 2K steps -----
echo "[$(date -Is)] Probe 2K starting" | tee -a "$LOG_PROBE"
run_train 2000 "$OUT/probe_2k" "$LOG_PROBE"
PROBE_RC=$?
echo "[$(date -Is)] Probe rc=$PROBE_RC" | tee -a "$LOG_PROBE"

# Parse last logged step + loss + grdn
last_line=$(grep -E "^[A-Z][a-z]+ +[0-9]+ +[0-9:]+ +[A-Z]+ +[a-zA-Z_/.]+:[0-9]+ +step:" "$LOG_PROBE" 2>/dev/null | tail -1)
if [ -z "$last_line" ]; then
  # fallback to any line with step:
  last_line=$(grep -E "step:[0-9]" "$LOG_PROBE" 2>/dev/null | tail -1)
fi
last_loss=$(echo "$last_line" | grep -oE "loss:[0-9.eE+-]+" | sed 's/loss://')
last_grdn=$(echo "$last_line" | grep -oE "grdn:[0-9.eE+-]+" | sed 's/grdn://')

# Get loss trend from probe: first vs last 10 logged points
losses=$(grep -oE "loss:[0-9.eE+-]+" "$LOG_PROBE" 2>/dev/null | sed 's/loss://')
n_losses=$(echo "$losses" | wc -l)
early_avg=$(echo "$losses" | head -10 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
late_avg=$(echo "$losses" | tail -10 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')

# Max grad over probe
max_grdn=$(grep -oE "grdn:[0-9.eE+-]+" "$LOG_PROBE" 2>/dev/null | sed 's/grdn://' | awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m+0}')

echo "[$(date -Is)] Probe metrics: last_loss=$last_loss last_grdn=$last_grdn early=$early_avg late=$late_avg max_grdn=$max_grdn n_log=$n_losses" | tee -a "$LOG_PROBE"

# Decision: loss decreasing AND max_grdn < 5 AND rc=0
decision="FAIL"
reason=""

if [ "$PROBE_RC" -ne 0 ]; then
  reason="probe_rc=$PROBE_RC"
elif [ -z "$late_avg" ] || [ "$late_avg" = "0" ]; then
  reason="no_loss_logged"
elif grep -qE "nan|NaN|inf|Inf" "$LOG_PROBE" 2>/dev/null; then
  # Confirm it's a loss line, not just any "inf" word
  if echo "$last_line" | grep -qE "loss:[^ ]*nan|loss:[^ ]*inf"; then
    reason="NaN_in_loss"
  else
    : # benign
  fi
fi

if [ -z "$reason" ]; then
  decreasing=$(awk -v a="$early_avg" -v b="$late_avg" 'BEGIN { print (b < a ? "1" : "0") }')
  grad_ok=$(awk -v g="$max_grdn" 'BEGIN { print (g < 5.0 ? "1" : "0") }')
  if [ "$decreasing" = "1" ] && [ "$grad_ok" = "1" ]; then
    decision="PASS"
  else
    reason="decreasing=$decreasing,max_grdn=$max_grdn"
  fi
fi

if [ "$decision" = "FAIL" ]; then
  echo "KILLED:2K_gate_fail:$reason" > "$STATUS_FILE"
  echo "[$(date -Is)] Probe FAIL: $reason" | tee -a "$LOG_PROBE"
  exit 0
fi

# ----- Extend: 15K steps (fresh run) -----
echo "PASS_2K_EXTENDING" > "$STATUS_FILE"
echo "[$(date -Is)] Probe PASS — launching 15K" | tee -a "$LOG_EXTEND"
run_train 15000 "$OUT/extend_15k" "$LOG_EXTEND"
EXT_RC=$?

# Final metrics
final_losses=$(grep -oE "loss:[0-9.eE+-]+" "$LOG_EXTEND" 2>/dev/null | sed 's/loss://')
final_avg100=$(echo "$final_losses" | tail -100 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
final_avg500=$(echo "$final_losses" | tail -500 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
final_last=$(echo "$final_losses" | tail -1)
final_max_grdn=$(grep -oE "grdn:[0-9.eE+-]+" "$LOG_EXTEND" 2>/dev/null | sed 's/grdn://' | awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m+0}')

echo "[$(date -Is)] Extension rc=$EXT_RC last=$final_last avg100=$final_avg100 avg500=$final_avg500 max_grdn=$final_max_grdn" | tee -a "$LOG_EXTEND"

if [ "$EXT_RC" -eq 0 ]; then
  echo "DONE_15K|final_loss=$final_last|avg100=$final_avg100|avg500=$final_avg500|max_grdn=$final_max_grdn" > "$STATUS_FILE"
else
  if grep -qiE "out of memory|cuda.*oom" "$LOG_EXTEND"; then
    echo "OOM_15K|rc=$EXT_RC" > "$STATUS_FILE"
  else
    echo "EXIT_${EXT_RC}_15K|avg100=$final_avg100|avg500=$final_avg500" > "$STATUS_FILE"
  fi
fi

echo "[$(date -Is)] Cell $NAME final: $(cat $STATUS_FILE)" | tee -a "$LOG_EXTEND"
