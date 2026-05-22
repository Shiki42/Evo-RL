#!/usr/bin/env bash
# Re-run cells that were killed by overly-strict probe gate. Skip probe; run 15K directly.
# Usage: run_cell_no_gate.sh <cell_name> <gamma> <enc> <dec> <lr> <stats>
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
LOG="$OUT/extend_15k.log"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "RUNNING_NO_GATE" > "$STATUS_FILE"
echo "[$(date -Is)] Cell $NAME (no gate) γ=$GAMMA L=$L_ENC/$L_DEC lr=$LR" | tee "$LOG"

"$VENV/bin/python" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET" --dataset.revision=main --dataset.video_backend=pyav \
  --policy.type=rlt_token \
  --policy.vla_pretrained_path="$SFT_MODEL" \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.rl_token_enc_layers="$L_ENC" --policy.rl_token_dec_layers="$L_DEC" \
  --policy.rl_token_ff_dim=4096 --policy.rl_token_nhead=8 \
  --policy.token_pool_size=0 --policy.image_only=false \
  --policy.norm_stats_path="$STATS" --policy.norm_gamma="$GAMMA" \
  --policy.push_to_hub=false \
  --batch_size=2 --steps=15000 --num_workers=2 \
  --log_freq=50 --save_freq=7500 --save_checkpoint=true \
  --output_dir="$OUT/extend_15k" --wandb.enable=false \
  --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr="$LR" \
  --optimizer.weight_decay=0.0 --optimizer.grad_clip_norm=1.0 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr="$LR" \
  --scheduler.decay_lr=5e-6 \
  --scheduler.num_warmup_steps=200 --scheduler.num_decay_steps=15000 \
  --seed=1000 \
  >> "$LOG" 2>&1
RC=$?

losses=$(grep -oE "loss:[0-9.eE+-]+" "$LOG" 2>/dev/null | sed 's/loss://')
avg100=$(echo "$losses" | tail -100 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
avg500=$(echo "$losses" | tail -500 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
last=$(echo "$losses" | tail -1)
max_grdn=$(grep -oE "grdn:[0-9.eE+-]+" "$LOG" 2>/dev/null | sed 's/grdn://' | awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m+0}')

if [ "$RC" -eq 0 ]; then
  echo "DONE_15K|final_loss=$last|avg100=$avg100|avg500=$avg500|max_grdn=$max_grdn" > "$STATUS_FILE"
else
  if grep -qiE "out of memory|cuda.*oom" "$LOG"; then
    echo "OOM_15K|rc=$RC" > "$STATUS_FILE"
  else
    echo "EXIT_${RC}_15K|avg100=$avg100" > "$STATUS_FILE"
  fi
fi
echo "[$(date -Is)] $NAME final: $(cat $STATUS_FILE)" | tee -a "$LOG"
