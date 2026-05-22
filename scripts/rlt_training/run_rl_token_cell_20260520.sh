#!/usr/bin/env bash
# Single RL Token training cell — invokes lerobot-train and writes a STATUS file.
# Usage: run_rl_token_cell.sh <name> <lr> <enc_L> <dec_L> <ff_dim> <nhead> <batch_size> <steps> <sft_path> <dataset_repo> <stats_path> <out_root> <wandb_proj>
set -uo pipefail

NAME="$1"; LR="$2"; L_ENC="$3"; L_DEC="$4"; FF="$5"; NHEAD="$6"; BS="$7"; STEPS="$8"
SFT="$9"; DATASET="${10}"; STATS="${11}"; OUT_ROOT="${12}"; WANDB_PROJ="${13}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV="${VENV:-$HOME/venv-lerobot}"
CELL_DIR="$OUT_ROOT/$NAME"
LEROBOT_OUT="$CELL_DIR/run"
mkdir -p "$CELL_DIR"
STATUS="$CELL_DIR/STATUS.txt"
LOG="$CELL_DIR/train.log"
OUT="$LEROBOT_OUT"  # passed to --output_dir below

if [ -e "$LEROBOT_OUT" ]; then
  echo "EXISTS|path=$LEROBOT_OUT" > "$STATUS"
  echo "Refusing to overwrite existing run dir: $LEROBOT_OUT" >&2
  exit 2
fi

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "RUNNING" > "$STATUS"
echo "[$(date -Is)] cell=$NAME lr=$LR L=$L_ENC/$L_DEC ff=$FF nhead=$NHEAD bs=$BS steps=$STEPS" | tee "$LOG"

cd "$REPO" || exit 1
"$VENV/bin/python" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET" --dataset.revision=main --dataset.video_backend=pyav \
  --policy.type=rlt_token \
  --policy.vla_pretrained_path="$SFT" \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.rl_token_enc_layers="$L_ENC" --policy.rl_token_dec_layers="$L_DEC" \
  --policy.rl_token_ff_dim="$FF" --policy.rl_token_nhead="$NHEAD" \
  --policy.token_pool_size=0 --policy.image_only=false \
  --policy.norm_stats_path="$STATS" --policy.norm_gamma=0.25 \
  --policy.push_to_hub=false \
  --batch_size="$BS" --steps="$STEPS" --num_workers=2 \
  --log_freq=50 --save_freq=$((STEPS / 2)) --save_checkpoint=true \
  --output_dir="$OUT" \
  --wandb.enable=true --wandb.project="$WANDB_PROJ" --wandb.entity=hushuyuan42 \
  --wandb.run_id="$NAME" \
  --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr="$LR" \
  --optimizer.weight_decay=0.0 --optimizer.grad_clip_norm=1.0 \
  --scheduler.type=cosine_decay_with_warmup --scheduler.peak_lr="$LR" \
  --scheduler.decay_lr=5e-6 \
  --scheduler.num_warmup_steps=200 --scheduler.num_decay_steps="$STEPS" \
  --seed=1000 \
  >> "$LOG" 2>&1
RC=$?

losses=$(grep -oE "loss:[0-9.eE+-]+" "$LOG" 2>/dev/null | sed 's/loss://')
avg100=$(echo "$losses" | tail -100 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
avg500=$(echo "$losses" | tail -500 | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n; else print "0"}')
last=$(echo "$losses" | tail -1)
max_grdn=$(grep -oE "grdn:[0-9.eE+-]+" "$LOG" 2>/dev/null | sed 's/grdn://' | awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m+0}')
nsteps=$(echo "$losses" | wc -l)

if [ "$RC" -eq 0 ]; then
  echo "DONE|rc=0|nsteps=$nsteps|last=$last|avg100=$avg100|avg500=$avg500|max_grdn=$max_grdn" > "$STATUS"
elif grep -qiE "out of memory|cuda.*oom" "$LOG"; then
  echo "OOM|rc=$RC|bs=$BS" > "$STATUS"
else
  echo "EXIT_${RC}|nsteps=$nsteps|last=$last|avg100=$avg100" > "$STATUS"
fi
echo "[$(date -Is)] cell=$NAME final: $(cat "$STATUS")" | tee -a "$LOG"
