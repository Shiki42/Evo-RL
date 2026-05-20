#!/usr/bin/env bash
# Master orchestrator for the 2026-05-14 overnight RLT run on coder b.
# Phases:
#   0. Wait until 09:00 Australia/Sydney (skippable with --skip-gate)
#   1. Task A bisect (mandatory items [1][2][3])
#   2. Task B Phase 1 γ sweep (5 cells, sequential)
# Status / decisions logged under outputs/rlt_sweep_b_2026-05-14/orchestrator.log
set -uo pipefail

REPO=/home/coder/code/Evo-RL-quick
VENV=/home/coder/venv-lerobot
SWEEP_DIR=$REPO/outputs/rlt_sweep_b_2026-05-14
mkdir -p "$SWEEP_DIR"
ORCH_LOG=$SWEEP_DIR/orchestrator.log
STATS_LEGACY=$REPO/outputs/legacy_stats/token_stats_full.pt

log() {
  echo "[$(TZ=Australia/Sydney date -Is)] $*" | tee -a "$ORCH_LOG"
}

SKIP_GATE=0
if [ "${1:-}" = "--skip-gate" ]; then
  SKIP_GATE=1
fi

log "=== Orchestrator start (skip_gate=$SKIP_GATE) ==="
log "Sweep dir: $SWEEP_DIR"
log "Stats (legacy): $STATS_LEGACY"

# ----- Phase 0: 9am Sydney gate -----
if [ "$SKIP_GATE" -eq 0 ]; then
  log "Waiting for 09:00 Australia/Sydney..."
  while [ "$(TZ=Australia/Sydney date +%H%M)" -lt 0900 ]; do
    sleep 60
  done
  log "Gate open: $(TZ=Australia/Sydney date -Is)"
else
  log "Gate skipped"
fi

# ----- Phase 1: Task A bisect -----
log "=== Task A bisect ==="
TASK_A_DIR=$SWEEP_DIR/task_a
mkdir -p "$TASK_A_DIR"

cd "$REPO" && "$VENV/bin/python" scripts/rlt_training/task_a_bisect.py \
  --out-dir "$TASK_A_DIR" \
  --model-path /home/coder/share/policy_pi05_screw \
  --dataset Shiki42/0420_0423screw \
  --legacy-stats "$STATS_LEGACY" \
  >> "$ORCH_LOG" 2>&1

TASK_A_RC=$?
log "Task A bisect rc=$TASK_A_RC"
log "Task A summary: $(cat $TASK_A_DIR/task_a_summary.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k: v.get(\"status\") for k,v in d.get(\"items\",{}).items()})' 2>/dev/null || echo 'parse_failed')"

# Decision: continue to Task B regardless (we will document in REPORT.md whatever Task A finds)
# Mandatory-fail conditions handled inside the bisect script

# ----- Phase 2: Task B Phase 1 γ sweep -----
log "=== Task B Phase 1 (γ sweep) ==="
RUN_CELL=$REPO/scripts/rlt_training/run_cell.sh
chmod +x "$RUN_CELL"

# Cell order: baseline → unweighted control → 0.25 → 0.75 → 1.0
declare -a CELLS=(
  "gamma_0p50_baseline 0.5 3 3 2e-4"
  "gamma_0p00_unweighted 0.0 3 3 2e-4"
  "gamma_0p25 0.25 3 3 2e-4"
  "gamma_0p75 0.75 3 3 2e-4"
  "gamma_1p00 1.0 3 3 2e-4"
)

for cell_spec in "${CELLS[@]}"; do
  read -r name gamma enc dec lr <<<"$cell_spec"
  log "--- Cell $name γ=$gamma L=$enc/$dec lr=$lr ---"

  start_ts=$(date +%s)
  "$RUN_CELL" "$name" "$gamma" "$enc" "$dec" "$lr" "$STATS_LEGACY" >> "$ORCH_LOG" 2>&1
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  status=$(cat "$SWEEP_DIR/cells/$name/STATUS.txt" 2>/dev/null || echo "UNKNOWN")
  log "Cell $name DONE: status=$status elapsed=${elapsed}s"

  # Optional: cool-down pause between cells
  sleep 30
done

log "=== Phase 1 complete; writing summary ==="

# Write summary CSV
SUMMARY=$SWEEP_DIR/phase1_summary.tsv
{
  echo -e "cell\tgamma\tL\tlr\tstatus\tfinal_loss\tavg_last_500\tmax_grad"
  for cell_spec in "${CELLS[@]}"; do
    read -r name gamma enc dec lr <<<"$cell_spec"
    cdir=$SWEEP_DIR/cells/$name
    st=$(cat "$cdir/STATUS.txt" 2>/dev/null || echo "MISSING")
    final_loss=$(grep -oE 'loss=[0-9.eE+-]+' "$cdir/train.log" 2>/dev/null | tail -1 | sed 's/loss=//' || echo "")
    avg500=$(grep -oE 'loss=[0-9.eE+-]+' "$cdir/train.log" 2>/dev/null | tail -10 | sed 's/loss=//' | awk '{s+=$1; n++} END {if (n>0) printf "%.6f", s/n}')
    grad_max=$(grep -oE 'grad_norm=[0-9.eE+-]+' "$cdir/train.log" 2>/dev/null | sed 's/grad_norm=//' | awk 'BEGIN {m=0} {if ($1>m) m=$1} END {print m}')
    echo -e "$name\t$gamma\t$enc\t$lr\t$st\t$final_loss\t$avg500\t$grad_max"
  done
} > "$SUMMARY"

log "Phase 1 summary: $SUMMARY"
cat "$SUMMARY" | tee -a "$ORCH_LOG"

log "=== Orchestrator done ==="
