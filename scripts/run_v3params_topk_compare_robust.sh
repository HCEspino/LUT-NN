#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

FIG_DIR="figures"
mkdir -p "$FIG_DIR"
LOG="$FIG_DIR/topk_compare_robust.log"
: > "$LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

COMMON_ARGS=(
  --env-id CartPole-v1
  --episodes 200
  --max-steps 500
  --batch-size 128
  --buffer-size 100000
  --warmup-steps 1000
  --gamma 0.99
  --lr 5e-4
  --target-update 200
  --tau 0.005
  --hidden 128
  --num-tables 8
  --num-comparisons 6
  --num-blocks 2
  --eps-start 1.0
  --eps-end 0.02
  --eps-decay-steps 6000
  --algo dqn
  --inter-block-norm none
  --train-freq 1
  --grad-steps 1
  --eval-every 25
  --eval-episodes 10
  --seed 42
  --device cuda
  --table-lr-mult 3.0
  --grad-clip 5.0
  --figures-dir "$FIG_DIR"
)

EXIT_ANY=0
for K in 1 2 3; do
  RUN="cartpole_v3params_norelu_topk${K}_e200"
  log "START $RUN"
  set +e
  python3 -u scripts/cartpole_lut_dqn_v4.py "${COMMON_ARGS[@]}" --surrogate-topk "$K" --run-name "$RUN" >> "$LOG" 2>&1
  RC=$?
  set -e
  log "END $RUN rc=$RC"
  if [ $RC -ne 0 ]; then
    EXIT_ANY=1
  fi
done

log "PLOTTING combined figure"
set +e
python3 -u scripts/plot_topk_compare.py \
  --runs cartpole_v3params_norelu_topk1_e200 cartpole_v3params_norelu_topk2_e200 cartpole_v3params_norelu_topk3_e200 \
  --figures-dir "$FIG_DIR" \
  --out "$FIG_DIR/cartpole_v3params_norelu_topk123_e200_compare.png" >> "$LOG" 2>&1
PRC=$?
set -e
log "PLOT rc=$PRC"

if [ $EXIT_ANY -ne 0 ] || [ $PRC -ne 0 ]; then
  log "DONE with failures"
  exit 1
else
  log "DONE success"
fi
