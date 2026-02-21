#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

FIG_DIR="figures"
mkdir -p "$FIG_DIR"

# v3-like baseline parameters from prior best run context, but with ReLU dropped.
COMMON_ARGS=(
  --env-id CartPole-v1
  --episodes 350
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

for K in 1 2 3; do
  RUN="cartpole_v3params_norelu_topk${K}"
  echo "=== Starting ${RUN} ==="
  python3 scripts/cartpole_lut_dqn_v4.py "${COMMON_ARGS[@]}" \
    --surrogate-topk "$K" \
    --run-name "$RUN"
  echo "=== Finished ${RUN} ==="
done

python3 scripts/plot_topk_compare.py \
  --runs cartpole_v3params_norelu_topk1 cartpole_v3params_norelu_topk2 cartpole_v3params_norelu_topk3 \
  --figures-dir "$FIG_DIR" \
  --out "$FIG_DIR/cartpole_v3params_norelu_topk123_compare.png"

echo "DONE: $FIG_DIR/cartpole_v3params_norelu_topk123_compare.png"
