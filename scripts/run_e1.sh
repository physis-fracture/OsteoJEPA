#!/usr/bin/env bash
# E1 - does the age mechanism work.
#
#   bash scripts/run_e1.sh            full: uses the Stage A checkpoint in runs/base
#   bash scripts/run_e1.sh --smoke    M0 path: 2-layer ViT, 200 images, 2 epochs
#
# The smoke path trains as well as scores, because the point of M0 is to prove
# that dataset -> valid_mask -> pretraining -> age sweep -> score -> r_study runs
# end to end before a single GPU-hour is spent on it.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  echo "no virtualenv found at .venv - create it with: python -m venv .venv" >&2
  exit 1
fi

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

if [ "$SMOKE" = "1" ]; then
  CONFIG="configs/exp/smoke.yaml"
  RUN_NAME="smoke_$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="runs/${RUN_NAME}"

  echo "== Stage A pretraining (smoke) =="
  "$PY" scripts/pretrain.py --config "$CONFIG" --set "run.name=${RUN_NAME}"
  CHECKPOINT="${RUN_DIR}/checkpoints/best.pt"
else
  CONFIG="configs/exp/e1.yaml"
  RUN_NAME="e1_$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="runs/${RUN_NAME}"
  CHECKPOINT="runs/base/checkpoints/best.pt"
  if [ ! -f "$CHECKPOINT" ]; then
    echo "missing Stage A checkpoint: ${CHECKPOINT}" >&2
    echo "run M3 first, or use --smoke to exercise the pipeline" >&2
    exit 1
  fi
fi

echo "== Calibration: lambda first, then (mu, sigma) =="
"$PY" scripts/calibrate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --run-subdir calibrate \
  --set "run.name=${RUN_NAME}"

echo "== Age sweep, surprise maps, r_image, r_study =="
"$PY" scripts/sweep_score.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --calibration "${RUN_DIR}/calibrate/calibration.json" \
  --run-subdir score \
  --set "run.name=${RUN_NAME}"

echo "== Done: ${RUN_DIR} =="
