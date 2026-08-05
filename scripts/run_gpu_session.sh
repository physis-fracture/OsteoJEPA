#!/usr/bin/env bash
# One GPU session: everything that needs a GPU, in order, in one go.
#
#   bash scripts/run_gpu_session.sh                  full session
#   bash scripts/run_gpu_session.sh --skip-smoke     skip the 4-minute check
#   bash scripts/run_gpu_session.sh --resume PATH    continue an interrupted run
#   bash scripts/run_gpu_session.sh --skip-train --run-dir runs/base   sweeps only
#
# The split is deliberate. Pretraining and the age sweep need the card; lambda
# selection, AUROC tables, the lambda ablation and all of E3 are arithmetic over
# s_rec, s_min and a_hat, which this session writes to disk. Download the
# archives at the end and the rest of the work needs no GPU at all.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  echo "no virtualenv found at .venv - run: bash scripts/setup_env.sh" >&2
  exit 1
fi

CONFIG="configs/base.yaml"
SKIP_SMOKE=0
SKIP_TRAIN=0
RESUME=""
RUN_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --resume) RESUME="${2:?--resume needs a checkpoint path}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?--run-dir needs a path}"; shift 2 ;;
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -f "data/manifest.csv" ] || { echo "data/manifest.csv missing - run scripts/setup_env.sh" >&2; exit 1; }

echo "== 0. data checks =="
"$PY" scripts/check_data.py --config "$CONFIG" --set "run.name=data_check_$(date +%H%M%S)"

if [ "$SKIP_SMOKE" = "0" ]; then
  echo
  echo "== 1. smoke: the whole pipeline on a 2-layer model =="
  bash scripts/run_e1.sh --smoke
fi

if [ "$SKIP_TRAIN" = "0" ]; then
  echo
  echo "== 2. benchmark: throughput, VRAM, margin cost =="
  "$PY" scripts/bench_step.py --config "$CONFIG" --steps 50 --warmup 5 \
    --set "run.name=bench_$(date +%H%M%S)"

  echo
  echo "== 3. Stage A pretraining =="
  PRETRAIN_LOG="$(mktemp)"
  if [ -n "$RESUME" ]; then
    "$PY" scripts/pretrain.py --config "$CONFIG" --resume "$RESUME" 2>&1 | tee "$PRETRAIN_LOG"
  else
    "$PY" scripts/pretrain.py --config "$CONFIG" 2>&1 | tee "$PRETRAIN_LOG"
  fi
  RUN_DIR="$(grep '^RUN_DIR=' "$PRETRAIN_LOG" | tail -1 | cut -d= -f2-)"
  rm -f "$PRETRAIN_LOG"
fi

[ -n "$RUN_DIR" ] || { echo "no run directory; pass --run-dir with --skip-train" >&2; exit 1; }
CHECKPOINT="${RUN_DIR}/checkpoints/best.pt"
[ -f "$CHECKPOINT" ] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
echo "run directory: ${RUN_DIR}"

# The sweeps cover every image of the validation and test folds, clean and
# fractured alike. Clean images give the comparator group and the calibration;
# images with boxes give the positive patches. Sweeping only the clean set would
# mean renting the card again for E1.
for SPLIT in val test; do
  echo
  echo "== 4.${SPLIT}. age sweep over the whole ${SPLIT} fold =="
  "$PY" scripts/sweep_score.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" --subset all \
    --no-normalize \
    --run-subdir "sweep_${SPLIT}"
done

echo
echo "== 5. calibration: lambda on raw scores, then (mu, sigma) =="
"$PY" scripts/calibrate.py \
  --config "$CONFIG" \
  --from-sweep "${RUN_DIR}/sweep_val/sweep_val_all.npz" \
  --run-subdir calibrate

CALIBRATION="${RUN_DIR}/calibrate/calibration.json"

for SPLIT in val test; do
  echo
  echo "== 6.${SPLIT}. scores from the archive (no GPU) =="
  "$PY" scripts/score_from_sweep.py \
    --config "$CONFIG" \
    --sweep "${RUN_DIR}/sweep_${SPLIT}/sweep_${SPLIT}_all.npz" \
    --calibration "$CALIBRATION" \
    --run-subdir "score_${SPLIT}"
done

ARCHIVE="physis_artifacts_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" \
  "${RUN_DIR}/checkpoints/best.pt" \
  "${RUN_DIR}/config.resolved.yaml" \
  "${RUN_DIR}/git.txt" \
  "${RUN_DIR}/metrics.json" \
  "${RUN_DIR}/log.txt" \
  "${RUN_DIR}/figures" \
  "${RUN_DIR}"/sweep_*/sweep_*.npz \
  "$CALIBRATION" \
  "${RUN_DIR}"/score_*/scores.json

echo
echo "== done =="
echo "artifacts: ${ARCHIVE}"
echo
echo "Download that one file. It carries the checkpoint, both sweep archives,"
echo "lambda* with the band statistics, the metrics and the monitor figures."
echo "M5, M7 and the calibration for M6 all run off it without a GPU."
