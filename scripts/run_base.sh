#!/usr/bin/env bash
# M3 - Stage A pretraining, the full run.
#
#   bash scripts/run_base.sh --bench            50-step benchmark, no training
#   bash scripts/run_base.sh                    full run into runs/base
#   bash scripts/run_base.sh --resume PATH      continue from a checkpoint
#
# Run --bench first. Twenty minutes spent measuring images/sec and peak VRAM
# saves a day of discovering that the batch size does not fit or that the run
# would take longer than the deadline allows.
#
# The run writes a checkpoint every epoch (last.pt), keeps the best epoch
# (best.pt), and snapshots every run.checkpoint_every epochs, so an interrupted
# GPU box costs at most one epoch.
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
MODE="train"
RESUME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bench) MODE="bench"; shift ;;
    --resume) RESUME="${2:?--resume needs a checkpoint path}"; shift 2 ;;
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ "$MODE" = "bench" ]; then
  echo "== 50-step benchmark =="
  "$PY" scripts/bench_step.py --config "$CONFIG" --steps 50 --warmup 5
  echo
  echo "Check three numbers before starting the full run:"
  echo "  peak CUDA memory   - must fit the card, lower optim.batch_size if not"
  echo "  images/sec         - and the projected wall clock against the deadline"
  echo "  margin_time_ratio  - around 1.25; near 2.0 means an encoder runs twice"
  exit 0
fi

if [ -n "$RESUME" ]; then
  echo "== Stage A pretraining, resuming from ${RESUME} =="
  "$PY" scripts/pretrain.py --config "$CONFIG" --resume "$RESUME"
else
  echo "== Stage A pretraining =="
  "$PY" scripts/pretrain.py --config "$CONFIG"
fi

echo
echo "Next: bash scripts/run_e1.sh"
