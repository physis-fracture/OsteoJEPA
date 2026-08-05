#!/usr/bin/env bash
# Create the virtualenv, install dependencies, and extract the dataset.
#
#   bash scripts/setup_env.sh              CUDA 12.8 build (RTX 5080 / 5090)
#   bash scripts/setup_env.sh --cpu        CPU build, for the smoke path
#   bash scripts/setup_env.sh --cuda cu124 another CUDA build
#
# torch >= 2.7 is required on Blackwell cards (sm_120); older builds have no
# kernels for them and fail at the first matmul rather than at import.
#
# The image archives are not in git. Put physis_meta.zip and physis_images_*.zip
# in dataset/ before running this, or extract them into data/ yourself.
set -euo pipefail

cd "$(dirname "$0")/.."

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu"; shift ;;
    --cuda) TORCH_INDEX="https://download.pytorch.org/whl/${2:?--cuda needs a tag}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ ! -d ".venv" ]; then
  echo "== creating .venv =="
  python3 -m venv .venv 2>/dev/null || python -m venv .venv
fi

if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
else
  PY=".venv/bin/python"
fi

echo "== installing dependencies =="
"$PY" -m pip install -q --upgrade pip uv
"$PY" -m uv pip install torch --index-url "$TORCH_INDEX"
"$PY" -m uv pip install -e ".[dev]"

echo "== extracting dataset =="
"$PY" - <<'PYTHON'
import pathlib, zipfile

out = pathlib.Path("data")
zips = sorted(pathlib.Path("dataset").glob("*.zip"))
if not zips:
    raise SystemExit(
        "no archives in dataset/. Fetch physis_meta.zip and physis_images_*.zip "
        "from the Hugging Face dataset repository first."
    )
out.mkdir(exist_ok=True)
for archive in zips:
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(out)
    print(f"extracted {archive.name}")

images = list((out / "images_384").glob("*.png"))
print(f"images: {len(images)}")
assert (out / "manifest.csv").exists(), "manifest.csv is missing; physis_meta.zip was not extracted"
PYTHON

echo "== verifying =="
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
"$PY" -m pytest -q

echo
echo "Ready. Next:"
echo "  bash scripts/run_e1.sh --smoke     verify the pipeline end to end"
echo "  bash scripts/run_base.sh --bench   measure before committing to M3"
