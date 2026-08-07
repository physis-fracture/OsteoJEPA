"""Modal entrypoint for the GPU stages.

This file orchestrates; it does not reimplement. Every stage shells out to the
same `scripts/*.py` the local bash path uses, with the same arguments, so there
is exactly one copy of the training loop, the sweep, and the calibration order.
A second implementation living here would be a second place for the padding rule
and the calibration order to drift.

What runs where is unchanged: pretraining and the age sweep need the card,
everything downstream is arithmetic over the saved sweep archives.

    pip install modal && modal setup                  # once, locally

    modal volume create physis-data
    modal volume put physis-data dataset /zips        # the four zip archives
    modal run modal_app.py::extract_data              # once, ~1 minute

    modal run modal_app.py --bench-only               # measure first
    modal run modal_app.py                            # the whole session
    modal run modal_app.py --skip-smoke --skip-train  # sweeps only
    modal run modal_app.py --resume /runs/base/checkpoints/last.pt
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import modal

APP_NAME = "physis"
HOUR = 3600
ROOT = "/root"

# Measured on A100-40GB: peak 14.23 GiB at batch 64, 347 img/s, margin cost
# ratio 1.09x. A 16 GB card would be uncomfortably tight. Raise to "H100" for
# speed; the benchmark stage prints peak memory either way, so the batch size is
# a measurement rather than a guess.
#
# 8 CPUs per GPU function: every training step decodes batch_size 16-bit PNGs,
# and a starved dataloader would make every other optimization irrelevant.
GPU = "A100-40GB"
CPUS = 8.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.7",
        "timm>=1.0.9",
        "omegaconf>=2.3",
        "numpy>=1.26",
        "pandas>=2.2",
        "pillow>=10.3",
        "scikit-learn>=1.5",
        "scipy>=1.13",
        "torchmetrics>=1.4",
        "faster-coco-eval>=1.6",
        "fastapi[standard]>=0.115",
        "matplotlib>=3.9",
    )
    .add_local_dir("src", remote_path=f"{ROOT}/src")
    .add_local_dir("scripts", remote_path=f"{ROOT}/scripts")
    .add_local_dir("configs", remote_path=f"{ROOT}/configs")
)

# Two volumes, and the split matters. Data is written once and read forever;
# runs are written constantly and are what you lose if a container dies without
# committing.
data_volume = modal.Volume.from_name("physis-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("physis-runs", create_if_missing=True)
VOLUMES = {"/data": data_volume, "/runs": runs_volume}

app = modal.App(APP_NAME, image=image)

# configs/data.yaml points at a relative data/ directory. On Modal the dataset
# lives on a volume, so the paths are overridden rather than the config edited:
# configs/data.yaml records fixed data facts and should not move because the
# compute did.
PATH_OVERRIDES = [
    "manifest=/data/manifest.csv",
    "fracture_boxes=/data/fracture_boxes.csv",
    "images_dir=/data/images_384",
]


def overrides(name: str, *extra: str) -> list[str]:
    return [*PATH_OVERRIDES, f"run.name={name}", f"run.out_dir=/runs/{name}", *extra]


def require_absent(path: str, remedy: str) -> None:
    """Fail rather than let a stale directory be read as a fresh one.

    The runs volume persists between calls, and `setup_run` refuses to overwrite
    an existing run directory by adding a timestamp suffix. On a fresh box those
    two facts never met. Here they do: a second session would train into
    `/runs/base_20260806_...` while every later stage still reads
    `/runs/base/checkpoints/best.pt` — the previous checkpoint, silently, with
    plausible numbers coming out the other end.
    """
    if pathlib.Path(path).exists():
        raise RuntimeError(f"{path} already exists.\n{remedy}")


def run_script(script: str, args: list[str]) -> None:
    """Run one pipeline script and commit the runs volume afterwards.

    Modal commits volumes in the background and on clean shutdown, but a stage
    that fails should still leave its checkpoints and metrics behind. Committing
    explicitly is what makes a crashed run resumable instead of lost.
    """
    command = [sys.executable, f"{ROOT}/scripts/{script}", *args]
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    runs_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"scripts/{script} exited with {result.returncode}")


@app.function(volumes={"/data": data_volume}, timeout=HOUR)
def extract_data() -> dict:
    """Expand the uploaded archives inside the volume.

    Uploading four zip files and extracting here beats pushing 20,327 small PNGs
    over the network.
    """
    import zipfile

    target = pathlib.Path("/data")
    archives = sorted((target / "zips").glob("*.zip"))
    assert archives, (
        "no archives under /data/zips. Run: modal volume put physis-data dataset /zips"
    )
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
        print(f"extracted {archive.name}", flush=True)

    data_volume.commit()
    images = len(list((target / "images_384").glob("*.png")))
    assert (target / "manifest.csv").exists(), (
        "manifest.csv missing; physis_meta.zip was not uploaded"
    )
    print(f"images: {images}")
    return {"images": images}


@app.function(volumes=VOLUMES, timeout=HOUR, cpu=CPUS)
def check_data() -> None:
    name = f"data_check_{int(time.time())}"
    run_script("check_data.py", ["--config", f"{ROOT}/configs/base.yaml", "--set", *overrides(name)])


@app.function(gpu=GPU, volumes=VOLUMES, timeout=HOUR, cpu=CPUS)
def smoke() -> None:
    """The whole pipeline on a 2-layer model, before any GPU-hour is committed."""
    config = f"{ROOT}/configs/exp/smoke.yaml"
    name = f"smoke_{int(time.time())}"
    base = f"/runs/{name}"
    run_script("pretrain.py", ["--config", config, "--set", *overrides(name)])
    run_script(
        "sweep_score.py",
        ["--config", config, "--checkpoint", f"{base}/checkpoints/best.pt",
         "--split", "val", "--subset", "all", "--no-normalize",
         "--run-subdir", "sweep_val", "--set", *overrides(name)],
    )
    run_script(
        "calibrate.py",
        ["--config", config, "--from-sweep", f"{base}/sweep_val/sweep_val_all.npz",
         "--run-subdir", "calibrate", "--set", *overrides(name)],
    )
    run_script(
        "score_from_sweep.py",
        ["--config", config, "--sweep", f"{base}/sweep_val/sweep_val_all.npz",
         "--calibration", f"{base}/calibrate/calibration.json",
         "--run-subdir", "score_val", "--set", *overrides(name)],
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=HOUR, cpu=CPUS)
def benchmark(steps: int = 50) -> None:
    """Throughput, peak VRAM, and the margin-loss cost ratio."""
    run_script(
        "bench_step.py",
        ["--config", f"{ROOT}/configs/base.yaml", "--steps", str(steps), "--warmup", "5",
         "--set", *overrides(f"bench_{int(time.time())}")],
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=24 * HOUR, cpu=CPUS)
def pretrain(name: str = "base", resume: str = "", config: str = "configs/base.yaml") -> None:
    """Stage A. 24 hours is Modal's ceiling for one call.

    A run needing longer does not need a bigger timeout; it needs to be called
    again with --resume, which continues into the same run directory and the
    same metrics.json so the loss curve stays in one piece.
    """
    if not resume:
        require_absent(
            f"/runs/{name}",
            f"Pass --resume /runs/{name}/checkpoints/last.pt to continue it, "
            f"choose another --run-name, or delete it: modal volume rm -r physis-runs /{name}",
        )
    args = ["--config", f"{ROOT}/{config}", "--set", *overrides(name)]
    if resume:
        args += ["--resume", resume]
    run_script("pretrain.py", args)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=6 * HOUR, cpu=CPUS)
def train_classifier(
    name: str,
    use_condition: bool = True,
    holdout_bands: str = "",
    split_mode: str = "grouped",
    init_from: str = "",
) -> None:
    """The supervised model the product ranks by, and three of the experiments.

    E2a is `use_condition`, E2b is `holdout_bands`, and the split-leakage number
    is `split_mode`; each is the same run with one thing changed, which is what
    makes the comparisons mean anything.
    """
    require_absent(
        f"/runs/{name}",
        f"Choose another --name, or delete it: modal volume rm -r physis-runs /{name}",
    )
    extra = [
        f"classifier.use_condition={str(bool(use_condition)).lower()}",
        f"classifier.split_mode={split_mode}",
        "data.augment=true",
    ]
    if init_from:
        extra.append(f"classifier.init_from={init_from}")
    if holdout_bands:
        extra.append("classifier.holdout_bands=[" + holdout_bands + "]")
    run_script(
        "train_classifier.py",
        ["--config", f"{ROOT}/configs/base.yaml", "--set", *overrides(name, *extra)],
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * HOUR, cpu=CPUS)
def score_classifier(name: str = "clf_main") -> None:
    """Re-score a trained classifier keeping logits. Inference only, no training.

    The training run saved sigmoid outputs, which have no float32 resolution
    left near 1: the top-20% flagged group shares 84 distinct values across 815
    studies, so a worklist cannot rank what it flags.
    """
    run_script(
        "score_classifier.py",
        ["--config", f"{ROOT}/configs/base.yaml",
         "--checkpoint", f"/runs/{name}/checkpoints/best.pt",
         "--run-subdir", "rescored", "--set", *overrides(name)],
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=6 * HOUR, cpu=CPUS)
def train_detector(name: str = "det_main", split_mode: str = "grouped") -> None:
    """The comparator detector, and the mAP half of the split-leakage figure."""
    require_absent(
        f"/runs/{name}",
        f"Choose another --name, or delete it: modal volume rm -r physis-runs /{name}",
    )
    run_script(
        "train_detector.py",
        ["--config", f"{ROOT}/configs/base.yaml", "--set",
         *overrides(name, f"detector.split_mode={split_mode}", "data.augment=true")],
    )


@app.function(gpu=GPU, volumes=VOLUMES, timeout=12 * HOUR, cpu=CPUS)
def sweep(split: str, name: str = "base") -> None:
    """Age sweep over a whole fold, clean and fractured alike.

    Clean images give the calibration and the comparator group; images carrying
    boxes give the positive patches. Sweeping only the clean set means renting
    the card again for E1.
    """
    require_absent(
        f"/runs/{name}/sweep_{split}",
        f"Delete it first: modal volume rm -r physis-runs /{name}/sweep_{split}",
    )
    run_script(
        "sweep_score.py",
        ["--config", f"{ROOT}/configs/base.yaml",
         "--checkpoint", f"/runs/{name}/checkpoints/best.pt",
         "--split", split, "--subset", "all", "--no-normalize",
         "--run-subdir", f"sweep_{split}", "--set", *overrides(name)],
    )


@app.function(volumes=VOLUMES, timeout=HOUR, cpu=4.0)
def calibrate(name: str = "base") -> None:
    """Lambda on raw scores, then (mu, sigma) at that lambda. No GPU needed."""
    require_absent(
        f"/runs/{name}/calibrate",
        f"Delete it first: modal volume rm -r physis-runs /{name}/calibrate",
    )
    run_script(
        "calibrate.py",
        ["--config", f"{ROOT}/configs/base.yaml",
         "--from-sweep", f"/runs/{name}/sweep_val/sweep_val_all.npz",
         "--run-subdir", "calibrate", "--set", *overrides(name)],
    )


@app.function(volumes=VOLUMES, timeout=HOUR, cpu=4.0)
def score(split: str, name: str = "base") -> None:
    run_script(
        "score_from_sweep.py",
        ["--config", f"{ROOT}/configs/base.yaml",
         "--sweep", f"/runs/{name}/sweep_{split}/sweep_{split}_all.npz",
         "--calibration", f"/runs/{name}/calibrate/calibration.json",
         "--run-subdir", f"score_{split}_{int(time.time())}", "--set", *overrides(name)],
    )


@app.function(volumes=VOLUMES, timeout=HOUR, cpu=2.0, min_containers=0)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    """The triage service. Scales to zero between requests.

    Serving from the volume rather than baking the checkpoint into the image
    means a recalibration is a file write, not a rebuild.
    """
    import sys as _sys

    _sys.path.insert(0, f"{ROOT}/src")
    from physis.serve.api import build_app
    from physis.serve.scorer import Scorer
    from physis.utils.config import load_config

    holder: dict = {}

    def factory():
        if "scorer" not in holder:
            checkpoint = pathlib.Path("/runs/clf_main/checkpoints/best.pt")
            calibration = pathlib.Path("/runs/clf_calibration/classifier_calibration.json")
            if not (checkpoint.exists() and calibration.exists()):
                return None
            cfg = load_config(f"{ROOT}/configs/base.yaml", PATH_OVERRIDES)
            detector = pathlib.Path("/runs/det_main/checkpoints/best.pt")
            holder["scorer"] = Scorer(
                cfg, str(checkpoint), str(calibration),
                detector_checkpoint=str(detector) if detector.exists() else None,
            )
        return holder["scorer"]

    return build_app(factory)


@app.local_entrypoint()
def main(
    run_name: str = "base",
    skip_smoke: bool = False,
    skip_train: bool = False,
    bench_only: bool = False,
    resume: str = "",
) -> None:
    """The full session, in the order the milestones require."""
    print("== 0. data checks ==")
    check_data.remote()

    if bench_only:
        print("== benchmark only ==")
        benchmark.remote()
        return

    if not skip_smoke:
        print("== 1. smoke ==")
        smoke.remote()

    if not skip_train:
        print("== 2. benchmark ==")
        benchmark.remote()
        print("== 3. Stage A ==")
        pretrain.remote(name=run_name, resume=resume)

    for split in ("val", "test"):
        print(f"== 4.{split}. age sweep ==")
        sweep.remote(split, name=run_name)

    print("== 5. calibration ==")
    calibrate.remote(name=run_name)

    for split in ("val", "test"):
        print(f"== 6.{split}. scores ==")
        score.remote(split, name=run_name)

    print(
        f"\n== done ==\n"
        f"Pull the artifacts:\n"
        f"  modal volume get physis-runs /{run_name}/checkpoints/best.pt .\n"
        f"  modal volume get physis-runs /{run_name}/calibrate/calibration.json .\n"
        f"  modal volume get physis-runs /{run_name}/sweep_val/sweep_val_all.npz .\n"
        f"  modal volume get physis-runs /{run_name}/sweep_test/sweep_test_all.npz .\n"
        f"  modal volume get physis-runs /{run_name}/metrics.json .\n"
        f"  modal volume get physis-runs /{run_name}/figures .\n"
        f"M5, M7 and the calibration M6 consumes all run off those without a GPU."
    )
