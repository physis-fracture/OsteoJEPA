"""E3, part three: end-to-end latency, from file bytes to triage score.

Measured through the serving path, not through an evaluation loop. The number a
user waits on includes decoding the upload, resizing, percentile clipping,
padding and the forward pass; a benchmark that starts after the tensor is
already built measures the wrong thing, which is the mistake the Stage A
benchmark made and paid for.

Reported beside the hardware it was measured on, as the paper's Section 3.3
requires.

    python scripts/measure_latency.py --checkpoint best.pt \\
        --calibration classifier_calibration.json --n 50
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.serve.scorer import Scorer
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="end-to-end serving latency")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def describe_hardware(device: torch.device) -> dict:
    info = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=latency")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg)
    log = run.log

    device = resolve_device(args.device or str(cfg.optim.device))
    scorer = Scorer(cfg, args.checkpoint, args.calibration, device=str(device))

    images_dir = Path(args.images_dir or cfg.images_dir)
    files = sorted(images_dir.glob("*.png"))[: args.n + args.warmup]
    assert files, f"no images under {images_dir}"
    # Read the bytes up front: the contract's latency starts when the service has
    # the upload, not when the disk does.
    payloads = [path.read_bytes() for path in files]

    stages: dict[str, list[float]] = {"total": [], "decode_preprocess": [], "forward": []}
    for index, blob in enumerate(payloads):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        from physis.serve.preprocess import load_grayscale, preprocess

        prepared = preprocess(load_grayscale(blob))
        if device.type == "cuda":
            torch.cuda.synchronize()
        after_preprocess = time.perf_counter()

        image = torch.from_numpy(prepared["image"])[None, None].to(device)
        valid = torch.from_numpy(prepared["valid_mask"])[None].to(device)
        with torch.no_grad():
            scorer.model(image, valid, scorer._meta(11.0, "M", 1, "L"))
        if device.type == "cuda":
            torch.cuda.synchronize()
        finished = time.perf_counter()

        if index >= args.warmup:
            stages["decode_preprocess"].append((after_preprocess - started) * 1000)
            stages["forward"].append((finished - after_preprocess) * 1000)
            stages["total"].append((finished - started) * 1000)

    report = {"hardware": describe_hardware(device), "n": len(stages["total"]), "stages": {}}
    for name, values in stages.items():
        array = np.asarray(values)
        report["stages"][name] = {
            "median_ms": float(np.median(array)),
            "p95_ms": float(np.percentile(array, 95)),
            "mean_ms": float(array.mean()),
        }
        log.info(
            "%-18s median %7.1f ms | p95 %7.1f ms", name,
            np.median(array), np.percentile(array, 95),
        )

    # A study is normally two projections, and the service scores them in
    # sequence, so the number a radiologist waits on is roughly twice one image.
    report["study_of_two_median_ms"] = 2 * report["stages"]["total"]["median_ms"]
    log.info(
        "a two-projection study: about %.0f ms on %s",
        report["study_of_two_median_ms"], report["hardware"]["device"],
    )
    run.write_json("latency.json", report)
    log.info("wrote %s", run.dir / "latency.json")


if __name__ == "__main__":
    main()
