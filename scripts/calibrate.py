"""Calibration: choose lambda on raw scores, then compute (mu, sigma) per band.

Runs the age sweep over the clean validation images and writes
`calibration.json` holding lambda*, the W(lambda) table, the per-band statistics,
and the MAE of a_hat that the pre-registered abort criterion is tested against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, build_eval_frame
from physis.data.geometry import age_band_index, band_list
from physis.models.osteojepa import load_osteojepa
from physis.scoring.age_sweep import sweep_batch
from physis.scoring.calibrate import PatchScores, band_statistics, select_lambda
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lambda selection and band statistics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="not needed with --from-sweep")
    parser.add_argument(
        "--from-sweep",
        default=None,
        help=(
            "a sweep_*.npz written by sweep_score.py. s_rec, s_min and a_hat do "
            "not depend on lambda, so calibration can be redone from the archive "
            "without touching a GPU. E3's recalibration budget needs exactly that."
        ),
    )
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default="calibrate")
    return parser.parse_args()


def collect_from_sweep(path: str, log) -> tuple[PatchScores, list[float], list[float]]:
    """Rebuild the calibration inputs from a saved sweep, clean images only."""
    archive = np.load(path, allow_pickle=False)
    clean = archive["clean"]
    log.info("loaded %s: %d images, %d clean", path, len(clean), int(clean.sum()))
    assert clean.any(), "the archive holds no clean images; calibration needs them"

    scores = PatchScores(s_rec=[], delta=[], band=[])
    ages_recorded, ages_hat = [], []
    for index in np.flatnonzero(clean):
        valid = archive["valid"][index]
        if not valid.any():
            continue
        s_rec = archive["s_rec"][index][valid]
        s_min = archive["s_min"][index][valid]
        scores.s_rec.append(s_rec)
        scores.delta.append(s_rec - s_min)
        scores.band.append(int(archive["band"][index]))
        ages_recorded.append(float(archive["age"][index]))
        ages_hat.append(float(np.median(archive["a_hat"][index][valid])))
    return scores, ages_recorded, ages_hat


def collect_from_model(cfg, checkpoint: str, bands, log):
    """Run the age sweep over the clean validation images."""
    frame = build_eval_frame(cfg, "val", "clean")
    log.info("clean validation images: %d", len(frame))
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(dataset, batch_size=int(cfg.inference.batch_size), shuffle=False)
    device = resolve_device(str(cfg.optim.device))
    model = load_osteojepa(cfg, checkpoint, device)

    scores = PatchScores(s_rec=[], delta=[], band=[])
    ages_recorded, ages_hat = [], []
    for batch in loader:
        result = sweep_batch(model, batch, cfg, device)
        for b in range(result["s_rec"].shape[0]):
            valid = ~np.isnan(result["s_rec"][b])
            if not valid.any():
                continue
            scores.s_rec.append(result["s_rec"][b][valid])
            scores.delta.append(result["delta"][b][valid])
            age = float(batch["age"][b])
            scores.band.append(age_band_index(age, bands))
            ages_recorded.append(age)
            # a_hat of the image is the median over its valid patches.
            ages_hat.append(float(np.nanmedian(result["a_hat"][b])))
    return scores, ages_recorded, ages_hat


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    bands = band_list(cfg)
    assert args.from_sweep or args.checkpoint, "pass either --from-sweep or --checkpoint"
    if args.from_sweep:
        scores, ages_recorded, ages_hat = collect_from_sweep(args.from_sweep, log)
    else:
        scores, ages_recorded, ages_hat = collect_from_model(cfg, args.checkpoint, bands, log)

    assert scores.s_rec, "the sweep produced no valid patches"

    # Step 1: lambda on raw scores. Step 2 must not run before this.
    lambda_star, table = select_lambda(scores, list(cfg.inference.lambda_grid), len(bands))
    log.info("lambda* = %.2f (W = %.6f)", lambda_star, min(row["W"] for row in table))

    mae = float(np.mean(np.abs(np.asarray(ages_hat) - np.asarray(ages_recorded))))
    max_mae = float(cfg.abort_criteria.max_mae_ahat_years)
    fallback = mae > max_mae
    if fallback:
        # Pre-registered: drop the sweep and continue at lambda = 0, reported as such.
        log.warning(
            "MAE of a_hat is %.2f years, above the pre-registered %.1f. "
            "Falling back to lambda = 0.", mae, max_mae
        )
        lambda_star = 0.0

    # Step 2: statistics at the chosen lambda, never before.
    stats = band_statistics(
        scores,
        lambda_star,
        bands,
        min_band_count=int(cfg.min_band_count),
        enforce_min_count=bool(cfg.inference.enforce_min_band_count),
    )

    payload = {
        "lambda_star": lambda_star,
        "lambda_table": table,
        "mae_a_hat": mae,
        "abort_fallback_applied": fallback,
        "n_images": len(scores.s_rec),
        "bands": stats,
    }
    path = run.write_json("calibration.json", payload)
    log.info("wrote %s", path)
    for entry in stats:
        log.info(
            "band %-6s n_img %4d  mu %+.5f  sigma %.5f",
            entry["name"], entry["n_images"], entry["mu"], entry["sigma"],
        )
    print(f"CALIBRATION={path}")


if __name__ == "__main__":
    main()
