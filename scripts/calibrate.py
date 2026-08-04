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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default="calibrate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    bands = band_list(cfg)

    frame = build_eval_frame(cfg, "val")
    log.info("clean validation images: %d", len(frame))
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(dataset, batch_size=int(cfg.inference.batch_size), shuffle=False)

    model = load_osteojepa(cfg, args.checkpoint, device)

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
