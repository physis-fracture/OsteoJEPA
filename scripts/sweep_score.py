"""Inference: age sweep to surprise map to r_image to r_study.

Also enforces the M0 acceptance criterion directly: every surprise map is
(24, 24) and holds NaN at exactly the padding positions, never anywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, build_eval_frame
from physis.data.geometry import age_band_index, band_list
from physis.models.osteojepa import load_osteojepa
from physis.scoring.age_sweep import score_map, sweep_batch
from physis.scoring.aggregate import aggregate_studies, normalize_map, r_image
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run

N_MAPS_SAVED = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="age sweep, surprise maps, triage scores")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default="score")
    return parser.parse_args()


def global_stats(stats: list[dict]) -> tuple[float, float]:
    """Patch-count-weighted mu and sigma across every band that holds patches."""
    weights = np.array([s["n_patches"] for s in stats], dtype=float)
    mus = np.array([s["mu"] for s in stats], dtype=float)
    sigmas = np.array([s["sigma"] for s in stats], dtype=float)
    usable = np.isfinite(mus) & np.isfinite(sigmas) & (weights > 0)
    assert usable.any(), "no age band carries usable statistics"
    weights = weights[usable] / weights[usable].sum()
    return float((weights * mus[usable]).sum()), float((weights * sigmas[usable]).sum())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    bands = band_list(cfg)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    lam = float(calibration["lambda_star"])
    stats = calibration["bands"]
    fallback_mu, fallback_sigma = global_stats(stats)
    strict_bands = bool(cfg.inference.enforce_min_band_count)
    log.info("lambda* = %.2f (fallback applied: %s)", lam, calibration["abort_fallback_applied"])

    frame = build_eval_frame(cfg, "val")
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(dataset, batch_size=int(cfg.inference.batch_size), shuffle=False)
    model = load_osteojepa(cfg, args.checkpoint, device)

    maps_dir = run.dir / "surprise_maps"
    maps_dir.mkdir(exist_ok=True)

    image_scores: dict[str, float] = {}
    stem_to_study: dict[str, str] = {}
    implicit_age_gap: list[float] = []
    saved = 0
    started = time.perf_counter()

    for batch in loader:
        result = sweep_batch(model, batch, cfg, device)
        for b in range(result["s_rec"].shape[0]):
            stem = batch["stem"][b]
            age = float(batch["age"][b])
            valid = result["valid"][b]

            raw = score_map(result["s_rec"][b], result["delta"][b], lam)

            # M0 acceptance: right shape, NaN at exactly the padding positions.
            assert raw.shape == valid.shape, f"surprise map shape {raw.shape} != {valid.shape}"
            assert np.array_equal(np.isnan(raw), ~valid), (
                f"{stem}: NaN positions do not match the padding mask"
            )

            entry = stats[age_band_index(age, bands)]
            mu, sigma = float(entry["mu"]), float(entry["sigma"])
            if not np.isfinite(sigma):
                assert not strict_bands, (
                    f"band {entry['name']} has no usable sigma while "
                    "inference.enforce_min_band_count is on"
                )
                mu, sigma = fallback_mu, fallback_sigma

            s_tilde = normalize_map(raw, mu, sigma)
            image_scores[stem] = r_image(s_tilde, 0.95)
            stem_to_study[stem] = batch["study_id"][b]
            implicit_age_gap.append(abs(float(np.nanmedian(result["a_hat"][b])) - age))

            if saved < N_MAPS_SAVED:
                np.save(maps_dir / f"{stem}.npy", s_tilde)
                np.save(maps_dir / f"{stem}.a_hat.npy", result["a_hat"][b])
                saved += 1

    elapsed = time.perf_counter() - started
    studies = aggregate_studies(stem_to_study, image_scores)
    ranked = sorted(studies.items(), key=lambda kv: kv[1], reverse=True)

    payload = {
        "lambda_star": lam,
        "n_images": len(image_scores),
        "n_studies": len(studies),
        "seconds_total": elapsed,
        "seconds_per_image": elapsed / max(len(image_scores), 1),
        "mean_abs_implicit_age_gap": float(np.mean(implicit_age_gap)),
        "r_image": image_scores,
        "r_study": studies,
        "worklist_top10": [{"study_id": s, "r_study": v} for s, v in ranked[:10]],
    }
    path = run.write_json("scores.json", payload)
    log.info("scored %d images in %d studies (%.2f s/image)", len(image_scores), len(studies), payload["seconds_per_image"])
    log.info("wrote %s and %d surprise maps", path, saved)
    for row in payload["worklist_top10"][:5]:
        log.info("worklist %s r_study %.4f", row["study_id"], row["r_study"])
    print(f"SCORES={path}")


if __name__ == "__main__":
    main()
