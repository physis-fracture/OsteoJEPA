"""Turn a saved sweep plus a calibration into triage scores. No GPU, no model.

    s_tilde(p) = (score(p) - mu_band) / sigma_band
    r_image    = q95(s_tilde)      over valid patches only
    r_study    = max over the images of the study

Separated from the sweep because everything here is cheap and everything there
is not. Re-running with a different lambda, with recalibrated band statistics, or
over a subsample of the validation set is arithmetic over an archive that
already exists - which is what the E3 recalibration budget measures and what a
new hospital would actually do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.scoring.aggregate import aggregate_studies, r_image
from physis.utils.config import load_config
from physis.utils.run import setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scores from a saved sweep archive")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep", required=True, help="sweep_*.npz from sweep_score.py")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--lambda-override", type=float, default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default="score_offline")
    return parser.parse_args()


def band_lookup(stats: list[dict]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per-band mu and sigma, plus patch-count-weighted fallbacks."""
    mus = np.array([s["mu"] for s in stats], dtype=float)
    sigmas = np.array([s["sigma"] for s in stats], dtype=float)
    weights = np.array([s["n_patches"] for s in stats], dtype=float)
    usable = np.isfinite(mus) & np.isfinite(sigmas) & (weights > 0)
    assert usable.any(), "no age band carries usable statistics"
    share = weights[usable] / weights[usable].sum()
    return mus, sigmas, float((share * mus[usable]).sum()), float((share * sigmas[usable]).sum())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    archive = np.load(args.sweep, allow_pickle=False)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    lam = float(args.lambda_override if args.lambda_override is not None else calibration["lambda_star"])
    mus, sigmas, fallback_mu, fallback_sigma = band_lookup(calibration["bands"])
    strict = bool(cfg.inference.enforce_min_band_count)
    log.info("%d images, lambda = %.2f", len(archive["stem"]), lam)

    scores: dict[str, float] = {}
    stem_to_study: dict[str, str] = {}
    for index, stem in enumerate(archive["stem"]):
        valid = archive["valid"][index]
        if not valid.any():
            continue
        s_rec = archive["s_rec"][index]
        delta = s_rec - archive["s_min"][index]
        raw = s_rec - lam * delta

        band = int(archive["band"][index])
        mu, sigma = mus[band], sigmas[band]
        if not np.isfinite(sigma):
            assert not strict, (
                f"band {calibration['bands'][band]['name']} has no usable sigma "
                "while inference.enforce_min_band_count is on"
            )
            mu, sigma = fallback_mu, fallback_sigma

        s_tilde = np.where(valid, (raw - mu) / sigma, np.nan)
        scores[str(stem)] = r_image(s_tilde, 0.95)
        stem_to_study[str(stem)] = str(archive["study_id"][index])

    studies = aggregate_studies(stem_to_study, scores)
    ranked = sorted(studies.items(), key=lambda kv: kv[1], reverse=True)
    payload = {
        "sweep": str(args.sweep),
        "lambda": lam,
        "n_images": len(scores),
        "n_studies": len(studies),
        "r_image": scores,
        "r_study": studies,
        "worklist_top10": [{"study_id": s, "r_study": v} for s, v in ranked[:10]],
    }
    path = run.write_json("scores.json", payload)
    log.info("scored %d images in %d studies -> %s", len(scores), len(studies), path)
    print(f"SCORES={path}")


if __name__ == "__main__":
    main()
