"""Inference: age sweep to surprise map to r_image to r_study.

Saves the raw sweep as a compressed archive, not only the finished scores. The
sweep is the expensive part of the pipeline and every downstream number - the
lambda ablation, AUROC per age band, the Delta distribution of the comparator
group, the recalibration budget in E3 - is arithmetic over s_rec, s_min and
a_hat. Keeping those means one GPU session produces everything, and the analysis
happens anywhere. At 576 patches per image the whole val and test folds come to
tens of megabytes.

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

from physis.data.dataset import EVAL_SUBSETS, PhysisDataset, build_eval_frame
from physis.data.geometry import age_band_index, band_list
from physis.data.patch_labels import boxes_by_stem, label_patches, load_boxes, patch_coverage
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
    parser.add_argument(
        "--calibration",
        default=None,
        help="calibration.json; omit to sweep without normalizing (see --no-normalize)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "produce the raw sweep only. s_rec, s_min and a_hat do not depend on "
            "lambda or on (mu, sigma), so this is what runs before calibration."
        ),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--subset", default="clean", choices=list(EVAL_SUBSETS))
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default="score")
    parser.add_argument("--no-arrays", action="store_true", help="skip the raw .npz")
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
    normalize = not args.no_normalize
    assert args.calibration or not normalize, (
        "--calibration is required unless --no-normalize is passed"
    )
    strict_bands = bool(cfg.inference.enforce_min_band_count)
    if normalize:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        lam = float(calibration["lambda_star"])
        stats = calibration["bands"]
        fallback_mu, fallback_sigma = global_stats(stats)
        log.info(
            "lambda* = %.2f (fallback applied: %s)", lam, calibration["abort_fallback_applied"]
        )
    else:
        # The archive is the deliverable here; lambda enters later.
        lam, stats, fallback_mu, fallback_sigma = 0.0, None, float("nan"), float("nan")
        log.info("raw sweep only: no normalization, no r_image")

    frame = build_eval_frame(cfg, args.split, args.subset)
    log.info("sweeping %d images (%s / %s)", len(frame), args.split, args.subset)
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(dataset, batch_size=int(cfg.inference.batch_size), shuffle=False)
    model = load_osteojepa(cfg, args.checkpoint, device)

    boxes = boxes_by_stem(load_boxes(cfg))
    thresholds = [float(t) for t in cfg.patch_labels.report_thresholds]
    margin = int(cfg.patch_labels.negative_margin_patches)

    maps_dir = run.dir / "surprise_maps"
    maps_dir.mkdir(exist_ok=True)

    collected: dict[str, list] = {
        "stem": [], "study_id": [], "age": [], "band": [], "clean": [],
        "s_rec": [], "s_min": [], "a_hat": [], "valid": [],
        **{f"label_{int(t * 100):03d}": [] for t in thresholds},
    }
    image_scores: dict[str, float] = {}
    stem_to_study: dict[str, str] = {}
    implicit_age_gap: list[float] = []
    saved = 0
    n_done = 0
    started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
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

            band = age_band_index(age, bands)
            stem_to_study[stem] = batch["study_id"][b]
            implicit_age_gap.append(abs(float(np.nanmedian(result["a_hat"][b])) - age))

            s_tilde = None
            if normalize:
                entry = stats[band]
                mu, sigma = float(entry["mu"]), float(entry["sigma"])
                if not np.isfinite(sigma):
                    assert not strict_bands, (
                        f"band {entry['name']} has no usable sigma while "
                        "inference.enforce_min_band_count is on"
                    )
                    mu, sigma = fallback_mu, fallback_sigma
                s_tilde = normalize_map(raw, mu, sigma)
                image_scores[stem] = r_image(s_tilde, 0.95)

            if not args.no_arrays:
                collected["stem"].append(stem)
                collected["study_id"].append(batch["study_id"][b])
                collected["age"].append(age)
                collected["band"].append(band)
                # Carried so calibration can be redone offline from the archive:
                # lambda selection and (mu, sigma) both run on clean images only.
                collected["clean"].append(bool(frame.iloc[int(batch["index"][b])]["clean_strict"]))
                collected["s_rec"].append(result["s_rec"][b].reshape(-1))
                collected["s_min"].append(result["s_min"][b].reshape(-1))
                collected["a_hat"].append(result["a_hat"][b].reshape(-1))
                collected["valid"].append(valid.reshape(-1))
                coverage = patch_coverage(
                    boxes.get(stem, np.zeros((0, 4))),
                    size=int(cfg.image.size), patch=int(cfg.image.patch),
                )
                for threshold in thresholds:
                    labels = label_patches(
                        coverage, valid,
                        positive_coverage=threshold, negative_margin_patches=margin,
                    )
                    collected[f"label_{int(threshold * 100):03d}"].append(labels.reshape(-1))

            n_done += 1
            if saved < N_MAPS_SAVED and s_tilde is not None:
                np.save(maps_dir / f"{stem}.npy", s_tilde)
                np.save(maps_dir / f"{stem}.a_hat.npy", result["a_hat"][b])
                saved += 1

        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            rate = n_done / (time.perf_counter() - started)
            log.info(
                "%d/%d images  %.2f img/s  eta %.1f min",
                n_done, len(frame), rate, (len(frame) - n_done) / max(rate, 1e-9) / 60,
            )

    elapsed = time.perf_counter() - started
    studies = aggregate_studies(stem_to_study, image_scores)
    ranked = sorted(studies.items(), key=lambda kv: kv[1], reverse=True)

    payload = {
        "split": args.split,
        "subset": args.subset,
        "normalized": normalize,
        "lambda_star": lam if normalize else None,
        "n_images": n_done,
        "n_studies": len(studies),
        "seconds_total": elapsed,
        # E3 reports latency end to end; this is its dominant component.
        "seconds_per_image": elapsed / max(n_done, 1),
        "mean_abs_implicit_age_gap": float(np.mean(implicit_age_gap)),
        "r_image": image_scores,
        "r_study": studies,
        "worklist_top10": [{"study_id": s, "r_study": v} for s, v in ranked[:10]],
    }
    path = run.write_json(f"scores_{args.split}_{args.subset}.json", payload)
    log.info(
        "swept %d images in %d studies (%.2f s/image)",
        n_done, len(studies), payload["seconds_per_image"],
    )

    if not args.no_arrays and collected["stem"]:
        archive = run.dir / f"sweep_{args.split}_{args.subset}.npz"
        np.savez_compressed(
            archive,
            stem=np.array(collected["stem"]),
            study_id=np.array(collected["study_id"]),
            age=np.array(collected["age"], dtype=np.float32),
            band=np.array(collected["band"], dtype=np.int16),
            clean=np.array(collected["clean"], dtype=bool),
            s_rec=np.stack(collected["s_rec"]).astype(np.float32),
            s_min=np.stack(collected["s_min"]).astype(np.float32),
            a_hat=np.stack(collected["a_hat"]).astype(np.float32),
            valid=np.stack(collected["valid"]),
            **{
                key: np.stack(values).astype(np.int8)
                for key, values in collected.items()
                if key.startswith("label_")
            },
            lambda_star=np.float32(lam),
        )
        log.info("wrote %s (%.1f MB)", archive, archive.stat().st_size / 1e6)

    log.info("wrote %s and %d surprise maps", path, saved)
    for row in payload["worklist_top10"][:5]:
        log.info("worklist %s r_study %.4f", row["study_id"], row["r_study"])
    print(f"SCORES={path}")


if __name__ == "__main__":
    main()
