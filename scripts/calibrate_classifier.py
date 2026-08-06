"""Fit the temperature and the per-band reference distribution. No GPU.

Both are fitted on **validation** scores and never on test, for the same reason
the OsteoJEPA band statistics were: a calibration read off the fold it is
reported on is not a calibration.

    python scripts/calibrate_classifier.py --val-scores scores_val.csv \\
        --test-scores scores_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.geometry import band_list
from physis.eval.metrics import safe_auroc
from physis.serve.calibration import (
    expected_calibration_error,
    fit_temperature,
    reference_quantiles,
    to_probability,
)
from physis.utils.config import load_config
from physis.utils.run import setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="temperature and band reference distribution")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--val-scores", required=True)
    parser.add_argument("--test-scores", default=None)
    parser.add_argument("--checkpoint-id", default="clf_main")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=clf_calibration")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log
    bands = band_list(cfg)

    val = pd.read_csv(args.val_scores)
    assert "logit" in val.columns, (
        "this needs the logit column from score_classifier.py; the training run's "
        "sigmoid outputs have no resolution left near 1"
    )
    manifest = pd.read_csv(cfg.manifest).set_index("stem")
    val = val.join(manifest[["clean_strict"]], on="stem")

    temperature = fit_temperature(val["logit"].to_numpy(), val["label"].to_numpy())
    before = expected_calibration_error(val["score"].to_numpy(), val["label"].to_numpy())
    after_probs = np.array([to_probability(z, temperature) for z in val["logit"]])
    after = expected_calibration_error(after_probs, val["label"].to_numpy())
    auroc_saved, _ = safe_auroc(val["label"].to_numpy(), val["score"].to_numpy())
    auroc_logit, _ = safe_auroc(val["label"].to_numpy(), val["logit"].to_numpy())
    auroc_after, _ = safe_auroc(val["label"].to_numpy(), after_probs)

    log.info("temperature T = %.3f", temperature)
    log.info("ECE on val: %.4f -> %.4f", before, after)
    # Temperature scaling is monotone, so it cannot move AUROC. The comparison
    # has to be logit against temperature-scaled logit; comparing against the
    # *saved* probability instead measures something else entirely, because
    # float32 near 1 collapses distinct logits into ties and AUROC scores ties at
    # half credit. That gap is the cost of the lost precision, not an effect of
    # the temperature.
    log.info("AUROC on val: logit %.6f -> scaled %.6f (must be identical)",
             auroc_logit, auroc_after)
    assert abs(auroc_logit - auroc_after) < 1e-9, "temperature scaling changed the ranking"
    log.info(
        "AUROC from the saved probability was %.6f, %.6f below the logit: that is "
        "what float32 ties near 1 cost, and why scoring now keeps the logit",
        auroc_saved, auroc_logit - auroc_saved,
    )

    log.info(
        "usable range: %.1f%% of val probabilities in (0.01, 0.99) before, %.1f%% after",
        100 * ((val["score"] > 0.01) & (val["score"] < 0.99)).mean(),
        100 * ((after_probs > 0.01) & (after_probs < 0.99)).mean(),
    )

    clean = val[val["clean_strict"]]
    # Two references, because the percentile has to be asked at the level it was
    # built at. The worklist ranks studies and queries with the max over a
    # study's images; the maximum of ~1.8 draws is stochastically larger than one
    # draw, so asking a per-image reference put a quarter of entirely normal
    # studies above the 90th percentile.
    clean_studies = clean.groupby("study_id").agg(
        logit=("logit", "max"), band=("band", "first")
    )
    quantiles = reference_quantiles(
        clean_studies["logit"].to_numpy(), clean_studies["band"].to_numpy(), len(bands)
    )
    quantiles_image = reference_quantiles(
        clean["logit"].to_numpy(), clean["band"].to_numpy(), len(bands)
    )
    # The >= 50 rule in DATA.md is fixed and is stated per image, so it is
    # asserted against the per-image reference. The study-level reference has
    # about 1.8x fewer units by construction; its counts are reported rather than
    # held to a threshold nobody has fixed.
    for band, entry, image_entry in zip(bands, quantiles, quantiles_image):
        log.info(
            "  band %-6s reference: %4d normal studies, %4d normal images",
            band["name"], entry["n"], image_entry["n"],
        )
        assert image_entry["n"] >= int(cfg.min_band_count) or not bool(
            cfg.inference.enforce_min_band_count
        ), f"band {band['name']} has only {image_entry['n']} normal images"
        if entry["n"] < 30:
            log.warning(
                "  band %s rests on %d studies; its percentile is coarse",
                band["name"], entry["n"],
            )

    payload = {
        "checkpoint": args.checkpoint_id,
        "temperature": temperature,
        "ece_before": before,
        "ece_after": after,
        "auroc_val": auroc_after,
        "auroc_val_from_saved_probability": auroc_saved,
        "precision_cost_auroc": auroc_logit - auroc_saved,
        "bands": [{"name": b["name"], **q} for b, q in zip(bands, quantiles)],
        "bands_image": [{"name": b["name"], **q} for b, q in zip(bands, quantiles_image)],
    }

    if args.test_scores:
        test = pd.read_csv(args.test_scores)
        probs = np.array([to_probability(z, temperature) for z in test["logit"]])
        payload["ece_test_before"] = expected_calibration_error(
            test["score"].to_numpy(), test["label"].to_numpy()
        )
        payload["ece_test_after"] = expected_calibration_error(probs, test["label"].to_numpy())
        log.info(
            "ECE on test: %.4f -> %.4f", payload["ece_test_before"], payload["ece_test_after"]
        )

    path = run.write_json("classifier_calibration.json", payload)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
