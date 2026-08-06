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
    auroc_before, _ = safe_auroc(val["label"].to_numpy(), val["score"].to_numpy())
    auroc_after, _ = safe_auroc(val["label"].to_numpy(), after_probs)

    log.info("temperature T = %.3f", temperature)
    log.info("ECE on val: %.4f -> %.4f", before, after)
    # A monotone transform cannot move AUROC. If these differ, something other
    # than temperature scaling has happened.
    log.info("AUROC on val: %.4f -> %.4f (must be identical)", auroc_before, auroc_after)
    assert abs(auroc_before - auroc_after) < 1e-9, "temperature scaling changed the ranking"

    log.info(
        "usable range: %.1f%% of val probabilities in (0.01, 0.99) before, %.1f%% after",
        100 * ((val["score"] > 0.01) & (val["score"] < 0.99)).mean(),
        100 * ((after_probs > 0.01) & (after_probs < 0.99)).mean(),
    )

    clean = val[val["clean_strict"]]
    quantiles = reference_quantiles(
        clean["logit"].to_numpy(), clean["band"].to_numpy(), len(bands)
    )
    for band, entry in zip(bands, quantiles):
        log.info("  band %-6s reference from %4d normal images", band["name"], entry["n"])
        assert entry["n"] >= int(cfg.min_band_count) or not bool(
            cfg.inference.enforce_min_band_count
        ), f"band {band['name']} has only {entry['n']} normal images"

    payload = {
        "checkpoint": args.checkpoint_id,
        "temperature": temperature,
        "ece_before": before,
        "ece_after": after,
        "auroc_val": auroc_after,
        "bands": [{"name": b["name"], **q} for b, q in zip(bands, quantiles)],
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
