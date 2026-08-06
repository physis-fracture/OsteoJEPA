"""E4 — is the score reading fractures, or reading their treatment?

Zech et al. documented a pneumothorax CNN that had learned to detect chest
tubes, a device fitted *after* the pneumothorax was treated. Cast appears on
28.7% of the test fold here and metal on 3.8%, and the paper cites that failure
and then defers the test to its limitations section. This runs it.

    E4a  cast and metal as a shortcut
    E4b  the 773 AO-only images: a known fracture with no box, because it was
         not clearly visualized in that projection

No GPU and no model: everything reads `scores_*.csv` and the manifest.

    python scripts/eval_confounders.py --scores artifacts/clf/clf_main/scores_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import mannwhitneyu

from physis.data.geometry import band_list
from physis.eval.metrics import safe_auroc
from physis.utils.config import load_config
from physis.utils.run import setup_run

CONFOUNDERS = {"cast": "tag_cast", "metal": "lbl_metal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 confounder and occult-fracture tests")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--scores", required=True, help="scores_test.csv from train_classifier")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def load_joined(cfg, scores_path: str) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)
    manifest = pd.read_csv(cfg.manifest).set_index("stem")
    columns = ["tag_cast", "lbl_metal", "tag_ao", "clean_strict", "n_fracture_box"]
    joined = scores.join(manifest[columns], on="stem")
    assert joined[columns].notna().all().all(), "some stems are missing from the manifest"
    return joined


def cast_shortcut(frame: pd.DataFrame, flag: str, log) -> dict:
    """Does performance rest on the confounder being visible?

    Two questions, and the second is the sharper one:

    1. Split the fold by the flag and compare AUROC. A large gap means the model
       leans on it where it is present.
    2. Among **fracture-negative** images only, do flagged images score higher?
       Those are cases where the device is present and there is no fracture to
       find, so any score difference is the model reacting to the device itself.
    """
    present = frame[frame[flag]]
    absent = frame[~frame[flag]]
    auroc_present, _ = safe_auroc(present["label"].to_numpy(), present["score"].to_numpy())
    auroc_absent, _ = safe_auroc(absent["label"].to_numpy(), absent["score"].to_numpy())

    negatives = frame[frame["label"] == 0]
    with_flag = negatives[negatives[flag]]["score"].to_numpy()
    without_flag = negatives[~negatives[flag]]["score"].to_numpy()
    if with_flag.size and without_flag.size:
        statistic, p_value = mannwhitneyu(with_flag, without_flag, alternative="greater")
        # Rank-biserial: the probability a flagged negative outscores an
        # unflagged one. 0.5 is indifference.
        effect = float(statistic / (with_flag.size * without_flag.size))
    else:
        p_value, effect = float("nan"), float("nan")

    log.info(
        "  %-6s present n=%4d AUROC %.4f | absent n=%4d AUROC %.4f | gap %+.4f",
        flag, len(present), auroc_present, len(absent), auroc_absent,
        auroc_present - auroc_absent,
    )
    log.info(
        "  %-6s among fracture-negatives: median score %.4f with vs %.4f without, "
        "P(flagged outscores unflagged) = %.3f, p = %.2e",
        flag,
        float(np.median(with_flag)) if with_flag.size else float("nan"),
        float(np.median(without_flag)) if without_flag.size else float("nan"),
        effect, p_value,
    )
    return {
        "n_present": int(len(present)),
        "n_absent": int(len(absent)),
        "auroc_present": auroc_present,
        "auroc_absent": auroc_absent,
        "auroc_gap": auroc_present - auroc_absent,
        "negatives_median_with": float(np.median(with_flag)) if with_flag.size else float("nan"),
        "negatives_median_without": (
            float(np.median(without_flag)) if without_flag.size else float("nan")
        ),
        "prob_flagged_outscores": effect,
        "p_value": float(p_value),
    }


def occult(frame: pd.DataFrame, log) -> dict:
    """E4b — are AO-only images scored above clean ones?

    These carry an AO classification and no fracture box: a fracture known to be
    present but not clearly visualized in that projection. They were excluded
    from the clean set for exactly that reason, and were never positives during
    training. Whether the score separates them from clean images asks whether it
    sees anything the boxes do not.
    """
    occult_rows = frame[(frame["tag_ao"]) & (frame["n_fracture_box"] == 0)]
    clean_rows = frame[frame["clean_strict"]]
    boxed = frame[frame["n_fracture_box"] > 0]

    labels = np.concatenate([np.ones(len(occult_rows)), np.zeros(len(clean_rows))])
    scores = np.concatenate([occult_rows["score"], clean_rows["score"]])
    auroc, _ = safe_auroc(labels, scores)

    log.info(
        "  occult n=%d median %.4f | clean n=%d median %.4f | boxed n=%d median %.4f",
        len(occult_rows), occult_rows["score"].median(),
        len(clean_rows), clean_rows["score"].median(),
        len(boxed), boxed["score"].median(),
    )
    log.info("  occult vs clean AUROC %.4f (0.5 means indistinguishable)", auroc)
    return {
        "n_occult": int(len(occult_rows)),
        "n_clean": int(len(clean_rows)),
        "median_occult": float(occult_rows["score"].median()),
        "median_clean": float(clean_rows["score"].median()),
        "median_boxed": float(boxed["score"].median()),
        "auroc_occult_vs_clean": auroc,
    }


def clean_only_performance(frame: pd.DataFrame, log) -> dict:
    """AUROC once every confounded image is removed from the negatives.

    The strictest reading: positives are boxed fractures, negatives are only the
    pretraining-clean set, which carries no cast, no metal, no AO classification
    and no indirect sign of injury. If performance holds here, it is not resting
    on treatment artefacts.
    """
    positives = frame[frame["n_fracture_box"] > 0]
    negatives = frame[frame["clean_strict"]]
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    scores = np.concatenate([positives["score"], negatives["score"]])
    auroc, auprc = safe_auroc(labels, scores)
    log.info(
        "  boxed fractures (n=%d) vs strictly clean (n=%d): AUROC %.4f AUPRC %.4f",
        len(positives), len(negatives), auroc, auprc,
    )
    return {
        "n_positive": int(len(positives)),
        "n_clean_negative": int(len(negatives)),
        "auroc": auroc,
        "auprc": auprc,
    }


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=e4_confounders")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    frame = load_joined(cfg, args.scores)
    bands = band_list(cfg)
    log.info(
        "%d images | %.1f%% positive | cast %.1f%% | metal %.1f%%",
        len(frame), 100 * frame["label"].mean(),
        100 * frame["tag_cast"].mean(), 100 * frame["lbl_metal"].mean(),
    )

    report = {"scores": str(args.scores), "n_images": int(len(frame))}

    log.info("--- E4a: cast and metal as a shortcut ---")
    report["confounders"] = {
        name: cast_shortcut(frame, column, log) for name, column in CONFOUNDERS.items()
    }

    log.info("--- E4a strict: confounded negatives removed ---")
    report["clean_only"] = clean_only_performance(frame, log)

    log.info("--- E4b: occult AO-only fractures ---")
    report["occult"] = occult(frame, log)

    # Per band, so a confounder concentrated in one age group cannot hide in the
    # aggregate the way overall performance does.
    log.info("--- cast rate by age band ---")
    by_band = []
    for index, band in enumerate(bands):
        subset = frame[frame["band"] == index]
        if subset.empty:
            continue
        by_band.append(
            {
                "band": band["name"],
                "n": int(len(subset)),
                "cast_rate": float(subset["tag_cast"].mean()),
                "positive_rate": float(subset["label"].mean()),
            }
        )
        log.info(
            "  band %-6s n %4d | cast %.3f | positive %.3f",
            band["name"], len(subset), subset["tag_cast"].mean(), subset["label"].mean(),
        )
    report["by_age_band"] = by_band

    run.write_json("e4_confounders.json", report)
    log.info("wrote %s", run.dir / "e4_confounders.json")


if __name__ == "__main__":
    main()
