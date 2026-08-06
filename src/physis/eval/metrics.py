"""Image-level and study-level metrics, stratified by age band.

The stratification is the point, not an extra. The academic baselines on
GRAZPEDWRI-DX report one aggregate number across ages 0 to 19, which is exactly
the hidden stratification Oakden-Rayner et al. describe: a drop confined to one
age group disappears into the average. Every table here is reported per band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """AUROC and AUPRC, or NaN when one class is missing."""
    labels = np.asarray(labels).astype(int)
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def study_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image scores to per-study, `max` over the study's images.

    One suspicious projection is enough to raise a case, which is what the
    maximum encodes. `r_image` is already a quantile, so the maximum is not
    exposed to a single outlier patch.
    """
    return frame.groupby("study_id").agg(
        score=("score", "max"),
        label=("label", "max"),
        age=("age", "first"),
        band=("band", "first"),
    )


def evaluate(frame: pd.DataFrame, bands: list) -> dict:
    """Image-level and study-level metrics plus the per-band breakdown.

    `frame` needs columns: score, label, age, band, study_id.
    """
    image_auroc, image_auprc = safe_auroc(frame["label"].to_numpy(), frame["score"].to_numpy())
    studies = study_scores(frame)
    study_auroc, study_auprc = safe_auroc(
        studies["label"].to_numpy(), studies["score"].to_numpy()
    )

    per_band = []
    for index, band in enumerate(bands):
        subset = frame[frame["band"] == index]
        auroc, auprc = safe_auroc(subset["label"].to_numpy(), subset["score"].to_numpy())
        sub_studies = studies[studies["band"] == index]
        study_band_auroc, _ = safe_auroc(
            sub_studies["label"].to_numpy(), sub_studies["score"].to_numpy()
        )
        per_band.append(
            {
                "band": band["name"],
                "n_images": int(len(subset)),
                "n_positive": int(subset["label"].sum()),
                "auroc": auroc,
                "auprc": auprc,
                "study_auroc": study_band_auroc,
            }
        )

    return {
        "n_images": int(len(frame)),
        "n_studies": int(len(studies)),
        "positive_rate_images": float(frame["label"].mean()),
        "positive_rate_studies": float(studies["label"].mean()),
        "image_auroc": image_auroc,
        "image_auprc": image_auprc,
        "study_auroc": study_auroc,
        "study_auprc": study_auprc,
        "by_age_band": per_band,
    }


def band_percentiles(clean_scores: np.ndarray, clean_bands: np.ndarray, n_bands: int) -> list:
    """Per-band score distribution of normal images, for `priority_percentile`.

    This is the piece of the OsteoJEPA calibration that survives the pivot: a
    worklist number a radiologist can act on is "94th percentile for a
    14-year-old", not a raw score. It needs a reference distribution per band and
    nothing else, so it works for any scoring model.
    """
    out = []
    for index in range(n_bands):
        values = np.sort(clean_scores[clean_bands == index])
        out.append(
            {
                "band_index": index,
                "n": int(values.size),
                "quantiles": (
                    np.quantile(values, np.linspace(0, 1, 101)).tolist()
                    if values.size
                    else []
                ),
            }
        )
    return out
