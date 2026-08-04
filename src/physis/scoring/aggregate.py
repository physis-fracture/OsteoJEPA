"""Surprise map to r_image to r_study.

    s_tilde(p) = (score(p) - mu_band) / sigma_band
    r_image    = q95(s_tilde)            over valid patches only
    r_study    = max over images in the study of r_image

q95 rather than the maximum: letter markers are deliberately not filtered out of
the dataset, so a marker in an unusual position produces one high-residual patch,
and a maximum would let it move the whole triage score. The maximum is used at
study level instead, where one suspicious projection is genuinely enough to
raise a case.
"""

from __future__ import annotations

import numpy as np


def normalize_map(score: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Per-band normalization. NaN at padding survives untouched."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid sigma for normalization: {sigma}")
    return (score - mu) / sigma


def r_image(s_tilde: np.ndarray, quantile: float = 0.95) -> float:
    """q95 over valid patches. Padding patches are NaN and are excluded."""
    values = s_tilde[~np.isnan(s_tilde)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, quantile))


def r_study(image_scores: list[float]) -> float:
    finite = [v for v in image_scores if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.max(finite))


def aggregate_studies(stems_to_study: dict[str, str], scores: dict[str, float]) -> dict[str, float]:
    """Group per-image scores into per-study scores."""
    grouped: dict[str, list[float]] = {}
    for stem, value in scores.items():
        grouped.setdefault(stems_to_study[stem], []).append(value)
    return {study: r_study(values) for study, values in grouped.items()}
