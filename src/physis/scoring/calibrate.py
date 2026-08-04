"""Calibration: lambda first, then (mu, sigma). The order is not interchangeable.

1. Choose lambda on **raw**, unnormalized scores over clean validation patches:

       W(lambda) = mean over age bands of [ q95(score) - median(score) ]
       lambda*   = argmin W(lambda)

   Developmental variants live in the upper tail of the normal distribution. If
   Delta genuinely explains them through some other age, that tail collapses
   toward the median. No fracture labels are involved, which is what keeps the
   pipeline annotation-free.

2. **Then** compute (mu, sigma) per age band at lambda*.

Computing (mu, sigma) first and sweeping lambda afterwards normalizes every
candidate by statistics belonging to a different lambda, and W(lambda) stops
meaning anything. Nothing in the output would look wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.geometry import age_band_index


@dataclass
class PatchScores:
    """Valid-patch scores of the clean validation images, kept per image.

    s_rec and delta are stored rather than a finished score, because lambda is
    not known until W(lambda) has been swept.
    """

    s_rec: list[np.ndarray]
    delta: list[np.ndarray]
    band: list[int]

    def score_at(self, lam: float) -> list[np.ndarray]:
        return [s - lam * d for s, d in zip(self.s_rec, self.delta)]


def _pool_by_band(values: list[np.ndarray], bands: list[int], n_bands: int) -> list[np.ndarray]:
    pooled: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    for array, band in zip(values, bands):
        pooled[band].append(array)
    return [np.concatenate(chunk) if chunk else np.zeros(0) for chunk in pooled]


def tail_width(scores: list[np.ndarray], bands: list[int], n_bands: int) -> float:
    """W(lambda), averaged over the bands that hold any patch."""
    widths = []
    for pooled in _pool_by_band(scores, bands, n_bands):
        if pooled.size == 0:
            continue
        widths.append(float(np.quantile(pooled, 0.95) - np.median(pooled)))
    assert widths, "no band held any patch; W(lambda) is undefined"
    return float(np.mean(widths))


def select_lambda(
    patch_scores: PatchScores, lambda_grid, n_bands: int
) -> tuple[float, list[dict]]:
    """Return lambda* and the full W(lambda) table, computed on raw scores."""
    table = []
    for lam in lambda_grid:
        lam = float(lam)
        width = tail_width(patch_scores.score_at(lam), patch_scores.band, n_bands)
        table.append({"lambda": lam, "W": width})
    best = min(table, key=lambda row: row["W"])
    return float(best["lambda"]), table


def band_statistics(
    patch_scores: PatchScores,
    lam: float,
    bands: list,
    *,
    min_band_count: int,
    enforce_min_count: bool = True,
) -> list[dict]:
    """(mu, sigma) per age band at the chosen lambda, plus the image counts.

    `min_band_count` is asserted rather than printed: a band estimated from a
    handful of images produces a plausible-looking sigma and a silently wrong
    normalization for every case that lands in it.
    """
    scores = patch_scores.score_at(lam)
    n_images = [0] * len(bands)
    for band in patch_scores.band:
        n_images[band] += 1

    stats = []
    for index, band in enumerate(bands):
        pooled = _pool_by_band(scores, patch_scores.band, len(bands))[index]
        if enforce_min_count:
            assert n_images[index] >= min_band_count, (
                f"age band {band['name']} has {n_images[index]} clean images, "
                f"below the required {min_band_count}"
            )
        mu = float(np.mean(pooled)) if pooled.size else float("nan")
        sigma = float(np.std(pooled)) if pooled.size else float("nan")
        stats.append(
            {
                "name": band["name"],
                "lo": band["lo"],
                "hi": band["hi"],
                "n_images": n_images[index],
                "n_patches": int(pooled.size),
                "mu": mu,
                # A degenerate sigma would divide the whole band by ~0.
                "sigma": sigma if sigma > 1e-8 else float("nan"),
            }
        )
    return stats


def stats_for_age(age: float, stats: list[dict], bands: list) -> tuple[float, float]:
    entry = stats[age_band_index(age, bands)]
    return float(entry["mu"]), float(entry["sigma"])
