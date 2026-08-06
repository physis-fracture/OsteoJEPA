"""Temperature scaling and the per-band reference distribution.

Two separate problems, and conflating them wastes effort on the wrong one.

**The displayed probability is meaningless.** 55.6% of test scores sit above
0.9999 and the top-20% flagged group shares 84 distinct values across 815
studies. A worklist cannot order that. Temperature scaling (Guo et al., 2017)
fits one scalar on the validation logits and spreads the outputs back into a
usable range.

**The percentile is a rank**, so temperature scaling cannot change it - the
transform is monotone. What makes the percentile hard is estimating the
reference distribution from few images at a new site, which is what the E3
recalibration budget measures and what temperature scaling has nothing to do
with.

Both are stored in one calibration file because the product needs both, but they
answer different questions.
"""

from __future__ import annotations

import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """One scalar T minimizing negative log-likelihood of sigmoid(logit / T).

    Ranking is untouched: dividing by a positive constant is monotone, so AUROC
    is identical before and after. Only the numbers a human reads change.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def nll(temperature: float) -> float:
        z = logits / temperature
        # log(1 + exp(-|z|)) form, so a large |z| cannot overflow.
        return float(np.mean(np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z)))))

    grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 400))
    losses = [nll(t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE, reported because an uncalibrated triage score cannot order a queue."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (probabilities > low) & (probabilities <= high)
        if not inside.any():
            continue
        error += inside.mean() * abs(labels[inside].mean() - probabilities[inside].mean())
    return float(error)


def reference_quantiles(logits: np.ndarray, bands: np.ndarray, n_bands: int) -> list:
    """Per-band logit distribution of **normal** images, for `priority_percentile`.

    Stored as logits rather than probabilities: the quantile grid keeps its
    resolution where the probabilities have none left.
    """
    grid = np.linspace(0.0, 1.0, 101)
    out = []
    for index in range(n_bands):
        values = logits[bands == index]
        out.append(
            {
                "band_index": int(index),
                "n": int(values.size),
                "quantiles": np.quantile(values, grid).tolist() if values.size else [],
            }
        )
    return out


def percentile_for(logit: float, band_entry: dict) -> float:
    """Where a case sits against normal images of its own age band, in [0, 1].

    The number a worklist should display. "94th percentile for a 14-year-old" is
    actionable; a raw score is not.
    """
    quantiles = band_entry.get("quantiles") or []
    if not quantiles:
        return float("nan")
    # searchsorted returns len(quantiles) for a value above every one of them,
    # which divides to 1.01 and puts an impossible percentile on the worklist.
    position = int(np.searchsorted(np.asarray(quantiles), logit))
    return float(min(position, len(quantiles) - 1) / (len(quantiles) - 1))


def to_probability(logit: float, temperature: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit / temperature)))
