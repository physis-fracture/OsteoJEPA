"""Run figures: loss curves and the two monitors that can kill a run early.

Kept out of the training script so M5 can render the same panels from a saved
metrics.json without rerunning anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

LOSS_KEYS = ["loss_loss", "loss_pred", "loss_var", "loss_cov", "loss_margin"]
LOSS_LABELS = ["total", "L_pred", "L_var", "L_cov", "L_margin"]


def plot_training_curves(history: list[dict], path: Path) -> Path:
    """Loss terms, Var(z) minimum across dimensions, and median V(p) per epoch."""
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    for key, label in zip(LOSS_KEYS, LOSS_LABELS):
        values = [record.get(key, np.nan) for record in history]
        axes[0].plot(epochs, values, label=label, linewidth=1.4)
    axes[0].set_yscale("symlog")
    axes[0].set_xlabel("epoch"), axes[0].set_title("loss terms")
    axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3)

    # The minimum across dimensions, not the mean: a collapse confined to a few
    # dimensions never shows up in an aggregate.
    axes[1].plot(epochs, [r.get("var_z_min_dim", np.nan) for r in history], color="crimson")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch"), axes[1].set_title("Var(z), minimum over dimensions")
    axes[1].grid(alpha=0.3)

    # If this stays flat, the predictor is ignoring the age channel and every
    # Stage B quantity loses its basis.
    axes[2].plot(epochs, [r.get("v_p_median", np.nan) for r in history], color="teal")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("epoch"), axes[2].set_title("median V(p) = Var_a s(p, a)")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def plot_band_curves(history: list[dict], path: Path) -> Path:
    """Median V(p) per age band. A mechanism that only works mid-range shows here.

    High V in the middle of the range and low V at both ends would mean the
    effect depends on two-sided distractor availability rather than on anatomy.
    """
    epochs = [record["epoch"] for record in history]
    bands = sorted({name for record in history for name in record.get("v_p_by_band", {})})
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    for name in bands:
        values = [record.get("v_p_by_band", {}).get(name, np.nan) for record in history]
        axis.plot(epochs, values, label=name, linewidth=1.2)
    axis.set_yscale("log")
    axis.set_xlabel("epoch"), axis.set_ylabel("median V(p)")
    axis.set_title("age sensitivity per age band")
    axis.legend(fontsize=7, ncol=2), axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def plot_spatial_map(array: np.ndarray, path: Path, title: str, cmap: str = "magma") -> Path:
    """A 24x24 patch map, NaN drawn as blank.

    For V(p) this is the figure that separates a real mechanism from an empty
    offset: age sensitivity concentrated on growth plates, carpal bones and
    metaphyses is anatomy, while an even spread including soft tissue and
    background is a uniform age-dependent shift that contains none.
    """
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    masked = np.ma.masked_invalid(array)
    image = axis.imshow(masked, cmap=cmap, interpolation="nearest")
    axis.set_xticks([]), axis.set_yticks([])
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
