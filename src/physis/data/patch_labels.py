"""Patch labels from fracture boxes.

Area coverage, not IoU: fracture boxes have a median area of 5.3 patches, so IoU
against a 16x16 patch is always small and always uninformative.

Not used by the M0 skeleton, which never evaluates. It exists here because the
coverage rule is one of the four things CONVENTIONS asks to unit test, and
because M5 will need it unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
LABEL_DISCARD = -1


def patch_coverage(
    boxes: np.ndarray | pd.DataFrame, size: int = 384, patch: int = 16
) -> np.ndarray:
    """Fraction of each patch covered by a fracture box, indexed [j, i].

    Coverage is the maximum over boxes, not the union: DATA.md defines a
    positive as a patch "at least 50% covered by a fracture box", singular. With
    a median box of 5.3 patches, overlapping boxes are rare enough that the two
    readings almost never differ.
    """
    grid = size // patch
    coverage = np.zeros((grid, grid), dtype=np.float32)
    if isinstance(boxes, pd.DataFrame):
        boxes = boxes[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return coverage

    edges = np.arange(grid + 1) * patch
    lo, hi = edges[:-1], edges[1:]
    for x0, y0, x1, y1 in boxes:
        overlap_i = np.clip(np.minimum(hi, x1) - np.maximum(lo, x0), 0, None)  # per column
        overlap_j = np.clip(np.minimum(hi, y1) - np.maximum(lo, y0), 0, None)  # per row
        area = np.outer(overlap_j, overlap_i) / float(patch * patch)  # [j, i]
        coverage = np.maximum(coverage, area.astype(np.float32))
    return coverage


def label_patches(
    coverage: np.ndarray,
    valid: np.ndarray,
    *,
    positive_coverage: float = 0.50,
    negative_margin_patches: int = 2,
) -> np.ndarray:
    """Positive / primary-negative / discarded labels, indexed [j, i].

    positive  coverage >= threshold
    discarded 0 < coverage < threshold (a mix of normal bone and fracture), and
              any patch within `negative_margin_patches` of a covered patch
    negative  coverage == 0 and far enough from every box
    Invalid (padding) patches are always discarded.
    """
    labels = np.full(coverage.shape, LABEL_DISCARD, dtype=np.int8)
    positive = coverage >= positive_coverage
    touched = coverage > 0

    near = _dilate(touched, negative_margin_patches)
    labels[~near & valid] = LABEL_NEGATIVE
    labels[positive & valid] = LABEL_POSITIVE
    labels[~valid] = LABEL_DISCARD
    return labels


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev dilation by `radius` patches, without a SciPy dependency.

    Shifts are clipped at the border rather than wrapped: np.roll would let a box
    on the left edge mark patches on the right edge as "near a box".
    """
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    out = mask.copy()
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            src_j = slice(max(0, -dj), height - max(0, dj))
            dst_j = slice(max(0, dj), height - max(0, -dj))
            src_i = slice(max(0, -di), width - max(0, di))
            dst_i = slice(max(0, di), width - max(0, -di))
            out[dst_j, dst_i] |= mask[src_j, src_i]
    return out


def boxes_for_stem(boxes_df: pd.DataFrame, stem: str) -> pd.DataFrame:
    return boxes_df[boxes_df["stem"] == stem]


def load_boxes(cfg) -> pd.DataFrame:
    """Load `fracture_boxes.csv`.

    Coordinates are already in 384-space, so no transformation happens at
    training time and there is no second place for the geometry to be applied
    wrongly.
    """
    frame = pd.read_csv(cfg.fracture_boxes)
    required = ["stem", "x0", "y0", "x1", "y1"]
    missing = [c for c in required if c not in frame.columns]
    assert not missing, f"fracture_boxes.csv is missing columns: {missing}"
    size = int(cfg.image.size)
    assert frame[["x0", "y0", "x1", "y1"]].to_numpy().max() <= size + 1e-6, (
        "box coordinates fall outside the 384 canvas; they are expected in 384-space"
    )
    assert (frame["x1"] > frame["x0"]).all() and (frame["y1"] > frame["y0"]).all()
    return frame


def boxes_by_stem(boxes_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Group boxes into {stem: (n, 4) array} once, instead of filtering per image."""
    grouped: dict[str, np.ndarray] = {}
    columns = ["x0", "y0", "x1", "y1"]
    for stem, chunk in boxes_df.groupby("stem", sort=False):
        grouped[str(stem)] = chunk[columns].to_numpy(dtype=np.float64)
    return grouped
