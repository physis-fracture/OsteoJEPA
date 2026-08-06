"""Patch geometry and age bands.

Index convention, fixed here once and relied on everywhere else
-----------------------------------------------------------------
`i` indexes the patch column (x axis), `j` indexes the patch row (y axis). The
manifest geometry columns and `fracture_boxes.csv` both use that convention
(`patch_i*` are derived from x, `patch_j*` from y).

Arrays in this codebase are stored image-style as ``[row, col] == [j, i]``, so
``mask[j, i]`` is the patch at column i and row j. Flattening row-major then
gives token index ``j * grid + i``, which is exactly the order a ViT patch
embedding produces. Getting this backwards is invisible in every summary
statistic and obvious only in the M1 overlay figure.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def valid_mask_from_geometry(
    pad_x: float, pad_y: float, new_w: float, new_h: float, size: int = 384, patch: int = 16
) -> np.ndarray:
    """Return the (grid, grid) bool mask of non-padding patches, indexed [j, i].

    A patch counts as valid only when its whole 16x16 box lies inside the
    content area, per SPEC section 3:

        16*i     >= pad_x  and  16*(i+1) <= pad_x + new_w
        16*j     >= pad_y  and  16*(j+1) <= pad_y + new_h

    Partly covered edge patches are therefore dropped. Mean valid fraction under
    this rule is 0.531 over the dataset, against a 0.569 mean *pixel* content
    fraction (= 1 - 0.431 padding). The two numbers measure different things and
    both are correct; the 0.57 quoted in SPEC is the pixel one.
    """
    grid = size // patch
    idx = np.arange(grid)
    valid_i = (patch * idx >= pad_x) & (patch * (idx + 1) <= pad_x + new_w)
    valid_j = (patch * idx >= pad_y) & (patch * (idx + 1) <= pad_y + new_h)
    return np.outer(valid_j, valid_i)  # [j, i]


def valid_mask_from_row(row, size: int = 384, patch: int = 16) -> np.ndarray:
    """`valid_mask_from_geometry` applied to one manifest row."""
    return valid_mask_from_geometry(
        row["pad_x"], row["pad_y"], row["new_w"], row["new_h"], size=size, patch=patch
    )


def age_band_index(age: float, bands: list) -> int:
    """Index of the reporting band holding `age`.

    Bands are contiguous and identified by their lower edge, so a value landing
    in the gap a literal `lo <= age <= hi` test would leave uncovered (6.9995,
    say) still falls in the band below it. Boundary behaviour: 6.999 -> "0-6",
    7.0 -> "7-8", 16.999 -> "16", 17.0 -> "17-19".
    """
    los = [float(b["lo"]) for b in bands]
    if age < los[0]:
        raise ValueError(f"age {age} below the first band edge {los[0]}")
    hi_last = float(bands[-1]["hi"])
    if age > hi_last:
        raise ValueError(f"age {age} above the last band edge {hi_last}")
    return int(np.searchsorted(np.asarray(los), age, side="right") - 1)


def age_band_name(age: float, bands: list) -> str:
    return str(bands[age_band_index(age, bands)]["name"])


def band_list(cfg: DictConfig) -> list:
    """Reporting bands as plain dicts, from configs/data.yaml."""
    return [{"name": b["name"], "lo": float(b["lo"]), "hi": float(b["hi"])} for b in cfg.age_bands]


def erode_valid_mask(mask: np.ndarray, rings: int) -> np.ndarray:
    """Drop `rings` patches inward from the content boundary.

    Excluding padding patches does not close the aspect-ratio leak the SPEC
    warns about. The leak is not in the padding itself but in the valid patches
    beside it: those carry the outer edge of the limb, and the limb's width in
    the frame is body size, which is age. A predictor can read age off that
    boundary without ever looking at bone, and the margin loss is satisfied
    either way.

    Eroding costs valid area - the mean valid fraction falls from 0.531 to about
    0.45 at one ring - so it is off by default and turned on deliberately.
    """
    if rings <= 0:
        return mask
    out = mask.copy()
    for _ in range(rings):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        shrunk[0, :] = False
        shrunk[-1, :] = False
        shrunk[:, 0] = False
        shrunk[:, -1] = False
        out = shrunk
    return out
