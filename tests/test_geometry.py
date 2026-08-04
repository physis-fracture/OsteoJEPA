"""Unit tests for the two things in geometry that fail silently.

CONVENTIONS asks for exactly four unit tests. Two of them live here:
valid_mask derivation against known geometry values, and age-band assignment at
boundary values.
"""

import numpy as np
import pytest

from physis.data.geometry import age_band_index, age_band_name, valid_mask_from_geometry

# Reporting bands from configs/data.yaml, repeated here so the test fails when
# the config changes rather than following it.
BANDS = [
    {"name": "0-6", "lo": 0.0, "hi": 6.999},
    {"name": "7-8", "lo": 7.0, "hi": 8.999},
    {"name": "9", "lo": 9.0, "hi": 9.999},
    {"name": "10", "lo": 10.0, "hi": 10.999},
    {"name": "11", "lo": 11.0, "hi": 11.999},
    {"name": "12", "lo": 12.0, "hi": 12.999},
    {"name": "13", "lo": 13.0, "hi": 13.999},
    {"name": "14", "lo": 14.0, "hi": 14.999},
    {"name": "15", "lo": 15.0, "hi": 15.999},
    {"name": "16", "lo": 16.0, "hi": 16.999},
    {"name": "17-19", "lo": 17.0, "hi": 19.001},
]


def test_valid_mask_matches_hand_computed_geometry():
    # Manifest row 0001_1297860395_01_WRI-L1_M014: portrait image, padded on x.
    # 16*i >= 61        -> i >= 4
    # 16*(i+1) <= 322   -> i <= 19          => 16 valid columns
    # y is unpadded and new_h == 384        => all 24 rows valid
    mask = valid_mask_from_geometry(pad_x=61, pad_y=0, new_w=261, new_h=384)
    assert mask.shape == (24, 24)
    assert mask.sum() == 16 * 24
    assert mask[:, 4].all() and mask[:, 19].all()
    assert not mask[:, 3].any() and not mask[:, 20].any()


def test_valid_mask_is_indexed_row_then_column():
    """mask[j, i]: padding on x must blank columns, not rows."""
    mask = valid_mask_from_geometry(pad_x=69, pad_y=0, new_w=246, new_h=384)
    assert mask.sum() == 14 * 24
    assert mask.all(axis=0).sum() == 14  # 14 fully valid columns
    assert mask.all(axis=1).sum() == 0   # no fully valid row: some columns are padding
    assert mask.any(axis=1).sum() == 24  # every row still holds content


def test_valid_mask_excludes_partly_covered_edge_patches():
    # Content starts at x = 8, i.e. halfway into patch column 0, which is
    # therefore not valid under the whole-box rule.
    mask = valid_mask_from_geometry(pad_x=8, pad_y=8, new_w=368, new_h=368)
    assert not mask[0, :].any() and not mask[:, 0].any()
    assert mask[1:23, 1:23].all()


def test_fully_unpadded_image_is_entirely_valid():
    mask = valid_mask_from_geometry(pad_x=0, pad_y=0, new_w=384, new_h=384)
    assert mask.all()


@pytest.mark.parametrize(
    "age,expected",
    [(6.999, "0-6"), (7.0, "7-8"), (16.999, "16"), (17.0, "17-19"),
     (0.2, "0-6"), (9.0, "9"), (19.0, "17-19")],
)
def test_age_band_boundaries(age, expected):
    assert age_band_name(age, BANDS) == expected


def test_age_band_covers_the_gap_between_hi_and_the_next_lo():
    """6.9995 sits between one band's hi and the next band's lo."""
    assert age_band_name(6.9995, BANDS) == "0-6"


def test_age_outside_the_range_raises():
    with pytest.raises(ValueError):
        age_band_index(19.5, BANDS)
