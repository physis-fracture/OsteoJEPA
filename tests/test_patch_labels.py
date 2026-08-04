"""Patch coverage labelling against a hand-computed box."""

import numpy as np

from physis.data.patch_labels import (
    LABEL_DISCARD,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    label_patches,
    patch_coverage,
)


def test_box_aligned_to_one_patch_is_fully_covered():
    # Patch (i=2, j=2) spans x in [32, 48) and y in [32, 48).
    coverage = patch_coverage(np.array([[32.0, 32.0, 48.0, 48.0]]))
    assert coverage[2, 2] == 1.0
    assert coverage.sum() == 1.0  # nothing else touched


def test_half_covered_patch_is_exactly_half():
    coverage = patch_coverage(np.array([[32.0, 32.0, 40.0, 48.0]]))
    assert np.isclose(coverage[2, 2], 0.5)


def test_quarter_covered_patch():
    # 8 x 8 of a 16 x 16 patch.
    coverage = patch_coverage(np.array([[32.0, 32.0, 40.0, 40.0]]))
    assert np.isclose(coverage[2, 2], 0.25)


def test_coverage_is_indexed_row_then_column():
    """A box wide in x and thin in y must cover columns, not rows."""
    coverage = patch_coverage(np.array([[32.0, 32.0, 80.0, 48.0]]))
    assert np.isclose(coverage[2, 2], 1.0)
    assert np.isclose(coverage[2, 4], 1.0)   # [j=2, i=4]
    assert np.isclose(coverage[4, 2], 0.0)


def test_labels_split_positive_ambiguous_and_negative():
    valid = np.ones((24, 24), dtype=bool)
    coverage = patch_coverage(np.array([[32.0, 32.0, 40.0, 48.0]]))  # 0.5 at (2, 2)
    labels = label_patches(coverage, valid, positive_coverage=0.50, negative_margin_patches=2)

    assert labels[2, 2] == LABEL_POSITIVE
    # Within two patches of the box: neither positive nor a primary negative.
    assert labels[3, 3] == LABEL_DISCARD
    assert labels[4, 4] == LABEL_DISCARD
    # Far away: a primary negative.
    assert labels[10, 10] == LABEL_NEGATIVE


def test_padding_patches_are_never_labelled():
    valid = np.ones((24, 24), dtype=bool)
    valid[:, :4] = False
    coverage = np.zeros((24, 24), dtype=np.float32)
    labels = label_patches(coverage, valid)
    assert (labels[:, :4] == LABEL_DISCARD).all()
    assert (labels[:, 4:] == LABEL_NEGATIVE).all()


def test_border_box_does_not_wrap_around_the_image():
    """Dilation must clip at the border, not wrap."""
    valid = np.ones((24, 24), dtype=bool)
    coverage = patch_coverage(np.array([[0.0, 0.0, 16.0, 16.0]]))  # patch (0, 0)
    labels = label_patches(coverage, valid, negative_margin_patches=2)
    assert labels[23, 23] == LABEL_NEGATIVE
    assert labels[0, 0] == LABEL_POSITIVE
