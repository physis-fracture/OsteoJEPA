"""The 4-group inference partition, and the padding rule for training masks."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from physis.data.geometry import valid_mask_from_geometry
from physis.data.masking import (
    mask_to_indices,
    partition_interleaved,
    sample_context_and_targets,
)

GEOMETRIES = [
    (61, 0, 261, 384),   # portrait, padded on x
    (0, 61, 384, 261),   # landscape, padded on y
    (8, 8, 368, 368),    # padded on both
    (0, 0, 384, 384),    # no padding at all
]

CFG = OmegaConf.create(
    {
        "masking": {
            "context_ratio": 0.60,
            "n_target_blocks": 4,
            "target_block_ratio": 0.15,
            "allow_overlap": False,
            "exclude_padding": True,
        }
    }
)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_partition_covers_every_valid_patch_exactly_once(geometry):
    valid = valid_mask_from_geometry(*geometry)
    groups = partition_interleaved(valid, k=4)

    assert len(groups) == 4
    counts = sum(group.astype(int) for group in groups)
    assert (counts[valid] == 1).all(), "a valid patch is in more or less than one group"
    assert (counts[~valid] == 0).all(), "a padding patch entered a group"


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_partition_is_spread_across_the_image(geometry):
    """Interleaved, not blocked: each group must touch most rows and columns."""
    valid = valid_mask_from_geometry(*geometry)
    for group in partition_interleaved(valid, k=4):
        rows = group.any(axis=1).sum()
        cols = group.any(axis=0).sum()
        assert rows >= valid.any(axis=1).sum() // 2
        assert cols >= valid.any(axis=0).sum() // 2


def test_partition_requires_a_perfect_square():
    valid = np.ones((24, 24), dtype=bool)
    with pytest.raises(AssertionError):
        partition_interleaved(valid, k=3)


def test_partition_is_deterministic():
    valid = valid_mask_from_geometry(61, 0, 261, 384)
    first = [g.copy() for g in partition_interleaved(valid, k=4)]
    second = partition_interleaved(valid, k=4)
    assert all((a == b).all() for a, b in zip(first, second))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_sampled_blocks_never_touch_padding(geometry):
    valid = valid_mask_from_geometry(*geometry)
    rng = np.random.default_rng(0)
    for _ in range(20):
        context, targets = sample_context_and_targets(valid, CFG, rng)
        assert context.any()
        assert not (context & ~valid).any()
        for block in targets:
            assert block.any()
            assert not (block & ~valid).any()
            assert not (block & context).any(), "context and target overlap"


def test_mask_to_indices_is_row_major():
    mask = np.zeros((24, 24), dtype=bool)
    mask[2, 3] = True  # row j=2, column i=3
    assert mask_to_indices(mask).tolist() == [2 * 24 + 3]
