"""Dataset reading and the augmentation that has to carry the mask with it."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from physis.data.dataset import PhysisDataset, content_pixel_mask
from physis.data.geometry import valid_mask_from_geometry

SIZE = 384
PAD_X, PAD_Y, NEW_W, NEW_H = 64, 0, 256, 384


def make_cfg(images_dir, rotate=0.0, translate=0.0, jitter=0.0):
    return OmegaConf.create(
        {
            "images_dir": str(images_dir),
            "image": {"size": SIZE, "patch": 16},
            "augment": {
                "hflip": False,
                "rotate_deg": rotate,
                "translate_frac": translate,
                "brightness_contrast": jitter,
            },
        }
    )


@pytest.fixture
def one_image(tmp_path):
    """A 16-bit PNG whose content region is bright and whose padding is zero."""
    array = np.zeros((SIZE, SIZE), dtype=np.uint16)
    array[PAD_Y : PAD_Y + NEW_H, PAD_X : PAD_X + NEW_W] = 40000
    Image.fromarray(array).save(tmp_path / "img0.png")  # uint16 -> mode I;16
    frame = pd.DataFrame(
        [
            {
                "stem": "img0", "patient_id": 1, "study_id": "s1", "age": 11.5,
                "gender": "M", "projection": 1, "laterality": "L",
                "pad_x": PAD_X, "pad_y": PAD_Y, "new_w": NEW_W, "new_h": NEW_H,
            }
        ]
    )
    return tmp_path, frame


def test_reads_sixteen_bit_and_scales_to_unit_range(one_image):
    tmp_path, frame = one_image
    dataset = PhysisDataset(frame, make_cfg(tmp_path))
    sample = dataset[0]
    image = sample["image"].numpy()[0]
    assert image.shape == (SIZE, SIZE)
    assert np.isclose(image.max(), 40000 / 65535.0, atol=1e-6)
    assert image[:, : PAD_X - 1].max() == 0.0


def test_unaugmented_mask_matches_the_geometry_rule(one_image):
    tmp_path, frame = one_image
    dataset = PhysisDataset(frame, make_cfg(tmp_path))
    mask = dataset[0]["valid_mask"].numpy()
    assert np.array_equal(mask, valid_mask_from_geometry(PAD_X, PAD_Y, NEW_W, NEW_H))


def test_identity_augmentation_leaves_the_mask_alone(one_image):
    tmp_path, frame = one_image
    dataset = PhysisDataset(frame, make_cfg(tmp_path), augment=True)
    mask = dataset[0]["valid_mask"].numpy()
    assert np.array_equal(mask, valid_mask_from_geometry(PAD_X, PAD_Y, NEW_W, NEW_H))


def test_rotation_moves_the_mask_and_never_admits_padding(one_image):
    """The mask must follow the transform, not stay pinned to the manifest."""
    tmp_path, frame = one_image
    dataset = PhysisDataset(frame, make_cfg(tmp_path, rotate=5.0, translate=0.02), augment=True)
    baseline = valid_mask_from_geometry(PAD_X, PAD_Y, NEW_W, NEW_H)

    moved = 0
    for index in range(12):
        dataset.seed = index
        sample = dataset[0]
        image = sample["image"].numpy()[0]
        mask = sample["valid_mask"].numpy()
        # Rotating a rectangle can only lose whole-box patches, never gain them.
        assert mask.sum() <= baseline.sum()
        moved += int(not np.array_equal(mask, baseline))
        # Every patch called valid must be entirely non-zero image content.
        blocks = image.reshape(24, 16, 24, 16).min(axis=(1, 3))
        assert (blocks[mask] > 0).all(), "a valid patch contains padding pixels"
    assert moved > 0, "rotation never changed the mask; the transform is not applied"


def test_padding_stays_exactly_zero_under_photometric_jitter(one_image):
    tmp_path, frame = one_image
    dataset = PhysisDataset(frame, make_cfg(tmp_path, jitter=0.1), augment=True)
    image = dataset[0]["image"].numpy()[0]
    outside = ~content_pixel_mask(PAD_X, PAD_Y, NEW_W, NEW_H, size=SIZE)
    assert image[outside].max() == 0.0


def test_hflip_is_rejected(one_image):
    tmp_path, frame = one_image
    cfg = make_cfg(tmp_path)
    cfg.augment.hflip = True
    with pytest.raises(AssertionError):
        PhysisDataset(frame, cfg)


def test_content_pixel_mask_is_indexed_row_then_column():
    mask = content_pixel_mask(PAD_X, PAD_Y, NEW_W, NEW_H, size=SIZE)
    assert mask[:, PAD_X - 1].sum() == 0
    assert mask[:, PAD_X].all()
    assert mask[0, PAD_X : PAD_X + NEW_W].all()
    assert mask[:, PAD_X + NEW_W :].sum() == 0
    assert mask.sum() == NEW_W * NEW_H
