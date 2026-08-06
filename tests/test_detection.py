"""The detection dataset, and the aliasing trap that only fires on CUDA.

`expand` returns a view whose channels share one buffer. Training on CPU never
notices, because `pin_memory` is off there; on a GPU the loader raises
"more than one element of the written-to tensor refers to a single memory
location" before the first step. These tests reach it without a GPU.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from physis.data.detection import DetectionDataset, collate

SIZE = 384


def make_cfg(images_dir):
    return OmegaConf.create({"images_dir": str(images_dir), "image": {"size": SIZE, "patch": 16}})


@pytest.fixture
def dataset(tmp_path):
    frame = []
    for index in range(3):
        stem = f"img{index}"
        array = (np.random.rand(SIZE, SIZE) * 60000).astype(np.uint16)
        Image.fromarray(array).save(tmp_path / f"{stem}.png")
        frame.append({"stem": stem, "n_fracture_box": index, "age": 10.0 + index})
    boxes = {
        "img1": np.array([[100.0, 120.0, 160.0, 170.0]]),
        "img2": np.array([[10.0, 20.0, 60.0, 80.0], [200.0, 210.0, 250.0, 260.0]]),
    }
    return DetectionDataset(pd.DataFrame(frame), boxes, make_cfg(tmp_path))


def test_image_is_three_real_channels_not_an_aliased_view(dataset):
    image, _ = dataset[0]
    assert image.shape == (3, SIZE, SIZE)
    assert image.is_contiguous()
    # An expanded view would hold one plane; three planes means three buffers.
    assert image.untyped_storage().nbytes() // 4 == 3 * SIZE * SIZE
    image[0, 0, 0] = 5.0
    assert float(image[1, 0, 0]) != 5.0, "channels alias the same memory"


def test_pin_memory_accepts_the_batch(dataset):
    """The exact operation that failed on the GPU, reachable without one."""
    images, _ = collate([dataset[i] for i in range(3)])
    for image in images:
        image.pin_memory() if torch.cuda.is_available() else image.clone()
        assert image.is_contiguous()


def test_images_without_boxes_are_kept_with_an_empty_target(dataset):
    _, target = dataset[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].numel() == 0
    assert target["boxes"].dtype == torch.float32


def test_boxes_pass_through_in_canvas_coordinates(dataset):
    _, target = dataset[2]
    assert target["boxes"].shape == (2, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([10.0, 20.0, 60.0, 80.0]))
    assert (target["labels"] == 1).all()


def test_collate_keeps_targets_ragged(dataset):
    images, targets = collate([dataset[i] for i in range(3)])
    assert isinstance(images, list) and isinstance(targets, list)
    assert [len(t["boxes"]) for t in targets] == [0, 1, 2]
