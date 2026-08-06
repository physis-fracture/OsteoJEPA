"""Fracture boxes as a detection dataset.

Coordinates in `fracture_boxes.csv` are already in 384-space, so nothing is
transformed here and there is no second place for the geometry to be applied
wrongly.

Images **without** boxes are kept. A detector trained only on images that
contain a fracture never learns what a wrist without one looks like, and would
fire on every clean film it is shown. That matters more here than usual: the
product's negatives are exactly the clean set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

FRACTURE_CLASS = 1  # 0 is background, as torchvision expects


class DetectionDataset(Dataset):
    """Returns (image, target) pairs in the format torchvision detection wants.

    The image is a 3-channel float tensor in [0, 1]: the backbone ships with
    3-channel ImageNet weights, so the single grayscale channel is repeated
    rather than the first convolution being rebuilt. Precision is preserved -
    the 16-bit PNG is divided by 65535, never quantised to 8 bits, which is the
    step a YOLO pipeline would have forced.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        boxes: dict[str, np.ndarray],
        data_cfg: DictConfig,
        *,
        augment: bool = False,
        seed: int = 0,
    ):
        self.df = frame.reset_index(drop=True)
        self.boxes = boxes
        self.images_dir = Path(data_cfg.images_dir)
        self.size = int(data_cfg.image.size)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        stem = str(row["stem"])
        with Image.open(self.images_dir / f"{stem}.png") as handle:
            array = np.asarray(handle)
        assert array.dtype == np.uint16, f"expected a 16-bit PNG, got {array.dtype}"
        image = torch.from_numpy(array.astype(np.float32) / 65535.0)

        boxes = self.boxes.get(stem)
        if boxes is None or len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        boxes = torch.as_tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))

        if self.augment and len(boxes):
            image, boxes = self._flip_free_jitter(image, boxes, index)

        target = {
            "boxes": boxes,
            "labels": torch.full((len(boxes),), FRACTURE_CLASS, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        if image.ndim == 2:
            # `repeat`, not `expand`. Expand returns a view whose three channels
            # share one buffer, and pin_memory refuses to write into a tensor
            # where several elements alias the same address. That only fires on
            # CUDA, because pin_memory is off on CPU - so a CPU-only smoke test
            # cannot reach it.
            image = image.unsqueeze(0).repeat(3, 1, 1)
        return image.contiguous(), target

    def _flip_free_jitter(self, image, boxes, index):
        """Brightness and contrast only.

        No flip: horizontal mirroring changes laterality, and no geometric
        augmentation either, because a box that is not moved with its image is a
        wrong label rather than a noisy one. The classifier makes the same
        choice for the same reason.
        """
        rng = np.random.default_rng((self.seed * 7919 + index) % (2**32))
        gain = 1.0 + float(rng.uniform(-0.1, 0.1))
        bias = float(rng.uniform(-0.05, 0.05))
        content = image > 0
        out = image.clone()
        out[content] = (image[content] * gain + bias).clamp(0.0, 1.0)
        return out, boxes


def collate(batch):
    """Detection targets are ragged; torchvision takes lists, not stacked tensors."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_frame(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    return manifest[manifest["split"] == split].reset_index(drop=True)
