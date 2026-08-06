"""Turn an arbitrary uploaded radiograph into the canvas the model was trained on.

The API contract says the service owns preprocessing and the client sends an
original image. This is that code, and it has to match the Kaggle notebook that
produced `data/images_384/` exactly, because a model trained on one
normalization and served another is wrong in a way nothing downstream reveals.

The order is load-bearing:

    1. resize the long side to 384, preserving aspect ratio
    2. clip at the 1st and 99th percentile **of the resized image**
    3. scale to [0, 1]
    4. zero-pad symmetrically to 384 x 384

Percentiles come before padding. Computing them afterwards would let the added
zeros - about 43% of the canvas - drag the 1st percentile to zero and flatten
the contrast of every image by an amount that depends on its aspect ratio.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CANVAS = 384
PATCH = 16
CLIP_PERCENTILES = (1.0, 99.0)


class UnreadableImage(ValueError):
    """The upload could not be decoded, or is not a 2D grayscale-convertible image."""


def load_grayscale(data: bytes | str) -> np.ndarray:
    """Decode bytes or a path to a 2D float array, preserving bit depth."""
    try:
        source = Image.open(data if isinstance(data, str) else _as_stream(data))
        array = np.asarray(source)
    except Exception as error:  # noqa: BLE001 - any decode failure is the same 415
        raise UnreadableImage(str(error)) from error

    if array.ndim == 3:
        # RGB or RGBA upload: collapse to luminance rather than refusing, since a
        # PACS export re-saved as PNG is a normal thing for a client to send.
        array = array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise UnreadableImage(f"expected a 2D image, got shape {array.shape}")
    if array.size == 0:
        raise UnreadableImage("image is empty")
    return array.astype(np.float32)


def _as_stream(data: bytes):
    import io

    return io.BytesIO(data)


def preprocess(array: np.ndarray) -> dict:
    """Return the padded canvas, its valid patch mask, and the geometry used.

    Geometry is returned rather than discarded because the valid mask is derived
    from it, and because `valid_patch_fraction` in the API response is how a
    client learns that a badly cropped upload produced a score resting on very
    few patches.
    """
    height, width = array.shape
    scale = CANVAS / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = np.asarray(
        Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32
    )

    low, high = np.percentile(resized, CLIP_PERCENTILES)
    if high <= low:
        # A blank or single-valued image: clipping would divide by zero.
        high = low + 1.0
    scaled = np.clip((resized - low) / (high - low), 0.0, 1.0)

    pad_x = (CANVAS - new_w) // 2
    pad_y = (CANVAS - new_h) // 2
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = scaled

    return {
        "image": canvas,
        "valid_mask": valid_mask(pad_x, pad_y, new_w, new_h),
        "geometry": {
            "scale": float(scale),
            "new_w": int(new_w),
            "new_h": int(new_h),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
        },
    }


def valid_mask(pad_x: int, pad_y: int, new_w: int, new_h: int) -> np.ndarray:
    """The same whole-box rule the training data uses, indexed [j, i]."""
    from ..data.geometry import valid_mask_from_geometry

    return valid_mask_from_geometry(pad_x, pad_y, new_w, new_h, size=CANVAS, patch=PATCH)
