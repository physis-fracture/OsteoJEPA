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


DTYPE_MAX = {"uint8": 255.0, "uint16": 65535.0, "int32": 65535.0}


def load_grayscale(data: bytes | str) -> tuple[np.ndarray, float]:
    """Decode to a 2D float array, plus the full-scale value of its dtype.

    The scale is returned rather than discarded because an already-preprocessed
    canvas has to be divided by the *dtype* range, exactly as PhysisDataset
    does. Dividing by the image's own maximum instead rescales every upload by a
    different factor - contrast the model never saw in training - and the scores
    come out plausible and wrong.
    """
    try:
        source = Image.open(data if isinstance(data, str) else _as_stream(data))
        array = np.asarray(source)
    except Exception as error:  # noqa: BLE001 - any decode failure is the same 415
        # PIL's message for undecodable bytes embeds the repr of the stream,
        # which puts a memory address in an HTTP error body: noise for the
        # client, and internals it has no use for. The original is chained, so
        # the server traceback still has it.
        raise UnreadableImage("not a decodable image") from error

    if array.ndim == 3:
        # RGB or RGBA upload: collapse to luminance rather than refusing, since a
        # PACS export re-saved as PNG is a normal thing for a client to send.
        array = array[..., :3].mean(axis=-1)
    if array.ndim != 2:
        raise UnreadableImage(f"expected a 2D image, got shape {array.shape}")
    if array.size == 0:
        raise UnreadableImage("image is empty")
    full_scale = DTYPE_MAX.get(str(array.dtype), 1.0 if array.dtype.kind == "f" else 255.0)
    return array.astype(np.float32), float(full_scale)


def _as_stream(data: bytes):
    import io

    return io.BytesIO(data)


def detect_preprocessed(array: np.ndarray) -> dict | None:
    """Geometry of an image that has already been through this pipeline, or None.

    The dataset ships 384x384 canvases that are already resized, clipped and
    zero-padded. Running the full pipeline over one of those does real damage:
    the percentile clip would include the padding zeros, and - far worse -
    pad_x and pad_y would come out zero, so every one of the 576 patches would
    be marked valid and the padding would enter the pooled representation. That
    is the one rule the whole project protects.

    Detection is deliberately narrow: exactly the canvas size, a strictly
    positive content region, and a zero border consistent with symmetric
    padding. An original radiograph of some other size never matches.
    """
    if array.shape != (CANVAS, CANVAS):
        return None
    content = array > 0
    if not content.any() or content.all():
        return None

    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    pad_y, pad_x = int(rows[0]), int(cols[0])
    new_h = int(rows[-1] - rows[0] + 1)
    new_w = int(cols[-1] - cols[0] + 1)

    # The bounding box of non-zero pixels is never wider than the true content,
    # and can be narrower: the 1st-percentile clip floors the darkest pixels, so
    # an outer row or column of genuine anatomy can come out entirely zero. The
    # geometry is therefore used exactly as observed and never widened. Losing a
    # patch at the edge costs almost nothing against a pooled mean over ~300 of
    # them; admitting one padding patch breaks the rule the whole project keeps.
    if max(new_w, new_h) < CANVAS - 8:
        return None
    if abs((CANVAS - new_w) // 2 - pad_x) > 4 or abs((CANVAS - new_h) // 2 - pad_y) > 4:
        return None
    return {"scale": 1.0, "new_w": new_w, "new_h": new_h, "pad_x": pad_x, "pad_y": pad_y}


def preprocess(array: np.ndarray, full_scale: float = 65535.0) -> dict:
    """Return the padded canvas, its valid patch mask, and the geometry used.

    Geometry is returned rather than discarded because the valid mask is derived
    from it, and because `valid_patch_fraction` in the API response is how a
    client learns that a badly cropped upload produced a score resting on very
    few patches.
    """
    already = detect_preprocessed(array)
    if already is not None:
        # Divide by the dtype range, matching PhysisDataset exactly. Not by
        # percentiles - the clipping already happened - and not by this image's
        # own maximum, which would apply a different contrast to every upload.
        canvas = (array / full_scale).astype(np.float32)
        return {
            "image": canvas,
            "valid_mask": valid_mask(
                already["pad_x"], already["pad_y"], already["new_w"], already["new_h"]
            ),
            "geometry": {**already, "already_preprocessed": True},
        }

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
            "already_preprocessed": False,
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
