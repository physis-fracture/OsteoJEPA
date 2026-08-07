"""Load the shipped artifacts and score a study. No FastAPI here.

Two artifacts ship: the classifier checkpoint and `classifier_calibration.json`.
The lambda and per-band (mu, sigma) the design called for belonged to OsteoJEPA
and no longer exist; see docs/EXPERIMENT_REVISION.md.

The service owns preprocessing, the padding mask, the age bands, and the
calibration. A client sends an original image and gets scores back, so none of
those rules has a second implementation to drift from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import (
    GENDER_UNKNOWN,
    GENDER_VOCAB,
    LATERALITY_UNKNOWN,
    LATERALITY_VOCAB,
    VIEW_UNKNOWN,
    VIEW_VOCAB,
)
from ..data.geometry import age_band_index, band_list
from ..models.classifier import build_classifier
from ..models.detector import load_detector, predict_boxes
from .calibration import percentile_for, to_probability
from .preprocess import UnreadableImage, load_grayscale, preprocess

AGE_MIN, AGE_MAX = 0.2, 19.0


class AgeRequired(ValueError):
    """Age is the one field with no `unknown` fallback: it selects the band."""


@dataclass
class ImageScore:
    image_id: str
    triage_score: float
    logit: float
    valid_patch_fraction: float
    boxes: list | None = None


class Scorer:
    def __init__(self, cfg, checkpoint: str, calibration: str, device: str = "cpu",
                 detector_checkpoint: str | None = None):
        self.cfg = cfg
        self.bands = band_list(cfg)
        self.device = torch.device(device)

        self.model = build_classifier(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.device).eval()

        # Optional, and skipped when `localize=False`. It costs about
        # a second per image on CPU against the classifier's 79 ms, so putting it
        # on the worklist path would make ranking a queue twelve times slower for
        # information the worklist is not allowed to show anyway.
        self.detector = (
            load_detector(cfg, detector_checkpoint, str(self.device))
            if detector_checkpoint
            else None
        )
        self.detector_threshold = float(cfg.detector.serve_threshold)

        self._checkpoint_path = checkpoint
        self._calibration_path = calibration
        self.calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
        self.temperature = float(self.calibration["temperature"])
        self.checkpoint_id = str(self.calibration.get("checkpoint", Path(checkpoint).stem))
        self.detector_id = (
            Path(detector_checkpoint).parent.parent.name if detector_checkpoint else None
        )

    # ---- metadata -----------------------------------------------------------

    def band_of(self, age_years: float) -> int:
        if age_years is None or not np.isfinite(age_years):
            raise AgeRequired("age_years is required")
        if not (AGE_MIN <= age_years <= AGE_MAX):
            raise AgeRequired(f"age_years must be within [{AGE_MIN}, {AGE_MAX}]")
        return age_band_index(float(age_years), self.bands)

    def _meta(self, age_years: float, sex, view, laterality) -> dict:
        # Missing categoricals map to the trained `unknown` embedding rather than
        # being rejected: an image with incomplete metadata is still scored.
        return {
            "age": torch.tensor([float(age_years)], dtype=torch.float32, device=self.device),
            "gender": torch.tensor(
                [GENDER_VOCAB.get(sex, GENDER_UNKNOWN)], device=self.device
            ),
            "view": torch.tensor([VIEW_VOCAB.get(view, VIEW_UNKNOWN)], device=self.device),
            "laterality": torch.tensor(
                [LATERALITY_VOCAB.get(laterality, LATERALITY_UNKNOWN)], device=self.device
            ),
        }

    # ---- scoring ------------------------------------------------------------

    @torch.no_grad()
    def score_image(self, data: bytes | str, *, image_id: str, age_years: float,
                    sex=None, view=None, laterality=None, localize: bool = False) -> ImageScore:
        array, full_scale = load_grayscale(data)
        prepared = preprocess(array, full_scale)

        image = torch.from_numpy(prepared["image"])[None, None].to(self.device)
        valid = torch.from_numpy(prepared["valid_mask"])[None].to(self.device)
        logit = float(
            self.model(image, valid, self._meta(age_years, sex, view, laterality)).item()
        )
        boxes = None
        if localize and self.detector is not None:
            canvas_boxes = predict_boxes(
                self.detector, image[0], self.device, self.detector_threshold
            )
            boxes = _to_upload_space(canvas_boxes, prepared["geometry"])

        return ImageScore(
            image_id=image_id,
            triage_score=to_probability(logit, self.temperature),
            logit=logit,
            valid_patch_fraction=float(prepared["valid_mask"].mean()),
            boxes=boxes,
        )

    def score_study(self, images: list[dict], *, study_id: str, age_years: float,
                    sex=None, localize: bool = False) -> dict:
        """Score every image, then aggregate with `max`.

        One suspicious projection is enough to raise a case, which is what the
        maximum encodes and why the study is the unit the worklist orders.
        """
        started = time.perf_counter()
        band = self.band_of(age_years)

        scored = [
            self.score_image(
                item["data"],
                image_id=item["image_id"],
                age_years=age_years,
                sex=sex,
                view=item.get("view"),
                laterality=item.get("laterality"),
                localize=localize,
            )
            for item in images
        ]
        assert scored, "a study needs at least one image"

        best = max(scored, key=lambda s: s.logit)
        # Study-level reference for a study-level query. Falling back to the
        # per-image one would rank a normal two-projection study far too high.
        entry = self.calibration["bands"][band]
        payload = {
            "study_id": study_id,
            "triage_score": best.triage_score,
            "priority_percentile": percentile_for(best.logit, entry),
            "age_band": self.bands[band]["name"],
            "images": [
                {
                    "image_id": s.image_id,
                    "triage_score": s.triage_score,
                    "valid_patch_fraction": s.valid_patch_fraction,
                }
                for s in scored
            ],
            "model": self.model_info(),
            "inference_time_ms": (time.perf_counter() - started) * 1000.0,
        }
        if localize and self.detector is not None:
            # Only when a detector is actually loaded. An empty list would mean
            # "looked and found nothing", which is a different statement from
            # "this deployment cannot localize" - and the API layer turns the
            # absent field into an explicit null with that reason attached.
            payload["localization"] = [
                {"image_id": s.image_id, "boxes": s.boxes or []} for s in scored
            ]
        return payload

    def model_info(self) -> dict:
        return {
            "checkpoint": self.checkpoint_id,
            "temperature": self.temperature,
            "calibration": self.calibration.get("calibration_id", "val"),
            "contract": "v1.1" if self.detector is not None else "v1",
            "detector": self.detector_id if self.detector is not None else None,
            # The age sweep and its lambda belonged to OsteoJEPA, which returned
            # a null result. The score is now a supervised classifier and the
            # response says so rather than carrying a field that means nothing.
            "scoring": "supervised_classifier",
        }


__all__ = ["Scorer", "ImageScore", "AgeRequired", "UnreadableImage"]


def _to_upload_space(boxes: list, geometry: dict) -> list:
    """Map canvas boxes back to the coordinates of the image the client sent.

    The detector sees the 384 canvas, but the client draws on the upload it
    holds, so the resize and padding this service applied have to be undone.

    Unless it applied none. An upload that was already a padded 384 canvas was
    not transformed here - its padding arrived inside the file - so its own
    coordinate space *is* the canvas and the inverse must not be applied.
    Subtracting a padding the client never saw put every box about 70 px off.
    """
    if geometry.get("already_preprocessed"):
        return [[round(v, 2) for v in box[:4]] + [round(box[4], 4)] for box in boxes]

    scale = float(geometry.get("scale", 1.0)) or 1.0
    pad_x = float(geometry.get("pad_x", 0))
    pad_y = float(geometry.get("pad_y", 0))
    return [
        [
            round((box[0] - pad_x) / scale, 2),
            round((box[1] - pad_y) / scale, 2),
            round((box[2] - pad_x) / scale, 2),
            round((box[3] - pad_y) / scale, 2),
            round(box[4], 4),
        ]
        for box in boxes
    ]
