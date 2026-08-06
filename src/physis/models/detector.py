"""Fracture box detector — Faster R-CNN from torchvision.

Not YOLO, and the reason is not performance. Ultralytics ships under AGPL-3.0
while this repository is MIT, and a copyleft dependency in a published,
distributed submission is a licensing conflict rather than a detail. Its
pipeline also decodes through OpenCV at 8 bits, which would throw away the
16-bit depth this project asserts on at every other boundary.

torchvision is BSD, already installed alongside torch, and is fed by our own
loader, so the detector sees exactly the images the classifier sees.

Reported as a comparator, not a contribution, per Section 3.2.3 of the paper.
The triage score stays with the calibrated classifier; the detector only fills
the radiologist's localization view.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 2  # background, fracture


def build_detector(cfg: DictConfig) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN with the head replaced for one class.

    `min_size` and `max_size` are pinned to the canvas so torchvision's internal
    transform leaves the geometry alone. Letting it rescale would move every box
    relative to the padding mask the rest of the project derives from the
    manifest.
    """
    detector = cfg.detector
    weights = "DEFAULT" if str(detector.init) == "coco" else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=int(cfg.image.size),
        max_size=int(cfg.image.size),
        box_score_thresh=float(detector.score_threshold),
        box_detections_per_img=int(detector.max_detections),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def load_detector(cfg: DictConfig, checkpoint: str, device: str = "cpu") -> torch.nn.Module:
    from omegaconf import OmegaConf

    eval_cfg = OmegaConf.merge(cfg, OmegaConf.create({"detector": {"init": "random"}}))
    model = build_detector(eval_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_boxes(model, image: torch.Tensor, device, score_threshold: float = 0.5) -> list:
    """Boxes above a threshold for one 384x384 grayscale image in [0, 1].

    Returned as [x0, y0, x1, y1, score] in canvas coordinates, which is the
    space the client draws in and the same space `fracture_boxes.csv` uses.
    """
    tensor = image if image.ndim == 3 else image.expand(3, -1, -1)
    output = model([tensor.to(device)])[0]
    keep = output["scores"] >= score_threshold
    boxes = output["boxes"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()
    return [
        [float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)]
        for b, s in zip(boxes, scores)
    ]
