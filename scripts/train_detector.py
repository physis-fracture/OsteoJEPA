"""Fracture box detector: training and mAP, on the split this project fixed.

Fills three placeholders the paper already reserved:

* Section 3.2.3, the comparator detector it describes and does not yet have
* Section 4, the split-leakage figure in mAP - run this with
  `detector.split_mode=random_per_image` for the other half of that comparison
* Section 3.3, the radiologist's localization view

The paper says the detector is fine-tuned from the Stage A backbone. Stage A
returned a null result and cost a classifier 0.21 AUROC when used as an
initializer, so COCO weights are used instead and the substitution is stated.

    modal run --detach modal_app.py::train_detector --name det_main
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import assign_splits, load_manifest, subset_by_study
from physis.data.detection import DetectionDataset, build_frame, collate
from physis.data.geometry import age_band_index, band_list
from physis.data.patch_labels import boxes_by_stem, load_boxes
from physis.models.detector import build_detector
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fracture box detector")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def evaluate(model, loader, device, frame, bands, log) -> dict:
    """mAP@50 and mAP@50:95, overall and per age band.

    torchmetrics rather than a hand-rolled average precision: the number is
    compared against published mAP figures, and a subtle bug in the
    interpolation would be invisible and wrong in exactly the way this project
    keeps guarding against.
    """
    from torchmetrics.detection import MeanAveragePrecision

    # faster_coco_eval rather than the pycocotools default: same COCO
    # definition, pure wheels, and pycocotools needs a compiler on Windows.
    def metric():
        return MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
        )

    overall = metric()
    per_band = {index: metric() for index in range(len(bands))}
    band_boxes = {index: 0 for index in range(len(bands))}
    band_images = {index: 0 for index in range(len(bands))}
    model.eval()
    row_bands = [age_band_index(float(a), bands) for a in frame["age"]]

    # The eval loader is never shuffled, so a running counter maps each item back
    # to its manifest row and therefore to its age band. Reading the band off the
    # target dict does not work: `image_id` is stripped when the targets are
    # narrowed to what torchmetrics accepts.
    position = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            truth = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            overall.update(predictions, truth)
            for prediction, target in zip(predictions, truth):
                band = row_bands[position]
                per_band[band].update([prediction], [target])
                band_images[band] += 1
                band_boxes[band] += int(len(target["boxes"]))
                position += 1
    assert position == len(frame), (
        f"evaluated {position} images against a frame of {len(frame)}; "
        "the per-band mapping depends on the loader order matching the frame"
    )

    result = overall.compute()
    report = {
        "map_50": float(result["map_50"]),
        "map_50_95": float(result["map"]),
        "map_small": float(result["map_small"]),
    }
    log.info("mAP@50 %.4f | mAP@50:95 %.4f", report["map_50"], report["map_50_95"])

    bands_out = []
    for index, band_metric in per_band.items():
        # COCO returns -1 for "no ground truth of this class", which is not a
        # score of -1 and must not be printed as one. A band with no boxes has
        # no mAP to report, and saying so beats showing a number.
        if band_boxes[index] == 0:
            value = float("nan")
        else:
            value = float(band_metric.compute()["map_50"])
            if value < 0:
                value = float("nan")
        bands_out.append({
            "band": bands[index]["name"],
            "n_images": band_images[index],
            "n_boxes": band_boxes[index],
            "map_50": value,
        })
        log.info(
            "  band %-6s images %4d boxes %4d | mAP@50 %s",
            bands[index]["name"], band_images[index], band_boxes[index],
            "     n/a" if np.isnan(value) else f"{value:8.4f}",
        )
    report["by_age_band"] = bands_out
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    bands = band_list(cfg)
    det = cfg.detector

    manifest = assign_splits(load_manifest(cfg), str(det.split_mode), int(cfg.run.seed))
    log.info("split mode: %s | backbone init: %s", det.split_mode, det.init)
    boxes = boxes_by_stem(load_boxes(cfg))

    train_df = build_frame(manifest, "train")
    if int(cfg.data.subset_train) > 0:
        train_df = subset_by_study(train_df, int(cfg.data.subset_train), int(cfg.run.seed))
    with_boxes = int((train_df["n_fracture_box"] > 0).sum())
    log.info(
        "train %d images, %d with a box (%.1f%%); images without one are kept so the "
        "detector learns what a wrist with no fracture looks like",
        len(train_df), with_boxes, 100 * with_boxes / max(len(train_df), 1),
    )

    loader = DataLoader(
        DetectionDataset(train_df, boxes, cfg, augment=bool(cfg.data.augment),
                         seed=int(cfg.run.seed)),
        batch_size=int(det.batch_size),
        shuffle=True,
        num_workers=int(cfg.optim.num_workers),
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = build_detector(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=float(det.lr), momentum=0.9, weight_decay=float(det.weight_decay)
    )

    epochs = int(det.epochs)
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * epochs
    warmup = min(500, steps_per_epoch)
    step = 0

    for epoch in range(epochs):
        model.train()
        running, seen, started = 0.0, 0, time.perf_counter()
        for images, targets in loader:
            if step < warmup:
                lr = float(det.lr) * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(total_steps - warmup, 1)
                lr = 0.5 * float(det.lr) * (1 + math.cos(math.pi * min(progress, 1.0)))
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())

            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
            optimizer.step()

            running += float(loss.detach())
            seen += 1
            step += 1

        log.info(
            "epoch %d | loss %.4f | %.1f s | lr %.2e",
            epoch, running / max(seen, 1), time.perf_counter() - started, lr,
        )

    report = {"split_mode": str(det.split_mode), "init": str(det.init), "splits": {}}
    for split in ("val", "test"):
        frame = build_frame(manifest, split)
        if int(cfg.data.get(f"subset_{split}", 0) or 0) > 0:
            frame = subset_by_study(frame, int(cfg.data[f"subset_{split}"]), int(cfg.run.seed))
        eval_loader = DataLoader(
            DetectionDataset(frame, boxes, cfg, augment=False),
            batch_size=int(det.batch_size),
            shuffle=False,
            num_workers=int(cfg.optim.num_workers),
            collate_fn=collate,
        )
        log.info("--- %s ---", split)
        report["splits"][split] = evaluate(model, eval_loader, device, frame, bands, log)

    torch.save({"model": model.state_dict(), "config": dict(det)},
               run.checkpoints / "best.pt")
    run.write_json("detector_metrics.json", report)
    log.info("wrote %s", run.dir / "detector_metrics.json")
    print(f"RUN_DIR={run.dir}")


if __name__ == "__main__":
    main()
