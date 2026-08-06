"""Supervised fracture classifier: the score the product ranks by.

Also carries three of the revised experiments, because each is the same training
run with one thing changed:

    E2a  classifier.use_condition        does age help a model that works
    E2b  classifier.holdout_bands        trajectory, or memorized age points
    leak classifier.split_mode           what a random per-image split inflates by

Target is `n_fracture_box > 0`. Images carrying an AO classification but no box
are deliberately **not** positives: they are the occult set E4b asks about
separately, and folding them in would make that question unanswerable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest, subset_by_study
from physis.data.geometry import age_band_index, band_list
from physis.eval.metrics import evaluate
from physis.models.classifier import build_classifier, load_backbone
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="supervised fracture classifier")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def assign_splits(manifest: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    """Return a manifest whose `split` column follows the requested scheme.

    `random_per_image` reproduces the practice the published baselines use: with
    3.3 images per patient, it puts the same patient in train and test. The gap
    between the two modes is the number the paper reports as split leakage.
    """
    if mode == "grouped":
        return manifest
    assert mode == "random_per_image", f"unknown split mode {mode!r}"

    sizes = manifest["split"].value_counts(normalize=True)
    rng = np.random.default_rng(seed)
    draw = rng.permutation(len(manifest))
    out = manifest.copy()
    n_test = int(round(sizes.get("test", 0.2) * len(manifest)))
    n_val = int(round(sizes.get("val", 0.2) * len(manifest)))
    labels = np.array(["train"] * len(manifest), dtype=object)
    labels[draw[:n_test]] = "test"
    labels[draw[n_test : n_test + n_val]] = "val"
    out["split"] = labels
    return out


def make_frame(manifest: pd.DataFrame, split: str, bands: list) -> pd.DataFrame:
    frame = manifest[manifest["split"] == split].reset_index(drop=True)
    frame = frame.assign(
        label=(frame["n_fracture_box"] > 0).astype(int),
        band=[age_band_index(float(a), bands) for a in frame["age"]],
    )
    return frame


@torch.no_grad()
def score_frame(model, frame, cfg, device, bands) -> pd.DataFrame:
    dataset = PhysisDataset(frame, cfg, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.classifier.batch_size),
        shuffle=False,
        num_workers=int(cfg.optim.num_workers),
    )
    model.eval()
    scores = []
    for batch in loader:
        meta = {
            "age": batch["age"].to(device),
            **{k: batch[k].to(device) for k in ("gender", "view", "laterality")},
        }
        logits = model(batch["image"].to(device), batch["valid_mask"].to(device), meta)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return frame.assign(score=np.concatenate(scores))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    use_amp = device.type == "cuda" and str(cfg.run.amp) == "bf16"
    bands = band_list(cfg)
    clf = cfg.classifier

    manifest = load_manifest(cfg)
    manifest = assign_splits(manifest, str(clf.split_mode), int(cfg.run.seed))
    log.info("split mode: %s", clf.split_mode)

    train_df = make_frame(manifest, "train", bands)
    holdout = [str(b) for b in (clf.holdout_bands or [])]
    if holdout:
        keep = ~train_df["band"].isin([i for i, b in enumerate(bands) if b["name"] in holdout])
        log.info(
            "leave-age-band-out: dropping bands %s, %d of %d training images",
            holdout, int((~keep).sum()), len(train_df),
        )
        train_df = train_df[keep].reset_index(drop=True)

    if int(cfg.data.subset_train) > 0:
        train_df = subset_by_study(train_df, int(cfg.data.subset_train), int(cfg.run.seed))
    log.info(
        "train %d images (%.1f%% positive) | condition input: %s",
        len(train_df), 100 * train_df["label"].mean(), bool(clf.use_condition),
    )

    dataset = PhysisDataset(train_df, cfg, augment=bool(cfg.data.augment), seed=int(cfg.run.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(clf.batch_size),
        shuffle=True,
        num_workers=int(cfg.optim.num_workers),
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg.optim.num_workers) > 0,
    )

    model = build_classifier(cfg).to(device)
    if str(clf.init_from):
        info = load_backbone(model, str(clf.init_from))
        log.info("encoder initialized from %s (%d tensors)", clf.init_from, info["loaded"])
    else:
        log.info("encoder initialized from ImageNet")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(clf.lr), weight_decay=float(clf.weight_decay)
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = int(clf.epochs)
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * epochs
    warmup = steps_per_epoch * int(clf.warmup_epochs)
    step = 0

    for epoch in range(epochs):
        model.train()
        running, correct, seen, started = 0.0, 0, 0, time.perf_counter()
        for batch in loader:
            if step < warmup:
                lr = float(clf.lr) * (step + 1) / max(warmup, 1)
            else:
                progress = (step - warmup) / max(total_steps - warmup, 1)
                lr = 0.5 * float(clf.lr) * (1 + math.cos(math.pi * min(progress, 1.0)))
            for group in optimizer.param_groups:
                group["lr"] = lr

            targets = batch["label"].float().to(device)
            meta = {
                "age": batch["age"].to(device),
                **{k: batch[k].to(device) for k in ("gender", "view", "laterality")},
            }
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(
                    batch["image"].to(device), batch["valid_mask"].to(device), meta
                )
                loss = criterion(logits.float(), targets)

            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip))
            optimizer.step()

            running += float(loss.detach()) * targets.numel()
            correct += int(((logits.float() > 0) == (targets > 0.5)).sum())
            seen += targets.numel()
            step += 1

        log.info(
            "epoch %d | loss %.4f | train acc %.3f | %.1f s",
            epoch, running / max(seen, 1), correct / max(seen, 1),
            time.perf_counter() - started,
        )

    report = {"split_mode": str(clf.split_mode), "use_condition": bool(clf.use_condition),
              "holdout_bands": holdout, "init_from": str(clf.init_from), "splits": {}}

    for split in ("val", "test"):
        frame = make_frame(manifest, split, bands)
        if int(cfg.data.get(f"subset_{split}", 0) or 0) > 0:
            frame = subset_by_study(
                frame, int(cfg.data[f"subset_{split}"]), int(cfg.run.seed)
            )
        scored = score_frame(model, frame, cfg, device, bands)
        metrics = evaluate(scored, bands)
        report["splits"][split] = metrics
        log.info(
            "%s | image AUROC %.4f AUPRC %.4f | study AUROC %.4f AUPRC %.4f",
            split, metrics["image_auroc"], metrics["image_auprc"],
            metrics["study_auroc"], metrics["study_auprc"],
        )
        for row in metrics["by_age_band"]:
            log.info(
                "  band %-6s n %4d pos %4d | AUROC %.4f | study AUROC %.4f",
                row["band"], row["n_images"], row["n_positive"], row["auroc"],
                row["study_auroc"],
            )
        scored[["stem", "study_id", "age", "band", "label", "score"]].to_csv(
            run.dir / f"scores_{split}.csv", index=False
        )

        if holdout:
            wanted = [i for i, b in enumerate(bands) if b["name"] in holdout]
            held = scored[scored["band"].isin(wanted)]
            held_metrics = evaluate(held, bands)
            report["splits"][f"{split}_heldout_bands"] = held_metrics
            log.info(
                "  held-out bands %s | image AUROC %.4f | study AUROC %.4f",
                holdout, held_metrics["image_auroc"], held_metrics["study_auroc"],
            )

    torch.save({"model": model.state_dict(), "config": dict(clf)}, run.checkpoints / "best.pt")
    run.write_json("classifier_metrics.json", report)
    log.info("wrote %s", run.dir / "classifier_metrics.json")
    print(f"RUN_DIR={run.dir}")


if __name__ == "__main__":
    main()
