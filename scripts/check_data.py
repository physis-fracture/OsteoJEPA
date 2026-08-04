"""M1 verification: geometry, split integrity, patch labels, and debug figures.

Every number M1 is accepted on comes from here, and the assertions that guard
the hard rules run on the whole manifest rather than on a sample.

    bash: .venv/Scripts/python.exe scripts/check_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest, select_split
from physis.data.geometry import age_band_index, band_list, valid_mask_from_geometry
from physis.data.masking import partition_interleaved
from physis.data.patch_labels import (
    LABEL_DISCARD,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    boxes_by_stem,
    label_patches,
    load_boxes,
    patch_coverage,
)
from physis.utils.config import load_config
from physis.utils.run import get_logger, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 data and masking checks")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--figures", type=int, default=4)
    return parser.parse_args()


def valid_fractions(manifest, size: int, patch: int) -> np.ndarray:
    """Valid patch fraction of every image, from the geometry columns."""
    grid = size // patch
    index = np.arange(grid)
    pad_x = manifest["pad_x"].to_numpy()[:, None]
    pad_y = manifest["pad_y"].to_numpy()[:, None]
    new_w = manifest["new_w"].to_numpy()[:, None]
    new_h = manifest["new_h"].to_numpy()[:, None]
    valid_i = (patch * index >= pad_x) & (patch * (index + 1) <= pad_x + new_w)
    valid_j = (patch * index >= pad_y) & (patch * (index + 1) <= pad_y + new_h)
    return valid_i.sum(axis=1) * valid_j.sum(axis=1) / (grid * grid)


def count_patch_labels(manifest, boxes, cfg, threshold: float) -> dict:
    """Positive / discarded / primary-negative patch counts at one threshold."""
    grouped = boxes_by_stem(boxes)
    size, patch = int(cfg.image.size), int(cfg.image.patch)
    margin = int(cfg.patch_labels.negative_margin_patches)

    totals = {"images": 0, "positive": 0, "discarded": 0, "negative": 0, "images_without_positive": 0}
    with_boxes = manifest[manifest["n_fracture_box"] > 0]
    for row in with_boxes.itertuples():
        stem_boxes = grouped.get(row.stem)
        if stem_boxes is None:
            continue
        valid = valid_mask_from_geometry(row.pad_x, row.pad_y, row.new_w, row.new_h, size, patch)
        coverage = patch_coverage(stem_boxes, size=size, patch=patch)
        labels = label_patches(
            coverage, valid, positive_coverage=threshold, negative_margin_patches=margin
        )
        positive = int((labels == LABEL_POSITIVE).sum())
        totals["images"] += 1
        totals["positive"] += positive
        totals["discarded"] += int((labels == LABEL_DISCARD).sum())
        totals["negative"] += int((labels == LABEL_NEGATIVE).sum())
        totals["images_without_positive"] += int(positive == 0)
    return totals


def debug_figure(dataset: PhysisDataset, index: int, boxes: dict, cfg, path: Path) -> None:
    """Image, valid mask overlay, fracture boxes, and positive patches.

    Misapplied pad_x is invisible in every summary statistic and immediately
    obvious here: the shaded padding would sit on the wrong side of the anatomy,
    or the boxes would float off the bone.
    """
    row = dataset.df.iloc[index]
    image, valid = dataset.read_image(index)
    patch = dataset.patch
    stem_boxes = boxes.get(str(row["stem"]), np.zeros((0, 4)))
    coverage = patch_coverage(stem_boxes, size=dataset.size, patch=patch)
    labels = label_patches(
        coverage, valid,
        positive_coverage=float(cfg.patch_labels.positive_coverage),
        negative_margin_patches=int(cfg.patch_labels.negative_margin_patches),
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for axis in axes:
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_xticks([]), axis.set_yticks([])

    axes[0].set_title(f"{row['stem']}\nage {row['age']} {row['gender']} view {row['projection']}")

    # Panel 2: padding shaded red, patch grid drawn, boxes in yellow.
    padding = np.repeat(np.repeat(~valid, patch, axis=0), patch, axis=1)
    overlay = np.zeros((*padding.shape, 4))
    overlay[padding] = (1.0, 0.0, 0.0, 0.35)
    axes[1].imshow(overlay, interpolation="nearest")
    for edge in range(0, dataset.size + 1, patch):
        axes[1].axhline(edge - 0.5, color="cyan", linewidth=0.25, alpha=0.5)
        axes[1].axvline(edge - 0.5, color="cyan", linewidth=0.25, alpha=0.5)
    for x0, y0, x1, y1 in stem_boxes:
        axes[1].add_patch(
            mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.6)
        )
    axes[1].set_title(
        f"padding {(~valid).sum()}/{valid.size} patches (red)\nvalid fraction {valid.mean():.3f}"
    )

    # Panel 3: positive patches green, one interleaved partition group in blue.
    label_overlay = np.zeros((dataset.size, dataset.size, 4))
    positive = np.repeat(np.repeat(labels == LABEL_POSITIVE, patch, 0), patch, 1)
    group0 = np.repeat(np.repeat(partition_interleaved(valid, 4)[0], patch, 0), patch, 1)
    label_overlay[group0] = (0.2, 0.5, 1.0, 0.22)
    label_overlay[positive] = (0.0, 1.0, 0.0, 0.45)
    axes[2].imshow(label_overlay, interpolation="nearest")
    for x0, y0, x1, y1 in stem_boxes:
        axes[2].add_patch(
            mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=1.6)
        )
    axes[2].set_title(
        f"positive patches at 0.50: {(labels == LABEL_POSITIVE).sum()} (green)\n"
        f"partition group 0 (blue)"
    )

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=data_check")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg)
    log = run.log

    manifest = load_manifest(cfg)  # asserts split integrity and the clean counts
    log.info("manifest rows: %d", len(manifest))

    # --- valid patch fraction -------------------------------------------------
    fractions = valid_fractions(manifest, int(cfg.image.size), int(cfg.image.patch))
    mean_fraction = float(fractions.mean())
    pixel_fraction = float(
        ((manifest["new_w"] * manifest["new_h"]) / (int(cfg.image.size) ** 2)).mean()
    )
    log.info(
        "mean valid patch fraction %.4f (min %.3f, max %.3f); mean pixel content fraction %.4f",
        mean_fraction, fractions.min(), fractions.max(), pixel_fraction,
    )
    assert abs(mean_fraction - 0.57) <= 0.05, (
        f"mean valid patch fraction {mean_fraction:.4f} is outside 0.57 +/- 0.05; "
        "the geometry columns are being applied wrong"
    )

    # --- split integrity ------------------------------------------------------
    per_patient = manifest.groupby("patient_id")["split"].nunique()
    assert int((per_patient > 1).sum()) == 0
    per_study = manifest.groupby("study_id")["split"].nunique()
    assert int((per_study > 1).sum()) == 0, "a study is split across folds"
    log.info(
        "split integrity: %d patients, %d studies, none crossing a split",
        manifest["patient_id"].nunique(), manifest["study_id"].nunique(),
    )

    # --- age bands ------------------------------------------------------------
    bands = band_list(cfg)
    clean_val = select_split(manifest, "val", clean_only=True)
    counts = [0] * len(bands)
    for age in clean_val["age"]:
        counts[age_band_index(float(age), bands)] += 1
    for band, count in zip(bands, counts):
        log.info("band %-6s clean val images %4d", band["name"], count)
        assert count >= int(cfg.min_band_count), (
            f"band {band['name']} holds {count} clean validation images, "
            f"below the required {cfg.min_band_count}"
        )

    # --- patch labels ---------------------------------------------------------
    boxes = load_boxes(cfg)
    log.info("fracture boxes: %d over %d images", len(boxes), boxes["stem"].nunique())
    label_summary = {}
    for threshold in [float(t) for t in cfg.patch_labels.report_thresholds]:
        totals = count_patch_labels(manifest, boxes, cfg, threshold)
        label_summary[f"{threshold:.2f}"] = totals
        log.info(
            "coverage %.2f: %d positive patches over %d images "
            "(%.2f per image), %d primary negatives, %d images with no positive",
            threshold, totals["positive"], totals["images"],
            totals["positive"] / max(totals["images"], 1),
            totals["negative"], totals["images_without_positive"],
        )

    # --- debug figures --------------------------------------------------------
    grouped = boxes_by_stem(boxes)
    with_boxes = manifest[manifest["n_fracture_box"] > 0].reset_index(drop=True)
    # One of each padding orientation, so a swapped pad_x/pad_y cannot hide.
    portrait = with_boxes[with_boxes["pad_x"] > with_boxes["pad_y"]].head(2)
    landscape = with_boxes[with_boxes["pad_y"] >= with_boxes["pad_x"]].head(2)
    chosen = list(portrait.index) + list(landscape.index)
    chosen = chosen[: max(args.figures, 1)]

    dataset = PhysisDataset(with_boxes, cfg, augment=False)
    figure_paths = []
    for index in chosen:
        stem = with_boxes.iloc[index]["stem"]
        path = run.figures / f"overlay_{stem}.png"
        debug_figure(dataset, index, grouped, cfg, path)
        figure_paths.append(str(path))
        log.info("wrote %s", path)

    run.write_json(
        "data_check.json",
        {
            "n_images": int(len(manifest)),
            "mean_valid_patch_fraction": mean_fraction,
            "mean_pixel_content_fraction": pixel_fraction,
            "valid_fraction_min": float(fractions.min()),
            "valid_fraction_max": float(fractions.max()),
            "bands_clean_val": {b["name"]: c for b, c in zip(bands, counts)},
            "patch_labels": label_summary,
            "figures": figure_paths,
        },
    )
    log.info("all M1 checks passed")


if __name__ == "__main__":
    main()
