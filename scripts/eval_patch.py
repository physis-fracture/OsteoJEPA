"""E1 — does the age mechanism work. Patch-level metrics from a saved sweep.

No GPU and no model: everything here is arithmetic over `s_rec`, `s_min` and
`a_hat`, which the sweep already wrote. That is what makes the full lambda
ablation affordable.

Reported in the order EXPERIMENTS.md demands, prerequisites first: if the
residual curve is flat against age the predictor is ignoring the age channel and
nothing downstream means anything.

    python scripts/eval_patch.py --config configs/base.yaml \\
        --sweep-val sweep_val_all.npz --sweep-test sweep_test_all.npz \\
        --calibration calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import average_precision_score, roc_auc_score

from physis.data.geometry import band_list
from physis.eval.figures import plot_spatial_map
from physis.utils.config import load_config
from physis.utils.run import setup_run

COMPARATOR_PERCENTILE = 95.0  # top 5% of clean-image patches by residual
N_CROPS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 patch-level metrics")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sweep-val", required=True)
    parser.add_argument("--sweep-test", default=None)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


class Sweep:
    """One sweep archive, flattened to patch level with its labels."""

    def __init__(self, path: str):
        archive = np.load(path, allow_pickle=False)
        self.stem = archive["stem"]
        self.age = archive["age"]
        self.band = archive["band"]
        self.clean = archive["clean"]
        self.valid = archive["valid"]
        self.s_rec = archive["s_rec"]
        self.s_min = archive["s_min"]
        self.a_hat = archive["a_hat"]
        self.labels = {
            key: archive[key] for key in archive.files if key.startswith("label_")
        }
        self.delta = self.s_rec - self.s_min

        # An image carries fracture boxes iff some valid patch is not a plain
        # negative: positives and the discarded ring around a box both differ
        # from 0. Clean images have no boxes by construction.
        any_label = self.labels["label_050"]
        touched = ((any_label != 0) & self.valid).any(axis=1)
        self.has_boxes = touched & ~self.clean

    def score(self, lam: float) -> np.ndarray:
        return self.s_rec - lam * self.delta

    def image_band(self) -> np.ndarray:
        return self.band


def band_statistics(scores: np.ndarray, valid: np.ndarray, bands: np.ndarray, n_bands: int):
    """(mu, sigma) per band over the patches given. Recomputed at every lambda."""
    mu = np.full(n_bands, np.nan)
    sigma = np.full(n_bands, np.nan)
    for index in range(n_bands):
        rows = np.flatnonzero(bands == index)
        if rows.size == 0:
            continue
        pooled = scores[rows][valid[rows]]
        if pooled.size < 2:
            continue
        mu[index] = pooled.mean()
        sigma[index] = pooled.std()
    fallback_mu = np.nanmean(mu)
    fallback_sigma = np.nanmean(sigma)
    mu = np.where(np.isfinite(mu), mu, fallback_mu)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, fallback_sigma)
    return mu, sigma


def normalize(scores: np.ndarray, bands: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (scores - mu[bands][:, None]) / sigma[bands][:, None]


def patch_sets(sweep: Sweep, threshold_key: str):
    """Positive, primary-negative and secondary-negative boolean patch masks.

    Primary negatives come from fracture-containing images, so a score difference
    cannot be explained by machine, exposure, or age differences between images.
    Secondary negatives come from clean images and are reported separately.
    """
    labels = sweep.labels[threshold_key]
    positive = (labels == 1) & sweep.valid
    negative = (labels == 0) & sweep.valid
    primary = negative & sweep.has_boxes[:, None]
    secondary = negative & sweep.clean[:, None]
    return positive, primary, secondary


def auroc_auprc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    y_score = np.concatenate([scores[positive], scores[negative]])
    y_true = np.concatenate([np.ones(int(positive.sum())), np.zeros(int(negative.sum()))])
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def a_hat_variants(sweep: Sweep) -> dict:
    """Image-level implicit age under three aggregations, on clean images.

    SPEC defines a_hat per patch and leaves the image-level aggregate open. The
    median over every valid patch is the obvious choice and also the weakest:
    most patches are ordinary cortical bone whose argmin over age is arbitrary,
    so the informative minority is outvoted. Delta measures how much the freedom
    to choose an age helped a patch, which is exactly the patches worth listening
    to, so it doubles as a weight.
    """
    rows = np.flatnonzero(sweep.clean)
    recorded = sweep.age[rows]
    out: dict[str, float] = {}

    def mae(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        return float(np.abs(values[finite] - recorded[finite]).mean())

    median_all = np.array(
        [np.median(sweep.a_hat[r][sweep.valid[r]]) for r in rows], dtype=float
    )
    out["median_all_patches"] = mae(median_all)

    for percentile in (75.0, 90.0):
        picked = []
        for r in rows:
            valid = sweep.valid[r]
            delta = sweep.delta[r][valid]
            ages = sweep.a_hat[r][valid]
            if delta.size == 0:
                picked.append(np.nan)
                continue
            cut = np.percentile(delta, percentile)
            chosen = ages[delta >= cut]
            picked.append(np.median(chosen) if chosen.size else np.nan)
        out[f"median_top_delta_{int(100 - percentile)}pct"] = mae(np.array(picked, dtype=float))

    weighted = []
    for r in rows:
        valid = sweep.valid[r]
        delta = np.clip(sweep.delta[r][valid], 0, None)
        ages = sweep.a_hat[r][valid]
        weighted.append(np.average(ages, weights=delta) if delta.sum() > 0 else np.nan)
    out["delta_weighted_mean"] = mae(np.array(weighted, dtype=float))

    grid = np.arange(0.5, 19.01, 0.5)
    out["best_constant_baseline"] = float(min(np.abs(recorded - c).mean() for c in grid))
    return out


def comparator_group(sweep: Sweep):
    """The most surprising *normal* patches, built from the residuals themselves.

    GRAZPEDWRI-DX does not annotate growth plates, so the developmental-variant
    group cannot be selected by label. The top 5% of clean-image patches by
    residual at lambda = 0 are, by construction, the normal anatomy that looks
    most abnormal - mostly physes and carpal ossification centres.

    This is the sharpest test of the Delta hypothesis. Fracture versus plain
    cortical bone is a contest any model wins.
    """
    clean_rows = np.flatnonzero(sweep.clean)
    residual = sweep.s_rec[clean_rows]
    valid = sweep.valid[clean_rows]
    pooled = residual[valid]
    cut = np.percentile(pooled, COMPARATOR_PERCENTILE)
    selected = (residual >= cut) & valid
    return clean_rows, selected, float(cut)


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=e1")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    bands = band_list(cfg)
    n_bands = len(bands)
    band_names = [b["name"] for b in bands]
    lambda_grid = [float(x) for x in cfg.inference.lambda_grid]

    val = Sweep(args.sweep_val)
    log.info(
        "val sweep: %d images, %d clean, %d with boxes",
        len(val.stem), int(val.clean.sum()), int(val.has_boxes.sum()),
    )
    evaluation = Sweep(args.sweep_test) if args.sweep_test else val
    split_name = "test" if args.sweep_test else "val"
    log.info("evaluating on the %s fold: %d images", split_name, len(evaluation.stem))

    report: dict = {"evaluation_split": split_name}

    # --- prerequisite 1: is the implicit age worth anything ------------------
    ages = a_hat_variants(val)
    report["a_hat_mae"] = ages
    log.info("--- prerequisite: MAE of a_hat on clean validation images ---")
    for name, value in ages.items():
        log.info("  %-28s %.3f years", name, value)
    baseline = ages["best_constant_baseline"]
    best_name = min(
        (k for k in ages if k != "best_constant_baseline"), key=lambda k: ages[k]
    )
    gain = baseline - ages[best_name]
    report["a_hat_gain_over_constant_years"] = gain
    # The pre-registered abort threshold of 3 years was fixed without a baseline.
    # Always print the constant predictor next to it: "MAE 2.6, under the 3-year
    # criterion" reads far stronger than it is if a model that always guesses the
    # median age scores 2.8.
    if gain > 0:
        log.info(
            "  best aggregation is %s, beating the constant predictor by %.3f years (%.1f%%)",
            best_name, gain, 100.0 * gain / baseline,
        )
    else:
        log.warning(
            "  no aggregation beats always guessing age %.1f (MAE %.3f). The implicit "
            "age carries no usable signal, whatever the abort threshold says.",
            12.0, baseline,
        )

    # --- prerequisite 2: Delta, the quantity the whole method rests on -------
    clean_rows, comparator, cut = comparator_group(val)
    comparator_delta = val.delta[clean_rows][comparator]
    positive_050, primary_050, _ = patch_sets(evaluation, "label_050")
    fracture_delta = evaluation.delta[positive_050]
    report["delta"] = {
        "comparator_cut_residual": cut,
        "comparator_n_patches": int(comparator.sum()),
        "comparator_median": float(np.median(comparator_delta)),
        "comparator_q75": float(np.percentile(comparator_delta, 75)),
        "fracture_n_patches": int(positive_050.sum()),
        "fracture_median": float(np.median(fracture_delta)),
        "fracture_q75": float(np.percentile(fracture_delta, 75)),
    }
    log.info("--- prerequisite: Delta distributions ---")
    log.info(
        "  developmental-variant comparator: n=%d median Delta %.5f",
        comparator.sum(), np.median(comparator_delta),
    )
    log.info(
        "  fracture patches (coverage 0.50): n=%d median Delta %.5f",
        positive_050.sum(), np.median(fracture_delta),
    )
    log.info(
        "  prediction is that the first is much larger; ratio %.2fx",
        np.median(comparator_delta) / max(np.median(fracture_delta), 1e-12),
    )

    # --- main comparison: lambda ablation ------------------------------------
    log.info("--- lambda ablation, %s fold, coverage 0.50 ---", split_name)
    ablation = []
    for lam in lambda_grid:
        val_scores = val.score(lam)
        mu, sigma = band_statistics(
            val_scores[val.clean], val.valid[val.clean], val.band[val.clean], n_bands
        )
        scores = normalize(evaluation.score(lam), evaluation.band, mu, sigma)
        auroc_primary, auprc_primary = auroc_auprc(scores, positive_050, primary_050)
        _, _, secondary = patch_sets(evaluation, "label_050")
        auroc_secondary, auprc_secondary = auroc_auprc(scores, positive_050, secondary)
        ablation.append(
            {
                "lambda": lam,
                "auroc_primary": auroc_primary,
                "auprc_primary": auprc_primary,
                "auroc_secondary": auroc_secondary,
                "auprc_secondary": auprc_secondary,
            }
        )
        log.info(
            "  lambda %.1f | AUROC %.4f AUPRC %.4f (primary neg) | AUROC %.4f (secondary)",
            lam, auroc_primary, auprc_primary, auroc_secondary,
        )
    report["lambda_ablation"] = ablation

    best = max(ablation, key=lambda row: row["auroc_primary"])
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    lambda_star = float(calibration["lambda_star"])
    report["lambda_star_label_free"] = lambda_star
    report["lambda_best_auroc"] = best["lambda"]
    log.info(
        "lambda* from the label-free criterion is %.1f; lambda maximising AUROC is %.1f",
        lambda_star, best["lambda"],
    )
    if abs(lambda_star - best["lambda"]) > 1e-9:
        log.warning(
            "they differ. Report both: the shipped score uses the label-free lambda, "
            "and the gap is the price of staying annotation-free."
        )

    # --- coverage thresholds --------------------------------------------------
    log.info("--- coverage thresholds, at the label-free lambda* = %.1f ---", lambda_star)
    by_threshold = {}
    val_scores = val.score(lambda_star)
    mu, sigma = band_statistics(
        val_scores[val.clean], val.valid[val.clean], val.band[val.clean], n_bands
    )
    scores_star = normalize(evaluation.score(lambda_star), evaluation.band, mu, sigma)
    for key in sorted(evaluation.labels):
        positive, primary, secondary = patch_sets(evaluation, key)
        auroc, auprc = auroc_auprc(scores_star, positive, primary)
        by_threshold[key] = {
            "n_positive": int(positive.sum()),
            "n_primary_negative": int(primary.sum()),
            "auroc": auroc,
            "auprc": auprc,
        }
        log.info(
            "  %s | positives %6d | AUROC %.4f AUPRC %.4f",
            key, positive.sum(), auroc, auprc,
        )
    report["coverage_thresholds"] = by_threshold

    # --- stratification by age band ------------------------------------------
    # Within a band the normalization is an affine transform, so it cannot change
    # the ranking; these numbers are identical on raw or normalized scores.
    log.info("--- AUROC by age band, coverage 0.50, lambda* = %.1f ---", lambda_star)
    stratified = []
    for index, name in enumerate(band_names):
        rows = evaluation.band == index
        mask = np.zeros_like(positive_050)
        mask[rows] = True
        auroc, auprc = auroc_auprc(scores_star, positive_050 & mask, primary_050 & mask)
        n_pos = int((positive_050 & mask).sum())
        stratified.append({"band": name, "n_positive": n_pos, "auroc": auroc, "auprc": auprc})
        log.info("  band %-6s positives %5d | AUROC %.4f AUPRC %.4f", name, n_pos, auroc, auprc)
    report["by_age_band"] = stratified

    # --- figures --------------------------------------------------------------
    _figures(run, ablation, calibration, comparator_delta, fracture_delta, evaluation, log)

    # --- spatial map of where age matters ------------------------------------
    # Delta stands in for V(p) here: the sweep archive does not carry the full
    # residual curve, and Delta measures the same thing one step coarser - how
    # much the freedom to choose an age helped this patch.
    grid = int(round(evaluation.valid.shape[1] ** 0.5))
    clean_delta = np.where(evaluation.valid, evaluation.delta, np.nan)[evaluation.clean]
    with warnings.catch_warnings():
        # Patch positions that are padding in every clean image are all-NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        spatial = np.nanmean(clean_delta, axis=0).reshape(grid, grid)
    plot_spatial_map(
        spatial, run.figures / "delta_spatial.png",
        "mean Delta per patch position, clean images",
    )
    np.save(run.figures / "delta_spatial.npy", spatial)

    _qualitative_crops(run, cfg, val, clean_rows, comparator, log)

    run.write_json("e1_metrics.json", report)
    log.info("wrote %s", run.dir / "e1_metrics.json")


def _qualitative_crops(run, cfg, sweep: Sweep, clean_rows, selected, log) -> None:
    """Eight crops from the developmental-variant group, no numbers attached.

    Presented without a quantitative claim, per EXPERIMENTS.md: these are the
    normal patches the model found most surprising, and the point is to let a
    reader see whether they are physes and carpal ossification centres or
    something uninteresting.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    images_dir = Path(cfg.images_dir)
    patch = int(cfg.image.patch)
    grid = int(round(sweep.valid.shape[1] ** 0.5))
    half = patch * 2  # a 5-patch-wide window around the patch of interest

    rows, cols = np.nonzero(selected)
    if rows.size == 0:
        log.warning("comparator group is empty; no crops written")
        return
    # Spread the picks across different images rather than eight from one.
    order = np.argsort(-sweep.s_rec[clean_rows][rows, cols])
    picked, seen = [], set()
    for index in order:
        image_row = int(rows[index])
        if image_row in seen:
            continue
        seen.add(image_row)
        picked.append((image_row, int(cols[index])))
        if len(picked) == N_CROPS:
            break

    figure, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for axis, (image_row, token) in zip(axes.ravel(), picked):
        stem = str(sweep.stem[clean_rows[image_row]])
        path = images_dir / f"{stem}.png"
        if not path.exists():
            axis.axis("off")
            continue
        with Image.open(path) as handle:
            array = np.asarray(handle, dtype=np.float32) / 65535.0
        j, i = divmod(token, grid)
        cy, cx = j * patch + patch // 2, i * patch + patch // 2
        # Shift the window at the border rather than clipping it, so every crop
        # is the same size and the eight are visually comparable.
        y0 = int(np.clip(cy - half, 0, array.shape[0] - 2 * half))
        x0 = int(np.clip(cx - half, 0, array.shape[1] - 2 * half))
        y1, x1 = y0 + 2 * half, x0 + 2 * half
        axis.imshow(array[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        axis.add_patch(
            plt.Rectangle(
                (i * patch - x0, j * patch - y0), patch, patch,
                fill=False, edgecolor="yellow", linewidth=1.4,
            )
        )
        axis.set_xticks([]), axis.set_yticks([])
        axis.set_title(f"{stem[:18]}  age {sweep.age[clean_rows[image_row]]:.1f}", fontsize=7)
    for axis in axes.ravel()[len(picked):]:
        axis.axis("off")

    figure.suptitle(
        "most surprising patches on clean images (lambda = 0), no quantitative claim",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(run.figures / "qualitative_crops.png", dpi=120)
    plt.close(figure)
    log.info("wrote %d qualitative crops", len(picked))


def _figures(run, ablation, calibration, comparator_delta, fracture_delta, evaluation, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [row["lambda"] for row in ablation]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    axes[0].plot(lambdas, [r["auroc_primary"] for r in ablation], "o-", label="AUROC")
    axes[0].plot(lambdas, [r["auprc_primary"] for r in ablation], "s-", label="AUPRC")
    axes[0].set_xlabel("lambda"), axes[0].set_title("patch separation vs lambda")
    axes[0].legend(), axes[0].grid(alpha=0.3)

    table = calibration["lambda_table"]
    twin = axes[1]
    twin.plot([r["lambda"] for r in table], [r["W"] for r in table], "o-", color="crimson")
    twin.set_xlabel("lambda"), twin.set_ylabel("W(lambda)")
    twin.set_title("label-free criterion: upper tail width")
    twin.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(run.figures / "lambda_ablation.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.4))
    bins = np.linspace(
        0, float(np.percentile(np.concatenate([comparator_delta, fracture_delta]), 99)), 60
    )
    axis.hist(comparator_delta, bins=bins, alpha=0.55, density=True,
              label=f"developmental variants (n={comparator_delta.size})")
    axis.hist(fracture_delta, bins=bins, alpha=0.55, density=True,
              label=f"fracture patches (n={fracture_delta.size})")
    axis.set_xlabel("Delta = s(a_rec) - min_a s(a)"), axis.set_ylabel("density")
    axis.set_title("does age explain the surprising normal patches?")
    axis.legend(fontsize=8), axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(run.figures / "delta_distributions.png", dpi=120)
    plt.close(figure)
    log.info("wrote figures to %s", run.figures)


if __name__ == "__main__":
    main()
