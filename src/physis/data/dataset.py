"""Manifest loading, split integrity, and the image dataset.

Assertions over prints (CONVENTIONS): split integrity and the padding rules are
asserted, because a printed warning scrolls past and a failed assertion stops
the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

from .geometry import erode_valid_mask, valid_mask_from_geometry

# Condition-vector categoricals. The trailing entry is the trained `unknown`
# embedding, so an image with missing metadata is scored rather than dropped.
GENDER_VOCAB = {"M": 0, "F": 1, "O": 2}
VIEW_VOCAB = {1: 0, 2: 1, 3: 2}
LATERALITY_VOCAB = {"L": 0, "R": 1}
GENDER_UNKNOWN = len(GENDER_VOCAB)
VIEW_UNKNOWN = len(VIEW_VOCAB)
LATERALITY_UNKNOWN = len(LATERALITY_VOCAB)

REQUIRED_COLUMNS = [
    "stem", "patient_id", "study_id", "age", "gender", "projection", "laterality",
    "fold", "split", "clean_strict", "n_fracture_box",
    "scale", "new_w", "new_h", "pad_x", "pad_y",
]


def load_manifest(cfg: DictConfig, *, verify_counts: bool = True) -> pd.DataFrame:
    """Load `manifest.csv` and assert the invariants the whole project rests on."""
    df = pd.read_csv(cfg.manifest)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"manifest is missing columns: {missing}"

    # Hard rule 1: the split was fixed once, on patient groups. A random
    # per-image split would put the same patient in train and test.
    per_patient_splits = df.groupby("patient_id")["split"].nunique()
    leaked = int((per_patient_splits > 1).sum())
    assert leaked == 0, f"{leaked} patients appear in more than one split"

    assert df["age"].notna().all(), "manifest has missing ages"

    if verify_counts:
        n_clean = int(df["clean_strict"].sum())
        assert n_clean == int(cfg.clean_set.n_total), (
            f"clean_strict count is {n_clean}, expected {cfg.clean_set.n_total}. "
            "The AO classification filter is the usual casualty when the filtering "
            "code is rewritten; without it 773 occult fractures re-enter the clean set."
        )
        n_train = int((df["clean_strict"] & (df["split"] == "train")).sum())
        assert n_train == int(cfg.clean_set.n_train), (
            f"clean train count is {n_train}, expected {cfg.clean_set.n_train}"
        )

    return df


def select_split(
    df: pd.DataFrame, split: str, *, clean_only: bool = False
) -> pd.DataFrame:
    """Rows for one split, optionally restricted to the pretraining-clean set."""
    assert split in {"train", "val", "test"}, f"unknown split {split!r}"
    out = df[df["split"] == split]
    if clean_only:
        out = out[out["clean_strict"]]
    return out.reset_index(drop=True)


def subset_by_study(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministically take about `n` images, keeping every study intact.

    Studies are kept whole so that `r_study = max over images in the study` is
    exercised on studies that actually hold more than one image.
    """
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    studies = df["study_id"].drop_duplicates().sort_values().to_numpy()
    order = np.random.default_rng(seed).permutation(len(studies))
    taken, total = [], 0
    for k in order:
        study = studies[k]
        size = int((df["study_id"] == study).sum())
        if total + size > n and taken:
            continue
        taken.append(study)
        total += size
        if total >= n:
            break
    return df[df["study_id"].isin(taken)].reset_index(drop=True)


def pick_demo_studies(
    manifest: pd.DataFrame,
    n: int,
    seed: int,
    *,
    split: str = "test",
    offline_scores: str | Path | None = None,
    exclude_patients: frozenset | set | None = None,
) -> pd.DataFrame:
    """Test-fold studies chosen so a demo shows the model, not its shortcut.

    Positives carry a fracture box and **no cast and no metal**, because E4a
    measured the model reading the cast: among fracture-negative images, 36 of
    the 38 carrying a cast score above 0.5. Negatives come from `clean_strict`,
    which excludes cast, metal, an AO classification, and every indirect sign of
    injury. Draw a negative without that filter and a plaster follow-up will
    score 1.0 with nothing broken in it, which looks like a broken model when
    the fault is in the choice of example.

    `offline_scores` switches from a random draw to the cases the model is most
    sure about. That is demo selection, not evaluation: ranking by the model's
    own score and then reporting how well it did would be circular.

    `exclude_patients` drops whole patients, not studies. Excluding by study
    would let a second draw return the other wrist of a child already in the
    first, which is not a fresh case to look at.

    Returns one row per study with `truth`, `age` and `n_images`.
    """
    rows = manifest[manifest["split"] == split]
    if exclude_patients:
        rows = rows[~rows["patient_id"].isin(exclude_patients)]
    by_study = rows.groupby("study_id").agg(
        boxes=("n_fracture_box", "max"),
        cast=("tag_cast", "max"),
        metal=("lbl_metal", "max"),
        clean=("clean_strict", "min"),
        age=("age", "first"),
        n_images=("stem", "count"),
    )
    positive = by_study[(by_study["boxes"] > 0) & (~by_study["cast"]) & (~by_study["metal"])]
    negative = by_study[by_study["clean"]]

    if offline_scores is not None:
        score = pd.read_csv(offline_scores).groupby("study_id")["score"].max()
        positive = positive.join(score).sort_values("score", ascending=False)
        negative = negative.join(score).sort_values("score", ascending=True)
        chosen = [positive.head(n).assign(truth=1), negative.head(n).assign(truth=0)]
    else:
        chosen = [
            positive.sample(min(n, len(positive)), random_state=seed).assign(truth=1),
            negative.sample(min(n, len(negative)), random_state=seed).assign(truth=0),
        ]
    return pd.concat(chosen).reset_index()


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

EVAL_SUBSETS = ("clean", "fracture", "all")


def build_eval_frame(
    cfg: DictConfig, split: str = "val", subset: str = "clean"
) -> pd.DataFrame:
    """Images of one split, subsetted for the smoke path if configured.

    `clean` is the pretraining-clean set, used for calibration and as the
    secondary-negative pool. `fracture` is images carrying at least one fracture
    box, which is where positive patches come from and which the clean set
    excludes by construction. `all` is both, and is what a single GPU session
    sweeps so that nothing has to be recomputed later.

    Calibration and scoring share this function so they see exactly the same
    images; lambda* and (mu, sigma) would otherwise be estimated on different
    sets.
    """
    assert subset in EVAL_SUBSETS, f"unknown eval subset {subset!r}"
    manifest = load_manifest(cfg)
    frame = select_split(manifest, split, clean_only=(subset == "clean"))
    if subset == "fracture":
        frame = frame[frame["n_fracture_box"] > 0].reset_index(drop=True)
    n = int(cfg.data.get(f"subset_{split}", 0) or 0)
    if n > 0:
        frame = subset_by_study(frame, n, int(cfg.run.seed))
    return frame


class PhysisDataset(Dataset):
    """384x384 preprocessed radiographs plus everything the condition vector needs.

    Returns images already in [0, 1]: the PNGs are 16-bit, so they are divided by
    65535. Reading them as 8-bit would quantise the whole intensity range into
    256 levels and cost exactly the fine cortical detail the model is meant to see.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_cfg: DictConfig,
        *,
        augment: bool = False,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(data_cfg.images_dir)
        self.size = int(data_cfg.image.size)
        self.patch = int(data_cfg.image.patch)
        self.grid = self.size // self.patch
        self.augment = augment
        self.aug_cfg = data_cfg.augment
        # Applied here so every consumer - training masks, the inference
        # partition, the surprise map - inherits the same valid region.
        self.erode_rings = int(data_cfg.get("masking", {}).get("erode_border_patches", 0) or 0)
        self.seed = seed
        assert not bool(self.aug_cfg.hflip), (
            "hflip is disabled by design: it changes laterality, and laterality is "
            "one of the condition vector fields"
        )

    def __len__(self) -> int:
        return len(self.df)

    def read_image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw image in [0, 1] and its geometry-derived valid mask, unaugmented."""
        row = self.df.iloc[index]
        path = self.images_dir / f"{row['stem']}.png"
        with Image.open(path) as im:
            arr = np.array(im)
        assert arr.dtype == np.uint16, f"expected a 16-bit PNG, got {arr.dtype} at {path}"
        assert arr.shape == (self.size, self.size), f"unexpected image shape {arr.shape}"
        image = arr.astype(np.float32) / 65535.0
        valid = valid_mask_from_geometry(
            row["pad_x"], row["pad_y"], row["new_w"], row["new_h"],
            size=self.size, patch=self.patch,
        )
        return image, erode_valid_mask(valid, self.erode_rings)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image, valid = self.read_image(index)

        if self.augment:
            image, valid = self._augment(image, row, index)

        return {
            "image": torch.from_numpy(image)[None],           # (1, 384, 384)
            "valid_mask": torch.from_numpy(valid),            # (24, 24) bool, [j, i]
            "age": torch.tensor(float(row["age"]), dtype=torch.float32),
            "gender": torch.tensor(GENDER_VOCAB.get(row["gender"], GENDER_UNKNOWN)),
            "view": torch.tensor(VIEW_VOCAB.get(int(row["projection"]), VIEW_UNKNOWN)),
            "laterality": torch.tensor(LATERALITY_VOCAB.get(row["laterality"], LATERALITY_UNKNOWN)),
            "stem": str(row["stem"]),
            "study_id": str(row["study_id"]),
            "index": index,
            # Supervised runs carry a target; the JEPA path has none, hence -1.
            "label": torch.tensor(float(row["label"]) if "label" in row else -1.0),
        }

    def _augment(self, image: np.ndarray, row, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Rotation, translation, and photometric jitter, with the mask following.

        Rotation and translation move the content box, so the valid mask cannot
        be read off the manifest geometry afterwards. A pixel-level content mask
        is therefore carried through the *same* affine transform and the patch
        mask is re-derived from it: a patch is valid only when all 16x16 of its
        pixels are still fully inside the transformed content. That keeps the
        whole-box rule intact under an arbitrary transform, where an
        axis-aligned formula would quietly start admitting padding.

        Conservative on purpose. Horizontal flip is not offered at all, because
        it changes laterality and laterality is a condition vector field.
        """
        rng = np.random.default_rng((self.seed * 1_000_003 + index) % (2**32))
        content = content_pixel_mask(
            row["pad_x"], row["pad_y"], row["new_w"], row["new_h"], size=self.size
        )

        angle = float(rng.uniform(-1, 1) * float(self.aug_cfg.rotate_deg))
        shift = float(self.aug_cfg.translate_frac) * self.size
        translate = (float(rng.uniform(-1, 1) * shift), float(rng.uniform(-1, 1) * shift))

        if angle != 0.0 or translate != (0.0, 0.0):
            image = _affine(image, angle, translate)
            content = _affine(content.astype(np.float32), angle, translate)

        # Interpolation blurs the content edge; require a patch to be entirely
        # inside before calling it valid.
        blocks = content.reshape(self.grid, self.patch, self.grid, self.patch)
        valid = erode_valid_mask(blocks.min(axis=(1, 3)) >= 1.0 - 1e-3, self.erode_rings)

        amount = float(self.aug_cfg.brightness_contrast)
        if amount > 0:
            gain = 1.0 + float(rng.uniform(-amount, amount))
            bias = float(rng.uniform(-amount, amount))
            inside = content >= 1.0 - 1e-3
            image = image.copy()
            image[inside] = np.clip(image[inside] * gain + bias, 0.0, 1.0)
        # Padding must stay exactly zero; otherwise "empty" patches stop being
        # identifiable from the pixels alone.
        image = np.where(content >= 1.0 - 1e-3, image, 0.0).astype(np.float32)
        return image, valid


def content_pixel_mask(
    pad_x: float, pad_y: float, new_w: float, new_h: float, size: int = 384
) -> np.ndarray:
    """Pixel-level mask of the non-padding region, indexed [y, x]."""
    mask = np.zeros((size, size), dtype=bool)
    y0, y1 = int(round(pad_y)), int(round(pad_y + new_h))
    x0, x1 = int(round(pad_x)), int(round(pad_x + new_w))
    mask[y0:y1, x0:x1] = True
    return mask


def _affine(array: np.ndarray, angle_deg: float, translate: tuple[float, float]) -> np.ndarray:
    """Rotate about the centre and translate, filling outside with zero."""
    image = Image.fromarray(array.astype(np.float32), mode="F")
    out = image.rotate(
        angle_deg, resample=Image.BILINEAR, translate=translate, fillcolor=0.0
    )
    return np.asarray(out, dtype=np.float32)
