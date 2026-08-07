"""Export a small set of real studies for testing the web application.

The web client needs more than images. Every request carries `age_years`, and
age selects the percentile band, so an image without its age cannot be scored at
all. This writes the images and the metadata that goes with them, in the enums
the web schema spells.

Studies come from the **test fold**, never seen during training, and are chosen
by the rule in `pick_demo_studies`: positives carry a box and no cast, negatives
are strictly clean. A random draw from the whole dataset would eventually hand
someone a plaster follow-up as its "normal" example, and that case scores 1.0
with nothing broken in it.

    python scripts/export_demo_set.py --n 8 --out demo_set
    python scripts/export_demo_set.py --confident --out demo_set   # for a video

Then zip `demo_set/` and send it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import pick_demo_studies
from physis.utils.config import load_config

# The web schema spells these differently from the manifest. Translating here
# rather than in the client keeps one definition of the mapping.
VIEW_NAME = {1: "PA", 2: "LATERAL", 3: "OTHER"}
LATERALITY_NAME = {"L": "left", "R": "right"}
SEX_NAME = {"M": "male", "F": "female"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="export demo studies for the web app")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="demo_set")
    parser.add_argument("--n", type=int, default=8, help="studies per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confident",
        action="store_true",
        help="pick the cases the model is surest about, for a recording",
    )
    parser.add_argument("--scores", default="artifacts/clf/clf_main/rescored/scores_test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.manifest)
    images_dir = Path(cfg.images_dir)

    scores = args.scores if args.confident else None
    if scores is not None and not Path(scores).exists():
        raise SystemExit(f"--confident needs {scores}; run score_classifier first")

    chosen = pick_demo_studies(manifest, args.n, args.seed, offline_scores=scores)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    records = []
    for study in chosen.itertuples():
        for row in manifest[manifest["study_id"] == study.study_id].itertuples():
            shutil.copyfile(images_dir / f"{row.stem}.png", out / "images" / f"{row.stem}.png")
            records.append({
                "study_id": study.study_id,
                "image_id": row.stem,
                "file": f"images/{row.stem}.png",
                "age_years": round(float(row.age), 1),
                "sex": SEX_NAME.get(str(row.gender), "unknown"),
                "view": VIEW_NAME.get(int(row.projection), "UNKNOWN"),
                "laterality": LATERALITY_NAME.get(str(row.laterality), "unknown"),
                # Ground truth, for checking a result. Not sent to the service.
                "has_fracture": bool(study.truth),
            })

    frame = pd.DataFrame(records)
    frame.to_csv(out / "studies.csv", index=False)
    (out / "studies.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (out / "README.md").write_text(_readme(frame), encoding="utf-8")

    print(f"{len(chosen)} studies, {len(frame)} images -> {out}/")
    print(f"  fracture: {int(chosen['truth'].sum())}   clean: {int((1 - chosen['truth']).sum())}")
    print(f"  ages {frame['age_years'].min():.1f} to {frame['age_years'].max():.1f}")


def _readme(frame: pd.DataFrame) -> str:
    """A note that travels with the folder, since whoever unzips it will not
    have read the script that produced it."""
    # One row per study. Projection varies within a study, so it is summarised
    # rather than taken from the first image, which would report every study as
    # PA and hide that each one has a lateral projection too.
    by_study = frame.groupby("study_id", sort=False).agg(
        age=("age_years", "first"),
        sex=("sex", "first"),
        views=("view", lambda v: " + ".join(v)),
        truth=("has_fracture", "max"),
    )
    listing = "\n".join(
        f"| `{r.Index}` | {r.age} | {r.sex} | {r.views} | "
        f"{'fracture' if r.truth else 'clean'} |"
        for r in by_study.itertuples()
    )
    return f"""# Demo studies

{frame['study_id'].nunique()} studies, {len(frame)} images, from the GRAZPEDWRI-DX test
fold. The model never saw any of them during training. CC0 Public Domain, same
as the source dataset, so they are safe to upload anywhere.

Images are already at 384x384, 16-bit PNG, preprocessed exactly as the model
expects. The service detects that and does not preprocess them a second time,
so what you upload is what it scores.

`studies.csv` and `studies.json` carry the metadata each request needs. Age is
required and has no fallback: it selects the percentile band, and defaulting it
would score a child against the wrong population.

| study | age | sex | projections | truth |
|---|---|---|---|---|
{listing}

The `truth` column is for checking a result, not for sending. It is not part of
any request.

## What to expect

Fracture studies should land high and clean ones low, but this is a handful of
cases and not a measurement. Study-level AUROC is 0.9580 over all 2,139 test
studies, so roughly one in twenty single cases will look wrong on its own.

None of these studies carries a plaster cast, on either side. That is
deliberate. The model partly reads cast as evidence of fracture, so a casted
study with nothing broken in it scores near 1.0 and would look like a failure
that is really a badly chosen example.
"""


if __name__ == "__main__":
    main()
