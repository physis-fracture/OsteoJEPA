"""Re-score a trained classifier, saving **logits** alongside probabilities.

The training run saved sigmoid outputs only, and that lost the ordering: 55.6%
of test scores land above 0.9999, and the top-20% flagged group shares 84
distinct values across 815 studies. Float32 has no resolution left that close to
1, so a worklist built on those probabilities cannot rank the cases it flags.

The logit does not have that problem. This is inference only - no training - so
it costs a few minutes.

    modal run --detach modal_app.py::score_classifier --name clf_main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import PhysisDataset, load_manifest
from physis.data.geometry import age_band_index, band_list
from physis.eval.metrics import evaluate
from physis.models.classifier import build_classifier
from physis.utils.config import load_config
from physis.utils.run import resolve_device, setup_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="score a trained classifier, keeping logits")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=clf_scores")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log

    device = resolve_device(str(cfg.optim.device))
    bands = band_list(cfg)
    manifest = load_manifest(cfg)

    model = build_classifier(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    log.info("loaded %s", args.checkpoint)

    for split in args.splits:
        frame = manifest[manifest["split"] == split].reset_index(drop=True)
        frame = frame.assign(
            label=(frame["n_fracture_box"] > 0).astype(int),
            band=[age_band_index(float(a), bands) for a in frame["age"]],
        )
        loader = DataLoader(
            PhysisDataset(frame, cfg, augment=False),
            batch_size=int(cfg.classifier.batch_size),
            shuffle=False,
            num_workers=int(cfg.optim.num_workers),
        )
        logits = []
        with torch.no_grad():
            for batch in loader:
                meta = {
                    "age": batch["age"].to(device),
                    **{k: batch[k].to(device) for k in ("gender", "view", "laterality")},
                }
                out = model(batch["image"].to(device), batch["valid_mask"].to(device), meta)
                logits.append(out.float().cpu().numpy())
        logits = np.concatenate(logits)

        scored = frame.assign(logit=logits, score=1.0 / (1.0 + np.exp(-logits)))
        metrics = evaluate(scored, bands)
        log.info(
            "%s | image AUROC %.4f | study AUROC %.4f | distinct logits %d of %d",
            split, metrics["image_auroc"], metrics["study_auroc"],
            int(pd.Series(logits).nunique()), len(logits),
        )
        scored[["stem", "study_id", "age", "band", "label", "logit", "score"]].to_csv(
            run.dir / f"scores_{split}.csv", index=False
        )
        log.info("wrote %s", run.dir / f"scores_{split}.csv")

    print(f"RUN_DIR={run.dir}")


if __name__ == "__main__":
    main()
