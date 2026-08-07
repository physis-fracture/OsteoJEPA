"""Call the running service with real studies and print what comes back.

End-to-end verification and demo rehearsal in one command. It picks studies from
the **test fold** - never seen in training - and chooses them deliberately:

* positives carry a fracture box and **no cast**, because E4a showed the model
  partly reads the cast rather than the injury
* negatives come from the strictly clean set, which carries no cast, no metal,
  no AO classification and no indirect sign of injury

Picking a casted negative would make the model look badly wrong when the fault
is in the choice of example. These are the studies to use in the video.

    python scripts/serve_local.py --checkpoint ... --calibration ...   # terminal 1
    python scripts/try_service.py                                      # terminal 2
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.dataset import pick_demo_studies
from physis.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exercise the running service")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=4, help="studies per class")
    parser.add_argument("--profile", default="triage", choices=["triage", "radiologist"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confident",
        action="store_true",
        help="pick studies the model is most sure about, for a demo rather than a check",
    )
    parser.add_argument("--offline-scores", default="artifacts/clf/clf_main/rescored/scores_test.csv")
    return parser.parse_args()


def build_payload(manifest, images_dir: Path, study_id: str, age: float, profile: str) -> dict:
    rows = manifest[manifest["study_id"] == study_id]
    images = []
    for row in rows.itertuples():
        blob = (images_dir / f"{row.stem}.png").read_bytes()
        images.append(
            {
                "image_id": row.stem,
                "content": base64.b64encode(blob).decode(),
                "view": int(row.projection),
                "laterality": str(row.laterality),
            }
        )
    return {
        "study_id": study_id,
        "profile": profile,
        "age_years": float(age),
        "sex": str(rows.iloc[0].gender),
        "images": images,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.manifest)
    images_dir = Path(cfg.images_dir)

    import urllib.error
    import urllib.request

    # 10 seconds was enough for a local process and not for Modal: the container
    # scales to zero, so the first request after an idle period waits through a
    # cold start and the check failed on a service that was working.
    try:
        with urllib.request.urlopen(args.url + "/v1/health", timeout=120) as response:
            import json as _json

            health = _json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as err:
        raise SystemExit(
            f"cannot reach {args.url}: {err}\n"
            "Local: start scripts/serve_local.py first.\n"
            "Modal: check the deployment with  modal app list"
        )
    print(f"service: {health['status']} | model: {health['model']}\n")

    chosen = pick_demo_studies(
        manifest,
        args.n,
        args.seed,
        offline_scores=args.offline_scores if args.confident else None,
    )
    if args.confident:
        print("CONFIDENT SELECTION - for a demo. Not a measurement of accuracy.\n")
    print(
        f"{'study':<12} {'age':>5} {'truth':>6} {'score':>7} {'pctile':>7} "
        f"{'band':>6} {'valid':>6} {'ms':>6}"
    )
    print("-" * 62)

    results = []
    for row in chosen.itertuples():
        payload = build_payload(manifest, images_dir, row.study_id, row.age, args.profile)
        request = urllib.request.Request(
            args.url + "/v1/score/study",
            data=__import__("json").dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = __import__("json").loads(response.read())

        valid = min(i["valid_patch_fraction"] for i in body["images"])
        results.append((row.truth, body["triage_score"]))
        print(
            f"{row.study_id:<12} {row.age:>5.1f} {row.truth:>6} "
            f"{body['triage_score']:>7.4f} {body['priority_percentile']:>7.3f} "
            f"{body['age_band']:>6} {valid:>6.3f} {body['inference_time_ms']:>6.0f}"
        )

    positives = [s for t, s in results if t == 1]
    negatives = [s for t, s in results if t == 0]
    print()
    print(f"fracture studies : median score {pd.Series(positives).median():.4f}")
    print(f"clean studies    : median score {pd.Series(negatives).median():.4f}")
    print(
        f"correct at 0.5   : {sum(s > 0.5 for s in positives)}/{len(positives)} fracture, "
        f"{sum(s <= 0.5 for s in negatives)}/{len(negatives)} clean"
    )
    # A handful of studies is a spot check, never a measurement. Printing the
    # published figure beside it stops a lucky or unlucky draw being read as one.
    print(
        f"\nThis is {len(results)} studies. Measured performance is study AUROC "
        f"0.9580 over all 2,139 test studies; see docs/RESULTS.md."
    )


if __name__ == "__main__":
    main()
