"""Call the running service with real studies and print what comes back.

End-to-end verification and demo rehearsal in one command. It picks studies from
the **test fold** - never seen in training - and chooses them deliberately:

* positives carry a fracture box and **no cast**, because E4a showed the model
  partly reads the cast rather than the injury
* negatives come from the strictly clean set, which carries no cast, no metal,
  no AO classification and no indirect sign of injury

Picking a casted negative would make the model look badly wrong when the fault
is in the choice of example. These are the studies to use in the video.

    python scripts/serve_local.py                # terminal 1
    python scripts/try_service.py                # terminal 2

`/v1/predict` takes presigned URLs rather than bytes, which is the arrangement
that keeps bucket credentials out of the service. A laptop cannot produce one,
so this serves the images over a throwaway local http server and the service is
told to accept loopback. That is why `serve_local.py` sets
PHYSIS_ALLOW_LOOPBACK_FETCH; a Modal deployment never does, and testing against
one needs images that are actually reachable over https.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confident",
        action="store_true",
        help="pick studies the model is most sure about, for a demo rather than a check",
    )
    parser.add_argument("--offline-scores", default="artifacts/clf/clf_main/rescored/scores_test.csv")
    parser.add_argument(
        "--token",
        default=None,
        help="bearer token, if the service requires one; defaults to PHYSIS_API_KEY",
    )
    return parser.parse_args()


def serve_directory(directory: Path):
    """A throwaway http server standing in for presigned R2 URLs."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    # Quiet: one request line per image would bury the results table.
    handler.log_message = lambda *args, **kwargs: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


VIEW_NAME = {1: "PA", 2: "LATERAL", 3: "OTHER"}
LATERALITY_NAME = {"L": "left", "R": "right"}
SEX_NAME = {"M": "male", "F": "female"}


def build_payload(manifest, study_id: str, age: float, base_url: str) -> dict:
    rows = manifest[manifest["study_id"] == study_id]
    return {
        "study_id": study_id,
        "age_years": round(float(age), 1),
        "sex": SEX_NAME.get(str(rows.iloc[0].gender), "unknown"),
        "images": [
            {
                "image_id": row.stem,
                "image_url": f"{base_url}/{row.stem}.png",
                "view": VIEW_NAME.get(int(row.projection), "UNKNOWN"),
                "laterality": LATERALITY_NAME.get(str(row.laterality), "unknown"),
            }
            for row in rows.itertuples()
        ],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.manifest)
    images_dir = Path(cfg.images_dir)

    # 10 seconds was enough for a local process and not for Modal: the container
    # scales to zero, so the first request after an idle period waits through a
    # cold start and the check failed on a service that was working.
    try:
        with urllib.request.urlopen(args.url + "/v1/health", timeout=120) as response:
            health = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as err:
        raise SystemExit(
            f"cannot reach {args.url}: {err}\n"
            "Local: start scripts/serve_local.py first.\n"
            "Modal: check the deployment with  modal app list"
        )
    print(f"service: {health['status']} | contract {health['contract_version']}\n")

    token = args.token if args.token is not None else os.environ.get("PHYSIS_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token}"

    chosen = pick_demo_studies(
        manifest,
        args.n,
        args.seed,
        offline_scores=args.offline_scores if args.confident else None,
    )
    if args.confident:
        print("CONFIDENT SELECTION - for a demo. Not a measurement of accuracy.\n")

    staging = Path(tempfile.mkdtemp(prefix="physis_try_"))
    for study in chosen.itertuples():
        for row in manifest[manifest["study_id"] == study.study_id].itertuples():
            shutil.copyfile(images_dir / f"{row.stem}.png", staging / f"{row.stem}.png")
    server, base_url = serve_directory(staging)

    print(
        f"{'study':<12} {'age':>5} {'truth':>6} {'score':>7} {'pctile':>7} "
        f"{'band':>6} {'boxes':>6} {'ms':>6}"
    )
    print("-" * 62)

    results = []
    try:
        for row in chosen.itertuples():
            payload = build_payload(manifest, row.study_id, row.age, base_url)
            request = urllib.request.Request(
                args.url + "/v1/predict", data=json.dumps(payload).encode(), headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read())
            except urllib.error.HTTPError as err:
                body = json.loads(err.read())

            if not body.get("success"):
                print(f"{row.study_id:<12} FAILED {body['error_code']}: {body['message']}")
                continue

            data = body["data"]
            boxes = sum(len(i["boxes"] or []) for i in data["images"])
            results.append((row.truth, data["triage_score"]))
            print(
                f"{row.study_id:<12} {row.age:>5.1f} {row.truth:>6} "
                f"{data['triage_score']:>7.4f} {data['priority_percentile']:>7.1f} "
                f"{data['age_band']:>6} {boxes:>6} {data['inference_time_ms']:>6}"
            )
    finally:
        server.shutdown()
        shutil.rmtree(staging, ignore_errors=True)

    if not results:
        raise SystemExit("no study scored; see the failures above")

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
        f"0.9580 over all 2,139 test studies."
    )


if __name__ == "__main__":
    main()
