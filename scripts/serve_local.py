"""Run the triage service on this machine. No Modal, no GPU.

The service is one forward pass of a ViT-S, so a laptop CPU is enough. Modal
buys a public URL and scale-to-zero; it is not a requirement, and for recording
a demo a local process is better because there is no cold start to wait through.

    python scripts/serve_local.py \\
        --checkpoint artifacts/clf/clf_main/best.pt \\
        --calibration artifacts/clf/clf_main/classifier_calibration.json

Then: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.serve.api import build_app
from physis.serve.scorer import Scorer
from physis.utils.config import load_config

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULTS = {
    "checkpoint": ARTIFACTS / "clf" / "clf_main" / "best.pt",
    "calibration": ARTIFACTS / "clf" / "clf_main" / "classifier_calibration.json",
    "detector": ARTIFACTS / "det" / "det_main" / "best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="serve the triage API locally")
    parser.add_argument("--config", default="configs/base.yaml")
    # Defaults point at the artifacts directory, so the fallback is one short
    # command rather than three paths typed correctly under pressure.
    parser.add_argument("--checkpoint", default=str(DEFAULTS["checkpoint"]))
    parser.add_argument("--calibration", default=str(DEFAULTS["calibration"]))
    parser.add_argument("--detector", default=str(DEFAULTS["detector"]),
                        help="box detector checkpoint; pass '' to serve without one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--token",
        default=None,
        help="require this bearer token on the scoring endpoints; "
             "defaults to PHYSIS_API_KEY, unset means open",
    )
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)

    # Loaded once at startup rather than on the first request: a demo should not
    # pay a model load in front of an audience.
    for label, path in (("checkpoint", args.checkpoint), ("calibration", args.calibration)):
        if not Path(path).exists():
            raise SystemExit(
                f"{label} not found: {path}\n"
                "Pull it first:  modal volume get physis-runs "
                "/clf_main/checkpoints/best.pt artifacts/clf/clf_main/"
            )
    detector = args.detector if args.detector and Path(args.detector).exists() else None
    if args.detector and detector is None:
        print(f"no detector at {args.detector}; serving without localization")

    # The auth dependency reads the environment, so --token is a convenience
    # over exporting it, not a second mechanism.
    if args.token is not None:
        os.environ["PHYSIS_API_KEY"] = args.token

    # /v1/predict fetches presigned https URLs, which a laptop cannot mint. The
    # local service accepts plain http to loopback so try_service.py can serve
    # test images from a throwaway server. Loopback only: the addresses an SSRF
    # attempt wants are link-local and private, and both stay blocked. The Modal
    # deployment never sets this.
    os.environ.setdefault("PHYSIS_ALLOW_LOOPBACK_FETCH", "1")

    scorer = Scorer(cfg, args.checkpoint, args.calibration, device=args.device,
                    detector_checkpoint=detector)
    print(f"loaded {args.checkpoint} on {args.device}")
    print(f"model: {scorer.model_info()}")
    auth = "bearer token required" if os.environ.get("PHYSIS_API_KEY", "").strip() else "open"
    print(f"listening on http://{args.host}:{args.port}  (docs at /docs, {auth})")

    import uvicorn

    uvicorn.run(build_app(lambda: scorer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
