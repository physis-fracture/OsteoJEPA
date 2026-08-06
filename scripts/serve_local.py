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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.serve.api import build_app
from physis.serve.scorer import Scorer
from physis.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="serve the triage API locally")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)

    # Loaded once at startup rather than on the first request: a demo should not
    # pay a model load in front of an audience.
    scorer = Scorer(cfg, args.checkpoint, args.calibration, device=args.device)
    print(f"loaded {args.checkpoint} on {args.device}")
    print(f"model: {scorer.model_info()}")

    import uvicorn

    uvicorn.run(build_app(lambda: scorer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
