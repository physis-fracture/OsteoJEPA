"""Run directories, seeding, logging.

Every run writes to runs/<name>/: resolved config, git commit hash, seed,
metrics as JSON, checkpoints, figures, log. When the paper tables get filled in,
provenance is what lets a number be trusted.

Existing run directories are never overwritten; a timestamp suffix is added
instead.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from .config import save_config

LOGGER_NAME = "physis"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch from the config."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_state() -> str:
    """Commit hash and dirty flag, or a marker when git is unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git must not kill a training run
        return "commit=unknown\ndirty=unknown\n"
    return f"commit={commit}\ndirty={'yes' if status else 'no'}\n"


@dataclass
class RunContext:
    """Handle on one run directory."""

    dir: Path
    cfg: DictConfig
    log: logging.Logger

    @property
    def checkpoints(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def figures(self) -> Path:
        return self.dir / "figures"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.json"

    def append_metrics(self, record: dict) -> None:
        """Append one record to metrics.json, rewriting the whole list.

        Runs here are short enough that rewriting beats managing a JSONL reader
        on the analysis side.
        """
        history = []
        if self.metrics_path.exists():
            history = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        history.append(record)
        self.metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def setup_run(cfg: DictConfig, *, subdir: str | None = None) -> RunContext:
    """Create runs/<name>/ and record everything needed to reproduce it."""
    base = Path(cfg.run.out_dir)
    if subdir:
        base = base / subdir
    if base.exists():
        base = base.with_name(f"{base.name}_{datetime.now():%Y%m%d_%H%M%S}")
    base.mkdir(parents=True)
    (base / "checkpoints").mkdir()
    (base / "figures").mkdir()

    save_config(cfg, base / "config.resolved.yaml")
    (base / "git.txt").write_text(git_state(), encoding="utf-8")
    (base / "seed.txt").write_text(f"{cfg.run.seed}\n", encoding="utf-8")

    log = _setup_logger(base / "log.txt")
    set_seed(int(cfg.run.seed))
    log.info("run dir: %s", base)
    log.info("seed: %s", cfg.run.seed)
    return RunContext(dir=base, cfg=cfg, log=log)


def _setup_logger(log_path: Path) -> logging.Logger:
    """Log to stdout and to runs/<name>/log.txt."""
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)

    file = logging.FileHandler(log_path, encoding="utf-8")
    file.setFormatter(fmt)
    log.addHandler(file)
    return log


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
