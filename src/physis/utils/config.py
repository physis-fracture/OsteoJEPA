"""Config loading.

One rule from CONVENTIONS: resolve the config once at process start, write the
resolved version to the run directory, and pass the resolved object down. No
module reads YAML on its own.

A config file may declare `defaults: [path, ...]`, where each path is relative
to the declaring file. Parents are merged first, so the declaring file only ever
contains overrides. That is what makes E1b's "identical configuration except for
the condition injection point" claim checkable by diffing one small file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def _load_with_defaults(path: Path, _seen: set[Path] | None = None) -> DictConfig:
    """Load one YAML file and merge its `defaults:` parents underneath it."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config defaults at {path}")
    _seen = _seen | {path}

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    node = OmegaConf.load(path)
    if not isinstance(node, DictConfig):
        raise TypeError(f"config root must be a mapping: {path}")

    parents = node.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for parent in parents:
        merged = OmegaConf.merge(merged, _load_with_defaults(path.parent / str(parent), _seen))
    return OmegaConf.merge(merged, node)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Resolve a config file plus optional `key.sub=value` command-line overrides.

    Interpolations such as ${run.name} are resolved eagerly, so the object handed
    to the rest of the process holds plain values and cannot drift when a field
    is mutated later.
    """
    cfg = _load_with_defaults(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    Path(path).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
