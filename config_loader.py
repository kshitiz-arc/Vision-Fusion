"""
VisionFusion AI — Configuration Loader
========================================
Loads, merges, and validates YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Hierarchical YAML configuration manager.

    Supports loading a *base* config and then selectively overriding
    values from an *experiment* config file or a flat ``dict``.

    Example
    -------
    >>> cfg = ConfigLoader.load("configs/default.yaml")
    >>> cfg["input"]["mode"]
    'webcam'
    """

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a single YAML file and return it as a nested dict."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as fh:
            return yaml.safe_load(fh) or {}

    @staticmethod
    def merge(base: dict, overrides: dict) -> dict:
        """Deep-merge *overrides* into *base* (modifies *base* in-place)."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader.merge(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def load_with_overrides(base_path: str | Path,
                            override_path: str | Path | None = None,
                            overrides: dict | None = None) -> dict[str, Any]:
        """Load base config and apply optional file + dict overrides."""
        cfg = ConfigLoader.load(base_path)
        if override_path:
            cfg = ConfigLoader.merge(cfg, ConfigLoader.load(override_path))
        if overrides:
            cfg = ConfigLoader.merge(cfg, overrides)
        return cfg

    @staticmethod
    def save(cfg: dict, path: str | Path) -> None:
        """Persist a config dict to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
