"""Typed, file-backed configuration for the lab."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


def load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_all() -> dict[str, dict[str, Any]]:
    return {name.removesuffix(".yaml"): load_config(name) for name in (
        "ball.yaml", "launcher.yaml", "physics.yaml", "environment.yaml", "rl.yaml", "ui.yaml", "algorithm.yaml"
    )}
