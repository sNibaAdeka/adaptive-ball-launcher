"""Gravity belongs to MuJoCo's world model, this exposes the configured value."""
from __future__ import annotations

import numpy as np


def gravity_vector(g: float = 9.81) -> np.ndarray:
    return np.array([0.0, 0.0, -g], dtype=float)
