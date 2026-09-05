"""Air drag in the body/world force application layer."""
from __future__ import annotations

import numpy as np


def drag_force(velocity: np.ndarray, wind_velocity: np.ndarray, rho: float, cd: float, area: float) -> np.ndarray:
    relative = np.asarray(velocity, dtype=float) - np.asarray(wind_velocity, dtype=float)
    speed = float(np.linalg.norm(relative))
    return -0.5 * rho * cd * area * speed * relative
