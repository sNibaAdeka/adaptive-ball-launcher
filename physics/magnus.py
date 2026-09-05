"""A stable, deliberately conservative Magnus approximation for a ping-pong ball."""
from __future__ import annotations

import numpy as np


def magnus_force(velocity: np.ndarray, wind_velocity: np.ndarray, angular_velocity: np.ndarray,
                 rho: float, radius: float, coefficient: float, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.zeros(3)
    relative = np.asarray(velocity, dtype=float) - np.asarray(wind_velocity, dtype=float)
    omega = np.asarray(angular_velocity, dtype=float)
    # Cross-product direction gives lift/side force; clamp avoids unstable experimental spin.
    force = coefficient * rho * (radius ** 3) * np.cross(omega, relative)
    magnitude = np.linalg.norm(force)
    return force if magnitude <= 0.12 else force * (0.12 / magnitude)
