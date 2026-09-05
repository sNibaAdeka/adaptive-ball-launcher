from __future__ import annotations

import numpy as np


def observation(target: np.ndarray, origin: np.ndarray, *, height: float, wind: np.ndarray, air_density: float,
                ball_mass: float, drag: float, previous: dict[str, float]) -> np.ndarray:
    relative = target - origin
    values = [*relative, height, *wind, air_density, ball_mass, drag,
              previous.get("error_x", 0.0), previous.get("error_y", 0.0), previous.get("distance_error", 0.0),
              previous.get("pitch", 0.0), previous.get("yaw", 0.0), previous.get("velocity", 0.0)]
    return np.asarray(values, dtype=np.float32)
