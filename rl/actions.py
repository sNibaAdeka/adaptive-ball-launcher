from __future__ import annotations

import numpy as np


def action_bounds(launcher: dict) -> tuple[np.ndarray, np.ndarray]:
    # Order is explicitly pitch, yaw, velocity throughout the RL API.
    return (np.array([launcher["pitch_limits"][0], launcher["yaw_limits"][0], launcher["velocity_limits"][0]], dtype=np.float32),
            np.array([launcher["pitch_limits"][1], launcher["yaw_limits"][1], launcher["velocity_limits"][1]], dtype=np.float32))


def unpack_action(action: np.ndarray, height: float, spin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    return {"pitch": float(action[0]), "yaw": float(action[1]), "velocity": float(action[2]), "height": height, "spin": spin}
