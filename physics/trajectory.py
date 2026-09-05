from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrajectorySample:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    forces: dict[str, np.ndarray] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "time": round(self.time, 4),
            "position": [round(float(v), 4) for v in self.position],
            "velocity": [round(float(v), 4) for v in self.velocity],
            "forces": {key: [round(float(v), 5) for v in value] for key, value in self.forces.items()},
        }


def metrics(samples: list[TrajectorySample]) -> dict[str, float]:
    if not samples:
        return {"flight_time": 0.0, "maximum_height": 0.0, "maximum_speed": 0.0}
    positions = np.array([sample.position for sample in samples])
    velocities = np.array([sample.velocity for sample in samples])
    return {
        "flight_time": float(samples[-1].time),
        "maximum_height": float(positions[:, 2].max()),
        "maximum_speed": float(np.linalg.norm(velocities, axis=1).max()),
    }
