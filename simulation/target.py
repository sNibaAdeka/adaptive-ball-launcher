from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Target:
    position: np.ndarray
    radius: float = 0.12

    @classmethod
    def from_iterable(cls, values: list[float] | tuple[float, float, float], radius: float = 0.12) -> "Target":
        return cls(np.asarray(values, dtype=float), radius)

    def metrics_from(self, origin: np.ndarray) -> dict[str, float | list[float]]:
        relative = self.position - origin
        horizontal = float(np.linalg.norm(relative[:2]))
        return {"xyz": self.position.tolist(), "horizontal_distance": horizontal, "distance": float(np.linalg.norm(relative)), "height_difference": float(relative[2])}
