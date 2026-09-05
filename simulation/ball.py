from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BallProperties:
    mass: float
    radius: float
    drag_coefficient: float
    restitution: float
    surface_friction: float
    magnus_coefficient: float

    @property
    def area(self) -> float:
        return float(np.pi * self.radius ** 2)
