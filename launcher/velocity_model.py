"""Replaceable virtual calibration curve; no direct motor/hardware access."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LauncherVelocityModel:
    min_velocity: float = 2.0
    max_velocity: float = 18.0
    exponent: float = 1.18

    def to_velocity(self, normalized_power: float) -> float:
        power = min(1.0, max(0.0, normalized_power))
        return self.min_velocity + (self.max_velocity - self.min_velocity) * power ** self.exponent

    def to_power(self, velocity: float) -> float:
        normalized = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity)
        return min(1.0, max(0.0, normalized)) ** (1.0 / self.exponent)
