"""Smooth temporal wind fields — deliberately not frame-to-frame white noise."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class WindField:
    base: np.ndarray
    mode: str = "no_wind"
    variation: float = 0.0
    gust_amplitude: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.base = np.asarray(self.base, dtype=float)
        self._rng = np.random.default_rng(self.seed)
        self._phase = self._rng.uniform(0, math.tau, size=3)

    def velocity_at(self, time_s: float) -> np.ndarray:
        if self.mode == "no_wind":
            return np.zeros(3)
        if self.mode == "constant":
            return self.base.copy()
        low_frequency = self.variation * np.sin(0.67 * time_s + self._phase)
        gust = np.zeros(3)
        if self.mode == "gusts":
            envelope = max(0.0, math.sin(0.17 * time_s + self._phase[0])) ** 8
            gust = envelope * self.gust_amplitude * np.array([math.cos(self._phase[1]), math.sin(self._phase[1]), 0.08])
        return self.base + low_frequency + gust

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.base))
