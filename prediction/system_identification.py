"""Lightweight online calibration record; it never changes real-world hardware."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationState:
    landing_bias: np.ndarray
    samples: int = 0


class SystemIdentifier:
    def __init__(self) -> None:
        self.state = CalibrationState(np.zeros(3), 0)

    def update(self, predicted: list[float], actual: list[float]) -> CalibrationState:
        residual = np.asarray(actual) - np.asarray(predicted)
        alpha = min(0.18, 1 / (self.state.samples + 1))
        self.state.landing_bias = (1 - alpha) * self.state.landing_bias + alpha * residual
        self.state.samples += 1
        return self.state

    def correct(self, predicted: list[float]) -> list[float]:
        return (np.asarray(predicted) + self.state.landing_bias).round(4).tolist()
