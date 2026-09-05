"""Finite-acceleration virtual servo, intentionally separate from rendering."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class ActuatorState:
    current_position: float
    target_position: float
    velocity: float
    ready: bool


class VirtualActuator:
    def __init__(self, *, initial: float, minimum: float, maximum: float, max_velocity: float,
                 max_acceleration: float, response_delay: float = 0.0, position_noise: float = 0.0,
                 backlash: float = 0.0, label: str = "actuator") -> None:
        self.minimum, self.maximum = minimum, maximum
        self.max_velocity, self.max_acceleration = max_velocity, max_acceleration
        self.response_delay, self.position_noise, self.backlash = response_delay, position_noise, backlash
        self.label = label
        self.current_position = self.target_position = initial
        self.velocity = 0.0
        self._delay_remaining = 0.0

    def set_target(self, position: float) -> float:
        self.target_position = max(self.minimum, min(self.maximum, position))
        self._delay_remaining = self.response_delay
        return self.target_position

    def reset(self, position: float) -> None:
        self.current_position = self.target_position = position
        self.velocity = self._delay_remaining = 0.0

    def update(self, dt: float, noisy: bool = True) -> ActuatorState:
        if self._delay_remaining > 0:
            self._delay_remaining = max(0.0, self._delay_remaining - dt)
            return self.state
        delta = self.target_position - self.current_position
        if abs(delta) <= max(0.001, self.backlash):
            self.current_position = self.target_position
            self.velocity = 0.0
        else:
            desired_velocity = math.copysign(self.max_velocity, delta)
            velocity_step = self.max_acceleration * dt
            self.velocity += max(-velocity_step, min(velocity_step, desired_velocity - self.velocity))
            next_position = self.current_position + self.velocity * dt
            if (delta > 0 and next_position >= self.target_position) or (delta < 0 and next_position <= self.target_position):
                self.current_position, self.velocity = self.target_position, 0.0
            else:
                self.current_position = next_position
        if noisy and self.position_noise:
            measured = self.current_position + random.gauss(0.0, self.position_noise)
        else:
            measured = self.current_position
        return ActuatorState(measured, self.target_position, self.velocity, self.is_ready)

    @property
    def is_ready(self) -> bool:
        return self._delay_remaining <= 0 and abs(self.target_position - self.current_position) < 0.002 and abs(self.velocity) < 0.01

    @property
    def state(self) -> ActuatorState:
        return ActuatorState(self.current_position, self.target_position, self.velocity, self.is_ready)
