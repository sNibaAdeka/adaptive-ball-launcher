"""Safe simulation controller interface; a hardware adapter can be added later."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math

from adaptive_ball_launcher.actuators.base import VirtualActuator
from adaptive_ball_launcher.actuators.yaw import YawActuator
from adaptive_ball_launcher.actuators.pitch import PitchActuator
from adaptive_ball_launcher.actuators.height import HeightActuator
from adaptive_ball_launcher.launcher.velocity_model import LauncherVelocityModel
from adaptive_ball_launcher.launcher.state_machine import LauncherStateMachine


class LauncherController(ABC):
    @abstractmethod
    def set_yaw(self, degrees: float) -> None: ...
    @abstractmethod
    def set_pitch(self, degrees: float) -> None: ...
    @abstractmethod
    def set_height(self, metres: float) -> None: ...
    @abstractmethod
    def set_launch_parameter(self, power: float) -> None: ...
    @abstractmethod
    def get_state(self) -> dict: ...
    @abstractmethod
    def is_ready(self) -> bool: ...
    @abstractmethod
    def launch(self) -> dict: ...


class SimulatedLauncherController(LauncherController):
    def __init__(self, config: dict) -> None:
        delay, noise, backlash = config["response_delay"], config["servo_position_noise"], config["gear_backlash"]
        self.yaw = YawActuator(initial=config["default_yaw"], minimum=config["yaw_limits"][0], maximum=config["yaw_limits"][1],
            max_velocity=config["max_yaw_velocity"], max_acceleration=config["max_yaw_acceleration"], response_delay=delay, position_noise=noise, backlash=backlash, label="yaw")
        self.pitch = PitchActuator(initial=config["default_pitch"], minimum=config["pitch_limits"][0], maximum=config["pitch_limits"][1],
            max_velocity=config["max_pitch_velocity"], max_acceleration=config["max_pitch_acceleration"], response_delay=delay, position_noise=noise, backlash=backlash, label="pitch")
        self.height = HeightActuator(initial=config["default_height"], minimum=config["height_limits"][0], maximum=config["height_limits"][1],
            max_velocity=config["max_height_velocity"], max_acceleration=config["max_height_acceleration"], response_delay=delay, position_noise=noise / 100, backlash=backlash / 100, label="height")
        self.velocity_model = LauncherVelocityModel(*config["velocity_limits"])
        self.power = 0.5
        self.machine = LauncherStateMachine()

    def set_yaw(self, degrees: float) -> None: self.yaw.set_target(degrees)
    def set_pitch(self, degrees: float) -> None: self.pitch.set_target(degrees)
    def set_height(self, metres: float) -> None: self.height.set_target(metres)
    def set_launch_parameter(self, power: float) -> None: self.power = min(1.0, max(0.0, power))

    def update(self, dt: float, noisy: bool = False) -> None:
        self.yaw.update(dt, noisy); self.pitch.update(dt, noisy); self.height.update(dt, noisy)
        if self.machine.state.value == "AIMING" and self.is_ready(): self.machine.ready()

    def is_ready(self) -> bool: return self.yaw.is_ready and self.pitch.is_ready and self.height.is_ready

    def wait_until_ready(self, dt: float = 0.01, timeout: float = 4.0) -> None:
        self.machine.aim()
        elapsed = 0.0
        while not self.is_ready() and elapsed < timeout:
            self.update(dt)
            elapsed += dt
        if not self.is_ready(): raise TimeoutError("Virtual launcher did not reach READY state")
        self.machine.ready()

    def launch(self) -> dict:
        if not self.is_ready(): raise RuntimeError("Launcher is not ready")
        self.machine.launch()
        return {"yaw": self.yaw.current_position, "pitch": self.pitch.current_position, "height": self.height.current_position,
                "power": self.power, "velocity": self.velocity_model.to_velocity(self.power)}

    def launch_origin(self) -> list[float]:
        yaw = math.radians(self.yaw.current_position)
        return [0.18 * math.sin(yaw), -4.7 + 0.18 * math.cos(yaw), self.height.current_position + 0.10]

    def get_state(self) -> dict:
        return {"state": self.machine.state.value, "yaw": self.yaw.state.__dict__, "pitch": self.pitch.state.__dict__, "height": self.height.state.__dict__, "power": self.power, "ready": self.is_ready()}
