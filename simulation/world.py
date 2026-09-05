"""MuJoCo-backed ball flight simulation and first-contact measurement."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import mujoco
import numpy as np

from adaptive_ball_launcher.physics.aerodynamics import drag_force
from adaptive_ball_launcher.physics.collisions import has_ball_surface_contact
from adaptive_ball_launcher.physics.magnus import magnus_force
from adaptive_ball_launcher.physics.trajectory import TrajectorySample, metrics
from adaptive_ball_launcher.physics.wind import WindField
from adaptive_ball_launcher.simulation.ball import BallProperties
from adaptive_ball_launcher.simulation.scene import make_mjcf
from adaptive_ball_launcher.simulation.target import Target


@dataclass
class ShotCommand:
    yaw: float
    pitch: float
    velocity: float
    height: float
    spin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ShotResult:
    target: list[float]
    command: dict[str, Any]
    trajectory: list[TrajectorySample]
    landing: list[float]
    contact_surface: str | None
    error: float
    reward: float
    metrics: dict[str, float]
    wind: list[float]
    air_density: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "command": self.command,
            "trajectory": [sample.as_dict() for sample in self.trajectory],
            "landing": self.landing, "contact_surface": self.contact_surface,
            "error": self.error, "reward": self.reward, "metrics": self.metrics,
            "wind": self.wind, "air_density": self.air_density,
        }


class PhysicsWorld:
    """One episode per MuJoCo model keeps reset deterministic and supports headless training."""
    def __init__(self, config: dict[str, dict], *, actual_model: bool = True, seed: int | None = None) -> None:
        self.config = config
        self.actual_model, self.rng = actual_model, np.random.default_rng(seed)
        ball_cfg, physics_cfg = config["ball"], config["physics"]
        launcher_cfg = config["launcher"]
        mass_scale = 1 + (self.rng.normal(0, launcher_cfg["ball_mass_variation"]) if actual_model else 0.0)
        drag_scale = 1 + (physics_cfg["actual_drag_bias"] + self.rng.normal(0, launcher_cfg["drag_variation"]) if actual_model else 0.0)
        self.ball = BallProperties(mass=ball_cfg["mass"] * mass_scale, radius=ball_cfg["radius"],
            drag_coefficient=ball_cfg["drag_coefficient"] * drag_scale, restitution=ball_cfg["restitution"],
            surface_friction=ball_cfg["surface_friction"], magnus_coefficient=ball_cfg["magnus_coefficient"])
        self.model = mujoco.MjModel.from_xml_string(make_mjcf(ball_cfg, config["environment"], physics_cfg, mass=self.ball.mass, drag=self.ball.drag_coefficient))
        self.data = mujoco.MjData(self.model)
        self.ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.wind = WindField(np.array(physics_cfg["wind"]), physics_cfg["wind_mode"], physics_cfg["wind_variation"], physics_cfg["gust_amplitude"], seed)
        self.air_density = physics_cfg["air_density"] * (1 + (self.rng.uniform(-0.035, 0.035) if actual_model else 0.0))

    @staticmethod
    def initial_velocity(command: ShotCommand) -> np.ndarray:
        yaw, pitch = math.radians(command.yaw), math.radians(command.pitch)
        horizontal = command.velocity * math.cos(pitch)
        return np.array([horizontal * math.sin(yaw), horizontal * math.cos(yaw), command.velocity * math.sin(pitch)], dtype=float)

    @staticmethod
    def launch_origin(command: ShotCommand) -> np.ndarray:
        yaw = math.radians(command.yaw)
        return np.array([0.18 * math.sin(yaw), -4.7 + 0.18 * math.cos(yaw), command.height + 0.10])

    def run_shot(self, command: ShotCommand, target: Target) -> ShotResult:
        mujoco.mj_resetData(self.model, self.data)
        origin, velocity = self.launch_origin(command), self.initial_velocity(command)
        self.data.qpos[:3] = origin
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:3], self.data.qvel[3:6] = velocity, np.asarray(command.spin, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        samples: list[TrajectorySample] = []
        landing: np.ndarray | None = None
        contact_surface: str | None = None
        max_steps = int(self.config["physics"]["max_flight_time"] / self.model.opt.timestep)
        stride = max(1, round(self.config["physics"]["sample_interval"] / self.model.opt.timestep))
        for step in range(max_steps):
            velocity = self.data.qvel[:3].copy()
            angular = self.data.qvel[3:6].copy()
            wind = self.wind.velocity_at(float(self.data.time))
            drag = drag_force(velocity, wind, self.air_density, self.ball.drag_coefficient, self.ball.area)
            magnus = magnus_force(velocity, wind, angular, self.air_density, self.ball.radius, self.ball.magnus_coefficient, self.config["physics"]["magnus_enabled"])
            self.data.xfrc_applied[self.ball_body, :3] = drag + magnus
            self.data.xfrc_applied[self.ball_body, 3:] = 0
            mujoco.mj_step(self.model, self.data)
            position = self.data.qpos[:3].copy()
            if step % stride == 0:
                samples.append(TrajectorySample(float(self.data.time), position, self.data.qvel[:3].copy(), {
                    "gravity": np.array([0.0, 0.0, -self.ball.mass * self.config["physics"]["gravity"]]), "drag": drag, "magnus": magnus, "wind": wind,
                }))
            contacted, surface = has_ball_surface_contact(self.model, self.data)
            if contacted and self.data.time > 0.035:
                landing, contact_surface = position.copy(), surface
                break
            if abs(position[0]) > self.config["environment"]["arena_width"] / 2 + 1 or abs(position[1]) > self.config["environment"]["arena_length"] / 2 + 1:
                landing, contact_surface = position.copy(), "out_of_arena"
                break
        if landing is None:
            landing, contact_surface = self.data.qpos[:3].copy(), "timeout"
        error = float(np.linalg.norm((landing - target.position)[:2]) + abs(landing[2] - target.position[2]) * 0.4)
        reward = -error + self._reward_bonus(error) - (3.0 if contact_surface in {"out_of_arena", "timeout"} else 0.0)
        return ShotResult(target.position.round(4).tolist(), asdict(command), samples, landing.round(4).tolist(), contact_surface, round(error, 4), round(reward, 4), metrics(samples), self.wind.velocity_at(float(self.data.time)).round(4).tolist(), round(float(self.air_density), 4))

    @staticmethod
    def _reward_bonus(error: float) -> float:
        return 10.0 if error < .05 else 6.0 if error < .10 else 3.0 if error < .20 else 1.5 if error < .50 else .5 if error < 1.0 else 0.0
