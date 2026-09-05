from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from adaptive_ball_launcher.physics.wind import WindField
from adaptive_ball_launcher.rl.actions import action_bounds, unpack_action
from adaptive_ball_launcher.rl.observations import observation
from adaptive_ball_launcher.rl.randomization import random_target, randomized_config
from adaptive_ball_launcher.simulation.target import Target
from adaptive_ball_launcher.simulation.world import PhysicsWorld, ShotCommand


class BallLauncherEnv(gym.Env):
    """Single-shot continuous-control environment. Rendering is intentionally disabled for training."""
    metadata = {"render_modes": []}

    def __init__(self, config: dict, curriculum_level: int = 1, seed: int | None = None) -> None:
        super().__init__()
        self.base_config, self.level = config, curriculum_level
        self.rng = np.random.default_rng(seed)
        low, high = action_bounds(config["launcher"])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.previous: dict[str, float] = {}
        self.world: PhysicsWorld | None = None
        self.target: Target | None = None
        self.episode_config: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode_config = randomized_config(self.base_config, self.level, self.rng)
        self.world = PhysicsWorld(self.episode_config, actual_model=True, seed=int(self.rng.integers(2**31)))
        self.target = Target(random_target(self.episode_config, self.level, self.rng), self.episode_config["environment"]["target_radius"])
        self.previous = {}
        return self._observation(), {"target": self.target.position.tolist(), "curriculum_level": self.level}

    def step(self, action: np.ndarray):
        if self.world is None or self.target is None: raise RuntimeError("Call reset before step")
        bounded = np.clip(action, self.action_space.low, self.action_space.high)
        values = unpack_action(bounded, self.episode_config["launcher"]["default_height"])
        result = self.world.run_shot(ShotCommand(**values), self.target)
        landing = np.asarray(result.landing)
        error_vec = landing - self.target.position
        self.previous = {"error_x": float(error_vec[0]), "error_y": float(error_vec[1]), "distance_error": result.error,
                         "pitch": values["pitch"], "yaw": values["yaw"], "velocity": values["velocity"]}
        return self._observation(), result.reward, True, False, {"landing_error": result.error, "success": result.error < .2, "shot": result.as_dict()}

    def _observation(self) -> np.ndarray:
        assert self.world and self.target and self.episode_config
        height = self.episode_config["launcher"]["default_height"]
        origin = np.array([0.0, -4.52, height + .10])
        wind = self.world.wind.velocity_at(0.0)
        return observation(self.target.position, origin, height=height, wind=wind, air_density=self.world.air_density,
                           ball_mass=self.world.ball.mass, drag=self.world.ball.drag_coefficient, previous=self.previous)
