"""Interactive decision adapter: uses a saved SAC/PPO policy when available."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from adaptive_ball_launcher.rl.observations import observation
from adaptive_ball_launcher.simulation.target import Target
from adaptive_ball_launcher.simulation.world import ShotCommand


class InteractiveAIPolicy:
    def __init__(self, config: dict, predictor) -> None:
        self.config, self.predictor, self.model = config, predictor, None
        self.source = "PHYSICS-AWARE"
        algorithm = config["algorithm"]["algorithm"].lower()
        candidate = Path(__file__).resolve().parents[1] / "models" / f"{algorithm}_launcher.zip"
        if candidate.exists():
            try:
                from stable_baselines3 import SAC, PPO
                self.model = (SAC if algorithm == "sac" else PPO).load(candidate)
                self.source = f"{algorithm.upper()} POLICY"
            except Exception:
                # A compatible saved policy is optional and never prevents safe simulation use.
                self.model = None

    def decide(self, target: Target, height: float, spin: tuple[float, float, float]) -> ShotCommand:
        if self.model is None:
            return self.predictor.estimate_command(target, height, spin)
        origin = np.array([0.0, -4.52, height + .10])
        wind = np.asarray(self.config["physics"]["wind"], dtype=float)
        ball = self.config["ball"]
        obs = observation(target.position, origin, height=height, wind=wind, air_density=self.config["physics"]["air_density"],
                          ball_mass=ball["mass"], drag=ball["drag_coefficient"], previous={})
        action, _ = self.model.predict(obs, deterministic=True)
        low = self.config["launcher"]["pitch_limits"][0], self.config["launcher"]["yaw_limits"][0], self.config["launcher"]["velocity_limits"][0]
        high = self.config["launcher"]["pitch_limits"][1], self.config["launcher"]["yaw_limits"][1], self.config["launcher"]["velocity_limits"][1]
        pitch, yaw, velocity = np.clip(np.asarray(action, dtype=float), low, high)
        return ShotCommand(float(yaw), float(pitch), float(velocity), height, spin)
