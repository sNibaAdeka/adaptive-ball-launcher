"""Physics-in-the-loop trajectory predictor used before every interactive shot."""
from __future__ import annotations

import math
import numpy as np

from adaptive_ball_launcher.simulation.target import Target
from adaptive_ball_launcher.simulation.world import PhysicsWorld, ShotCommand, ShotResult


class TrajectoryPredictor:
    def __init__(self, config: dict, seed: int = 7) -> None:
        self.config = config
        self.seed = seed

    def estimate_command(self, target: Target, height: float, spin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> ShotCommand:
        origin = np.array([0.0, -4.52, height + 0.1])
        d = target.position - origin
        horizontal, dz = float(np.linalg.norm(d[:2])), float(d[2])
        yaw = math.degrees(math.atan2(d[0], d[1]))
        limits = self.config["launcher"]["velocity_limits"]
        candidates: list[ShotCommand] = []
        # Ballistic candidates are cheap initial guesses; MuJoCo then picks the least-error flight.
        for pitch in np.linspace(20.0, 68.0, 13):
            theta = math.radians(pitch)
            denominator = 2 * (math.cos(theta) ** 2) * (horizontal * math.tan(theta) - dz)
            if denominator <= 0:
                continue
            v2 = self.config["physics"]["gravity"] * horizontal**2 / denominator
            if v2 <= 0:
                continue
            velocity = math.sqrt(v2)
            # Ballistic estimate omits drag; examine a tight speed fan in the same MuJoCo model.
            for scale in (.78, .90, 1.02, 1.14, 1.28, 1.42):
                candidate_velocity = velocity * scale
                if limits[0] <= candidate_velocity <= limits[1]:
                    candidates.append(ShotCommand(yaw, pitch, candidate_velocity, height, spin))
        if not candidates:
            candidates = [ShotCommand(yaw, 45.0, min(limits[1], max(limits[0], 8.0)), height, spin)]
        world = PhysicsWorld(self.config, actual_model=False, seed=self.seed)
        best = min(((world.run_shot(command, target).error, command) for command in candidates), key=lambda item: item[0])
        # A local velocity sweep removes the coarse-grid bias without taking over the physics engine.
        coarse = best[1]
        refined = [ShotCommand(coarse.yaw, coarse.pitch, coarse.velocity + offset, height, spin) for offset in np.linspace(-.72, .72, 17)
                   if limits[0] <= coarse.velocity + offset <= limits[1]]
        return min(((world.run_shot(command, target).error, command) for command in refined), key=lambda item: item[0])[1]

    def predict(self, command: ShotCommand, target: Target) -> ShotResult:
        return PhysicsWorld(self.config, actual_model=False, seed=self.seed).run_shot(command, target)
