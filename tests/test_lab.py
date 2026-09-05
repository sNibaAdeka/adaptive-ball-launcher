from __future__ import annotations

import tempfile
import unittest

import numpy as np

from adaptive_ball_launcher.actuators.base import VirtualActuator
from adaptive_ball_launcher.config import load_all
from adaptive_ball_launcher.logging.shot_logger import ShotLogger
from adaptive_ball_launcher.physics.aerodynamics import drag_force
from adaptive_ball_launcher.physics.magnus import magnus_force
from adaptive_ball_launcher.prediction.trajectory_predictor import TrajectoryPredictor
from adaptive_ball_launcher.rl.ai_policy import InteractiveAIPolicy
from adaptive_ball_launcher.rl.env import BallLauncherEnv
from adaptive_ball_launcher.simulation.target import Target
from adaptive_ball_launcher.simulation.world import PhysicsWorld, ShotCommand


class LabValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_all()

    def test_actuator_moves_gradually_then_settles(self) -> None:
        servo = VirtualActuator(initial=0, minimum=-90, maximum=90, max_velocity=20, max_acceleration=100, response_delay=.02)
        servo.set_target(20)
        first = servo.update(.01, noisy=False)
        self.assertEqual(first.current_position, 0)
        servo.update(.01, noisy=False)
        moved = servo.update(.1, noisy=False)
        self.assertGreater(moved.current_position, 0)
        self.assertLess(moved.current_position, 20)
        for _ in range(60): servo.update(.05, noisy=False)
        self.assertTrue(servo.is_ready)
        self.assertAlmostEqual(servo.current_position, 20)

    def test_drag_opposes_relative_air_velocity(self) -> None:
        force = drag_force(np.array([4., -2., 0.]), np.array([1., 0., 0.]), 1.225, .47, .001256)
        relative = np.array([3., -2., 0.])
        self.assertLess(float(np.dot(force, relative)), 0)

    def test_magnus_can_be_disabled(self) -> None:
        kwargs = dict(velocity=np.array([4., 3., 0.]), wind_velocity=np.zeros(3), angular_velocity=np.array([0., 0., 80.]), rho=1.225, radius=.02, coefficient=.000018)
        self.assertTrue(np.linalg.norm(magnus_force(**kwargs, enabled=True)) > 0)
        self.assertTrue(np.allclose(magnus_force(**kwargs, enabled=False), 0))

    def test_mujoco_shot_records_first_landing_without_nan(self) -> None:
        target = Target.from_iterable([0, 1.5, .02])
        command = TrajectoryPredictor(self.config).estimate_command(target, .62)
        result = PhysicsWorld(self.config, actual_model=False, seed=4).run_shot(command, target)
        self.assertIn(result.contact_surface, {"floor", "low_platform", "high_platform", "bridge_platform", "out_of_arena", "timeout"})
        self.assertGreater(len(result.trajectory), 3)
        self.assertTrue(np.isfinite(result.error))
        self.assertTrue(np.isfinite(np.asarray([item.position for item in result.trajectory])).all())

    def test_interactive_ai_has_a_bounded_safe_fallback(self) -> None:
        target = Target.from_iterable([0, 1.5, .02])
        policy = InteractiveAIPolicy(self.config, TrajectoryPredictor(self.config))
        command = policy.decide(target, .62, (0, 0, 0))
        self.assertEqual(policy.source, "PHYSICS-AWARE")
        self.assertGreaterEqual(command.velocity, self.config["launcher"]["velocity_limits"][0])
        self.assertLessEqual(command.velocity, self.config["launcher"]["velocity_limits"][1])

    def test_gym_observation_and_action_contract(self) -> None:
        env = BallLauncherEnv(self.config, curriculum_level=1, seed=11)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (16,))
        self.assertTrue(env.observation_space.contains(observation))
        next_observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        self.assertTrue(terminated); self.assertFalse(truncated)
        self.assertTrue(np.isfinite(next_observation).all()); self.assertTrue(np.isfinite(reward))
        self.assertIn("landing_error", info)

    def test_shot_logger_persists_replay_data(self) -> None:
        target = Target.from_iterable([0, .5, .02])
        shot = PhysicsWorld(self.config, actual_model=False, seed=8).run_shot(ShotCommand(0, 35, 9, .62), target).as_dict()
        with tempfile.TemporaryDirectory() as tmp:
            logger = ShotLogger(tmp); record = logger.append(result=shot, predicted=shot)
            self.assertTrue(logger.csv_path.exists()); self.assertTrue(logger.json_path.exists())
            self.assertEqual(len(logger.history()), 1)
            self.assertIn("actual_trajectory", record)


if __name__ == "__main__":
    unittest.main()
