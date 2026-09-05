"""Headless SAC/PPO training entry point."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from adaptive_ball_launcher.logging.training_logger import TrainingLogger
from adaptive_ball_launcher.rl.env import BallLauncherEnv
from adaptive_ball_launcher.rl.evaluate import evaluate_policy


class AnalyticsCallback(BaseCallback):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.training_logger = TrainingLogger(directory)
        self.episode = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "landing_error" not in info:
                continue
            self.episode += 1
            shot = info.get("shot", {})
            target = shot.get("target", [0, 0, 0])
            self.training_logger.log(episode=self.episode, reward=float(self.locals["rewards"][0]), landing_error=info["landing_error"],
                            success=float(info["success"]), target_distance=float((target[0] ** 2 + target[1] ** 2) ** .5),
                            wind=float(sum(v * v for v in shot.get("wind", [0, 0, 0])) ** .5), training_step=self.num_timesteps)
        return True


def train(config: dict, algorithm: str | None = None, timesteps: int | None = None) -> Path:
    algo = (algorithm or config["algorithm"]["algorithm"]).lower()
    data_dir = Path(__file__).resolve().parents[1] / "data"; data_dir.mkdir(exist_ok=True)
    state_path = data_dir / "training_state.json"; total = timesteps or config["rl"]["timesteps"]
    state_path.write_text(json.dumps({"status":"running", "algorithm":algo, "total_timesteps":total, "started_at":datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    try:
        env = Monitor(BallLauncherEnv(config, curriculum_level=config["rl"]["curriculum_level"], seed=config["rl"]["seed"]))
        params = config["algorithm"].get(algo, {})
        model_class = SAC if algo == "sac" else PPO if algo == "ppo" else None
        if model_class is None: raise ValueError("algorithm must be 'sac' or 'ppo'")
        model = model_class("MlpPolicy", env, verbose=1, seed=config["rl"]["seed"], **params)
        model.learn(total_timesteps=total, callback=AnalyticsCallback(data_dir), progress_bar=False)
        path = Path(__file__).resolve().parents[1] / "models" / f"{algo}_launcher"
        model.save(path)
        evaluation = evaluate_policy(model, config, episodes=100)
        (data_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        state_path.write_text(json.dumps({"status":"complete", "algorithm":algo, "total_timesteps":total, "finished_at":datetime.now(timezone.utc).isoformat(), "evaluation":evaluation}), encoding="utf-8")
        return path.with_suffix(".zip")
    except Exception as error:
        state_path.write_text(json.dumps({"status":"failed", "algorithm":algo, "total_timesteps":total, "error":str(error)}), encoding="utf-8")
        raise
