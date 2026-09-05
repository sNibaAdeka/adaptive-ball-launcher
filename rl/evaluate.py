from __future__ import annotations

import math
import numpy as np

from adaptive_ball_launcher.rl.env import BallLauncherEnv


def evaluate_policy(model, config: dict, episodes: int = 100) -> dict[str, float]:
    env = BallLauncherEnv(config, curriculum_level=7, seed=917)
    errors: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        errors.append(info["landing_error"])
    arr = np.asarray(errors)
    return {"episodes": episodes, "mean_error": float(arr.mean()), "median_error": float(np.median(arr)), "rmse": float(math.sqrt(np.mean(arr**2))),
            "p90_error": float(np.percentile(arr, 90)), "p95_error": float(np.percentile(arr, 95)),
            **{f"success_{int(limit*100)}cm": float(np.mean(arr < limit)) for limit in (1, .5, .2, .1, .05)}}
