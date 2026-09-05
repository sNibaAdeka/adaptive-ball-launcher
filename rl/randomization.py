"""Episode-level domain randomization, never stochastic jitter every physics frame."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def randomized_config(base: dict, level: int, rng: np.random.Generator) -> dict:
    config = deepcopy(base)
    if level >= 2:
        config["launcher"]["default_height"] = float(rng.uniform(.35, 1.6))
    if level >= 3:
        config["physics"]["wind_mode"] = "constant"
        config["physics"]["wind"] = rng.uniform(-.45, .45, size=3).round(3).tolist(); config["physics"]["wind"][2] *= .2
    if level >= 4:
        config["physics"]["wind_mode"] = "gusts"; config["physics"]["wind_variation"] = .32; config["physics"]["gust_amplitude"] = .65
    if level >= 5:
        config["launcher"]["ball_mass_variation"] = .04; config["launcher"]["drag_variation"] = .08; config["launcher"]["velocity_noise"] = .05
    if level >= 7:
        config["physics"]["air_density"] = float(rng.uniform(1.05, 1.32))
    return config


def random_target(config: dict, level: int, rng: np.random.Generator) -> np.ndarray:
    distance = rng.uniform(1, 3) if level == 1 else rng.uniform(1.0, 6.0)
    angle = rng.uniform(-.70, .70)
    point = np.array([math_sin(angle) * distance, -4.52 + math_cos(angle) * distance, .02])
    if level >= 6 and rng.random() < .28:
        platform = rng.choice(config["environment"]["platforms"])
        point = np.array([platform["position"][0], platform["position"][1], platform["position"][2] + platform["size"][2] + .02])
    return point


def math_sin(value: float) -> float:
    from math import sin
    return sin(value)


def math_cos(value: float) -> float:
    from math import cos
    return cos(value)
