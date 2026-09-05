from __future__ import annotations


def landing_reward(error: float, valid: bool = True) -> float:
    bonus = 10.0 if error < .05 else 6.0 if error < .10 else 3.0 if error < .20 else 1.5 if error < .50 else .5 if error < 1.0 else 0.0
    return -error + bonus - (3.0 if not valid else 0.0)
