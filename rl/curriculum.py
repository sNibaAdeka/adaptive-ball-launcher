from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Curriculum:
    level: int = 1
    success_rate: float = 0.0

    descriptions = {
        1: "No wind · fixed height · targets 1–3 m",
        2: "Random distance and launcher height",
        3: "Small constant wind",
        4: "Variable wind",
        5: "Physical parameter noise",
        6: "Elevated platforms",
        7: "Full domain randomization",
    }

    def update(self, success_rate: float) -> bool:
        self.success_rate = success_rate
        if success_rate >= .78 and self.level < 7:
            self.level += 1
            return True
        return False
