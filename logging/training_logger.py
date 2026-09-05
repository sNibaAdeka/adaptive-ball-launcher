from __future__ import annotations

import csv
from pathlib import Path


class TrainingLogger:
    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / "training.csv"; self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **metrics: float | int) -> None:
        exists = self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            if not exists: writer.writeheader()
            writer.writerow(metrics)
