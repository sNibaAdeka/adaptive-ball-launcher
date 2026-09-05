from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class ShotLogger:
    fields = ["timestamp", "target_position", "launcher_height", "pitch", "yaw", "velocity", "spin", "wind", "air_density", "predicted_landing", "actual_landing", "error", "reward", "contact_surface"]

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path, self.json_path = self.directory / "shots.csv", self.directory / "shots.json"

    def append(self, *, result: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
        command = result["command"]
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "target_position": result["target"], "launcher_height": command["height"],
            "pitch": command["pitch"], "yaw": command["yaw"], "velocity": command["velocity"], "spin": command["spin"],
            "wind": result["wind"], "air_density": result["air_density"], "predicted_landing": predicted["landing"], "actual_landing": result["landing"],
            "error": result["error"], "reward": result["reward"], "contact_surface": result["contact_surface"],
            "predicted_trajectory": predicted["trajectory"], "actual_trajectory": result["trajectory"],
        }
        exists = self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields); 
            if not exists: writer.writeheader()
            writer.writerow({key: json.dumps(record[key]) if isinstance(record[key], (list, dict)) else record[key] for key in self.fields})
        history = json.loads(self.json_path.read_text(encoding="utf-8")) if self.json_path.exists() else []
        history.append(record)
        self.json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return record

    def history(self) -> list[dict[str, Any]]:
        return json.loads(self.json_path.read_text(encoding="utf-8")) if self.json_path.exists() else []
