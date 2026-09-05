"""Offline analytics charts; intentionally only imported by the CLI analytics command."""
from __future__ import annotations

from pathlib import Path
import csv


def build_training_graphs(csv_path: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(Path(csv_path).open()))
    if not rows: return
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    x = [int(row["episode"]) for row in rows]
    for metric, label in (("reward", "Reward"), ("landing_error", "Mean error (m)"), ("success", "Success rate")):
        plt.figure(figsize=(7, 3.5)); plt.plot(x, [float(row[metric]) for row in rows], color="#44d7ff")
        plt.title(label); plt.xlabel("Episode"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / f"{metric}.png", dpi=160); plt.close()
