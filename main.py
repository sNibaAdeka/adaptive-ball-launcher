"""Run the interactive Adaptive Ball Launcher RL Lab in a browser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Ball Launcher RL Lab")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--train", choices=("sac", "ppo"), help="Run headless training instead of the UI")
    parser.add_argument("--timesteps", type=int, help="Training steps when using --train")
    parser.add_argument("--evaluate", choices=("sac", "ppo"), help="Evaluate a saved policy over 100 random targets")
    args = parser.parse_args()
    if args.train:
        from adaptive_ball_launcher.config import load_all
        from adaptive_ball_launcher.rl.train import train
        print(train(load_all(), args.train, args.timesteps))
        return
    if args.evaluate:
        from stable_baselines3 import PPO, SAC
        from adaptive_ball_launcher.config import load_all
        from adaptive_ball_launcher.rl.evaluate import evaluate_policy
        model_path = Path(__file__).resolve().parent / "models" / f"{args.evaluate}_launcher.zip"
        if not model_path.exists():
            raise SystemExit(f"No saved {args.evaluate.upper()} policy: {model_path}")
        model = (SAC if args.evaluate == "sac" else PPO).load(model_path)
        print(json.dumps(evaluate_policy(model, load_all(), 100), indent=2))
        return
    uvicorn.run("adaptive_ball_launcher.ui.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
