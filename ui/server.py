"""Thin interactive API: all shot calculations remain inside Python/MuJoCo."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Literal
import csv
import json
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adaptive_ball_launcher.config import load_all
from adaptive_ball_launcher.launcher.controller import SimulatedLauncherController
from adaptive_ball_launcher.logging.shot_logger import ShotLogger
from adaptive_ball_launcher.prediction.system_identification import SystemIdentifier
from adaptive_ball_launcher.prediction.trajectory_predictor import TrajectoryPredictor
from adaptive_ball_launcher.rl.ai_policy import InteractiveAIPolicy
from adaptive_ball_launcher.simulation.target import Target
from adaptive_ball_launcher.simulation.world import PhysicsWorld, ShotCommand

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
config = load_all()
controller = SimulatedLauncherController(config["launcher"])
predictor = TrajectoryPredictor(config)
ai_policy = InteractiveAIPolicy(config, predictor)
identifier = SystemIdentifier()
logger = ShotLogger(ROOT / "data")
app = FastAPI(title="Adaptive Ball Launcher RL Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TargetRequest(BaseModel):
    x: float = Field(ge=-6.0, le=6.0)
    y: float = Field(ge=-7.0, le=7.0)
    z: float = Field(ge=0.0, le=6.0)


class ShotRequest(TargetRequest):
    mode: Literal["ai", "manual"] = "ai"
    yaw: float | None = None
    pitch: float | None = None
    velocity: float | None = None
    height: float | None = None
    spin_x: float = Field(default=0, ge=-180, le=180)
    spin_y: float = Field(default=0, ge=-180, le=180)
    spin_z: float = Field(default=0, ge=-180, le=180)


def scene_payload() -> dict:
    environment = config["environment"]
    return {"arena": {key: environment[key] for key in ("arena_width", "arena_length", "arena_height_limit")},
            "platforms": environment["platforms"], "launcher": config["launcher"], "physics": {key: config["physics"][key] for key in ("wind_mode", "wind", "air_density", "magnus_enabled")},
            "controller": controller.get_state(), "history_count": len(logger.history())}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/training")
def training_page():
    return FileResponse(STATIC / "training.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/scene")
def get_scene():
    return scene_payload()


@app.post("/api/target")
def select_target(request: TargetRequest):
    if request.y < -4.2 and abs(request.x) < 1.0:
        raise HTTPException(400, "The protected launcher area is not targetable.")
    controller.machine.select_target()
    target = Target.from_iterable([request.x, request.y, request.z], config["environment"]["target_radius"])
    origin = controller.launch_origin()
    return {"target": target.metrics_from(__import__("numpy").asarray(origin)), "state": controller.get_state()}


@app.post("/api/shot")
def shoot(request: ShotRequest):
    if controller.machine.state.value not in {"IDLE", "TARGET_SELECTED", "RESULT"}:
        raise HTTPException(409, "A shot is currently in progress. Wait for its result before launching again.")
    target = Target.from_iterable([request.x, request.y, request.z], config["environment"]["target_radius"])
    controller.machine.select_target()
    spin = (request.spin_x, request.spin_y, request.spin_z)
    height = request.height if request.height is not None else controller.height.current_position
    if request.mode == "ai":
        command = ai_policy.decide(target, height, spin)
    else:
        if None in (request.yaw, request.pitch, request.velocity):
            raise HTTPException(422, "Manual mode requires yaw, pitch, and velocity.")
        command = ShotCommand(request.yaw, request.pitch, request.velocity, height, spin)
    controller.set_yaw(command.yaw); controller.set_pitch(command.pitch); controller.set_height(command.height)
    controller.set_launch_parameter(controller.velocity_model.to_power(command.velocity))
    controller.wait_until_ready()
    launch_state = controller.launch()
    launch_state["spin"] = list(command.spin)
    predicted = predictor.predict(command, target).as_dict()
    # Imperfections only affect the actual virtual lab model, never the prediction model.
    noisy = ShotCommand(command.yaw + random.gauss(0, config["launcher"]["angle_noise"]), command.pitch + random.gauss(0, config["launcher"]["angle_noise"]),
                        command.velocity * (1 + random.gauss(0, config["launcher"]["velocity_noise"])), command.height, command.spin)
    controller.machine.flying()
    actual = PhysicsWorld(config, actual_model=True).run_shot(noisy, target).as_dict()
    controller.machine.landed(); controller.machine.result()
    record = logger.append(result=actual, predicted=predicted)
    calibration = identifier.update(predicted["landing"], actual["landing"])
    return {"state": controller.get_state(), "decision": launch_state, "decision_source": ai_policy.source if request.mode == "ai" else "MANUAL INPUT", "predicted": predicted, "actual": actual,
            "prediction_error": round(float(__import__("numpy").linalg.norm(__import__("numpy").asarray(predicted["landing"]) - __import__("numpy").asarray(actual["landing"]))), 4),
            "calibration_bias": calibration.landing_bias.round(4).tolist(), "record": record}


@app.get("/api/history")
def history():
    return logger.history()


@app.get("/api/training/status")
def training_status():
    data_dir = ROOT / "data"; state_path = data_dir / "training_state.json"; csv_path = data_dir / "training.csv"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status":"idle", "total_timesteps":0}
    if not csv_path.exists(): return {"state":state, "points":[]}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        points = list(deque(csv.DictReader(stream), maxlen=180))
    return {"state":state, "points":points}


@app.post("/api/replay/{kind}")
def replay(kind: Literal["last", "best", "worst"]):
    shots = logger.history()
    if not shots: raise HTTPException(404, "No recorded shots yet.")
    selected = shots[-1] if kind == "last" else min(shots, key=lambda item: item["error"]) if kind == "best" else max(shots, key=lambda item: item["error"])
    return selected


@app.post("/api/settings")
def update_settings(payload: dict):
    allowed = {"wind_mode", "wind", "air_density", "magnus_enabled"}
    for key, value in payload.items():
        if key not in allowed: raise HTTPException(422, f"Unsupported setting: {key}")
        config["physics"][key] = value
    return scene_payload()
