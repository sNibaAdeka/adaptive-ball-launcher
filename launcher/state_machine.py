from __future__ import annotations

from enum import StrEnum


class LauncherState(StrEnum):
    IDLE = "IDLE"
    TARGET_SELECTED = "TARGET_SELECTED"
    AIMING = "AIMING"
    READY = "READY"
    LAUNCHING = "LAUNCHING"
    BALL_IN_FLIGHT = "BALL_IN_FLIGHT"
    BALL_LANDED = "BALL_LANDED"
    RESULT = "RESULT"
    RESETTING = "RESETTING"


class LauncherStateMachine:
    def __init__(self) -> None:
        self.state = LauncherState.IDLE

    def select_target(self) -> None:
        if self.state in (LauncherState.IDLE, LauncherState.RESULT):
            self.state = LauncherState.TARGET_SELECTED

    def aim(self) -> None:
        if self.state == LauncherState.TARGET_SELECTED:
            self.state = LauncherState.AIMING

    def ready(self) -> None:
        if self.state == LauncherState.AIMING:
            self.state = LauncherState.READY

    def launch(self) -> None:
        if self.state == LauncherState.READY:
            self.state = LauncherState.LAUNCHING

    def flying(self) -> None:
        if self.state == LauncherState.LAUNCHING:
            self.state = LauncherState.BALL_IN_FLIGHT

    def landed(self) -> None:
        if self.state == LauncherState.BALL_IN_FLIGHT:
            self.state = LauncherState.BALL_LANDED

    def result(self) -> None:
        if self.state == LauncherState.BALL_LANDED:
            self.state = LauncherState.RESULT

    def reset(self) -> None:
        self.state = LauncherState.RESETTING
        self.state = LauncherState.IDLE
