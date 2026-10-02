"""Typed capability contracts and program-owned task state."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Direction = Literal["forward", "back", "turn_left", "turn_right"]
Kind = Literal["reach", "translation", "rotation"]
Phase = Literal["origin", "empty", "pending", "holding", "final"]
NOTE_MAX_CHARS = 340


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(Record):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class TargetIdentity(Record):
    label: str = Field(max_length=120)


class Target(TargetIdentity):
    center: Point
    path: Literal["clear", "blocked", "unknown"]
    radius: float | None = Field(default=None, gt=0, le=1, description="Ball radius / image WIDTH; not diameter")
    visibility: Literal["full", "partial", "unknown"] = "unknown"


class BallTarget(Target):
    radius: float = Field(gt=0, le=1, description="Tight ball circle radius / image WIDTH; exclude shadow/reflection")
    visibility: Literal["full", "partial"]


class Detection(Record):
    status: Literal["visible", "absent", "unknown"]
    target: Target | None = None
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)

    @model_validator(mode="after")
    def visible_target(self):
        if (self.status == "visible") != (self.target is not None):
            raise ValueError("Only visible detections must contain a target")
        return self


class BallDetection(Detection):
    target: BallTarget | None = None


class InfoRequest(Record):
    kind: Kind
    direction: Direction | None = None

    @model_validator(mode="after")
    def matching_direction(self):
        if self.kind == "reach" and self.direction is not None:
            raise ValueError("reach has no direction")
        if self.kind == "translation" and self.direction not in ("forward", "back"):
            raise ValueError("translation needs forward/back")
        if self.kind == "rotation" and self.direction not in ("turn_left", "turn_right"):
            raise ValueError("rotation needs turn_left/turn_right")
        return self


class Match(Record):
    status: Literal["yes", "no", "unknown", "need_info"]
    request: InfoRequest | None = None
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)

    @model_validator(mode="after")
    def requested(self):
        if (self.status == "need_info") != (self.request is not None):
            raise ValueError("need_info must contain exactly one request")
        return self


class SpatialMatch(Match):
    alignment: Literal["left", "aligned", "right", "unknown"]
    distance: Literal["far", "ready", "too_close", "unknown"]


class Motion(Record):
    status: Literal["move", "unknown", "need_info"]
    direction: Direction | None = None
    seconds: float | None = Field(default=None, ge=0.1, description="Total planned duration; must not exceed input max_seconds")
    request: InfoRequest | None = None
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)

    @model_validator(mode="after")
    def complete(self):
        if (self.status == "need_info") != (self.request is not None):
            raise ValueError("need_info must contain a request")
        if self.status == "move":
            if self.direction is None or self.seconds is None:
                raise ValueError("move requires direction and seconds")
        elif self.direction is not None or self.seconds is not None:
            raise ValueError("Only move may contain an action")
        return self


class Verdict(Record):
    status: Literal["yes", "no", "unknown"]
    note: str = Field(default="", max_length=NOTE_MAX_CHARS)


class ReachAnalysis(Record):
    """valid means the estimated polygon is usable, not whether the photo exists."""

    valid: bool
    region: list[Point] = Field(default_factory=list, max_length=8)
    note: str = Field(max_length=NOTE_MAX_CHARS)

    @model_validator(mode="after")
    def polygon(self):
        if self.valid:
            if len(self.region) < 3:
                raise ValueError("A valid region needs at least three vertices")
            area = sum(a.x*b.y - b.x*a.y for a, b in
                       zip(self.region, self.region[1:] + self.region[:1]))
            if abs(area) < 1e-6:
                raise ValueError("Degenerate region")
        return self


class MotionAnalysis(Record):
    valid: bool
    landmark: str = Field(max_length=120)
    source: Literal["shared_target", "background"] = "background"
    before: Point | None = None
    after: Point | None = None
    note: str = Field(max_length=NOTE_MAX_CHARS)

    @model_validator(mode="after")
    def evidence(self):
        if self.valid and (not self.landmark or (self.source == "background" and
                                               (self.before is None or self.after is None))):
            raise ValueError("A valid motion sample needs the same visible landmark twice")
        return self


class MotionReview(MotionAnalysis):
    source: Literal["shared_target", "background"]

    @model_validator(mode="after")
    def shared_coordinates_are_program_owned(self):
        if self.source == "shared_target" and (self.before is not None or self.after is not None):
            raise ValueError("shared_target reuses input detections: before/after must be null")
        return self


class Frame(Record):
    id: str
    path: str
    config_id: str
    pose: str
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class Recovery(Record):
    """Task-local visual failure evidence, separate from motion calibration."""

    kind: Literal["grasp_failed"] = "grasp_failed"
    note: str = Field(max_length=NOTE_MAX_CHARS)
    failed_frame_id: str
    failures: int = Field(default=1, ge=1)
    correction_commands: list[dict] = Field(default_factory=list)


class CapabilityInput(Record):
    """Allowlisted context: only images, own commands and requested evidence."""

    frame: Frame
    target: Target | None = None
    previous_target: TargetIdentity | None = None
    comparison_target: Target | None = None
    assessment: SpatialMatch | None = None
    task: str = ""
    intent: str = ""
    summary: dict = Field(default_factory=dict)
    evidence: list[dict] = Field(default_factory=list)
    comparison: Frame | None = None
    commands: list[dict] = Field(default_factory=list)
    max_seconds: float = 0.6
    max_command_seconds: float = 2.0
    info_feedback: str = ""
    recovery: Recovery | None = None

    @classmethod
    def from_recorded(cls, value):
        """Migrate old replay inputs without accepting stale coordinates live."""
        value = dict(value)
        if value.get("previous_target"):
            value["previous_target"] = {"label": value["previous_target"]["label"]}
        return cls.model_validate(value)


@dataclass
class TaskState:
    phase: Phase = "origin"
    started: bool = False
    step: str = "start"
    can_act: bool = False
    target: Target | None = None
    target_frame_id: str | None = None
    assessment: SpatialMatch | None = None
    learned_frame_id: str | None = None
    carry_frame: Frame | None = None
    frame: Frame | None = None
    last_transition: dict | None = None
    request: InfoRequest | None = None
    permission: tuple | None = None
    pending_commands: list[dict] = field(default_factory=list)
    loaded: list[dict] = field(default_factory=list)
    unknown_count: int = 0
    pose: str = "unknown"
    final_verdict: str | None = None
    recovery: Recovery | None = None


def direction_request(direction: Direction) -> InfoRequest:
    return InfoRequest(kind="translation" if direction in ("forward", "back")
                       else "rotation", direction=direction)
