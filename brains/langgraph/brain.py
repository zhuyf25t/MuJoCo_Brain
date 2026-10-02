"""Bridge fresh front images and command history into the modular state graph."""

import hashlib
import json
from pathlib import Path

import config
from brains.base import Brain
from .capabilities import Capabilities, load_profile
from .contracts import TaskState
from .graph import build_graph
from .memory import ExperienceStore, RunLog
from .stages import Stages


class LangGraphBrain(Brain):
    name = "langgraph"
    requires_final_check = True
    stop_on_tool_error = True
    gui_realtime = True

    def __init__(self, *, profile_path=None, memory_dir=None, backend=None, overrides=None):
        self.profile = load_profile(profile_path)
        self.max_decisions = self.profile["max_decisions"]
        self.root = Path(memory_dir or config.ROOT / "output" / "langgraph")
        self.capabilities = Capabilities(self.profile, backend=backend, overrides=overrides)
        self.memory = None
        self.graph = None
        self.reset()

    def reset(self):
        self.state = TaskState()
        self.history_count = 0
        self.decision_round = 0
        self.log = RunLog(self.root)
        self.task_text = None

    def close(self):
        self.capabilities.close()

    def _prepare(self, rgb):
        # Fingerprint settings without passing numeric geometry or pose truth to any model.
        settings = [self.profile["calibration_id"], list(rgb.shape), config.ARM_STOW,
                    config.ARM_CARRY, config.ARM_REACH, config.ARM_DROP,
                    config.CAM_FRONT_FOVY, config.CAM_HEIGHT, config.CAM_PITCH_DEG,
                    config.CAM_FORWARD_OFFSET, config.PRIM_V, config.PRIM_W]
        config_id = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:20]
        if self.memory is None or self.memory.config_id != config_id:
            if self.state.started:
                raise RuntimeError("Configuration changed mid-task; start a new task")
            self.memory = ExperienceStore(self.root, config_id)
            self.stages = Stages(self.profile, self.memory, self.capabilities)
            self.graph = build_graph(self.stages)
        return config_id

    def _feedback(self, history, current):
        if len(history) < self.history_count:
            raise RuntimeError("Command history reset without brain.reset()")
        commands = [{"tool": h["tool"], "args": h.get("args", {})}
                    for h in history[self.history_count:]]
        if commands != self.state.pending_commands:
            raise RuntimeError("Actual command sequence does not match the pending batch")
        previous = self.state.frame
        self.history_count = len(history)
        self.state.pending_commands = []
        self.state.can_act = False
        self.state.permission = None
        self.state.loaded = []
        self.state.request = None
        if previous is None:
            return
        transition = {"before": previous.model_dump(), "commands": commands,
                      "after": current.model_dump(), "before_target":
                      self.state.target.model_dump() if self.state.target is not None
                      and self.state.target_frame_id == previous.id else None}
        self.state.last_transition = transition
        self.log.append("transition", **transition)
        # Learning waits for this frame's one fresh detection in Stages.locate().

    def decide(self, obs, task_text, tool_schemas, history):
        # Deliberately do not read holding, qpos, coordinates, extra cameras or tool result text.
        rgb = obs.get("images", {}).get("front")
        if rgb is None:
            raise ValueError("LangGraph requires images.front")
        if self.task_text is not None and task_text != self.task_text:
            raise RuntimeError("Task changed without reset()")
        self.task_text = task_text
        config_id = self._prepare(rgb)
        pose = self.state.pose
        for h in history[self.history_count:]:
            if h["tool"] == "arm_pose":
                pose = h.get("args", {}).get("pose", "unknown")
        current = self.log.frame(rgb, config_id, pose)
        self.decision_round += 1
        self.log.append("round_started", round=self.decision_round, phase=self.state.phase,
                        step=self.state.step, frame=current.model_dump())
        self.capabilities.begin_frame(self.log)
        try:
            self._feedback(history, current)
            self.state.pose, self.state.frame = pose, current
            self.stages.task, self.stages.log = task_text, self.log
            result = self.graph.invoke({"task": self.state, "actions": []}, {"recursion_limit": 8})
            self.state = result["task"]
            available = {schema["name"] for schema in tool_schemas}
            if any(a.tool not in available for a in result["actions"]):
                raise RuntimeError("Required robot tool is unavailable")
            self.log.append("decision", phase=self.state.phase, step=self.state.step,
                            frame_id=current.id, commands=self.state.pending_commands)
            return result["actions"]
        except Exception as error:
            self.log.append("task_error", phase=self.state.phase, frame_id=current.id,
                            error=type(error).__name__, message=str(error))
            raise

    def record_outcome(self, outcome):
        self.log.append("outcome", **outcome, final_verdict=self.state.final_verdict)
