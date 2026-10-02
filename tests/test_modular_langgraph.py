"""Behavioral tests for image boundaries, phase progression, memory and swaps."""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from brains.langgraph import LangGraphBrain
from brains.langgraph.backends import CapabilityError, LLMBackend, polygon_match
from brains.langgraph.capabilities import OUTPUTS
from brains.langgraph.contracts import (
    BallDetection, CapabilityInput, Frame, InfoRequest, Match, SpatialMatch, MotionAnalysis, MotionReview, ReachAnalysis,
    TaskState, Target, direction_request,
)
from brains.langgraph.execution import authorize, command, issue, motion_batch
from brains.langgraph.memory import ExperienceStore, RunLog
from brains.langgraph.image_context import COMMON_PROMPT, image_content
from brains.langgraph.stages import FinalCheckError

TOOLS = [{"name": name} for name in ("arm_pose", "open_gripper", "close_gripper",
         "forward", "back", "turn_left", "turn_right", "observe", "done")]
REGION = [{"x": .4, "y": .6}, {"x": .6, "y": .6}, {"x": .6, "y": .8}, {"x": .4, "y": .8}]
TARGET = {"label": "visible ball", "center": {"x": .5, "y": .7}, "radius": .05,
          "visibility": "full", "path": "clear"}


class Scenario:
    def __init__(self):
        self.calls = []
        self.match = "yes"
        self.held = "yes"
        self.final = "yes"
        self.ball = "visible"
        self.sample_valid = True

    def callback(self, name):
        def invoke(value):
            self.calls.append((name, value))
            if name in ("find_ball", "find_box"):
                status = self.ball if name == "find_ball" else "visible"
                return {"status": status, "target": TARGET if status == "visible" else None}
            if name in ("match_grasp", "match_drop"):
                if not value.evidence:
                    return {"status": "need_info", "request": {"kind": "reach"},
                            "alignment": "unknown", "distance": "unknown"}
                status = self.match if name == "match_grasp" else "yes"
                return {"status": status, "alignment": "aligned", "distance": "ready" if status == "yes" else "far"}
            if name == "check_held":
                return {"status": self.held}
            if name == "check_drop":
                return {"status": self.final}
            if name == "analyze_reach":
                return {"valid": True, "region": REGION, "note": "visible fingers"}
            if name == "analyze_motion":
                return {"valid": self.sample_valid, "landmark": "box corner", "source": "background",
                        "before": {"x": .6, "y": .5}, "after": {"x": .5, "y": .5}, "note": "left"}
            if name == "plan_motion":
                direction = "turn_right" if value.intent == "search_right" else "forward"
                request = direction_request(direction)
                if not any(s["direction"] == direction for s in value.evidence):
                    return {"status": "need_info", "request": request.model_dump()}
                return {"status": "move", "direction": direction, "seconds": .2}
            raise AssertionError(name)
        return invoke


class Runner:
    def __init__(self, root, scenario=None, **kwargs):
        self.scenario = scenario or Scenario()
        self.brain = LangGraphBrain(memory_dir=root, overrides={n: self.scenario.callback(n) for n in OUTPUTS}, **kwargs)
        self.history = []
        self.round = 0

    def step(self):
        self.round += 1
        obs = {"images": {"front": np.full((24, 32, 3), self.round, np.uint8),
                          "overhead": "PRIVATE_OVERHEAD"},
               "holding": "PRIVATE_HOLDING", "arm_qpos": "PRIVATE_JOINTS"}
        batch = self.brain.decide(obs, "collect one ball", TOOLS, self.history)
        self.history.extend({"tool": d.tool, "args": d.args, "ok": False,
                             "result": "PRIVATE_SIMULATOR_RESULT"} for d in batch)
        return [(d.tool, d.args) for d in batch]

    def initialize(self):
        assert self.step() == [("arm_pose", {"pose": "stow"})]
        assert self.step() == [("arm_pose", {"pose": "reach"})]
        assert self.step() == [("arm_pose", {"pose": "stow"})]


@pytest.fixture
def runner(tmp_path):
    run = Runner(tmp_path)
    yield run
    run.brain.close()


def test_five_stages_wait_for_real_frames_and_keep_gripper_closed(runner):
    runner.initialize()
    assert [x[0] for x in runner.step()] == ["open_gripper", "arm_pose", "close_gripper", "arm_pose"]
    assert runner.brain.state.phase == "pending"
    assert runner.step() == [("arm_pose", {"pose": "reach"})]
    carry = runner.brain.state.carry_frame
    assert carry.pose == "carry"
    assert runner.step() == [("arm_pose", {"pose": "carry"})]
    held_inputs = [v for n, v in runner.scenario.calls if n == "check_held"]
    assert held_inputs[0].comparison == carry and held_inputs[0].frame.id != carry.id
    assert [x[0] for x in runner.step()] == ["arm_pose", "open_gripper", "observe"]
    assert runner.brain.state.phase == "final"
    assert runner.step() == [("done", {"success": True})]
    assert runner.brain.state.final_verdict == "yes"
    assert not any(n == "analyze_motion" for n, _ in runner.scenario.calls)
    assert set(runner.brain.graph.get_graph().nodes) == {"__start__", "__end__", "origin", "empty", "pending", "holding", "final"}
    sent = json.dumps([v.model_dump() for _, v in runner.scenario.calls])
    assert "PRIVATE_" not in sent


def test_miss_performs_one_probe_and_learns_only_after_new_frame(runner):
    runner.scenario.match = "no"
    runner.initialize()
    assert runner.step() == [("forward", {"seconds": .2})]
    assert not runner.brain.memory.query(direction_request("forward"))
    assert runner.brain.state.can_act is False
    assert runner.step() == [("forward", {"seconds": .2})]
    sample = runner.brain.memory.query(direction_request("forward"))[0]
    assert sample["frames"][0]["id"] != sample["frames"][1]["id"]
    assert sample["commands"] == [command("forward", seconds=.2)]
    assert sample["image_rate"]["duration_seconds"] == .2
    assert sample["image_rate"]["normalized_dx_per_second"] == pytest.approx(-.5)
    assert sample["image_rate"]["normalized_dy_per_second"] == 0
    assert not runner.brain.memory.query(direction_request("back"))


def test_no_landmark_search_does_not_deadlock_or_create_experience(runner):
    runner.scenario.ball = "absent"
    runner.scenario.sample_valid = False
    runner.initialize()
    for _ in range(3):
        assert runner.step() == [("turn_right", {"seconds": .2})]
    assert not runner.brain.memory.query(direction_request("turn_right"))


def test_known_reference_and_recent_motion_do_not_need_extra_model_requests(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()  # First forward probe.
    runner.scenario.calls.clear()
    runner.step()
    names = [name for name, _ in runner.scenario.calls]
    assert names.count("match_grasp") == names.count("plan_motion") == 1
    plan_input = next(value for name, value in runner.scenario.calls if name == "plan_motion")
    assert any(s["direction"] == "forward" for s in plan_input.evidence)
    assert any(s["kind"] == "reach" for s in plan_input.evidence)
    assert plan_input.assessment.distance == "far"
    assert plan_input.max_seconds == 4.0


def test_failed_grasp_survives_stow_and_requires_acknowledged_correction(runner):
    runner.initialize()
    runner.step()  # Grasp.
    runner.step()  # Closed reach to check.
    runner.brain.capabilities.overrides["check_held"] = lambda v: {
        "status": "no", "note": "球仍在地面，两指空夹；原因不能确定"}
    assert runner.step() == [("arm_pose", {"pose": "stow"})]
    failure = runner.brain.state.recovery
    assert failure.note == "球仍在地面，两指空夹；原因不能确定"
    assert failure.failed_frame_id == runner.brain.state.frame.id
    seen = []
    def recover(value):
        seen.append(value)
        return {"status": "need_info", "request": {"kind": "translation", "direction": "back"}}
    runner.brain.capabilities.overrides["plan_motion"] = recover
    # Even a fresh match=yes must not trigger the identical grasp again.
    assert runner.step() == [("back", {"seconds": .2})]
    assert seen[0].recovery == failure and seen[0].max_seconds == .6
    assert seen[0].intent.startswith("recover_grasp:")
    assert not runner.brain.state.recovery.correction_commands  # Issuing is not execution.
    matches = [v for n, v in runner.scenario.calls if n == "match_grasp"]
    assert matches[-1].recovery == failure
    assert all(v.recovery is None for n, v in runner.scenario.calls if n == "find_ball")
    assert runner.step()[0][0] == "open_gripper"  # New frame acknowledges actual back.
    corrected = runner.brain.state.recovery
    assert corrected.note == failure.note
    assert corrected.correction_commands == [command("back", seconds=.2)]
    runner.brain.capabilities.overrides["check_held"] = lambda v: {"status": "yes"}
    runner.step()
    runner.step()
    assert runner.brain.state.phase == "holding" and runner.brain.state.recovery is None
    sent = json.dumps([v.model_dump() for _, v in runner.scenario.calls] + [v.model_dump() for v in seen])
    assert "PRIVATE_" not in sent


def test_uncertain_recovery_does_not_repeat_grasp_or_forget_failure(runner):
    runner.scenario.held = "no"
    runner.initialize()
    runner.step()
    runner.step()
    runner.step()
    runner.brain.capabilities.overrides["plan_motion"] = lambda v: {
        "status": "unknown", "note": "无法判断应怎样修正"}
    for _ in range(2):
        assert runner.step() == [("observe", {})]
        assert runner.brain.state.recovery.failures == 1
        assert not runner.brain.state.recovery.correction_commands
    with pytest.raises(RuntimeError, match="连续无法判定"):
        runner.step()


def test_new_failure_resets_correction_and_task_reset_clears_recovery(runner):
    runner.scenario.held = "no"
    runner.initialize()
    for _ in range(3):
        runner.step()
    for _ in range(4):  # Recovery probe, retry, closed reach, failed verification.
        runner.step()
    assert runner.brain.state.recovery.failures == 2
    assert not runner.brain.state.recovery.correction_commands
    runner.brain.reset()
    assert runner.brain.state.recovery is None


def test_long_forward_is_one_batch_and_learned_with_full_duration(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()  # Missing experience: still only one 0.2s probe.
    calls = []
    def plan(value):
        calls.append(value)
        return {"status": "move", "direction": "forward", "seconds": 4.0}
    runner.brain.capabilities.overrides["plan_motion"] = plan
    assert runner.step() == [("forward", {"seconds": 2.0}), ("forward", {"seconds": 2.0})]
    assert len(calls) == 1  # No model call between the two commands.
    runner.step()
    sample = runner.brain.memory.query(direction_request("forward"))[0]
    assert sample["commands"] == [command("forward", seconds=2.0)] * 2
    assert sample["image_rate"]["duration_seconds"] == 4.0
    assert sample["image_rate"]["normalized_dx_per_second"] == pytest.approx(-.025)


@pytest.mark.parametrize("distance,direction,seconds", [
    ("ready", "forward", 4.0), ("far", "forward", 4.1), ("far", "back", 4.0),
])
def test_invalid_long_plans_do_not_issue_any_commands(runner, distance, direction, seconds):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()
    runner.brain.capabilities.overrides["match_grasp"] = lambda v: {
        "status": "no", "alignment": "aligned", "distance": distance}
    runner.brain.capabilities.overrides["plan_motion"] = lambda v: {
        "status": "move", "direction": direction, "seconds": seconds}
    if direction == "back":
        # Supply direction evidence so the invalid long back reaches plan validation.
        runner.brain.memory.add(direction_request("back"), [runner.brain.state.frame] * 2,
                               [command("back", seconds=.2)], MotionAnalysis(
                                   valid=True, landmark="corner", before={"x": .6, "y": .5},
                                   after={"x": .5, "y": .5}, note="shift"))
    with pytest.raises(RuntimeError, match="planning limit|Long motion"):
        runner.step()
    assert not runner.brain.state.pending_commands


@pytest.mark.parametrize("total,expected", [(4, [2, 2]), (4.3, [2, 2, .3]), (2.05, [1.95, .1])])
def test_motion_batch_preserves_total_and_valid_remainders(total, expected):
    batch = motion_batch("forward", total, 2)
    durations = [c["args"]["seconds"] for c in batch]
    assert durations == expected
    assert sum(durations) == pytest.approx(total)


def test_duplicate_request_gets_feedback_without_aborting_or_duplicating_samples(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()
    seen = []
    def plan(value):
        seen.append(value)
        if not value.info_feedback:
            return {"status": "need_info", "request": {"kind": "translation", "direction": "forward"}}
        return {"status": "move", "direction": "forward", "seconds": 1.2}
    runner.brain.capabilities.overrides["plan_motion"] = plan
    assert runner.step() == [("forward", {"seconds": 1.2})]
    assert len(seen) == 2 and seen[1].info_feedback
    assert len(seen[0].evidence) == len(seen[1].evidence) == 2


@pytest.mark.parametrize("held,phase,pose", [("no", "empty", "stow"), ("unknown", "pending", "carry")])
def test_pending_branches_do_not_open_gripper(runner, held, phase, pose):
    runner.scenario.held = held
    runner.initialize()
    runner.step()
    runner.step()
    assert runner.step() == [("arm_pose", {"pose": pose})]
    assert runner.brain.state.phase == phase


@pytest.mark.parametrize("verdict", ["no", "unknown"])
def test_final_failure_raises_without_recovery(runner, verdict):
    runner.scenario.final = verdict
    for _ in range(7):
        runner.step()
    with pytest.raises(FinalCheckError):
        runner.step()
    assert runner.brain.state.final_verdict == verdict


def test_reset_keeps_experience_but_clears_task_and_compares_fresh_images(runner):
    runner.initialize()
    sample_id = runner.brain.memory.samples[0]["id"]
    runner.brain.reset()
    runner.history = []
    assert runner.brain.state.phase == "origin" and runner.brain.state.carry_frame is None
    assert runner.step() == [("arm_pose", {"pose": "stow"})]
    assert runner.step()[0][0] == "open_gripper"
    assert runner.brain.memory.samples[0]["id"] == sample_id


def test_rejects_missing_execution_feedback(runner):
    runner.step()
    runner.history.clear()
    with pytest.raises(RuntimeError, match="command sequence"):
        runner.step()


def test_permission_is_scoped_to_direction_frame_and_target():
    state = TaskState(frame=Frame(id="f1", path="unused", config_id="c", pose="stow"))
    state.loaded = [{"kind": "rotation", "direction": "turn_right"}]
    with pytest.raises(RuntimeError, match="Missing requested"):
        authorize(state, "forward", [direction_request("forward")])
    authorize(state, "turn_right", [direction_request("turn_right")])
    state.frame = state.frame.model_copy(update={"id": "f2"})
    with pytest.raises(RuntimeError, match="No permission"):
        issue(state, [command("turn_right", seconds=.2)], "", action="turn_right")


def test_polygon_implementation_swaps_without_graph_changes(runner):
    runner.brain.capabilities.overrides["match_grasp"] = polygon_match
    runner.initialize()
    assert runner.step()[0][0] == "open_gripper"
    frame = runner.brain.state.frame
    sample = runner.brain.memory.samples[0]
    value = CapabilityInput(frame=frame, target=Target.model_validate(TARGET), evidence=[sample])
    assert polygon_match(value).status == "yes"
    incompatible = value.model_copy(update={"frame": frame.model_copy(update={"config_id": "changed"})})
    assert polygon_match(incompatible).status == "unknown"


def test_experience_caps_and_configuration_invalidation(tmp_path):
    store = ExperienceStore(tmp_path, "c1")
    log = RunLog(tmp_path)
    frame = log.frame(np.zeros((3, 3, 3), np.uint8), "c1", "reach")
    analysis = ReachAnalysis(valid=True, region=REGION, note="fingers")
    for _ in range(4):
        store.add(InfoRequest(kind="reach"), [frame], [], analysis)
    assert len(store.samples) == 2
    movement = MotionAnalysis(valid=True, landmark="corner", before={"x": .1, "y": .2},
                              after={"x": .2, "y": .2}, note="shift")
    for direction in ("forward", "back", "forward", "turn_right", "turn_left", "turn_right"):
        store.add(direction_request(direction), [frame, frame], [command(direction, seconds=.2)], movement)
    assert len(store.samples) == 6
    assert len(ExperienceStore(tmp_path, "c1").samples) == 6
    assert ExperienceStore(tmp_path, "c2").samples == []


def test_model_cannot_bypass_required_information(runner):
    runner.brain.capabilities.overrides["match_grasp"] = lambda v: SpatialMatch(status="yes", alignment="aligned", distance="ready")
    runner.initialize()
    assert runner.step()[0][0] == "open_gripper"
    events = [json.loads(line) for line in (runner.brain.log.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "request_info" and e["request"]["kind"] == "reach" for e in events)


def test_uncertain_reach_region_keeps_photo_and_finishes_initialization(runner):
    runner.brain.capabilities.overrides["analyze_reach"] = lambda v: {
        "valid": False, "region": [], "note": "Visible fingers, uncertain projected grasp area"}
    runner.initialize()
    sample = runner.brain.memory.query(InfoRequest(kind="reach"))[0]
    assert sample["analysis"]["valid"] is False
    assert Path(sample["frames"][0]["path"]).is_file()
    loaded = ExperienceStore(runner.brain.root, runner.brain.memory.config_id)
    assert loaded.query(InfoRequest(kind="reach"))[0] == sample
    value = CapabilityInput(frame=runner.brain.state.frame.model_copy(update={"pose": "stow"}),
                            target=Target.model_validate(TARGET), evidence=[sample])
    assert polygon_match(value).status == "unknown"
    # A reference exists; the following phase can request it without another reach probe.
    runner.brain.capabilities.overrides["match_grasp"] = lambda v: (
        {"status": "unknown", "alignment": "unknown", "distance": "unknown", "note": "cannot match this ball"} if v.evidence
        else {"status": "need_info", "alignment": "unknown", "distance": "unknown", "request": {"kind": "reach"}})
    assert runner.step() == [("observe", {})]
    assert runner.brain.state.phase == "empty"


def test_backend_uses_per_module_model_prompt_limits_and_logs_usage(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://mock.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "PRIVATE_KEY")
    captured, logs = [], []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"total_tokens": 123}, "choices": [{"message": {
            "tool_calls": [{"function": {"name": "submit_result", "arguments": '{"status":"no"}'}}]}}]})

    backend = LLMBackend(transport=httpx.MockTransport(handler))
    frame = RunLog(tmp_path).frame(np.zeros((4, 4, 3), np.uint8), "c", "stow")
    value = CapabilityInput(frame=frame)
    try:
        for name, model in (("match_grasp", "small-model"), ("match_drop", "other-model")):
            result = backend.invoke(name, {"model": model, "max_tokens": 111, "timeout_s": 1,
                                          "attempts": 1, "thinking": "disabled"},
                                    f"short prompt {name}", value, Match,
                                    lambda event, **kw: logs.append({"event": event, **kw}))
            assert result.status == "no"
    finally:
        backend.close()
    assert [r["model"] for r in captured] == ["small-model", "other-model"]
    assert captured[0]["messages"][0] != captured[1]["messages"][0]
    assert all(r["max_tokens"] == 111 and len(r["tools"]) == 1 for r in captured)
    assert all(r["thinking"] == {"type": "disabled"} for r in captured)
    assert all(e["usage"]["total_tokens"] == 123 for e in logs)
    assert "PRIVATE_KEY" not in json.dumps(logs + captured)


@pytest.mark.parametrize("thinking", ["enabled", "disabled"])
def test_deepseek_thinking_uses_supported_tool_choice(tmp_path, monkeypatch, thinking):
    import config
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://mock.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "PRIVATE_KEY")
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        if thinking == "enabled" and body["tool_choice"] != "auto":
            return httpx.Response(400, json={"error": {"message": "Forced tool choice is not supported in thinking mode"}})
        return httpx.Response(200, json={"choices": [{"message": {
            "tool_calls": [{"function": {"name": "submit_result", "arguments": '{"status":"no"}'}}]}}]})

    backend = LLMBackend(transport=httpx.MockTransport(handler))
    frame = RunLog(tmp_path).frame(np.zeros((4, 4, 3), np.uint8), "c", "stow")
    spec = {"model": "deepseek-flash", "max_tokens": 8192, "timeout_s": 1,
            "attempts": 1, "thinking": thinking, "reasoning_effort": "low"}
    try:
        result = backend.invoke("match_grasp", spec, "short", CapabilityInput(frame=frame),
                                Match, lambda *a, **k: None)
    finally:
        backend.close()
    assert result.status == "no"
    assert captured[0]["reasoning_effort"] == "low"
    assert captured[0]["thinking"] == {"type": thinking}
    assert len(captured[0]["tools"]) == 1
    if thinking == "disabled":
        assert captured[0]["tool_choice"]["function"]["name"] == "submit_result"


def test_backend_bad_schema_fails_after_bounded_retries(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://mock.invalid/v1")
    count = []
    def handler(request):
        count.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"status":"invented"}'}}]})
    backend = LLMBackend(transport=httpx.MockTransport(handler))
    frame = RunLog(tmp_path).frame(np.zeros((4, 4, 3), np.uint8), "c", "stow")
    try:
        with pytest.raises(CapabilityError):
            backend.invoke("match_grasp", {"model": "test", "max_tokens": 100, "timeout_s": 1, "attempts": 2},
                           "short", CapabilityInput(frame=frame), Match, lambda *a, **k: None)
    finally:
        backend.close()
    assert len(count) == 2


@pytest.mark.parametrize("visual", [True, False])
def test_collector_waits_for_visual_final_even_if_environment_says_success(tmp_path, monkeypatch, visual):
    from types import SimpleNamespace
    from brains.base import Decision
    import run_collect

    calls = []
    class Brain:
        name = "test_final"
        requires_final_check = True
        max_decisions = 3
        def decide(self, *args):
            calls.append(1)
            if len(calls) == 1:
                return [Decision("", "observe")]
            if not visual:
                raise FinalCheckError("unknown")
            return [Decision("", "done", {"success": True})]
    env = SimpleNamespace(model=None, data=SimpleNamespace(time=0),
                          robot=SimpleNamespace(base=None, arm=None, hal=SimpleNamespace(attached_object=lambda: None)),
                          get_obs=lambda: {}, success=lambda: True)
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=lambda *a: (True, "ok")))
    result = run_collect.run_episode(env, Brain(), [], "test", tmp_path / "episode", save_images=False)
    assert len(calls) == 2 and result["environment_success"] is True
    assert result["success"] is visual


def test_collector_aborts_macro_after_tool_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from brains.base import Decision
    import run_collect

    calls = []
    def execute(tool, args):
        calls.append(tool)
        return False, "motor command failed"
    env = SimpleNamespace(model=None, data=SimpleNamespace(time=0),
                          robot=SimpleNamespace(base=None, arm=None, hal=SimpleNamespace(attached_object=lambda: None)),
                          get_obs=lambda: {}, success=lambda: False)
    brain = SimpleNamespace(name="test_failure", stop_on_tool_error=True,
                            decide=lambda *args: [Decision("", "arm_pose", {"pose": "drop"}), Decision("", "open_gripper")])
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: SimpleNamespace(execute=execute))
    result = run_collect.run_episode(env, brain, [], "test", tmp_path / "episode", save_images=False)
    assert calls == ["arm_pose"] and result["decisions"] == 1


@pytest.mark.parametrize("schema,payload", [
    (OUTPUTS["find_ball"], {"status": "unknown"}),
    (OUTPUTS["match_grasp"], {"status": "no", "alignment": "unknown", "distance": "unknown"}),
    (OUTPUTS["plan_motion"], {"status": "unknown"}),
    (OUTPUTS["check_held"], {"status": "unknown"}),
    (ReachAnalysis, {"valid": False}),
    (MotionAnalysis, {"valid": False, "landmark": ""}),
])
def test_notes_accept_340_characters_and_reject_341_without_truncation(schema, payload):
    note = "中 a。" * 85
    assert len(note) == 340
    assert schema.model_validate({**payload, "note": note}).note == note
    with pytest.raises(ValidationError) as caught:
        schema.model_validate({**payload, "note": note + "!"})
    assert caught.value.errors()[0]["type"] == "string_too_long"


def test_failed_response_keeps_full_text_exact_error_and_retry_feedback(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://mock.invalid/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "PRIVATE_KEY")
    logs, requests = [], []
    raw = json.dumps({"status": "no", "note": "a" * 341})

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "tool_calls": [{"function": {"name": "submit_result", "arguments": raw}}]}}]})

    frame = RunLog(tmp_path).frame(np.zeros((4, 4, 3), np.uint8), "c", "stow")
    backend = LLMBackend(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(CapabilityError, match=r"note: .*340.*actual=341"):
            backend.invoke("match_grasp", {"model": "test", "max_tokens": 1000, "timeout_s": 1, "attempts": 2},
                           "short", CapabilityInput(frame=frame), Match,
                           lambda event, **kw: logs.append({"event": event, **kw}))
    finally:
        backend.close()
    errors = [e for e in logs if e["event"] == "invalid_result"]
    assert len(errors) == 2 and all(e["raw_result"] == raw for e in errors)
    assert errors[0]["validation_errors"][0]["actual_length"] == 341
    assert "actual=341" in requests[1]["messages"][0]["content"]
    assert "340 个字符" in requests[0]["messages"][0]["content"]
    assert "PRIVATE_KEY" not in json.dumps(logs + requests)


def test_image_order_roles_and_public_path_clearance(tmp_path):
    import base64
    import io
    from PIL import Image

    log = RunLog(tmp_path)
    before = log.frame(np.full((4, 4, 3), 30, np.uint8), "c", "stow")
    after = log.frame(np.full((4, 4, 3), 220, np.uint8), "c", "stow")
    action = [command("turn_left", seconds=.2)]
    value = CapabilityInput(frame=after, comparison=before, commands=action,
                            target=Target.model_validate(TARGET))
    for evidence in ([], [{"id": "sample", "kind": "rotation", "direction": "turn_left",
                          "frames": [before.model_dump(), after.model_dump()], "commands": action}]):
        content, count = image_content(value.model_copy(update={"evidence": evidence}))
        assert count == 2
        captions = [c["text"] for c in content[1:] if c["type"] == "text"]
        assert "BEFORE" in captions[0] and before.id in captions[0]
        assert "AFTER CURRENT" in captions[1] and after.id in captions[1]
        if evidence:
            assert "REFERENCE sample AFTER" in captions[1]
        pixels = [np.array(Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1]))))
                  for c in content if c["type"] == "image_url"]
        assert pixels[0].mean() < 40 and pixels[1].mean() > 210
        context = json.loads(content[0]["text"])
        assert context["target"]["path"] == "clear" and "path" not in context["frame"]
        assert before.path not in content[0]["text"]
    assert "x向右、y向下" in COMMON_PROMPT


def test_detection_relocalizes_target_without_receiving_previous_coordinates(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()
    old = runner.brain.state.target
    new_target = {**TARGET, "center": {"x": .65, "y": .72}}
    seen = []
    def detection(value):
        seen.append(value)
        return {"status": "visible", "target": new_target}
    runner.brain.capabilities.overrides["find_ball"] = detection
    runner.scenario.calls.clear()
    runner.step()
    assert seen[0].previous_target.model_dump() == {"label": old.label}
    assert seen[0].target is None and seen[0].evidence == []
    matched = next(v for n, v in runner.scenario.calls if n == "match_grasp")
    assert matched.target.center.x == .65
    assert matched.frame.id == seen[0].frame.id
    assert runner.brain.state.target.center.x == .65
    runner.brain.capabilities.overrides["find_ball"] = lambda v: {"status": "absent"}
    runner.step()
    assert runner.brain.state.target is None


def test_old_replay_strips_coordinates_but_live_input_rejects_them(tmp_path):
    frame = RunLog(tmp_path).frame(np.zeros((4, 4, 3), np.uint8), "c", "stow")
    old = {"frame": frame.model_dump(), "previous_target": TARGET}
    with pytest.raises(ValidationError):
        CapabilityInput.model_validate(old)
    assert CapabilityInput.from_recorded(old).previous_target.model_dump() == {"label": TARGET["label"]}
    assert old["previous_target"] == TARGET


def test_legacy_motion_analysis_is_archived_and_not_reused(tmp_path):
    store = ExperienceStore(tmp_path, "c")
    log = RunLog(tmp_path)
    frame = log.frame(np.zeros((4, 4, 3), np.uint8), "c", "reach")
    store.add(InfoRequest(kind="reach"), [frame], [], ReachAnalysis(valid=True, region=REGION, note="old"))
    analysis = MotionAnalysis(valid=True, landmark="box", before={"x": .6, "y": .5},
                              after={"x": .4, "y": .5}, note="reversed")
    store.add(direction_request("turn_left"), [frame, frame], [command("turn_left", seconds=.2)], analysis)
    legacy = {"config_id": "c", "samples": store.samples}
    store.path.write_text(json.dumps(legacy), encoding="utf-8")
    migrated = ExperienceStore(tmp_path, "c")
    assert migrated.query(direction_request("turn_left")) == []
    reach = migrated.query(InfoRequest(kind="reach"))[0]
    assert reach["needs_analysis"] is True
    assert reach["analysis"]["valid"] is False and reach["analysis"]["region"] == []
    assert reach["frames"][0] == frame.model_dump() and Path(frame.path).is_file()
    archives = list(tmp_path.glob("experience-before-camera-*.json"))
    assert len(archives) == 1 and json.loads(archives[0].read_text(encoding="utf-8")) == legacy
    assert ExperienceStore(tmp_path, "c").samples == migrated.samples
    assert len(list(tmp_path.glob("experience-before-camera-*.json"))) == 1
    migrated.add(direction_request("turn_left"), [frame, frame], [command("turn_left", seconds=.2)],
                 analysis.model_copy(update={"before": analysis.after, "after": analysis.before}))
    assert migrated.query(direction_request("turn_left"))[0]["image_rate"]["normalized_dx_per_second"] > 0


def test_origin_reanalyzes_migrated_reach_once_using_the_saved_photo(runner):
    runner.initialize()
    original = runner.brain.memory.samples[0]
    original["needs_analysis"] = True
    original["analysis"] = {"valid": False, "region": [], "note": "old protocol"}
    runner.brain.memory.save()
    runner.brain.reset()
    runner.history = []
    runner.scenario.calls.clear()
    runner.step()  # stow
    assert runner.step()[0][0] == "open_gripper"
    calls = [v for n, v in runner.scenario.calls if n == "analyze_reach"]
    assert len(calls) == 1 and calls[0].frame.id == original["frames"][0]["id"]
    assert "needs_analysis" not in original
    assert original["analysis"]["valid"] is True
    loaded = ExperienceStore(runner.brain.root, runner.brain.memory.config_id)
    assert "needs_analysis" not in loaded.samples[0]


def test_collector_preserves_and_displays_actual_schema_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_collect
    def fail(*args):
        raise CapabilityError("match_grasp: note: String should have at most 340 characters (actual=341)")
    env = SimpleNamespace(model=None, data=SimpleNamespace(time=0),
                          robot=SimpleNamespace(base=None, arm=None, hal=SimpleNamespace(attached_object=lambda: None)),
                          get_obs=lambda: {}, success=lambda: False)
    brain = SimpleNamespace(name="test_failure", requires_final_check=True, decide=fail)
    monkeypatch.setattr(run_collect, "ToolLayer", lambda *a, **k: None)
    result = run_collect.run_episode(env, brain, [], "test", tmp_path / "episode", save_images=False)
    assert result["stop_code"] == "brain_error" and "actual=341" in result["error"]
    status = run_collect.langgraph_status(TaskState(phase="empty"), result)
    assert "Stop: brain_error" in status and "actual=341" in status and "match_grasp" in status


def test_ball_requires_whole_object_circle_and_partial_visibility_is_explicit():
    assert BallDetection(status="visible", target=TARGET).target.radius == .05
    for missing in ("radius", "visibility"):
        incomplete = {k: v for k, v in TARGET.items() if k != missing}
        with pytest.raises(ValidationError):
            BallDetection(status="visible", target=incomplete)
    partial = {**TARGET, "visibility": "partial", "center": {"x": .5, "y": .99}, "radius": .2}
    assert BallDetection(status="visible", target=partial).target.visibility == "partial"
    with pytest.raises(ValidationError):
        BallDetection(status="visible", target={**TARGET, "radius": -1})


def test_motion_reuses_exact_detections_then_passes_spatial_assessment_to_planner(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()  # First detection and forward probe.
    old_frame = runner.brain.state.frame.id
    fresh = {**TARGET, "center": {"x": .54, "y": .75}, "radius": .06}
    runner.brain.capabilities.overrides["find_ball"] = lambda v: {"status": "visible", "target": fresh}
    runner.brain.capabilities.overrides["analyze_motion"] = lambda v: {
        "valid": True, "source": "shared_target", "landmark": "same ball", "note": "both circles checked"}
    runner.scenario.calls.clear()
    runner.step()
    sample = runner.brain.memory.query(direction_request("forward"))[0]
    assert sample["analysis"]["before"] == TARGET["center"]
    assert sample["analysis"]["after"] == fresh["center"]
    assert sample["observation"]["before_frame_id"] == old_frame
    assert sample["observation"]["after_frame_id"] == runner.brain.state.frame.id
    assert sample["image_rate"]["radius_ratio"] == pytest.approx(1.2)
    assert sample["image_rate"]["normalized_dy_per_second"] == pytest.approx(.25)
    plan = next(v for n, v in runner.scenario.calls if n == "plan_motion")
    assert plan.target.model_dump() == fresh
    assert plan.assessment.distance == "far" and plan.assessment.alignment == "aligned"
    assert {s["kind"] for s in plan.evidence} == {"translation", "reach"}
    events = [json.loads(s) for s in (runner.brain.log.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [e for e in events if e["event"] == "capability_input" and e["input"]["frame"]["id"] == runner.brain.state.frame.id]
    assert [c["capability"] for c in calls][:2] == ["find_ball", "analyze_motion"]
    assert calls[1]["input"]["target"] == fresh
    assert calls[1]["input"]["comparison_target"]["center"] == TARGET["center"]


def test_shared_motion_cannot_invent_new_coordinates_or_learn_from_missing_target(runner):
    with pytest.raises(ValidationError, match="before/after must be null"):
        MotionReview(valid=True, source="shared_target", landmark="ball", note="",
                     before={"x": .1, "y": .2}, after={"x": .2, "y": .3})
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()
    runner.brain.capabilities.overrides["find_ball"] = lambda v: {"status": "absent"}
    runner.brain.capabilities.overrides["analyze_motion"] = lambda v: {
        "valid": True, "source": "shared_target", "landmark": "ball", "note": "invented"}
    with pytest.raises(ValueError, match="requires both frame detections"):
        runner.step()
    assert not runner.brain.memory.query(direction_request("forward"))


def test_unreliable_or_pushed_target_sample_is_not_saved(runner):
    runner.scenario.match = "no"
    runner.initialize()
    runner.step()
    runner.brain.capabilities.overrides["analyze_motion"] = lambda v: {
        "valid": False, "source": "shared_target", "landmark": "ball", "note": "may have been pushed"}
    runner.step()
    assert not runner.brain.memory.query(direction_request("forward"))


def test_repeated_request_for_delivered_reference_observes_then_stops_clearly(runner):
    runner.initialize()
    seen = []
    def insist(value):
        seen.append(value)
        return {"status": "need_info", "request": {"kind": "reach"},
                "alignment": "unknown", "distance": "unknown"}
    runner.brain.capabilities.overrides["match_grasp"] = insist
    assert runner.step() == [("observe", {})]
    assert len(seen) == 2 and seen[-1].info_feedback and seen[-1].evidence
    assert runner.step() == [("observe", {})]
    with pytest.raises(RuntimeError, match="提醒后仍重复申请"):
        runner.step()
    assert len(seen) == 6  # One call and one reminder per new frame; no same-input loop.


def test_gui_status_avoids_unrenderable_chinese_and_points_to_full_report():
    from run_collect import langgraph_status
    outcome = {"stop_code": "brain_error", "error": "RuntimeError: 提醒后仍重复申请"}
    text = langgraph_status(TaskState(phase="empty"), outcome)
    assert text.isascii() and "Reference supplied" in text and "review/index.html" in text
    assert outcome["error"] == "RuntimeError: 提醒后仍重复申请"
