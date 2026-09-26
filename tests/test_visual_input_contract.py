"""Verify the OpenAI request boundary without changing simulation feedback."""

import copy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import config
from brains import OpenAICompatBrain
from brains.llm_common import encode_image_block
from control import perception
from control.drivers import ArmDriver
from control.tools import ToolLayer


@pytest.fixture
def captured_brain(monkeypatch):
    # Never use credentials or settings from the project's private .env.
    for name, value in dict(OPENAI_MAX_TOKENS=8192, OPENAI_STREAM="false",
                            OPENAI_THINKING="", OPENAI_REASONING_EFFORT="",
                            LLM_TERM_IMAGES=False).items():
        monkeypatch.setattr(config, name, value)
    requests = []

    def respond(request):
        assert request.url.host == "mock.invalid"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "观察后决定", "tool_calls": [{"function": {
                "name": "close_gripper", "arguments": "{}"}}]}}]})

    brain = OpenAICompatBrain(base_url="https://mock.invalid/v1", api_key="test-only",
                              model="mock-model", transport=httpx.MockTransport(respond))
    try:
        yield brain, requests
    finally:
        brain._client.close()


def schemas():
    return ToolLayer(None, None, None).schemas_anthropic()


def images():
    return {"front": np.full((24, 32, 3), [120, 60, 30], dtype=np.uint8)}


def test_wire_request_is_invariant_to_privileged_state_and_results(captured_brain, monkeypatch):
    brain, requests = captured_brain

    def forbidden(*args, **kwargs):
        pytest.fail("OpenAI visual input must not run legacy perception")

    monkeypatch.setattr(perception, "analyze", forbidden)
    monkeypatch.setattr(perception, "sense", forbidden)
    obs = {"images": images(), "base_pose": [111, 222, 333],
           "arm_qpos": [444, 555], "tcp_pos": [666, 777, 888],
           "finger": 1, "holding": False, "qpos": [999],
           "camera_geometry": {"sentinel": "PRIVATE_CAMERA_POSE"},
           "ball_positions": "PRIVATE_BALL_TRUTH", "success": False}
    history = [{"tool": "forward", "args": {"seconds": .2}, "ok": True,
                "result": "PRIVATE_RESULT: 当前位姿 [1,2,3]; 成功夹到球",
                "thought": "PRIVATE_THOUGHT", "holding": True}]
    brain.decide(obs, "把球放进箱子", schemas(), history)
    original = requests[-1]

    changed = copy.deepcopy(obs)
    for key in changed:
        if key != "images":
            changed[key] = "DIFFERENT_PRIVATE_STATE"
    changed["images"].update(overhead=np.zeros((24, 32, 3), dtype=np.uint8),
                             side=np.ones((24, 32, 3), dtype=np.uint8))
    changed_history = copy.deepcopy(history)
    changed_history[0].update(ok=False, result="DIFFERENT_RESULT", thought="OTHER", holding=False)
    brain.decide(changed, "把球放进箱子", schemas(), changed_history)
    assert requests[-1] == original

    body = json.dumps(original, ensure_ascii=False)
    for forbidden_value in ("PRIVATE_", "DIFFERENT_", "0.55m", "±15%", "夹爪视角"):
        assert forbidden_value not in body
    parts = original["messages"][1]["content"]
    picture_blocks = [p for p in parts if p["type"] == "image_url"]
    assert picture_blocks == [encode_image_block(obs["images"]["front"], fmt="openai")]
    assert original["tools"] == ToolLayer(None, None, None).schemas_openai()
    # The command itself remains available, while its real-world outcome is absent.
    assert '"seconds": 0.2' in "\n".join(p["text"] for p in parts if p["type"] == "text")


def test_current_image_task_and_bounded_command_history_reach_model(captured_brain):
    brain, requests = captured_brain
    history = [{"tool": "forward", "args": {"seconds": i / 10},
                "ok": False, "result": "MUST_NOT_SEND"}
               for i in range(config.HISTORY_LINES + 3)]
    obs = {"images": images()}  # No robot state is needed.
    brain.decide(obs, "TASK_ONE", schemas(), history)
    parts = requests[-1]["messages"][1]["content"]
    text = parts[-1]["text"]
    commands = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    assert commands == [{"tool": e["tool"], "args": e["args"]}
                        for e in history[-config.HISTORY_LINES:]]
    assert "TASK_ONE" in text and "MUST_NOT_SEND" not in text
    original_image = parts[1]

    obs["images"]["front"][:] = 255
    brain.decide(obs, "TASK_TWO", schemas(), [])
    parts = requests[-1]["messages"][1]["content"]
    assert parts[1] != original_image and "TASK_TWO" in parts[-1]["text"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 1
    assert "尚无动作指令记录" in parts[-1]["text"]


@pytest.mark.parametrize("obs", [{}, {"images": {}},
                                 {"images": {"overhead": np.zeros((4, 4, 3), np.uint8)}}])
def test_missing_front_stops_before_request(captured_brain, obs):
    brain, requests = captured_brain
    with pytest.raises(ValueError, match="images.front"):
        brain.decide(obs, "任务", schemas(), [])
    assert not requests


def test_parse_retry_keeps_front_only_contract(captured_brain):
    brain, _ = captured_brain
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        content = "请重试" if len(bodies) == 1 else '{"tool":"done","args":{"success":false}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    brain._client.close()
    brain._client = httpx.Client(transport=httpx.MockTransport(respond))
    obs = {"images": {**images(), "overhead": np.zeros((4, 4, 3), np.uint8)},
           "holding": "PRIVATE_HOLDING"}
    brain.decide(obs, "任务", schemas(), [{"tool": "close_gripper", "args": {},
                                        "result": "PRIVATE_RESULT", "ok": True}])
    assert len(bodies) == 2
    for body in bodies:
        assert "PRIVATE_" not in json.dumps(body)
        assert sum(p["type"] == "image_url" for p in body["messages"][1]["content"]) == 1


@pytest.mark.parametrize("tool,goal", [("open_gripper", config.FINGER_OPEN_CTRL),
                                      ("close_gripper", config.FINGER_CLOSE_CTRL)])
def test_original_gripper_feedback_is_kept_locally_but_not_sent(
        tool, goal, captured_brain, monkeypatch):
    brain, requests = captured_brain
    commands, ticks, callbacks = [], [], []
    # The original tool is allowed to generate a truth-dependent visual hint.
    # The API boundary must suppress that result without altering the tool.
    monkeypatch.setattr(perception, "sense", lambda robot: {})
    monkeypatch.setattr(perception, "gripper_hint", lambda state: "PRIVATE_GRASP_HINT")

    def step_ticks(n, callback):
        ticks.append(n)
        for _ in range(n):
            callback()

    robot = SimpleNamespace(
        arm_geometry=lambda: (.38, .40, np.array([-.3, -2.8]), np.array([2.2, 2.6])),
        finger_ctrl_range=lambda: (0., .05), command_finger=commands.append,
        step_ticks=step_ticks, attached_object=lambda: held)
    layer = ToolLayer(None, ArmDriver(robot), robot, on_tick=lambda: callbacks.append(True))
    results = []
    for held in (None, "ball_0"):
        ok, result = layer.execute(tool, {})
        assert ok
        results.append(result)
        brain.decide({"images": images()}, "把球放进箱子", schemas(),
                     [{"tool": tool, "args": {}, "ok": ok, "result": result}])
    assert commands == [goal, goal] and ticks == [8, 8] and len(callbacks) == 16
    if tool == "close_gripper":
        assert "没有夹到球" in results[0] and "PRIVATE_GRASP_HINT" in results[0]
        assert "成功夹到球" in results[1]
    else:
        assert "球已释放" in results[0] and "球已释放" not in results[1]
    assert requests[-1] == requests[-2]
    body = json.dumps(requests[-1], ensure_ascii=False)
    assert all(word not in body for word in ("成功夹到", "没有夹到", "球已释放", "PRIVATE_GRASP_HINT"))
