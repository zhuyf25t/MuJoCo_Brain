"""openclaw_sim 冒烟与单元测试.

运行: ../../.venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


# ============================ 模型与组合 ============================
def test_model_loads():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(config.SCENE_XML))
    probes = (m.joint, m.body, m.site, m.actuator, m.camera)
    for name in (config.JOINT_BASE, config.JOINT_WHEEL_L, config.JOINT_WHEEL_R,
                 config.JOINT_SHOULDER, config.JOINT_ELBOW, config.JOINT_FINGER,
                 config.SITE_TCP, config.ACT_SHOULDER, config.ACT_ELBOW,
                 config.ACT_FINGER, config.CAM_FRONT, *config.BALL_NAMES):
        assert any(1 for p in probes if _ok(p, name)), f"缺少 {name}"


def _ok(fn, name) -> bool:
    try:
        fn(name)
        return True
    except KeyError:
        return False


# ============================ LLM Brain (mock) ============================
class _Resp:
    def __init__(self, payload):
        self.payload = payload


def test_openai_brain_toolcall():
    import httpx
    from brains import OpenAICompatBrain

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": "先去抓红方块",
                "tool_calls": [{"function": {
                    "name": "move_base",
                    "arguments": json.dumps({"dx": 0.8, "dy": 0.0})}}],
            }}]
        })

    brain = OpenAICompatBrain(base_url="http://mock/v1", api_key="k",
                              model="mock-model", transport=httpx.MockTransport(handler))
    dec = brain.decide(_fake_obs(), "把红色方块放进箱子", _fake_schemas(), [])[0]
    assert dec.tool == "move_base" and dec.args["dx"] == 0.8
    assert "先去抓红方块" in dec.thought
    # 请求结构
    assert captured["url"].endswith("/chat/completions")
    body = captured["body"]
    assert body["model"] == "mock-model"
    assert body["tools"][0]["type"] == "function"
    fn = body["tools"][0]["function"]
    assert fn["name"] and "properties" in fn["parameters"]
    user = body["messages"][1]["content"]
    assert any(p["type"] == "image_url" and
               p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in user)


def test_openai_brain_json_fallback():
    import httpx
    from brains import OpenAICompatBrain

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '思考中... ```json\n{"thought": "观察", "tool": "look", "args": {}}\n```'
        }}]})

    brain = OpenAICompatBrain(base_url="http://mock/v1", transport=httpx.MockTransport(handler))
    dec = brain.decide(_fake_obs(), "任务", _fake_schemas(), [])
    dec = dec[0] if isinstance(dec, list) else dec
    assert dec.tool == "look"


def test_openai_brain_retry_then_safe_fallback():
    import httpx
    from brains import OpenAICompatBrain
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "我不会调用工具"}}]})

    brain = OpenAICompatBrain(base_url="http://mock/v1", transport=httpx.MockTransport(handler))
    dec = brain.decide(_fake_obs(), "任务", _fake_schemas(), [])
    dec = dec[0] if isinstance(dec, list) else dec
    assert dec.tool == "forward" and calls["n"] == 2   # 重试一次后安全回退(原地重观察)


def test_anthropic_brain_tool_use():
    from brains import AnthropicBrain

    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Msgs:
        def create(self, **kw):
            captured.update(kw)
            return Block(content=[Block(type="text", text="好的"),
                                  Block(type="tool_use", name="grasp", input={})],
                         stop_reason="tool_use")

    captured: dict = {}
    brain = AnthropicBrain(api_key="k", model="claude-opus-5")
    brain.client = type("C", (), {"messages": Msgs()})()
    dec = brain.decide(_fake_obs(), "任务", _fake_schemas(), [])
    assert dec[0].tool == "grasp" and "好的" in dec[0].thought
    # 请求结构: 不带 temperature (新模型不支持), 工具用 input_schema
    assert "temperature" not in captured
    assert captured["tools"][0]["input_schema"]["type"] == "object"
    assert captured["system"] and captured["messages"][0]["role"] == "user"


def _fake_obs():
    import numpy as np
    return {"images": {"overhead": np.zeros((48, 64, 3), dtype=np.uint8)},
            "arm_qpos": [0.0] * 7, "gripper": 1.0,
            "base_pose": [0, 0, 0], "tcp_pos": [0, 0, 0]}


def _fake_schemas():
    return [{"name": "move_base", "description": "底盘移动",
             "input_schema": {"type": "object", "properties": {"dx": {"type": "number"}},
                              "required": ["dx"]}},
            {"name": "look", "description": "观察", "input_schema": {"type": "object"}},
            {"name": "grasp", "description": "抓取", "input_schema": {"type": "object"}}]


# ============================ 端到端冒烟 ============================
def test_scripted_episode_end_to_end(tmp_path):
    """1 集 ScriptedBrain 全流程 (无图加速): 决策→工具→记录→成功判定."""
    from sim.env import MobileManipEnv, parse_task
    from brains import ScriptedBrain
    from recorder import EpisodeRecorder
    from control.tools import ToolLayer
    from run_collect import run_episode  # noqa: E402

    env = MobileManipEnv()
    env.reset(seed=0, task=parse_task("0号网球"))
    brain = ScriptedBrain(env)
    schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()
    try:
        out = run_episode(env, brain, schemas, "把0号网球捡起来放进收纳箱",
                          tmp_path / "ep_test", save_images=False, verbose=False)
    finally:
        env.close()
    ep = tmp_path / "ep_test"
    assert (ep / "meta.json").exists()
    decisions = [json.loads(l) for l in (ep / "decisions.jsonl").read_text().splitlines()]
    assert 3 <= len(decisions) <= 20
    # 视觉闭环管线应完成抓取 (投放端可靠性问题单独跟踪, 见 README)
    assert any(r["tool"] == "close_gripper" for r in decisions)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
