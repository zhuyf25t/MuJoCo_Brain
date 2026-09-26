"""Graph/collector contracts with mock replies, NOT evidence of visual skill."""

from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

import config
from brains.langgraph_brain import LangGraphBrain, initial_state
from brains.llm_common import encode_image_block
from control.tools import ToolLayer


def action(tool, **args):
    return {"tool": tool, "args": args}


def plan(*actions, phase="explore", holding="empty", arm_view="other", **extra):
    return {"phase": phase, "holding": holding, "arm_view": arm_view, "reason": "测试用的当前图依据",
            "expected": "下一轮检查动作段的预期变化", "actions": list(actions) or [action("observe")], **extra}


GRASP = {"region": [.44, .72, .56, .88], "note": "测试假设：实体张爪接近竖直、靠近地面"}


CLEARANCE = {"save_current": True, "note": "爪已离地让开视线，肘未调整"}
RELEASE = {"above_rim": True, "inside_opening": True, "evidence": "测试假设：接近前后图确认球越过近沿且高于箱口"}


def workflow():
    return [
        plan(action("open_gripper"), action("arm_pose", pose="reach"), holding="unclear"),
        plan(action("shoulder", delta=-.3), action("shoulder", delta=-.2),
             arm_view="grasp", grasp_reference=GRASP, learning="当前低位看见两指与地面"),
        plan(action("shoulder", delta=.2), action("shoulder", delta=.3),
             arm_view="clearance", clearance_reference=CLEARANCE),
        plan(action("shoulder", delta=-.3), action("shoulder", delta=-.2), action("turn_left", seconds=.2),
             action("forward", seconds=.9), phase="pick", arm_view="grasp"),
        plan(action("forward", seconds=.1), action("open_gripper"), action("shoulder", delta=.2),
             action("shoulder", delta=.3), action("close_gripper"), action("shoulder", delta=-.2),
             phase="pick", arm_view="clearance", target={"kind": "ball", "note": "中央球"}),
        plan(action("turn_right", seconds=.3), action("forward", seconds=.8), action("shoulder", delta=-.1),
             action("elbow", delta=.2), phase="place", holding="held", target={"kind": "bin", "note": "接近前的箱子图"}),
        plan(action("open_gripper"), action("shoulder", delta=-.1), phase="place", holding="held", release_check=RELEASE),
        plan(action("done", success=True), phase="place"),
    ]


def obs(value=0):
    return {"images": {"front": np.full((12, 16, 3), value, np.uint8)}, "holding": "PRIVATE_HOLDING"}


def schemas():
    return ToolLayer(None, None, None).schemas_anthropic()


@pytest.fixture
def make_brain(monkeypatch, tmp_path):
    for name, value in dict(OPENAI_MAX_TOKENS=8192, OPENAI_STREAM="false", OPENAI_THINKING="",
                            OPENAI_REASONING_EFFORT="", LLM_TERM_IMAGES=False).items():
        monkeypatch.setattr(config, name, value)
    brains = []

    def create(plans=None, *, db_path=None, raw=False):
        requests = []
        queue = iter(plans) if plans is not None else None

        def respond(request):
            assert request.url.host == "mock.invalid"
            requests.append(json.loads(request.content))
            result = next(queue) if queue is not None else plan(action("shoulder", delta=.1))
            message = {"content": json.dumps(result)} if raw else {"tool_calls": [
                {"function": {"name": "report_plan", "arguments": json.dumps(result)}}]}
            return httpx.Response(200, json={"choices": [{"message": message}]})

        folder = tmp_path / f"brain-{len(brains)}"
        brain = LangGraphBrain(base_url="https://mock.invalid/v1", api_key="test-only",
                              model="deepseek-test", transport=httpx.MockTransport(respond),
                              db_path=db_path or folder / "state.sqlite", trace_dir=folder / "traces")
        brains.append(brain)
        return brain, requests

    yield create
    for brain in brains:
        brain.close()


def tick(brain, history, value=0, observation=None):
    batch = brain.decide(observation or obs(value), "TASK", schemas(), history)
    history.extend({"tool": d.tool, "args": d.args, "ok": True, "result": "PRIVATE_RESULT"} for d in batch)
    return batch


def input_text(body):
    return json.loads(body["messages"][1]["content"][0]["text"])


def run_ticks(brain, history, n):
    return [tick(brain, history, i * 20) for i in range(n)]


def test_three_stages_whole_chunks_and_observed_bidirectional_routes(make_brain):
    brain, requests = make_brain(workflow())
    history, states = [], []
    for i in range(8):
        assert len(tick(brain, history, i * 20)) == [2, 2, 2, 4, 6, 4, 2, 1][i]
        states.append(brain.state)
    assert len(requests) == 8 and len(history) == 23
    assert set(brain.graph.get_graph().nodes) == {"__start__", "explore", "pick", "place", "__end__"}
    assert all(set(state) == set(initial_state()) for state in states)
    assert not any(states[1]["calibration"]["moves"].values())
    assert states[2]["calibration"]["moves"]["to_grasp"] is None
    moves = states[3]["calibration"]["moves"]
    assert moves["to_clearance"]["actions"] == [action("shoulder", delta=-.3), action("shoulder", delta=-.2)]
    assert moves["to_grasp"]["actions"] == [action("shoulder", delta=.2), action("shoulder", delta=.3)]
    assert moves["to_grasp"]["before"]["history_n"] == 4 and moves["to_grasp"]["after"]["history_n"] == 6
    assert states[4]["calibration"]["moves"] == moves
    assert states[4]["visual_memory"]["held"] is None and states[5]["visual_memory"]["held"]
    assert not any(states[6]["calibration"]["moves"].values())
    assert states[7]["visual_memory"]["held"] is None
    with sqlite3.connect(brain.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM brain_images").fetchone()[0] == 8
        assert conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id='main'").fetchone()[0] > 0
    traces = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["entry_node"] for r in traces] == ["explore"] * 4 + ["pick"] * 2 + ["place"] * 2
    for row in traces:
        for request in row["requests"]:
            for part in request["messages"][1]["content"]:
                if part["type"] == "image_url":
                    assert (brain.trace_dir / part["image_url"]["url"]).is_file()
    assert "test-only" not in json.dumps(traces)


def test_blocked_pushes_excluded_and_new_lessons_cannot_erase_verified_moves(make_brain):
    prefix = [plan(action("shoulder", delta=.15), grasp_reference=GRASP, arm_view="grasp"),
              plan(action("shoulder", delta=.3), arm_view="grasp"),
              plan(action("shoulder", delta=.5), arm_view="grasp"),
              plan(action("shoulder", delta=-.5), arm_view="grasp"),
              plan(action("shoulder", delta=.5), arm_view="clearance", clearance_reference=CLEARANCE),
              plan(action("shoulder", delta=-.5), arm_view="grasp", phase="pick")]
    extra = [plan(action("forward", seconds=.2), phase="pick", arm_view="clearance", learning=f"新球位置{i}") for i in range(5)]
    brain, requests = make_brain(prefix + extra)
    history = []
    for value in (0, 0, 0, 0, 80, 0):
        tick(brain, history, value)
    moves = deepcopy(brain.state["calibration"]["moves"])
    assert moves["to_clearance"]["actions"] == [action("shoulder", delta=-.5)]
    assert moves["to_clearance"]["before"]["history_n"] == 3
    assert moves["to_grasp"]["actions"] == [action("shoulder", delta=.5)]
    for i in range(5): tick(brain, history, 100 + i * 20)
    assert brain.state["calibration"]["moves"] == moves
    assert len(brain.state["calibration"]["lessons"]) == 4
    assert all("arm_offset" not in input_text(r)["状态"] for r in requests)


def test_one_way_prediction_cannot_enter_pick(make_brain):
    bad = plan(phase="pick", arm_view="clearance", clearance_reference=CLEARANCE)
    brain, requests = make_brain(workflow()[:2] + [bad, workflow()[2], workflow()[3]])
    history = []
    run_ticks(brain, history, 3)
    assert brain.state["phase"] == "explore" and len(requests) == 4
    tick(brain, history, 80)
    assert brain.state["phase"] == "pick"


@pytest.mark.parametrize("command", [action("elbow", delta=.1), action("arm_pose", pose="stow")])
def test_changed_elbow_invalidates_moves_and_requires_reexploration(command, make_brain):
    brain, requests = make_brain(workflow()[:4] + [plan(command, phase="pick"), plan(command), plan(phase="pick"), plan()])
    history = []
    run_ticks(brain, history, 6)
    assert len(requests) == 8 and brain.state["phase"] == "explore"
    assert brain.state["arm_anchor"] is None and not any(brain.state["calibration"]["moves"].values())
    assert brain.state["calibration"]["grasp"] and brain.state["calibration"]["clearance"]


@pytest.mark.parametrize("movement", [action("turn_left", seconds=.15), action("forward", seconds=.1),
                                      action("shoulder", delta=-.1), action("elbow", delta=.1)])
def test_positioning_and_release_need_a_new_image(movement, make_brain):
    bad = plan(movement, action("open_gripper"), phase="place", holding="held", release_check=RELEASE)
    brain, requests = make_brain(workflow()[:6] + [bad, plan(movement, phase="place", holding="held"), workflow()[6]])
    batches = run_ticks(brain, [], 8)
    assert len(requests) == 9 and all(d.tool != "open_gripper" for d in batches[6])
    assert batches[7][0].tool == "open_gripper"


@pytest.mark.parametrize("check", [None, {**RELEASE, "above_rim": False}, {**RELEASE, "inside_opening": False},
                                  {**RELEASE, "evidence": ""}, {**RELEASE, "above_rim": 1}])
def test_release_requires_current_explicit_assessment(check, make_brain):
    bad = plan(action("open_gripper"), phase="place", holding="held", release_check=check)
    brain, requests = make_brain(workflow()[:6] + [bad, plan(phase="place", holding="held")])
    assert run_ticks(brain, [], 7)[-1][0].tool == "observe" and len(requests) == 8


def test_only_rotating_does_not_establish_bin_depth(make_brain):
    locate = plan(action("turn_right", seconds=.3), phase="place", holding="held", target={"kind": "bin", "note": "刚找到箱子"})
    brain, requests = make_brain(workflow()[:5] + [locate, workflow()[6],
                                plan(action("forward", seconds=.2), phase="place", holding="held"), workflow()[6]])
    batches = run_ticks(brain, [], 8)
    assert len(requests) == 9 and batches[6][0].tool == "forward" and batches[7][0].tool == "open_gripper"


def test_unchanged_bin_image_cannot_certify_release(make_brain):
    brain, requests = make_brain(workflow()[:6] + [workflow()[6], plan(phase="place", holding="held")])
    history = []
    run_ticks(brain, history, 6)
    assert tick(brain, history, 100)[0].tool == "observe" and len(requests) == 8


def test_unknown_hold_never_releases_or_returns_to_pick(make_brain):
    recovery = plan(action("shoulder", delta=.1), phase="place", holding="unclear")
    brain, requests = make_brain(workflow()[:6] + [plan(action("open_gripper"), phase="place", holding="unclear"),
                            recovery, plan(phase="pick", holding="unclear"), recovery])
    batches = run_ticks(brain, [], 8)
    assert len(requests) == 10 and all(d.tool != "open_gripper" for batch in batches[6:] for d in batch)
    assert brain.state["visual_memory"]["held"]


def test_private_truth_never_changes_requests_or_state(make_brain):
    a, req_a = make_brain(workflow()); b, req_b = make_brain(workflow())
    ha, hb = [], []
    for i in range(8):
        changed = obs(i * 20)
        changed.update(base_pose=[1, 2, 3], arm_qpos=[99, 88], tcp_pos=[1, 2, 3], holding=True)
        changed["images"]["overhead"] = np.zeros((40, 40, 3), np.uint8)
        for entry in hb: entry.update(ok=False, result="SECRET_SIM_TRUTH", thought="PRIVATE_THOUGHT")
        assert tick(a, ha, i * 20) == tick(b, hb, observation=changed)
    assert req_a == req_b and a.state == b.state and "SECRET_SIM_TRUTH" not in json.dumps(req_b)
    assert encode_image_block(changed["images"]["front"], fmt="openai") in req_b[-1]["messages"][1]["content"]
    snapshot = a.state; snapshot["calibration"]["grasp"]["note"] = "changed"
    assert a.state["calibration"]["grasp"]["note"] != "changed"


@pytest.mark.parametrize("change", ["partial", "extra", "reordered"])
def test_history_must_match_whole_batch_before_request(change, make_brain):
    brain, requests = make_brain(workflow()); history = []
    tick(brain, history); before = brain.state
    if change == "partial": history.pop()
    elif change == "extra": history.append(action("observe"))
    else: history[0], history[1] = history[1], history[0]
    with pytest.raises(RuntimeError, match="history"): brain.decide(obs(120), "TASK", schemas(), history)
    assert len(requests) == 1 and brain.state == before


def test_reset_and_lock_keep_episodes_independent(make_brain):
    brain, _ = make_brain(workflow()); history = []
    run_ticks(brain, history, 4); archive, before = brain.trace_dir, brain.state
    b, req_b = make_brain(db_path=brain.db_path)
    with pytest.raises(RuntimeError, match="已有运行实例"): b.decide(obs(), "TASK", schemas(), [])
    assert brain.state == before and not req_b
    brain.reset(); assert brain.state == {}; brain.close()
    fresh, _ = make_brain(db_path=brain.db_path); tick(fresh, [], 200)
    assert fresh.state["calibration"] == initial_state()["calibration"]
    with sqlite3.connect(brain.db_path) as conn: assert conn.execute("SELECT COUNT(*) FROM brain_images").fetchone()[0] == 1
    assert (archive / "rounds.jsonl").exists()


@pytest.mark.parametrize("invalid", [
    plan(action("shoulder", delta=.1), action("forward", seconds=0)), plan(action("shoulder", delta=float("nan"))),
    plan(action("done", success="false")), plan(action("done", success=True)), plan(action("close_gripper"), action("open_gripper")),
    plan(action("open_gripper"), action("done", success=True)), plan(target="not an object"),
    plan(grasp_reference=GRASP, holding="unclear", arm_view="grasp"), plan(grasp_reference=GRASP, arm_view="grasp", clearance_reference=CLEARANCE),
    plan(grasp_reference={"region": [.5, .5, .4, .8], "note": "反框"}, arm_view="grasp"),
    plan(phase="pick", grasp_reference=GRASP, arm_view="grasp"), plan(phase="place", holding="unclear"), plan(arm_view="made-up")])
def test_bad_plan_is_atomic_and_fallback_does_not_end_episode(invalid, make_brain):
    brain, requests = make_brain([invalid, invalid, plan(action("shoulder", delta=.2))]); history = []
    assert tick(brain, history)[0].tool == "observe" and len(requests) == 2
    assert brain.state["calibration"] == initial_state()["calibration"]
    assert tick(brain, history, 40)[0].tool == "shoulder" and len(requests) == 3


@pytest.mark.parametrize("commands", [[action("observe")], [action("shoulder", delta=-.3), action("elbow", delta=-.2)],
                                      [action("shoulder", delta=.3), action("shoulder", delta=-.5)]])
def test_mixed_or_reversing_actions_cannot_certify_a_shoulder_route(commands, make_brain):
    brain, requests = make_brain([plan(*commands, grasp_reference=GRASP, arm_view="grasp"),
                                 plan(arm_view="clearance", clearance_reference=CLEARANCE), plan()])
    run_ticks(brain, [], 2)
    assert len(requests) == 3 and brain.state["calibration"]["clearance"] is None


def test_same_direction_return_is_rejected(make_brain):
    brain, requests = make_brain(workflow()[:2] + [plan(action("shoulder", delta=-.2), arm_view="clearance", clearance_reference=CLEARANCE),
                                                 plan(arm_view="grasp", phase="pick"), plan()])
    run_ticks(brain, [], 4)
    assert len(requests) == 5 and brain.state["calibration"]["moves"]["to_grasp"] is None


def test_closed_gripper_cannot_save_open_geometry(make_brain):
    brain, requests = make_brain(workflow()[:5] + [plan(action("open_gripper"), grasp_reference=GRASP, arm_view="grasp"),
                                                plan(action("shoulder", delta=.1), phase="pick", holding="unclear")])
    history = []; run_ticks(brain, history, 5); saved = deepcopy(brain.state["calibration"]["grasp"])
    assert tick(brain, history, 120)[0].tool == "shoulder" and len(requests) == 7
    assert brain.state["calibration"]["grasp"] == saved


def test_missing_image_http_error_and_old_history_never_replay_actions(make_brain):
    brain, requests = make_brain([plan()], raw=True)
    with pytest.raises(ValueError, match="images.front"): brain.decide({}, "TASK", schemas(), [])
    assert not brain.db_path.exists()
    history = []; tick(brain, history)
    with pytest.raises(RuntimeError, match="history"): brain.decide(obs(), "TASK", schemas(), [])
    assert len(requests) == 1
    before = brain.state; brain.llm._client.close()
    brain.llm._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))
    with pytest.raises(httpx.HTTPStatusError): brain.decide(obs(30), "TASK", schemas(), history)
    assert brain.state == before
    last = json.loads((brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["decisions"] == [] and last["requests"][0]["error"] == "HTTPStatusError"


def test_observe_preserves_arm_and_records_ticks():
    drive, ticks = [], []
    robot = SimpleNamespace(command_drive=lambda *args: drive.append(args), step_ticks=lambda n, callback: [callback() for _ in range(n)])
    ok, _ = ToolLayer(None, None, robot, on_tick=lambda: ticks.append(True)).execute("observe", {})
    assert ok and drive == [(0.0, 0.0)] and len(ticks) == 2


def test_factory_selects_langgraph(monkeypatch):
    from run_collect import build_brain
    sentinel = object()
    monkeypatch.setattr("brains.langgraph_brain.LangGraphBrain", lambda: sentinel)
    assert build_brain("langgraph", None) is sentinel


def test_reset_same_instance_does_not_carry_routes_into_next_episode(make_brain):
    brain, requests = make_brain(workflow() + workflow()[:2])
    history = []
    run_ticks(brain, history, 8)
    old_trace = brain.trace_dir
    brain.reset()
    history = []
    run_ticks(brain, history, 2)
    assert len(requests) == 10 and brain.trace_dir != old_trace
    assert not any(brain.state["calibration"]["moves"].values())
    assert brain.state["calibration"]["clearance"] is None
    assert brain.state["visual_memory"]["held"] is None
    assert input_text(requests[8])["状态"] == initial_state()
    assert (old_trace / "rounds.jsonl").exists()


def test_visible_open_geometry_can_be_saved_when_initial_hold_is_unclear(make_brain):
    brain, requests = make_brain([
        plan(action("open_gripper"), holding="unclear"),
        plan(action("shoulder", delta=-.3), holding="unclear", arm_view="grasp", grasp_reference=GRASP),
        plan(action("shoulder", delta=.3), holding="unclear", arm_view="clearance", clearance_reference=CLEARANCE),
        plan(action("shoulder", delta=-.3), holding="unclear", arm_view="grasp", phase="pick")])
    run_ticks(brain, [], 4)
    assert len(requests) == 4 and brain.state["phase"] == "pick"
    assert brain.state["visual_memory"]["held"] is None


def test_no_visual_change_cannot_certify_a_lift(make_brain):
    brain, requests = make_brain([
        plan(action("shoulder", delta=-.3), arm_view="grasp", grasp_reference=GRASP),
        plan(arm_view="clearance", clearance_reference=CLEARANCE), plan(arm_view="grasp")])
    history = []
    tick(brain, history, 0)
    tick(brain, history, 0)
    assert len(requests) == 3 and brain.state["calibration"]["clearance"] is None


def test_current_image_is_last_even_when_identical_to_a_historical_reference(make_brain):
    brain, requests = make_brain([workflow()[0], workflow()[1], plan(), plan()])
    history = []
    for value in (0, 20, 20, 60):
        tick(brain, history, value)
    for request, value in zip(requests, (0, 20, 20, 60)):
        content = request["messages"][1]["content"]
        assert content[-1] == encode_image_block(obs(value)["images"]["front"], fmt="openai")
        assert content[-2]["text"].startswith("唯一当前图")
        blocks = [p for p in content if p["type"] == "image_url"]
        assert len(blocks) == len({p["image_url"]["url"] for p in blocks})
    # Same pixels as the low reference do not make the latest observation historical.
    assert "已调用动作数=4" in requests[2]["messages"][1]["content"][-2]["text"]
