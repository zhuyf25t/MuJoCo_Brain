"""Graph/collector contracts with mock replies, NOT evidence of visual skill."""

from collections import deque
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
from brains.langgraph_policy import CALIBRATION_COMPARE_PROMPT, CALIBRATION_VIEW_PROMPT
from brains.llm_common import encode_image_block
from control.tools import ToolLayer


def action(tool, **args):
    return {"tool": tool, "args": args}


def plan(*actions, phase="explore", holding="empty", arm_view="other", **extra):
    return {"phase": phase, "holding": holding, "arm_view": arm_view, "reason": "测试用的当前图依据",
            "expected": "下一轮检查动作段的预期变化", "actions": list(actions) or [action("observe")], **extra}


GRASP = {"region": [.44, .72, .56, .88], "note": "测试假设：实体张爪接近竖直、靠近地面"}


CLEARANCE = {"save_current": True, "note": "爪已离地让开视线，肘未调整"}
RELEASE = {"clear_drop_path": True, "inside_opening": True, "evidence": "测试假设：接近前后图确认球向下可落入箱内，不撞沿"}


def scene(holding="empty", release_view="unclear"):
    return {"ground_balls": [{"center": [.5, .3], "description": "上方地面的球"}],
            "gripper": "测试用的独立实体手指观测", "holding": holding,
            "release_view": release_view, "evidence": "测试用的当前图空间关系"}


class RequestLog(list):
    """Keep the existing planning-contract assertions separate from observation IO."""
    def __init__(self):
        super().__init__()
        self.observations = []
        self.gripper_observations = []


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

    def create(plans=None, *, db_path=None, raw=False, scenes=None, gripper_scenes=None):
        requests = RequestLog()
        queue = deque(plans) if plans is not None else None
        views = iter(scenes) if scenes is not None else None
        gripper_views = iter(gripper_scenes) if gripper_scenes is not None else None

        def respond(request):
            assert request.url.host == "mock.invalid"
            body = json.loads(request.content)
            name = body["tools"][0]["function"]["name"]
            if name == "describe_scene" and body["messages"][0]["content"] in {CALIBRATION_VIEW_PROMPT, CALIBRATION_COMPARE_PROMPT}:
                requests.gripper_observations.append(body)
                result = next(gripper_views) if gripper_views is not None else scene()
                if isinstance(result, Exception):
                    raise result
            elif name == "describe_scene":
                requests.observations.append(body)
                planned_hold = queue[0].get("holding", "empty") if queue else "empty"
                result = next(views) if views is not None else scene(planned_hold, "candidate" if planned_hold == "held" else "unclear")
            else:
                requests.append(body)
                result = queue.popleft() if queue is not None else plan(action("shoulder", delta=.1))
            message = {"content": json.dumps(result)} if raw else {"tool_calls": [
                {"function": {"name": name, "arguments": json.dumps(result)}}]}
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


def test_refine_ball_region_retains_verified_arm_routes(make_brain):
    correction = {"region": [.43, .5, .57, .6], "note": "同一低爪当前看见球仍在两指前方，修正候选位置"}
    refine = plan(action("shoulder", delta=-.5), phase="pick", arm_view="grasp", grasp_region=correction)
    brain, _ = make_brain(workflow()[:4] + [refine])
    history = []
    run_ticks(brain, history, 4)
    prior = brain.state["calibration"]
    tick(brain, history, 100)
    current = brain.state["calibration"]
    assert current["grasp"]["region"] == correction["region"]
    assert current["grasp"]["frame"]["image_id"] != prior["grasp"]["frame"]["image_id"]
    assert current["grasp"]["frame"] == brain.state["arm_anchor"]["frame"]
    assert current["clearance"] == prior["clearance"] and current["moves"] == prior["moves"]
    assert prior["grasp"]["region"] == GRASP["region"]


@pytest.mark.parametrize("case", ["raised", "new_pose", "no_ball", "elbow_changed", "closed"])
def test_region_correction_cannot_replace_new_pose_or_unseen_ball(case, make_brain):
    bad = plan(phase="pick", arm_view="grasp", grasp_region=GRASP)
    prefix = workflow()[:4]
    views = [scene() for _ in range(6)]
    if case == "raised":
        bad["arm_view"] = "clearance"
    elif case == "new_pose":
        bad["grasp_reference"] = GRASP
    elif case == "no_ball":
        views[4]["ground_balls"] = []
    elif case == "elbow_changed":
        prefix.append(plan(action("elbow", delta=.1)))
        bad["phase"] = "explore"
    elif case == "closed":
        prefix.append(plan(action("close_gripper"), phase="pick"))
    brain, requests = make_brain(prefix + [bad, bad], scenes=views)
    history = []
    run_ticks(brain, history, len(prefix))
    old_grasp = brain.state["calibration"]["grasp"]
    batch = tick(brain, history, 180)
    assert [d.tool for d in batch] == ["observe"]
    assert brain.state["calibration"]["grasp"] == old_grasp
    assert len(requests) == len(prefix) + 2


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


@pytest.mark.parametrize("check", [None, {**RELEASE, "clear_drop_path": False}, {**RELEASE, "inside_opening": False},
                                  {**RELEASE, "evidence": ""}, {**RELEASE, "clear_drop_path": 1}])
def test_release_requires_current_explicit_assessment(check, make_brain):
    bad = plan(action("open_gripper"), phase="place", holding="held", release_check=check)
    brain, requests = make_brain(workflow()[:6] + [bad, plan(phase="place", holding="held")])
    assert run_ticks(brain, [], 7)[-1][0].tool == "observe" and len(requests) == 8


@pytest.mark.parametrize("first_claim", ["held", "unclear", "empty"])
def test_release_retry_cannot_discard_required_positioning(first_claim, make_brain):
    release = workflow()[6]
    mixed = deepcopy(release)
    mixed["holding"] = first_claim
    mixed["actions"].insert(0, action("forward", seconds=.1))
    views = [scene() for _ in range(5)] + [scene("held", "candidate") for _ in range(3)]
    brain, requests = make_brain(workflow()[:6] + [mixed, release, release], scenes=views)
    batches = run_ticks(brain, [], 8)
    assert [d.tool for d in batches[6]] == ["observe"]
    assert batches[7][0].tool == "open_gripper"
    assert len(requests) == 9
    trace = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [request.get("validation_error", "") for request in trace[6]["requests"]]
    assert any("删除定位" in error for error in errors)
    assert any("没有新图" in error for error in errors)


@pytest.mark.parametrize("holding_claim", ["held", "unclear"])
def test_planner_cannot_release_when_independent_grip_is_obscured(holding_claim, make_brain):
    prefix = workflow()[:6]
    release = plan(action("open_gripper"), phase="place", holding=holding_claim, release_check=RELEASE)
    views = [scene() for _ in prefix]
    views[-1] = scene("held", "candidate")
    views.append(scene("unclear", "candidate"))
    retry = {**release, "holding": "held"}
    plans = prefix + [release, retry]
    brain, requests = make_brain(plans, scenes=views)
    batches = run_ticks(brain, [], len(prefix) + 1)
    assert [d.tool for d in batches[-1]] == ["observe"]
    assert len(requests) == len(prefix) + 2


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
    assert req_a.observations == req_b.observations
    assert "SECRET_SIM_TRUTH" not in json.dumps(req_b.observations)
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


def test_held_label_preserves_pick_close_and_lift_without_relaxing_release(make_brain):
    confirm = plan(action("close_gripper"), action("shoulder", delta=-.3), phase="pick", holding="held")
    release = plan(action("open_gripper"), phase="place", holding="held", release_check=RELEASE)
    views = [scene(p["holding"]) for p in workflow()[:4]] + [scene("held", "blocked")] * 2
    brain, requests = make_brain(workflow()[:4] + [confirm, release, plan(phase="place", holding="held")], scenes=views)
    history = []
    run_ticks(brain, history, 4)
    batch = tick(brain, history, 80)
    assert [d.tool for d in batch] == ["close_gripper", "shoulder"]
    assert len(requests) == 5 and brain.state["phase"] == "pick"  # No correction discards the grasp.
    assert brain.state["visual_memory"]["held"] is not None
    batch = tick(brain, history, 100)
    assert [d.tool for d in batch] == ["observe"] and len(requests) == 7
    assert brain.state["phase"] == "place" and brain.state["visual_memory"]["held"] is not None


@pytest.mark.parametrize("actions", [
    [action("observe")],
    [action("shoulder", delta=-.3), action("close_gripper")],
    [action("close_gripper"), action("forward", seconds=.2)],
    [action("close_gripper"), action("turn_right", seconds=.2)],
    [action("close_gripper"), action("open_gripper")],
])
def test_held_pick_exception_does_not_allow_transport_release_or_late_close(actions, make_brain):
    bad = plan(*actions, phase="pick", holding="held")
    corrected = plan(action("close_gripper"), action("shoulder", delta=-.3), phase="pick", holding="unclear")
    brain, requests = make_brain(workflow()[:4] + [bad, corrected])
    batches = run_ticks(brain, [], 5)
    assert [d.tool for d in batches[-1]] == ["close_gripper", "shoulder"]
    assert len(requests) == 6 and brain.state["phase"] == "pick"
    assert brain.state["visual_memory"]["held"] is not None


@pytest.mark.parametrize("boundary", ["explore", "last_close", "place"])
def test_held_pick_confirmation_requires_existing_pick_and_last_open(boundary, make_brain):
    if boundary == "explore":
        prefix = workflow()[:3]
    elif boundary == "last_close":
        prefix = workflow()[:4] + [plan(action("close_gripper"), phase="pick")]
    else:
        prefix = workflow()[:7]  # Has released; last gripper command is open, phase is place.
    bad = plan(action("close_gripper"), action("shoulder", delta=-.3),
               phase="pick", holding="held", arm_view="grasp" if boundary == "explore" else "other")
    brain, requests = make_brain(prefix + [bad, plan(phase="place", holding="held")])
    batches = run_ticks(brain, [], len(prefix) + 1)
    assert [d.tool for d in batches[-1]] == ["observe"]
    assert len(requests) == len(prefix) + 2 and brain.state["phase"] == "place"


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
        assert content[-2] == encode_image_block(obs(value)["images"]["front"], fmt="openai")
        assert content[-3]["text"].startswith("唯一当前图")
        assert content[-1]["text"].startswith("current_scene")
        blocks = [p for p in content if p["type"] == "image_url"]
        assert len(blocks) == len({p["image_url"]["url"] for p in blocks})
    # Same pixels as the low reference do not make the latest observation historical.
    assert "已调用动作数=4" in requests[2]["messages"][1]["content"][-3]["text"]


@pytest.mark.parametrize("verdict", ["blocked", "unclear"])
def test_independent_scene_denies_release_and_empty_claim_on_retry(verdict, make_brain):
    release = workflow()[6]
    claim_empty = plan(action("open_gripper"), phase="place", holding="empty")
    views = [scene(p["holding"], "candidate" if p["holding"] == "held" else "unclear") for p in workflow()[:6]]
    views.append(scene("unclear", verdict))
    brain, requests = make_brain(workflow()[:6] + [release, claim_empty], scenes=views)
    batches = run_ticks(brain, [], 7)
    assert [d.tool for d in batches[-1]] == ["observe"]
    assert brain.state["visual_memory"]["held"] and brain.state["phase"] == "place"
    assert len(requests.observations) == 7 and len(requests) == 8
    trace = json.loads((brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert trace["current_scene"]["release_view"] == verdict
    assert [r["kind"] for r in trace["requests"]] == ["observation", "plan", "plan"]


def test_observation_only_receives_current_image_and_reuses_identical_pixels(make_brain):
    brain, requests = make_brain([plan(action("shoulder", delta=.1), learning="OLD_TARGET_PRIVATE"), plan()])
    history = []
    tick(brain, history, 20)
    tick(brain, history, 20)
    assert len(requests.observations) == 1 and len(requests) == 2
    observation = requests.observations[0]
    assert len(observation["messages"]) == 2
    assert observation["messages"][1]["content"] == [
        {"type": "text", "text": "只描述这一张当前图，调用describe_scene。"},
        encode_image_block(obs(20)["images"]["front"], fmt="openai")]
    assert "TASK" not in json.dumps(observation) and "OLD_TARGET_PRIVATE" not in json.dumps(observation)
    rows = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["scene_reused"] and rows[-1]["current_scene"] == rows[0]["current_scene"]


def planner_scene(request):
    return json.loads(request["messages"][1]["content"][-1]["text"].split("：", 1)[1].split("\n", 1)[0])


def described_scene(holding="empty", release_view="unclear"):
    return {**scene(holding, release_view), "ground_balls": [{"description": "上方地面的球"}]}


@pytest.mark.parametrize("holding,release_view", [("empty", "unclear"), ("held", "blocked")])
def test_observer_coordinate_outlier_cannot_change_planner_input(holding, release_view, make_brain):
    requests_by_center = []
    for center in ([.46, .38], [.46, .70]):
        observed = scene(holding, release_view)
        observed["ground_balls"][0]["center"] = center
        brain, requests = make_brain(
            [plan(phase="place" if holding == "held" else "explore", holding=holding)], scenes=[observed])
        tick(brain, [], 20)
        requests_by_center.append(requests[0])
        assert planner_scene(requests[0]) == described_scene(holding, release_view)
        assert brain._scene_cache[1] == observed
        row = json.loads((brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert row["current_scene"] == observed
        assert brain.state["calibration"]["grasp"] is None
    # A large change in the observer's guessed center cannot become new evidence
    # of ball motion for the planner. The raw estimates remain available to audit.
    assert requests_by_center[0] == requests_by_center[1]


@pytest.mark.parametrize("probe_holding", ["empty", "unclear"])
def test_calibration_probe_only_projects_geometry_and_preserves_base_cache(probe_holding, make_brain):
    base = {**scene("unclear", "blocked"), "gripper": "BASE_SHADOW_IDENTITY", "evidence": "BASE_SHADOW_EVIDENCE"}
    probe = {**scene(probe_holding, "candidate"), "gripper": "TOP_REAL_JAWS_WITH_VISIBLE_OPENING",
             "ground_balls": [{"center": [.9, .9], "description": "WRONG_PROBE_BALL"}], "evidence": "WRONG_PROBE_RELEASE"}
    brain, requests = make_brain([plan(action("open_gripper")), plan(holding="unclear"), plan(holding="unclear")],
                                scenes=[scene(), base], gripper_scenes=[probe])
    history = []
    for value in (0, 20, 20):
        tick(brain, history, value)
    assert len(requests.observations) == 2 and len(requests.gripper_observations) == 1
    for request in requests[1:]:
        projected = planner_scene(request)
        assert projected["gripper"] == probe["gripper"]
        assert "evidence" not in projected
        assert projected["ground_balls"] == [{"description": "上方地面的球"}]
        for field in ("holding", "release_view"):
            assert projected[field] == base[field]
    assert brain._scene_cache[1] == base
    rows = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["current_scene"] == base
    assert brain.state["visual_memory"]["held"] is None


def test_initial_shoulder_comparison_uses_adjacent_images_without_predictions(make_brain):
    base = {**scene("unclear", "blocked"), "gripper": "BASE_GEOMETRY"}
    probe = {**scene("unclear", "candidate"), "gripper": "CURRENT_GEOMETRY_AND_DIRECT_CHANGE",
             "ground_balls": [{"center": [.9, .9], "description": "WRONG_PROBE_BALL"}]}
    plans = [workflow()[0],
             plan(action("shoulder", delta=-.2), arm_view="grasp", grasp_reference=GRASP),
             plan(action("shoulder", delta=-.1), action("observe"), action("shoulder", delta=-.1),
                  learning="OLD_COMPARISON_PRIVATE", expected="EXPECTED_CHANGE_PRIVATE"), plan()]
    brain, requests = make_brain(plans, scenes=[scene(), scene(), scene(), base],
                                gripper_scenes=[scene(), scene(), probe])
    history = []
    for value in (0, 20, 40, 60):
        tick(brain, history, value)
    pair = requests.gripper_observations[-1]
    assert pair["messages"][0]["content"] == CALIBRATION_COMPARE_PROMPT
    images = [p for p in pair["messages"][1]["content"] if p["type"] == "image_url"]
    # The previous batch began at 40; the saved grasp reference is the older 20.
    assert images == [encode_image_block(obs(v)["images"]["front"], fmt="openai") for v in (40, 60)]
    payload = json.dumps(pair)
    for private in ("OLD_COMPARISON_PRIVATE", "EXPECTED_CHANGE_PRIVATE", "PRIVATE_HOLDING", "PRIVATE_RESULT", "TASK"):
        assert private not in payload
    assert "delta" not in json.dumps(pair["messages"][1])
    assert len(requests) == len(requests.observations) == 4
    assert len(requests.gripper_observations) == 3  # Replaces the probe, adds no request.
    projected = planner_scene(requests[-1])
    assert projected["gripper"] == probe["gripper"]
    assert projected["ground_balls"] == [{"description": "上方地面的球"}]
    for field in ("holding", "release_view"):
        assert projected[field] == base[field]
    assert brain._scene_cache[1] == base and brain.state["visual_memory"]["held"] is None
    for view, value in zip(requests.observations, (0, 20, 40, 60)):
        assert [p for p in view["messages"][1]["content"] if p["type"] == "image_url"] == [
            encode_image_block(obs(value)["images"]["front"], fmt="openai")]


def test_initial_comparison_cache_distinguishes_before_image_and_single_view(make_brain):
    brain, requests = make_brain([plan(action("open_gripper"))] +
                                [plan(action("shoulder", delta=.1))] * 3 + [plan()])
    history = []
    for value in (0, 20, 40, 40, 40):
        tick(brain, history, value)
    assert len(requests.observations) == 3  # Base still caches only current pixels.
    probes = requests.gripper_observations
    assert len(probes) == 3  # Single 20, pair 20->40, pair 40->40; final pair is reused.
    for probe, values in zip(probes, ((20,), (20, 40), (40, 40))):
        assert [p for p in probe["messages"][1]["content"] if p["type"] == "image_url"] == [
            encode_image_block(obs(v)["images"]["front"], fmt="openai") for v in values]
    assert brain.state["calibration"]["moves"] == {"to_grasp": None, "to_clearance": None}


@pytest.mark.parametrize("actions", [
    [action("observe")], [action("arm_pose", pose="reach")], [action("elbow", delta=.1)],
    [action("shoulder", delta=.1), action("forward", seconds=.2)],
    [action("shoulder", delta=.1), action("elbow", delta=.1)],
    [action("shoulder", delta=.1), action("shoulder", delta=-.1)],
])
def test_initial_comparison_excludes_non_shoulder_or_opposing_batches(actions, make_brain):
    brain, requests = make_brain([plan(action("open_gripper")), plan(*actions), plan()])
    history = []
    for value in (0, 20, 40):
        tick(brain, history, value)
    probe = requests.gripper_observations[-1]
    assert probe["messages"][0]["content"] == CALIBRATION_VIEW_PROMPT
    assert [p for p in probe["messages"][1]["content"] if p["type"] == "image_url"] == [
        encode_image_block(obs(40)["images"]["front"], fmt="openai")]


def test_same_pixels_entering_pick_use_unmodified_base_without_new_observation(make_brain):
    brain, requests = make_brain(workflow()[:5], gripper_scenes=[scene("unclear")] * 3)
    history = []
    for value in (0, 20, 40, 60, 60):
        tick(brain, history, value)
    assert len(requests.observations) == 4
    assert len(requests.gripper_observations) == 3
    assert planner_scene(requests[-1]) == described_scene()
    rows = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["scene_reused"] and rows[-1]["calibration_gripper"] is None


def test_close_then_open_in_explore_does_not_reenable_initial_geometry_probe(make_brain):
    brain, requests = make_brain([plan(action("open_gripper")), plan(action("close_gripper")),
                                 plan(action("open_gripper")), plan()], gripper_scenes=[scene("unclear")])
    run_ticks(brain, [], 4)
    assert len(requests.gripper_observations) == 1
    assert brain.state["phase"] == "explore"
    assert planner_scene(requests[-1]) == described_scene()


@pytest.mark.parametrize("probe", [{}, scene("held")])
def test_unaccepted_geometry_probe_leaves_valid_base_scene_usable(probe, make_brain):
    brain, requests = make_brain([plan(action("open_gripper")), plan(action("shoulder", delta=.1))],
                                gripper_scenes=[probe])
    batches = run_ticks(brain, [], 2)
    assert batches[-1][0].tool == "shoulder"
    assert planner_scene(requests[-1]) == described_scene()


def test_direct_held_evidence_skips_initial_geometry_probe(make_brain):
    brain, requests = make_brain([plan(action("open_gripper")), plan(phase="place", holding="held")],
                                scenes=[scene(), scene("held", "blocked")])
    run_ticks(brain, [], 2)
    assert not requests.gripper_observations
    assert planner_scene(requests[-1]) == described_scene("held", "blocked")


def test_optional_geometry_timeout_preserves_valid_observation_and_planning(make_brain):
    brain, requests = make_brain([plan(action("open_gripper")), plan(action("shoulder", delta=.1)),
                                 plan(action("shoulder", delta=.1)), plan()],
                                gripper_scenes=[httpx.ReadTimeout("single timeout"), httpx.ReadTimeout("pair timeout")])
    history = []
    for value in (0, 20, 20, 20):
        tick(brain, history, value)
    assert len(requests) == 4 and len(requests.gripper_observations) == 2
    assert all(planner_scene(request) == described_scene() for request in requests[1:])
    rows = [json.loads(s) for s in (brain.trace_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows[1:3]:
        assert next(r for r in row["requests"] if r["kind"] == "calibration_gripper")["error"] == "ReadTimeout"
    assert not any(r["kind"] == "calibration_gripper" for r in rows[-1]["requests"])


@pytest.mark.parametrize("invalid", [{}, {**scene(), "holding": "possibly"},
                                    {**scene(), "ground_balls": [{"center": [1.2, .3], "description": "越界"}]}])
def test_invalid_observation_does_not_plan_or_default_to_empty(invalid, make_brain):
    brain, requests = make_brain([plan(action("open_gripper"))], scenes=[invalid])
    assert tick(brain, [])[0].tool == "observe"
    assert not requests and len(requests.observations) == 1
    assert brain.state["visual_memory"]["held"] is None


def test_independent_empty_allows_preparation_after_failed_grasp(make_brain):
    first = plan(action("close_gripper"))
    prepare = plan(action("open_gripper"), holding="empty")
    brain, requests = make_brain([first, prepare], scenes=[scene(), scene("empty", "blocked")])
    batches = run_ticks(brain, [], 2)
    assert [d.tool for d in batches[-1]] == ["open_gripper"]


def test_fresh_observer_held_cannot_release_without_history_or_candidate(make_brain):
    bad = plan(action("open_gripper"), holding="empty")
    brain, requests = make_brain([bad, bad], scenes=[scene("held", "blocked")])
    assert tick(brain, [])[0].tool == "observe"
    assert len(requests.observations) == 1 and len(requests) == 2


def test_observed_held_is_remembered_when_planner_is_uncertain_across_rounds(make_brain):
    look = plan(action("shoulder", delta=.1), holding="unclear")
    release = plan(action("open_gripper"), holding="empty")
    brain, requests = make_brain([look, release, release], scenes=[scene("held"), scene("unclear")])
    batches = run_ticks(brain, [], 2)
    assert [d.tool for d in batches[-1]] == ["observe"]
    assert brain.state["visual_memory"]["held"]
