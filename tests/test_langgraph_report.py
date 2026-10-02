"""Reports correlate real module calls, image evidence and executed commands."""

import json

from PIL import Image
import pytest

from brains.langgraph.report import Images, build_report


def write_lines(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


@pytest.mark.parametrize("explicit", [True, False])
def test_round_report_keeps_reference_calls_in_current_round_and_matches_effects(tmp_path, explicit):
    run = tmp_path / "memory" / "runs" / "run1"
    (run / "frames").mkdir(parents=True)
    frames = []
    for name in ("reference", "one", "two", "three"):
        path = run / "frames" / (name + ".png")
        Image.new("RGB", (8, 6), "red").save(path)
        frames.append({"id": name, "path": str(path), "config_id": "c", "pose": "reach" if name == "reference" else "stow"})
    ref, one, two, three = frames
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "meta.json").write_text(json.dumps({"brain_run": str(run), "success": False}), encoding="utf-8")
    records = []
    if explicit:
        records.append({"event": "round_started", "round": 1, "phase": "origin", "frame": one})
    records.append({"event": "decision", "phase": "origin", "frame_id": "one", "commands": [{"tool": "arm_pose", "args": {"pose": "stow"}}]})
    if explicit:
        records.append({"event": "round_started", "round": 2, "phase": "origin", "frame": two})
    records += [
        {"event": "transition", "before": one, "after": two, "commands": []},
        {"event": "capability_input", "call_id": "reference", "capability": "analyze_reach", "input": {"frame": ref}},
        {"event": "capability_result", "call_id": "reference", "capability": "analyze_reach", "result": {"valid": False}},
        {"event": "capability_input", "call_id": "find", "capability": "find_ball", "input": {"frame": two}, "prompt": "<script>bad</script>"},
        {"event": "capability_result", "call_id": "find", "capability": "find_ball", "result": {"status": "visible"}},
        {"event": "capability_input", "call_id": "match", "capability": "match_grasp", "input": {
            "frame": two, "target": {"center": {"x": .5, "y": .1}}, "evidence": [{"kind": "reach", "analysis": {
                "valid": True, "region": [{"x": .4, "y": .4}, {"x": .6, "y": .4}, {"x": .5, "y": .6}]}}]}},
        {"event": "capability_result", "call_id": "match", "capability": "match_grasp", "result": {"status": "yes"}},
        {"event": "decision", "phase": "pending", "frame_id": "two", "commands": [{"tool": "close_gripper", "args": {}}]},
    ]
    if explicit:
        records.append({"event": "round_started", "round": 3, "phase": "pending", "frame": three})
    records += [
        {"event": "transition", "before": two, "after": three, "commands": []},
        {"event": "capability_input", "call_id": "held", "capability": "check_held", "input": {"frame": three, "comparison": two}},
        {"event": "api_call", "capability": "check_held", "ok": False, "error": "ConnectError", "seconds": 5},
        {"event": "capability_error", "call_id": "held", "error": "CapabilityError"},
        {"event": "task_error", "error": "CapabilityError", "message": "ConnectError"},
    ]
    write_lines(run / "events.jsonl", records)
    (episode / "imgs").mkdir()
    Image.new("RGB", (8, 6), "blue").save(episode / "imgs" / "post.jpg")
    actions = [{"decision_round": 1, "tool": "arm_pose", "args": {"pose": "stow"}},
               {"decision_round": 2, "tool": "close_gripper", "args": {}, "ok": True,
                "result": "No ball attached", "img_after": {"front_cam": "imgs/post.jpg"}}]
    write_lines(episode / "decisions.jsonl", actions)
    page = build_report(episode)
    source = page.with_name("data.js").read_text(encoding="utf-8")
    data = json.loads(source.removeprefix("window.REPORT = ").strip().removesuffix(";"))
    assert "<script>" not in source
    assert len(data["rounds"]) == 3
    first, second, last = data["rounds"]
    assert first["frame"]["id"] == "one" and first["calls"] == []
    assert second["frame"]["id"] == "two"
    assert second["calls"][0]["input"]["frame"]["id"] == "reference"
    assert second["phase_after"] == "pending" and second["warnings"]
    assert second["actions"][0]["tool"] == "close_gripper"
    assert (page.parent / second["actions"][0]["img_after_urls"]["front_cam"]).is_file()
    assert last["calls"][0]["error"] == "CapabilityError" and last["actions"] == []
    assert last["error"]["message"] == "ConnectError"
    assert data["stats"]["api_failures"] == 1 and data["warnings"] == []
    assert data["default_round"] == 2


def test_report_does_not_copy_images_outside_record_roots(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "unrelated.png"
    Image.new("RGB", (3, 3)).save(outside)
    images = Images(tmp_path / "output", [allowed])
    assert images.copy(allowed / ".." / "unrelated.png") is None
    assert images.warnings and not (tmp_path / "output").exists()


def test_live_snapshot_skips_only_unfinished_tail_and_marks_run_in_progress(tmp_path):
    run = tmp_path / "memory" / "runs" / "active"
    run.mkdir(parents=True)
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "meta.json").write_text(json.dumps({"brain_run": str(run)}), encoding="utf-8")
    write_lines(run / "events.jsonl", [{"event": "round_started", "round": 1, "phase": "empty",
                                      "frame": {"id": "live"}}])
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('\n{"event": "capability_in')
    (episode / "decisions.jsonl").write_text('{"decision_round":', encoding="utf-8")
    page = build_report(episode)
    data = json.loads(page.with_name("data.js").read_text(encoding="utf-8")
                      .removeprefix("window.REPORT = ").strip().removesuffix(";"))
    assert data["in_progress"] is True and data["stats"]["rounds"] == 1
    assert data["rounds"][0]["actions"] == []
    # A completed report must not silently hide corruption.
    (episode / "meta.json").write_text(json.dumps({"brain_run": str(run), "success": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        build_report(episode)
