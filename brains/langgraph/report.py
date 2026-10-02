"""Build an offline, round-by-round inspection page from existing run records.

python -m brains.langgraph.report EPISODE --events EVENTS_JSONL --serve
No model requests and no robot commands are made by this module.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from functools import partial

ASSETS = Path(__file__).with_name("report_assets")


def read_records(path, *, allow_partial=False):
    records = []
    source = Path(path).read_text(encoding="utf-8")
    lines = source.splitlines()
    if allow_partial and source and not source.endswith("\n"):
        lines = lines[:-1]
    for number, line in enumerate(lines, 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
    return records


class Images:
    def __init__(self, output, roots):
        self.output = output
        self.roots = [Path(root).resolve() for root in roots]
        self.saved = {}
        self.warnings = []

    def copy(self, source):
        path = Path(source).resolve()
        if path in self.saved:
            return self.saved[path]
        if (not any(path.is_relative_to(root) for root in self.roots)
                or path.suffix.lower() not in (".png", ".jpg", ".jpeg") or not path.is_file()):
            self.warnings.append(f"图片不可用或不在本次记录目录内：{path.name}")
            self.saved[path] = None
            return None
        name = hashlib.sha256(str(path).encode()).hexdigest()[:24] + path.suffix.lower()
        destination = self.output / "images" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        self.saved[path] = "images/" + name
        return self.saved[path]

    def frames(self, value):
        if isinstance(value, dict):
            result = {key: self.frames(item) for key, item in value.items()}
            if {"id", "path", "config_id", "pose"} <= value.keys():
                result["image_url"] = self.copy(value["path"])
            return result
        if isinstance(value, list):
            return [self.frames(item) for item in value]
        return value


def new_round(number, phase):
    return {"number": number, "phase_before": phase, "phase_after": phase, "frame": None,
            "calls": [], "info": [], "actions": [], "events": [], "decision": None,
            "error": None, "warnings": []}


def check_consistency(round_data):
    """Flag contradictions in reported numbers, never certify physical graspability."""
    for call in round_data["calls"]:
        value, result = call.get("input", {}), call.get("result", {})
        if call["capability"] != "match_grasp" or result.get("status") != "yes":
            continue
        target = value.get("target") or {}
        point = target.get("center")
        refs = [s for s in value.get("evidence", []) if s.get("kind") == "reach"]
        if not point or not refs:
            continue
        analysis = refs[-1].get("analysis", {})
        region = analysis.get("region", [])
        if not analysis.get("valid") or len(region) < 3:
            continue
        xmin, xmax = min(p["x"] for p in region), max(p["x"] for p in region)
        ymin, ymax = min(p["y"] for p in region), max(p["y"] for p in region)
        if not (xmin <= point["x"] <= xmax and ymin <= point["y"] <= ymax):
            message = (f"判定可抓，但模型球心 ({point['x']:g}, {point['y']:g}) 在其参考范围 "
                       f"x={xmin:g}…{xmax:g}、y={ymin:g}…{ymax:g} 之外。"
                       "这是记录内的数值矛盾，仍需对照原图核实。")
            call["warning"] = message
            round_data["warnings"].append(message)


def build_report(episode, *, events=None, output=None):
    episode = Path(episode).resolve()
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    if events is None:
        if not meta.get("brain_run"):
            raise ValueError("旧记录没有 brain_run，请显式指定 --events")
        events = Path(meta["brain_run"]) / "events.jsonl"
    events = Path(events).resolve()
    output = Path(output).resolve() if output else episode / "review"
    output.mkdir(parents=True, exist_ok=True)
    running = "success" not in meta and not meta.get("t_wall_end")
    # Read actions first: their round_started events have already been written.
    actions = read_records(episode / "decisions.jsonl", allow_partial=running)
    records = read_records(events, allow_partial=running)
    # A run may use reference photos from sibling runs in the same experience store.
    images = Images(output, [episode, events.parent.parent.parent])
    records = images.frames(records)
    explicit_rounds = any(e["event"] == "round_started" for e in records)
    rounds, current, active_call = [], None, None
    calls = {}
    phase = "origin"
    outcome = None
    for event in records:
        kind = event["event"]
        if kind == "round_started":
            current = new_round(event["round"], event["phase"])
            current["frame"] = event["frame"]
            rounds.append(current)
            active_call = None
        elif current is None:
            current = new_round(1, phase)
            rounds.append(current)
        elif kind == "transition" and not explicit_rounds:
            current = new_round(len(rounds) + 1, phase)
            rounds.append(current)
            active_call = None
        current["events"].append(event)
        if kind == "transition":
            current["frame"] = event["after"]
            # The transition's BEFORE is authoritative for the previous round.
            if len(rounds) > 1 and rounds[-2]["frame"] is None:
                rounds[-2]["frame"] = event["before"]
        elif kind == "capability_input":
            active_call = {"call_id": event["call_id"], "capability": event["capability"],
                           "input": event["input"], "prompt": event.get("prompt", ""),
                           "common_prompt": event.get("common_prompt", ""),
                           "profile": event.get("profile", {}), "version": event.get("version"),
                           "attempts": [], "invalid_results": []}
            calls[event["call_id"]] = active_call
            current["calls"].append(active_call)
        elif kind == "capability_result":
            call = calls.get(event["call_id"])
            if call is not None:
                call["result"] = event["result"]
                call["seconds"] = event.get("seconds")
        elif kind == "capability_error":
            call = calls.get(event["call_id"])
            if call is not None:
                call["error"] = event["error"]
        elif kind in ("api_call", "invalid_result") and active_call is not None:
            active_call["attempts" if kind == "api_call" else "invalid_results"].append(event)
        elif kind in ("request_info", "probe", "repeated_request", "cache_hit", "motion_sample", "reference_reanalyzed", "information_unavailable"):
            current["info"].append(event)
        elif kind == "decision":
            current["decision"] = event
            current["phase_after"] = phase = event["phase"]
        elif kind == "task_error":
            current["error"] = event
        elif kind == "outcome":
            outcome = event
    for action in actions:
        number = action.get("decision_round")
        match = next((r for r in rounds if r["number"] == number), None)
        if match is None:
            raise ValueError(f"Action cannot be correlated to a round: {number}")
        item = dict(action)
        for field in ("img_before", "img_after"):
            item[field + "_urls"] = {cam: images.copy(episode / path)
                                      for cam, path in (action.get(field) or {}).items()}
        match["actions"].append(item)
    for item in rounds:
        # Round 1 has no transition in old logs; find its recorded image by id.
        if item["frame"] is None and item["decision"]:
            frame_id = item["decision"]["frame_id"]
            source = events.parent / "frames" / (frame_id + ".png")
            item["frame"] = {"id": frame_id, "pose": "unknown", "image_url": images.copy(source)}
        check_consistency(item)
    api = [e for e in records if e["event"] == "api_call"]
    data = {"meta": meta, "outcome": outcome, "rounds": rounds,
            "in_progress": running and outcome is None,
            "sources": {"episode": str(episode), "events": str(events)},
            "warnings": images.warnings,
            "stats": {"rounds": len(rounds), "actions": len(actions), "api_calls": len(api),
                      "api_seconds": round(sum(e.get("seconds", 0) for e in api), 2),
                      "tokens": sum((e.get("usage") or {}).get("total_tokens", 0) for e in api),
                      "api_failures": sum(not e.get("ok", False) for e in api)},
            "default_round": next((r["number"] for r in rounds if r["warnings"]), 1)}
    # Data is only rendered with textContent; escape '<' as an extra script safeguard.
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    (output / "data.js").write_text("window.REPORT = " + encoded + ";\n", encoding="utf-8")
    for name in ("index.html", "review.css", "review.js"):
        shutil.copyfile(ASSETS / name, output / name)
    return output / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args(argv)
    page = build_report(args.episode, events=args.events, output=args.out)
    print(f"Review: {page}", flush=True)
    if args.serve:
        handler = partial(SimpleHTTPRequestHandler, directory=str(page.parent))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Open http://127.0.0.1:{args.port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
