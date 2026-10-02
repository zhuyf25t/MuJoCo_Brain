"""Bounded operation experience and separate, replayable run records."""

import json
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .contracts import Frame, InfoRequest, MotionAnalysis, ReachAnalysis
from .image_context import PROTOCOL_VERSION


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class ExperienceStore:
    def __init__(self, root, config_id):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_id = config_id
        self.path = self.root / "experience.json"
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.samples = data.get("samples", []) if data.get("config_id") == config_id else []
        if self.samples and data.get("image_protocol") != PROTOCOL_VERSION:
            # Old before/after analyses may be reversed. Archive all original data;
            # retain reach pixels but withdraw its unverified numerical polygon.
            archive = self.root / f"experience-before-camera-v{PROTOCOL_VERSION}-{uuid4().hex}.json"
            write_json(archive, data)
            self.samples = [{**s, "needs_analysis": True, "analysis": {"valid": False, "region": [],
                            "note": "旧协议分析已失效；仅保留 reach 原图供重新判断。"}}
                            for s in self.samples if s["kind"] == "reach"]
            self.save()
        # Reject malformed or stale records rather than grant permission from them.
        for sample in self.samples:
            InfoRequest.model_validate({k: sample[k] for k in ("kind", "direction")})
            analysis = ReachAnalysis if sample["kind"] == "reach" else MotionAnalysis
            parsed_analysis = analysis.model_validate(sample["analysis"])
            if sample["kind"] != "reach" and not parsed_analysis.valid:
                raise ValueError("Invalid experience record")
            for frame in sample["frames"]:
                parsed = Frame.model_validate(frame)
                if parsed.config_id != config_id:
                    raise ValueError("Experience frame configuration mismatch")
                self.checked_path(parsed.path)
        for kind in ("reach", "translation", "rotation"):
            if sum(s["kind"] == kind for s in self.samples) > 2:
                raise ValueError("Experience capacity exceeded")

    def save(self):
        write_json(self.path, {"config_id": self.config_id, "image_protocol": PROTOCOL_VERSION,
                               "samples": self.samples})

    def checked_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Experience image missing or outside storage root")
        return path

    def directory(self):
        return {kind: [{"id": s["id"], "direction": s["direction"]}
                       for s in self.samples if s["kind"] == kind]
                for kind in ("reach", "translation", "rotation")}

    def query(self, request: InfoRequest):
        return [s for s in self.samples if s["kind"] == request.kind
                and s["direction"] == request.direction]

    def refresh_reach(self, sample_id, analysis: ReachAnalysis):
        sample = next(s for s in self.samples if s["id"] == sample_id and s["kind"] == "reach")
        sample["analysis"] = analysis.model_dump()
        sample.pop("needs_analysis", None)
        self.save()

    def add(self, request, frames, commands, analysis, *, observation=None):
        # A reach photo remains useful evidence even if its estimated region is unknown.
        # Motion requires a matched landmark to count as a valid experience sample.
        if request.kind != "reach" and not analysis.valid:
            return None
        if request.kind != "reach" and (analysis.before is None or analysis.after is None):
            raise ValueError("Stored motion analysis needs resolved image measurements")
        for frame in frames:
            if frame.config_id != self.config_id:
                raise ValueError("Cannot save samples from a different configuration")
            self.checked_path(frame.path)
        sample = {"id": uuid4().hex, **request.model_dump(),
                  "frames": [f.model_dump() for f in frames], "commands": commands,
                  "analysis": analysis.model_dump()}
        if request.kind != "reach":
            seconds = sum(float(c["args"].get("seconds", 1.0)) for c in commands)
            if seconds <= 0:
                raise ValueError("Motion sample needs positive executed duration")
            sample["image_rate"] = {
                "duration_seconds": seconds,
                "normalized_dx_per_second": (analysis.after.x - analysis.before.x) / seconds,
                "normalized_dy_per_second": (analysis.after.y - analysis.before.y) / seconds,
                "source": analysis.source + "; local image motion, not physical speed",
                "scope": "Only this scene/depth interval; perspective is nonlinear, do not extrapolate globally",
            }
            if observation:
                sample["observation"] = observation
                before, after = observation["before"], observation["after"]
                if before.get("radius") is not None and after.get("radius") is not None:
                    sample["image_rate"]["radius_ratio"] = after["radius"] / before["radius"]
        others = [s for s in self.samples if s["kind"] != request.kind]
        same = [s for s in self.samples if s["kind"] == request.kind]
        if request.kind != "reach":
            same = [s for s in same if s["direction"] != request.direction]
        self.samples = others + (same + [sample])[-2:]
        self.save()
        return sample


class RunLog:
    def __init__(self, root):
        self.path = Path(root) / "runs" / uuid4().hex
        (self.path / "frames").mkdir(parents=True)

    def frame(self, rgb, config_id, pose):
        frame_id = uuid4().hex
        path = (self.path / "frames" / f"{frame_id}.png").resolve()
        Image.fromarray(rgb).save(path)
        return Frame(id=frame_id, path=str(path), config_id=config_id, pose=pose,
                     width=rgb.shape[1], height=rgb.shape[0])

    def append(self, event, **data):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
