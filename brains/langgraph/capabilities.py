"""Per-task model/prompt dispatch, input contracts, caching and call records."""

import hashlib
import json
from pathlib import Path
import time

import yaml

import config
from .backends import LLMBackend, polygon_match
from .contracts import BallDetection, Detection, SpatialMatch, Motion, MotionReview, ReachAnalysis, Verdict
from .image_context import COMMON_PROMPT, PROTOCOL_VERSION

OUTPUTS = {
    "find_ball": BallDetection, "find_box": Detection, "match_grasp": SpatialMatch,
    "plan_motion": Motion, "check_held": Verdict, "match_drop": SpatialMatch,
    "check_drop": Verdict, "analyze_reach": ReachAnalysis, "analyze_motion": MotionReview,
}
PACKAGE = Path(__file__).parent


def load_profile(path=None):
    path = Path(path) if path else PACKAGE / "profiles.yaml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(profile["capabilities"]) != set(OUTPUTS):
        raise ValueError("Profile must configure exactly the nine capabilities")
    for key in ("max_decisions", "max_unknown", "max_info_requests"):
        if not isinstance(profile[key], int) or profile[key] < 1:
            raise ValueError(f"Invalid profile setting: {key}")
    profile.setdefault("max_plan_seconds", profile["max_motion_seconds"])
    profile.setdefault("max_recovery_seconds", min(0.6, profile["max_motion_seconds"]))
    if not (0.1 <= profile["probe_seconds"] <= profile["max_motion_seconds"] <= config.PRIM_MAX_S
            and profile["max_motion_seconds"] <= profile["max_plan_seconds"] <= 30
            and profile["probe_seconds"] <= profile["max_recovery_seconds"] <= profile["max_motion_seconds"]):
        raise ValueError("Invalid motion durations")
    for name, override in profile["capabilities"].items():
        spec = {**profile["defaults"], **override}
        if spec["backend"] not in ("llm", "polygon") or (spec["backend"] == "polygon" and name != "match_grasp"):
            raise ValueError(f"Unsupported backend for {name}")
        if not 1 <= spec["attempts"] <= 3 or spec["max_tokens"] < 1 or spec["timeout_s"] <= 0:
            raise ValueError(f"Invalid API limits for {name}")
        if spec.get("thinking") not in (None, "enabled", "disabled"):
            raise ValueError("thinking must be enabled/disabled")
        prompt = Path(spec["prompt"])
        override["prompt_path"] = str(prompt if prompt.is_absolute() else
                                      (path.parent / "prompts" / prompt))
        if not Path(override["prompt_path"]).is_file():
            raise ValueError(f"Missing prompt: {name}")
    return profile


class Capabilities:
    def __init__(self, profile, backend=None, overrides=None):
        self.profile = profile
        self.backend = backend or LLMBackend()
        self.overrides = overrides or {}
        self.cache = {}
        self.log = None

    def close(self):
        self.backend.close()

    def begin_frame(self, log):
        self.cache.clear()
        self.log = log

    def call(self, name, value):
        spec = {**self.profile["defaults"], **self.profile["capabilities"][name]}
        prompt = Path(spec["prompt_path"]).read_text(encoding="utf-8").strip()
        schema = OUTPUTS[name]
        signature = {"spec": spec, "prompt": prompt, "common_prompt": COMMON_PROMPT,
                     "image_protocol": PROTOCOL_VERSION, "schema": schema.model_json_schema()}
        version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        key = hashlib.sha256((name + version + value.model_dump_json()).encode()).hexdigest()
        if key in self.cache:
            self.log.append("cache_hit", capability=name, frame_id=value.frame.id, version=version)
            return self.cache[key]
        started = time.monotonic()
        self.log.append("capability_input", call_id=key, capability=name, version=version,
                        input=value.model_dump(), profile=spec, prompt=prompt,
                        common_prompt=COMMON_PROMPT, image_protocol=PROTOCOL_VERSION)
        print(f"    [langgraph] {name} ({spec['backend']})", flush=True)
        try:
            if name in self.overrides:
                result = self.overrides[name](value)
            elif spec["backend"] == "polygon":
                result = polygon_match(value)
            else:
                result = self.backend.invoke(name, spec, prompt, value, schema, self.log.append)
            result = schema.model_validate(result)
        except Exception as error:
            self.log.append("capability_error", call_id=key, capability=name, error=type(error).__name__)
            raise
        self.log.append("capability_result", call_id=key, capability=name, version=version,
                        result=result.model_dump(), seconds=round(time.monotonic()-started, 3))
        self.cache[key] = result
        return result
