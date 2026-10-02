"""Small structured LLM adapter and a replaceable polygon matching backend."""

import json
import os
import time

import httpx
from pydantic import ValidationError

import config
from .contracts import InfoRequest, Match
from .image_context import COMMON_PROMPT, image_content


class CapabilityError(RuntimeError):
    pass


def validation_details(error, raw):
    details = error.errors(include_url=False, include_context=False, include_input=False)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    for item in details:
        value = parsed
        try:
            for key in item["loc"]:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(value, str):
            item["actual_length"] = len(value)
    return details


class LLMBackend:
    def __init__(self, *, transport=None):
        self.base_url = (config.LLM_BASE_URL or os.getenv("LLM_BASE_URL")
                         or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.key = config.LLM_API_KEY or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.default_model = config.LLM_MODEL or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "")
        self.client = httpx.Client(transport=transport)
        self.endpoint = None

    def close(self):
        self.client.close()

    def invoke(self, name, spec, prompt, value, output_type, log):
        model = spec.get("model") or self.default_model
        if not self.base_url or not model:
            raise CapabilityError("请在项目 .env 配置 LLM_BASE_URL 和 LLM_MODEL")
        content, image_count = image_content(value)
        messages = [{"role": "system", "content": prompt + "\n" + COMMON_PROMPT},
                    {"role": "user", "content": content}]
        body = {"model": model, "messages": messages, "stream": False,
                "max_tokens": spec["max_tokens"],
                "tools": [{"type": "function", "function": {
                    "name": "submit_result", "description": name,
                    "parameters": output_type.model_json_schema()}}],
                "tool_choice": {"type": "function", "function": {"name": "submit_result"}}}
        if spec.get("thinking"):
            body["thinking"] = {"type": spec["thinking"]}
        if spec.get("reasoning_effort"):
            body["reasoning_effort"] = spec["reasoning_effort"]
        if model.startswith("deepseek") and spec.get("thinking") == "enabled":
            # DeepSeek thinking rejects named/required tool_choice with HTTP 400.
            # Keep the same single result tool and validate its output as before.
            body["tool_choice"] = "auto"
            messages[0]["content"] += "\n请调用 submit_result 提交结构化结果。"
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        for attempt in range(spec["attempts"]):
            started = time.monotonic()
            usage = {}
            raw, result_message, finish_reason = None, None, None
            try:
                urls = [self.endpoint] if self.endpoint else [f"{self.base_url}/chat/completions"]
                if not self.endpoint and not self.base_url.endswith("/v1"):
                    urls.append(f"{self.base_url}/v1/chat/completions")
                for url in urls:
                    response = self.client.post(url, headers=headers, json=body, timeout=spec["timeout_s"])
                    if response.status_code != 404:
                        break
                response.raise_for_status()
                self.endpoint = url
                data = response.json()
                usage = data.get("usage", {})
                message = data["choices"][0]["message"]
                finish_reason = data["choices"][0].get("finish_reason")
                result_message = {k: message[k] for k in ("content", "tool_calls") if k in message}
                calls = message.get("tool_calls") or []
                if calls:
                    if len(calls) != 1 or calls[0]["function"]["name"] != "submit_result":
                        raise ValueError("Expected one submit_result")
                    raw = calls[0]["function"]["arguments"]
                else:
                    raw = message.get("content") or ""
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                result = output_type.model_validate_json(raw)
                log("api_call", capability=name, attempt=attempt + 1, model=model,
                    seconds=round(time.monotonic()-started, 3), usage=usage,
                    image_count=image_count, ok=True, finish_reason=finish_reason)
                return result
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                details = validation_details(error, raw) if isinstance(error, ValidationError) else []
                detail = "; ".join(
                    f"{'.'.join(map(str, e['loc'])) or 'result'}: {e['msg']}"
                    + (f" (actual={e['actual_length']})" if "actual_length" in e else "") for e in details)
                if not detail:
                    detail = f"{type(error).__name__}, status={status}"
                if result_message is not None:
                    # Keep the model result intact, never HTTP headers/auth/provider error bodies.
                    log("invalid_result", capability=name, attempt=attempt + 1, model=model,
                        raw_result=raw, result_message=result_message, validation_errors=details,
                        error=type(error).__name__, finish_reason=finish_reason)
                log("api_call", capability=name, attempt=attempt + 1, model=model,
                    seconds=round(time.monotonic()-started, 3), usage=usage,
                    image_count=image_count, ok=False, error=type(error).__name__, status=status,
                    detail=detail, finish_reason=finish_reason)
                if status in (400, 401, 403) or attempt + 1 == spec["attempts"]:
                    raise CapabilityError(f"{name}: {detail}") from None
                messages[0]["content"] += "\n上次结果校验失败，请重新提交完整结果。错误：" + detail
        raise AssertionError("unreachable")


def polygon_match(value):
    """No vision magic: requires a detected center and a compatible learned region."""
    from .contracts import SpatialMatch

    def result(**kwargs):
        return SpatialMatch(**kwargs, alignment="unknown", distance="unknown")

    refs = [s for s in value.evidence if s["kind"] == "reach"]
    if not refs:
        return result(status="need_info", request=InfoRequest(kind="reach"))
    sample = refs[-1]
    reference = sample["frames"][0]
    if (value.target is None or value.target.path != "clear" or value.frame.pose != "stow"
            or reference["config_id"] != value.frame.config_id or reference["pose"] != "reach"):
        return result(status="unknown", note="需要清楚的目标及兼容的相机/姿态配置")
    points = sample["analysis"]["region"]
    if not sample["analysis"]["valid"] or len(points) < 3:
        return result(status="unknown", note="没有有效抓取区域")
    x, y = value.target.center.x, value.target.center.y
    inside = False
    for a, b in zip(points, points[1:] + points[:1]):
        ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        if abs(cross) < 1e-10 and min(ax,bx) <= x <= max(ax,bx) and min(ay,by) <= y <= max(ay,by):
            return result(status="yes", note="球心位于参考区域边界")
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay) + ax:
            inside = not inside
    return result(status="yes" if inside else "no", note="球心与参考多边形匹配")
