"""OpenAI-compatible Chat Completions, preserving tool requests across rounds.

Exactly one POST per invoke: no path probing, streaming, or automatic retries.
The configured endpoint must support both image inputs and function tools.
"""

from __future__ import annotations

import base64
import io
import json
import os
from copy import deepcopy
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import config
from .settings import local_values, positive_integer, setting
from .state import calls_of


def image_message(views: list[tuple[dict, object]], caption: str) -> HumanMessage:
    """Label actual PNG image blocks; provenance is retained only in local metadata."""
    content = [{"type": "text", "text": caption}]
    references = []
    for metadata, image in views:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.extend([
            {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ])
        references.append(deepcopy(metadata))
    return HumanMessage(content=content, additional_kwargs={"vista_images": references})


def api_messages(messages, system: str) -> list[dict]:
    result = [{"role": "system", "content": system}]
    for message in messages:
        if isinstance(message, AIMessage):
            saved = message.additional_kwargs.get("provider_message")
            if saved is not None:
                result.append(deepcopy(saved))
            else:
                calls = []
                for call in calls_of(message):
                    arguments = call["args"]
                    calls.append({"id": call["id"], "type": "function", "function": {
                        "name": call["name"], "arguments": (
                            json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                            if isinstance(arguments, dict) else arguments)}})
                result.append({"role": "assistant", "content": message.content or None,
                               "tool_calls": calls})
        elif isinstance(message, ToolMessage):
            result.append({"role": "tool", "tool_call_id": message.tool_call_id,
                           "content": message.content})
        elif isinstance(message, HumanMessage):
            result.append({"role": "user", "content": deepcopy(message.content)})
        elif isinstance(message, SystemMessage):
            result.append({"role": "system", "content": message.content})
        else:
            raise RuntimeError("VISTA 存在无法发送到模型的消息类型")
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def parse_reply(data: dict) -> AIMessage:
    try:
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected one choice")
        choice = choices[0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise RuntimeError("VISTA 模型回复被截断或拦截；未执行其中的工具调用。")
        raw = choice["message"]
        if raw.get("role", "assistant") != "assistant":
            raise ValueError("invalid role")
        content = raw.get("content") or ""
        if not isinstance(content, (str, list)):
            raise ValueError("invalid content")
        raw_calls = raw.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError("invalid calls")
        calls, invalid = [], []
        for call in raw_calls:
            if call.get("type") != "function":
                raise ValueError("unsupported tool type")
            function = call["function"]
            name, identifier = function["name"], call.get("id")
            if (not isinstance(name, str) or not isinstance(identifier, str)
                    or not identifier.strip()):
                raise ValueError("invalid call identity")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                raise ValueError("tool arguments must be JSON text in the API response")
            try:
                parsed = json.loads(arguments, parse_constant=_reject_constant)
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be an object")
            except (ValueError, TypeError):
                invalid.append({"id": identifier, "name": name,
                                "args": arguments if isinstance(arguments, str) else "null",
                                "error": "参数必须是合法的 JSON 对象"})
            else:
                calls.append({"id": identifier, "name": name, "args": parsed})
        # Preserve provider-required reasoning fields and the exact original calls.
        # Do not echo response-level metadata or any request/authentication headers.
        wire = {key: deepcopy(raw[key]) for key in (
            "content", "tool_calls", "refusal", "reasoning_content", "reasoning", "reasoning_details"
        ) if key in raw}
        wire["role"] = "assistant"
        return AIMessage(content=content, tool_calls=calls, invalid_tool_calls=invalid,
                         additional_kwargs={"provider_message": wire})
    except RuntimeError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError):
        raise RuntimeError("VISTA 模型响应格式错误或工具调用缺少有效 ID，程序终止。") from None


class ChatModel:
    def __init__(self, *, base_url=None, api_key=None, model=None, transport=None,
                 request_options=None, env_file=None, environ=None):
        values = local_values(env_file)
        env = os.environ if environ is None else environ
        self.model = model if model is not None else setting(("LLM_MODEL", "OPENAI_MODEL"), values, env)
        base = base_url if base_url is not None else setting(("LLM_BASE_URL", "OPENAI_BASE_URL"), values, env)
        key = api_key if api_key is not None else setting(("LLM_API_KEY", "OPENAI_API_KEY"), values, env, "")
        if not isinstance(base, str) or not base.strip():
            raise RuntimeError("VISTA 未配置 LLM_BASE_URL / OPENAI_BASE_URL")
        if not isinstance(self.model, str) or not self.model.strip():
            raise RuntimeError("VISTA 未配置 LLM_MODEL / OPENAI_MODEL")
        base = base.strip().rstrip("/")
        parsed = urlsplit(base)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise RuntimeError("VISTA 模型地址必须是 http(s) API 根地址或 chat/completions 地址")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        if request_options is None:
            max_tokens = positive_integer(setting(("OPENAI_MAX_TOKENS", "max_tokens"), values, env,
                                                   config.LLM_MAX_TOKENS), "max_tokens")
            stream = setting(("OPENAI_STREAM", "stream"), values, env, "false")
            if str(stream).strip().lower() != "false":
                raise RuntimeError("VISTA 第一版只接收完整 JSON 回复，请配置 stream=false")
            request_options = {"max_tokens": max_tokens}
            thinking = setting(("OPENAI_THINKING", "thinking"), values, env)
            if thinking:
                if thinking not in ("enabled", "disabled"):
                    raise RuntimeError("thinking 必须是 enabled 或 disabled")
                request_options["thinking"] = {"type": thinking}
            effort = setting(("OPENAI_REASONING_EFFORT", "reasoning_effort"), values, env)
            if effort:
                request_options["reasoning_effort"] = effort
        if set(request_options) - {"max_tokens", "max_completion_tokens", "thinking", "reasoning_effort"}:
            raise RuntimeError("VISTA request_options 含不支持的参数")
        self.options = deepcopy(request_options)
        self._client = httpx.Client(timeout=config.LLM_TIMEOUT_S, transport=transport,
                                    follow_redirects=False)

    def invoke(self, messages, *, tools: list[dict], system: str) -> AIMessage:
        body = {"model": self.model, "messages": api_messages(messages, system),
                "tools": tools, "tool_choice": "auto", "stream": False, **self.options}
        try:
            response = self._client.post(self.endpoint, headers=self._headers, json=body)
        except httpx.HTTPError as error:
            raise RuntimeError(f"VISTA 模型请求失败（{type(error).__name__}），未自动重试。") from None
        if not response.is_success:
            advice = {401: "检查 API 凭据", 403: "检查访问权限", 404: "检查完整 API 地址（包括所需 /v1）",
                      413: "上下文或图片过大", 429: "请求限流或配额不足"}.get(
                          response.status_code, "检查端点的参数、图像/工具支持及上下文限制")
            raise RuntimeError(f"VISTA 模型请求失败，HTTP {response.status_code}；{advice}。未自动重试。")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("VISTA 模型端点未返回合法 JSON，未自动尝试其他地址。") from None
        return parse_reply(data)

    def close(self):
        self._client.close()
