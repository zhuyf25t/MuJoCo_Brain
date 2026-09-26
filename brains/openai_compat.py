"""OpenAI 兼容 API Brain (httpx 直连 /chat/completions).

适配 DeepSeek/Qwen/GLM/OpenRouter/vLLM 等一切兼容端点.
环境变量: LLM_BASE_URL (如 https://api.deepseek.com/v1), LLM_API_KEY, LLM_MODEL.
"""

from __future__ import annotations

import json
import os

import httpx

import config
from .base import Brain, Decision
from .llm_common import (SYSTEM_PROMPT, encode_image_block, history_summary,
                         llm_log, obs_text, parse_json_toolcall, term_show_image)


class OpenAICompatBrain(Brain):
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, transport=None):
        # 优先级: 显式参数 > 项目 config > 环境变量
        self.base_url = (base_url
                         or config.LLM_BASE_URL
                         or os.environ.get(config.ENV_OPENAI_BASE, "")).rstrip("/")
        self.api_key = (api_key
                        or config.LLM_API_KEY
                        or os.environ.get(config.ENV_OPENAI_KEY, ""))
        self.model = (model
                      or config.LLM_MODEL
                      or os.environ.get(config.ENV_OPENAI_MODEL, ""))
        if not self.base_url:
            raise RuntimeError("未配置 LLM 端点: 设 config.LLM_BASE_URL 或环境变量 LLM_BASE_URL")
        self._endpoint = None                    # 运行时探测: {base}/chat/completions 或 {base}/v1/chat/completions
        self._client = httpx.Client(timeout=config.LLM_TIMEOUT_S, transport=transport)

    def _request(self, messages: list, tools: list[dict],
                 retries: int = 3) -> dict:
        """POST chat/completions; 自动探测路径(±/v1), 5xx 指数退避重试."""
        import time as _time

        import httpx as _hx
        if self._endpoint is None:
            candidates = [f"{self.base_url}/chat/completions"]
            if not self.base_url.endswith("/v1"):
                candidates.append(f"{self.base_url}/v1/chat/completions")
        else:
            candidates = [self._endpoint]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": config.LLM_MAX_TOKENS}
        last_err: Exception | None = None
        for attempt in range(retries):
            for url in candidates:
                r = self._client.post(url, headers=headers, json=body)
                ct = r.headers.get("content-type", "")
                if r.status_code == 404 or ("json" not in ct and r.status_code == 200):
                    # 路径不对 (404 或返回了 HTML 首页) → 试下一个候选
                    last_err = _hx.HTTPStatusError(f"路径不可用: {url}", request=r.request, response=r)
                    continue
                if r.status_code >= 500:          # 上游过载/耗尽 → 退避后重试
                    last_err = _hx.HTTPStatusError(
                        f"{r.status_code} {url}: {r.text[:120]}", request=r.request, response=r)
                    continue
                r.raise_for_status()
                self._endpoint = url              # 记住可用路径
                return r.json()
            if attempt < retries - 1:
                _time.sleep(2.0 * (attempt + 1))
        raise last_err or RuntimeError("无可用端点")

    def decide(self, obs, task_text, tool_schemas, history) -> list[Decision]:
        tools = [{"type": "function",
                  "function": {"name": t["name"], "description": t["description"],
                               "parameters": t["input_schema"]}}
                 for t in tool_schemas]
        note = ""
        for attempt in range(2):
            user_content: list = []
            for cam, rgb in obs.get("images", {}).items():
                user_content.append({"type": "text", "text": f"相机 {cam} 当前画面:"})
                user_content.append(encode_image_block(rgb, fmt="openai"))
            user_content.append({"type": "text", "text": obs_text(obs, task_text, history) + note})
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            user_txt = " ".join(p["text"] for p in user_content if p.get("type") == "text")
            n_img = sum(1 for p in user_content if p.get("type") == "image_url")
            for cam, rgb in obs.get("images", {}).items():
                term_show_image(rgb, label=f"→ 发给模型的相机图 [{cam}]")
            llm_log("LLM→模型", f"图x{n_img} | {user_txt[:400]}")
            data = self._request(messages, tools)
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            calls = msg.get("tool_calls") or []
            if calls:
                decisions = []
                for tc in calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    decisions.append(Decision(thought="", tool=fn.get("name", ""), args=args))
                thought = (msg.get("content") or "")[:400]
                decisions[0].thought = thought
                llm_log("模型→工具", f"{[d.tool for d in decisions]} | 思考: {thought[:200]}")
                return decisions
            # 兜底: 文本里的 JSON
            parsed = parse_json_toolcall(msg.get("content") or "")
            if parsed:
                return Decision(thought=str(parsed.get("thought", ""))[:400],
                                tool=str(parsed.get("tool", "")),
                                args=parsed.get("args", {}) or {})
            note = f"(上次输出无法解析为工具调用: {(msg.get('content') or '')[:120]!r}, 请用 tool_calls 或 JSON 重试)"
        return Decision("连续两次无法解析, 保守原位重观察", "forward", {"seconds": 0.0})
