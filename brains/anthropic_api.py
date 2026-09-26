"""Anthropic Claude Brain (官方 SDK).

环境变量: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (默认 claude-opus-5),
          ANTHROPIC_BASE_URL (可选, SDK 原生支持).
注意: 新模型不支持 temperature/top_p (发即 400), 这里不发送.
"""

from __future__ import annotations

import os
import time

import anthropic

import config
from .base import Brain, Decision
from .llm_common import (SYSTEM_PROMPT, encode_image_block, history_summary,
                         llm_log, obs_text, parse_json_toolcall, term_show_image)


class AnthropicBrain(Brain):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None):
        import httpx2
        # 优先级: 显式参数 > 项目 config > 环境变量.
        # (机器上常有给其他工具配的 ANTHROPIC_BASE_URL/AUTH_TOKEN, 不能劫持本项目)
        kwargs = {}
        key = (api_key or config.ANTHROPIC_API_KEY
               or os.environ.get(config.ENV_ANTHROPIC_KEY, ""))
        url = (base_url or config.ANTHROPIC_BASE_URL
               or os.environ.get("ANTHROPIC_BASE_URL", ""))
        if key:
            kwargs["api_key"] = key
        if url:
            kwargs["base_url"] = url
        # 部分中转网关对 SDK 默认客户端(truststore TLS)返回 401, 显式用裸 httpx2 客户端稳定
        kwargs["http_client"] = httpx2.Client(timeout=config.LLM_TIMEOUT_S)
        self.client = anthropic.Anthropic(**kwargs)
        self.model = (model or config.ANTHROPIC_MODEL_DEFAULT
                      or os.environ.get(config.ENV_ANTHROPIC_MODEL, "") or "claude-opus-5")

    def _create(self, **kw):
        """messages.create 带瞬时错误重试(超时/断连/5xx), 避免单次网络抖动炸掉整集."""
        last: Exception | None = None
        for i in range(max(1, config.LLM_RETRIES)):
            try:
                return self.client.messages.create(**kw)
            except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                last = e
                if i < config.LLM_RETRIES - 1:
                    time.sleep(2.0 * (i + 1))
        raise last  # type: ignore[misc]

    def decide(self, obs, task_text, tool_schemas, history) -> list[Decision]:
        tools = [{"name": t["name"], "description": t["description"],
                  "input_schema": t["input_schema"]}
                 for t in tool_schemas]
        note = ""
        for attempt in range(2):
            content: list = []
            for cam, rgb in obs.get("images", {}).items():
                content.append({"type": "text", "text": f"相机 {cam} 当前画面:"})
                content.append(encode_image_block(rgb, fmt="anthropic"))
            content.append({"type": "text", "text": (
                f"任务: {task_text}\n\n当前状态: 底盘位姿="
                f"{[round(v,3) for v in obs.get('base_pose',[])]}, "
                f"夹爪开度={obs['finger']:.0%}, TCP="
                f"{[round(v,3) for v in obs.get('tcp_pos',[])]}\n\n"
                f"历史动作与结果:\n{history_summary(history)}\n{note}\n"
                "请决定下一个工具调用。")})
            user_txt = " ".join(p["text"] for p in content if p.get("type") == "text")
            n_img = sum(1 for p in content if p.get("type") == "image")
            for cam, rgb in obs.get("images", {}).items():
                term_show_image(rgb, label=f"→ 发给模型的相机图 [{cam}]")
            llm_log("LLM→模型", f"图x{n_img} | {user_txt[:400]}")
            try:
                # adaptive thinking + 摘要显示: 思考过程进入 decisions.jsonl 的 thought 字段
                resp = self._create(
                    model=self.model,
                    max_tokens=config.LLM_MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    tool_choice={"type": "auto"},
                    thinking={"type": "adaptive", "display": "summarized"},
                    messages=[{"role": "user", "content": content}],
                )
            except anthropic.BadRequestError as e:
                # 个别中转端点不支持较新参数, 降级重试一次(去掉 tool_choice)
                if attempt == 0 and "tool_choice" in str(e):
                    resp = self._create(
                        model=self.model, max_tokens=config.LLM_MAX_TOKENS, system=SYSTEM_PROMPT,
                        tools=tools, messages=[{"role": "user", "content": content}])
                else:
                    raise
            thought_parts = []
            calls = []
            for block in resp.content:
                if block.type == "tool_use":
                    args = block.input if isinstance(block.input, dict) else {}
                    calls.append(Decision(thought="", tool=block.name, args=dict(args)))
                elif block.type == "thinking":
                    thought_parts.append(getattr(block, "thinking", "") or "")
                elif block.type == "text":
                    thought_parts.append(block.text)
            if calls:
                thought = " ".join(thought_parts)[:400]
                llm_log("模型→工具", f"{[c.tool for c in calls]} | 思考: {thought[:200]}")
                calls[0].thought = thought
                return calls
            llm_log("模型→文本", f"{' '.join(thought_parts)[:300]}")
            parsed = parse_json_toolcall(" ".join(thought_parts))
            if parsed:
                return Decision(thought=str(parsed.get("thought", ""))[:400],
                                tool=str(parsed.get("tool", "")),
                                args=parsed.get("args", {}) or {})
            note = f"(上次输出无法解析为工具调用: {' '.join(thought_parts)[:120]!r}, 请调用一个工具)"
        return Decision("连续两次无法解析, 保守原位重观察", "forward", {"seconds": 0.0})
