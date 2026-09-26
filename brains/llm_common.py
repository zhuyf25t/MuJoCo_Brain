"""LLM Brain 公共工具: 系统提示 / 图像编码 / 历史摘要 / JSON 兜底解析."""

from __future__ import annotations

import base64
import io
import json
import re

import numpy as np

import config

SYSTEM_PROMPT = """你是一台地面捡球机器人的大脑, 通过调用低层动作原语控制机器人.

机器人: "差速底盘 + 两自由度机械臂(肩/肘俯仰) + 滑动指夹爪 + 车头前向相机".
- 相机装在车头, 俯角看前方地面; 你没有任何精确坐标, 只有视觉检测结果(自动附带在每轮观测中)
  (球的方位角与距离估算) 和本体状态(里程计/关节角/是否持有球).
- 动作是时间盒原语: forward/back/turn_left/turn_right(秒, 速度固定),
  以及臂预设(arm_pose: stow/carry/reach/drop)、关节微调(shoulder/elbow)、
  手指开合(open_gripper/close_gripper).
- 抓取原理: 把车开到球正前方约 0.45-0.60m → 张开手指 → arm_pose("reach") 臂伸到地面 →
  close_gripper 闭合 → **先 arm_pose("carry") 抬起** → 下一轮自动收到视觉验证:
  如果手里有球, 视野内球会少一个且持有状态为"有球"; 如果球仍在地上,
  说明没夹到(结果文本会明确说 ✗), 需张开手指→收臂→微调位置→重试.
  臂在 reach 位时球可能贴着地面不易看清, 抬起后再看视野更开阔.
- **精确停距(重要)**: arm_pose("reach") 的夹爪落点恰在**车头正前方 0.55m**、离地 0.11m.
  视觉测距有 ±15% 误差(0.55m 处约±8cm), 而成功窗口只有约±5cm, 所以开环停靠
  经常差一点——差 2-12cm 失败完全正常, 不是方法错误, 按下面规则小步修正即可:
  抓取失败且臂在 reach 位时, 视觉会报告"球比夹爪远/近 + 偏差米数"(垂直像素差判别,
  近低远高). 球比夹爪**远** = 车太靠后 → 先 stow, 再 forward(0.2~0.3s≈6~9cm);
  球比夹爪**近** = 车太靠前 → 先 stow, 再 back(0.2~0.3s).
  **每次只修正一小步(≤0.3s), 修正后直接重试抓取序列; 禁止一次修正 >0.5s 或
  来回大幅移动** —— 那样只会重复错过成功窗口.
- 投放: 开到洋红色收纳箱正前方, **以视觉报的箱距为准, 箱距约 0.40~0.50m 时停**
  → arm_pose("drop") → open_gripper → 等 1 轮看球是否落进箱内.
  几何: drop 的 TCP 在车头前 0.55m、高 0.50m —— 箱距 0.45m 时球恰好越过壁沿落向箱中央;
  箱距 <0.35m 释放点会压到远壁, 球被弹出箱外; >0.6m 会落在箱前地面.
  落箱失败(球在箱外地面上)就重新捡起再投.
- 视野约 ±40°, 球不在视野时用 turn_left/turn_right 扫描寻找.
- **近距离对齐(重要)**: 臂在 reach 位时夹爪在画面下方可见. 视觉检测结果会报告
  球与夹爪的相对位置(偏移px和角度): "正下方"=已对准可闭合; "左侧/右侧"=
  需小幅转向. 这是比车头方位角更精确的近距离参照, 优先使用它做最终对准.

每次决策你自动收到: 车头相机图像 + 视觉检测结果 + 本体状态(无此工具).
你可以**一次输出多个工具调用**, 它们会按顺序执行完才回来问你下一步:
- 抓取序列: open_gripper → arm_pose("reach") → close_gripper → arm_pose("carry")
- 投放序列: arm_pose("drop") → open_gripper
- 驾驶: turn_left(0.3) → forward(1.5) → turn_right(0.2)
批量调用适合已确定的操作序列; 不确定时只输出一步, 下次看结果再决定.

规则:
1. 每轮可输出 1 个或多个工具调用(按顺序执行); 视觉检测结果自动附带在每轮观测中.
2. **方位角在 ±15° 以内就直接 forward, 不要继续微调!** 转向最小粒度约8°,
   视觉有±3°噪声, 追求精确归零只会来回振荡. 粗略对准 → 前进 → 到近处再修正.
3. 方位角"左正右负": 球偏左(+度)就 turn_left, 偏右就 turn_right; 角度大转久一点.
4. 距离估算有 ±10% 误差, 靠近目标时小步前进, 多轮确认.
5. close_gripper 后若结果说"没有持有球", 说明没对准: 张开手指退回重来.
6. 行驶前 arm_pose("stow"); 任务完成或确认无法完成时调用 done."""


def llm_log(tag: str, text: str) -> None:
    """把 LLM 的输入/输出摘要打印到终端 (青色标记便于过滤)."""
    print(f"\033[36m[{tag}]\033[0m {text}", flush=True)


def term_show_image(rgb, label: str = "", max_w: int = 400) -> None:
    """在支持 Kitty 图形协议的终端 (Ghostty/kitty/WezTerm) 内联显示图像."""
    if not getattr(config, "LLM_TERM_IMAGES", False):
        return
    import base64
    import sys as _sys
    from PIL import Image
    im = Image.fromarray(rgb)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    if label:
        print(f"\033[2m{label}\033[0m", flush=True)
    CHUNK, n, i, first = 4096, len(b64), 0, True
    while i < n:
        chunk = b64[i:i + CHUNK]
        last = i + CHUNK >= n
        if first and last:
            esc = f"\x1b_Gf=100,a=T;{chunk}\x1b\\"
        elif first:
            esc = f"\x1b_Gf=100,a=T,m=1;{chunk}\x1b\\"
        elif last:
            esc = f"\x1b_Gm=0;{chunk}\x1b\\"
        else:
            esc = f"\x1b_Gm=1;{chunk}\x1b\\"
        _sys.stdout.write(esc)
        first = False
        i += CHUNK
    _sys.stdout.write("\r\n")
    _sys.stdout.flush()


def encode_image_block(rgb: np.ndarray, fmt: str = "anthropic") -> dict:
    """rgb uint8 → jpeg base64 content block (anthropic 或 openai 格式)."""
    from PIL import Image
    im = Image.fromarray(rgb)
    if max(im.size) > config.LLM_MAX_IMAGE_EDGE:
        scale = config.LLM_MAX_IMAGE_EDGE / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if fmt == "anthropic":
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def obs_text(obs: dict, task_text: str, history: list[dict]) -> str:
    """构建发给 LLM 的观测文本: 任务 + 视觉检测 + 本体状态 + 历史."""
    from control import perception
    img = obs.get("images", {}).get("front")
    vision = ""
    if img is not None:
        dets = perception.analyze(img)
        grip = perception._gripper_tip(img.astype(np.float32) / 255.0,
                                       perception._rgb_to_hsv(img)[2])
        s = {"detections": dets, "gripper_tip": grip, "base_pose": obs.get("base_pose", [0,0,0]),
             "arm_q": np.array(obs.get("arm_qpos", [0,0])),
             "tcp_pos": np.array(obs.get("tcp_pos", [0,0,0])),
             "finger_open": obs.get("finger", 1.0), "holding": obs.get("holding", False)}
        vision = perception.sense_text(s)
    bp = obs.get("base_pose", [0, 0, 0])
    return (f"任务: {task_text}\n\n{vision}\n\n"
            f"历史动作与结果:\n{history_summary(history)}\n"
            f"请决定下一步 (可一次输出多个工具调用, 会按顺序执行):")


def history_summary(history: list[dict]) -> str:
    """最近 K 次决策的单行摘要 (省 token, 历史图像不重发)."""
    if not history:
        return "(尚无历史动作)"
    lines = []
    for h in history[-config.HISTORY_LINES:]:
        args = json.dumps(h.get("args", {}), ensure_ascii=False)
        flag = "OK" if h.get("ok") else "失败"
        res = str(h.get("result", ""))[:160].replace("\n", " ")
        lines.append(f"- {h.get('tool')}({args}) [{flag}] {res}")
    return "\n".join(lines)


_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_json_toolcall(text: str) -> dict | None:
    """兜底: 从自由文本里提取 {"thought":..,"tool":..,"args":{}} JSON."""
    if not text:
        return None
    m = _JSON_RE.search(text) or re.search(r"(\{[^{}]*\"tool\"[^{}]*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        if isinstance(d, dict) and "tool" in d:
            return d
    except json.JSONDecodeError:
        pass
    return None
