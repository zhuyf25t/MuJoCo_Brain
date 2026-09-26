"""Three task stages, one visual decision and a whole action chunk per round.

Only the collector executes robot tools. A graph invocation always ends after
planning, so a graph edge can never masquerade as a new camera observation.
"""

from __future__ import annotations

import atexit
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import TypedDict

import httpx

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime
    from .langgraph_memory import EpisodeMemory
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("langgraph"):
        raise
    raise ImportError("安装依赖: python -m pip install -r requirements-langgraph.txt") from exc

import config
from .base import Brain, Decision
from .langgraph_policy import (CALIBRATION_COMPARE_PROMPT, CALIBRATION_VIEW_PROMPT, COMMON_PROMPT,
                              CURRENT_VIEW_PROMPT, STAGE_GOALS, plan_tool, scene_tool)
from .llm_common import llm_log, term_show_image
from .openai_compat import OpenAICompatBrain

TOOL_NAMES = {"shoulder", "elbow", "arm_pose", "open_gripper", "close_gripper",
              "forward", "back", "turn_left", "turn_right", "observe", "done"}
POSITION_TOOLS = {"forward", "back", "turn_left", "turn_right", "shoulder", "elbow", "arm_pose"}
DEPTH_TOOLS = {"forward", "back", "shoulder", "elbow", "arm_pose"}


class EpisodeState(TypedDict):
    phase: str
    calibration: dict
    arm_anchor: dict | None
    visual_memory: dict
    last_batch: dict | None


def initial_state() -> EpisodeState:
    return {"phase": "explore", "calibration": {"lessons": [], "grasp": None, "clearance": None,
            "moves": {"to_grasp": None, "to_clearance": None}}, "arm_anchor": None,
            "visual_memory": {"target": None, "held": None}, "last_batch": None}


@dataclass
class RoundContext:
    frame: dict
    task: str
    tools: list[dict]
    commands: list[dict]
    round_no: int
    entry_node: str = ""
    decisions: list[Decision] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    scene: dict | None = None
    scene_reused: bool = False
    calibration_gripper: str | None = None
    release_needs_new_frame: bool = False


class LangGraphBrain(Brain):
    name = "langgraph"
    thread_id = "main"

    def __init__(self, *, base_url=None, api_key=None, model=None, transport=None,
                 db_path=None, trace_dir=None):
        self.db_path = Path(db_path) if db_path else Path(__file__).with_name("langgraph_temporary_file.sqlite")
        self.memory = EpisodeMemory(self.db_path, trace_dir or config.DATA_DIR / "langgraph_runs")
        self.llm = OpenAICompatBrain(base_url=base_url, api_key=api_key, model=model, transport=transport)
        self.graph = None
        self._graph_config = {"configurable": {"thread_id": self.thread_id}}
        self._closed = False
        self.reset()
        atexit.register(self.close)

    def reset(self):
        """The existing collector calls this at each episode; initialize on its first frame."""
        self._fresh, self._round = True, 0
        self._scene_cache = None
        self._gripper_cache = None

    def _start_episode(self):
        self.memory.open()
        if self.graph is None:
            builder = StateGraph(EpisodeState, context_schema=RoundContext)
            for name, node in (("explore", self._explore), ("pick", self._pick), ("place", self._place)):
                builder.add_node(name, node)
                builder.add_edge(name, END)
            builder.add_conditional_edges(START, lambda state: state["phase"], {name: name for name in STAGE_GOALS})
            self.graph = builder.compile(checkpointer=self.memory.checkpointer)
        self.memory.new_episode(self.thread_id)
        self._fresh = False

    @staticmethod
    def _consume_batch(state, commands):
        """Reconcile the ENTIRE prior batch, using command records only, never ok/result."""
        state = deepcopy(state)
        previous = state["last_batch"]
        if previous is None:
            if commands:
                raise RuntimeError("新 episode 必须从空 history 开始")
            return state
        n = previous["history_n"]
        if len(commands) != n + len(previous["actions"]) or commands[n:] != previous["actions"]:
            raise RuntimeError("history 与上批动作不一致；停止以避免重复执行")
        for command in commands[n:]:
            tool = command["tool"]
            if tool in {"arm_pose", "elbow"}:
                # Keep the pictures for comparison, but changing the elbow loses the learned route.
                state["arm_anchor"] = None
                state["calibration"]["moves"] = {"to_grasp": None, "to_clearance": None}
        return state

    @staticmethod
    def _shoulder_segment(anchor, ctx):
        """An observed endpoint pair can certify a segment, never a sum of attempted motion."""
        if anchor is None or anchor["frame"]["image_id"] == ctx.frame["image_id"]:
            return None
        commands = ctx.commands[anchor["frame"]["history_n"]:]
        if any(c["tool"] not in {"shoulder", "observe"} for c in commands):
            return None
        actions = [deepcopy(c) for c in commands if c["tool"] == "shoulder"]
        deltas = [a["args"]["delta"] for a in actions]
        if not deltas or not (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)):
            return None
        return {"actions": actions, "before": anchor["frame"], "after": ctx.frame}

    def _observe_arm(self, state, ctx, view):
        if view not in {"grasp", "clearance", "other", "unclear"}:
            raise ValueError("arm_view须描述当前爪形态：grasp/clearance/other/unclear")
        calibration, anchor = state["calibration"], state["arm_anchor"]
        if view not in {"grasp", "clearance"}:
            return
        if calibration[view] is None:
            raise ValueError("当前没有这张姿态参考；先保存当前实图参考，或用other/unclear")
        if anchor and anchor["pose"] != view:
            segment = self._shoulder_segment(anchor, ctx)
            other = calibration["moves"]["to_" + anchor["pose"]]
            if segment and other and segment["actions"][0]["args"]["delta"] * other["actions"][0]["args"]["delta"] > 0:
                raise ValueError("固定肘的两个相反肩往返方向不能同号；重新核对当前图，不要把影子或预测当到位")
            if segment and calibration["moves"]["to_" + view] is None:
                calibration["moves"]["to_" + view] = segment
        # Seeing the same low pose after an ineffective push rebases the next experiment.
        # Its failed pushes are excluded from a subsequent lift, without reading ok/qpos.
        state["arm_anchor"] = {"pose": view, "frame": ctx.frame}

    @staticmethod
    def _may_hold(state, ctx, report):
        gripper = [c["tool"] for c in ctx.commands if c["tool"] in {"open_gripper", "close_gripper"}]
        return (bool(state["visual_memory"]["held"]) or bool(gripper and gripper[-1] == "close_gripper") or
                ctx.scene["holding"] == "held" or report["holding"] == "held")

    def _validate_release(self, original, ctx, report):
        names = [a["tool"] for a in report["actions"]]
        if "open_gripper" not in names or not self._may_hold(original, ctx, report):
            return
        if ctx.scene["holding"] == "empty":
            return  # An independent current view allows preparation after an empty grasp.
        if any(n in POSITION_TOOLS for n in names[:names.index("open_gripper")]):
            ctx.release_needs_new_frame = True
            raise ValueError("持球时定位与释放不能同批：先完成转向/接近/调臂，看下一张图后再松爪；本轮重规划也不能删除定位动作直接松爪")
        if ctx.release_needs_new_frame:
            raise ValueError("本轮先前计划还需要定位，没有新图不能改称已经到位；保持闭爪，完成必要定位或观察后再判断释放")
        if ctx.scene["holding"] != "held" or report["holding"] != "held":
            raise ValueError("可能持球但当前夹持关系看不清，先保持闭爪改善观察；规划改称held不能替代独立当前图确认")
        if ctx.scene["release_view"] != "candidate":
            raise ValueError("独立当前图观测不支持释放：" + ctx.scene["evidence"] + "。保持闭爪，先解决高度、前后位置或观察问题")
        if report["phase"] != "place":
            raise ValueError("持球松爪属于place，先确认投放位置")
        check = report.get("release_check")
        if (not isinstance(check, dict) or set(check) != {"clear_drop_path", "inside_opening", "evidence"} or
                check["clear_drop_path"] is not True or check["inside_opening"] is not True):
            raise ValueError("释放须确认球向下落不撞箱沿或外壁、整颗落点在箱内；release_check两项均须为true")
        self._text(check["evidence"], "release_check.evidence")
        target = original["visual_memory"]["target"]
        if not target or target["kind"] != "bin":
            raise ValueError("刚找到箱子不能凭单图重叠释放；先保存箱子图，保持闭爪接近/调臂再比较")
        since = ctx.commands[target["frame"]["history_n"]:]
        if (target["frame"]["image_id"] == ctx.frame["image_id"] or
                not any(c["tool"] in DEPTH_TOOLS for c in since)):
            raise ValueError("须比较接近或调臂前后的箱子图；只有转向、等待或同一张图不能确认前后关系")

    def _messages(self, state, ctx, note=""):
        candidates = []
        if state["last_batch"]:
            candidates.append(("历史：上批动作前图", state["last_batch"]["before"]))
        pose_refs = () if state["phase"] == "place" else (("grasp", "张爪朝下、靠近地面的抓球参考"), ("clearance", "抬肩后能看球看路的就绪参考"))
        for key, label in pose_refs:
            if state["calibration"][key]:
                candidates.append(("历史：" + label, state["calibration"][key]["frame"]))
        for key, label in (("held", "最近清楚的持球证据"), ("target", "上次目标所在图")):
            if state["visual_memory"][key]:
                if key == "target" and state["visual_memory"][key]["kind"] == "ball":
                    continue  # A previous ball scene is not a current target measurement.
                candidates.append(("历史：" + label, state["visual_memory"][key]["frame"]))
        current_label = "唯一当前图（本轮决策依据；前面的图片全是历史）"
        candidates.append((current_label, ctx.frame))
        images = []
        for label, frame in candidates:
            found = next((entry for entry in images if entry[1]["image_id"] == frame["image_id"]), None)
            if found:
                if label == current_label:
                    # Even an unchanged frame belongs at the end, with the current timestamp.
                    images.remove(found)
                    images.append([f"{current_label}；也与{found[0]}(动作数{found[1]['history_n']})图像相同", frame])
                else:
                    found[0] += f"；也对应{label}(动作数{frame['history_n']})"
            else:
                images.append([label, frame])
        tools = deepcopy(ctx.tools)
        for tool in tools:
            if tool["name"] in {"shoulder", "elbow"}:
                tool["description"] = "让这个关节改变delta弧度；不是持续秒数，爪抬高还是降低要比较前后图"
        body = {"任务": ctx.task, "决策轮": f"{ctx.round_no}/{config.MAX_DECISIONS}", "状态": state,
                "剩余看图决策次数（含本轮）": max(0, config.MAX_DECISIONS - ctx.round_no + 1),
                "最近调用的指令": ctx.commands[-config.HISTORY_LINES:], "可用工具": tools}
        body["图像与历史文字的时间"] = ("最后一张才是唯一当前图。历史note/learning里的位置、大小以及‘当前’都是当时的描述，"
                                  "不能作为现在的球位置；先看最后一图，历史只用于比较变化和复用动作。")
        body["抓球区域含义"] = ("grasp.region是估计的球心接近位置，不是球的外接框或已证明能夹住的位置。"
                                "首次抓取和抓空后，先降到低位取得球与实体夹指同图，再决定闭爪。")
        moves = state["calibration"]["moves"]
        if moves["to_clearance"] and not moves["to_grasp"]:
            body["待验证的降回动作（仅从就绪参考出发）"] = [
                {"tool": "shoulder", "args": {"delta": -a["args"]["delta"]}}
                for a in reversed(moves["to_clearance"]["actions"])]
        body["参考关系"] = ("moves保存两端实图确认过的肩动作，不是当前关节角。arm_anchor只说明最近一次认出的姿态；"
                          "之后执行过肩动作就不能当成当前位置。照片比较看爪相对车头的形态，车外背景可以变化。"
                          "肘或预设改变后，照片保留用于恢复，但往返动作须重新确认。")
        content = [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]
        if note:
            content.append({"type": "text", "text": note})
        for label, frame in images:
            content.extend([{"type": "text", "text": f"{label}；已调用动作数={frame['history_n']}"},
                            self.memory.image_block(frame)])
        scene = deepcopy(ctx.scene)
        geometry_note = ""
        if ctx.calibration_gripper:
            scene["gripper"] = ctx.calibration_gripper
            # The base evidence can repeat its mistaken shadow identity. Keep it
            # in the trace and release checks, not alongside the revised geometry.
            scene.pop("evidence")
            geometry_note = (
                "本轮初始标定的gripper由另一次结合本车外观说明的看图复核提供，帮助辨认实体掌/夹块的位置、形态和遮挡。"
                "其中若有‘与前图比较’，是直接对照上批动作前和现在的实图，不读取动作预期或旧结论；把可见变化与实际指令对应，不把两次粗估坐标的差当成运动。"
                "球坐标、holding、release_view仍来自原独立观测，未被复核替换；夹爪描述中提到的空爪或持球不能覆盖这些字段。"
                "保存姿态仍须核对当前实体与地面，形态复核不是已经到达低位或就绪位的证明。")
        scene_scope = ("当前图观测；初始gripper复核的来源与比较范围见下文" if ctx.calibration_gripper else
                       "独立只看最后这张当前图得到；不是历史或动作预测")
        content.append({"type": "text", "text": f"current_scene（{scene_scope}）：" +
                        json.dumps(scene, ensure_ascii=False) +
                        "\n" + geometry_note +
                        "\n先根据这份当前观测说明下一批的目的。历史只用来比较动作效果，不能把旧球位置当成现在。"})
        # Include adjacent goals so a visually confirmed boundary needs no second model call.
        goals = "\n\n".join(f"{name}: {goal}" for name, goal in STAGE_GOALS.items())
        messages = [{"role": "system", "content": COMMON_PROMPT + "\n阶段目标：\n" + goals +
                     f"\n本轮从 {ctx.entry_node} 开始：上一批实际变成了什么样？这批要达到什么目的、具体怎样做？"
                     "请根据当前图回答；需要下一张图才能判断的地方，就留给下一轮。"},
                    {"role": "user", "content": content}]
        return messages, plan_tool([t["name"] for t in tools]), images

    def _explore(self, state: EpisodeState, runtime: Runtime[RoundContext]):
        return self._plan(state, runtime.context, "explore")

    def _pick(self, state: EpisodeState, runtime: Runtime[RoundContext]):
        return self._plan(state, runtime.context, "pick")

    def _place(self, state: EpisodeState, runtime: Runtime[RoundContext]):
        return self._plan(state, runtime.context, "place")

    def _plan(self, state, ctx, entry_node):
        ctx.entry_node = entry_node
        state = self._consume_batch(state, ctx.commands)
        try:
            self._read_scene(ctx)
            if (entry_node == "explore" and not state["visual_memory"]["held"] and ctx.scene["holding"] != "held" and
                    any(c["tool"] == "open_gripper" for c in ctx.commands) and
                    not any(c["tool"] == "close_gripper" for c in ctx.commands)):
                self._read_calibration_gripper(ctx, state)
        except (ValueError, TypeError, KeyError) as exc:
            return self._emit_batch(state, ctx, [{"tool": "observe", "args": {}}],
                                    "当前图观测格式不完整，先保持姿态重新观察", str(exc))
        # Preserve direct visual evidence even if the planner is uncertain or malformed.
        if ctx.scene["holding"] == "held":
            state["visual_memory"]["held"] = {"frame": ctx.frame, "note": ctx.scene["gripper"]}
        elif ctx.scene["holding"] == "empty":
            state["visual_memory"]["held"] = None
        note = ""
        for _ in range(2):  # One planner request plus one correction; the observation is fixed.
            messages, tool, images = self._messages(state, ctx, note)
            request = {"kind": "plan", "phase": state["phase"], "messages": self.memory.archive_messages(messages, images), "tools": [tool]}
            ctx.requests.append(request)
            started = time.monotonic()
            try:
                data = self.llm._request(messages, [tool])
                msg = data.get("choices", [{}])[0].get("message", {})
                request.update(reply=msg.get("tool_calls") or msg.get("content"), usage=data.get("usage"))
            except Exception as exc:
                request["error"] = type(exc).__name__
                raise
            finally:
                request["elapsed_s"] = round(time.monotonic() - started, 3)
            try:
                return self._apply_plan(state, ctx, self._parse(msg))
            except (ValueError, TypeError, KeyError) as exc:
                request["validation_error"] = str(exc)
                note = f"\n上份计划未通过校验：{exc}。尚未执行任何动作，请修正整批计划。"
                llm_log("LangGraph重试", str(exc))
        # Bad JSON is not a failed physical task. Keep memory and acquire a fresh frame.
        return self._emit_batch(state, ctx, [{"tool": "observe", "args": {}}],
                                "计划未通过校验，原位重观察", "保持姿态，下轮修正计划。" + note)

    @staticmethod
    def _parse(message, name="report_plan"):
        calls = message.get("tool_calls") or []
        if calls:
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != name:
                raise ValueError(f"请调用一次 {name}")
            return json.loads(calls[0]["function"].get("arguments", "{}"))
        text = (message.get("content") or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    def _read_scene(self, ctx):
        if self._scene_cache and self._scene_cache[0] == ctx.frame["image_id"]:
            ctx.scene, ctx.scene_reused = deepcopy(self._scene_cache[1]), True
            return
        ctx.scene = self._describe_current(ctx, CURRENT_VIEW_PROMPT, "observation")
        self._scene_cache = (ctx.frame["image_id"], deepcopy(ctx.scene))

    def _read_calibration_gripper(self, ctx, state):
        previous = state["last_batch"]
        actions = previous["actions"] if previous else []
        deltas = [a["args"]["delta"] for a in actions if a["tool"] == "shoulder"]
        before = None
        # _consume_batch already matched this complete batch against command history.
        # Compare the adjacent images, not a pose reference or a predicted result.
        if (deltas and all(a["tool"] in {"shoulder", "observe"} for a in actions) and
                (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))):
            before = previous["before"]
        cache_key = (before["image_id"] if before else None, ctx.frame["image_id"])
        if self._gripper_cache and self._gripper_cache[0] == cache_key:
            ctx.calibration_gripper = self._gripper_cache[1]
            return
        try:
            prompt = CALIBRATION_COMPARE_PROMPT if before else CALIBRATION_VIEW_PROMPT
            report = self._describe_current(ctx, prompt, "calibration_gripper", before=before)
        except (ValueError, TypeError, KeyError, httpx.HTTPError):
            self._gripper_cache = (cache_key, None)
            return  # An optional shape reading cannot invalidate a valid base scene.
        # Never replace holding, release evidence or coordinates with this probe.
        # A cropped opening can make occupancy unclear while the visible jaws
        # are identifiable. Initial-calibration eligibility is checked by _plan.
        ctx.calibration_gripper = report["gripper"] if report["holding"] in {"empty", "unclear"} else None
        self._gripper_cache = (cache_key, ctx.calibration_gripper)

    def _describe_current(self, ctx, prompt, kind, *, before=None):
        images = [["唯一当前图", ctx.frame]]
        content = [{"type": "text", "text": "只描述这一张当前图，调用describe_scene。"}]
        if before is not None:
            images.insert(0, ["上批动作前图", before])
            content = [{"type": "text", "text": "第一张：上批动作之前的实图。"}, self.memory.image_block(before),
                       {"type": "text", "text": "第二张：唯一当前实图。调用describe_scene；gripper附上与第一张实体夹爪的直接比较。"}]
        content.append(self.memory.image_block(ctx.frame))
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
        tool = scene_tool()
        if before is not None:
            tool["function"]["description"] = "根据第二张当前图片报告物体和空间关系；gripper还需与第一张的同一实体夹爪直接比较，不作动作规划。"
        request = {"kind": kind, "messages": self.memory.archive_messages(messages, images), "tools": [tool]}
        ctx.requests.append(request)
        started = time.monotonic()
        try:
            data = self.llm._request(messages, [tool])
            msg = data.get("choices", [{}])[0].get("message", {})
            request.update(reply=msg.get("tool_calls") or msg.get("content"), usage=data.get("usage"))
            scene = self._parse(msg, "describe_scene")
            self._validate_scene(scene)
            return scene
        except (ValueError, TypeError, KeyError) as exc:
            request["validation_error"] = str(exc)
            raise
        except Exception as exc:
            request["error"] = type(exc).__name__
            raise
        finally:
            request["elapsed_s"] = round(time.monotonic() - started, 3)

    def _validate_scene(self, scene):
        if not isinstance(scene, dict) or set(scene) != {"ground_balls", "gripper", "holding", "release_view", "evidence"}:
            raise ValueError("独立观测字段缺失或多余")
        if scene["holding"] not in {"held", "empty", "unclear"} or scene["release_view"] not in {"blocked", "candidate", "unclear"}:
            raise ValueError("独立观测的夹持或释放关系不合法")
        self._text(scene["gripper"], "scene.gripper")
        self._text(scene["evidence"], "scene.evidence")
        if not isinstance(scene["ground_balls"], list):
            raise ValueError("ground_balls必须是列表")
        for ball in scene["ground_balls"]:
            if not isinstance(ball, dict) or set(ball) != {"center", "description"}:
                raise ValueError("地面球需要center和description")
            center = ball["center"]
            if (not isinstance(center, list) or len(center) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in center)):
                raise ValueError("球心须为左上原点、0到1的[x,y]")
            self._text(ball["description"], "scene.ball.description")

    @staticmethod
    def _validate_actions(actions, schemas):
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions 必须是非空的有序动作列表")
        by_name = {t["name"]: t["input_schema"] for t in schemas}
        for action in actions:
            if not isinstance(action, dict) or set(action) != {"tool", "args"}:
                raise ValueError("每个动作只能包含 tool 和 args")
            name, args = action["tool"], action["args"]
            if name not in by_name or not isinstance(args, dict):
                raise ValueError("未知工具或参数格式错误")
            schema = by_name[name]
            props = schema.get("properties", {})
            if set(args) - set(props) or set(schema.get("required", [])) - set(args):
                raise ValueError("工具参数缺失或包含未知字段")
            for key, value in args.items():
                spec = props[key]
                kind = spec.get("type")
                if kind == "number":
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("动作参数必须是有限数字")
                    if not spec.get("minimum", -math.inf) <= value <= spec.get("maximum", math.inf):
                        raise ValueError("动作参数超出工具范围")
                if (kind == "string" and not isinstance(value, str)) or (kind == "boolean" and type(value) is not bool):
                    raise ValueError("动作参数类型错误")
            if name == "arm_pose" and args.get("pose") not in {"stow", "carry", "reach", "drop"}:
                raise ValueError("未知臂姿态")
        names = [a["tool"] for a in actions]
        if "done" in names and len(names) != 1:
            raise ValueError("done 必须单独返回，不能根据本批尚未执行的动作宣告完成")
        if "close_gripper" in names and "open_gripper" in names[names.index("close_gripper") + 1:]:
            raise ValueError("闭爪后须等下一轮观察，再决定是否松爪")

    @staticmethod
    def _text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 需要非空文字依据")
        return value.strip()

    def _grasp_ref(self, ref, ctx, field):
        if not isinstance(ref, dict) or set(ref) != {"region", "note"}:
            raise ValueError(f"{field} 需要 region 和 note")
        box = ref["region"]
        if (not isinstance(box, list) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box) or
                not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError("region 需要 [left,top,right,bottom]，满足0到1且面积非零")
        return {"frame": ctx.frame, "region": deepcopy(box), "note": self._text(ref["note"], field + ".note")}

    def _apply_plan(self, original, ctx, report):
        if not isinstance(report, dict):
            raise ValueError("计划必须是 JSON 对象")
        if set(report) - {"phase", "holding", "reason", "expected", "actions", "learning",
                          "grasp_reference", "grasp_region", "clearance_reference", "target", "arm_view", "release_check"}:
            raise ValueError("计划包含未知字段，请按report_plan定义输出")
        if report.get("phase") not in STAGE_GOALS:
            raise ValueError("phase 必须为 explore/pick/place，表示当前证据支持的工作阶段")
        if report.get("holding") not in {"held", "empty", "unclear"}:
            raise ValueError("holding 必须描述当前图：held/empty/unclear")
        reason = self._text(report.get("reason"), "reason")
        expected = self._text(report.get("expected"), "expected")
        actions = report.get("actions")
        self._validate_actions(actions, ctx.tools)  # Validate the whole batch before returning any part.
        # Remember unexecuted positioning before other report errors can trigger
        # a retry that drops those actions and changes its holding claim.
        self._validate_release(original, ctx, report)
        names = [a["tool"] for a in actions]
        state = deepcopy(original)
        calibration = state["calibration"]
        view = report["holding"]
        gripper = [c["tool"] for c in ctx.commands if c["tool"] in {"open_gripper", "close_gripper"}]
        may_hold = self._may_hold(state, ctx, report)
        if may_hold and view == "empty" and ctx.scene["holding"] != "empty":
            raise ValueError("独立当前图没有确认空爪，不能用规划中的empty清除可能持球；先保持闭爪改善观察")
        if view == "held" and ctx.scene["holding"] == "empty":
            raise ValueError("独立当前图报告空爪，不能直接声称持球；先核对图片、改善观察")
        if "learning" in report and state["last_batch"] is not None:
            lesson = {"text": self._text(report["learning"], "learning"), "through_history_n": len(ctx.commands)}
            if not calibration["lessons"] or lesson["text"] != calibration["lessons"][-1]["text"]:
                calibration["lessons"] = (calibration["lessons"] + [lesson])[-4:]
        if "grasp_reference" in report:
            ref = report["grasp_reference"]
            if ref is not None:
                opened_without_possible_hold = bool(gripper and gripper[-1] == "open_gripper") and not may_hold
                if (view == "held" or (gripper and gripper[-1] == "close_gripper") or
                    (view == "unclear" and not opened_without_possible_hold)):
                    raise ValueError("抓取几何参考须来自当前可见张爪；已执行open且无可能持球记录时可unclear，不能用本批张爪预测")
                ref = self._grasp_ref(ref, ctx, "grasp_reference")
            calibration["grasp"], calibration["clearance"] = ref, None
            calibration["moves"] = {"to_grasp": None, "to_clearance": None}
            state["arm_anchor"] = {"pose": "grasp", "frame": ctx.frame} if ref else None
            if ref is not None and report.get("arm_view") != "grasp":
                raise ValueError("保存低位当前图时arm_view须为grasp")
        if "clearance_reference" in report:
            ref = report["clearance_reference"]
            grasp = calibration["grasp"]
            if ref is not None:
                if not isinstance(ref, dict) or set(ref) != {"save_current", "note"} or ref["save_current"] is not True:
                    raise ValueError("clearance_reference须为{save_current:true,note:当前实图依据}；尚无可用参考就省略或null")
                anchor = state["arm_anchor"]
                if (grasp is None or anchor is None or anchor["pose"] != "grasp" or
                        self._shoulder_segment(anchor, ctx) is None):
                    raise ValueError("就绪参考须来自最近确认的低位之后、仅单方向肩动作的新图；受阻或反复换向先重新确认低位")
                if report.get("arm_view") != "clearance":
                    raise ValueError("保存就绪当前图时arm_view须为clearance")
                ref = {"frame": ctx.frame, "note": self._text(ref["note"], "clearance_reference.note")}
            else:
                calibration["moves"] = {"to_grasp": None, "to_clearance": None}
                state["arm_anchor"] = None
            calibration["clearance"] = ref
        self._observe_arm(state, ctx, report.get("arm_view"))
        if "grasp_region" in report:
            if ("grasp_reference" in report or "clearance_reference" in report or
                    report.get("arm_view") != "grasp" or not calibration["grasp"] or
                    not calibration["clearance"] or not all(calibration["moves"].values())):
                raise ValueError("修正抓球区域须回到同一低位且原肩往返有效；不能同时更换姿态参考")
            if may_hold or not gripper or gripper[-1] != "open_gripper" or not ctx.scene["ground_balls"]:
                raise ValueError("修正抓球区域须当前已张爪、没有可能持球，并看见地面球；不能依据本批未来动作")
            # A better estimate of where the ball belongs does not relearn the arm route.
            calibration["grasp"] = self._grasp_ref(report["grasp_region"], ctx, "grasp_region")
        phase = report["phase"]
        if names == ["done"] and actions[0]["args"]["success"]:
            if phase != "place" or original["phase"] != "place" or view == "held" or not gripper or gripper[-1] != "open_gripper":
                raise ValueError("done(true)须在投放后的当前实图确认完成，不能根据本批预测")
        if names != ["done"]:
            if phase == "pick" and not (calibration["grasp"] and calibration["clearance"] and
                                         all(calibration["moves"].values())):
                raise ValueError("捡球须先用实图验证低位→就绪、就绪→低位两个方向；继续explore验证，不能把反向预测当已完成")
            if phase == "pick" and any(n in {"elbow", "arm_pose"} for n in names):
                raise ValueError("pick复用固定肘的抓球方法；确实需要改肘/换预设时先回explore重建")
            if phase == "place" and original["phase"] != "place" and view != "held":
                raise ValueError("进入place须当前看清持球；预测闭爪抬起后仍用pick，下轮再核对")
            # A ball between open jaws can be mistaken for an already lifted ball.
            # Preserve a planned close/lift check; the label must not force transport.
            securing_grasp = (original["phase"] == phase == "pick" and gripper and
                              gripper[-1] == "open_gripper" and names[0] == "close_gripper" and
                              all(n in {"close_gripper", "shoulder", "observe"} for n in names))
            if view == "held" and phase != "place" and not securing_grasp:
                raise ValueError("holding=held与当前阶段不一致；若确已夹稳可进入place，若证据仍不确定应如实填unclear，不要为换阶段省掉尚未执行的抓取与确认动作")
            if original["phase"] == "place" and phase == "pick" and view != "empty":
                raise ValueError("回捡球须当前看清空爪/掉球；看不清可保持闭爪改善视野或回explore恢复参照")
        if "target" in report:
            target = report["target"]
            if target is not None:
                if not isinstance(target, dict) or target.get("kind") not in {"ball", "bin"}:
                    raise ValueError("target 需要 kind(ball/bin) 和一句 note，或为 null")
                target = {"kind": target["kind"], "note": self._text(target.get("note"), "target.note"), "frame": ctx.frame}
            state["visual_memory"]["target"] = target
        if view == "held":
            state["visual_memory"]["held"] = {"frame": ctx.frame, "note": report["reason"][:400]}
        elif view == "empty":
            state["visual_memory"]["held"] = None
        # Neither predicted holding nor predicted joint positions are observations.
        state["phase"] = phase
        return self._emit_batch(state, ctx, actions, reason, expected)

    @staticmethod
    def _emit_batch(state, ctx, actions, reason, expected):
        state = deepcopy(state)
        state["last_batch"] = {"before": ctx.frame, "history_n": len(ctx.commands),
                               "actions": deepcopy(actions), "expected": expected}
        ctx.decisions = [Decision(f"[{state['phase']}] {reason} | 预期：{expected}" if i == 0 else "",
                                  action["tool"], deepcopy(action["args"])) for i, action in enumerate(actions)]
        return state

    def decide(self, obs, task_text, tool_schemas, history) -> list[Decision]:
        if self._closed:
            raise RuntimeError("LangGraphBrain 已关闭")
        front = obs.get("images", {}).get("front")
        if front is None:
            raise ValueError("LangGraph 需要 images.front 车头图片")
        if not isinstance(history, list):
            raise ValueError("history 应是列表；每集第一轮传 []")
        if self._fresh:
            if history:
                raise ValueError("新 episode 必须从空 history 开始")
            self._start_episode()
        commands = [{"tool": e.get("tool"), "args": deepcopy(e.get("args", {}))} for e in history]
        tools = [t for t in tool_schemas if t["name"] in TOOL_NAMES]
        before = self.state
        ctx = RoundContext(self.memory.save_image(front, len(commands)), task_text, tools, commands, self._round + 1)
        term_show_image(front, label="LangGraph → 当前车头图")
        llm_log("LangGraph", f"轮{ctx.round_no} | 入口阶段={(before or initial_state())['phase']} | 等待看图规划")
        try:
            self.graph.invoke({} if before else initial_state(), config=self._graph_config, context=ctx)
            if not ctx.decisions:
                raise RuntimeError("图未产生动作批次")
            self._round = ctx.round_no
        finally:
            self.memory.log({"format_version": 7, "round": ctx.round_no, "model": self.llm.model,
                             "entry_node": ctx.entry_node, "action_phase": self.state.get("phase"),
                             "current_frame": ctx.frame, "current_scene": ctx.scene, "scene_reused": ctx.scene_reused,
                             "calibration_gripper": ctx.calibration_gripper,
                             "state_before": before, "state_after": self.state,
                             "requests": ctx.requests, "decisions": [asdict(d) for d in ctx.decisions]})
        phase = self.state["phase"]
        labels = {"explore": "探索", "pick": "捡球", "place": "投放"}
        transition = f" | 阶段切换 {ctx.entry_node} → {phase}" if ctx.entry_node != phase else ""
        llm_log("LangGraph→批次", f"轮{ctx.round_no} | 当前动作阶段={phase}（{labels[phase]}）{transition} | "
                f"{len(ctx.decisions)}个动作：{[d.tool for d in ctx.decisions]} | {ctx.decisions[0].thought}")
        return ctx.decisions

    @property
    def state(self):
        return {} if self._fresh or self.graph is None else deepcopy(self.graph.get_state(self._graph_config).values)

    @property
    def trace_dir(self):
        return self.memory.trace_dir

    def close(self):
        if not self._closed:
            self._closed = True
            atexit.unregister(self.close)
            self.llm._client.close()
            self.memory.close()
