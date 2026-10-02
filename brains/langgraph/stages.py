"""Five small task stages; capability results never control actuators directly."""

from .contracts import CapabilityInput, Frame, InfoRequest, TargetIdentity, direction_request
from .execution import authorize, command, drop, grasp, issue
from .learning import learn_motion


class FinalCheckError(RuntimeError):
    pass


class ProbeRequired(Exception):
    def __init__(self, request):
        self.request = request


class InformationUnavailable(Exception):
    """The model cannot decide from a reference it already received."""


class Stages:
    def __init__(self, profile, memory, capabilities):
        self.profile = profile
        self.memory = memory
        self.capabilities = capabilities
        self.task = ""
        self.log = None

    def context(self, state, **kwargs):
        return CapabilityInput(frame=state.frame, summary=self.memory.directory(), **kwargs)

    def fixed(self, state, commands, note):
        return issue(state, commands, note, fixed=True)

    def locate(self, state, name):
        detection = self.capabilities.call(name, self.context(
            state, task=self.task,
            previous_target=TargetIdentity(label=state.target.label) if state.target else None))
        state.target = detection.target
        state.target_frame_id = state.frame.id
        state.assessment = None
        learn_motion(state, self.capabilities, self.memory, self.log)
        return detection

    def unknown(self, state, note):
        state.unknown_count += 1
        if state.unknown_count >= self.profile["max_unknown"]:
            raise RuntimeError(f"连续无法判定，保留现场: {note}")
        return self.fixed(state, [command("observe")], note)

    def resolve(self, state, name, context, allowed):
        """Same-frame information loop; a miss unwinds before any task action."""
        # Planning receives the same reach reference used for the spatial assessment.
        evidence = ([s for s in state.loaded if s["kind"] == "reach"]
                    if name == "plan_motion" and context.intent != "search_right" else [])

        def load(request, source):
            state.request = request
            samples = self.memory.query(request)
            self.log.append("request_info", phase=state.phase, frame_id=state.frame.id,
                            request=request.model_dump(), hit=bool(samples), source=source)
            if not samples:
                raise ProbeRequired(request)
            evidence.extend(s for s in samples if s["id"] not in {e["id"] for e in evidence})
            loaded_ids = {s["id"] for s in state.loaded}
            state.loaded.extend(s for s in samples if s["id"] not in loaded_ids)

        # A visible target makes the reach dependency deterministic; do not spend a
        # model call merely asking for the same reference on every new frame.
        if name in ("match_grasp", "match_drop"):
            load(InfoRequest(kind="reach"), "stage_dependency")
        elif name == "plan_motion" and state.last_transition:
            commands = state.last_transition["commands"]
            directions = {c["tool"] for c in commands}
            if len(directions) == 1 and next(iter(directions)) in ("forward", "back", "turn_left", "turn_right"):
                direction = next(iter(directions))
                recent = direction_request(direction)
                if (context.intent != "search_right" or direction == "turn_right") and self.memory.query(recent):
                    load(recent, "recent_motion")

        feedback = ""
        for _ in range(self.profile["max_info_requests"]):
            value = context.model_copy(update={"evidence": evidence, "info_feedback": feedback})
            result = self.capabilities.call(name, value)
            if result.status == "need_info":
                request = result.request
            else:
                # Enforce declared prerequisites even if a model skips its request.
                request = (InfoRequest(kind="reach") if name in ("match_grasp", "match_drop")
                           else direction_request(result.direction) if result.status == "move" else None)
                if request is None or any(s["kind"] == request.kind and s["direction"] == request.direction
                                          for s in evidence):
                    return result
            if request.kind not in allowed:
                raise RuntimeError(f"{name} requested unrelated information")
            if any(s["kind"] == request.kind and s["direction"] == request.direction for s in evidence):
                self.log.append("repeated_request", capability=name, request=request.model_dump())
                if feedback:
                    raise InformationUnavailable(
                        f"{name} 已收到 {request.kind}/{request.direction} 原图与分析，"
                        "提醒后仍重复申请，当前依据无法完成判断")
                feedback = (f"{request.kind}/{request.direction} 的全部现有样本已在 evidence 中，"
                            "没有更细的额外样本。请据此判断；依据仍不足就返回 unknown。")
                continue
            load(request, "model")
        raise RuntimeError("资料申请次数超限")

    def move(self, state, intent, note=""):
        value = self.context(state, target=state.target, intent=intent,
                             assessment=state.assessment,
                             max_seconds=self.profile["max_motion_seconds"])
        result = self.resolve(state, "plan_motion", value, {"translation", "rotation"})
        if result.status == "unknown":
            return self.unknown(state, result.note)
        if intent == "search_right" and result.direction != "turn_right":
            raise RuntimeError("Search may only turn right")
        if result.seconds > self.profile["max_motion_seconds"]:
            raise RuntimeError("Motion exceeds configured small-step limit")
        if result.direction == "forward" and (state.target is None or state.target.path != "clear"):
            raise RuntimeError("Forward approach requires a visible clear path")
        required = direction_request(result.direction)
        authorize(state, result.direction, [required])
        state.unknown_count = 0
        return issue(state, [command(result.direction, seconds=result.seconds)],
                     note or result.note, action=result.direction)

    def probe(self, state, request):
        if request.kind == "reach":
            raise RuntimeError("reach 资料必须在起源阶段先补齐")
        if request.direction == "forward" and (state.target is None or state.target.path != "clear"):
            raise RuntimeError("Cannot probe forward without a clear visible path")
        self.log.append("probe", frame_id=state.frame.id, request=request.model_dump())
        # Exactly one operation; no catch/drop may accompany a database miss.
        return self.fixed(state, [command(request.direction, seconds=self.profile["probe_seconds"])],
                          "经验缺失：只执行一次小步试探，等待新图")

    def origin(self, state):
        if not state.started:
            state.started = True
            state.step = "check_reach"
            return self.fixed(state, [command("arm_pose", pose="stow")], "新任务先收臂")
        request = InfoRequest(kind="reach")
        if state.step == "reach_image":
            result = self.capabilities.call("analyze_reach", self.context(state))
            self.memory.add(request, [state.frame], [], result)
            state.step = "stowed"
            state.unknown_count = 0
            return self.fixed(state, [command("arm_pose", pose="stow")],
                              "reach 原图与分析已保存，恢复收臂；区域是否可用由匹配模块判断")
        references = self.memory.query(request)
        if not references:
            state.step = "reach_image"
            return self.fixed(state, [command("arm_pose", pose="reach")], "首次采集 reach 参考图")
        for sample in references:
            if sample.get("needs_analysis"):
                reference = Frame.model_validate(sample["frames"][0])
                result = self.capabilities.call("analyze_reach", CapabilityInput(frame=reference))
                self.memory.refresh_reach(sample["id"], result)
                self.log.append("reference_reanalyzed", sample_id=sample["id"], frame_id=reference.id)
        state.phase = "empty"
        state.step = "search"
        return []

    def empty(self, state):
        detection = self.locate(state, "find_ball")
        # Old positions never survive a new-frame detection, including unknown/absent.
        state.target = detection.target
        if detection.status == "unknown":
            return self.unknown(state, detection.note)
        if detection.status == "absent":
            return self.move(state, "search_right")
        state.target = detection.target
        if state.target.path == "unknown":
            return self.unknown(state, "目标路径看不清")
        if state.target.path == "blocked":
            state.target = None
            return self.move(state, "search_right", "目标路径受阻，继续搜索")
        match = self.resolve(state, "match_grasp", self.context(state, target=state.target), {"reach"})
        state.assessment = match
        if match.status == "unknown":
            return self.unknown(state, match.note)
        if match.status == "no":
            return self.move(state, "approach_ball: " + match.note)
        authorize(state, "grasp", [InfoRequest(kind="reach")])
        actions = issue(state, grasp(), "范围匹配，执行抓取组合；结果待新图确认", action="grasp")
        state.phase = "pending"
        state.step = "capture_carry"
        state.unknown_count = 0
        return actions

    def pending(self, state):
        if state.step == "capture_carry":
            state.carry_frame = state.frame
            state.step = "check_reach"
            return self.fixed(state, [command("arm_pose", pose="reach")], "保持闭爪，伸出后检查持球")
        if state.carry_frame is None:
            raise RuntimeError("Missing protected carry frame")
        verdict = self.capabilities.call("check_held", self.context(state, comparison=state.carry_frame))
        if verdict.status == "yes":
            state.phase, state.step = "holding", "search"
            state.target = None
            state.unknown_count = 0
            return self.fixed(state, [command("arm_pose", pose="carry")], "视觉确认持球，回到携带姿态")
        if verdict.status == "no":
            state.phase, state.step = "empty", "search"
            state.unknown_count = 0
            return self.fixed(state, [command("arm_pose", pose="stow")], "视觉确认未持球，收臂后继续寻球")
        state.unknown_count += 1
        if state.unknown_count >= self.profile["max_unknown"]:
            raise RuntimeError("多次无法确认持球，停止并保留现场")
        state.step = "capture_carry"
        return self.fixed(state, [command("arm_pose", pose="carry")], "持球不确定，保持闭爪并回 carry")

    def holding(self, state):
        detection = self.locate(state, "find_box")
        state.target = detection.target
        if detection.status == "unknown":
            return self.unknown(state, detection.note)
        if detection.status == "absent":
            return self.move(state, "search_right")
        state.target = detection.target
        match = self.resolve(state, "match_drop", self.context(state, target=state.target), {"reach"})
        state.assessment = match
        if match.status == "unknown":
            return self.unknown(state, match.note)
        if match.status == "no":
            if state.target.path != "clear":
                return self.unknown(state, "箱子路径不可确认")
            return self.move(state, "approach_box: " + match.note)
        authorize(state, "drop", [InfoRequest(kind="reach")])
        actions = issue(state, drop(), "投放条件满足，drop 后张爪并等待落稳", action="drop")
        state.phase, state.step = "final", "check"
        return actions

    def final(self, state):
        verdict = self.capabilities.call("check_drop", self.context(state, target=state.target))
        state.final_verdict = verdict.status
        if verdict.status != "yes":
            raise FinalCheckError(f"最终检查 {verdict.status}: {verdict.note}")
        return self.fixed(state, [command("done", success=True)], "当前图片确认球已在箱内")

    def run(self, phase, state):
        try:
            return getattr(self, phase)(state)
        except ProbeRequired as missing:
            return self.probe(state, missing.request)
        except InformationUnavailable as unavailable:
            self.log.append("information_unavailable", capability_phase=phase, message=str(unavailable))
            return self.unknown(state, str(unavailable))
