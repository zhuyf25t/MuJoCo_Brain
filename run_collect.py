#!/usr/bin/env python3
"""数据采集主入口 —— LLM 大脑 + 工具调用 + 记录器 闭环.

用法:
  python run_collect.py --brain scripted --episodes 5              # 脚本策略批量产数据
  python run_collect.py --brain openai --task "把蓝色方块放进箱子"  # OpenAI 兼容 LLM
  python run_collect.py --brain anthropic                          # Claude
  python run_collect.py --brain scripted --gui                     # 边看边采
  python run_collect.py --brain vista --episodes 1 --gui           # VISTA 消息/工具闭环
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import ExitStack

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" or "--gui" in sys.argv
                      else "egl")              # 必须在 import mujoco 之前

import mujoco  # noqa: E402

import config  # noqa: E402
from sim.env import MobileManipEnv, parse_task  # noqa: E402
from recorder import EpisodeRecorder, next_episode_dir  # noqa: E402
from control.tools import ToolLayer  # noqa: E402
from viz import CameraRig, GuiViewer  # noqa: E402
from brains.base import Decision  # noqa: E402


def build_brain(name: str, env: MobileManipEnv):
    if name == "vista":
        from brains.vista import VistaBrain
        return VistaBrain()
    if name == "scripted":
        from brains import ScriptedBrain
        return ScriptedBrain(env)
    if name == "openai":
        from brains import OpenAICompatBrain
        return OpenAICompatBrain()
    if name == "langgraph":
        from brains.langgraph_brain import LangGraphBrain
        return LangGraphBrain()
    if name == "anthropic":
        from brains import AnthropicBrain
        return AnthropicBrain()
    raise ValueError(f"未知 brain: {name}")


def _run_vista_rounds(env, brain, tools, recorder, task_text, tool_schemas,
                      history, outcome, gui, verbose):
    """Execute each accepted play exactly once and settle it before any next model call."""
    from brains.vista.state import action_result, batch_status

    obs = env.get_obs()
    for index in range(getattr(brain, "max_decisions", config.MAX_DECISIONS)):
        if gui is not None and not gui.is_running():
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
            break
        outcome["decision_rounds"] = index + 1
        if gui is not None:
            gui.set_status(f"VISTA | round {index + 1}\nThinking...")
        decisions = brain.decide(obs, task_text, tool_schemas, history)
        call = brain.handoff()
        if call["name"] == "finish":
            expected = [Decision(call["args"]["reason"], "done", {"success": call["args"]["success"]})]
            if decisions != expected:
                raise RuntimeError("VISTA finish 与返回的 Decision 不一致")
            brain.accept_finish_result({"tool_call_id": call["id"], "accepted": True})
            decision = decisions[0]
            recorder.log_decision(t=env.data.time, thought=decision.thought, tool="done",
                                  args=decision.args, ok=True, result="已接受模型结束请求",
                                  extra={"decision_round": index + 1, "tool_call_id": call["id"]})
            history.append({"tool": "done", "args": decision.args, "ok": True, "result": "已接受结束请求"})
            outcome.update(brain_success=call["args"]["success"], reason="brain 请求结束", stop_code="brain_done")
            break

        actions = call["args"]["actions"]
        if [{"action": decision.tool, "args": decision.args} for decision in decisions] != actions:
            raise RuntimeError("VISTA play 与返回的 Decision 批次不一致")
        batch_id = brain.next_batch_id
        brain.record_batch_start(batch_id)
        results = [action_result(i, action, "not_executed") for i, action in enumerate(actions, 1)]
        fatal = None
        cancelled = False
        for action_index, decision in enumerate(decisions):
            if gui is not None and not gui.is_running():
                cancelled = True
                break
            if gui is not None:
                gui.set_status(f"VISTA | batch {batch_id} | action {action_index + 1}/{len(actions)}\n{decision.tool}")
            try:
                ok, raw_result = tools.execute(decision.tool, decision.args)
                if type(ok) is not bool or not isinstance(raw_result, str):
                    raise RuntimeError("控制器返回了无效结果")
            except BaseException as error:
                ok, raw_result = False, f"控制执行中断: {type(error).__name__}"
                fatal = error
            # A false boolean from this controller does not prove the robot never moved.
            results[action_index] = action_result(action_index + 1, actions[action_index],
                                                   "completed" if ok else "unknown")
            if not ok and fatal is None:
                fatal = RuntimeError(f"VISTA 批次 {batch_id} 第 {action_index + 1} 个动作执行进度未知；"
                                     "剩余动作未执行，程序终止。")
            history.append({"tool": decision.tool, "args": dict(decision.args), "ok": ok, "result": raw_result})
            outcome["decisions"] = len(history)
            try:
                # Raw controller diagnostics stay in human-facing collector logs only.
                recorder.log_decision(t=env.data.time, thought=decision.thought, tool=decision.tool,
                                      args=decision.args, ok=ok, result=raw_result,
                                      extra={"decision_round": index + 1, "batch_id": batch_id,
                                             "action_index": action_index + 1, "tool_call_id": call["id"]})
            except Exception as error:
                if fatal is None:
                    fatal = RuntimeError(f"VISTA 批次 {batch_id} 的采集记录写入失败（{type(error).__name__}）")
                else:
                    fatal.add_note("采集器未能保存该子动作的日志")
            if verbose:
                print(f"  [批次{batch_id:02d}/动作{action_index+1:02d}/轮{index+1:02d}] "
                      f"{decision.tool}({decision.args}) [{'OK' if ok else 'UNKNOWN'}]", flush=True)
            if fatal is not None:
                cancelled = ok  # A recorder failure can cancel the remainder after a known completion.
                break

        receipt = {"batch_id": batch_id, "tool_call_id": call["id"],
                   "execution_status": batch_status(results, cancelled=cancelled), "action_results": results}
        obs_after = None
        try:
            obs_after = env.get_obs()
        except Exception as error:
            if fatal is None:
                fatal = RuntimeError(f"FRAME_UNAVAILABLE: VISTA 批次 {batch_id} 结束后取图失败（{type(error).__name__}）")
            else:
                fatal.add_note("批次停止后也未能取得最终车头图")
        try:
            brain.accept_batch_result(receipt, obs_after)
        except Exception as error:
            if fatal is None:
                raise
            if str(error) != str(fatal):
                fatal.add_note(str(error))
        if fatal is not None:
            raise fatal
        obs = obs_after
        # Even the last batch has already received its ToolMessage and final PNG.
        if cancelled:
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
            break
        if env.success():
            outcome.update(reason="环境判定成功", stop_code="environment_success")
            break


def _settle_episode(env, outcome, on_tick, gui=None, verbose=True):
    """Advance a fixed physical tail, without new decisions or arm/gripper commands."""
    if gui is not None and not gui.is_running():
        outcome.update(reason="用户关闭窗口", stop_code="window_closed")
        return
    seconds = float(config.EPISODE_SETTLE_SECONDS)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("EPISODE_SETTLE_SECONDS 必须是非负有限数")
    ticks = math.ceil(seconds / config.CTRL_DT)
    start = float(env.data.time)
    settling = {"requested_seconds": seconds, "start_t_sim": round(start, 4),
                "environment_success_before": bool(env.success()), "completed": False}
    outcome["settling"] = settling
    try:
        if ticks:
            if verbose:
                print(f"  [收尾] 继续仿真 {ticks * config.CTRL_DT:g} 秒并采集，再进行最终判定", flush=True)
            if gui is not None:
                gui.set_status(f"Settling physics | {ticks * config.CTRL_DT:g}s")
            env.robot.hal.command_drive(0.0, 0.0)
        for _ in range(ticks):
            if gui is not None and not gui.is_running():
                outcome.update(reason="用户关闭窗口", stop_code="window_closed")
                break
            env.robot.hal.tick(on_tick)
        else:
            settling["completed"] = True
        if gui is not None and not gui.is_running():
            outcome.update(reason="用户关闭窗口", stop_code="window_closed")
    finally:
        settling.update(end_t_sim=round(float(env.data.time), 4),
                        elapsed_seconds=round(float(env.data.time) - start, 4))


def run_episode(env, brain, tool_schemas, task_text, ep_dir, gui=None,
                save_images: bool = True, verbose: bool = True,
                seed: int | None = None) -> dict:
    # LIFO 释放: 记录文件 → 录像相机. 环境相机由调用方 env.close() 释放.
    with ExitStack() as resources:
        rig = (resources.enter_context(CameraRig(env.model, cams=config.REC_CAMS))
               if save_images else None)
        capture = rig.render if rig else None
        if hasattr(brain, "reset"):
            brain.reset()
        recorder = resources.enter_context(EpisodeRecorder(ep_dir, {
            "task": task_text, "brain": brain.name, "task_kind": "pickball",
            "seed": seed,
        }, capture=capture))

        def on_tick():
            recorder.on_tick(env.model, env.data, env.robot)
            if gui is not None:
                gui.sync()

        tools = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal, on_tick=on_tick)
        history: list[dict] = []
        outcome = {"success": False, "brain_success": None, "decisions": 0,
                   "decision_rounds": 0, "reason": "决策轮数耗尽", "stop_code": "decision_limit"}
        try:
            if brain.name == "vista":
                brain.begin_episode(ep_dir, task_text, tool_schemas)
                _run_vista_rounds(env, brain, tools, recorder, task_text, tool_schemas,
                                  history, outcome, gui, verbose)
            rounds = () if brain.name == "vista" else range(getattr(brain, "max_decisions", config.MAX_DECISIONS))
            for i in rounds:
                if gui is not None and not gui.is_running():
                    outcome["reason"] = "用户关闭窗口"
                    outcome["stop_code"] = "window_closed"
                    break
                obs = env.get_obs()  # 车头相机; overhead 仅用于录制
                obs["holding"] = env.robot.hal.attached_object() is not None
                outcome["decision_rounds"] = i + 1
                try:
                    batch = brain.decide(obs, task_text, tool_schemas, history)
                except Exception as e:          # API 异常仍保存本集
                    outcome["reason"] = f"brain 决策异常: {type(e).__name__}: {e}"
                    outcome["stop_code"] = "brain_error"
                    outcome["error"] = f"{type(e).__name__}: {e}"
                    print(f"  [!] {outcome['reason']}")
                    break
                if isinstance(batch, Decision):
                    batch = [batch]
                if not batch:
                    outcome["reason"] = "brain 未返回工具调用"
                    outcome["stop_code"] = "empty_decision"
                    break
                done_called = False
                for decision in batch:
                    img_before = recorder.snapshot_paths(env.data, "pre") if capture else None
                    ok, result = tools.execute(decision.tool, decision.args)
                    img_after = recorder.snapshot_paths(env.data, "post") if capture else None
                    recorder.log_decision(
                        t=env.data.time, thought=decision.thought, tool=decision.tool,
                        args=decision.args, ok=ok, result=result,
                        img_before=img_before, img_after=img_after,
                        extra={"decision_round": i + 1})
                    history.append({"tool": decision.tool, "args": decision.args,
                                    "ok": ok, "result": result})
                    if verbose:
                        print(f"  [动作{len(history):02d}/轮{i+1:02d}] brain: "
                              f"{decision.tool}({decision.args})")
                        print(f"      tool[{'OK' if ok else 'FAIL'}]: "
                              f"{result.replace(chr(10), ' ')}")
                    if decision.tool == "done":
                        done_called = True
                        outcome["brain_success"] = decision.args.get("success")
                        outcome["reason"] = ("brain 请求结束 "
                                             f"(自评 success={outcome['brain_success']})")
                        outcome["stop_code"] = "brain_done"
                        break
                if done_called:
                    break
                if env.success():
                    outcome["reason"] = "环境判定成功"
                    outcome["stop_code"] = "environment_success"
                    break
            if outcome["stop_code"] in (
                    "brain_done", "decision_limit", "environment_success"):
                _settle_episode(env, outcome, on_tick, gui, verbose)
            outcome["success"] = env.success()
            if "settling" in outcome:
                outcome["settling"]["environment_success_after"] = bool(outcome["success"])
                if outcome["stop_code"] == "environment_success" and not outcome["success"]:
                    outcome["reason"] = "曾检测到成功，收尾后环境判定未通过"
            outcome["decisions"] = len(history)
            if brain.name == "vista":
                outcome["environment_success"] = outcome["success"]
                outcome["vista_episode"] = str(brain.storage.episode_dir)
                brain.end_episode(outcome)
            recorder.finish(outcome["success"], {
                "reason": outcome["reason"], "brain_success": outcome["brain_success"],
                "stop_code": outcome["stop_code"], "error": outcome.get("error"),
                **({"environment_success": outcome["environment_success"]}
                   if "environment_success" in outcome else {}),
                **({"settling": outcome["settling"]} if "settling" in outcome else {}),
                "decision_rounds": outcome["decision_rounds"],
                "t_sim": round(float(env.data.time), 2),
            })
        except BaseException as e:
            if brain.name == "vista":
                # Fatal errors must leave all episode loops, and diagnostics must not mask them.
                brain.abort_episode(e, outcome)
                try:
                    recorder.finish(False, {"reason": f"采集中断: {type(e).__name__}", "error": str(e),
                                            **({"settling": outcome["settling"]} if "settling" in outcome else {}),
                                            "t_sim": round(float(env.data.time), 2)})
                except Exception as secondary:
                    e.add_note(f"采集中断记录也未能保存：{type(secondary).__name__}")
            else:
                recorder.finish(False, {"reason": f"采集中断: {type(e).__name__}",
                                        **({"settling": outcome["settling"]} if "settling" in outcome else {}),
                                        "t_sim": round(float(env.data.time), 2)})
            raise
        return outcome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MuJoCo_Brain 数据采集")
    ap.add_argument("--brain", default="scripted",
                    choices=["scripted", "openai", "anthropic", "langgraph", "vista"])
    ap.add_argument("--brain-memory", help="VISTA 的经验、图片与调用记录根目录")
    ap.add_argument("--max-decisions", type=int, help="本轮最大决策次数")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task", default=None, help="任务文本, 如 '把红色方块放进箱子'")
    ap.add_argument("--gui", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--keep-open", action="store_true", help="GUI 采集结束后保持最后画面，关闭窗口退出")
    ap.add_argument("--no-images", action="store_true", help="关闭通用录像图片；VISTA 必需的车头 PNG 仍保存")
    ap.add_argument("--out", default=str(config.EPISODES_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes 必须至少为 1")
    if args.keep_open and not args.gui:
        ap.error("--keep-open 需要配合 --gui")
    if args.brain_memory and args.brain != "vista":
        ap.error("--brain-memory 只用于 vista")
    if args.max_decisions is not None and args.max_decisions < 1:
        ap.error("--max-decisions 必须至少为 1")

    config.ensure_dirs()
    task = parse_task(args.task) if args.task else parse_task("把0号网球捡起来放进收纳箱")
    task_text = args.task or task.text()

    results = []
    with ExitStack() as resources:
        env = resources.enter_context(MobileManipEnv())
        if args.brain == "vista" and args.brain_memory:
            from brains.vista import VistaBrain
            brain = VistaBrain(memory_dir=args.brain_memory)
        else:
            brain = build_brain(args.brain, env)
        if hasattr(brain, "close"):
            resources.callback(brain.close)
        if args.max_decisions is not None:
            brain.max_decisions = args.max_decisions
        schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()
        print(f"brain={brain.name} task='{task_text}' episodes={args.episodes} out={args.out}")
        gui = None
        if args.gui:
            print("正在创建 GUI 窗口；若长期停在这里，请检查 GLFW/WSLg 图形环境。", flush=True)
            gui = resources.enter_context(GuiViewer(env.model, env.data))
            print("GUI 已打开，开始采集。", flush=True)
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep, task=task)
            if gui:
                gui.sync()
            ep_dir = next_episode_dir(args.out)
            print(f"\n== episode {ep+1}/{args.episodes} -> {ep_dir.name}")
            r = run_episode(env, brain, schemas, task_text, ep_dir, gui=gui,
                            save_images=not args.no_images, verbose=not args.quiet,
                            seed=args.seed + ep)
            results.append(r)
            print(f"   环境判定: {'PASS' if r['success'] else 'FAIL'} "
                  f"({r['reason']}, 决策 {r['decision_rounds']} 轮 / 工具调用 {r['decisions']} 次)")
            if gui is not None and not gui.is_running():
                break
        n_ok = sum(1 for r in results if r["success"])
        print(f"\n===== 汇总: {n_ok}/{len(results)} 成功 =====")
        if args.keep_open and gui.is_running():
            print("采集已结束，窗口保留最后画面；关闭窗口退出。", flush=True)
            while gui.is_running():
                gui.sync()
                time.sleep(0.02)
    return 0 if n_ok == args.episodes else 1


def cli(argv=None) -> int:
    try:
        return main(argv)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        for note in getattr(error, "__notes__", []):
            print(f"  {note}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
