#!/usr/bin/env python3
"""数据采集主入口 —— LLM 大脑 + 工具调用 + 记录器 闭环.

用法:
  python run_collect.py --brain scripted --episodes 5              # 脚本策略批量产数据
  python run_collect.py --brain openai --task "把蓝色方块放进箱子"  # OpenAI 兼容 LLM
  python run_collect.py --brain anthropic                          # Claude
  python run_collect.py --brain scripted --gui                     # 边看边采
  python run_collect.py --brain test --episodes 1 --gui --keep-open # 四种臂预设演示
  python run_collect.py --brain test --gui --loop                  # 连续演示直到关闭窗口
  python run_collect.py --brain langgraph --episodes 1 --gui       # 模块化视觉流程
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap
from contextlib import ExitStack
from itertools import count

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" or "--gui" in sys.argv
                      else "egl")              # 必须在 import mujoco 之前

import mujoco  # noqa: E402

import config  # noqa: E402
from sim.env import MobileManipEnv, parse_task  # noqa: E402
from recorder import EpisodeRecorder, next_episode_dir  # noqa: E402
from control.tools import ToolLayer  # noqa: E402
from viz import CameraRig, GuiViewer  # noqa: E402
from brains.base import Decision  # noqa: E402


def langgraph_status(state, outcome):
    lines = [f"LangGraph finished | {state.phase}",
             f"Visual: {state.final_verdict or 'incomplete'} | "
             f"Environment: {'PASS' if outcome.get('environment_success') else 'FAIL'}",
             f"Stop: {outcome['stop_code']}"]
    if outcome.get("error"):
        error = outcome["error"]
        if error.isascii():
            lines.extend(textwrap.wrap(error, width=62))
        else:
            # MuJoCo's built-in font cannot render Chinese; keep full text in the report.
            error_type = error.split(":", 1)[0]
            lines.append(error_type if error_type.isascii() else "Decision stopped")
            if "重复申请" in error or "资料申请" in error:
                lines.append("Reference supplied; model could not decide")
            elif "无法" in error:
                lines.append("Visual decision remained uncertain")
        lines.append("Details: episode/review/index.html")
    return "\n".join(lines)


def build_brain(name: str, env: MobileManipEnv):
    if name == "langgraph":
        from brains.langgraph import LangGraphBrain
        return LangGraphBrain()
    if name == "test":
        from brains.arm_pose_test import ArmPoseTestBrain
        return ArmPoseTestBrain()
    if name == "scripted":
        from brains import ScriptedBrain
        return ScriptedBrain(env)
    if name == "openai":
        from brains import OpenAICompatBrain
        return OpenAICompatBrain()
    if name == "anthropic":
        from brains import AnthropicBrain
        return AnthropicBrain()
    raise ValueError(f"未知 brain: {name}")


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
        demo_only = getattr(brain, "demo_only", False)
        gui_hold_s = getattr(brain, "gui_hold_s", 0.0) if gui is not None else 0.0
        gui_realtime = gui_hold_s or getattr(brain, "gui_realtime", False)
        recorder = resources.enter_context(EpisodeRecorder(ep_dir, {
            "task": task_text, "brain": brain.name,
            "task_kind": "arm_pose_demo" if demo_only else "pickball",
            "seed": seed,
            **({"brain_run": str(brain.log.path.resolve())}
               if brain.name == "langgraph" and hasattr(brain, "log") else {}),
        }, capture=capture))

        last_gui_tick = time.monotonic()

        def on_tick():
            nonlocal last_gui_tick
            recorder.on_tick(env.model, env.data, env.robot)
            if gui is not None:
                if gui_realtime:
                    time.sleep(max(0.0, config.CTRL_DT - (time.monotonic() - last_gui_tick)))
                gui.sync()
                last_gui_tick = time.monotonic()

        tools = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal, on_tick=on_tick)
        history: list[dict] = []
        outcome = {"success": False, "brain_success": None, "decisions": 0,
                   "decision_rounds": 0, "reason": "决策轮数耗尽", "stop_code": "decision_limit"}
        try:
            rounds = (count() if demo_only and gui is not None and getattr(brain, "loop", False)
                      else range(getattr(brain, "max_decisions", config.MAX_DECISIONS)))
            for i in rounds:
                if gui is not None and not gui.is_running():
                    outcome["reason"] = "用户关闭窗口"
                    outcome["stop_code"] = "window_closed"
                    break
                obs = env.get_obs()  # 车头相机; overhead 仅用于录制
                obs["holding"] = env.robot.hal.attached_object() is not None
                outcome["decision_rounds"] = i + 1
                if gui is not None and brain.name == "langgraph":
                    gui.set_status(f"LangGraph | round {i+1} | {brain.state.phase}\nThinking...")
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
                tool_failed = False
                for decision in batch:
                    if gui is not None and not gui.is_running():
                        break
                    if gui_hold_s and decision.tool == "arm_pose":
                        gui.set_status(f'arm_pose("{decision.args["pose"]}")\nMoving...')
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
                    if gui_hold_s and decision.tool == "arm_pose":
                        status = "OK" if ok else "FAIL - see console"
                        gui.set_status(f'arm_pose("{decision.args["pose"]}")\n'
                                       f'{status} | hold {gui_hold_s:g}s')
                        for _ in range(round(gui_hold_s / config.CTRL_DT)):
                            if not gui.is_running():
                                break
                            env.robot.hal.tick(on_tick)
                    if decision.tool == "done":
                        done_called = True
                        outcome["brain_success"] = decision.args.get("success")
                        outcome["reason"] = ("brain 请求结束 "
                                             f"(自评 success={outcome['brain_success']})")
                        outcome["stop_code"] = "brain_done"
                        break
                    if not ok and getattr(brain, "stop_on_tool_error", False):
                        outcome["reason"] = f"工具执行失败: {decision.tool}"
                        outcome["stop_code"] = "tool_error"
                        outcome["error"] = decision.tool
                        tool_failed = True
                        break
                if done_called or tool_failed:
                    break
                if (not demo_only and not getattr(brain, "requires_final_check", False)
                        and env.success()):
                    outcome["reason"] = "环境判定成功"
                    outcome["stop_code"] = "environment_success"
                    break
            outcome["success"] = (outcome["brain_success"] is True and
                                  all(entry["ok"] for entry in history)) if demo_only else env.success()
            if getattr(brain, "requires_final_check", False):
                outcome["environment_success"] = outcome["success"]
                outcome["success"] = outcome["success"] and outcome["brain_success"] is True
            outcome["decisions"] = len(history)
            if hasattr(brain, "record_outcome"):
                brain.record_outcome(outcome)
            recorder.finish(outcome["success"], {
                "reason": outcome["reason"], "brain_success": outcome["brain_success"],
                "stop_code": outcome["stop_code"], "error": outcome.get("error"),
                **({"environment_success": outcome["environment_success"]}
                   if "environment_success" in outcome else {}),
                "decision_rounds": outcome["decision_rounds"],
                "t_sim": round(float(env.data.time), 2),
            })
            if brain.name == "langgraph" and hasattr(brain, "log"):
                try:
                    from brains.langgraph.report import build_report
                    page = build_report(ep_dir)
                    outcome["review_page"] = str(page)
                    print(f"逐轮运行网页: {page}", flush=True)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    # Report generation must not replace the original task outcome.
                    print(f"  [!] 逐轮网页生成失败: {type(error).__name__}: {error}", flush=True)
        except BaseException as e:
            recorder.finish(False, {"reason": f"采集中断: {type(e).__name__}",
                                    "t_sim": round(float(env.data.time), 2)})
            raise
        return outcome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MuJoCo_Brain 数据采集")
    ap.add_argument("--brain", default="scripted",
                    choices=["scripted", "openai", "anthropic", "test", "langgraph"])
    ap.add_argument("--brain-profile", help="LangGraph 各模块的 YAML 配置文件")
    ap.add_argument("--brain-memory", help="LangGraph 操作经验和调用记录目录")
    ap.add_argument("--max-decisions", type=int, help="本轮最大决策次数，LangGraph 默认 64")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task", default=None, help="任务文本, 如 '把红色方块放进箱子'")
    ap.add_argument("--gui", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--keep-open", action="store_true", help="GUI 采集结束后保持最后画面，关闭窗口退出")
    ap.add_argument("--loop", action="store_true", help="仅用于 --brain test --gui：持续循环姿态，关闭窗口停止")
    ap.add_argument("--no-images", action="store_true", help="不存图(更快)")
    ap.add_argument("--out", default=str(config.EPISODES_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes 必须至少为 1")
    if args.keep_open and not args.gui:
        ap.error("--keep-open 需要配合 --gui")
    if args.loop and (args.brain != "test" or not args.gui):
        ap.error("--loop 需要配合 --brain test --gui")
    if (args.brain_profile or args.brain_memory) and args.brain != "langgraph":
        ap.error("--brain-profile / --brain-memory 只用于 langgraph")
    if args.max_decisions is not None and args.max_decisions < 1:
        ap.error("--max-decisions 必须至少为 1")

    config.ensure_dirs()
    task = parse_task(args.task) if args.task else parse_task("把0号网球捡起来放进收纳箱")
    task_text = args.task or ("依次演示 stow、reach、carry、drop，最后返回 stow"
                              if args.brain == "test" else task.text())

    results = []
    with ExitStack() as resources:
        env = resources.enter_context(MobileManipEnv())
        if args.brain == "langgraph" and (args.brain_profile or args.brain_memory):
            from brains.langgraph import LangGraphBrain
            brain = LangGraphBrain(profile_path=args.brain_profile, memory_dir=args.brain_memory)
        else:
            brain = build_brain(args.brain, env)
        if hasattr(brain, "close"):
            resources.callback(brain.close)
        if args.max_decisions is not None:
            brain.max_decisions = args.max_decisions
        if args.brain == "test":
            brain.loop = args.loop
        schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()
        print(f"brain={brain.name} task='{task_text}' episodes={args.episodes} out={args.out}")
        gui = None
        if args.gui:
            print("正在创建 GUI 窗口；若长期停在这里，请检查 GLFW/WSLg 图形环境。", flush=True)
            gui = resources.enter_context(GuiViewer(env.model, env.data, focus_robot=args.brain == "test"))
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
            if args.loop and r["reason"] == "用户关闭窗口":
                print("连续姿态演示已停止。", flush=True)
                return 0
            result_label = ("姿态演示" if args.brain == "test" else
                            "视觉和环境共同判定" if args.brain == "langgraph" else "环境判定")
            print(f"   {result_label}: {'PASS' if r['success'] else 'FAIL'} "
                  f"({r['reason']}, 决策 {r['decision_rounds']} 轮 / 工具调用 {r['decisions']} 次)")
            if gui is not None and not gui.is_running():
                break
        n_ok = sum(1 for r in results if r["success"])
        print(f"\n===== 汇总: {n_ok}/{len(results)} 成功 =====")
        if args.keep_open and gui.is_running():
            if args.brain == "test":
                gui.set_status("Arm pose demo finished\nFinal pose: stow | Close window to exit")
            elif args.brain == "langgraph":
                gui.set_status(langgraph_status(brain.state, results[-1]))
            print("采集已结束，窗口保留最后画面；关闭窗口退出。", flush=True)
            while gui.is_running():
                gui.sync()
                time.sleep(0.02)
    return 0 if n_ok == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
