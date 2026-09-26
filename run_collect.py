#!/usr/bin/env python3
"""数据采集主入口 —— LLM 大脑 + 工具调用 + 记录器 闭环.

用法:
  python run_collect.py --brain scripted --episodes 5              # 脚本策略批量产数据
  python run_collect.py --brain openai --task "把蓝色方块放进箱子"  # OpenAI 兼容 LLM
  python run_collect.py --brain anthropic                          # Claude
  python run_collect.py --brain scripted --gui                     # 边看边采
"""

from __future__ import annotations

import argparse
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
                   "decision_rounds": 0, "reason": "决策轮数耗尽"}
        try:
            for i in range(config.MAX_DECISIONS):
                if gui is not None and not gui.is_running():
                    outcome["reason"] = "用户关闭窗口"
                    break
                obs = env.get_obs()  # 车头相机; overhead 仅用于录制
                obs["holding"] = env.robot.hal.attached_object() is not None
                outcome["decision_rounds"] = i + 1
                try:
                    batch = brain.decide(obs, task_text, tool_schemas, history)
                except Exception as e:          # API 异常仍保存本集
                    outcome["reason"] = f"brain 决策异常: {type(e).__name__}: {e}"
                    print(f"  [!] {outcome['reason']}")
                    break
                if isinstance(batch, Decision):
                    batch = [batch]
                if not batch:
                    outcome["reason"] = "brain 未返回工具调用"
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
                        break
                if done_called:
                    break
                if env.success():
                    outcome["reason"] = "环境判定成功"
                    break
            outcome["success"] = env.success()
            outcome["decisions"] = len(history)
            recorder.finish(outcome["success"], {
                "reason": outcome["reason"], "brain_success": outcome["brain_success"],
                "decision_rounds": outcome["decision_rounds"],
                "t_sim": round(float(env.data.time), 2),
            })
        except BaseException as e:
            recorder.finish(False, {"reason": f"采集中断: {type(e).__name__}",
                                    "t_sim": round(float(env.data.time), 2)})
            raise
        return outcome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MuJoCo_Brain 数据采集")
    ap.add_argument("--brain", default="scripted",
                    choices=["scripted", "openai", "anthropic"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task", default=None, help="任务文本, 如 '把红色方块放进箱子'")
    ap.add_argument("--gui", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--keep-open", action="store_true", help="GUI 采集结束后保持最后画面，关闭窗口退出")
    ap.add_argument("--no-images", action="store_true", help="不存图(更快)")
    ap.add_argument("--out", default=str(config.EPISODES_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes 必须至少为 1")
    if args.keep_open and not args.gui:
        ap.error("--keep-open 需要配合 --gui")

    config.ensure_dirs()
    task = parse_task(args.task) if args.task else parse_task("把0号网球捡起来放进收纳箱")
    task_text = args.task or task.text()

    results = []
    with ExitStack() as resources:
        env = resources.enter_context(MobileManipEnv())
        brain = build_brain(args.brain, env)
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


if __name__ == "__main__":
    sys.exit(main())
