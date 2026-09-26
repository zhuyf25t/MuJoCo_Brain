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

os.environ.setdefault("MUJOCO_GL", "egl")       # 必须在 import mujoco 之前

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
                save_images: bool = True, verbose: bool = True) -> dict:
    rig = CameraRig(env.model, cams=config.REC_CAMS) if save_images else None
    capture = (lambda data: rig.render(data)) if rig else None

    def on_tick():
        recorder.on_tick(env.model, env.data, env.robot)
        if gui is not None:
            gui.sync()

    tools = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal, on_tick=on_tick)
    if hasattr(brain, "reset"):
        brain.reset()
    recorder = EpisodeRecorder(ep_dir, {
        "task": task_text,
        "brain": brain.name,
        "task_kind": "pickball",
    }, capture=capture)

    history: list[dict] = []
    outcome = {"success": False, "decisions": 0, "reason": "决策数耗尽"}
    done_called = False
    for i in range(config.MAX_DECISIONS):
        if gui is not None and not gui.is_running():
            outcome["reason"] = "用户关闭窗口"
            break
        obs = env.get_obs()          # 大脑只看车头相机 (overhead 仅录制用)
        obs["holding"] = env.robot.hal.attached_object() is not None
        try:
            batch = brain.decide(obs, task_text, tool_schemas, history)
        except Exception as e:                     # noqa: BLE001  API 异常不炸采集
            print(f"  [!] brain 决策异常: {type(e).__name__}: {e}")
            break
        if isinstance(batch, Decision):
            batch = [batch]
        # 批量执行本轮回的所有工具调用
        for decision in batch:
            img_before = (recorder.snapshot_paths(env.data, "pre")
                          if capture else None)
            ok, result = tools.execute(decision.tool, decision.args)
            img_after = (recorder.snapshot_paths(env.data, "post")
                         if capture else None)
            recorder.log_decision(t=env.data.time, thought=decision.thought,
                                  tool=decision.tool, args=decision.args, ok=ok,
                                  result=result, img_before=img_before,
                                  img_after=img_after)
            history.append({"tool": decision.tool, "args": decision.args,
                            "ok": ok, "result": result})
            if verbose:
                print(f"  [{i+1:02d}] {decision.tool}({decision.args}) -> "
                      f"{'OK' if ok else 'FAIL'}: {result[:80].replace(chr(10), ' ')}")
            if decision.tool == "done":
                break
        if decision.tool == "done":
            done_called = True
            outcome["reason"] = f"brain 宣告完成 (success={decision.args.get('success')})"
            break
        if env.success():
            outcome["reason"] = "环境判定成功"
            break
    else:
        pass

    outcome["success"] = env.success()
    outcome["decisions"] = len(history)
    recorder.finish(outcome["success"], {
        "reason": outcome["reason"],
        "t_sim": round(float(env.data.time), 2),
    })
    if rig:
        rig.close()
    return outcome


def main() -> None:
    ap = argparse.ArgumentParser(description="openclaw_sim 数据采集")
    ap.add_argument("--brain", default="scripted",
                    choices=["scripted", "openai", "anthropic"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task", default=None, help="任务文本, 如 '把红色方块放进箱子'")
    ap.add_argument("--gui", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--no-images", action="store_true", help="不存图(更快)")
    ap.add_argument("--out", default=str(config.EPISODES_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    task = parse_task(args.task) if args.task else parse_task("把0号网球捡起来放进收纳箱")
    task_text = args.task or task.text()

    env = MobileManipEnv()
    brain = build_brain(args.brain, env)
    schemas = ToolLayer(env.robot.base, env.robot.arm, env.robot.hal).schemas_anthropic()

    print(f"brain={brain.name} task='{task_text}' episodes={args.episodes} "
          f"out={args.out}")
    gui = None
    if args.gui:
        gui = GuiViewer(env.model, env.data)

    results = []
    try:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep, task=task)
            if gui:
                gui.sync()
            ep_dir = next_episode_dir(args.out)
            print(f"\n== episode {ep+1}/{args.episodes} -> {ep_dir.name}")
            r = run_episode(env, brain, schemas, task_text, ep_dir, gui=gui,
                            save_images=not args.no_images, verbose=not args.quiet)
            results.append(r)
            print(f"   结果: {'PASS' if r['success'] else 'FAIL'} "
                  f"({r['reason']}, 决策 {r['decisions']} 次)")
    finally:
        if gui:
            gui.close()

    n_ok = sum(1 for r in results if r["success"])
    print(f"\n===== 汇总: {n_ok}/{len(results)} 成功 =====")
    code = 0 if n_ok == len(results) and results else 1
    if args.gui:
        # EGL 渲染器 + glfw viewer 在解释器拆卸阶段可能因 GL 上下文销毁顺序段错误,
        # 任务与数据均已完整落盘, 这里跳过析构直接退出以保住退出码
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
