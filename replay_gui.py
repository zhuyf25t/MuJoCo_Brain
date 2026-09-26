#!/usr/bin/env python3
"""在 MuJoCo GUI 中回放 trajectory.jsonl，支持自由视角、暂停与重播。

  python replay_gui.py ep_0007
  python replay_gui.py data/episodes/ep_0007/trajectory.jsonl --speed 0.5
  python replay_gui.py ep_0007 --check
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402

import config  # noqa: E402
from playback import PlaybackCursor, apply_frame, load_trajectory, trajectory_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode", help="episode 名、目录或 trajectory.jsonl 路径")
    ap.add_argument("--root", type=Path, default=config.EPISODES_DIR)
    ap.add_argument("--model", type=Path, default=config.SCENE_XML, help="录制时的场景 XML")
    ap.add_argument("--speed", type=float, default=1.0, help="播放倍速，例如 0.5 或 2")
    ap.add_argument("--loop", action="store_true", help="循环播放")
    ap.add_argument("--exit-on-end", action="store_true", help="播放完自动关闭；默认保留最后画面")
    ap.add_argument("--check", action="store_true", help="只校验轨迹与模型，不打开窗口")
    args = ap.parse_args(argv)
    if args.loop and args.exit_on_end:
        ap.error("--loop 与 --exit-on-end 不能同时使用")
    try:
        path = trajectory_path(args.episode, args.root)
        model = mujoco.MjModel.from_xml_path(str(args.model))
        frames = load_trajectory(path, model)
        cursor = PlaybackCursor(frames, args.speed, args.loop)
    except (OSError, ValueError) as e:
        ap.error(str(e))
    print(f"轨迹: {path}\n{len(frames)} 帧，时长 {frames[-1].t - frames[0].t:.2f}s，"
          f"倍速 {args.speed:g}，模型 nq={model.nq}", flush=True)
    if args.check:
        print("校验通过。位置维度一致；请仍确认使用的是录制时的同一模型。")
        return 0

    from mujoco import viewer as mj_viewer

    data = mujoco.MjData(model)
    apply_frame(model, data, frames[0])
    keys = SimpleQueue()
    print("正在创建 GUI 窗口；若长期停在这里，请检查 GLFW/WSLg 图形环境。", flush=True)
    with mj_viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        print("GUI 已打开。空格: 暂停/继续；R: 从头重播；鼠标可调整视角；关闭窗口退出。", flush=True)
        with viewer.lock():
            viewer.cam.lookat[:] = [0.55, 0.0, 0.55]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -35
        previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            was_finished = cursor.finished
            cursor.advance(now - previous)
            previous = now
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if key == 32:
                    cursor.toggle_pause()
                elif key in (ord("R"), ord("r")):
                    cursor.restart()
            with viewer.lock():
                apply_frame(model, data, frames[cursor.index])
            viewer.sync()
            if cursor.finished:
                if args.exit_on_end:
                    break
                if not was_finished:
                    print("播放结束，保留最后画面；按 R 重播，或关闭窗口退出。", flush=True)
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    sys.exit(main())
