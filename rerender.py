#!/usr/bin/env python3
"""离线重渲染: 回放 trajectory.jsonl 里的全量 qpos, 从任意相机导出视频.

修复相机参数/换视角/提分辨率后无需重跑 episode 即可重新出片:
  python rerender.py ep_0005                    # 默认 overhead
  python rerender.py ep_0005 --cams side front_cam
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "egl")

from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402

import config  # noqa: E402
from playback import apply_frame, load_trajectory, trajectory_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ep", help="episode 名, 如 ep_0005")
    ap.add_argument("--root", type=Path, default=config.EPISODES_DIR)
    ap.add_argument("--model", type=Path, default=config.SCENE_XML, help="录制时的场景 XML")
    ap.add_argument("--cams", nargs="+", default=["overhead"])
    ap.add_argument("--fps", type=int, default=config.CTRL_HZ)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        ap.error("fps、width 和 height 必须大于 0")
    traj_path = trajectory_path(args.ep, args.root)
    try:
        model = mujoco.MjModel.from_xml_path(str(args.model))
        traj = load_trajectory(traj_path, model)
    except (OSError, ValueError) as e:
        ap.error(str(e))
    data = mujoco.MjData(model)
    outs = {c: traj_path.parent / f"rerender_{c}.mp4" for c in args.cams}
    with ExitStack() as resources:
        renderers = {}
        for cam in outs:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cid < 0:
                ap.error(f"相机不存在: {cam}")
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            resources.callback(renderer.close)
            renderers[cam] = (renderer, cid)
        writers = {}
        for cam, path in outs.items():
            writer = imageio.get_writer(str(path), fps=args.fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            resources.callback(writer.close)
            writers[cam] = writer
        for frame in traj:
            apply_frame(model, data, frame)
            for cam, (renderer, cid) in renderers.items():
                renderer.update_scene(data, camera=cid)
                writers[cam].append_data(renderer.render())
    for c, p in outs.items():
        print(f"已导出: {p} ({len(traj)} 帧 @ {args.fps}fps)")


if __name__ == "__main__":
    main()
