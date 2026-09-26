#!/usr/bin/env python3
"""离线重渲染: 回放 trajectory.jsonl 里的全量 qpos, 从任意相机导出视频.

修复相机参数/换视角/提分辨率后无需重跑 episode 即可重新出片:
  python rerender.py ep_0005                    # 默认 overhead
  python rerender.py ep_0005 --cams side front_cam
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")       # 必须在 import mujoco 之前

from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402

import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ep", help="episode 名, 如 ep_0005")
    ap.add_argument("--cams", nargs="+", default=["overhead"])
    ap.add_argument("--fps", type=int, default=config.CTRL_HZ)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    ep_dir = config.EPISODES_DIR / args.ep
    traj_path = ep_dir / "trajectory.jsonl"
    if not traj_path.exists():
        sys.exit(f"找不到 {traj_path}")
    traj = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    if not traj:
        sys.exit("trajectory.jsonl 为空")

    model = mujoco.MjModel.from_xml_path(str(config.SCENE_XML))
    data = mujoco.MjData(model)
    if len(traj[0]["qpos"]) != model.nq:
        sys.exit(f"qpos 维度不符: 记录 {len(traj[0]['qpos'])} vs 模型 {model.nq} "
                 "(模型改过? 换回录制时的模型)")

    renderers: dict[str, tuple[mujoco.Renderer, int]] = {}
    for cam in args.cams:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            sys.exit(f"相机不存在: {cam}")
        renderers[cam] = (mujoco.Renderer(model, height=args.height,
                                          width=args.width), cid)

    outs = {c: ep_dir / f"rerender_{c}.mp4" for c in args.cams}
    writers = {c: imageio.get_writer(str(p), fps=args.fps, codec="libx264",
                                     quality=8, macro_block_size=None)
               for c, p in outs.items()}
    for rec in traj:
        data.qpos[:] = rec["qpos"]
        mujoco.mj_forward(model, data)      # 仅渲染, 不需要步进物理
        for cam, (r, cid) in renderers.items():
            r.update_scene(data, camera=cid)
            writers[cam].append_data(r.render())
    for w in writers.values():
        w.close()
    for c, p in outs.items():
        print(f"已导出: {p} ({len(traj)} 帧 @ {args.fps}fps)")


if __name__ == "__main__":
    main()
