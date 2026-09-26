#!/usr/bin/env python3
"""数据集检查与回放.

用法:
  python inspect_data.py                    # 全部 episode 统计
  python inspect_data.py --replay ep_0000   # 导出该 episode 俯视相机回放视频
  python inspect_data.py --decisions ep_0000   # 打印决策日志
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config


def load_ep(ep_dir: Path) -> dict:
    meta = json.loads((ep_dir / "meta.json").read_text())
    decisions = [json.loads(l) for l in
                 (ep_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    traj_path = ep_dir / "trajectory.jsonl"
    traj = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
    return {"meta": meta, "decisions": decisions, "traj": traj}


def stats(root: Path) -> None:
    eps = sorted(p for p in root.glob("ep_*") if p.is_dir())
    if not eps:
        print(f"{root} 下没有 episode")
        return
    n_ok = n_dec = n_frames = 0
    total_sim_t = 0.0
    tools: dict[str, int] = {}
    print(f"{'episode':<10}{'brain':<10}{'成功':<5}{'决策':<5}{'帧数':<7}{'时长s':<8}原因")
    for p in eps:
        d = load_ep(p)
        m = d["meta"]
        ok = m.get("success")
        n_ok += bool(ok)
        n_dec += len(d["decisions"])
        n_frames += len(d["traj"])
        total_sim_t += d["traj"][-1]["t"] if d["traj"] else 0
        for r in d["decisions"]:
            tools[r["tool"]] = tools.get(r["tool"], 0) + 1
        print(f"{p.name:<10}{m.get('brain','?'):<10}{'✓' if ok else '✗':<5}"
              f"{len(d['decisions']):<5}{len(d['traj']):<7}"
              f"{(d['traj'][-1]['t'] if d['traj'] else 0):<8.1f}"
              f"{m.get('reason', '')[:40]}")
    print(f"\n汇总: {n_ok}/{len(eps)} 成功 | 平均决策 {n_dec/len(eps):.1f} 次/集 | "
          f"平均时长 {total_sim_t/len(eps):.1f}s/集 | 总帧数 {n_frames}")
    print(f"工具调用分布: {dict(sorted(tools.items(), key=lambda x: -x[1]))}")


def replay(ep_dir: Path, out: str | None = None, cam: str = "overhead") -> None:
    import imageio.v2 as imageio
    d = load_ep(ep_dir)
    frames = []
    for rec in d["traj"]:
        img = rec.get("imgs", {}).get(cam)
        if img:
            frames.append(imageio.imread(ep_dir / img))
    if not frames:
        print(f"该 episode 没有 {cam} 相机图像 (--no-images 采集?)")
        return
    out = out or str(ep_dir / f"replay_{cam}.mp4")
    w = imageio.get_writer(out, fps=config.CTRL_HZ, codec="libx264", quality=8,
                           macro_block_size=None)
    for f in frames:
        w.append_data(f)
    w.close()
    print(f"回放视频: {out} ({len(frames)} 帧 @ {config.CTRL_HZ}fps)")


def show_decisions(ep_dir: Path) -> None:
    d = load_ep(ep_dir)
    for r in d["decisions"]:
        flag = "OK  " if r["ok"] else "FAIL"
        thought = (r.get("thought") or "").replace("\n", " ")[:60]
        print(f"[{r['i']:02d}] t={r['t']:6.1f}s {flag} {r['tool']}({json.dumps(r['args'], ensure_ascii=False)})")
        print(f"      思考: {thought}")
        print(f"      结果: {r['result'][:120].replace(chr(10), ' ')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(config.EPISODES_DIR))
    ap.add_argument("--replay", metavar="EP", help="导出回放视频, 如 ep_0000")
    ap.add_argument("--decisions", metavar="EP", help="打印决策日志")
    ap.add_argument("--cam", default="overhead")
    args = ap.parse_args()
    root = Path(args.root)

    if args.decisions:
        show_decisions(root / args.decisions)
    elif args.replay:
        replay(root / args.replay, cam=args.cam)
    else:
        stats(root)


if __name__ == "__main__":
    sys.exit(main())
