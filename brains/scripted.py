"""确定性脚本策略 —— 纯视觉闭环版.

只用与 LLM 完全相同的感知(车头相机 + 本体感觉)和动作原语,
不读任何仿真真值: 找球(转头扫描) → 对准 → 前进 → 伸手 → 闭合 →
抬起 → 找箱 → 对准 → 投放位 → 张开. 状态机驱动.
"""

from __future__ import annotations

import numpy as np

import config
from control import perception
from sim.env import MobileManipEnv
from .base import Brain, Decision

TURN_RATE_DEG = config.PRIM_W * 57.3          # 估计角速度 (度/秒), 用于换算时长


def _turn_seconds(bearing_deg: float) -> float:
    """把方位角修正换算成原语时长 (增益1.0, 限幅防过冲震荡)."""
    return float(np.clip(abs(bearing_deg) / TURN_RATE_DEG, 0.2, 0.9))


class ScriptedBrain(Brain):
    name = "scripted"

    def __init__(self, env: MobileManipEnv, seed: int | None = None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self._phase = "find_ball"     # find_ball/align_ball/approach/grasp_seq
        #                               /find_bin/align_bin/approach_bin/drop_seq/done
        self._seq = 0                 # 阶段内子步骤
        self._search = 0              # 搜索计数

    def reset(self) -> None:
        self._phase = "find_ball"
        self._seq = 0
        self._search = 0

    # ---------- 感知 ----------
    def _see(self):
        return perception.sense(self.env.robot.hal)

    def _pick(self, dets, kind):
        """目标承诺: 优先选与上次目标方位连续的检测, 防止多目标间跳变."""
        cands = [d for d in dets if d.kind == kind and d.dist_m is not None]
        if not cands:
            return None
        key = "_last_ball_br" if kind == "ball" else "_last_bin_br"
        last = getattr(self, key, None)
        if last is not None:
            near = [d for d in cands if abs(d.bearing_deg - last) < 25]
            if near:
                return min(near, key=lambda d: abs(d.bearing_deg - last))
        d = min(cands, key=lambda d: abs(d.bearing_deg))
        setattr(self, key, d.bearing_deg)
        return d

    # ---------- 主决策 ----------
    def decide(self, obs, task_text, tool_schemas, history) -> Decision:
        self._n = getattr(self, "_n", 0) + 1
        s = self._see()
        holding = s["holding"]

        # ============ 持球: 去箱子 ============
        if holding:
            return self._go_bin(s)

        # ============ 未持球: 去球 ============
        if self._phase == "drop_seq":          # 刚投放完没持有 → 成功收尾
            return Decision("已投放, 任务完成", "done", {"success": True})
        # 顺手记忆箱位(视觉+里程计估计). 只接受远距(>0.9m)目击:
        # 近距 blob 失真大, 写入会把记忆污染成"箱在身边"
        bin_det = self._pick(s["detections"], "bin")
        if bin_det is not None and bin_det.dist_m is not None and bin_det.dist_m > 0.9:
            bp = s["base_pose"]
            ang = bp[2] + np.radians(bin_det.bearing_deg)
            self._bin_mem = np.array([bp[0] + bin_det.dist_m * np.cos(ang),
                                      bp[1] + bin_det.dist_m * np.sin(ang)])
        return self._go_ball(s)

    # ---------- 找球/对准/接近/抓取 ----------
    def _go_ball(self, s) -> Decision:
        R = self.env.robot
        ball = self._pick(s["detections"], "ball")
        if ball is not None:
            self._last_ball_br = ball.bearing_deg
        if ball is None:
            self._search += 1
            if self._search > 22:
                return Decision("视野内始终找不到球, 放弃", "done", {"success": False})
            return Decision("视野内没有球, 转头扫描", "turn_right",
                            {"seconds": 0.8})
        self._search = 0

        if self._phase in ("find_ball", "align_ball"):
            self._phase = "align_ball"
            if abs(ball.bearing_deg) > 10:
                return Decision(f"把球转到视野中心 (方位{ball.bearing_deg:+.0f}°)",
                                "turn_left" if ball.bearing_deg > 0 else "turn_right",
                                {"seconds": round(_turn_seconds(ball.bearing_deg), 2)})
            self._phase = "approach"

        if self._phase == "approach":
            # 边走边对准: 视觉近距会低估距离, 用紧阈值 + 航向修正
            if abs(ball.bearing_deg) > 18:
                return Decision(f"接近中修正航向 (方位{ball.bearing_deg:+.0f}°)",
                                "turn_left" if ball.bearing_deg > 0 else "turn_right",
                                {"seconds": round(_turn_seconds(ball.bearing_deg) * 0.7, 2)})
            d = ball.dist_m
            if d is None or d > 0.55:
                sec = float(np.clip((d - 0.45) / config.PRIM_V * 0.9 if d else 1.0, 0.4, 2.0))
                return Decision(f"球在 {d if d else '?'}m 外, 前进接近", "forward",
                                {"seconds": round(sec, 2)})
            self._phase = "grasp_seq"
            self._seq = 0

        if self._phase == "grasp_seq":
            steps = [
                ("张开手指准备抓取", "open_gripper", {}),
                ("臂伸到地面抓取位", "arm_pose", {"pose": "reach"}),
                ("闭合手指抓球", "close_gripper", {}),
                ("抬起携带", "arm_pose", {"pose": "carry"}),
            ]
            if self._seq < len(steps):
                th, tool, args = steps[self._seq]
                self._seq += 1
                return Decision(th, tool, args)
            # 验证: 用持有状态(本体感知)判断
            if self.env.robot.hal.attached_object() is not None:
                self._phase = "find_bin"
                self._grasp_fails = 0
                return Decision("已持有球, 开始找箱", "arm_pose", {"pose": "carry"})
            # 没抓到: 回到对准阶段, 重新转向+靠近 (视觉有误差, 迭代收敛)
            self._grasp_fails = getattr(self, "_grasp_fails", 0) + 1
            if self._grasp_fails > 3:
                return Decision("多次没抓到, 放弃", "done", {"success": False})
            self._phase = "align_ball"
            return Decision("没抓到, 张开手指重新对准再试", "open_gripper", {})

        return Decision("继续", "arm_pose", {"pose": "stow"})

    # ---------- 找箱/对准/接近/投放 ----------
    def _go_bin(self, s) -> Decision:
        # 投放序列中不再依赖视觉
        if self._phase == "drop_seq":
            steps = [
                ("臂到高位投放位", "arm_pose", {"pose": "drop"}),
                ("张开手指投放", "open_gripper", {}),
            ]
            if self._seq < len(steps):
                th, tool, args = steps[self._seq]
                self._seq += 1
                return Decision(th, tool, args)
            self._phase = "done"
            return Decision("投放完成", "done", {"success": True})

        if self._phase == "final_go":                # 终点推进完成 → 直接投放
            self._phase = "drop_seq"
            self._seq = 0
            return Decision("到位, 开始投放", "arm_pose", {"pose": "drop"})

        b = self._pick(s["detections"], "bin")
        if b is not None:
            self._last_bin_br = b.bearing_deg
            self._bin_seen = (b.bearing_deg, b.dist_m, self._n)   # 目击记录
            self._occl = 0
            self._close_steps = 0
        if b is None:
            # ① 新鲜目击(≤2拍)后消失 = 转到正前方被手里的球遮挡 —— 按末次距离行动
            seen = getattr(self, "_bin_seen", None)
            self._occl = getattr(self, "_occl", 0)
            if seen is not None and self._n - seen[2] <= 2 and seen[1] is not None:
                # 刚目击后消失 = 正前方被手里的球遮挡 —— 按末次目击距离一把推进
                if seen[1] > 0.50:
                    sec = float(np.clip((seen[1] - 0.30) / config.PRIM_V, 0.2, 1.0))
                    self._phase = "final_go"
                    return Decision("箱在正前方(遮挡), 一把推进到投放位", "forward",
                                    {"seconds": round(sec, 2)})
                self._phase = "drop_seq"
                self._seq = 0
                return Decision("已到箱前(遮挡推断), 开始投放", "arm_pose", {"pose": "drop"})
            self._search += 1
            if self._search > 22:
                return Decision("找不到收纳箱, 放弃", "done", {"success": False})
            # ② 扫3次不见 → 朝抓球前记忆的箱世界方位定向转 (26°步距会跳过可视窗)
            mem = getattr(self, "_bin_mem", None)
            if self._search > 3 and mem is not None:
                bp = s["base_pose"]
                want = np.degrees(np.arctan2(mem[1] - bp[1], mem[0] - bp[0]))
                diff = (want - np.degrees(bp[2]) + 180) % 360 - 180
                if abs(diff) > 12:
                    return Decision(f"按记忆方位找箱 (还需转{diff:+.0f}°)",
                                    "turn_left" if diff > 0 else "turn_right",
                                    {"seconds": round(min(abs(diff) / TURN_RATE_DEG, 0.9), 2)})
                d_mem = float(np.hypot(mem[0] - bp[0], mem[1] - bp[1]))
                if d_mem > 0.8:
                    return Decision("朝记忆箱位方向前进搜索", "forward", {"seconds": 0.8})
            return Decision("视野内没有箱, 转头扫描", "turn_right", {"seconds": 0.45})
        self._search = 0

        if abs(b.bearing_deg) > 18:
            return Decision(f"把箱转到视野中心 (方位{b.bearing_deg:+.0f}°)",
                            "turn_left" if b.bearing_deg > 0 else "turn_right",
                            {"seconds": round(_turn_seconds(b.bearing_deg) * 0.5, 2)})
        d = b.dist_m
        if d is not None and d <= 0.45:
            # 终点: 一把推进 (洋红箱检测干净: est+低估0.22-壁到心0.20-余量) 后直接投放
            advance = float(np.clip(d - 0.05, 0.10, 0.60))
            self._phase = "final_go"
            return Decision(f"对准箱(d={d:.2f}m), 推进{advance:.2f}m后投放", "forward",
                            {"seconds": round(advance / config.PRIM_V, 2)})
        sec = float(np.clip((d - 0.65) / config.PRIM_V * 0.9 if d else 0.8, 0.3, 1.6))
        return Decision(f"箱在 {d if d else '?'}m 外, 前进接近", "forward",
                        {"seconds": round(sec, 2)})
