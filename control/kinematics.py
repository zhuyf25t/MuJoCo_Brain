"""两自由度平面臂解析运动学 (纯数学, 零仿真依赖)."""

from __future__ import annotations

import numpy as np


class PlanarArm2R:
    """肩+肘两连杆 (均在矢状面内俯仰, q=0 时竖直向上, q 正 = 向前倒).

    位置正解: p = L1·u(q1) + L2·u(q1+q2),  u(θ) = (sinθ, cosθ) 取 (前向, 上向).
    """

    def __init__(self, l1: float, l2: float,
                 q_lo: np.ndarray, q_hi: np.ndarray, margin: float = 0.04):
        self.L1 = float(l1)
        self.L2 = float(l2)
        self.q_lo = np.asarray(q_lo, dtype=float)
        self.q_hi = np.asarray(q_hi, dtype=float)
        self.reach_lo = abs(self.L1 - self.L2) + margin
        self.reach_hi = self.L1 + self.L2 - margin

    def fk(self, q: np.ndarray) -> np.ndarray:
        """TCP 相对肩轴的 (前向, 上向) 坐标."""
        q1, q2 = float(q[0]), float(q[1])
        f = self.L1 * np.sin(q1) + self.L2 * np.sin(q1 + q2)
        u = self.L1 * np.cos(q1) + self.L2 * np.cos(q1 + q2)
        return np.array([f, u])

    def ik(self, df: float, dz: float) -> np.ndarray | None:
        """目标 (前方 df, 高度差 dz) → [q1, q2]; 无解(超臂展/越限位)返回 None.

        双肘分支都试: 先肘下卷(高位目标), 不在限位内再试肘上翻(地面低目标).
        """
        r = float(np.hypot(df, dz))
        if not (self.reach_lo <= r <= self.reach_hi):
            return None
        c2 = (r * r - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        phi = float(np.arctan2(df, dz))              # 从竖直向上量起
        for sign in (-1.0, 1.0):
            q2 = sign * float(np.arccos(np.clip(c2, -1.0, 1.0)))
            psi = float(np.arctan2(self.L2 * np.sin(q2), self.L1 + self.L2 * np.cos(q2)))
            q1 = phi - psi
            q = np.array([q1, q2])
            if np.all(q >= self.q_lo - 1e-3) and np.all(q <= self.q_hi + 1e-3):
                return q
        return None
