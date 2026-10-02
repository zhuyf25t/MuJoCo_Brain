"""固定演示四种 arm_pose；不调用模型，不移动底盘，不抓球。"""

from __future__ import annotations

from .base import Brain, Decision


class ArmPoseTestBrain(Brain):
    name = "test"
    demo_only = True
    gui_hold_s = 2.0
    poses = ("stow", "reach", "carry", "drop", "stow")

    def __init__(self):
        self.loop = False
        self.reset()

    def reset(self):
        self._index = 0

    def decide(self, obs, task_text, tool_schemas, history):
        if not self.loop and self._index >= len(self.poses):
            ok = all(entry.get("ok", False) for entry in history)
            return [Decision("四种预设演示结束，已返回 stow", "done", {"success": ok})]
        poses = self.poses[:-1] if self.loop else self.poses
        pose = poses[self._index % len(poses)]
        self._index += 1
        actions = []
        if self._index == 1:
            actions.append(Decision("张开夹爪，保持底盘不动", "open_gripper"))
        actions.append(Decision(f"{'循环' if self.loop else '顺序'}演示 {self._index}: {pose}",
                                "arm_pose", {"pose": pose}))
        return actions
