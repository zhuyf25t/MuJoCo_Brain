"""OpenAI-compatible input contract: front RGB, task, tools and own commands.

Keep this separate from the legacy Anthropic context. Do not accept robot
state, perception output, tool results or success flags in the text builder.
"""

from __future__ import annotations

import json

import config


SYSTEM_PROMPT = """你是捡球小车的大脑，通过提供的工具控制底盘、两自由度机械臂和夹爪。
根据任务说明，在车头图片中寻找球和收纳箱，尝试抓取并投放。

每轮的外部观测只有当前一张车头图片。另有任务、工具定义和最近的动作指令记录。
动作记录只说明程序调用过这些指令，不代表实际移动到位、夹住球或完成投放。
没有额外的测距、定位、关节回读、持球检测或相机标定信息；不要编造精确距离和状态。

工具参数的单位与范围以工具定义为准。预设姿态表示发送对应运动目标，不保证实际到位。
接近目标时小步操作，通过后续图片重新判断。机械臂遮挡目标时，先恢复可观察的视野。
闭爪后根据图像检查，必要时小幅抬臂或调整姿态；球从视野消失不等于已经夹住。
张爪后同样要看图确认投放结果。无法确认时保留不确定性，不反复盲目执行相同序列。

一轮可以返回一个或多个工具调用，程序会依次执行，中间不会重新询问你。
需要根据新图片判断的动作应放在本批末尾，下一轮观察后再继续。
完成或决定放弃时调用 done；success 是你根据视觉作出的自评。
"""


def command_history(history: list[dict]) -> str:
    """Whitelist only commands; even an OK flag can depend on simulator truth."""
    if not history:
        return "(尚无动作指令记录)"
    return "\n".join(
        json.dumps({"tool": entry.get("tool"), "args": entry.get("args", {})},
                   ensure_ascii=False)
        for entry in history[-config.HISTORY_LINES:]
    )


def task_text(task: str, history: list[dict]) -> str:
    """Intentionally takes no observation dict: no path for state text to leak."""
    return (f"任务: {task}\n\n"
            f"最近动作指令（仅调用记录，不是实际执行结果）:\n{command_history(history)}\n\n"
            "请根据当前车头图片决定下一步工具调用。")
