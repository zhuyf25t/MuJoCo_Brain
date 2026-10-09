"""全局配置: 路径 / 命名 / 频率 / 视觉标定 / 动作原语参数."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# ---- 路径 ----
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR  # 项目根
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
EPISODES_DIR = DATA_DIR / "episodes"

BASE_XML = MODELS_DIR / "base_chassis.xml"
SCENE_XML = MODELS_DIR / "scene_pickball.xml"
GENERATED_XML = MODELS_DIR / "mobile_manip.generated.xml"
ARM_XML = MODELS_DIR / "reference_arm.xml"

# ---- 模型命名 ----
ARM_PREFIX = "arm_"
JOINT_BASE = "base_ref"
JOINT_WHEEL_L = "wheel_left_joint"
JOINT_WHEEL_R = "wheel_right_joint"
ACT_WHEEL_L = "wheel_left_vel"
ACT_WHEEL_R = "wheel_right_vel"
BODY_BASE = "base"

JOINT_SHOULDER = f"{ARM_PREFIX}shoulder_joint"
JOINT_ELBOW = f"{ARM_PREFIX}elbow_joint"
JOINT_FINGER = f"{ARM_PREFIX}finger_slide"
ACT_SHOULDER = f"{ARM_PREFIX}shoulder_servo"
ACT_ELBOW = f"{ARM_PREFIX}elbow_servo"
ACT_FINGER = f"{ARM_PREFIX}finger_servo"
SITE_TCP = f"{ARM_PREFIX}gripper_tcp"
BODY_GRIPPER = f"{ARM_PREFIX}gripper_tip"
BODY_ARM_BASE = f"{ARM_PREFIX}arm_base"

# 指滑动 ctrl: 0 = 全开, 0.015 = 轻贴球面
FINGER_OPEN_CTRL = 0.0
FINGER_CLOSE_CTRL = 0.05    # 满行程闭合: 左指面到+0.027, 与固定右指(-0.077)夹住0.12m球; 行程不足时正对球差2mm永远无接触
FINGER_CLOSED_THRESHOLD = 0.008   # ctrl 超过此值视为"已闭合"(自动吸附条件)
GRASP_DIST = 0.15                 # 球心距 TCP 此距离内才做接触检测(预筛)

# ---- 相机 (车头前向, 唯一的"眼睛") ----
CAM_FRONT = "front_cam"
CAM_FRONT_RES = (320, 240)        # 感知分辨率 (W, H)
CAM_FRONT_FOVY = 75.0             # 垂直视场角 (度, 与 XML 一致)
CAM_HEIGHT = 0.26                 # 相机离地高度 (m)
CAM_PITCH_DEG = 30.0              # 光轴俯角 (度)
CAM_FORWARD_OFFSET = 0.16         # 相机相对底盘中心的前移量 (m)

# 记录用相机 (不给大脑, 只给人看回放)
CAM_OVERHEAD = "overhead"
CAM_SIDE = "side"
OBS_CAMS = [CAM_FRONT]            # 大脑观测相机
REC_CAMS = [CAM_FRONT, CAM_OVERHEAD]

# ---- 频率 / 步长 ----
SIM_DT = 0.002
CTRL_HZ = 10
CTRL_DT = 1.0 / CTRL_HZ

# 正常结束决策后继续运行的仿真秒数，再评定结果；继续采集，不调用模型。
# 保持臂/爪目标并停止底盘；按控制周期向上取整，0 禁用。异常、关窗和演示跳过。
EPISODE_SETTLE_SECONDS = 3.0

# ---- 动作原语 (低层时间盒控制) ----
PRIM_V = 0.30          # forward/back 速度 (m/s)
PRIM_W = 1.0           # turn_left/right 角速度 (rad/s)
PRIM_MAX_S = 3.0       # 单次命令最长秒数
ARM_NUDGE_MAX = 0.5    # shoulder/elbow 单次微调最大弧度

# ---- 臂预设姿态 [肩, 肘] ----
ARM_STOW = [0.15, -0.6]      # 行驶收纳
ARM_CARRY = [0.55, -1.1]     # 持球携带 (高位)
ARM_REACH = [1.15, 1.46]     # 前伸抓取 (TCP≈前方0.55m, 离地0.11m)
# 投放使用高肘分支：肘部约高0.64m，夹爪朝前下方；TCP仍≈前方0.55m、高0.50m。
ARM_DROP = [0.48503, 1.44]

# ---- 任务场景 ----
BALL_NAMES = ["ball_0", "ball_1", "ball_2", "ball_3"]
BALL_R = 0.06
BALL_SPAWN_X = (0.45, 1.75)
BALL_SPAWN_Y = (-1.30, 1.30)
BIN_POS = (1.35, 0.0)
BIN_HALF_INT = (0.17, 0.17)
BIN_WALL_TOP = 0.16
BASE_SPAWN = (-0.9, 0.0, 0.0)

# ---- 视觉检测参数 ----
BALL_HUE = (55, 100)    # 网球黄绿色相范围 (度)
BALL_SAT = 0.35
BALL_VAL = 0.25
BALL_MIN_AREA = 25      # 像素
# 收纳箱=洋红色 (perception._bin_mask 颜色比判别)
BIN_SAT = 0.30
BIN_VAL = 0.20
BIN_MIN_AREA = 150

# ---- 随机化 ----
RAND_PHYS = 0.2

# ---- Brain / LLM ----
MAX_DECISIONS = 32
LLM_TIMEOUT_S = 180.0   # 大 max_tokens 下思考链长, 单请求可能超 60s
LLM_RETRIES = 3          # 超时/连接错/5xx 自动重试
LLM_MAX_TOKENS = 65536   # 单次模型输出 token 上限；开启思考时需同时容纳 reasoning 和工具调用
LLM_MAX_IMAGE_EDGE = 1024
LLM_TERM_IMAGES = True      # 终端内联显示发给 LLM 的相机图 (需 Kitty 图形协议终端, 如 Ghostty)
HISTORY_LINES = 12

ENV_OPENAI_BASE = "LLM_BASE_URL"
ENV_OPENAI_KEY = "LLM_API_KEY"
ENV_OPENAI_MODEL = "LLM_MODEL"
ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
ENV_ANTHROPIC_MODEL = "ANTHROPIC_MODEL"

# 只读取本项目 .env，不改写进程环境，不展开其中的 ${...} 或执行 shell。
# dotenv_values 支持引号、注释与 Windows BOM；密钥不写入已跟踪的源码。
LOCAL_LLM_FILE = ROOT / ".env"
_local_llm = dotenv_values(LOCAL_LLM_FILE, encoding="utf-8-sig", interpolate=False)


def _local_setting(*names: str, default=""):
    return next((_local_llm[n] for n in names if _local_llm.get(n)), default)


# ---- OpenAI 兼容端点；本地配置文件优先，default 为源码默认值 ----
LLM_BASE_URL = _local_setting("LLM_BASE_URL", "OPENAI_BASE_URL")
LLM_API_KEY = _local_setting("LLM_API_KEY", "OPENAI_API_KEY")
LLM_MODEL = _local_setting("LLM_MODEL", "OPENAI_MODEL")
OPENAI_MAX_TOKENS = _local_setting("OPENAI_MAX_TOKENS", "max_tokens", default=LLM_MAX_TOKENS)
OPENAI_THINKING = _local_setting("OPENAI_THINKING", "thinking")
OPENAI_REASONING_EFFORT = _local_setting("OPENAI_REASONING_EFFORT", "reasoning_effort")
OPENAI_STREAM = _local_setting("OPENAI_STREAM", "stream", default="false")

# 两类 API 的端点和密钥分别配置，避免把 OpenAI 协议的凭据发到 Anthropic 端点。
ANTHROPIC_BASE_URL = _local_setting("ANTHROPIC_BASE_URL")
ANTHROPIC_API_KEY = _local_setting("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL_DEFAULT = _local_setting("ANTHROPIC_MODEL")


def ensure_dirs() -> None:
    for d in (DATA_DIR, EPISODES_DIR):
        os.makedirs(d, exist_ok=True)
