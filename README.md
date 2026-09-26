# MuJoCo_Brain

基于 MuJoCo 的机器人智能体实验平台，支持可替换大脑、工具调用与仿真轨迹记录。

命令与参数速查、输出文件及覆盖规则见 [README-simple.md](README-simple.md)。

项目关注大脑、工具与环境之间的闭环:大脑根据观测选择动作，控制层执行动作，
MuJoCo 推进物理模拟并生成新图像，记录器保存决策与轨迹。
`scripted` 是用于验证流程的规则基线；`openai` 和 `anthropic` 是可替换的 LLM 接口。
采集过程不训练模型；LeRobot 导出与 ACT 训练属于后续规划。

```text
                                          底层控制循环
            ┌───────────────────────────────────────────────────────────────────────┐
            v                                                                       │
┌──────────────────────┐            ┌──────────────────────┐            ┌───────────┴──────────┐
│ MuJoCo 仿真          │  观测      │ Brain (统一接口)     │  工具调用  │ ToolLayer            │
│ 差速小车+两自由度臂  │ -------->  │ scripted             │ -------->  │ forward / back       │
│ 网球 + 收纳箱        │            │ openai / anthropic   │ <--------  │ turn_left/right ...  │
└───────────┬──────────┘            └───────────┬──────────┘  工具反馈  └───────────┬──────────┘
            │ 状态/图像                         │ 决策                              │ 执行结果
            └───────────────────────────────────┼───────────────────────────────────┘
                                                v
                             EpisodeRecorder (决策级 + 控制级 10 Hz)
                                                │
                                                v
                                     data/episodes/ep_XXXX/
                                                │
                                                v
                         GUI 回放 / 视频导出 / (后续: LeRobot v3 -> ACT)
```

观测 = 车头相机图像 + 状态；工具执行反馈也会交给 Brain，供下一轮决策使用。

**当前任务**: 场地上散落 4 个网球，把任意一个球捡起来放进洋红色收纳箱。
视觉策略不区分球编号；环境独立检查球是否入箱并基本静止。
(机器人/任务形态参照 `mujoco-pzc/MuJoCo` 网球抓取小车)。

## 机器人(参照 mujoco-pzc/MuJoCo 网球抓取小车)

- **底盘**:差速小车,蓝色车身+顶板、参考车式万向脚轮(叉架+滚轮外观);
  轮位拓扑为"中置双驱动轮承滚转 + 前后脚轮承俯仰"——高重心臂下比参考车的
  "前驱+单后脚轮"稳定得多(实测前驱单脚轮构型会单轮卸载打滑/翘轮);
  脚轮碰撞体用各向同性近零摩擦球体(自旋不卡)
- **机械臂**:参考车形式的两自由度臂(肩俯仰 + 肘俯仰位置舵机)+ 掌部单边滑动指
  (右指固定,与参考车一致);挂点高 0.30m,臂展 0.06-0.74m,工作面为车头正前方(±7cm)
- **夹取**:手指闭合 + 球心距 TCP < 0.15m(预筛)且指面与球**有实际接触** → 后端登记
  "持有"(运动学吸附,物理抓取的仿真等效);张开即释放。对大脑只暴露持有/未持有状态
- **感知**:`control/perception.py` 颜色比例分割 + 针孔地面投影测距,
  方位 ±3° / 距离 ±15% (实测);收纳箱为洋红色(场景唯一,杜绝与球影色域冲突)
- **组合**:`compose_model.py` 用 MjSpec `spec.attach()` 组合底盘与臂
  (生成 `models/mobile_manip.generated.xml`)
- **相机**:`front_cam` 装在**车头**(俯角30°,机器人的唯一"眼睛",感知走视觉);
  `overhead/side` 仅用于录像回放,大脑不可见
- **控制**:底盘前进/后退按固定速度执行指定时长，转向用里程计反馈减速到位；
  机械臂通过预设关节姿态与微调控制，项目另保留 2R 解析 IK 工具

## 目录结构(控制与仿真分离)

```
models/                          # 场景/底盘/臂 XML + 生成的组合模型
hal.py                           # ★ 硬件抽象层: RobotInterface / WorldInterface
control/                         # ★ 控制逻辑 (零 mujoco 依赖, 可移植真机)
  kinematics.py                  #   两自由度解析 IK (纯数学)
  drivers.py                     #   BaseDriver(动作原语) / ArmDriver(预设+微调)
  tools.py                       #   ToolLayer: LLM 工具集
  perception.py                  #   车头相机视觉感知(颜色分割+地面投影)
sim/                             # ★ MuJoCo 仿真后端 (实现 hal 接口)
  backend.py                     #   MjRobot / MjWorld
  env.py                         #   MobileManipEnv: 场景组装/reset随机化
brains/                          # scripted / openai_compat / anthropic
config.py                        # 全部参数
compose_model.py                 # 模型组合 (python compose_model.py)
recorder.py                      # 双层数据记录
viz.py                           # EGL 相机组 + GUI viewer
run_collect.py                   # 采集主入口
run                              # 一键运行 (自举 venv → 生成模型 → 采集)
inspect_data.py                  # 数据统计 + 回放视频导出
rerender.py                      # 离线重渲染 (回放 trajectory, 任意相机出片)
replay_gui.py                    # 历史轨迹的交互式 GUI 回放
playback.py                      # 轨迹校验与状态恢复 (GUI/视频共用)
termimg_test.py                  # 终端内联图像(Kitty 协议)自检
tests/                           # 纯控制层(无mujoco) + 仿真管线 + LLM 协议 mock
data/episodes/                   # 采集输出
```

**控制/仿真分离**: `control/` 只 import `hal.py` 与纯数学, 不知道 MuJoCo 的存在
(tests/test_control_pure.py 用理想 2D 单车"假硬件"跑真实 BaseDriver 验证了这一点)。
将来接真机: 写一个 `RealRobot`(串口/CAN 发轮速与舵机目标, tick=sleep+回读)
实现 `RobotInterface`, 控制逻辑、工具层、LLM 大脑全部原样复用。
仿真专属机制(捡球的"运动学吸附"= 每步写球 qpos)封装在 sim 后端; 真机上由
物理夹爪天然实现, 接口的 attach/detach 只登记状态。

## 快速开始

使用 **Python 3.12**。根目录 `requirements.txt` 包含当前代码的运行、视频导出和
测试依赖,固定为此前在 Python 3.12 上通过项目测试的直接依赖版本。
这不是完整的传递依赖锁文件;安装后可用 `python -m pip check` 检查包依赖冲突。

Ubuntu 24.04 / WSL 首次安装:

```bash
cd /mnt/f/AstraBel/mujoco    # 按实际项目位置修改
sudo apt update
sudo apt install -y python3.12-venv libgl1 libegl1 libglfw3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

以后进入项目先执行 `source .venv/bin/activate`。如果 `.venv` 是 Windows 创建的,
请在 WSL 中另建环境,例如 `python3.12 -m venv ~/.venvs/mujoco`,然后
`source ~/.venvs/mujoco/bin/activate`;两个系统不能共用同一虚拟环境。
`./run` 首次自举时也从 `requirements.txt` 安装依赖,但它需要预装 `uv`,
并固定使用项目根目录的 Linux `.venv`。已有环境更新依赖时,重新执行上面的
`python -m pip install -r requirements.txt`。

**WSL GUI**: 以下窗口命令使用 WSL 2 的 WSLg 图形支持。在 Windows PowerShell
运行 `wsl --list --verbose` 确认 `VERSION` 为 `2`;WSL 1 无法直接使用 WSLg。
已有 Ubuntu 可以原地转换,不需要重装:保存工作并备份重要文件后,在 Windows
管理员 PowerShell 执行 `wsl --update`,再执行
`wsl --set-version Ubuntu-24.04 2`（发行版名称以列表为准）。
参考 [微软 WSL GUI 配置说明](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps)。
图形系统和 OpenGL 系统库需要单独配置,不能通过 `requirements.txt` 安装。

```bash
source .venv/bin/activate    # 或激活自己创建的 WSL 虚拟环境

# 1. 规则基线采集(无需 API Key，不进行训练)
python run_collect.py --brain scripted --episodes 10

# 2. 查看数据
python inspect_data.py
python inspect_data.py --decisions ep_0000     # 决策日志
python rerender.py ep_0000 --cams overhead side # 离线重渲染(960x720, 不用重跑)

# 3. GUI 实时采集；--keep-open 在采集结束后保留窗口
MUJOCO_GL=glfw python run_collect.py --brain scripted --episodes 1 --gui --keep-open

# 已有轨迹的交互式回放，不重新调用 brain
MUJOCO_GL=glfw python replay_gui.py ep_0007

# 只查看场景,不运行捡球策略
MUJOCO_GL=glfw python -m mujoco.viewer --mjcf=models/scene_pickball.xml

# 4. 测试
MUJOCO_GL=egl python -m pytest tests/ -v
```

`Pillow` 对应代码中的 `PIL`;`imageio-ffmpeg` 提供导出 MP4 所需的 FFmpeg 后端。
MuJoCo 会自动安装 Python 的 `glfw` 和 `PyOpenGL` 依赖。
当前 OpenAI 兼容接口直接使用 `httpx`,Anthropic 接口使用 `anthropic` + `httpx2`;
`brains/__init__.py` 会导入所有大脑,因此脚本模式也需要这些包。
当前尚无 LangGraph、PyTorch 或 LeRobot 的实际代码依赖,训练工作流实现后再添加。

## 查看记录与 GUI 回放

以下命令在项目根目录、激活虚拟环境后执行。`ep_0007` 是示例，换成自己已有的
episode 名；仓库不包含 `data/` 中的本地采集数据。先执行 `python inspect_data.py`
可以列出已有记录。

| 文件 | 内容与来源 | 用途 |
|---|---|---|
| `meta.json` | 任务、brain、种子、工具调用次数、环境成功判定和结束原因 | 查看本集概况 |
| `decisions.jsonl` | brain 的工具名、参数、说明，以及工具执行反馈 | 解释每个动作 |
| `trajectory.jsonl` | 约 10 Hz 的仿真时间、全量 `qpos`、机器人状态与 `ctrl` | 重建历史画面 |
| `imgs/` | 录制相机图像及每次工具调用前后的快照 | 看图与导出原始录像 |

JSONL 每行都是一个独立的 JSON 对象。决策记录的行数等于工具调用次数，
轨迹记录按控制采样产生，二者不一一对应。`thought` 在 scripted 模式下是固定规则
填写的说明；LLM 模式下保存适配器返回的文本说明，可能为空。

```bash
python inspect_data.py --meta ep_0007
python inspect_data.py --decisions ep_0007
python inspect_data.py --frame ep_0007 --index 0
```

三条命令都只读取已有记录并打印到终端，不运行 brain，也不打开 GUI：

- `--meta ep_0007`：格式化显示本集的 `meta.json`，包括任务、brain 类型、
  环境成功判定、结束原因、工具调用次数、轨迹帧数和模拟时长。
- `--decisions ep_0007`：逐条显示 `decisions.jsonl` 中的工具名、参数、
  brain 说明及工具执行反馈，最后显示环境判定。
- `--frame ep_0007 --index 0`：显示 `trajectory.jsonl` 的第 0 条记录。
  索引从 0 开始；输出是模拟时间、位置/姿态、控制量和图片路径等 JSON 数据，
  不是打开图片，也不是第 0 次 brain 决策。第一条记录不一定在模拟时间 0 秒。

### 实时演示与历史回放的区别

- `run_collect.py --gui`：运行新的任务，brain 持续决策，并生成新记录。
  加 `--keep-open` 可保留最后画面；关闭窗口会停止后续采集。
- `replay_gui.py`：读取已经保存的状态，不调用 brain，也不重新执行工具或推进物理。
  可以自由调整视角，空格暂停/继续，`R` 从头重播。默认播放结束后保留窗口。
- `inspect_data.py --replay`：把已存图片合成 MP4。
- `rerender.py`：读取轨迹重新渲染 MP4，可改变相机与分辨率。

```bash
# 按 episode 名回放，也可以直接传入 JSONL 文件
MUJOCO_GL=glfw python replay_gui.py ep_0007
MUJOCO_GL=glfw python replay_gui.py data/episodes/ep_0007/trajectory.jsonl --speed 0.5

# 其他数据目录、循环播放、只校验文件
python replay_gui.py ep_0000 --root output/my_run --loop
python replay_gui.py ep_0007 --check

# 导出视频（不打开 GUI）
python inspect_data.py --replay ep_0007 --cam overhead
python rerender.py ep_0007 --cams overhead side
```

`--speed 2` 为两倍速，`--exit-on-end` 可在播放结束后自动关闭窗口。
回放必须使用录制时对应的模型；自定义场景可通过 `--model path/to/scene.xml` 指定。
程序会检查每帧的位置/控制量维度、时间顺序和数值，但同维度不代表同一模型。
当前轨迹用于姿态回放，没有完整保存速度、接触与随机化参数，不能当作精确续跑
物理模拟的存档。显示状态仍为原有约 10 Hz 采样，GUI 刷新更快不会补出新状态。

Windows 原生 Python 环境也可执行 `python replay_gui.py ep_0007`，默认使用 GLFW；
它需要独立安装 Windows 版依赖，不能使用 WSL 的 `.venv`。
`--check` 不创建窗口，可用于区分轨迹问题与图形环境问题。

如果 WSL 的 GPU 初始化卡住，可以先用进程级软件渲染验证离屏采集/导出，
不会修改系统配置（速度可能更慢）：

```bash
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  python run_collect.py --brain scripted --episodes 1 --out output/software-check
```

WSLg 窗口创建仍依赖系统图形服务；软件 EGL 验证通过不代表 WSLg 窗口问题已经解决。

### 终端每一行是谁产生的

```text
[动作01/轮01] brain: turn_right({'seconds': 0.37})
    tool[OK]: 右转 0.4s (实际转-22°) 完成, 当前位姿 [-0.9, 0.01, -0.39]
```

- `动作01/轮01`：采集程序的工具调用编号与 brain 决策轮次。LLM 一轮可返回多个动作。
- `brain: turn_right(...)`：brain 选出的工具名称与参数。
- `tool[OK]`：工具/控制层的执行反馈，表示动作执行成功，不保证抓球或入箱成功。
- `实际转-22°` 和 `当前位姿`：控制层读取仿真状态后生成的反馈；中文不是 MuJoCo 自动生成的。
- 位姿是 `[x 米, y 米, yaw 弧度]`，第三项是朝向，不是高度。
- `done(success=True)`：brain 的自评；最终 `环境判定: PASS/FAIL` 由评估代码独立读取
  球的位置和速度决定。新记录还用 `meta.brain_success` 单独保存自评。

默认最多 32 轮 brain 决策；scripted 每轮一个工具调用，LLM 批量调用时总动作数可能更大。
物理步长是 0.002 秒，控制/记录周期是 0.1 秒，brain 则在上一批动作完成后再决策。
`forward(seconds=2)` 中的时间是模拟时间，运行和渲染所用的实际时间可能更长。

### 生命周期与当前边界

每个 episode 都清空规则 brain 的阶段、失败计数、搜索状态及目标记忆，避免上一集
影响下一集。车头相机、录制相机和记录文件均显式关闭，正常结束、异常和 Ctrl+C
都会走资源释放路径；GUI 不再依赖强制跳过 Python 清理来退出。

这两项修复保证回合独立和资源收尾，不代表规则策略已具备稳定抓取/投放能力。
感知误差、接近策略和投放判断仍是基线的已知限制。

## 接入 LLM 大脑

端点、密钥和模型的配置优先级：**函数参数 > 项目根目录 `.env` >
`config.py` 默认值 > 进程环境变量**。项目配置优先，避免其他工具的同名环境变量
改变本项目使用的端点。程序只读取本项目的 `.env`，不执行文件内容或改写进程环境。

密钥放在被 Git 忽略的 `.env` 中；不要写进 `config.py`。新建配置可参考
[`.env.example`](.env.example)，已有 `.env` 时直接使用，避免覆盖。
例如 DeepSeek 的 OpenAI 兼容接口：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
max_tokens=8192
thinking=enabled
reasoning_effort=low
stream=false
```

也支持原有 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 名称；两组同时存在时
`LLM_*` 优先。上述可选请求参数从本地配置文件读取，只用于 OpenAI 兼容适配器；
目标服务需支持 `thinking` 和 `reasoning_effort`，不支持时删掉对应配置。
当前只支持非流式响应，`stream` 必须为 `false`。`max_tokens` 是回复 token 上限，
不是每次都会消耗的数量。

Anthropic 适配器使用独立的 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、
`ANTHROPIC_MODEL` 配置；不会复用 OpenAI 端点或密钥。

```bash
python run_collect.py --brain openai      # OpenAI 兼容协议
python run_collect.py --brain anthropic   # Anthropic 协议(实测走通)
```

本机 Windows GUI 已验证可用，PowerShell 中可边运行 DeepSeek 边采集：

```powershell
cd F:\AstraBel\mujoco
.\.venv-gui\Scripts\python.exe run_collect.py --brain openai --episodes 1 --gui --keep-open
```

`openai` 指兼容协议，实际调用的是本地配置文件里的模型。其他机器需要先创建自己的
Windows 虚拟环境并安装 `requirements.txt`；WSL 的 `.venv` 不能直接用于 Windows。
新增依赖 `python-dotenv` 用于读取本地配置文件，更新项目后需安装最新依赖。

备注:
- OpenAI 兼容 Brain 会自动探测 `{base}/chat/completions` 与 `{base}/v1/chat/completions`
  (中转站可能任一路由可用), 5xx 自动退避重试
- Anthropic SDK 默认客户端对部分中转网关返回 401(TLS 指纹类问题), Brain 内已显式
  构造裸 httpx2 客户端规避; anthropic 1.x 基于 httpx2(非 httpx)
- Anthropic Brain 开启 adaptive thinking + 摘要显示, 模型思考过程写入
  decisions.jsonl 的 thought 字段(数据飞轮的一部分)

两个 Brain 都:原生 tool-use 协议 → 文本 JSON 兜底解析 → 解析失败回喂错误重试 1 次 →
仍失败安全回退原地重观察；**每轮观测都会发送车头相机图片**（无单独的 look 工具）。
OpenAI 兼容适配器另发送程序估算的视觉检测结果与本体状态；Anthropic 适配器当前
发送底盘位姿、夹爪开度和 TCP 坐标，二者的文字摘要尚不完全一致。
历史以单行摘要传递（不累积历史图）；
工具名、参数、执行反馈和 brain 文本说明写入 `decisions.jsonl`；当前 LLM 文本说明
最多保留 400 字符。API 的输入/输出 token 用量和请求耗时尚未记录。
这里的 token 是模型处理内容的计量单位；请求耗时是实际等待 API 的时间，
与仿真的 `t` 时间不同。
LLM 输入/输出实时打印到终端,支持 Kitty 图形协议(Ghostty)内联显示相机图像
(`config.LLM_TERM_IMAGES`)。

### LLM 可用的工具(动作原语)

| 工具 | 说明 |
|---|---|
| `forward(seconds)` / `back(seconds)` | 直行/倒车 |
| `turn_left(seconds)` / `turn_right(seconds)` | 原地转向(闭环, 里程计反馈) |
| `arm_pose(pose)` | `stow` 行驶收纳 / `carry` 持球携带 / `reach` 下探抓取 / `drop` 投放 |
| `shoulder(delta)` / `elbow(delta)` | 肩/肘关节微调(弧度) |
| `open_gripper()` / `close_gripper()` | 手指开合(闭合要求真实接触球才算持有) |
| `done(success)` | 宣告任务结束 |

没有 move-to 类指令,也没有 look()——机器人像真机一样只靠车头相机 + 动作原语完成任务。
LLM 每轮可一次产出**一批 tool call**,按顺序执行完再带着新观测回来(批量决策)。

工具失败会返回原因(横向超限/超出臂展/附近无球等)并回喂给 LLM —— 失败也是数据。

## 数据格式与 ACT 对接

每集一个目录:

```
data/episodes/ep_0042/
  meta.json          # task/brain/success/seed/时长/失败原因
  decisions.jsonl    # 决策级: {t, img_before/after, thought, tool, args, ok, result}
  trajectory.jsonl   # 控制级 10Hz: {t, imgs(2路), qpos全量, base_pose, arm_qpos, finger, ctrl}
  imgs/              # 000123_front_cam.jpg / 000123_overhead.jpg + 决策快照
```

**映射到 LeRobotDataset v3 / ACT**:

| LeRobot 键 | 本数据 |
|---|---|
| `observation.state` (6维) | `arm_qpos`(2) + `finger`(1) + `base_pose`(3) |
| `action` (6维) | 同维目标值(下一帧 state 的目标/ctrl 可直接回归绝对动作) |
| `observation.images.front` | `imgs/*_front_cam.jpg`(车头相机,即策略输入) |
| `observation.images.overhead` | `imgs/*_overhead.jpg`(仅回放/调试,非策略输入) |
| `timestamp` / fps | `t` / 10 |
| tasks 文本 | `meta.json.task` |

参考实现(下一步迭代):
```python
from lerobot.common.datasets.v3 import LeRobotDataset
ds = LeRobotDataset.create("openclaw_pickplace", fps=10, ...   # 需 lerobot>=0.4
    features={"observation.state": 11维, "action": 11维, 两路图像})
for ep in episodes:
    for frame in trajectory:  # 只保留 success=True 的集
        ds.add_frame({...})
    ds.save_episode()
ds.finalize()                 # 必须调用, 否则 parquet footer 损坏
# 之后: lerobot 官方 ACT policy (ResNet18 + CVAE + 动作分块) 直接训练
```

**训练建议**(LeRobot 官方经验):相机固定 ✓、抓取行为一致 ✓(脚本策略)、
~50 条演示起步、ScriptedBrain 批量 200–1000 条很便宜(本机 headless 约 1 分钟/10 集)。

## 两阶段工作流(本仓库定位)

- **阶段一(已交付)**:LLM/脚本 控制 + 全量记录。ScriptedBrain 批量产数据;
  LLM Brain 处理新任务/冷启动,失败数据同样有价值
- **阶段二(下一步)**:success=True 的轨迹 → 导出 LeRobot v3 → `lerobot` ACT 训练 →
  部署 `ACTBrain`(实现同一个 Brain 接口,策略推理代替 LLM)→ 新任务回退 LLM,循环

## 已知限制与下一步

- **脚本策略(纯视觉+原语)当前不稳定**: 找球/对准/接近/抓取段可靠,
  投放段(持球找箱→对准→停靠)受单相机+手中球遮挡/阴影影响, 成功率低。
  这正是 LLM 大脑(会推理)与阶段二 ACT 策略(学出来的鲁棒性)要解决的问题
- 转向原语在持球负载下有时打滑(角速度下降), 待查(轮地摩擦/重心)
- 本机 nvidia 驱动异常(`nvidia-smi` 失败,渲染不受影响):ACT 训练前需修驱动或用远程 GPU
- 臂只有 2 自由度:抓取依赖底盘对准;斜侧向目标需要多步 reposition
- LLM 决策上限 `MAX_DECISIONS=32`(每轮可批量 tool call),任务文本解析支持"N号"球编号
