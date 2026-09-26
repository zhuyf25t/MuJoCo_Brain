# MuJoCo_Brain 命令速查

## 每轮发给 DeepSeek API 的内容

适用于 `--brain openai` 且 `.env` 配置为 DeepSeek 的情况：

| 内容 | 实际发送的东西 |
|---|---|
| `front_cam` 车头图片 | 当前一张车头画面，实时编码成 JPEG/base64 放入请求；与记录的 `*_front_cam.jpg` 来自同一个相机，不是读取某个固定 JPG 文件 |
| 机器人状态 | 底盘 x/y/朝向、两个机械臂关节角、手指张开或闭合、夹爪是否持球，以文字发送 |
| 视觉估计 | 从图片估算的球和收纳箱方位、距离，以及可用的夹爪对齐提示；不是直接提供球的仿真真值坐标 |
| 任务与工具 | 任务文字、系统提示词、可调用工具的名称/说明/参数定义 |
| 动作历史 | 最近最多 12 次工具调用的名称、参数及执行反馈；不发送历史图片 |
| API 请求参数 | `.env` 中的模型名、`max_tokens`、`thinking`、`reasoning_effort`、`stream`；密钥用于请求认证，不上传 `.env` 文件 |

**不发送** `overhead`（俯视相机，不是后置相机）或 `side` 图片，也不上传整份 `meta.json`、`decisions.jsonl`、`trajectory.jsonl` 或场景 XML；这些主要用于本地记录、检查和回放。

**每个 episode 默认最多 32 轮决策**（`config.MAX_DECISIONS`），可以提前结束。一轮可返回多个工具调用；失败重试可能增加 API 请求次数，因此“决策轮数”不一定等于“工具调用次数”或“API 请求次数”。

这里按“命令：作用、参数、输出位置、覆盖规则”列出项目的运行入口。架构和实现细节见 [README.md](README.md)。

## 在 PowerShell 中准备环境

本页暂时只列 Windows PowerShell 命令，使用本机已能正常打开 GUI 的 `.venv-gui` 环境。下面所有命令均在项目根目录执行，直接指定 Python 路径，无需激活虚拟环境。

- `cd F:\AstraBel\mujoco`：进入项目根目录。
- `.\.venv-gui\Scripts\python.exe -c "import sys, mujoco; print(sys.executable); print(mujoco.__version__)"`：检查实际使用的 Python 和 MuJoCo；路径应包含本项目 `.venv-gui\Scripts\python.exe`。
- `.\.venv-gui\Scripts\python.exe -m pip install -r requirements.txt`：在 Windows 环境中安装或更新依赖。
- `.\.venv-gui\Scripts\python.exe -m pip check`：检查依赖冲突，不运行任务、不生成数据。
- `$env:MUJOCO_GL = "glfw"`：显式选择 Windows 图形后端，对当前 PowerShell 会话中后续启动的程序生效；采集、回放和重渲染入口在 Windows 上默认已使用它。

## 本机现在直接看演示：使用 PowerShell

- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --loop --speed 0.5`：半速循环播放已有记录；不调用 API，不生成新的 episode。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain scripted --episodes 1 --gui --keep-open`：运行一集规则策略，边模拟边采集；结束后保留窗口。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain openai --episodes 1 --out output/deepseek --gui --keep-open`：使用 `.env` 指定的模型运行一集并采集；当前配置为 DeepSeek，数据写到 `output/deepseek/ep_XXXX/`，会调用付费 API。

本机已验证 Windows GUI 可以播放，当前优先使用以上命令。

## 正式运行和采集：run_collect.py

- `.\.venv-gui\Scripts\python.exe run_collect.py`：默认使用 `scripted` 运行 3 集，初始种子为 0，保存到项目内 `data/episodes/`，不显示 GUI。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain scripted --episodes 10`：运行 10 集规则策略；无需 API Key，采集过程不训练模型。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain openai --episodes 1`：使用 OpenAI 兼容协议调用 `.env` 中的端点和模型；`openai` 是协议适配器名字，模型可以是 DeepSeek。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain anthropic --episodes 1`：使用 Anthropic 协议；需要单独配置 `.env` 中的 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain scripted --episodes 2 --out output/my_run`：将新记录写到 `output/my_run/ep_0000/`、`ep_0001/` 等；输出根目录不存在时自动创建。`--out` 接收数据根目录，不是单个 JSON 文件。
- `.\.venv-gui\Scripts\python.exe run_collect.py --episodes 3 --seed 42`：三集分别使用种子 42、43、44；默认 `--seed 0`。新一次命令不会按已有 episode 编号自动接续种子。
- `.\.venv-gui\Scripts\python.exe run_collect.py --task "把一个网球放进收纳箱" --brain openai --episodes 1`：改变给 brain 的任务文字；目前场景与评估仍是“任意网球入箱”，不会因文字自动生成其他物体或任务。
- `.\.venv-gui\Scripts\python.exe run_collect.py --episodes 1 --gui`：显示实时模拟窗口；结束时自动关闭。
- `.\.venv-gui\Scripts\python.exe run_collect.py --episodes 1 --gui --keep-open`：结束后保留最后画面，关闭窗口退出；`--keep-open` 必须配合 `--gui`。关闭窗口会停止后续 episode。
- `.\.venv-gui\Scripts\python.exe run_collect.py --episodes 1 --no-images`：保存元信息、决策和状态轨迹，但不保存相机 JPG；brain 仍会渲染和使用车头相机图像，LLM 仍会收到图片，因此这不等于关闭视觉或完全不需要图形后端。
- `.\.venv-gui\Scripts\python.exe run_collect.py --episodes 1 --quiet`：隐藏逐动作的终端日志；仍有每集结果和汇总，LLM 适配器自己的日志也可能显示；不减少数据记录。
- `.\.venv-gui\Scripts\python.exe run_collect.py --help`：显示全部命令行参数与帮助，不运行采集；`-h` 等价。

**覆盖规则**：一次正常、串行运行会在 `--out` 下寻找现有最大 episode 编号，再从下一个编号开始。例如已有 `ep_0000` 到 `ep_0009`，再次采集将从 `ep_0010` 开始，不覆盖旧集。不要让两个采集进程同时使用同一个输出根目录；当前编号分配没有并发锁。

每集保存 `meta.json`、`decisions.jsonl`、`trajectory.jsonl`；保存图片时还会产生 `imgs/`。`--out` 的相对路径以当前工作目录为准，默认输出路径则固定在本项目下。`data/` 与 `output/` 都被 Git 忽略。

`--episodes` 必须至少为 1。默认最多 32 轮 brain 决策，修改 `config.py` 的 `MAX_DECISIONS` 可调整；没有 `--max-decisions` 参数。一轮 LLM 决策可以返回多个工具调用。只要有一集未成功，程序的退出码就是 1，这本身不表示 Python 崩溃，仍应看具体结束原因。

## 查看记录：inspect_data.py

- `.\.venv-gui\Scripts\python.exe inspect_data.py`：列出默认 `data/episodes/` 中所有 episode 的统计和汇总；不运行仿真、不修改记录。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --root output/my_run`：改为查看 `output/my_run/` 中的记录；该参数可与下列查看或视频导出模式搭配。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --meta ep_0007`：打印该集的 `meta.json`，包括 brain、环境成功判定、结束原因、工具调用次数、帧数等。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --decisions ep_0007`：依次打印 brain 选择的工具和参数、文字说明、执行反馈，最后打印环境判定。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --frame ep_0007 --index 0`：打印 `trajectory.jsonl` 中第 0 条状态；索引从 0 开始，默认也是 0。它输出 JSON，不打开图片，且不等于第 0 次决策。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --replay ep_0007 --cam overhead`：将已保存的俯视相机 JPG 合成为 `data/episodes/ep_0007/replay_overhead.mp4`，不打开 GUI，也不重新渲染。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --replay ep_0007 --cam front_cam`：导出车头相机视频 `replay_front_cam.mp4`。`--cam` 默认 `overhead`，只能选择这集实际保存过的相机；当前默认录制 `front_cam` 和 `overhead`，没有录制 `side`。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --root output/my_run --meta ep_0000`：查看自定义输出根目录中的某一集。
- `.\.venv-gui\Scripts\python.exe inspect_data.py --help`：显示完整参数，`-h` 等价。

`--meta`、`--decisions`、`--frame`、`--replay` 四种模式互斥，每次选一个。`--index` 用于 `--frame`，`--cam` 用于 `--replay`。

**覆盖规则**：查看模式只读。导出视频时，同一 episode、同一相机的 `replay_相机名.mp4` 会被重新写入；想保留旧视频需先改名或复制。`--no-images` 采集的记录不能用这种方式导出视频，可以使用下面的 `rerender.py`。

## GUI 播放历史轨迹：replay_gui.py

- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007`：读取默认根目录中该集的 `trajectory.jsonl`，正常速度播放；默认结束后保留窗口。
- `.\.venv-gui\Scripts\python.exe replay_gui.py data/episodes/ep_0007`：直接指定 episode 目录。
- `.\.venv-gui\Scripts\python.exe replay_gui.py data/episodes/ep_0007/trajectory.jsonl`：直接指定轨迹文件；无需先运行 `inspect_data.py` 或在 GUI 中手动导入。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0000 --root output/my_run`：按 episode 名查找其他数据根目录；显式传入目录/JSONL 路径时直接使用该路径。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --speed 0.5`：半速播放；`--speed 2` 为两倍速，默认 1，必须大于 0。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --loop`：循环播放。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --exit-on-end`：播放结束自动关闭；与 `--loop` 不能同时使用。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --check`：只检查轨迹能否读取、维度与模型是否匹配、时间顺序和数值是否有效，不创建窗口。
- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --model models/scene_pickball.xml`：指定录制时对应的场景模型；默认就是这个 XML。相同维度不保证模型相同，需要自己保留正确模型。
- `.\.venv-gui\Scripts\python.exe replay_gui.py --help`：显示完整参数，`-h` 等价。

窗口中：空格暂停/继续，`R` 从头重播，鼠标调整视角，关闭窗口退出。播放结束后按空格也会从头开始。

**输出和覆盖规则**：回放只读已有数据，不创建新 episode、不调用 brain/API、不执行原工具。它用 XML 场景和记录的姿态重建画面，因此没有 JPG 也可回放。当前不保存完整速度、接触和随机化参数，不能从这里精确续跑物理模拟；状态记录约 10 Hz，GUI 刷新更快也不会增加原始轨迹信息。

## 从轨迹重新渲染视频：rerender.py

- `.\.venv-gui\Scripts\python.exe rerender.py ep_0007`：用轨迹重新渲染默认 `overhead` 相机，生成该集目录下的 `rerender_overhead.mp4`；默认 960×720、10 fps，不打开 GUI。
- `.\.venv-gui\Scripts\python.exe rerender.py ep_0007 --cams overhead side front_cam`：分别导出多个相机的视频，文件名为 `rerender_相机名.mp4`；相机必须存在于当前场景模型中，不要求采集时存过该相机的 JPG。
- `.\.venv-gui\Scripts\python.exe rerender.py ep_0007 --width 640 --height 480`：改变渲染分辨率；宽高必须为正整数，并受模型离屏缓冲区大小限制。
- `.\.venv-gui\Scripts\python.exe rerender.py ep_0007 --fps 10`：设置视频编码帧率；当前实现逐条轨迹写一帧，改变 fps 会改变播放速度，不会插值生成新状态。
- `.\.venv-gui\Scripts\python.exe rerender.py ep_0000 --root output/my_run`：从其他数据根目录读取轨迹。
- `.\.venv-gui\Scripts\python.exe rerender.py output/my_run/ep_0000/trajectory.jsonl`：直接传入轨迹文件；也支持 episode 目录。
- `.\.venv-gui\Scripts\python.exe rerender.py ep_0007 --model models/scene_pickball.xml`：指定对应的场景 XML，默认使用项目的捡球场景。
- `.\.venv-gui\Scripts\python.exe rerender.py --help`：显示完整参数，`-h` 等价。

**覆盖规则**：同一集目录下同名 `rerender_相机名.mp4` 会被覆盖，原 JSONL 和 JPG 不变。这个入口没有 `--out` 参数；输出位置固定为轨迹所在目录。渲染仍需要可用的图形后端，在 Windows 上默认使用 GLFW。

## 常见故障判断

遇到“无法识别 `.\.venv-gui\Scripts\python.exe`”：先确认当前目录是 `F:\AstraBel\mujoco`，并确认 Windows 虚拟环境目录 `.venv-gui` 存在。

遇到 `ModuleNotFoundError: No module named 'mujoco'`：先用上面的检查命令确认 `sys.executable`；应使用 `.venv-gui\Scripts\python.exe`。若已经是该环境，再用它安装 `requirements.txt`。

停在 `正在创建 GUI 窗口`：尚未进入采集/播放，需检查图形环境。只有看到 `GUI 已打开` 才表示窗口创建成功。`replay_gui.py --check` 能通过并不代表图形环境可用。

停在 `LLM→模型`：这时可能是在等待 API；与窗口创建阻塞不同。当前单请求超时为 `config.LLM_TIMEOUT_S=180` 秒，重试会增加总等待时间。

`--no-images` 仍需要渲染 brain 的车头观测，不能用来绕过图形初始化；不显示窗口也仍需要可用的渲染后端。

## 其他入口

- `.\.venv-gui\Scripts\python.exe compose_model.py`：组合底盘和机械臂模型并自检，**覆盖** `models/mobile_manip.generated.xml`；修改模型源文件后才需要重新生成。除 `-h/--help` 外没有自定义命令行参数。
- `.\.venv-gui\Scripts\python.exe -m mujoco.viewer --mjcf=models/scene_pickball.xml`：只打开场景，不运行项目 brain、不采集、不回放历史轨迹。
- `.\.venv-gui\Scripts\python.exe -m pytest tests/ -v`：运行全部自动测试；不是采集任务。渲染测试需要可用图形后端。

## Brain 到底能看到什么

Python 层的 observation 是一个字典，包含 `images.front`（车头 RGB 像素数组）、`base_pose`（底盘 x/y/yaw）、`arm_qpos`（两个臂关节角）、`finger`（开度 0～1）、`tcp_pos`（夹爪末端位置）；采集循环另加入 `holding`（是否持球）。此外会传入任务文字、工具定义和动作历史。

- `scripted`：没有神经网络或远程模型；当前实现通过 HAL 再取车头图片和本体状态，做本地颜色检测、距离/方位估计，再用规则选择动作。它不是只读一份坐标 JSON。
- `openai`（当前 DeepSeek）：发送车头图片、任务、可用工具、最近最多 12 次动作及反馈；还发送程序从图片估算的球/箱方位与距离，以及底盘位姿、臂角、手指开合、是否持球等文字。
- `anthropic`：同样发送车头图片、任务、工具和动作历史；当前文字观测包含底盘位姿、夹爪开度和 TCP 坐标，与 OpenAI 适配器的文字摘要尚不完全一致。

图片被编码成 JPEG/base64，放进 API 的 JSON 请求中的图片块。**传输格式是 JSON，不代表里面只有文字或坐标；视觉模型确实会接收图片。** 默认 brain 不接收俯视/侧视调试图，也不接收所有球的仿真真值位置。

`trajectory.jsonl` 保存的全场景 `qpos` 用于回放，其中包含的信息比 brain 的 observation 更多；这个文件不会作为整份观测直接发给 LLM。

这里有两个不同含义的“模型”：`.env` 的 `OPENAI_MODEL=deepseek-flash` 是远程 AI 模型，通过 `--brain openai` 调用；回放命令的 `--model models/scene_pickball.xml` 是 MuJoCo 的场景/机器人/物理参数文件。当前项目不下载或训练 DeepSeek 权重，也没有实现 ACT 训练。

API 配置只放 `.env`；仓库中的 `.env.example` 只含占位值。`max_tokens`、`thinking`、`reasoning_effort`、`stream` 是 `.env` 请求设置，不是 `run_collect.py` 的命令行参数。API 用量和实际耗时目前尚未自动写进 episode 日志。
