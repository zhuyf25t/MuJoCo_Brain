# MuJoCo_Brain 命令速查

这里按“命令：作用、参数、输出位置、覆盖规则”列出项目的运行入口。架构和实现细节见 [README.md](README.md)。

## 先选对 Python 环境

本机有两套独立环境：WSL 的 `.venv`、Windows 的 `.venv-gui`。两者共享项目和数据文件，但 Python 与依赖不能混用。

- WSL：`cd /mnt/f/AstraBel/mujoco`：进入项目根目录。
- WSL：`source .venv/bin/activate`：激活 Linux 虚拟环境；每次新开终端都要重新激活，此后 `python` 和 `python3` 都应指向它。
- WSL：`python3 -c "import sys, mujoco; print(sys.executable); print(mujoco.__version__)"`：检查实际使用的 Python 和 MuJoCo；路径应包含本项目 `.venv/bin/python`。
- WSL：`.venv/bin/python inspect_data.py`：直接指定虚拟环境的 Python，无需先激活；其他入口也可使用这个前缀。
- PowerShell：`cd F:\AstraBel\mujoco`：进入同一项目的 Windows 路径。
- PowerShell：`.\.venv-gui\Scripts\python.exe inspect_data.py`：直接使用现有 Windows 环境，无需激活。
- WSL：`python3 -m pip install -r requirements.txt`：在已激活的 Linux 环境中安装或更新依赖。
- PowerShell：`.\.venv-gui\Scripts\python.exe -m pip install -r requirements.txt`：在 Windows 环境中安装或更新依赖。
- `python3 -m pip check`：检查依赖冲突，不运行任务、不生成数据。

下面通用命令使用 `python3`，默认已在 WSL 中激活环境并进入项目根目录。在 PowerShell 中把 `python3` 换成 `.\.venv-gui\Scripts\python.exe`。`MUJOCO_GL=...` 这种前缀属于 Bash，不能原样粘贴到 PowerShell。

首次创建环境、系统库与 WSLg 的安装步骤见主 README；这里使用已经建好的环境。

## 本机现在直接看演示：使用 PowerShell

- `.\.venv-gui\Scripts\python.exe replay_gui.py ep_0007 --loop --speed 0.5`：半速循环播放已有记录；不调用 API，不生成新的 episode。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain scripted --episodes 1 --gui --keep-open`：运行一集规则策略，边模拟边采集；结束后保留窗口。
- `.\.venv-gui\Scripts\python.exe run_collect.py --brain openai --episodes 1 --out output/deepseek --gui --keep-open`：使用 `.env` 指定的模型运行一集并采集；当前配置为 DeepSeek，数据写到 `output/deepseek/ep_XXXX/`，会调用付费 API。

2026-09-26 本机检查结果：Windows GUI 已能播放；WSL GUI 尚未稳定跑通，此前发现 Xwayland 卡在 WSL GPU 适配器枚举。这个问题与是否激活 Python 环境是两件事，不能靠重复安装 Python 包解决，也不能保证多等一会就会出现窗口。

## 正式运行和采集：run_collect.py

- `python3 run_collect.py`：默认使用 `scripted` 运行 3 集，初始种子为 0，保存到项目内 `data/episodes/`，不显示 GUI。
- `python3 run_collect.py --brain scripted --episodes 10`：运行 10 集规则策略；无需 API Key，采集过程不训练模型。
- `python3 run_collect.py --brain openai --episodes 1`：使用 OpenAI 兼容协议调用 `.env` 中的端点和模型；`openai` 是协议适配器名字，模型可以是 DeepSeek。
- `python3 run_collect.py --brain anthropic --episodes 1`：使用 Anthropic 协议；需要单独配置 `.env` 中的 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`。
- `python3 run_collect.py --brain scripted --episodes 2 --out output/my_run`：将新记录写到 `output/my_run/ep_0000/`、`ep_0001/` 等；输出根目录不存在时自动创建。`--out` 接收数据根目录，不是单个 JSON 文件。
- `python3 run_collect.py --episodes 3 --seed 42`：三集分别使用种子 42、43、44；默认 `--seed 0`。新一次命令不会按已有 episode 编号自动接续种子。
- `python3 run_collect.py --task "把一个网球放进收纳箱" --brain openai --episodes 1`：改变给 brain 的任务文字；目前场景与评估仍是“任意网球入箱”，不会因文字自动生成其他物体或任务。
- `python3 run_collect.py --episodes 1 --gui`：显示实时模拟窗口；结束时自动关闭。Linux 需要正常工作的 GLFW/WSLg 图形环境。
- `python3 run_collect.py --episodes 1 --gui --keep-open`：结束后保留最后画面，关闭窗口退出；`--keep-open` 必须配合 `--gui`。关闭窗口会停止后续 episode。
- `python3 run_collect.py --episodes 1 --no-images`：保存元信息、决策和状态轨迹，但不保存相机 JPG；brain 仍会渲染和使用车头相机图像，LLM 仍会收到图片，因此这不等于关闭视觉或完全不需要图形后端。
- `python3 run_collect.py --episodes 1 --quiet`：隐藏逐动作的终端日志；仍有每集结果和汇总，LLM 适配器自己的日志也可能显示；不减少数据记录。
- `python3 run_collect.py --help`：显示全部命令行参数与帮助，不运行采集；`-h` 等价。

**覆盖规则**：一次正常、串行运行会在 `--out` 下寻找现有最大 episode 编号，再从下一个编号开始。例如已有 `ep_0000` 到 `ep_0009`，再次采集将从 `ep_0010` 开始，不覆盖旧集。不要让两个采集进程同时使用同一个输出根目录；当前编号分配没有并发锁。

每集保存 `meta.json`、`decisions.jsonl`、`trajectory.jsonl`；保存图片时还会产生 `imgs/`。`--out` 的相对路径以当前工作目录为准，默认输出路径则固定在本项目下。`data/` 与 `output/` 都被 Git 忽略。

`--episodes` 必须至少为 1。默认最多 32 轮 brain 决策，修改 `config.py` 的 `MAX_DECISIONS` 可调整；没有 `--max-decisions` 参数。一轮 LLM 决策可以返回多个工具调用。只要有一集未成功，程序的退出码就是 1，这本身不表示 Python 崩溃，仍应看具体结束原因。

## 查看记录：inspect_data.py

- `python3 inspect_data.py`：列出默认 `data/episodes/` 中所有 episode 的统计和汇总；不运行仿真、不修改记录。
- `python3 inspect_data.py --root output/my_run`：改为查看 `output/my_run/` 中的记录；该参数可与下列查看或视频导出模式搭配。
- `python3 inspect_data.py --meta ep_0007`：打印该集的 `meta.json`，包括 brain、环境成功判定、结束原因、工具调用次数、帧数等。
- `python3 inspect_data.py --decisions ep_0007`：依次打印 brain 选择的工具和参数、文字说明、执行反馈，最后打印环境判定。
- `python3 inspect_data.py --frame ep_0007 --index 0`：打印 `trajectory.jsonl` 中第 0 条状态；索引从 0 开始，默认也是 0。它输出 JSON，不打开图片，且不等于第 0 次决策。
- `python3 inspect_data.py --replay ep_0007 --cam overhead`：将已保存的俯视相机 JPG 合成为 `data/episodes/ep_0007/replay_overhead.mp4`，不打开 GUI，也不重新渲染。
- `python3 inspect_data.py --replay ep_0007 --cam front_cam`：导出车头相机视频 `replay_front_cam.mp4`。`--cam` 默认 `overhead`，只能选择这集实际保存过的相机；当前默认录制 `front_cam` 和 `overhead`，没有录制 `side`。
- `python3 inspect_data.py --root output/my_run --meta ep_0000`：查看自定义输出根目录中的某一集。
- `python3 inspect_data.py --help`：显示完整参数，`-h` 等价。

`--meta`、`--decisions`、`--frame`、`--replay` 四种模式互斥，每次选一个。`--index` 用于 `--frame`，`--cam` 用于 `--replay`。

**覆盖规则**：查看模式只读。导出视频时，同一 episode、同一相机的 `replay_相机名.mp4` 会被重新写入；想保留旧视频需先改名或复制。`--no-images` 采集的记录不能用这种方式导出视频，可以使用下面的 `rerender.py`。

## GUI 播放历史轨迹：replay_gui.py

- `python3 replay_gui.py ep_0007`：读取默认根目录中该集的 `trajectory.jsonl`，正常速度播放；默认结束后保留窗口。
- `python3 replay_gui.py data/episodes/ep_0007`：直接指定 episode 目录。
- `python3 replay_gui.py data/episodes/ep_0007/trajectory.jsonl`：直接指定轨迹文件；无需先运行 `inspect_data.py` 或在 GUI 中手动导入。
- `python3 replay_gui.py ep_0000 --root output/my_run`：按 episode 名查找其他数据根目录；显式传入目录/JSONL 路径时直接使用该路径。
- `python3 replay_gui.py ep_0007 --speed 0.5`：半速播放；`--speed 2` 为两倍速，默认 1，必须大于 0。
- `python3 replay_gui.py ep_0007 --loop`：循环播放。
- `python3 replay_gui.py ep_0007 --exit-on-end`：播放结束自动关闭；与 `--loop` 不能同时使用。
- `python3 replay_gui.py ep_0007 --check`：只检查轨迹能否读取、维度与模型是否匹配、时间顺序和数值是否有效，不创建窗口。
- `python3 replay_gui.py ep_0007 --model models/scene_pickball.xml`：指定录制时对应的场景模型；默认就是这个 XML。相同维度不保证模型相同，需要自己保留正确模型。
- `python3 replay_gui.py --help`：显示完整参数，`-h` 等价。

窗口中：空格暂停/继续，`R` 从头重播，鼠标调整视角，关闭窗口退出。播放结束后按空格也会从头开始。

**输出和覆盖规则**：回放只读已有数据，不创建新 episode、不调用 brain/API、不执行原工具。它用 XML 场景和记录的姿态重建画面，因此没有 JPG 也可回放。当前不保存完整速度、接触和随机化参数，不能从这里精确续跑物理模拟；状态记录约 10 Hz，GUI 刷新更快也不会增加原始轨迹信息。

## 从轨迹重新渲染视频：rerender.py

- `python3 rerender.py ep_0007`：用轨迹重新渲染默认 `overhead` 相机，生成该集目录下的 `rerender_overhead.mp4`；默认 960×720、10 fps，不打开 GUI。
- `python3 rerender.py ep_0007 --cams overhead side front_cam`：分别导出多个相机的视频，文件名为 `rerender_相机名.mp4`；相机必须存在于当前场景模型中，不要求采集时存过该相机的 JPG。
- `python3 rerender.py ep_0007 --width 640 --height 480`：改变渲染分辨率；宽高必须为正整数，并受模型离屏缓冲区大小限制。
- `python3 rerender.py ep_0007 --fps 10`：设置视频编码帧率；当前实现逐条轨迹写一帧，改变 fps 会改变播放速度，不会插值生成新状态。
- `python3 rerender.py ep_0000 --root output/my_run`：从其他数据根目录读取轨迹。
- `python3 rerender.py output/my_run/ep_0000/trajectory.jsonl`：直接传入轨迹文件；也支持 episode 目录。
- `python3 rerender.py ep_0007 --model models/scene_pickball.xml`：指定对应的场景 XML，默认使用项目的捡球场景。
- `python3 rerender.py --help`：显示完整参数，`-h` 等价。

**覆盖规则**：同一集目录下同名 `rerender_相机名.mp4` 会被覆盖，原 JSONL 和 JPG 不变。这个入口没有 `--out` 参数；输出位置固定为轨迹所在目录。渲染仍需要可用的图形后端，WSL 可以使用下面的软件 EGL 配置。

## WSL 无窗口运行与故障判断

- `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=llvmpipe python3 run_collect.py --brain scripted --episodes 1 --out output/software-check`：在 WSL 使用软件 EGL 渲染采集一集，不创建窗口；这是本机已验证的无窗口运行方式，速度可能较慢。
- `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=llvmpipe python3 rerender.py ep_0007 --cams overhead`：在 WSL 使用软件 EGL 导出历史轨迹的视频。
- `MUJOCO_GL=glfw python3 replay_gui.py ep_0007`：通过 WSLg 打开历史回放窗口；需要 WSLg 正常，当前本机尚未稳定验证通过。
- `MUJOCO_GL=glfw python3 run_collect.py --brain scripted --episodes 1 --gui --keep-open`：通过 WSLg 实时采集；同样依赖正常的窗口服务。

遇到 `python: command not found`：先激活虚拟环境，或显式使用 `.venv/bin/python`。仅把命令改成系统 `python3` 不会自动使用项目依赖。

遇到 `ModuleNotFoundError: No module named 'mujoco'`：先检查 `sys.executable`。截图中未激活环境时使用的是系统 Python；若已经是项目 Python，再安装 `requirements.txt`。

停在 `正在创建 GUI 窗口`：尚未进入采集/播放；检查 GLFW/WSLg，优先使用本机已验证的 Windows GUI。只有看到 `GUI 已打开` 才表示窗口创建成功。`replay_gui.py --check` 能通过并不代表图形环境可用。

停在 `LLM→模型`：这时可能是在等待 API；与窗口创建阻塞不同。当前单请求超时为 `config.LLM_TIMEOUT_S=180` 秒，重试会增加总等待时间。

`--no-images` 仍需要渲染 brain 的车头观测，不能用来绕过图形初始化。软件 EGL 用于无窗口渲染，也不等于修好了 WSLg 窗口服务。

## 其他入口

- `python3 compose_model.py`：组合底盘和机械臂模型并自检，**覆盖** `models/mobile_manip.generated.xml`；修改模型源文件后才需要重新生成。除 `-h/--help` 外没有自定义命令行参数。
- `python3 -m mujoco.viewer --mjcf=models/scene_pickball.xml`：只打开场景，不运行项目 brain、不采集、不回放历史轨迹；WSL 仍需要 WSLg 正常。
- `./run --brain scripted --episodes 3 --out output/quick`：Linux/WSL 的快捷入口；调用项目 `.venv/bin/python`，环境不存在时用 `uv` 创建并安装依赖，组合模型不存在时生成，随后运行 `run_collect.py`。它透传采集参数；默认 scripted、3 集、seed 0。已有环境不会自动更新依赖。
- `python3 -m pytest tests/ -v`：运行全部自动测试；不是采集任务。渲染测试需要可用图形后端。
- `MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=llvmpipe python3 -m pytest tests/ -v`：本机 WSL 下使用软件渲染运行测试。
- `python3 termimg_test.py`：仅在 Linux/WSL 的交互式终端中检查 Kitty 内联图片协议，并渲染一张车头画面；没有命令行参数，需要可用离屏渲染。Windows 原生 Python 不支持它使用的 `termios`；普通终端不能显示内联图片不影响独立 GUI。

## Brain 到底能看到什么

Python 层的 observation 是一个字典，包含 `images.front`（车头 RGB 像素数组）、`base_pose`（底盘 x/y/yaw）、`arm_qpos`（两个臂关节角）、`finger`（开度 0～1）、`tcp_pos`（夹爪末端位置）；采集循环另加入 `holding`（是否持球）。此外会传入任务文字、工具定义和动作历史。

- `scripted`：没有神经网络或远程模型；当前实现通过 HAL 再取车头图片和本体状态，做本地颜色检测、距离/方位估计，再用规则选择动作。它不是只读一份坐标 JSON。
- `openai`（当前 DeepSeek）：发送车头图片、任务、可用工具、最近最多 12 次动作及反馈；还发送程序从图片估算的球/箱方位与距离，以及底盘位姿、臂角、手指开合、是否持球等文字。
- `anthropic`：同样发送车头图片、任务、工具和动作历史；当前文字观测包含底盘位姿、夹爪开度和 TCP 坐标，与 OpenAI 适配器的文字摘要尚不完全一致。

图片被编码成 JPEG/base64，放进 API 的 JSON 请求中的图片块。**传输格式是 JSON，不代表里面只有文字或坐标；视觉模型确实会接收图片。** 默认 brain 不接收俯视/侧视调试图，也不接收所有球的仿真真值位置。

`trajectory.jsonl` 保存的全场景 `qpos` 用于回放，其中包含的信息比 brain 的 observation 更多；这个文件不会作为整份观测直接发给 LLM。

这里有两个不同含义的“模型”：`.env` 的 `OPENAI_MODEL=deepseek-flash` 是远程 AI 模型，通过 `--brain openai` 调用；回放命令的 `--model models/scene_pickball.xml` 是 MuJoCo 的场景/机器人/物理参数文件。当前项目不下载或训练 DeepSeek 权重，也没有实现 ACT 训练。

API 配置只放 `.env`；仓库中的 `.env.example` 只含占位值。`max_tokens`、`thinking`、`reasoning_effort`、`stream` 是 `.env` 请求设置，不是 `run_collect.py` 的命令行参数。API 用量和实际耗时目前尚未自动写进 episode 日志。
