#!/usr/bin/env python3
"""终端图像显示自检 + 协议探测. 在 Ghostty 里直接运行, 不要接管道."""
import os, sys, tty, termios, select, time

os.environ.setdefault("MUJOCO_GL", "egl")
pass

print(f"TERM = {os.environ.get('TERM')}")
print(f"TERM_PROGRAM = {os.environ.get('TERM_PROGRAM', '(未设置)')}")

# ---- 1. Kitty 图形协议探测: 发查询, 看终端是否应答 ----
def query_graphics_support(timeout=0.5):
    """发送 Kitty graphics 查询序列, 读应答."""
    if not sys.stdin.isatty():
        return None, "stdin 不是 tty(在管道里运行?)"
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        # a=q 查询; i=32 请求支持信息
        os.write(sys.stdout.fileno(), b"\x1b_Gi=31,a=q;\x1b\\")
        os.write(sys.stdout.fileno(), b"\x1b_Gi=32,a=q;\x1b\\")
        time.sleep(0.05)
        resp = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                chunk = os.read(sys.stdin.fileno(), 4096)
                if not chunk:
                    break
                resp += chunk
                deadline = time.time() + 0.1   # 有数据则稍等后续
        return resp, None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

resp, err = query_graphics_support()
if err:
    print(f"[探测失败] {err}")
elif resp:
    print(f"[探测] 终端应答了图形协议查询 ({len(resp)} 字节) → 支持 Kitty graphics ✓")
    print(f"       应答片段: {resp[:80]!r}")
else:
    print("[探测] 终端无应答 → 不支持 Kitty 图形协议(或被中间层拦截) ✗")

# ---- 2. 实际显示一帧 ----
from brains.llm_common import term_show_image
from sim.env import MobileManipEnv

print("\n↓ 下面应显示车头相机画面 (若上面探测为 ✗ 则只有文字):")
env = MobileManipEnv()
env.reset(seed=0)
term_show_image(env.hal.front_image(), label="[车头相机 front_cam]")
print("↑ 有图 = 一切正常, ./run 里也能显示\n")
env.hal.close()

print("""若仍无图:
  1. Ghostty 版本: ghostty +version (需 >= 1.0.0)
  2. Ghostty 配置: 检查 image-storage 是否被设为 disabled
  3. tmux/screen 会拦截: 需 passthrough 配置, 或退出后直接跑
  4. SSH 远程: 显示发生在本地终端, 确认本地也在 Ghostty 里""")
