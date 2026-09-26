"""车头相机视觉感知: 网球/收纳箱检测 + 地面投影测距 (纯 numpy+scipy).

输入一张 RGB 图 + 相机标定, 输出检测列表(方位角/距离估算).
不依赖仿真 —— 真机上同一份代码处理 USB 相机帧.

注意: 相机光轴俯角 30°, 地平线在画面上部; 距离由"地面投影"估算
(像素行 → 俯角 → 三角函数), 球用质心行, 箱用 blob 底行(地面落点).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import config


@dataclass
class Detection:
    kind: str            # "ball" | "bin"
    area: int            # 像素面积
    u: float             # 质心列
    v: float             # 质心行
    bearing_deg: float   # 相对车头指向 (左正右负)
    dist_m: float | None  # 相对底盘中心的地面距离估计; 视线过高时 None


def _rgb_to_hsv(rgb: np.ndarray):
    rgb = rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    diff = mx - mn + 1e-8
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = 60.0 * (((g - b) / diff)[m] % 6)
    m = mx == g
    h[m] = 60.0 * ((b - r) / diff)[m] + 120.0
    m = mx == b
    h[m] = 60.0 * ((r - g) / diff)[m] + 240.0
    s = diff / (mx + 1e-8)
    return h, s, mx


def _ball_mask(rgb01, v):
    """网球: 黄绿色 (R≈0.8G 且 R<G, G≫B). 例: rgb(194,245,20).
    R<G 排除橙色机械臂(R>G)进入镜头视野造成的误检."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return ((r > g * 0.68) & (r < g * 1.02) & (g > b * 2.0) & (v > config.BALL_VAL))


def _bin_mask(rgb01, v):
    """收纳箱: 洋红色 (R,G 双高). 场景中唯一洋红色物体, 与球影永不存在色域冲突."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return ((r > g * 1.8) & (b > g * 1.8) & (v > config.BIN_VAL))


def _blobs(mask, min_area, bottom_of=None):
    """连通域 → [(area, u, v_centroid, v_bottom)] 按面积降序, 最多 6 个."""
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    areas = ndimage.sum(mask, labels, index=range(1, n + 1))
    out = []
    for idx in np.argsort(areas)[::-1][:6]:
        area = int(areas[idx])
        if area < min_area:
            break
        sel = labels == (idx + 1)
        cy, cx = ndimage.center_of_mass(mask, labels, idx + 1)
        v_bottom = float(np.where(sel.any(axis=1))[0].max())
        out.append((area, float(cx), float(cy), v_bottom))
    return out


def _bearing(u: float) -> float:
    W, H = config.CAM_FRONT_RES
    fx = (H / 2) / np.tan(np.radians(config.CAM_FRONT_FOVY) / 2)
    return -float(np.degrees(np.arctan((u - W / 2) / fx)))   # 画面右=车右=负


def _ground_dist(v: float, obj_height: float = 0.0) -> float | None:
    """像素行 → 地面距离. v 取物体接触地面的行最准; 近水平线返回 None."""
    W, H = config.CAM_FRONT_RES
    fy = (H / 2) / np.tan(np.radians(config.CAM_FRONT_FOVY) / 2)
    beta = np.arctan((v - H / 2) / fy)
    depression = np.radians(config.CAM_PITCH_DEG) + beta
    if depression <= np.radians(4.0):
        return None
    cam_h = config.CAM_HEIGHT - obj_height
    ground_from_cam = cam_h / np.tan(depression)
    return float(max(ground_from_cam - config.CAM_FORWARD_OFFSET, 0.05))


def _gripper_tip(rgb01, v):
    """检测夹爪末端: 取暖色 blob 的最下缘(臂指向地面, 最下缘=指尖).
    返回 (u, v) 像素坐标, 不可见返回 None."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    # 暖色: R > G*1.3 且 R > B*1.5 (橙色臂 + 红色指垫)
    mask = (r > g * 1.3) & (r > b * 1.5) & (v > 0.25)
    if not mask.any():
        return None
    from scipy import ndimage
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    # 合并所有暖色 blob, 取整体最下缘的中心列(指尖位置)
    all_mask = labels > 0
    rows = np.where(all_mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    v_bottom = int(rows.max())               # 最下缘行 = 指尖
    cols = np.where(all_mask[v_bottom - 3:v_bottom + 1].any(axis=0))[0]
    u_tip = int(cols.mean()) if len(cols) > 0 else None
    if u_tip is None:
        return None
    return u_tip, v_bottom


def analyze(rgb: np.ndarray) -> list[Detection]:
    rgb01 = rgb.astype(np.float32) / 255.0
    _h, _s, v = _rgb_to_hsv(rgb)
    dets: list[Detection] = []
    ball_bboxes = []
    # 球: 质心行 + 球心高度 0.06 修正
    for area, u, vc, _vb in _blobs(_ball_mask(rgb01, v), config.BALL_MIN_AREA):
        dets.append(Detection("ball", area, u, vc, _bearing(u),
                              _ground_dist(vc, obj_height=config.BALL_R)))
        # 记录球包围盒(供 bin 掩码剔除: 手里球的影子会与箱连通成一个大blob)
        ball_bboxes.append((u, vc, area))
    # 箱: 方位用质心列, 距离用底行(地面接触).
    # 先剔除球blob区域(扩20px)—— 手中球的阴影会把 bin 掩码与箱连成一片,
    # 质心被拉向球方向且不随转向移动, 导致对准死循环
    bin_mask = _bin_mask(rgb01, v)
    if ball_bboxes:
        H, W = bin_mask.shape
        for u, vc, area in ball_bboxes:
            r = int(np.sqrt(area)) + 20
            y0, y1 = max(0, int(vc) - r), min(H, int(vc) + r)
            x0, x1 = max(0, int(u) - r), min(W, int(u) + r)
            bin_mask[y0:y1, x0:x1] = False
    for area, u, vc, vb in _blobs(bin_mask, config.BIN_MIN_AREA):
        dets.append(Detection("bin", area, u, vc, _bearing(u), _ground_dist(vb)))
    return dets


def sense(robot) -> dict:
    """机器人主动感知: 拍一帧车头相机 + 本体感觉. 大脑/脚本共用, 无真值."""
    img = robot.front_image()
    dets = analyze(img)
    grip = _gripper_tip(img.astype(np.float32) / 255.0,
                        _rgb_to_hsv(img)[2])
    holding = robot.attached_object() is not None
    if holding:
        # 持球过滤: ①手里的球从 ball 检测滤掉; ②球的阴影漏进 bin 掩码形成
        # ~0.26m 假箱 —— 0.38m 内 bin 丢弃(终点策略下不需要近距箱检测)
        dets = [d for d in dets
                if (d.kind != "ball" or d.dist_m is None or d.dist_m > 0.45)
                and (d.kind != "bin" or d.area >= 2500
                     or (d.dist_m is not None and d.dist_m > 0.38))]
    return {
        "image": img,
        "detections": dets,
        "gripper_tip": grip,
        "base_pose": robot.base_pose,
        "arm_q": robot.arm_q,
        "tcp_pos": robot.tcp_pos,
        "finger_open": robot.finger_opening,
        "holding": holding,
    }


def gripper_hint(s: dict) -> str:
    """夹爪可见且有近球时的对齐提示(前后 dv + 左右 du); 无参考返回空串.

    以夹爪末端(暖色 blob 最下缘)为参照: 垂直像素差 dv=前后(画面里越低=离车越近),
    水平像素差 du=左右. 仅在臂处于 reach 位(指尖贴地)时几何上成立."""
    grip = s.get("gripper_tip")
    if grip is None:
        return ""
    balls = [d for d in s["detections"] if d.kind == "ball"]
    near = [d for d in balls if d.dist_m is not None and d.dist_m < 0.8]
    if not near:
        return ""
    # 正在抓的球 = 横向最贴近夹爪的那个(按 dist 选会误选侧面的近球)
    nb = min(near, key=lambda d: abs(d.u - grip[0]))
    W = config.CAM_FRONT_RES[0]
    du = nb.u - grip[0]                      # 正=球在夹爪右侧
    offset_deg = -du / W * 75                # 画面右=车右=负
    dv = nb.v - grip[1]                      # 正=球比指尖低=比夹爪近
    parts = []
    if abs(dv) >= 5:                         # 标定: dv 过零≈0.54m, ±5px≈物理成功窗(实测7.4mm/px)
        where = ("比夹爪近(车太靠前, 应先 stow 再 back)" if dv > 0
                 else "比夹爪远(车太靠后, 应先 stow 再 forward)")
        parts.append(f"前后向: 球{where}, 偏差≈{abs(dv) * 0.0074:.2f}m")
    if abs(du) < 30:
        parts.append(f"左右向: 球在夹爪{'正下方' if abs(du) < 15 else '近旁'}"
                     f"(偏移{du:+.0f}px≈{offset_deg:+.0f}°), 可闭合")
    else:
        side = "右" if du > 0 else "左"
        parts.append(f"左右向: 球在夹爪{side}侧{abs(du):.0f}px(≈{abs(offset_deg):.0f}°), "
                     f"需{'turn_right' if du > 0 else 'turn_left'}")
    return "; ".join(parts)


def sense_text(s: dict) -> str:
    balls = [d for d in s["detections"] if d.kind == "ball"]
    bins = [d for d in s["detections"] if d.kind == "bin"]
    lines = ["车头相机视觉感知结果:"]

    # 夹爪参照: 近距离对齐的关键反馈(臂 reach 位时夹爪在画面下方可见)
    hint = gripper_hint(s)
    if hint:
        lines.append(f"  ★ 夹爪参照: {hint}")
    if balls:
        lines.append(f"  检测到 {len(balls)} 个网球:")
        for d in balls:
            dist = f"{d.dist_m:.2f}m" if d.dist_m is not None else "超出可信范围"
            lines.append(f"    方位 {d.bearing_deg:+.0f}° (左正右负), 距离约 {dist}")
    else:
        lines.append("  视野内没有网球")
    if bins:
        d = bins[0]
        dist = f"{d.dist_m:.2f}m" if d.dist_m is not None else "超出可信范围"
        lines.append(f"  收纳箱: 方位 {d.bearing_deg:+.0f}°, 距离约 {dist}")
    else:
        lines.append("  视野内没有收纳箱")
    bp = s["base_pose"]
    lines.append(f"本体状态: 里程计位姿 x={bp[0]:.2f} y={bp[1]:.2f} 朝向{np.degrees(bp[2]):.0f}°, "
                 f"臂角[{np.degrees(s['arm_q'][0]):.0f}°,{np.degrees(s['arm_q'][1]):.0f}°], "
                 f"手指{'张开' if s['finger_open'] > 0.5 else '闭合'}, "
                 f"夹爪{'持有球' if s['holding'] else '空'}")
    return "\n".join(lines)
