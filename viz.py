"""渲染封装: EGL 离屏相机组 + GUI viewer 包装."""

from __future__ import annotations

import numpy as np
import mujoco

import config


class CameraRig:
    """每个观测相机一个离屏 Renderer (EGL)."""

    def __init__(self, model: mujoco.MjModel,
                 cams: list[str] | None = None,
                 width: int = 640, height: int = 480):
        self.cams = cams or config.OBS_CAMS
        self._renderers = {c: mujoco.Renderer(model, height=height, width=width)
                           for c in self.cams}
        self._cam_ids = {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
                         for c in self.cams}

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for c, r in self._renderers.items():
            r.update_scene(data, camera=self._cam_ids[c])
            out[c] = r.render()
        return out

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()


class GuiViewer:
    """launch_passive 包装: 主循环跑物理, 每控制拍 sync()."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(model, data)
        v = self._viewer
        v.cam.lookat[:] = [0.55, 0.0, 0.55]
        v.cam.distance = 2.2
        v.cam.azimuth = 130.0
        v.cam.elevation = -25.0

    def sync(self) -> None:
        if self._viewer.is_running():
            self._viewer.sync()

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def close(self) -> None:
        try:
            self._viewer.close()
        except Exception:
            pass

    def __enter__(self) -> "GuiViewer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
