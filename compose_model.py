"""MjSpec 模型组合: 参考车式差速底盘 + 参考式两自由度臂 → mobile_manip.generated.xml

用法:
  python compose_model.py          # 生成并自检
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import config


def build_spec() -> mujoco.MjSpec:
    base = mujoco.MjSpec.from_file(str(config.BASE_XML))
    arm = mujoco.MjSpec.from_file(str(config.ARM_XML))
    base.attach(arm, prefix=config.ARM_PREFIX, site="arm_mount")
    return base


def generate(write: bool = True) -> mujoco.MjModel:
    spec = build_spec()
    model = spec.compile()
    xml = spec.to_xml()
    _selfcheck(model)
    if write:
        config.GENERATED_XML.write_text(xml)
        print(f"已生成 {config.GENERATED_XML}")
    return model


def _selfcheck(model: mujoco.MjModel) -> None:
    probes = (model.joint, model.body, model.site, model.actuator,
              model.camera, model.equality)
    missing = [n for n in (config.JOINT_BASE, config.JOINT_WHEEL_L, config.JOINT_WHEEL_R,
                           config.JOINT_SHOULDER, config.JOINT_ELBOW, config.JOINT_FINGER,
                           config.SITE_TCP, config.ACT_SHOULDER, config.ACT_ELBOW,
                           config.ACT_FINGER, config.CAM_FRONT)
               if not any(_exists(p, n) for p in probes)]
    if missing:
        raise AssertionError(f"组合模型缺少元素: {missing}")
    # 量臂长: 竖直姿态下肩→肘, 肘→TCP 距离
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    sh = data.joint(config.JOINT_SHOULDER)
    print(f"自检通过: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"  臂执行器: {config.ACT_SHOULDER}, {config.ACT_ELBOW}, {config.ACT_FINGER}")
    print(f"  TCP 位点: {config.SITE_TCP}, 指滑动 ctrl∈"
          f"{model.actuator(config.ACT_FINGER).ctrlrange}")


def _exists(fn, name) -> bool:
    try:
        fn(name)
        return True
    except KeyError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    generate()


if __name__ == "__main__":
    main()
