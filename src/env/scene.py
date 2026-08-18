"""Panda 场景构建 —— Day0 脚手架。

用 MjSpec 在官方 Franka Panda 模型上程序化添加:
  - TCP site (两指抓取中心, Franka 官方 flange->TCP 偏移 z=0.1034)
  - wrist 相机 (挂在 hand 上)
  - front 相机 (第三人称, 挂在 world)

为什么用 MjSpec 而不是改 XML:
  官方 panda.xml 由 include 组装, <include> 无法向已定义的 body 内注入元素。
  MjSpec 是 MuJoCo 3.x 的程序化模型 API, 改完 compile() 即可, 不污染上游文件。

TCP 定义说明 (Day1-2 会反复用到):
  MuJoCo 的 `hand` body 原点 = Franka flange (link8 法兰面)。
  官方两指夹爪的抓取中心在法兰沿 +z 方向 0.1034m 处。
  finger body 在 z=0.0584, 指尖再往前约 0.045 -> 0.1034 是标准 TCP。
  注意 hand 带 quat="0 0 0 1" (绕 z 转 180°), 所以 TCP 的姿态不等于 flange 姿态。
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# scene.py 在 <root>/src/env/, 所以项目根是 parents[2]
# 可用 MENAGERIE 环境变量覆盖 (env.sh 里已导出)
import os

MENAGERIE = Path(
    os.environ.get("MENAGERIE", Path(__file__).resolve().parents[2] / "menagerie")
) / "franka_emika_panda"

# Franka 官方: flange -> 两指抓取中心
TCP_OFFSET_Z = 0.1034


def build_spec(
    *,
    add_tcp: bool = True,
    add_cameras: bool = True,
    img_size: int = 128,
    debug_viz: bool = False,
) -> mujoco.MjSpec:
    """加载官方 scene.xml 并注入 TCP site + 双相机。

    debug_viz=False 时 site 的 alpha 设为 0 —— 关键的防泄漏措施:
    site 标记若渲进观测图像, 策略会直接盯着红点回归位置, 学不到真东西,
    而且真机上没有这个标记, sim2real 直接崩。site 不可见不影响运动学计算。
    """
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "scene.xml"))

    hand = spec.body("hand")
    site_alpha = 0.6 if debug_viz else 0.0

    if add_tcp:
        # site 挂在 hand 上, 沿 hand 局部 +z 偏移到抓取中心
        site = hand.add_site()
        site.name = "tcp"
        site.pos = [0.0, 0.0, TCP_OFFSET_Z]
        site.size = [0.008, 0.008, 0.008]
        site.rgba = [1.0, 0.2, 0.2, site_alpha]

        # 法兰参考 site, Day1 验证 FK 链时用来分段对齐
        flange = hand.add_site()
        flange.name = "flange"
        flange.pos = [0.0, 0.0, 0.0]
        flange.size = [0.006, 0.006, 0.006]
        flange.rgba = [0.2, 0.6, 1.0, site_alpha]

    if add_cameras:
        # 腕部相机 —— 朝向必须在 hand 局部系里指定, 不能用世界系的 _look_at。
        #
        # 实测(Day0): 夹爪朝下时 hand 局部 z 轴指向世界 -z, 即局部 +z == 抓取方向。
        # MuJoCo 相机沿自身 -z 看, 所以要让视线朝局部 +z, 相机需绕 x 轴转 180°:
        #   quat = (0, 1, 0, 0)  ->  cam_z = -hand_z, cam 视线 = -cam_z = +hand_z ✓
        # 再侧偏 y 并稍微后撤(-z), 避免两根手指占满画面(实测过, 正前方会被挡死)。
        wrist = hand.add_camera()
        wrist.name = "wrist"
        wrist.pos = [0.0, -0.05, -0.01]
        wrist.quat = _quat_mul(
            (0.0, 1.0, 0.0, 0.0),  # 绕 x 转 180°: 视线转到局部 +z
            _quat_about_x(np.deg2rad(-22.0)),  # 再俯仰一点, 把抓取区收进画面中心
        )
        wrist.fovy = 75.0

        front = spec.worldbody.add_camera()
        front.name = "front"
        front.pos = [1.05, 0.0, 0.75]
        front.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        front.quat = _look_at(eye=(1.05, 0.0, 0.75), target=(0.45, 0.0, 0.15))
        front.fovy = 58.0

    spec.visual.global_.offwidth = max(img_size, 640)
    spec.visual.global_.offheight = max(img_size, 480)
    return spec


def _quat_about_x(angle: float) -> tuple[float, float, float, float]:
    """绕 x 轴旋转 angle 弧度的四元数 (w, x, y, z)。"""
    return (float(np.cos(angle / 2)), float(np.sin(angle / 2)), 0.0, 0.0)


def _quat_mul(a, b) -> list[float]:
    """四元数乘法 (w,x,y,z), 返回 a∘b。"""
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out.tolist()


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """生成让相机 -z 轴指向 target 的四元数 (MuJoCo 相机看向自身 -z)。"""

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    # 相机看向 -z, 故 z 轴 = -forward
    z = -forward
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-8:  # up 与视线平行, 换一个 up
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)

    R = np.column_stack([x, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat.tolist()


def build_model(**kwargs) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """便捷入口: 返回编译好的 (model, data)。"""
    spec = build_spec(**kwargs)
    model = spec.compile()
    return model, mujoco.MjData(model)


if __name__ == "__main__":
    m, d = build_model()
    mujoco.mj_forward(m, d)
    print(f"MuJoCo {mujoco.__version__}")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nsite={m.nsite} ncam={m.ncam}")
    for i in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"  site[{i}] {name:8s} xpos={d.site_xpos[i]}")
    for i in range(m.ncam):
        print(f"  cam[{i}]  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)}")
