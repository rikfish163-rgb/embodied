"""抓取-放置任务场景 —— Day0 脚手架。

在 Panda 基础场景上加:
  - 工作台面 (薄板, 避免机械臂贴地)
  - 目标立方体 (freejoint, 位置/朝向/颜色可随机化)
  - 放置容器 (开口盒, 四壁 + 底)
  - 干扰立方体 (可选, 给 eval 的 L2 扰动档用)

qpos 布局 (加了 freejoint 后会变, 必须显式记住, 这是最容易错的地方):
    qpos[0:7]   7 个臂关节
    qpos[7:9]   2 个夹爪指关节
    qpos[9:16]  目标立方体 freejoint (xyz + wxyz 四元数)
    qpos[16:]   干扰物 freejoint, 每个 7 维

尺寸依据 (来自 Day0 实测):
    夹爪单指行程 0-0.04m -> 最大开口 8cm, 所以立方体边长取 4cm(半边 0.02) 有余量
    夹爪执行器 actuator8 ctrl 范围 [0, 255]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import build_spec

# --- 几何常量 ---
TABLE_H = 0.02  # 台面厚度, 顶面 z = TABLE_H
CUBE_HALF = 0.02  # 立方体半边长 -> 4cm 立方体
BOX_INNER = 0.06  # 容器内半宽
BOX_WALL = 0.006  # 容器壁厚
BOX_H = 0.05  # 容器壁高
FINGER_OPEN_QPOS = 0.04
GRIPPER_CTRL_MAX = 255.0

# --- 随机化范围 (这就是"训练分布", eval 的 L1 外推档要落在它之外) ---
CUBE_X = (0.42, 0.60)
CUBE_Y = (-0.16, 0.16)
BOX_X = (0.44, 0.56)
BOX_Y = (0.26, 0.36)

COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "blue": (0.15, 0.30, 0.85, 1.0),
    "green": (0.15, 0.70, 0.25, 1.0),
}


@dataclass
class TaskConfig:
    n_distractors: int = 0
    img_size: int = 128
    debug_viz: bool = False
    colors: tuple[str, ...] = field(default_factory=lambda: ("red",))
    control_hz: float = 20.0
    success_hold_s: float = 1.0
    success_z_tolerance: float = 0.01


def build_task_spec(cfg: TaskConfig) -> mujoco.MjSpec:
    spec = build_spec(img_size=cfg.img_size, debug_viz=cfg.debug_viz)
    wb = spec.worldbody

    # 台面: 薄板, 覆盖工作区
    table = wb.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [0.40, 0.40, TABLE_H / 2]
    table.pos = [0.50, 0.0, TABLE_H / 2]
    table.rgba = [0.72, 0.66, 0.55, 1.0]

    # 目标立方体: freejoint 使其可自由落体/被抓起
    cube = wb.add_body()
    cube.name = "cube"
    cube.pos = [0.50, 0.0, TABLE_H + CUBE_HALF]
    cube.add_freejoint()
    cg = cube.add_geom()
    cg.name = "cube_geom"
    cg.type = mujoco.mjtGeom.mjGEOM_BOX
    cg.size = [CUBE_HALF] * 3
    cg.rgba = list(COLORS["red"])
    cg.mass = 0.05
    cg.friction = [1.0, 0.02, 0.001]  # 高滑动摩擦, 否则夹爪夹不住

    # 干扰立方体
    for i in range(cfg.n_distractors):
        b = wb.add_body()
        b.name = f"distractor{i}"
        b.pos = [0.50, 0.0, TABLE_H + CUBE_HALF]
        b.add_freejoint()
        g = b.add_geom()
        g.name = f"distractor{i}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [CUBE_HALF] * 3
        g.rgba = list(COLORS["blue"])
        g.mass = 0.05
        g.friction = [1.0, 0.02, 0.001]

    # 放置容器: 静态 body, 四壁 + 底
    box = wb.add_body()
    box.name = "box"
    box.pos = [0.50, 0.30, TABLE_H]
    base = box.add_geom()
    base.name = "box_base"
    base.type = mujoco.mjtGeom.mjGEOM_BOX
    base.size = [BOX_INNER, BOX_INNER, BOX_WALL / 2]
    base.pos = [0, 0, BOX_WALL / 2]
    base.rgba = [0.35, 0.35, 0.40, 1.0]
    for name, (dx, dy, sx, sy) in {
        "xp": (BOX_INNER, 0, BOX_WALL / 2, BOX_INNER),
        "xn": (-BOX_INNER, 0, BOX_WALL / 2, BOX_INNER),
        "yp": (0, BOX_INNER, BOX_INNER, BOX_WALL / 2),
        "yn": (0, -BOX_INNER, BOX_INNER, BOX_WALL / 2),
    }.items():
        w = box.add_geom()
        w.name = f"box_{name}"
        w.type = mujoco.mjtGeom.mjGEOM_BOX
        w.size = [sx, sy, BOX_H / 2]
        w.pos = [dx, dy, BOX_H / 2]
        w.rgba = [0.40, 0.40, 0.45, 1.0]

    return spec


class PickPlace:
    """抓取-放置环境，只定义任务契约，不包含专家或学习策略。"""

    def __init__(self, cfg: TaskConfig | None = None):
        self.cfg = cfg or TaskConfig()
        self.spec = build_task_spec(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        if self.cfg.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.cfg.success_hold_s <= 0:
            raise ValueError("success_hold_s must be positive")
        if self.cfg.success_z_tolerance <= 0:
            raise ValueError("success_z_tolerance must be positive")

        control_period = 1.0 / self.cfg.control_hz
        self.steps_per_control = round(control_period / self.model.opt.timestep)
        if not np.isclose(
            self.steps_per_control * self.model.opt.timestep,
            control_period,
            atol=1e-12,
        ):
            raise ValueError(
                "control_hz must map to an integer number of MuJoCo physics steps"
            )
        self.required_success_steps = math.ceil(
            self.cfg.success_hold_s / self.model.opt.timestep
        )

        self.nq_arm = 7
        self.qadr_fingers = 7
        # freejoint 无名字, 用 body->joint 反查 qpos 地址, 不硬编码。
        # (实测布局确为 fingers@7 / cube@9 / distractor@16,23, 但下面的断言保证
        #  以后往场景里加东西时不会静默错位 —— 这类错位极难调试。)
        self.qadr_cube = self._free_qadr("cube")
        self.qadr_distractors = [
            self._free_qadr(f"distractor{i}") for i in range(self.cfg.n_distractors)
        ]
        assert self.qadr_fingers == 7 and self.qadr_cube == 9, (
            f"qpos 布局意外: fingers@{self.qadr_fingers} cube@{self.qadr_cube}"
        )

        self.sid_tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.bid_cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.bid_box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")

        self.box_inner = BOX_INNER
        self.box_height = BOX_H

        self._renderer: mujoco.Renderer | None = None
        self.home_q = np.array([0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785])
        self._placement_hold_steps = 0
        self._success_confirmed = False

    def _free_qadr(self, body_name: str) -> int:
        """查某 body 的 freejoint 在 qpos 中的起始地址。"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"找不到 body: {body_name}")
        jadr = self.model.body_jntadr[bid]
        if jadr < 0 or self.model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body {body_name} 没有 freejoint")
        return int(self.model.jnt_qposadr[jadr])

    # --- 随机化与复位 ---
    def reset(self, rng: np.random.Generator, *, ranges: dict | None = None) -> dict:
        """ranges 可覆盖默认随机化区间 —— eval 的 L1 外推档就靠它。"""
        r = {"cube_x": CUBE_X, "cube_y": CUBE_Y, "box_x": BOX_X, "box_y": BOX_Y}
        if ranges:
            r.update(ranges)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.nq_arm] = self.home_q
        self.data.qpos[self.qadr_fingers : self.qadr_fingers + 2] = (
            FINGER_OPEN_QPOS  # 张开
        )

        cx = rng.uniform(*r["cube_x"])
        cy = rng.uniform(*r["cube_y"])
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self._set_free(self.qadr_cube, (cx, cy, TABLE_H + CUBE_HALF), yaw)

        for i, adr in enumerate(self.qadr_distractors):
            while True:  # 不与目标重叠
                dx, dy = rng.uniform(*r["cube_x"]), rng.uniform(*r["cube_y"])
                if np.hypot(dx - cx, dy - cy) > 4 * CUBE_HALF:
                    break
            self._set_free(
                adr, (dx, dy, TABLE_H + CUBE_HALF), rng.uniform(-np.pi, np.pi)
            )

        bx, by = rng.uniform(*r["box_x"]), rng.uniform(*r["box_y"])
        self.model.body_pos[self.bid_box] = [bx, by, TABLE_H]

        self.data.ctrl[:] = 0.0
        self.data.ctrl[: self.nq_arm] = self.home_q
        self.data.ctrl[7] = GRIPPER_CTRL_MAX  # 夹爪张开
        self._reset_success_tracker()
        mujoco.mj_forward(self.model, self.data)
        return {"cube_xy": (cx, cy), "cube_yaw": yaw, "box_xy": (bx, by)}

    def _set_free(self, adr: int, pos, yaw: float) -> None:
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # --- 状态查询 ---
    @property
    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.sid_tcp].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.bid_cube].copy()

    @property
    def gripper_state(self) -> float:
        """返回 [0, 1] 夹爪开度；0 为闭合，1 为完全张开。"""

        fingers = self.data.qpos[self.qadr_fingers : self.qadr_fingers + 2]
        return float(np.clip(np.mean(fingers) / FINGER_OPEN_QPOS, 0.0, 1.0))

    def observe(self) -> dict[str, np.ndarray]:
        """返回策略唯一允许读取的双相机图像与 8 维本体状态。"""

        state = np.concatenate(
            [self.data.qpos[: self.nq_arm].copy(), [self.gripper_state]]
        )
        return {
            "observation.images.front": self.render("front"),
            "observation.images.wrist": self.render("wrist"),
            "observation.state": state,
        }

    def step(
        self,
        action: np.ndarray,
        *,
        physics_steps: int | None = None,
    ) -> dict[str, float | int | bool]:
        """执行 8 维关节目标动作并更新连续成功判定。

        ``action[:7]`` 是 7 个臂关节的位置目标（弧度）；``action[7]``
        是归一化夹爪开度，0 为闭合、1 为张开。臂目标会裁剪到执行器范围，
        非有限值、错误 shape 和越界夹爪命令会直接拒绝。
        """

        target = np.asarray(action, dtype=float)
        if target.shape != (8,):
            raise ValueError(f"action shape must be (8,), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("action must contain only finite values")
        if not 0.0 <= target[7] <= 1.0:
            raise ValueError("gripper action must be in [0, 1]")

        if physics_steps is None:
            physics_steps = self.steps_per_control
        if not isinstance(physics_steps, int) or isinstance(physics_steps, bool):
            raise TypeError("physics_steps must be a positive integer")
        if physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer")

        arm_limits = self.model.actuator_ctrlrange[: self.nq_arm]
        self.data.ctrl[: self.nq_arm] = np.clip(
            target[: self.nq_arm], arm_limits[:, 0], arm_limits[:, 1]
        )
        self.data.ctrl[7] = target[7] * GRIPPER_CTRL_MAX

        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
            self._update_success_tracker()

        status = self.placement_status()
        return {
            "sim_time": float(self.data.time),
            "success": self.success(),
            "placement_ready": status["ready"],
            "hold_steps": status["hold_steps"],
        }

    def placement_status(self) -> dict[str, float | int | bool]:
        """返回当前几何条件和连续保持进度，不推进仿真。"""

        cube = self.cube_pos
        box = self.model.body_pos[self.bid_box]
        full_cube_margin = BOX_INNER - CUBE_HALF
        inside_xy = bool(
            abs(cube[0] - box[0]) <= full_cube_margin
            and abs(cube[1] - box[1]) <= full_cube_margin
        )
        expected_center_z = box[2] + BOX_WALL + CUBE_HALF
        height_error = abs(cube[2] - expected_center_z)
        near_bottom = bool(height_error <= self.cfg.success_z_tolerance)
        return {
            "inside_xy": inside_xy,
            "near_bottom": near_bottom,
            "ready": inside_xy and near_bottom,
            "height_error": float(height_error),
            "hold_steps": self._placement_hold_steps,
            "required_hold_steps": self.required_success_steps,
        }

    def _reset_success_tracker(self) -> None:
        self._placement_hold_steps = 0
        self._success_confirmed = False

    def _update_success_tracker(self) -> None:
        if self.placement_status()["ready"]:
            self._placement_hold_steps += 1
            self._success_confirmed = (
                self._placement_hold_steps >= self.required_success_steps
            )
            return
        self._reset_success_tracker()

    def success(self) -> bool:
        """仅在 cube 完整入盒、接近底部并保持配置时长后返回成功。"""

        return self._success_confirmed

    def render(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            n = self.cfg.img_size
            self._renderer = mujoco.Renderer(self.model, n, n)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    import imageio

    env = PickPlace(TaskConfig(n_distractors=2, colors=("red",)))
    info = env.reset(np.random.default_rng(0))
    print(f"nq={env.model.nq} nu={env.model.nu}  {info}")
    print("TCP =", np.round(env.tcp, 4), " cube =", np.round(env.cube_pos, 4))
    for cam in ("front", "wrist"):
        imageio.imwrite(f"/tmp/task_{cam}.png", env.render(cam))
    env.close()
