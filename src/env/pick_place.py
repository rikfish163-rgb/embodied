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

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import TCP_OFFSET_Z, build_spec

# --- 几何常量 ---
TABLE_H = 0.02  # 台面厚度, 顶面 z = TABLE_H
CUBE_HALF = 0.02  # 立方体半边长 -> 4cm 立方体
BOX_INNER = 0.06  # 容器内半宽
BOX_WALL = 0.006  # 容器壁厚
BOX_H = 0.05  # 容器壁高

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
    """抓取-放置环境。只提供 reset/step/obs/success, 不含任何策略逻辑。"""

    def __init__(self, cfg: TaskConfig | None = None):
        self.cfg = cfg or TaskConfig()
        self.spec = build_task_spec(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

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

        self._renderer: mujoco.Renderer | None = None
        self.home_q = np.array([0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785])

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
        self.data.qpos[self.qadr_fingers : self.qadr_fingers + 2] = 0.04  # 张开

        cx = rng.uniform(*r["cube_x"])
        cy = rng.uniform(*r["cube_y"])
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self._set_free(self.qadr_cube, (cx, cy, TABLE_H + CUBE_HALF), yaw)

        for i, adr in enumerate(self.qadr_distractors):
            while True:  # 不与目标重叠
                dx, dy = rng.uniform(*r["cube_x"]), rng.uniform(*r["cube_y"])
                if np.hypot(dx - cx, dy - cy) > 4 * CUBE_HALF:
                    break
            self._set_free(adr, (dx, dy, TABLE_H + CUBE_HALF), rng.uniform(-np.pi, np.pi))

        bx, by = rng.uniform(*r["box_x"]), rng.uniform(*r["box_y"])
        self.model.body_pos[self.bid_box] = [bx, by, TABLE_H]

        self.data.ctrl[:] = 0.0
        self.data.ctrl[: self.nq_arm] = self.home_q
        self.data.ctrl[7] = 255.0  # 夹爪张开
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

    def success(self) -> bool:
        """立方体落入容器内且高度低于壁顶 -> 成功。"""
        c = self.cube_pos
        b = self.model.body_pos[self.bid_box]
        inside_xy = abs(c[0] - b[0]) < BOX_INNER and abs(c[1] - b[1]) < BOX_INNER
        landed = c[2] < TABLE_H + BOX_H
        return bool(inside_xy and landed)

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
