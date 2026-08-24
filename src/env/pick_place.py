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

import hashlib
import json
import math
from collections.abc import Mapping
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
DISTRACTOR_MIN_SEPARATION = 4 * CUBE_HALF
MAX_DISTRACTOR_PLACEMENT_ATTEMPTS = 256
MAX_BOX_PLACEMENT_ATTEMPTS = 256
RESET_SAMPLER_VERSION = "pick_place_collision_free_rejection_v1"
RESET_CANDIDATE_HASH_VERSION = "sha256-canonical-json-array-v1"
RESET_RECEIPT_SCHEMA_VERSION = "pick_place_reset_receipt_v1"
_CONTAINMENT_EPS = 1e-9

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


class _CandidateSequenceHasher:
    """Constant-memory SHA-256 over a canonical JSON candidate array."""

    def __init__(self):
        self._hasher = hashlib.sha256()
        self._hasher.update(b"[")
        self.count = 0
        self._records: list[dict[str, object]] = []

    def add(self, xy: tuple[float, float], *, collision_free: bool) -> None:
        record = {
            "candidate_index": self.count,
            "collision_free": bool(collision_free),
            "xy": [float(xy[0]), float(xy[1])],
        }
        self._records.append(record)
        encoded = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if self.count:
            self._hasher.update(b",")
        self._hasher.update(encoded)
        self.count += 1

    def hexdigest(self) -> str:
        completed = self._hasher.copy()
        completed.update(b"]")
        return completed.hexdigest()

    def ledger(self) -> list[dict[str, object]]:
        """Return the bounded (at most 256 row) auditable candidate sequence."""

        return [
            {
                "candidate_index": record["candidate_index"],
                "collision_free": record["collision_free"],
                "xy": list(record["xy"]),  # type: ignore[arg-type]
            }
            for record in self._records
        ]


class DistractorPlacementError(RuntimeError):
    """Raised when reset cannot place a distractor within its finite budget."""

    def __init__(
        self,
        distractor_index: int,
        candidate_sequence_sha256: str,
        candidate_ledger: list[dict[str, object]],
    ):
        self.distractor_index = distractor_index
        self.receipt_schema_version = RESET_RECEIPT_SCHEMA_VERSION
        self.sampler_version = RESET_SAMPLER_VERSION
        self.candidate_hash_version = RESET_CANDIDATE_HASH_VERSION
        self.attempts = MAX_DISTRACTOR_PLACEMENT_ATTEMPTS
        self.rejections = self.attempts
        self.accepted_candidate_index = None
        self.collision_free = False
        self.candidate_sequence_sha256 = candidate_sequence_sha256
        self.candidate_ledger = candidate_ledger
        self.minimum_separation = DISTRACTOR_MIN_SEPARATION
        super().__init__(
            f"distractor {distractor_index} could not be placed after "
            f"{self.attempts} attempts with minimum separation "
            f"{self.minimum_separation:.2f} m; sampler={self.sampler_version} "
            f"candidate_sha256={self.candidate_sequence_sha256}"
        )


class BoxPlacementError(RuntimeError):
    """Raised when reset cannot separate the box from all movable cubes."""

    def __init__(
        self,
        candidate_sequence_sha256: str,
        candidate_ledger: list[dict[str, object]],
    ):
        self.receipt_schema_version = RESET_RECEIPT_SCHEMA_VERSION
        self.sampler_version = RESET_SAMPLER_VERSION
        self.candidate_hash_version = RESET_CANDIDATE_HASH_VERSION
        self.attempts = MAX_BOX_PLACEMENT_ATTEMPTS
        self.rejections = self.attempts
        self.accepted_candidate_index = None
        self.collision_free = False
        self.candidate_sequence_sha256 = candidate_sequence_sha256
        self.candidate_ledger = candidate_ledger
        super().__init__(
            f"box could not be placed after {self.attempts} attempts without "
            f"intersecting target and distractors; sampler={self.sampler_version} "
            f"candidate_sha256={self.candidate_sequence_sha256}"
        )


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
        self.gid_cube = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.gid_box_base = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_base"
        )
        self.gid_box_walls = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"box_{name}")
            for name in ("xp", "xn", "yp", "yn")
        }
        self.gid_distractors = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"distractor{index}_geom",
            )
            for index in range(self.cfg.n_distractors)
        ]

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
        r = self._validated_ranges(ranges)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_success_tracker()
        self.data.qpos[: self.nq_arm] = self.home_q
        self.data.qpos[self.qadr_fingers : self.qadr_fingers + 2] = (
            FINGER_OPEN_QPOS  # 张开
        )

        cx = rng.uniform(*r["cube_x"])
        cy = rng.uniform(*r["cube_y"])
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self._set_free(self.qadr_cube, (cx, cy, TABLE_H + CUBE_HALF), yaw)

        placed_xy = [np.array([cx, cy])]
        distractor_sampling: list[dict[str, object]] = []
        for i, adr in enumerate(self.qadr_distractors):
            candidate_audit = _CandidateSequenceHasher()
            for _ in range(MAX_DISTRACTOR_PLACEMENT_ATTEMPTS):
                dx, dy = rng.uniform(*r["cube_x"]), rng.uniform(*r["cube_y"])
                candidate = np.array([dx, dy])
                min_center_separation = min(
                    float(np.linalg.norm(candidate - other)) for other in placed_xy
                )
                collision_free = min_center_separation > DISTRACTOR_MIN_SEPARATION
                candidate_audit.add(
                    (float(dx), float(dy)), collision_free=collision_free
                )
                if collision_free:
                    break
            else:
                raise DistractorPlacementError(
                    i, candidate_audit.hexdigest(), candidate_audit.ledger()
                )
            distractor_yaw = rng.uniform(-np.pi, np.pi)
            self._set_free(adr, (dx, dy, TABLE_H + CUBE_HALF), distractor_yaw)
            placed_xy.append(candidate)
            distractor_sampling.append(
                {
                    "distractor_index": i,
                    "attempts": candidate_audit.count,
                    "rejections": candidate_audit.count - 1,
                    "accepted_candidate_index": candidate_audit.count - 1,
                    "accepted_xy": [float(dx), float(dy)],
                    "accepted_yaw_rad": float(distractor_yaw),
                    "collision_free": True,
                    "accepted_min_center_separation_m": min_center_separation,
                    "candidate_ledger": candidate_audit.ledger(),
                    "candidate_sequence_sha256": candidate_audit.hexdigest(),
                }
            )

        box_candidate_audit = _CandidateSequenceHasher()
        for _ in range(MAX_BOX_PLACEMENT_ATTEMPTS):
            bx, by = rng.uniform(*r["box_x"]), rng.uniform(*r["box_y"])
            self.model.body_pos[self.bid_box] = [bx, by, TABLE_H]
            mujoco.mj_forward(self.model, self.data)
            min_box_clearance = self._box_min_separating_clearance()
            collision_free = min_box_clearance > _CONTAINMENT_EPS
            box_candidate_audit.add(
                (float(bx), float(by)), collision_free=collision_free
            )
            if collision_free:
                break
        else:
            raise BoxPlacementError(
                box_candidate_audit.hexdigest(), box_candidate_audit.ledger()
            )

        self.data.ctrl[:] = 0.0
        self.data.ctrl[: self.nq_arm] = self.home_q
        self.data.ctrl[7] = GRIPPER_CTRL_MAX  # 夹爪张开
        mujoco.mj_forward(self.model, self.data)
        return {
            "cube_xy": (cx, cy),
            "cube_yaw": yaw,
            "box_xy": (bx, by),
            "receipt_schema_version": RESET_RECEIPT_SCHEMA_VERSION,
            "sampler_version": RESET_SAMPLER_VERSION,
            "candidate_hash_version": RESET_CANDIDATE_HASH_VERSION,
            "collision_free": True,
            "target_sampling": {
                "accepted_xy": [float(cx), float(cy)],
                "accepted_yaw_rad": float(yaw),
            },
            "distractor_sampling": distractor_sampling,
            "box_sampling": {
                "attempts": box_candidate_audit.count,
                "rejections": box_candidate_audit.count - 1,
                "accepted_candidate_index": box_candidate_audit.count - 1,
                "accepted_xy": [float(bx), float(by)],
                "collision_free": True,
                "accepted_min_clearance_m": min_box_clearance,
                "candidate_ledger": box_candidate_audit.ledger(),
                "candidate_sequence_sha256": box_candidate_audit.hexdigest(),
            },
        }

    def _geom_xy_bounds(self, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
        axes = self.data.geom_xmat[geom_id].reshape(3, 3)
        half_extent = np.abs(axes[:2]) @ self.model.geom_size[geom_id]
        center = self.data.geom_xpos[geom_id, :2]
        return center - half_extent, center + half_extent

    def _box_min_separating_clearance(self) -> float:
        wall_bounds = [
            self._geom_xy_bounds(geom_id) for geom_id in self.gid_box_walls.values()
        ]
        box_low = np.min([bounds[0] for bounds in wall_bounds], axis=0)
        box_high = np.max([bounds[1] for bounds in wall_bounds], axis=0)
        object_clearances: list[float] = []
        for geom_id in (self.gid_cube, *self.gid_distractors):
            object_low, object_high = self._geom_xy_bounds(geom_id)
            object_clearances.append(
                max(
                    float(box_low[0] - object_high[0]),
                    float(object_low[0] - box_high[0]),
                    float(box_low[1] - object_high[1]),
                    float(object_low[1] - box_high[1]),
                )
            )
        return min(object_clearances)

    @staticmethod
    def _validated_ranges(
        ranges: Mapping[str, object] | None,
    ) -> dict[str, tuple[float, float]]:
        defaults: dict[str, object] = {
            "cube_x": CUBE_X,
            "cube_y": CUBE_Y,
            "box_x": BOX_X,
            "box_y": BOX_Y,
        }
        if ranges is not None:
            if not isinstance(ranges, Mapping):
                raise TypeError("ranges must be a mapping")
            unknown = sorted(set(ranges) - set(defaults))
            if unknown:
                raise ValueError(
                    f"unknown randomization range key(s): {', '.join(unknown)}"
                )
            defaults.update(ranges)

        validated: dict[str, tuple[float, float]] = {}
        for name, value in defaults.items():
            try:
                array = np.asarray(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"{name} must contain two finite numbers with low < high"
                ) from None
            valid_dtype = array.dtype.kind in "iuf" and array.dtype.kind != "b"
            if array.shape == (2,) and valid_dtype:
                numeric = array.astype(float)
                valid = bool(np.all(np.isfinite(numeric)) and numeric[0] < numeric[1])
            else:
                valid = False
            if not valid:
                raise ValueError(
                    f"{name} must contain two finite numbers with low < high"
                )
            validated[name] = (float(numeric[0]), float(numeric[1]))
        return validated

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
        result = {
            "sim_time": float(self.data.time),
            "success": self.success(),
            "placement_ready": status["ready"],
        }
        result.update(status)
        return result

    def placement_status(self) -> dict[str, float | int | bool]:
        """返回当前几何条件和连续保持进度，不推进仿真。"""

        cube_center = self.data.geom_xpos[self.gid_cube]
        cube_axes = self.data.geom_xmat[self.gid_cube].reshape(3, 3)
        cube_half_size = self.model.geom_size[self.gid_cube]

        def obb_clearance(point: np.ndarray, inward_normal: np.ndarray) -> float:
            support = float(np.abs(cube_axes.T @ inward_normal) @ cube_half_size)
            return float((cube_center - point) @ inward_normal - support)

        wall_specs = {
            "xp": (0, np.array([-1.0, 0.0, 0.0])),
            "xn": (0, np.array([1.0, 0.0, 0.0])),
            "yp": (1, np.array([0.0, -1.0, 0.0])),
            "yn": (1, np.array([0.0, 1.0, 0.0])),
        }
        wall_clearances: dict[str, float] = {}
        top_clearances: list[float] = []
        for name, (thickness_axis, local_inward) in wall_specs.items():
            gid = self.gid_box_walls[name]
            wall_axes = self.data.geom_xmat[gid].reshape(3, 3)
            inward = wall_axes @ local_inward
            inner_face = (
                self.data.geom_xpos[gid]
                + inward * self.model.geom_size[gid, thickness_axis]
            )
            wall_clearances[name] = obb_clearance(inner_face, inward)

            top_normal = wall_axes[:, 2]
            wall_top = (
                self.data.geom_xpos[gid] + top_normal * self.model.geom_size[gid, 2]
            )
            top_clearances.append(obb_clearance(wall_top, -top_normal))

        base_axes = self.data.geom_xmat[self.gid_box_base].reshape(3, 3)
        base_normal = base_axes[:, 2]
        base_top = (
            self.data.geom_xpos[self.gid_box_base]
            + base_normal * self.model.geom_size[self.gid_box_base, 2]
        )
        bottom_clearance = obb_clearance(base_top, base_normal)
        top_clearance = min(top_clearances)
        min_wall_clearance = min(wall_clearances.values())

        inside_xy = bool(min_wall_clearance >= -_CONTAINMENT_EPS)
        above_bottom = bool(bottom_clearance >= -self.cfg.success_z_tolerance)
        below_wall_top = bool(top_clearance >= -_CONTAINMENT_EPS)
        fully_contained = inside_xy and above_bottom and below_wall_top
        height_error = abs(bottom_clearance)
        near_bottom = bool(height_error <= self.cfg.success_z_tolerance)
        return {
            "inside_xy": inside_xy,
            "above_bottom": above_bottom,
            "below_wall_top": below_wall_top,
            "fully_contained": fully_contained,
            "near_bottom": near_bottom,
            "ready": fully_contained and near_bottom,
            "height_error": float(height_error),
            "corner_clearance_xp": wall_clearances["xp"],
            "corner_clearance_xn": wall_clearances["xn"],
            "corner_clearance_yp": wall_clearances["yp"],
            "corner_clearance_yn": wall_clearances["yn"],
            "min_corner_wall_clearance": min_wall_clearance,
            "min_corner_bottom_clearance": bottom_clearance,
            "min_corner_top_clearance": top_clearance,
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
