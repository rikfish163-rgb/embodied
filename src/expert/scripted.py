"""Privileged DLS expert for the Panda pick-and-place task.

The expert may read cube, box, and TCP ground truth. Learned policies must not.
All physical motion still goes through :meth:`env.pick_place.PickPlace.step`;
this module never writes the cube pose after the environment reset.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from env.pick_place import PickPlace

StepCallback = Callable[[PickPlace, str, np.ndarray], None]


@dataclass(frozen=True)
class ExpertConfig:
    damping: float = 0.025
    cartesian_step_m: float = 0.008
    rotation_step_rad: float = 0.04
    joint_step_rad: float = 0.05
    nullspace_gain: float = 0.015
    command_lead_norm: float = 0.12
    command_lead_per_joint: float = 0.04
    position_tolerance_m: float = 0.004
    velocity_tolerance: float = 0.15
    max_move_steps: int = 180
    max_attempts: int = 2
    pregrasp_z: float = 0.18
    grasp_z: float = 0.052
    carry_z: float = 0.25
    release_z: float = 0.105
    minimum_lifted_cube_z: float = 0.10
    dropped_cube_z: float = 0.08
    close_steps: int = 20
    open_steps: int = 20
    settle_steps: int = 30
    recovery_wait_steps: int = 12
    diagonal_grasp_offset_rad: float = math.pi / 4

    def __post_init__(self) -> None:
        positive_floats = {
            "damping": self.damping,
            "cartesian_step_m": self.cartesian_step_m,
            "rotation_step_rad": self.rotation_step_rad,
            "joint_step_rad": self.joint_step_rad,
            "command_lead_norm": self.command_lead_norm,
            "command_lead_per_joint": self.command_lead_per_joint,
            "position_tolerance_m": self.position_tolerance_m,
            "velocity_tolerance": self.velocity_tolerance,
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        positive_ints = {
            "max_move_steps": self.max_move_steps,
            "max_attempts": self.max_attempts,
            "close_steps": self.close_steps,
            "open_steps": self.open_steps,
            "settle_steps": self.settle_steps,
            "recovery_wait_steps": self.recovery_wait_steps,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ExpertStep:
    stage: str
    sim_time_s: float
    action: np.ndarray
    tcp: np.ndarray
    cube_pos: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "sim_time_s": self.sim_time_s,
            "action": self.action.tolist(),
            "tcp": self.tcp.tolist(),
            "cube_pos": self.cube_pos.tolist(),
        }


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    failure_stage: str | None
    attempts: int
    recovered: bool
    sim_time_s: float
    control_steps: int
    reset_info: dict[str, Any]
    final_cube_pos: np.ndarray
    final_box_pos: np.ndarray
    steps: tuple[ExpertStep, ...] = ()

    def to_dict(self, *, include_steps: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "seed": self.seed,
            "success": self.success,
            "failure_stage": self.failure_stage,
            "attempts": self.attempts,
            "recovered": self.recovered,
            "sim_time_s": self.sim_time_s,
            "control_steps": self.control_steps,
            "reset_info": _jsonable(self.reset_info),
            "final_cube_pos": self.final_cube_pos.tolist(),
            "final_box_pos": self.final_box_pos.tolist(),
        }
        if include_steps:
            output["steps"] = [step.to_dict() for step in self.steps]
        return output


class DLSController:
    """Six-dimensional damped-least-squares TCP controller."""

    def __init__(self, env: PickPlace, config: ExpertConfig | None = None):
        self.env = env
        self.config = config or ExpertConfig()
        self.q_reference = env.data.qpos[: env.nq_arm].copy()
        self.q_command = self.q_reference.copy()
        self.home_rotation = env.data.site_xmat[env.sid_tcp].reshape(3, 3).copy()
        self.target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(self.target_quaternion, self.home_rotation.ravel())
        self.jacp = np.zeros((3, env.model.nv))
        self.jacr = np.zeros((3, env.model.nv))
        self.identity_task = np.eye(6)
        self.identity_joints = np.eye(env.nq_arm)

    def set_grasp_yaw(self, cube_yaw: float) -> None:
        """Choose the nearest cube diagonal for a deeper, symmetric grasp."""

        offset = self.config.diagonal_grasp_offset_rad
        grasp_yaw = cube_yaw + (offset if cube_yaw <= 0 else -offset)
        cosine = math.cos(grasp_yaw)
        sine = math.sin(grasp_yaw)
        rotation_z = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        target_rotation = rotation_z @ self.home_rotation
        mujoco.mju_mat2Quat(self.target_quaternion, target_rotation.ravel())

    def action(self, target_position: np.ndarray, gripper: float) -> np.ndarray:
        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be a finite 3-vector")
        if not 0.0 <= gripper <= 1.0:
            raise ValueError("gripper must be in [0, 1]")

        mujoco.mj_jacSite(
            self.env.model,
            self.env.data,
            self.jacp,
            self.jacr,
            self.env.sid_tcp,
        )
        jacobian = np.vstack(
            [
                self.jacp[:, : self.env.nq_arm],
                self.jacr[:, : self.env.nq_arm],
            ]
        )

        position_delta = _limited_vector(
            target - self.env.tcp, self.config.cartesian_step_m
        )

        current_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            current_quaternion, self.env.data.site_xmat[self.env.sid_tcp]
        )
        local_rotation_error = np.empty(3)
        mujoco.mju_subQuat(
            local_rotation_error,
            self.target_quaternion,
            current_quaternion,
        )
        # mju_subQuat returns the correction in the current TCP frame, while
        # mj_jacSite's rotational rows are expressed in the world frame.
        current_rotation = self.env.data.site_xmat[self.env.sid_tcp].reshape(3, 3)
        world_rotation_error = current_rotation @ local_rotation_error
        rotation_delta = _limited_vector(
            world_rotation_error, self.config.rotation_step_rad
        )

        task_delta = np.concatenate([position_delta, rotation_delta])
        regularized = (
            jacobian @ jacobian.T + self.config.damping**2 * self.identity_task
        )
        pseudo_inverse = jacobian.T @ np.linalg.solve(regularized, self.identity_task)
        nullspace = self.identity_joints - pseudo_inverse @ jacobian
        joint_delta = pseudo_inverse @ task_delta
        joint_delta += nullspace @ (
            self.config.nullspace_gain
            * (self.q_reference - self.env.data.qpos[: self.env.nq_arm])
        )
        joint_delta = np.clip(
            joint_delta,
            -self.config.joint_step_rad,
            self.config.joint_step_rad,
        )

        limits = self.env.model.actuator_ctrlrange[: self.env.nq_arm]
        self.q_command = np.clip(
            self.q_command + joint_delta,
            limits[:, 0],
            limits[:, 1],
        )
        command_lead = self.q_command - self.env.data.qpos[: self.env.nq_arm]
        if np.linalg.norm(command_lead) > self.config.command_lead_norm:
            self.q_command = self.env.data.qpos[: self.env.nq_arm] + np.clip(
                command_lead,
                -self.config.command_lead_per_joint,
                self.config.command_lead_per_joint,
            )

        return np.concatenate([self.q_command.copy(), [gripper]])


def run_episode(
    env: PickPlace,
    *,
    seed: int,
    config: ExpertConfig | None = None,
    record_steps: bool = False,
    step_callback: StepCallback | None = None,
) -> EpisodeResult:
    """Run one deterministic privileged expert episode."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    cfg = config or ExpertConfig()
    reset_info = env.reset(np.random.default_rng(seed))
    controller = DLSController(env, cfg)
    controller.set_grasp_yaw(float(reset_info["cube_yaw"]))
    box_position = env.model.body_pos[env.bid_box].copy()
    records: list[ExpertStep] = []
    control_steps = 0

    def step(stage: str, target: np.ndarray, gripper: float) -> None:
        nonlocal control_steps
        action = controller.action(target, gripper)
        if step_callback is not None:
            step_callback(env, stage, action.copy())
        if record_steps:
            records.append(
                ExpertStep(
                    stage=stage,
                    sim_time_s=float(env.data.time),
                    action=action.copy(),
                    tcp=env.tcp,
                    cube_pos=env.cube_pos,
                )
            )
        env.step(action)
        control_steps += 1

    def move(stage: str, target: np.ndarray, gripper: float) -> bool:
        for _ in range(cfg.max_move_steps):
            step(stage, target, gripper)
            position_error = float(np.linalg.norm(env.tcp - target))
            joint_speed = float(np.linalg.norm(env.data.qvel[: env.nq_arm]))
            if (
                position_error < cfg.position_tolerance_m
                and joint_speed < cfg.velocity_tolerance
            ):
                return True
        return False

    def hold(stage: str, target: np.ndarray, gripper: float, count: int) -> None:
        for _ in range(count):
            step(stage, target, gripper)

    def finish(
        *, failure_stage: str | None, attempts: int, success: bool | None = None
    ) -> EpisodeResult:
        actual_success = env.success() if success is None else success
        return EpisodeResult(
            seed=seed,
            success=actual_success,
            failure_stage=None if actual_success else failure_stage,
            attempts=attempts,
            recovered=actual_success and attempts > 1,
            sim_time_s=float(env.data.time),
            control_steps=control_steps,
            reset_info=reset_info,
            final_cube_pos=env.cube_pos,
            final_box_pos=box_position.copy(),
            steps=tuple(records),
        )

    transported = False
    failure_stage: str | None = None
    attempts = 0
    for attempt_index in range(cfg.max_attempts):
        attempts = attempt_index + 1
        if attempt_index:
            recovery_target = env.tcp
            hold(
                "recover",
                recovery_target,
                1.0,
                cfg.recovery_wait_steps,
            )
            controller.set_grasp_yaw(_cube_yaw(env))

        cube_position = env.cube_pos
        pregrasp = np.array([cube_position[0], cube_position[1], cfg.pregrasp_z])
        grasp = np.array([cube_position[0], cube_position[1], cfg.grasp_z])
        lift = np.array([cube_position[0], cube_position[1], cfg.carry_z])
        transport = np.array([box_position[0], box_position[1], cfg.carry_z])

        if not move("pregrasp", pregrasp, 1.0):
            return finish(failure_stage="pregrasp", attempts=attempts, success=False)
        if not move("descend", grasp, 1.0):
            return finish(failure_stage="descend", attempts=attempts, success=False)
        hold("close", grasp, 0.0, cfg.close_steps)
        if not move("lift", lift, 0.0):
            return finish(failure_stage="lift", attempts=attempts, success=False)
        if env.cube_pos[2] < cfg.minimum_lifted_cube_z:
            failure_stage = "lift"
            continue
        if not move("transport", transport, 0.0):
            return finish(failure_stage="transport", attempts=attempts, success=False)
        if (
            env.cube_pos[2] < cfg.dropped_cube_z
            and not env.placement_status()["inside_xy"]
        ):
            failure_stage = "transport"
            continue
        transported = True
        failure_stage = None
        break

    if not transported:
        return finish(failure_stage=failure_stage or "transport", attempts=attempts)

    release = np.array([box_position[0], box_position[1], cfg.release_z])
    retreat = np.array([box_position[0], box_position[1], cfg.carry_z])
    if not move("lower", release, 0.0):
        return finish(failure_stage="lower", attempts=attempts)
    hold("open", release, 1.0, cfg.open_steps)
    if not move("retreat", retreat, 1.0):
        return finish(failure_stage="retreat", attempts=attempts)
    hold("settle", retreat, 1.0, cfg.settle_steps)
    return finish(failure_stage="settle", attempts=attempts)


def _cube_yaw(env: PickPlace) -> float:
    quaternion = env.data.qpos[env.qadr_cube + 3 : env.qadr_cube + 7]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    matrix = rotation.reshape(3, 3)
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


def _limited_vector(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit:
        return vector
    return vector * (limit / norm)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_dict(config: ExpertConfig) -> dict[str, Any]:
    """Return a JSON-serializable controller configuration."""

    return asdict(config)
