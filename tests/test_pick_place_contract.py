from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from env.pick_place import BOX_WALL, CUBE_HALF, PickPlace, TaskConfig


@pytest.fixture
def env() -> PickPlace:
    instance = PickPlace(TaskConfig(img_size=128))
    yield instance
    instance.close()


def _place_cube(env: PickPlace, *, xy: np.ndarray, z: float) -> None:
    env.data.qpos[env.qadr_cube : env.qadr_cube + 3] = [xy[0], xy[1], z]
    env.data.qpos[env.qadr_cube + 3 : env.qadr_cube + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _home_action(env: PickPlace, gripper: float = 1.0) -> np.ndarray:
    return np.concatenate([env.home_q, [gripper]])


def test_reset_and_observation_contract_are_deterministic(env: PickPlace) -> None:
    first = env.reset(np.random.default_rng(42))
    second = env.reset(np.random.default_rng(42))

    assert first == second

    observation = env.observe()
    assert set(observation) == {
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
    }
    assert observation["observation.images.front"].shape == (128, 128, 3)
    assert observation["observation.images.wrist"].shape == (128, 128, 3)
    assert observation["observation.images.front"].dtype == np.uint8
    assert observation["observation.images.wrist"].dtype == np.uint8
    assert observation["observation.state"].shape == (8,)
    assert np.all(np.isfinite(observation["observation.state"]))
    assert 0.0 <= observation["observation.state"][-1] <= 1.0


def test_step_maps_normalized_gripper_and_rejects_invalid_actions(
    env: PickPlace,
) -> None:
    env.reset(np.random.default_rng(0))

    env.step(_home_action(env, gripper=0.25), physics_steps=1)
    assert env.data.ctrl[7] == pytest.approx(0.25 * 255.0)

    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(7), physics_steps=1)
    with pytest.raises(ValueError, match="finite"):
        bad = _home_action(env)
        bad[0] = np.nan
        env.step(bad, physics_steps=1)
    with pytest.raises(ValueError, match="gripper"):
        env.step(_home_action(env, gripper=1.1), physics_steps=1)


def test_default_step_advances_one_control_period_and_clips_arm_targets(
    env: PickPlace,
) -> None:
    env.reset(np.random.default_rng(0))
    action = np.concatenate([np.full(7, 100.0), [0.5]])

    result = env.step(action)

    assert result["sim_time"] == pytest.approx(1.0 / env.cfg.control_hz)
    np.testing.assert_array_less(
        env.data.ctrl[:7], env.model.actuator_ctrlrange[:7, 1] + 1e-12
    )
    np.testing.assert_array_less(
        env.model.actuator_ctrlrange[:7, 0] - 1e-12, env.data.ctrl[:7]
    )


def test_success_requires_bottom_placement_and_full_hold_time(env: PickPlace) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]

    # Being barely below the wall top is not a valid bottom placement.
    wall_top = box[2] + env.box_height
    _place_cube(env, xy=box[:2], z=wall_top - 1e-4)
    assert not env.placement_status()["near_bottom"]
    assert not env.success()

    floor_center_z = box[2] + BOX_WALL + CUBE_HALF
    _place_cube(env, xy=box[:2], z=floor_center_z)
    assert env.placement_status()["ready"]
    assert not env.success()

    required_steps = math.ceil(env.cfg.success_hold_s / env.model.opt.timestep)
    env.step(_home_action(env), physics_steps=required_steps - 1)
    assert not env.success()
    env.step(_home_action(env), physics_steps=1)
    assert env.success()


def test_success_resets_as_soon_as_cube_leaves_box(env: PickPlace) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]
    floor_center_z = box[2] + BOX_WALL + CUBE_HALF
    _place_cube(env, xy=box[:2], z=floor_center_z)

    required_steps = math.ceil(env.cfg.success_hold_s / env.model.opt.timestep)
    env.step(_home_action(env), physics_steps=required_steps)
    assert env.success()

    env.reset(np.random.default_rng(1))
    assert not env.success()
    assert env.placement_status()["hold_steps"] == 0

    _place_cube(env, xy=box[:2] + [env.box_inner, 0.0], z=floor_center_z)
    env.step(_home_action(env), physics_steps=1)
    assert not env.success()
    assert env.placement_status()["hold_steps"] == 0


def test_tcp_jacobian_is_finite_and_full_rank(env: PickPlace) -> None:
    env.reset(np.random.default_rng(0))
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env.sid_tcp)
    arm_jacobian = np.vstack([jacp[:, :7], jacr[:, :7]])

    assert arm_jacobian.shape == (6, 7)
    assert np.all(np.isfinite(arm_jacobian))
    assert np.linalg.matrix_rank(arm_jacobian) == 6
