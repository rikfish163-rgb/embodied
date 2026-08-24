from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

import env.pick_place as pick_place
from data.manifest import compiled_model_fingerprint
from env.pick_place import BOX_WALL, CUBE_HALF, PickPlace, TaskConfig


@pytest.fixture
def env() -> PickPlace:
    instance = PickPlace(TaskConfig(img_size=128))
    yield instance
    instance.close()


def _place_cube(
    env: PickPlace,
    *,
    xy: np.ndarray,
    z: float,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    env.data.qpos[env.qadr_cube : env.qadr_cube + 3] = [xy[0], xy[1], z]
    env.data.qpos[env.qadr_cube + 3 : env.qadr_cube + 7] = quaternion
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


def test_compiled_model_fingerprint_is_stable_across_reset(env: PickPlace) -> None:
    before = compiled_model_fingerprint(env)

    env.reset(np.random.default_rng(42))

    assert compiled_model_fingerprint(env) == before


def test_reset_without_distractors_preserves_the_seed_zero_rng_sequence(
    env: PickPlace,
) -> None:
    rng = np.random.default_rng(0)

    info = env.reset(rng)

    assert info["cube_xy"] == pytest.approx((0.5346531037178618, -0.0736682515955615))
    assert info["cube_yaw"] == pytest.approx(-0.7210371025026309)
    assert info["box_xy"] == pytest.approx((0.4419833162634235, 0.34132702392002723))
    assert rng.random() == pytest.approx(0.9127555772777217)
    assert info["receipt_schema_version"] == "pick_place_reset_receipt_v1"
    assert info["sampler_version"] == "pick_place_collision_free_rejection_v1"
    assert info["candidate_hash_version"] == "sha256-canonical-json-array-v1"
    assert info["collision_free"] is True
    assert info["target_sampling"] == {
        "accepted_xy": pytest.approx([0.5346531037178618, -0.0736682515955615]),
        "accepted_yaw_rad": pytest.approx(-0.7210371025026309),
    }
    assert info["distractor_sampling"] == []
    assert info["box_sampling"]["attempts"] == 1
    assert info["box_sampling"]["rejections"] == 0
    assert info["box_sampling"]["accepted_candidate_index"] == 0
    assert info["box_sampling"]["collision_free"] is True
    assert pick_place.RESET_SAMPLER_VERSION == "pick_place_collision_free_rejection_v1"


class _ScriptedGenerator:
    """Finite deterministic RNG double for the reset draw schedule."""

    def __init__(self, values: list[float]):
        self._values = iter(values)
        self.calls = 0

    def uniform(self, low: float, high: float) -> float:
        self.calls += 1
        value = next(self._values)
        assert float(low) <= value <= float(high)
        return value


def test_reset_audits_first_box_rejection_and_second_candidate_acceptance() -> None:
    instance = PickPlace()
    rng = _ScriptedGenerator([0.5, 0.224, 0.0, 0.5, 0.26, 0.5, 0.36])
    ranges = {
        "cube_x": (0.49, 0.51),
        "cube_y": (0.223, 0.225),
        "box_x": (0.49, 0.51),
        "box_y": (0.25, 0.37),
    }
    try:
        info = instance.reset(rng, ranges=ranges)  # type: ignore[arg-type]
        repeated = instance.reset(
            _ScriptedGenerator([0.5, 0.224, 0.0, 0.5, 0.26, 0.5, 0.36]),
            ranges=ranges,  # type: ignore[arg-type]
        )
    finally:
        instance.close()

    assert rng.calls == 7
    assert info["cube_xy"] == pytest.approx((0.5, 0.224))
    assert info["box_xy"] == pytest.approx((0.5, 0.36))
    assert info["sampler_version"] == "pick_place_collision_free_rejection_v1"
    assert info["candidate_hash_version"] == "sha256-canonical-json-array-v1"
    assert info["collision_free"] is True
    assert info["distractor_sampling"] == []
    box_audit = info["box_sampling"]
    assert box_audit["attempts"] == 2
    assert box_audit["rejections"] == 1
    assert box_audit["accepted_candidate_index"] == 1
    assert box_audit["collision_free"] is True
    assert box_audit["accepted_min_clearance_m"] == pytest.approx(0.053)
    assert box_audit["accepted_xy"] == pytest.approx([0.5, 0.36])
    assert box_audit["candidate_ledger"] == [
        {"candidate_index": 0, "collision_free": False, "xy": [0.5, 0.26]},
        {"candidate_index": 1, "collision_free": True, "xy": [0.5, 0.36]},
    ]
    assert box_audit["candidate_sequence_sha256"] == (
        "7ec52f749827dc692697e2e4497ad8be6f9e75de512a64e65468d8c252fe3e84"
    )
    assert repeated["box_sampling"] == box_audit


def _free_body_xy_positions(env: PickPlace) -> list[np.ndarray]:
    addresses = [env.qadr_cube, *env.qadr_distractors]
    return [env.data.qpos[address : address + 2].copy() for address in addresses]


def _assert_conservative_pairwise_spacing(env: PickPlace) -> None:
    positions = _free_body_xy_positions(env)
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            assert np.linalg.norm(first - second) > 0.08


def _geom_xy_bounds(env: PickPlace, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    axes = env.data.geom_xmat[geom_id].reshape(3, 3)
    half_extent = np.abs(axes[:2]) @ env.model.geom_size[geom_id]
    center = env.data.geom_xpos[geom_id, :2]
    return center - half_extent, center + half_extent


def _assert_objects_clear_the_box_outer_footprint(env: PickPlace) -> None:
    wall_ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"box_{name}")
        for name in ("xp", "xn", "yp", "yn")
    ]
    wall_bounds = [_geom_xy_bounds(env, geom_id) for geom_id in wall_ids]
    box_low = np.min([bounds[0] for bounds in wall_bounds], axis=0)
    box_high = np.max([bounds[1] for bounds in wall_bounds], axis=0)
    object_ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        *[
            mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"distractor{index}_geom",
            )
            for index in range(env.cfg.n_distractors)
        ],
    ]
    for geom_id in object_ids:
        object_low, object_high = _geom_xy_bounds(env, geom_id)
        has_separating_gap = bool(
            object_high[0] < box_low[0]
            or object_low[0] > box_high[0]
            or object_high[1] < box_low[1]
            or object_low[1] > box_high[1]
        )
        assert has_separating_gap


def test_two_distractors_avoid_each_other_for_known_seed_one() -> None:
    instance = PickPlace(TaskConfig(n_distractors=2))
    try:
        instance.reset(np.random.default_rng(1))

        _assert_conservative_pairwise_spacing(instance)
    finally:
        instance.close()


def test_distractor_spacing_is_pairwise_safe_and_deterministic_across_seeds() -> None:
    instance = PickPlace(TaskConfig(n_distractors=3))
    try:
        for seed in range(16):
            first_info = instance.reset(np.random.default_rng(seed))
            first_positions = np.stack(_free_body_xy_positions(instance))
            _assert_conservative_pairwise_spacing(instance)
            _assert_objects_clear_the_box_outer_footprint(instance)
            assert first_info["sampler_version"] == (
                "pick_place_collision_free_rejection_v1"
            )
            assert first_info["collision_free"] is True
            assert len(first_info["distractor_sampling"]) == 3
            for index, audit in enumerate(first_info["distractor_sampling"]):
                assert audit["distractor_index"] == index
                assert audit["attempts"] == audit["rejections"] + 1
                assert audit["accepted_candidate_index"] == audit["rejections"]
                assert audit["collision_free"] is True
                assert audit["accepted_min_center_separation_m"] > 0.08
                assert len(audit["candidate_sequence_sha256"]) == 64

            second_info = instance.reset(np.random.default_rng(seed))
            second_positions = np.stack(_free_body_xy_positions(instance))
            assert second_info == first_info
            np.testing.assert_array_equal(second_positions, first_positions)
    finally:
        instance.close()


def test_default_box_sampling_clears_all_objects_across_many_seeds() -> None:
    instance = PickPlace(TaskConfig(n_distractors=3))
    try:
        for seed in range(100):
            instance.reset(np.random.default_rng(seed))
            _assert_objects_clear_the_box_outer_footprint(instance)
    finally:
        instance.close()


@pytest.mark.parametrize(
    "bad_range",
    [
        0.5,
        (0.4,),
        (0.4, 0.5, 0.6),
        ([0.4], [0.5, 0.6]),
        ("left", "right"),
        (0.4, np.inf),
        (0.5, 0.5),
        (0.6, 0.5),
    ],
)
def test_reset_rejects_malformed_randomization_ranges_before_sampling(
    env: PickPlace,
    bad_range: object,
) -> None:
    with pytest.raises(ValueError, match=r"cube_x.*two finite numbers.*low < high"):
        env.reset(np.random.default_rng(0), ranges={"cube_x": bad_range})


def test_reset_rejects_non_mapping_and_unknown_randomization_ranges(
    env: PickPlace,
) -> None:
    with pytest.raises(TypeError, match="ranges must be a mapping"):
        env.reset(np.random.default_rng(0), ranges=[(0.4, 0.6)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown randomization range.*unknown"):
        env.reset(np.random.default_rng(0), ranges={"unknown": (0.4, 0.6)})


class _ConstantGenerator:
    """Deterministic sampler that exposes an unbounded retry loop quickly."""

    def __init__(self) -> None:
        self.calls = 0

    def uniform(self, low: float, high: float) -> float:
        self.calls += 1
        if self.calls > 600:
            raise AssertionError("distractor sampling exceeded its finite budget")
        return (float(low) + float(high)) / 2.0


def test_impossible_distractor_ranges_fail_after_a_frozen_attempt_budget() -> None:
    instance = PickPlace(TaskConfig(n_distractors=2))
    rng = _ConstantGenerator()
    try:
        with pytest.raises(
            RuntimeError,
            match=r"distractor 0.*256 attempts.*minimum separation 0\.08",
        ) as exc_info:
            instance.reset(
                rng,  # type: ignore[arg-type]
                ranges={"cube_x": (0.49, 0.51), "cube_y": (-0.01, 0.01)},
            )
        error = exc_info.value
        assert error.receipt_schema_version == "pick_place_reset_receipt_v1"  # type: ignore[attr-defined]
        assert error.sampler_version == "pick_place_collision_free_rejection_v1"  # type: ignore[attr-defined]
        assert error.candidate_hash_version == "sha256-canonical-json-array-v1"  # type: ignore[attr-defined]
        assert error.attempts == 256  # type: ignore[attr-defined]
        assert error.rejections == 256  # type: ignore[attr-defined]
        assert error.accepted_candidate_index is None  # type: ignore[attr-defined]
        assert error.collision_free is False  # type: ignore[attr-defined]
        assert len(error.candidate_ledger) == 256  # type: ignore[attr-defined]
        assert error.candidate_ledger[-1]["candidate_index"] == 255  # type: ignore[attr-defined]
        assert error.candidate_sequence_sha256 == (  # type: ignore[attr-defined]
            "81514e8dbba4f9cd7c5a3413a9b4eba8978f3dffc59ce543b0a75051bd5af00e"
        )
    finally:
        instance.close()


def test_l1_boundary_ranges_cannot_spawn_a_cube_through_the_box_wall() -> None:
    instance = PickPlace()
    rng = _ConstantGenerator()
    try:
        with pytest.raises(
            RuntimeError,
            match=r"box.*256 attempts.*target and distractors",
        ) as exc_info:
            instance.reset(
                rng,  # type: ignore[arg-type]
                ranges={
                    "cube_x": (0.4999, 0.5001),
                    "cube_y": (0.2239, 0.2241),
                    "box_x": (0.4999, 0.5001),
                    "box_y": (0.2599, 0.2601),
                },
            )
        error = exc_info.value
        assert rng.calls == 515
        assert error.receipt_schema_version == "pick_place_reset_receipt_v1"  # type: ignore[attr-defined]
        assert error.sampler_version == "pick_place_collision_free_rejection_v1"  # type: ignore[attr-defined]
        assert error.candidate_hash_version == "sha256-canonical-json-array-v1"  # type: ignore[attr-defined]
        assert error.attempts == 256  # type: ignore[attr-defined]
        assert error.rejections == 256  # type: ignore[attr-defined]
        assert error.accepted_candidate_index is None  # type: ignore[attr-defined]
        assert error.collision_free is False  # type: ignore[attr-defined]
        assert len(error.candidate_ledger) == 256  # type: ignore[attr-defined]
        assert error.candidate_ledger[-1]["candidate_index"] == 255  # type: ignore[attr-defined]
        assert error.candidate_sequence_sha256 == (  # type: ignore[attr-defined]
            "0d31076cf69b759ce5271612eca89778936b8d4cb46f5f6ee2c2d7d7c62e2da9"
        )
    finally:
        instance.close()


def test_l1_locked_seeds_expose_every_conditioned_box_rejection() -> None:
    instance = PickPlace()
    rejection_counts: dict[int, int] = {}
    try:
        for seed in range(10000, 10050):
            info = instance.reset(
                np.random.default_rng(seed),
                ranges={"cube_x": (0.384, 0.636), "cube_y": (0.16, 0.224)},
            )
            audit = info["box_sampling"]
            assert audit["attempts"] == audit["rejections"] + 1
            assert audit["accepted_candidate_index"] == audit["rejections"]
            assert audit["collision_free"] is True
            assert audit["accepted_min_clearance_m"] > 0.0
            assert len(audit["candidate_sequence_sha256"]) == 64
            if audit["rejections"]:
                rejection_counts[seed] = audit["rejections"]
    finally:
        instance.close()

    assert rejection_counts == {
        10015: 1,
        10020: 1,
        10025: 4,
        10028: 1,
        10034: 1,
        10044: 1,
        10045: 1,
    }
    assert sum(rejection_counts.values()) == 10


def test_failed_physical_reset_cannot_retain_a_prior_success(env: PickPlace) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]
    floor_center_z = box[2] + BOX_WALL + CUBE_HALF
    _place_cube(env, xy=box[:2], z=floor_center_z)
    env.step(_home_action(env), physics_steps=env.required_success_steps)
    assert env.success()

    with pytest.raises(RuntimeError, match=r"box.*256 attempts"):
        env.reset(
            np.random.default_rng(0),
            ranges={
                "cube_x": (0.4999, 0.5001),
                "cube_y": (0.2239, 0.2241),
                "box_x": (0.4999, 0.5001),
                "box_y": (0.2599, 0.2601),
            },
        )

    assert not env.success()
    assert env.placement_status()["hold_steps"] == 0


def test_failed_distractor_reset_cannot_retain_a_prior_success() -> None:
    instance = PickPlace(TaskConfig(n_distractors=1))
    try:
        instance.reset(np.random.default_rng(0))
        box = instance.model.body_pos[instance.bid_box]
        floor_center_z = box[2] + BOX_WALL + CUBE_HALF
        _place_cube(instance, xy=box[:2], z=floor_center_z)
        instance.step(
            _home_action(instance), physics_steps=instance.required_success_steps
        )
        assert instance.success()

        with pytest.raises(RuntimeError, match=r"distractor 0.*256 attempts"):
            instance.reset(
                _ConstantGenerator(),  # type: ignore[arg-type]
                ranges={"cube_x": (0.49, 0.51), "cube_y": (-0.01, 0.01)},
            )

        assert not instance.success()
        assert instance.placement_status()["hold_steps"] == 0
    finally:
        instance.close()


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
    status = env.placement_status()
    assert status["ready"]
    assert status["fully_contained"]
    assert status["corner_clearance_xp"] == pytest.approx(0.037)
    assert status["corner_clearance_xn"] == pytest.approx(0.037)
    assert status["corner_clearance_yp"] == pytest.approx(0.037)
    assert status["corner_clearance_yn"] == pytest.approx(0.037)
    assert status["min_corner_wall_clearance"] == pytest.approx(0.037)
    assert status["min_corner_bottom_clearance"] == pytest.approx(0.0)
    assert status["min_corner_top_clearance"] == pytest.approx(0.004)
    assert not env.success()

    required_steps = math.ceil(env.cfg.success_hold_s / env.model.opt.timestep)
    env.step(_home_action(env), physics_steps=required_steps - 1)
    assert not env.success()
    result = env.step(_home_action(env), physics_steps=1)
    assert env.success()
    assert result["fully_contained"]
    assert result["min_corner_wall_clearance"] > 0.0


def test_rotated_cube_corner_outside_inner_wall_never_accumulates_success(
    env: PickPlace,
) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]
    floor_center_z = box[2] + BOX_WALL + CUBE_HALF
    yaw = np.pi / 4
    _place_cube(
        env,
        xy=box[:2] + [0.039, 0.0],
        z=floor_center_z,
        quaternion=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    )

    status = env.placement_status()
    assert status["corner_clearance_xp"] == pytest.approx(-0.0102842712474619)
    assert not status["inside_xy"]
    assert not status["fully_contained"]
    assert not status["ready"]

    result = env.step(
        _home_action(env),
        physics_steps=env.required_success_steps,
    )
    assert not result["success"]
    assert result["hold_steps"] == 0


def test_tilted_cube_top_must_be_below_the_actual_wall_top(env: PickPlace) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]
    roll = np.pi / 4
    vertical_half_extent = math.sqrt(2.0) * CUBE_HALF
    base_top = box[2] + BOX_WALL
    _place_cube(
        env,
        xy=box[:2],
        z=base_top + vertical_half_extent,
        quaternion=(math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0),
    )

    status = env.placement_status()
    assert status["inside_xy"]
    assert status["near_bottom"]
    assert status["min_corner_bottom_clearance"] == pytest.approx(0.0)
    assert status["min_corner_top_clearance"] == pytest.approx(-0.0125685424949238)
    assert not status["below_wall_top"]
    assert not status["fully_contained"]
    assert not status["ready"]


def test_containment_uses_compiled_cube_geom_size_and_wall_inner_face(
    env: PickPlace,
) -> None:
    env.reset(np.random.default_rng(0))
    box = env.model.body_pos[env.bid_box]
    cube_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    positive_x_wall = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "box_xp")

    env.model.geom_size[cube_geom, 0] = 0.03
    env.model.geom_size[positive_x_wall, 0] = 0.01
    floor_center_z = box[2] + BOX_WALL + CUBE_HALF
    _place_cube(env, xy=box[:2] + [0.021, 0.0], z=floor_center_z)

    status = env.placement_status()
    assert status["corner_clearance_xp"] == pytest.approx(-0.001)
    assert not status["inside_xy"]
    assert not status["fully_contained"]
    assert not status["ready"]


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
