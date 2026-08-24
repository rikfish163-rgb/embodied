"""Pre-registered M4 trial plans; this module never mutates an environment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

# Reserved evaluation seeds. Their disjointness from train/validation data must
# be checked against the final collection manifest before a formal evaluation.
M4_EVAL_SEEDS = tuple(range(10_000, 10_050))
REACTIVITY_K_EXEC = (1, 4, 8, 16)
M4_PROTOCOL_ID = "m4-reactivity-robustness-v1"
M4_SELECTED_K_EXEC = 8
M4_ACTION_DIM = 8
M4_ACTION_SCALE = (1.0,) * M4_ACTION_DIM
M4_ACTION_REPRESENTATION = "joint_position_targets_plus_normalized_gripper"
M4_ACTION_COMPONENTS = tuple(
    {"name": f"joint_{index}_target", "unit": "radian"} for index in range(1, 8)
) + ({"name": "gripper_opening_target", "unit": "normalized"},)


@dataclass(frozen=True)
class PerturbationSpec:
    """A declarative perturbation request, not executable environment code."""

    kind: str
    parameters: dict[str, Any]
    trigger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "parameters": self.parameters,
            "trigger": self.trigger,
        }
        _require_json(payload)
        # A JSON round trip also prevents callers mutating our nested constants.
        return json.loads(json.dumps(payload, allow_nan=False))


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    description: str
    perturbation: PerturbationSpec
    enabled: bool = True
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "perturbation": self.perturbation.to_dict(),
        }


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    policy_id: str
    checkpoint_id: str
    condition_id: str
    seed: int
    k_exec: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "policy_id": self.policy_id,
            "checkpoint_id": self.checkpoint_id,
            "condition_id": self.condition_id,
            "seed": self.seed,
            "k_exec": self.k_exec,
        }


@dataclass(frozen=True)
class EvaluationPlan:
    protocol_id: str
    protocol_mode: str
    customization_reasons: tuple[str, ...]
    policy_id: str
    checkpoint_id: str
    seeds: tuple[int, ...]
    reactivity_k_exec: tuple[int, ...]
    selected_k_exec: int
    conditions: dict[str, ConditionSpec]
    reactivity_trials: tuple[TrialSpec, ...]
    robustness_trials: tuple[TrialSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "evaluation-plan.v1",
            "protocol_id": self.protocol_id,
            "protocol_mode": self.protocol_mode,
            "customization_reasons": list(self.customization_reasons),
            "policy_id": self.policy_id,
            "checkpoint_id": self.checkpoint_id,
            "seeds": list(self.seeds),
            "reactivity_k_exec": list(self.reactivity_k_exec),
            "selected_k_exec": self.selected_k_exec,
            "seed_disjointness": {
                "status": "unchecked",
                "collection_manifest_id": None,
            },
            "reset_admission": _reset_admission(),
            "action_spec": _action_spec(),
            "conditions": {
                name: condition.to_dict()
                for name, condition in sorted(self.conditions.items())
            },
            "reactivity_trials": [trial.to_dict() for trial in self.reactivity_trials],
            "robustness_trials": [trial.to_dict() for trial in self.robustness_trials],
            "planned_trials": len(self.reactivity_trials) + len(self.robustness_trials),
        }
        _require_json(payload)
        return payload


def build_m4_plan(
    *,
    policy_id: str,
    checkpoint_id: str,
    seeds: tuple[int, ...] = M4_EVAL_SEEDS,
    reactivity_k_exec: tuple[int, ...] = REACTIVITY_K_EXEC,
    selected_k_exec: int,
    include_language: bool = False,
) -> EvaluationPlan:
    """Build a paired-seed M4 plan without running any trial.

    Only the exact v1 seed, K, condition, and action contract receives the
    frozen protocol identity. Any override remains useful for diagnostics but
    receives a deterministic, explicitly non-formal identity.
    """

    policy_id = _canonical_id("policy_id", policy_id)
    checkpoint_id = _canonical_id("checkpoint_id", checkpoint_id)
    if not isinstance(include_language, bool):
        raise TypeError("include_language must be bool")
    seeds = _validated_unique_non_negative_ints("seeds", seeds)
    reactivity_k_exec = _validated_unique_positive_ints(
        "reactivity_k_exec", reactivity_k_exec
    )
    _require_positive_int("selected_k_exec", selected_k_exec)

    conditions = _condition_specs(include_language=include_language)
    customization_reasons = []
    if seeds != M4_EVAL_SEEDS:
        customization_reasons.append("seeds")
    if reactivity_k_exec != REACTIVITY_K_EXEC:
        customization_reasons.append("reactivity_k_exec")
    if selected_k_exec != M4_SELECTED_K_EXEC:
        customization_reasons.append("selected_k_exec")
    if conditions != _condition_specs(include_language=False):
        customization_reasons.append("conditions")

    if customization_reasons:
        protocol_mode = "custom_non_formal"
        protocol_id = _custom_protocol_id(
            seeds=seeds,
            reactivity_k_exec=reactivity_k_exec,
            selected_k_exec=selected_k_exec,
            conditions=conditions,
        )
    else:
        protocol_mode = "frozen"
        protocol_id = M4_PROTOCOL_ID

    reactivity_trials = tuple(
        TrialSpec(
            trial_id=f"R1-k{k_exec:03d}-s{seed:05d}",
            policy_id=policy_id,
            checkpoint_id=checkpoint_id,
            condition_id="R1",
            seed=seed,
            k_exec=k_exec,
        )
        for k_exec in reactivity_k_exec
        for seed in seeds
    )
    robustness_condition_ids = ["L0", "L1", "L2", "L3", "L4"]
    if include_language:
        robustness_condition_ids.append("L5")
    robustness_trials = tuple(
        TrialSpec(
            trial_id=f"{condition_id}-k{selected_k_exec:03d}-s{seed:05d}",
            policy_id=policy_id,
            checkpoint_id=checkpoint_id,
            condition_id=condition_id,
            seed=seed,
            k_exec=selected_k_exec,
        )
        for condition_id in robustness_condition_ids
        for seed in seeds
    )
    return EvaluationPlan(
        protocol_id=protocol_id,
        protocol_mode=protocol_mode,
        customization_reasons=tuple(customization_reasons),
        policy_id=policy_id,
        checkpoint_id=checkpoint_id,
        seeds=seeds,
        reactivity_k_exec=reactivity_k_exec,
        selected_k_exec=selected_k_exec,
        conditions=conditions,
        reactivity_trials=reactivity_trials,
        robustness_trials=robustness_trials,
    )


def _action_spec() -> dict[str, Any]:
    return {
        "representation": M4_ACTION_REPRESENTATION,
        "dimension": M4_ACTION_DIM,
        "components": [dict(component) for component in M4_ACTION_COMPONENTS],
        "scale": list(M4_ACTION_SCALE),
    }


def _custom_protocol_id(
    *,
    seeds: tuple[int, ...],
    reactivity_k_exec: tuple[int, ...],
    selected_k_exec: int,
    conditions: dict[str, ConditionSpec],
) -> str:
    identity_payload = {
        "schema_version": "evaluation-plan.v1",
        "seeds": list(seeds),
        "reactivity_k_exec": list(reactivity_k_exec),
        "selected_k_exec": selected_k_exec,
        "reset_admission": _reset_admission(),
        "conditions": {
            name: condition.to_dict() for name, condition in sorted(conditions.items())
        },
        "action_spec": _action_spec(),
    }
    encoded = json.dumps(
        identity_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"custom-m4-reactivity-robustness-{digest}"


def assert_disjoint_seeds(
    evaluation_seeds: tuple[int, ...],
    occupied_seeds: list[int] | tuple[int, ...] | set[int],
) -> None:
    """Fail if reserved evaluation seeds occur in train/validation data."""

    eval_values = set(
        _validated_unique_non_negative_ints("evaluation_seeds", evaluation_seeds)
    )
    occupied_values = set(
        _validated_non_negative_ints("occupied_seeds", tuple(occupied_seeds))
    )
    overlap = sorted(eval_values & occupied_values)
    if overlap:
        raise ValueError(f"seed overlap with train/validation manifest: {overlap}")


def _reset_admission() -> dict[str, Any]:
    return {
        "algorithm": "pick_place_collision_free_rejection_v1",
        "receipt_schema_version": "pick_place_reset_receipt_v1",
        "sampler_version": "pick_place_collision_free_rejection_v1",
        "candidate_hash": {
            "version": "sha256-canonical-json-array-v1",
            "algorithm": "sha256",
            "payload": "json_array",
            "candidate_fields": [
                "candidate_index",
                "collision_free",
                "xy",
            ],
            "candidate_index_origin": 0,
            "json_encoding": {
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": True,
                "allow_nan": False,
            },
        },
        "rng": {
            "api": "numpy.random.default_rng",
            "bit_generator": "PCG64",
            "numpy_version": "bound_by_environment_provenance",
            "trial_seed_source": "trial.seed",
            "draw_schedule": [
                "target_xy_from_condition_proposal",
                "target_yaw_uniform_negative_pi_over_4_to_positive_pi_over_4",
                "each_distractor_xy_retry_until_accept",
                "accepted_distractor_yaw_uniform_negative_pi_to_positive_pi",
                "box_xy_retry_until_accept",
            ],
        },
        "ranges": {
            "interval_semantics": (
                "numpy_uniform_low_inclusive_high_exclusive_subject_to_rounding"
            ),
            "training_cube_x_m": [0.42, 0.60],
            "training_cube_y_m": [-0.16, 0.16],
            "l1_outer_cube_x_m": [0.384, 0.636],
            "l1_outer_cube_y_m": [-0.224, 0.224],
            "box_x_m": [0.44, 0.56],
            "box_y_m": [0.26, 0.36],
            "default_proposal_id": "training_rectangle_uniform_conditioned",
            "l1_proposal_id": (
                "outer_minus_training_area_weighted_uniform_conditioned"
            ),
            "l1_partition": ["left", "right", "bottom", "top"],
            "conditioning": "collision_free_joint_admission",
        },
        "admission": {
            "object_order": [
                "target",
                "distractor_index_ascending",
                "box",
            ],
            "distractor_distance": "center_xy_l2",
            "cube_half_extent_m": 0.02,
            "distractor_minimum_separation_m": 0.08,
            "distractor_comparison": ">",
            "box_geometry": ("actual_rotated_geom_xy_aabb_vs_four_wall_outer_envelope"),
            "box_outer_half_extent_m": 0.063,
            "box_minimum_clearance_m": 1e-9,
            "box_comparison": ">",
        },
        "requires_collision_free": [
            "target_vs_box",
            "distractors_vs_box",
            "target_vs_distractors",
            "distractor_pairs",
        ],
        "max_attempts_per_distractor": 256,
        "max_box_attempts": 256,
        "exhaustion_classification": "invalid_reset",
        "include_in_metric_denominator": False,
        "receipt": {
            "top_level_fields": [
                "schema_version",
                "sampler_version",
                "candidate_hash_version",
                "seed",
                "condition_id",
                "proposal_id",
                "effective_ranges",
                "rng",
                "collision_free",
                "target_sampling",
                "distractor_sampling",
                "box_sampling",
            ],
            "target_fields": [
                "accepted_xy",
                "accepted_yaw_rad",
                "partition",
            ],
            "distractor_fields": [
                "distractor_index",
                "attempts",
                "rejections",
                "accepted_candidate_index",
                "accepted_xy",
                "accepted_yaw_rad",
                "collision_free",
                "accepted_min_center_separation_m",
                "candidate_ledger",
                "candidate_sequence_sha256",
            ],
            "box_fields": [
                "attempts",
                "rejections",
                "accepted_candidate_index",
                "accepted_xy",
                "collision_free",
                "accepted_min_clearance_m",
                "candidate_ledger",
                "candidate_sequence_sha256",
            ],
            "candidate_ledger_fields": [
                "candidate_index",
                "collision_free",
                "xy",
            ],
            "success_invariants": [
                "attempts_equals_rejections_plus_one",
                "accepted_candidate_index_equals_rejections",
                "candidate_sequence_sha256_is_lower_hex_64",
                "accepted_clearance_strictly_exceeds_threshold",
            ],
            "exhaustion_invariants": [
                "attempts_equals_rejections_equals_budget",
                "accepted_candidate_index_is_null",
                "collision_free_is_false",
                "whole_formal_run_fails_without_retry_or_attrition",
            ],
        },
    }


def _condition_specs(*, include_language: bool) -> dict[str, ConditionSpec]:
    no_perturbation = PerturbationSpec(
        kind="none",
        parameters={},
        trigger={"event": "reset"},
    )
    return {
        "L0": ConditionSpec(
            condition_id="L0",
            description="IID baseline; training-distribution reset ranges",
            perturbation=no_perturbation,
        ),
        "L1": ConditionSpec(
            condition_id="L1",
            description="cube position in a 20%-expanded boundary shell",
            perturbation=PerturbationSpec(
                kind="cube_position_boundary_shell",
                parameters={
                    "training_cube_x_m": [0.42, 0.60],
                    "training_cube_y_m": [-0.16, 0.16],
                    "outer_cube_x_m": [0.384, 0.636],
                    "outer_cube_y_m": [-0.224, 0.224],
                    "expansion_per_side_fraction": 0.20,
                    "sampling_rule": (
                        "outer_rectangle_minus_training_rectangle_then_"
                        "collision_free_admission"
                    ),
                },
                trigger={"event": "reset"},
            ),
        ),
        "L2": ConditionSpec(
            condition_id="L2",
            description="two non-target cube distractors",
            perturbation=PerturbationSpec(
                kind="distractor_count",
                parameters={"n_distractors": 2},
                trigger={"event": "reset"},
            ),
        ),
        "L3": ConditionSpec(
            condition_id="L3",
            description="fixed side-light reduction and table texture swap",
            perturbation=PerturbationSpec(
                kind="lighting_texture",
                parameters={
                    "key_light_intensity_multiplier": 0.5,
                    "key_light_direction_xyz": [1.0, -1.0, -1.0],
                    "table_texture_id": "eval_checker_v1",
                },
                trigger={"event": "reset"},
            ),
        ),
        "L4": ConditionSpec(
            condition_id="L4",
            description="fixed camera extrinsic offset outside the zero-jitter training range",
            perturbation=PerturbationSpec(
                kind="camera_pose_offset",
                parameters={
                    "cameras": ["front", "wrist"],
                    "translation_xyz_m": [0.02, 0.0, 0.0],
                    "rotation_rpy_deg": [0.0, 5.0, 0.0],
                },
                trigger={"event": "reset"},
            ),
        ),
        "L5": ConditionSpec(
            condition_id="L5",
            description="red instruction token replaced with crimson",
            perturbation=PerturbationSpec(
                kind="instruction_substitution",
                parameters={"from": "red", "to": "crimson"},
                trigger={"event": "episode_start"},
            ),
            enabled=include_language,
            disabled_reason=None
            if include_language
            else "single-color task has no meaningful language dimension",
        ),
        "R1": ConditionSpec(
            condition_id="R1",
            description="3 cm lateral cube translation before grasp",
            perturbation=PerturbationSpec(
                kind="cube_translation",
                parameters={"delta_xyz_m": [0.0, 0.03, 0.0]},
                trigger={
                    "event": "stage_entry",
                    "stage": "pregrasp",
                    "requires_cube_ungrasped": True,
                },
            ),
        ),
    }


def _validated_unique_non_negative_ints(
    name: str, values: tuple[int, ...]
) -> tuple[int, ...]:
    result = _validated_non_negative_ints(name, values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _validated_unique_positive_ints(
    name: str, values: tuple[int, ...]
) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _require_positive_int(name, value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(values)


def _validated_non_negative_ints(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(values)
    for value in result:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integers")
        if value < 0:
            raise ValueError(f"{name} must contain non-negative integers")
    return result


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _canonical_id(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("protocol fields must be finite JSON values") from error
