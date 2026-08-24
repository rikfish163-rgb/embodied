from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from env.pick_place import BOX_INNER, BOX_WALL, PickPlace
from evaluation.metrics import (
    action_smoothness,
    failure_taxonomy,
    latency_percentiles,
    wilson_interval,
)
from evaluation.protocol import (
    M4_ACTION_DIM,
    M4_ACTION_SCALE,
    M4_EVAL_SEEDS,
    M4_PROTOCOL_ID,
    M4_SELECTED_K_EXEC,
    REACTIVITY_K_EXEC,
    assert_disjoint_seeds,
    build_m4_plan,
)
from evaluation.report import build_report, main as report_main


def test_wilson_interval_handles_known_value_and_boundaries() -> None:
    interval = wilson_interval(8, 10)

    assert interval["confidence"] == pytest.approx(0.95)
    assert interval["low"] == pytest.approx(0.4901624715)
    assert interval["high"] == pytest.approx(0.9433178485)
    assert wilson_interval(0, 10)["low"] == 0.0
    assert wilson_interval(10, 10)["high"] == 1.0


def _valid_reset_receipt(trial: dict[str, object]) -> dict[str, object]:
    condition_id = trial["condition_id"]
    is_l1 = condition_id == "L1"
    distractor_count = 2 if condition_id == "L2" else 0
    target_xy = [0.5, 0.2] if is_l1 else [0.5, 0.0]
    target_yaw = 0.0
    accepted_positions = [target_xy]
    distractor_sampling = []
    distractor_positions = ([0.42, -0.16], [0.59, 0.15])
    for index in range(distractor_count):
        accepted_xy = distractor_positions[index]
        clearance = min(
            math.dist(accepted_xy, previous) for previous in accepted_positions
        )
        candidate_ledger = [
            {
                "candidate_index": 0,
                "collision_free": True,
                "xy": accepted_xy,
            }
        ]
        distractor_sampling.append(
            {
                "distractor_index": index,
                "attempts": 1,
                "rejections": 0,
                "accepted_candidate_index": 0,
                "accepted_xy": accepted_xy,
                "accepted_yaw_rad": 0.0,
                "collision_free": True,
                "accepted_min_center_separation_m": clearance,
                "candidate_ledger": candidate_ledger,
                "candidate_sequence_sha256": hashlib.sha256(
                    json.dumps(
                        candidate_ledger,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        accepted_positions.append(accepted_xy)
    box_xy = [0.5, 0.35]
    box_candidate_ledger = [
        {"candidate_index": 0, "collision_free": True, "xy": box_xy}
    ]
    box_clearance = 0.117 if distractor_count else (0.067 if is_l1 else 0.267)
    return {
        "schema_version": "pick_place_reset_receipt_v1",
        "sampler_version": "pick_place_collision_free_rejection_v1",
        "candidate_hash_version": "sha256-canonical-json-array-v1",
        "seed": trial["seed"],
        "condition_id": condition_id,
        "proposal_id": (
            "outer_minus_training_area_weighted_uniform_conditioned"
            if is_l1
            else "training_rectangle_uniform_conditioned"
        ),
        "effective_ranges": {
            "cube_x_m": [0.384, 0.636] if is_l1 else [0.42, 0.60],
            "cube_y_m": [-0.224, 0.224] if is_l1 else [-0.16, 0.16],
            "box_x_m": [0.44, 0.56],
            "box_y_m": [0.26, 0.36],
        },
        "rng": {
            "api": "numpy.random.default_rng",
            "bit_generator": "PCG64",
        },
        "collision_free": True,
        "target_sampling": {
            "accepted_xy": target_xy,
            "accepted_yaw_rad": target_yaw,
            "partition": "top" if is_l1 else "training_rectangle",
        },
        "distractor_sampling": distractor_sampling,
        "box_sampling": {
            "attempts": 1,
            "rejections": 0,
            "accepted_candidate_index": 0,
            "accepted_xy": box_xy,
            "collision_free": True,
            "accepted_min_clearance_m": box_clearance,
            "candidate_ledger": box_candidate_ledger,
            "candidate_sequence_sha256": hashlib.sha256(
                json.dumps(
                    box_candidate_ledger,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


class _ScriptedResetGenerator:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def uniform(self, low: float, high: float) -> float:
        value = next(self._values)
        assert float(low) <= value <= float(high)
        return value


def test_report_verifies_complete_reset_receipt_against_frozen_contract() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L2"
    )

    report = build_report(
        [{**trial, "success": True, "reset_receipt": _valid_reset_receipt(trial)}],
        protocol=protocol,
    )

    evidence = report["completion"]["reset_admission_evidence"]
    assert evidence == {
        "required": True,
        "complete": False,
        "content_recomputed": True,
        "runner_provenance_verified": False,
        "observed_record_count": 1,
        "valid_record_count": 1,
        "missing_record_count": 0,
        "invalid_record_count": 0,
        "missing_trial_ids": [],
        "invalid_records": [],
    }
    assert (
        "reset_admission_unverified"
        in report["completion"]["formal_readiness_blockers"]
    )
    assert report["qualified_episode_count"] == 1


def test_report_accepts_environment_receipt_at_true_box_outer_boundary() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L1"
    )
    env = PickPlace()
    try:
        reset_info = env.reset(
            _ScriptedResetGenerator([0.5, 0.176, 0.0, 0.5, 0.26]),
            ranges={"cube_x": (0.384, 0.636), "cube_y": (-0.224, 0.224)},
        )
    finally:
        env.close()

    assert reset_info["box_sampling"]["accepted_min_clearance_m"] == pytest.approx(
        0.001
    )
    assert protocol["reset_admission"]["admission"][
        "box_outer_half_extent_m"
    ] == pytest.approx(BOX_INNER + BOX_WALL / 2)
    receipt = {
        "schema_version": reset_info["receipt_schema_version"],
        "sampler_version": reset_info["sampler_version"],
        "candidate_hash_version": reset_info["candidate_hash_version"],
        "seed": trial["seed"],
        "condition_id": trial["condition_id"],
        "proposal_id": ("outer_minus_training_area_weighted_uniform_conditioned"),
        "effective_ranges": {
            "cube_x_m": [0.384, 0.636],
            "cube_y_m": [-0.224, 0.224],
            "box_x_m": [0.44, 0.56],
            "box_y_m": [0.26, 0.36],
        },
        "rng": {"api": "numpy.random.default_rng", "bit_generator": "PCG64"},
        "collision_free": reset_info["collision_free"],
        "target_sampling": {
            **reset_info["target_sampling"],
            "partition": "top",
        },
        "distractor_sampling": reset_info["distractor_sampling"],
        "box_sampling": reset_info["box_sampling"],
    }

    report = build_report(
        [{**trial, "success": True, "reset_receipt": receipt}],
        protocol=protocol,
    )

    evidence = report["completion"]["reset_admission_evidence"]
    assert evidence["valid_record_count"] == 1
    assert evidence["invalid_records"] == []
    assert report["qualified_episode_count"] == 1


def test_report_rejects_l1_target_inside_training_rectangle() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L1"
    )
    receipt = _valid_reset_receipt(trial)
    receipt["target_sampling"]["accepted_xy"] = [0.5, 0.0]  # type: ignore[index]

    report = build_report(
        [{**trial, "success": True, "reset_receipt": receipt}],
        protocol=protocol,
    )

    invalid = report["completion"]["reset_admission_evidence"]["invalid_records"]
    assert (
        "reset_receipt.target_sampling.accepted_xy:not_in_l1_outer_shell"
        in (invalid[0]["errors"])
    )
    assert report["qualified_episode_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda receipt: receipt["box_sampling"].update(attempts=2),
            "reset_receipt.box_sampling:attempts_must_equal_rejections_plus_one",
        ),
        (
            lambda receipt: receipt["box_sampling"].update(
                candidate_sequence_sha256="c" * 64
            ),
            "reset_receipt.box_sampling.candidate_sequence_sha256:content_mismatch",
        ),
        (
            lambda receipt: receipt["box_sampling"].update(
                accepted_min_clearance_m=99.0
            ),
            "reset_receipt.box_sampling.accepted_min_clearance_m:recomputed_value_mismatch",
        ),
        (
            lambda receipt: receipt.update(proposal_id="unconditioned_uniform"),
            "reset_receipt.proposal_id:conflicts_with_condition",
        ),
    ],
)
def test_report_excludes_tampered_reset_receipt_from_metrics(
    mutation: object,
    expected_error: str,
) -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L0"
    )
    receipt = _valid_reset_receipt(trial)
    mutation(receipt)  # type: ignore[operator]

    report = build_report(
        [{**trial, "success": True, "reset_receipt": receipt}],
        protocol=protocol,
    )

    evidence = report["completion"]["reset_admission_evidence"]
    assert evidence["complete"] is False
    assert evidence["invalid_record_count"] == 1
    assert expected_error in evidence["invalid_records"][0]["errors"]
    assert (
        "reset_admission_invalid" in report["completion"]["formal_readiness_blockers"]
    )
    assert report["qualified_episode_count"] == 0
    assert report["metrics"]["success"]["trials"] == 0


def test_report_recomputes_l2_pairwise_candidate_admission() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L2"
    )
    receipt = _valid_reset_receipt(trial)
    sample = receipt["distractor_sampling"][1]  # type: ignore[index]
    sample["accepted_xy"] = [0.43, -0.16]
    sample["accepted_min_center_separation_m"] = 0.01
    sample["candidate_ledger"][0]["xy"] = [0.43, -0.16]
    sample["candidate_sequence_sha256"] = hashlib.sha256(
        json.dumps(
            sample["candidate_ledger"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    report = build_report(
        [{**trial, "success": True, "reset_receipt": receipt}],
        protocol=protocol,
    )

    errors = report["completion"]["reset_admission_evidence"]["invalid_records"][0][
        "errors"
    ]
    assert (
        "reset_receipt.distractor_sampling[1].candidate_ledger[0].collision_free:geometry_mismatch"
        in errors
    )
    assert "reset_receipt.distractor_sampling[1].accepted_xy:not_collision_free" in (
        errors
    )
    assert report["qualified_episode_count"] == 0


@pytest.mark.parametrize("component", ["distractor", "box"])
def test_report_rejects_self_consistent_candidate_ledger_outside_frozen_ranges(
    component: str,
) -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    condition = "L2" if component == "distractor" else "L0"
    trial = next(
        item
        for item in protocol["robustness_trials"]
        if item["condition_id"] == condition
    )
    receipt = _valid_reset_receipt(trial)
    if component == "distractor":
        sample = receipt["distractor_sampling"][0]  # type: ignore[index]
        sample["accepted_xy"] = [9.0, 9.0]
        sample["accepted_min_center_separation_m"] = math.dist([9.0, 9.0], [0.5, 0.0])
        expected_error = (
            "reset_receipt.distractor_sampling[0].candidate_ledger[0].xy:"
            "outside_protocol_range"
        )
    else:
        sample = receipt["box_sampling"]  # type: ignore[assignment]
        sample["accepted_xy"] = [9.0, 9.0]
        sample["accepted_min_clearance_m"] = 8.414
        expected_error = (
            "reset_receipt.box_sampling.candidate_ledger[0].xy:outside_protocol_range"
        )
    sample["candidate_ledger"][0]["xy"] = [9.0, 9.0]
    sample["candidate_sequence_sha256"] = hashlib.sha256(
        json.dumps(
            sample["candidate_ledger"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    report = build_report(
        [{**trial, "success": True, "reset_receipt": receipt}],
        protocol=protocol,
    )

    errors = report["completion"]["reset_admission_evidence"]["invalid_records"][0][
        "errors"
    ]
    assert expected_error in errors
    assert report["qualified_episode_count"] == 0


def test_observed_trial_identity_whitespace_is_not_canonicalized_to_match() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trial = next(
        item for item in protocol["robustness_trials"] if item["condition_id"] == "L0"
    )

    report = build_report(
        [
            {
                **trial,
                "trial_id": f" {trial['trial_id']}",
                "condition_id": f"{trial['condition_id']} ",
                "success": True,
            }
        ],
        protocol=protocol,
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert evidence["invalid_record_count"] == 1
    assert "trial_id_must_be_canonical" in evidence["invalid_records"][0]["errors"]
    assert "condition_id_must_be_canonical" in evidence["invalid_records"][0]["errors"]
    assert report["qualified_episode_count"] == 0


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(-1, 10), (11, 10), (0, 0), (True, 10)],
)
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        wilson_interval(successes, trials)


def test_latency_percentiles_use_all_finite_non_negative_samples() -> None:
    summary = latency_percentiles([1, 2, 3, 4, 5])

    assert summary == {
        "count": 5,
        "p50": 3.0,
        "p95": pytest.approx(4.8),
        "p99": pytest.approx(4.96),
    }
    assert latency_percentiles([]) == {
        "count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    with pytest.raises(ValueError, match="non-negative"):
        latency_percentiles([1.0, -0.1])


def test_action_smoothness_is_mean_scaled_l2_first_difference() -> None:
    # Each transition is [3, 4], whose L2 norm is 5.
    summary = action_smoothness([[0, 0], [3, 4], [6, 8]])

    assert summary["metric"] == "mean_scaled_l2_delta"
    assert summary["transitions"] == 2
    assert summary["value"] == pytest.approx(5.0)
    assert summary["action_dim"] == 2
    assert action_smoothness([[1, 2]])["value"] is None

    scaled = action_smoothness([[0, 0], [3, 4], [6, 8]], action_scale=[3, 4])
    assert scaled["value"] == pytest.approx(2**0.5)


def test_failure_taxonomy_is_cross_tabulated_by_execution_stage() -> None:
    records = [
        {"success": True},
        {
            "success": False,
            "failure_stage": "pregrasp",
            "failure_type": "empty_grasp",
        },
        {
            "success": False,
            "failure_stage": "lift",
            "failure_type": "object_slip",
        },
        {"success": False, "failure_stage": "lift"},
    ]

    summary = failure_taxonomy(records)

    assert summary["total_failures"] == 3
    assert summary["by_stage"] == {"lift": 2, "pregrasp": 1}
    assert summary["by_type"] == {
        "empty_grasp": 1,
        "object_slip": 1,
        "unclassified": 1,
    }
    assert summary["stage_by_type"]["lift"] == {
        "object_slip": 1,
        "unclassified": 1,
    }


@dataclass
class _Step:
    action: tuple[float, ...]


@dataclass
class _EpisodeObject:
    seed: int
    success: bool
    failure_stage: str | None
    failure_type: str | None
    steps: tuple[_Step, ...]


def test_report_accepts_mapping_and_expert_shaped_episode_objects() -> None:
    records = [
        {
            "episode_id": "act-L0-10000",
            "seed": 10_000,
            "condition_id": "L0",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "k_exec": 8,
            "success": True,
            "actions": [[0.0, 0.0], [1.0, 0.0]],
            "inference_latency_ms": [3.0, 5.0],
            "control_latency_ms": [50.0, 50.0],
        },
        _EpisodeObject(
            seed=10_001,
            success=False,
            failure_stage="lift",
            failure_type="object_slip",
            steps=(_Step((0.0, 0.0)), _Step((0.0, 2.0))),
        ),
    ]

    report = build_report(records, protocol={"protocol_id": "unit-test"})

    assert report["schema_version"] == "evaluation-report.v1"
    assert report["metrics"]["success"]["successes"] == 1
    assert report["metrics"]["success"]["trials"] == 2
    assert report["metrics"]["failures"]["by_stage"] == {"lift": 1}
    assert report["metrics"]["latency_ms"]["inference"]["count"] == 2
    assert report["metrics"]["latency_ms"]["control"]["p95"] == 50.0
    assert report["metrics"]["action_smoothness"]["episodes"] == 2
    assert report["metrics"]["action_smoothness"]["value"] == pytest.approx(1.5)
    assert json.loads(json.dumps(report)) == report


def test_report_uses_protocol_action_scale_for_smoothness() -> None:
    report = build_report(
        [
            {
                "success": True,
                "actions": [[0.0, 0.0], [2.0, 4.0]],
                "action_scale": [2.0, 4.0],
            }
        ],
        protocol={
            "protocol_id": "unit-test",
            "action_spec": {"dimension": 2, "scale": [2.0, 4.0]},
        },
    )

    smoothness = report["metrics"]["action_smoothness"]
    assert smoothness["action_scale"] == [2.0, 4.0]
    assert smoothness["action_dimension"] == 2
    assert smoothness["value"] == pytest.approx(2**0.5)


def test_report_rejects_mixed_action_dimensions() -> None:
    report = build_report(
        [
            {"success": True, "actions": [[0.0, 0.0], [1.0, 1.0]]},
            {
                "success": True,
                "actions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            },
        ],
        protocol={"protocol_id": "unit-test", "planned_trials": 2},
    )

    assert report["completion"]["complete"] is False
    assert report["completion"]["record_schema_evidence"]["complete"] is False
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("records", "protocol", "message"),
    [
        (
            [
                {
                    "success": True,
                    "actions": [[0.0, 0.0], [1.0, 1.0]],
                    "action_scale": [1.0, 1.0],
                },
                {
                    "success": True,
                    "actions": [[0.0, 0.0], [1.0, 1.0]],
                    "action_scale": [2.0, 2.0],
                },
            ],
            {"protocol_id": "unit-test"},
            "mixed action scales",
        ),
        (
            [
                {
                    "success": True,
                    "actions": [[0.0, 0.0], [1.0, 1.0]],
                    "action_scale": [1.0, 1.0],
                }
            ],
            {
                "protocol_id": "unit-test",
                "action_spec": {"dimension": 2, "scale": [2.0, 2.0]},
            },
            "conflicts with protocol action scale",
        ),
        (
            [{"success": True, "actions": [[0.0, 0.0], [1.0, 1.0]]}],
            {
                "protocol_id": "unit-test",
                "action_spec": {"dimension": 3, "scale": [1.0, 1.0, 1.0]},
            },
            "action dimension conflicts with protocol",
        ),
        (
            [
                {"success": True, "actions": [[0.0, 0.0], [1.0, 1.0]]},
                {
                    "success": True,
                    "actions": [[0.0, 0.0], [1.0, 1.0]],
                    "action_scale": [2.0, 2.0],
                },
            ],
            {"protocol_id": "unit-test"},
            "mixed action scales",
        ),
    ],
)
def test_report_rejects_incompatible_action_contracts(
    records: list[dict[str, object]],
    protocol: dict[str, object],
    message: str,
) -> None:
    protocol["planned_trials"] = len(records)

    report = build_report(records, protocol=protocol)

    assert report["completion"]["complete"] is False
    assert any(
        message.replace(" ", "_") in error
        for invalid in report["completion"]["record_schema_evidence"]["invalid_records"]
        for error in invalid["errors"]
    )
    assert report["metrics"]["success"]["trials"] == 0


def test_frozen_protocol_rejects_boolean_scale_as_wrong_json_scalar_type() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    protocol["action_spec"]["scale"][0] = True
    records = [
        {**trial, "success": True}
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    completion = build_report(records, protocol=protocol)["completion"]

    assert completion["trial_coverage_complete"] is True
    assert completion["protocol_validation"]["valid"] is False
    assert completion["protocol_validation"]["formal"] is False
    assert (
        "protocol.action_spec.scale[0]:boolean_is_not_number"
        in completion["protocol_validation"]["errors"]
    )
    assert (
        "frozen_protocol_field_mismatch:action_spec"
        in completion["protocol_validation"]["errors"]
    )
    assert completion["complete"] is False


@pytest.mark.parametrize(
    ("action_spec", "expected_error"),
    [
        (7, "protocol.action_spec:must_be_object"),
        (
            {"dimension": True, "scale": [1.0]},
            "protocol.action_spec.dimension:boolean_is_not_integer",
        ),
        (
            {"dimension": 2, "scale": [1.0]},
            "protocol.action_spec.scale:length_must_match_dimension",
        ),
        (
            {"dimension": 1, "scale": [True]},
            "protocol.action_spec.scale[0]:boolean_is_not_number",
        ),
        (
            {"dimension": 1, "scale": [float("nan")]},
            "protocol.action_spec.scale[0]:non_finite_number",
        ),
        (
            {"dimension": 1, "scale": [float("inf")]},
            "protocol.action_spec.scale[0]:non_finite_number",
        ),
        (
            {"dimension": 1, "scale": [" 1 "]},
            "protocol.action_spec.scale[0]:must_be_number",
        ),
    ],
)
def test_malformed_action_spec_is_structured_protocol_invalid(
    action_spec: object,
    expected_error: str,
) -> None:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [{**trial, "success": True}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
            "action_spec": action_spec,
        },
    )

    completion = report["completion"]
    assert completion["protocol_validation"]["valid"] is False
    assert expected_error in completion["protocol_validation"]["errors"]
    assert completion["complete"] is False
    assert report["metrics"]["success"]["trials"] == 0


def test_metrics_exclude_unexpected_success_from_planned_failure() -> None:
    planned = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    unexpected = {**planned, "trial_id": "L0-k008-s10001", "seed": 10_001}
    protocol = {
        "protocol_id": "unit-test",
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        "planned_trials": 1,
        "reactivity_trials": [],
        "robustness_trials": [planned],
        "action_spec": {"dimension": 2, "scale": [1.0, 1.0]},
    }

    report = build_report(
        [
            {
                **planned,
                "success": False,
                "failure_stage": "lift",
                "failure_type": "object_slip",
                "actions": [[0.0, 0.0], [1.0, 0.0]],
            },
            {
                **unexpected,
                "success": True,
                "actions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            },
        ],
        protocol=protocol,
    )

    assert report["observed_episode_count"] == 2
    assert report["qualified_episode_count"] == 1
    assert report["metrics"]["success"]["successes"] == 0
    assert report["metrics"]["success"]["trials"] == 1
    assert report["metrics"]["action_smoothness"]["value"] == 1.0
    assert len(report["groups"]) == 1
    assert report["completion"]["unexpected"][0]["trial_id"] == ("L0-k008-s10001")
    assert report["completion"]["record_qualification"]["excluded_record_count"] == 1


def test_metrics_exclude_all_copies_of_duplicate_trial() -> None:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    failure = {
        **trial,
        "success": False,
        "failure_stage": "lift",
        "failure_type": "object_slip",
    }

    report = build_report(
        [failure, failure],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
        },
    )

    assert report["qualified_episode_count"] == 0
    assert report["metrics"]["success"] == {
        "successes": 0,
        "trials": 0,
        "rate": None,
        "wilson": None,
    }
    assert report["completion"]["record_qualification"]["excluded_record_count"] == 2


def test_non_finite_expected_action_is_structured_record_invalid() -> None:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [
            {
                **trial,
                "success": True,
                "actions": [[0.0, 0.0], [float("nan"), 1.0]],
            }
        ],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
            "action_spec": {"dimension": 2, "scale": [1.0, 1.0]},
        },
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert evidence["complete"] is False
    assert evidence["invalid_trial_ids"] == ["L0-k008-s10000"]
    assert "actions_must_be_finite" in evidence["invalid_records"][0]["errors"]
    assert report["completion"]["complete"] is False
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("actions", "expected_error"),
    [
        ([[0.0, True]], "actions_boolean_is_not_number"),
        ([[0.0, " 1 "]], "actions_must_be_numeric_matrix"),
        ([[0.0, float("inf")]], "actions_must_be_finite"),
        ([0.0, 1.0], "actions_must_have_time_by_dimension_shape"),
        ([], "actions_must_be_non_empty_matrix"),
    ],
)
def test_malformed_action_values_are_structured_record_invalid(
    actions: object,
    expected_error: str,
) -> None:
    report = build_report(
        [{"success": True, "actions": actions}],
        protocol={"protocol_id": "unit-test", "planned_trials": 1},
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert evidence["complete"] is False
    assert expected_error in evidence["invalid_records"][0]["errors"]
    assert report["completion"]["complete"] is False
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("action_scale", "expected_error"),
    [
        ([True, 1.0], "action_scale_boolean_is_not_number"),
        ([" 1 ", 1.0], "action_scale_must_be_numeric_vector"),
        ([float("nan"), 1.0], "action_scale_must_contain_positive_finite_values"),
        ([float("inf"), 1.0], "action_scale_must_contain_positive_finite_values"),
    ],
)
def test_malformed_episode_action_scale_is_structured_record_invalid(
    action_scale: object,
    expected_error: str,
) -> None:
    report = build_report(
        [
            {
                "success": True,
                "actions": [[0.0, 0.0], [1.0, 1.0]],
                "action_scale": action_scale,
            }
        ],
        protocol={"protocol_id": "unit-test", "planned_trials": 1},
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert expected_error in evidence["invalid_records"][0]["errors"]
    assert report["metrics"]["success"]["trials"] == 0


def test_action_scale_without_actions_is_audited_and_excluded() -> None:
    report = build_report(
        [
            {"success": True, "actions": [[0.0, 0.0], [1.0, 1.0]]},
            {"success": True, "action_scale": [2.0, 2.0]},
        ],
        protocol={"protocol_id": "unit-test", "planned_trials": 2},
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert evidence["invalid_record_count"] == 1
    assert "action_scale_without_actions" in evidence["invalid_records"][0]["errors"]
    assert report["qualified_episode_count"] == 1
    assert report["metrics"]["success"]["trials"] == 1


@pytest.mark.parametrize(
    "failure_metadata",
    [
        {"failure_stage": float("nan"), "failure_type": "timeout"},
        {"failure_stage": "lift", "failure_type": float("inf")},
        {"failure": {"stage": float("-inf"), "type": "timeout"}},
    ],
)
def test_non_finite_failure_labels_are_audited_without_breaking_json(
    failure_metadata: dict[str, object],
) -> None:
    report = _single_trial_outcome_report(
        success=False,
        failure_metadata=failure_metadata,
    )

    evidence = report["completion"]["outcome_evidence"]
    assert evidence["complete"] is False
    assert any(
        error.endswith("_must_be_string")
        for error in evidence["invalid_records"][0]["errors"]
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("latency", "expected_error"),
    [
        ([True], "inference_latency_ms_boolean_is_not_number"),
        ([" 1 "], "inference_latency_ms_must_be_numeric_vector"),
        ([float("nan")], "inference_latency_ms_must_be_finite"),
        ([float("inf")], "inference_latency_ms_must_be_finite"),
        ([-1.0], "inference_latency_ms_must_be_non_negative"),
    ],
)
def test_malformed_latency_is_structured_record_invalid(
    latency: object,
    expected_error: str,
) -> None:
    report = build_report(
        [{"success": True, "inference_latency_ms": latency}],
        protocol={"protocol_id": "unit-test", "planned_trials": 1},
    )

    evidence = report["completion"]["record_schema_evidence"]
    assert expected_error in evidence["invalid_records"][0]["errors"]
    assert report["completion"]["complete"] is False
    assert report["metrics"]["latency_ms"]["inference"]["count"] == 0


def test_fixed_m4_plan_pairs_seeds_and_records_perturbation_parameters() -> None:
    plan = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=8,
    )
    payload = plan.to_dict()

    assert payload["protocol_id"] == M4_PROTOCOL_ID
    assert payload["protocol_mode"] == "frozen"
    assert payload["seeds"] == list(M4_EVAL_SEEDS)
    assert payload["reactivity_k_exec"] == list(REACTIVITY_K_EXEC)
    assert payload["selected_k_exec"] == M4_SELECTED_K_EXEC
    assert payload["action_spec"] == {
        "representation": "joint_position_targets_plus_normalized_gripper",
        "dimension": M4_ACTION_DIM,
        "components": [
            {"name": f"joint_{index}_target", "unit": "radian"} for index in range(1, 8)
        ]
        + [{"name": "gripper_opening_target", "unit": "normalized"}],
        "scale": list(M4_ACTION_SCALE),
    }
    assert payload["seed_disjointness"] == {
        "status": "unchecked",
        "collection_manifest_id": None,
    }
    assert payload["reset_admission"] == {
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
            "box_minimum_clearance_m": 1e-09,
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
    assert (
        payload["conditions"]["L1"]["perturbation"]["parameters"]["sampling_rule"]
        == "outer_rectangle_minus_training_rectangle_then_collision_free_admission"
    )
    assert len(payload["reactivity_trials"]) == len(M4_EVAL_SEEDS) * len(
        REACTIVITY_K_EXEC
    )
    assert {
        trial["seed"] for trial in payload["reactivity_trials"] if trial["k_exec"] == 1
    } == set(M4_EVAL_SEEDS)

    perturbation = payload["conditions"]["R1"]["perturbation"]
    assert perturbation["kind"] == "cube_translation"
    assert perturbation["parameters"]["delta_xyz_m"] == [0.0, 0.03, 0.0]
    assert perturbation["trigger"] == {
        "event": "stage_entry",
        "stage": "pregrasp",
        "requires_cube_ungrasped": True,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_report_rejects_tampered_frozen_reset_admission() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    protocol["reset_admission"]["include_in_metric_denominator"] = True
    records = [
        {**trial, "success": True}
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    completion = build_report(records, protocol=protocol)["completion"]

    assert completion["complete"] is False
    assert completion["protocol_validation"]["formal"] is False
    assert (
        "frozen_protocol_field_mismatch:reset_admission"
        in completion["protocol_validation"]["errors"]
    )


@pytest.mark.parametrize(
    ("overrides", "custom_reason"),
    [
        ({"seeds": (7,)}, "seeds"),
        ({"reactivity_k_exec": (2,)}, "reactivity_k_exec"),
        ({"selected_k_exec": 4}, "selected_k_exec"),
        ({"include_language": True}, "conditions"),
    ],
)
def test_custom_m4_parameters_receive_non_formal_protocol_identity(
    overrides: dict[str, object],
    custom_reason: str,
) -> None:
    arguments = {
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        "selected_k_exec": M4_SELECTED_K_EXEC,
        **overrides,
    }

    payload = build_m4_plan(**arguments).to_dict()

    assert payload["protocol_id"] != M4_PROTOCOL_ID
    assert payload["protocol_id"].startswith("custom-m4-reactivity-robustness-")
    assert payload["protocol_mode"] == "custom_non_formal"
    assert custom_reason in payload["customization_reasons"]


def test_build_m4_plan_canonicalizes_frozen_policy_identity() -> None:
    payload = build_m4_plan(
        policy_id="  act  ",
        checkpoint_id="  sha256:abc  ",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()

    assert payload["policy_id"] == "act"
    assert payload["checkpoint_id"] == "sha256:abc"
    assert {
        (trial["policy_id"], trial["checkpoint_id"])
        for trial in payload["reactivity_trials"] + payload["robustness_trials"]
    } == {("act", "sha256:abc")}


def test_report_rejects_custom_grid_spoofing_frozen_protocol_id() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        seeds=(7,),
        reactivity_k_exec=(2,),
        selected_k_exec=3,
    ).to_dict()
    protocol["protocol_id"] = M4_PROTOCOL_ID
    protocol["protocol_mode"] = "frozen"
    records = [
        {**trial, "success": True}
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    report = build_report(records, protocol=protocol)

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"]["valid"] is False
    assert completion["protocol_validation"]["formal"] is False
    assert any(
        "frozen_protocol_field_mismatch:seeds" in error
        for error in completion["protocol_validation"]["errors"]
    )


def test_report_keeps_intact_custom_m4_plan_non_formal_but_auditable() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        seeds=(7,),
        reactivity_k_exec=(2,),
        selected_k_exec=3,
    ).to_dict()
    records = [
        {**trial, "success": True}
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    completion = build_report(records, protocol=protocol)["completion"]

    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"] == {
        "valid": True,
        "formal": False,
        "mode": "custom_non_formal",
        "errors": [],
    }
    assert "non_formal_protocol" in completion["formal_readiness_blockers"]
    assert "reset_admission_unverified" in completion["formal_readiness_blockers"]


def test_report_rejects_tampered_custom_m4_trial_list() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        seeds=(7,),
        reactivity_k_exec=(2,),
        selected_k_exec=3,
    ).to_dict()
    protocol["reactivity_trials"] = []
    protocol["planned_trials"] = len(protocol["robustness_trials"])
    records = [{**trial, "success": True} for trial in protocol["robustness_trials"]]

    completion = build_report(records, protocol=protocol)["completion"]

    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"]["valid"] is False
    assert (
        "custom_protocol_field_mismatch:reactivity_trials"
        in completion["protocol_validation"]["errors"]
    )


def test_report_rejects_frozen_plan_with_missing_r1_grid() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    protocol["reactivity_trials"] = []
    protocol["planned_trials"] = len(protocol["robustness_trials"])
    records = [{**trial, "success": True} for trial in protocol["robustness_trials"]]

    report = build_report(records, protocol=protocol)

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"]["valid"] is False
    assert "r1_plan_missing" in completion["formal_readiness_blockers"]


def test_report_rejects_frozen_protocol_id_without_trial_lists() -> None:
    report = build_report(
        [{"success": True}],
        protocol={
            "protocol_id": M4_PROTOCOL_ID,
            "protocol_mode": "frozen",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
        },
    )

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"]["formal"] is False
    assert (
        "frozen_protocol_field_mismatch:reactivity_trials"
        in completion["protocol_validation"]["errors"]
    )
    assert "r1_plan_missing" in completion["formal_readiness_blockers"]


def test_report_keeps_exact_frozen_protocol_incomplete_without_reset_provenance() -> (
    None
):
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    records = [
        {
            **trial,
            "success": True,
            "actions": [[0.0] * M4_ACTION_DIM, [0.0] * M4_ACTION_DIM],
            "inference_latency_ms": [1.0],
            "control_latency_ms": [2.0],
        }
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    report = build_report(records, protocol=protocol)

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["protocol_validation"] == {
        "valid": True,
        "formal": True,
        "mode": "frozen",
        "errors": [],
    }
    assert "reset_admission_unverified" in completion["formal_readiness_blockers"]
    assert completion["reset_admission_evidence"]["missing_record_count"] == 450
    assert report["qualified_episode_count"] == 0
    assert report["metrics"]["success"]["trials"] == 0


def test_self_reported_reset_receipts_cannot_complete_exact_frozen_protocol() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    trials = protocol["reactivity_trials"] + protocol["robustness_trials"]
    records = [
        {
            **trial,
            "success": True,
            "actions": [[0.0] * M4_ACTION_DIM, [0.0] * M4_ACTION_DIM],
            "inference_latency_ms": [1.0],
            "control_latency_ms": [2.0],
            "reset_receipt": _valid_reset_receipt(trial),
        }
        for trial in trials
    ]

    report = build_report(records, protocol=protocol)

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["protocol_validation"]["formal"] is True
    assert completion["reset_admission_evidence"] == {
        "required": True,
        "complete": False,
        "content_recomputed": True,
        "runner_provenance_verified": False,
        "observed_record_count": 450,
        "valid_record_count": 450,
        "missing_record_count": 0,
        "invalid_record_count": 0,
        "missing_trial_ids": [],
        "invalid_records": [],
    }
    assert completion["complete"] is False
    assert completion["formal_ready"] is False
    assert "reset_admission_unverified" in completion["formal_readiness_blockers"]


def test_exact_frozen_failures_without_classification_fail_closed() -> None:
    protocol = build_m4_plan(
        policy_id="act",
        checkpoint_id="sha256:abc",
        selected_k_exec=M4_SELECTED_K_EXEC,
    ).to_dict()
    records = [
        {
            **trial,
            "success": False,
            "actions": [[0.0] * M4_ACTION_DIM, [0.0] * M4_ACTION_DIM],
            "inference_latency_ms": [1.0],
            "control_latency_ms": [2.0],
        }
        for trial in protocol["reactivity_trials"] + protocol["robustness_trials"]
    ]

    report = build_report(records, protocol=protocol)

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["protocol_validation"]["formal"] is True
    assert completion["outcome_evidence"]["complete"] is False
    assert completion["outcome_evidence"]["invalid_record_count"] == 450
    assert len(completion["outcome_evidence"]["invalid_trial_ids"]) == 450
    assert completion["outcome_evidence"]["invalid_trial_ids"][0] == ("R1-k001-s10000")
    assert completion["outcome_evidence"]["invalid_trial_ids"][-1] == ("L4-k008-s10049")
    assert completion["complete"] is False
    assert completion["formal_readiness_blockers"] == [
        "outcome_evidence_invalid",
        "record_schema_invalid",
        "seed_disjointness_unverified",
        "reset_admission_unverified",
        "r1_perturbation_unverified",
    ]
    assert report["qualified_episode_count"] == 0
    assert report["metrics"]["success"]["trials"] == 0
    assert report["metrics"]["failures"]["total_failures"] == 0


def _single_trial_outcome_report(
    *,
    success: bool,
    failure_metadata: dict[str, object],
) -> dict[str, object]:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    return build_report(
        [{**trial, "success": success, **failure_metadata}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
        },
    )


@pytest.mark.parametrize(
    ("failure_metadata", "expected_error"),
    [
        ({"failure_type": "timeout"}, "failed_episode_missing_failure_stage"),
        ({"failure_stage": "lift"}, "failed_episode_missing_failure_type"),
        (
            {"failure_stage": "lift", "failure_type": "unclassified"},
            "failed_episode_unclassified_failure_type",
        ),
        (
            {"failure_stage": "lift", "failure_type": "new_failure"},
            "failed_episode_unknown_failure_type",
        ),
        (
            {"failure_stage": 3, "failure_type": "timeout"},
            "failure_stage_must_be_string",
        ),
        (
            {"failure_stage": " lift ", "failure_type": "timeout"},
            "failure_stage_must_be_canonical",
        ),
        (
            {"failure_stage": "Unknown", "failure_type": "timeout"},
            "failure_stage_must_be_canonical",
        ),
        (
            {"failure_stage": "UNKNOWN", "failure_type": "timeout"},
            "failure_stage_must_be_canonical",
        ),
        (
            {"failure_stage": "Lift", "failure_type": "timeout"},
            "failure_stage_must_be_canonical",
        ),
        (
            {"failure_stage": "lift", "failure_type": " timeout "},
            "failure_type_must_be_canonical",
        ),
        ({"failure": "timeout"}, "failure_payload_must_be_object"),
    ],
)
def test_report_rejects_invalid_failed_outcome_metadata(
    failure_metadata: dict[str, object],
    expected_error: str,
) -> None:
    report = _single_trial_outcome_report(
        success=False,
        failure_metadata=failure_metadata,
    )

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert completion["outcome_evidence"]["invalid_trial_ids"] == ["L0-k008-s10000"]
    assert (
        expected_error in completion["outcome_evidence"]["invalid_records"][0]["errors"]
    )
    assert "outcome_evidence_invalid" in completion["formal_readiness_blockers"]
    assert report["metrics"]["success"]["trials"] == 0
    assert report["metrics"]["failures"]["total_failures"] == 0


@pytest.mark.parametrize(
    ("failure_metadata", "expected_error"),
    [
        (
            {"failure_stage": "lift", "failure_type": "object_slip"},
            "successful_episode_has_failure_metadata",
        ),
        (
            {"failure": {"stage": "lift", "type": "object_slip"}},
            "successful_episode_has_failure_metadata",
        ),
        (
            {"failure_stage": "   ", "failure_type": None},
            "failure_stage_must_be_canonical",
        ),
    ],
)
def test_report_rejects_success_with_failure_metadata(
    failure_metadata: dict[str, object],
    expected_error: str,
) -> None:
    report = _single_trial_outcome_report(
        success=True,
        failure_metadata=failure_metadata,
    )

    completion = report["completion"]
    assert completion["trial_coverage_complete"] is True
    assert completion["outcome_evidence"]["complete"] is False
    assert completion["complete"] is False
    assert (
        expected_error in completion["outcome_evidence"]["invalid_records"][0]["errors"]
    )
    assert report["metrics"]["success"]["successes"] == 0
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("failure_metadata", "expected_error"),
    [
        (
            {
                "failure_stage": "lift",
                "failure_type": "object_slip",
                "failure": {"stage": "lift", "type": "object_slip"},
            },
            "duplicate_failure_stage_sources",
        ),
        (
            {
                "failure_stage": "lift",
                "failure_type": "object_slip",
                "failure": {"stage": "transport", "type": "object_slip"},
            },
            "conflicting_failure_stage_sources",
        ),
        (
            {
                "failure_stage": "lift",
                "failure": {"type": "object_slip"},
            },
            "failure_stage_type_source_mismatch",
        ),
    ],
)
def test_report_rejects_duplicate_or_mixed_failure_sources(
    failure_metadata: dict[str, object],
    expected_error: str,
) -> None:
    report = _single_trial_outcome_report(
        success=False,
        failure_metadata=failure_metadata,
    )

    completion = report["completion"]
    assert completion["complete"] is False
    assert (
        expected_error in completion["outcome_evidence"]["invalid_records"][0]["errors"]
    )


@pytest.mark.parametrize(
    "failure_metadata",
    [
        {"failure_stage": "lift", "failure_type": "object_slip"},
        {"failure": {"stage": "lift", "type": "object_slip"}},
    ],
)
def test_report_accepts_one_complete_canonical_failure_source(
    failure_metadata: dict[str, object],
) -> None:
    report = _single_trial_outcome_report(
        success=False,
        failure_metadata=failure_metadata,
    )

    completion = report["completion"]
    assert completion["outcome_evidence"] == {
        "complete": True,
        "invalid_record_count": 0,
        "invalid_trial_ids": [],
        "invalid_records": [],
    }
    assert completion["complete"] is True
    assert report["metrics"]["failures"]["stage_by_type"] == {
        "lift": {"object_slip": 1}
    }


def test_report_treats_null_or_empty_success_failure_fields_as_absent() -> None:
    report = _single_trial_outcome_report(
        success=True,
        failure_metadata={"failure_stage": "", "failure_type": None},
    )

    assert report["completion"]["outcome_evidence"]["complete"] is True
    assert report["completion"]["complete"] is True


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
@pytest.mark.parametrize("top_level_value", [None, "   "])
def test_trial_plan_requires_non_empty_top_level_frozen_identity(
    field: str,
    top_level_value: str | None,
) -> None:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    protocol = {
        "protocol_id": "unit-test",
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        "planned_trials": 1,
        "reactivity_trials": [],
        "robustness_trials": [trial],
    }
    protocol[field] = top_level_value

    report = build_report([{**trial, "success": True}], protocol=protocol)

    completion = report["completion"]
    assert completion["complete"] is False
    assert completion["protocol_validation"]["valid"] is False
    assert any(
        f"protocol.{field}" in error
        for error in completion["protocol_validation"]["errors"]
    )


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
def test_trial_plan_requires_present_top_level_frozen_identity(field: str) -> None:
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    protocol = {
        "protocol_id": "unit-test",
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        "planned_trials": 1,
        "reactivity_trials": [],
        "robustness_trials": [trial],
    }
    del protocol[field]

    report = build_report([{**trial, "success": True}], protocol=protocol)

    assert report["completion"]["complete"] is False
    assert any(
        f"protocol.{field}" in error
        for error in report["completion"]["protocol_validation"]["errors"]
    )


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
@pytest.mark.parametrize("trial_value", [None, "   ", "conflicting-id"])
def test_trial_identity_cannot_override_top_level_freeze(
    field: str,
    trial_value: str | None,
) -> None:
    top_level = {"policy_id": "act", "checkpoint_id": "sha256:abc"}
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        **top_level,
        field: trial_value,
    }
    observed = {
        **trial,
        field: "observed-conflict",
        "success": True,
    }

    report = build_report(
        [observed],
        protocol={
            "protocol_id": "unit-test",
            **top_level,
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
        },
    )

    completion = report["completion"]
    assert completion["complete"] is False
    assert completion["protocol_validation"]["valid"] is False
    assert any(
        f"robustness_trials[0].{field}" in error
        for error in completion["protocol_validation"]["errors"]
    )


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
def test_planned_trial_requires_propagated_top_level_identity(field: str) -> None:
    top_level = {"policy_id": "act", "checkpoint_id": "sha256:abc"}
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        **top_level,
    }
    del planned_trial[field]
    observed_trial = {**planned_trial, field: top_level[field], "success": True}

    completion = build_report(
        [observed_trial],
        protocol={
            "protocol_id": "unit-test",
            **top_level,
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
        },
    )["completion"]

    assert completion["trial_coverage_complete"] is True
    assert completion["complete"] is False
    assert any(
        f"robustness_trials[0].{field}" in error
        for error in completion["protocol_validation"]["errors"]
    )


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
def test_trial_identity_cannot_be_omitted_at_both_levels(field: str) -> None:
    top_level = {"policy_id": "act", "checkpoint_id": "sha256:abc"}
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        **top_level,
    }
    del top_level[field]
    del trial[field]
    observed = {**trial, "success": True}

    report = build_report(
        [observed],
        protocol={
            "protocol_id": "unit-test",
            **top_level,
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [trial],
        },
    )

    assert report["completion"]["complete"] is False
    assert report["completion"]["protocol_validation"]["valid"] is False


@pytest.mark.parametrize("field", ["policy_id", "checkpoint_id"])
@pytest.mark.parametrize(
    "observed_value",
    [None, "   ", "  padded-but-nonempty  ", "conflicting-id"],
)
def test_observed_trial_requires_exact_top_level_frozen_identity(
    field: str,
    observed_value: str | None,
) -> None:
    top_level = {"policy_id": "act", "checkpoint_id": "sha256:abc"}
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        **top_level,
    }
    observed_trial = {
        **planned_trial,
        **top_level,
        field: observed_value,
        "success": True,
    }

    report = build_report(
        [observed_trial],
        protocol={
            "protocol_id": "unit-test",
            **top_level,
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
        },
    )

    completion = report["completion"]
    assert completion["complete"] is False
    assert completion["missing"][0][field] == top_level[field]
    assert completion["unexpected"]


def test_seed_disjointness_must_be_checked_against_the_collection_manifest() -> None:
    assert_disjoint_seeds(M4_EVAL_SEEDS, [0, 1, 2])

    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_seeds(M4_EVAL_SEEDS, [M4_EVAL_SEEDS[0]])


def test_report_marks_exact_trial_identity_set_complete() -> None:
    reactivity_trial = {
        "trial_id": "R1-k001-s10000",
        "seed": 10_000,
        "condition_id": "R1",
        "k_exec": 1,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    robustness_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [
            {**reactivity_trial, "success": True},
            {
                **robustness_trial,
                "success": False,
                "failure_stage": "lift",
                "failure_type": "object_slip",
            },
        ],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 2,
            "reactivity_trials": [reactivity_trial],
            "robustness_trials": [robustness_trial],
        },
    )

    completion = report["completion"]
    assert completion["complete"] is True
    assert completion["missing"] == []
    assert completion["unexpected"] == []
    assert completion["duplicate"] == []
    assert completion["seed_disjointness"] == {
        "status": "unchecked",
        "collection_manifest_id": None,
        "reason": "collection_manifest_not_verified",
    }
    assert completion["formal_ready"] is False
    assert completion["formal_readiness_blockers"] == [
        "non_formal_protocol",
        "seed_disjointness_unverified",
        "r1_perturbation_unverified",
        "required_actions_missing",
        "required_inference_latency_missing",
        "required_control_latency_missing",
    ]


@pytest.mark.parametrize(
    ("mismatched_field", "mismatched_value"),
    [
        ("trial_id", "R1-k004-s10001"),
        ("seed", 10_001),
        ("condition_id", "L0"),
        ("k_exec", 4),
        ("policy_id", "bc"),
        ("checkpoint_id", "sha256:def"),
    ],
)
def test_report_rejects_equal_count_trial_identity_mismatches(
    mismatched_field: str,
    mismatched_value: str | int,
) -> None:
    planned_trial = {
        "trial_id": "R1-k001-s10000",
        "seed": 10_000,
        "condition_id": "R1",
        "k_exec": 1,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    observed_trial = {**planned_trial, mismatched_field: mismatched_value}

    report = build_report(
        [{**observed_trial, "success": True}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [planned_trial],
            "robustness_trials": [],
        },
    )

    completion = report["completion"]
    assert completion["observed"] == completion["planned"] == 1
    assert completion["complete"] is False
    assert completion["matching"] == "trial_identity"
    assert completion["missing"][0]["trial_id"] == "R1-k001-s10000"
    assert completion["unexpected"][0][mismatched_field] == mismatched_value
    assert completion["duplicate"] == []
    assert report["metrics"]["success"]["trials"] == 0


def test_report_rejects_equal_count_duplicate_trials() -> None:
    first_trial = {
        "trial_id": "R1-k001-s10000",
        "seed": 10_000,
        "condition_id": "R1",
        "k_exec": 1,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    second_trial = {
        "trial_id": "R1-k001-s10001",
        "seed": 10_001,
        "condition_id": "R1",
        "k_exec": 1,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [
            {**first_trial, "success": True},
            {
                **first_trial,
                "success": False,
                "failure_stage": "lift",
                "failure_type": "object_slip",
            },
        ],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 2,
            "reactivity_trials": [first_trial, second_trial],
            "robustness_trials": [],
        },
    )

    completion = report["completion"]
    assert completion["observed"] == completion["planned"] == 2
    assert completion["complete"] is False
    assert [trial["trial_id"] for trial in completion["missing"]] == ["R1-k001-s10001"]
    assert completion["unexpected"] == []
    assert completion["duplicate"] == [
        {
            "trial_id": "R1-k001-s10000",
            "seed": 10_000,
            "condition_id": "R1",
            "k_exec": 1,
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "observed_count": 2,
        }
    ]
    assert report["metrics"]["success"]["trials"] == 0


@pytest.mark.parametrize(
    ("mismatched_field", "mismatched_value"),
    [("policy_id", "bc"), ("checkpoint_id", "sha256:def")],
)
def test_report_uses_protocol_level_optional_identity_when_trial_omits_it(
    mismatched_field: str,
    mismatched_value: str,
) -> None:
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
    }
    observed_trial = {
        **planned_trial,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        mismatched_field: mismatched_value,
        "success": True,
    }

    report = build_report(
        [observed_trial],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
        },
    )

    completion = report["completion"]
    assert completion["complete"] is False
    assert completion["missing"][0][mismatched_field] != mismatched_value
    assert completion["unexpected"][0][mismatched_field] == mismatched_value


@pytest.mark.parametrize("missing_field", ["policy_id", "checkpoint_id"])
def test_report_rejects_observation_missing_planned_optional_identity(
    missing_field: str,
) -> None:
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    observed_trial = {
        key: value for key, value in planned_trial.items() if key != missing_field
    }

    report = build_report(
        [{**observed_trial, "success": True}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
        },
    )

    completion = report["completion"]
    assert completion["complete"] is False
    assert completion["missing"][0][missing_field] == planned_trial[missing_field]
    assert missing_field not in completion["unexpected"][0]


def test_report_requires_manifest_evidence_for_formal_readiness() -> None:
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [{**planned_trial, "success": True}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
            "seed_disjointness": {"status": "checked"},
        },
    )

    assert report["completion"]["complete"] is True
    assert report["completion"]["seed_disjointness"] == {
        "status": "unchecked",
        "collection_manifest_id": None,
        "declared_status": "checked",
        "reason": "missing_collection_manifest_id",
    }
    assert report["completion"]["formal_ready"] is False


def test_report_does_not_trust_self_declared_manifest_evidence() -> None:
    planned_trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }

    report = build_report(
        [{**planned_trial, "success": True}],
        protocol={
            "protocol_id": "unit-test",
            "policy_id": "act",
            "checkpoint_id": "sha256:abc",
            "planned_trials": 1,
            "reactivity_trials": [],
            "robustness_trials": [planned_trial],
            "seed_disjointness": {
                "status": "checked",
                "collection_manifest_id": "sha256:manifest-fixture",
            },
        },
    )

    assert report["completion"]["complete"] is True
    assert report["completion"]["seed_disjointness"] == {
        "status": "unchecked",
        "collection_manifest_id": "sha256:manifest-fixture",
        "declared_status": "checked",
        "reason": "collection_manifest_not_verified",
    }
    assert report["completion"]["formal_ready"] is False
    assert (
        "seed_disjointness_unverified"
        in report["completion"]["formal_readiness_blockers"]
    )


def test_report_cli_writes_incomplete_report_and_fails_closed_by_default(
    tmp_path: Path,
) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    protocol_path = tmp_path / "protocol.json"
    output_path = tmp_path / "report.json"
    episodes_path.write_text(
        json.dumps({"seed": 1, "success": True}) + "\n",
        encoding="utf-8",
    )
    protocol_path.write_text(
        json.dumps({"protocol_id": "unit-test", "planned_trials": 2}),
        encoding="utf-8",
    )

    exit_code = report_main(
        [
            "--episodes",
            str(episodes_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["observed_episode_count"] == 1
    assert report["protocol"]["planned_trials"] == 2
    completion = report["completion"]
    assert completion["observed"] == 1
    assert completion["planned"] == 2
    assert completion["complete"] is False
    assert completion["seed_disjointness"] == {
        "status": "unchecked",
        "collection_manifest_id": None,
        "reason": "collection_manifest_not_verified",
    }
    assert completion["formal_ready"] is False
    assert "trial_coverage_incomplete" in completion["formal_readiness_blockers"]


def test_report_cli_allows_explicit_incomplete_override(tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    protocol_path = tmp_path / "protocol.json"
    output_path = tmp_path / "report.json"
    episodes_path.write_text(
        json.dumps({"seed": 1, "success": True}) + "\n",
        encoding="utf-8",
    )
    protocol_path.write_text(
        json.dumps({"protocol_id": "unit-test", "planned_trials": 2}),
        encoding="utf-8",
    )

    exit_code = report_main(
        [
            "--episodes",
            str(episodes_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output_path),
            "--allow-incomplete",
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["completion"]["complete"] is False


@pytest.mark.parametrize(
    "seed_disjointness",
    [
        None,
        {
            "status": "checked",
            "collection_manifest_id": "sha256:manifest-fixture",
        },
    ],
)
def test_report_cli_does_not_trust_self_declared_seed_disjointness(
    tmp_path: Path,
    seed_disjointness: dict[str, str] | None,
) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    protocol_path = tmp_path / "protocol.json"
    output_path = tmp_path / "report.json"
    trial = {
        "trial_id": "L0-k008-s10000",
        "seed": 10_000,
        "condition_id": "L0",
        "k_exec": 8,
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
    }
    episodes_path.write_text(
        json.dumps({**trial, "success": True}) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "protocol_id": "unit-test",
        "policy_id": "act",
        "checkpoint_id": "sha256:abc",
        "planned_trials": 1,
        "reactivity_trials": [],
        "robustness_trials": [trial],
    }
    if seed_disjointness is not None:
        protocol["seed_disjointness"] = seed_disjointness
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    exit_code = report_main(
        [
            "--episodes",
            str(episodes_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["completion"]["complete"] is True
    assert report["completion"]["formal_ready"] is False
    assert (
        "seed_disjointness_unverified"
        in report["completion"]["formal_readiness_blockers"]
    )
