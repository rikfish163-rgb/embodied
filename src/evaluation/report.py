"""Adapt episode records and write model-independent evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    CANONICAL_FAILURE_TYPES,
    action_smoothness,
    failure_taxonomy,
    latency_percentiles,
    wilson_interval,
)
from .protocol import M4_PROTOCOL_ID, M4_SELECTED_K_EXEC, build_m4_plan

GROUP_FIELDS = ("policy_id", "checkpoint_id", "condition_id", "k_exec")
INCOMPLETE_EXIT_CODE = 2
TRIAL_LIST_FIELDS = ("reactivity_trials", "robustness_trials")
REQUIRED_TRIAL_IDENTITY_FIELDS = ("trial_id", "seed", "condition_id", "k_exec")
OPTIONAL_TRIAL_IDENTITY_FIELDS = ("policy_id", "checkpoint_id")
CUSTOM_M4_PROTOCOL_PREFIX = "custom-m4-reactivity-robustness-"
FORMAL_FAILURE_TYPES = frozenset(CANONICAL_FAILURE_TYPES) - {"unclassified"}
_FAILURE_LABEL_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def build_report(
    episode_records: Sequence[Mapping[str, Any] | Any],
    *,
    protocol: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Build a JSON-serializable report from expert, BC, or ACT records.

    Required per-record field: ``success``. All other fields are optional for
    count-only summaries so the current expert ``EpisodeResult`` can be
    summarized. Protocols with trial lists reconcile the available identity
    fields and report records with missing identities as unexpected. Missing
    actions or latency arrays produce explicit zero counts / ``null`` metrics.
    """

    protocol_payload, protocol_input_errors = _protocol_dict(protocol)
    action_dimension, action_scale, action_contract_errors = _protocol_action_contract(
        protocol_payload
    )
    protocol_errors = protocol_input_errors + action_contract_errors
    records = [_adapt_episode(record) for record in episode_records]
    if not records:
        raise ValueError("cannot build an evaluation report without episode records")

    completion, qualified_indices = _completion(
        records,
        protocol_payload,
        extra_protocol_errors=protocol_errors,
        protocol_action_dimension=action_dimension,
        protocol_action_scale=action_scale,
    )
    qualified_records = [records[index] for index in qualified_indices]
    metrics = _summarize(
        qualified_records,
        protocol_action_dimension=action_dimension,
        protocol_action_scale=action_scale,
    )
    groups = []
    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in qualified_records:
        key = tuple(record[field] for field in GROUP_FIELDS)
        grouped[key].append(record)
    for key in sorted(grouped, key=_sortable_group_key):
        values = dict(zip(GROUP_FIELDS, key, strict=True))
        group_records = grouped[key]
        groups.append(
            {
                "group": values,
                "episode_count": len(group_records),
                "metrics": _summarize(
                    group_records,
                    protocol_action_dimension=action_dimension,
                    protocol_action_scale=action_scale,
                ),
            }
        )

    report = {
        "schema_version": "evaluation-report.v1",
        "protocol": protocol_payload,
        "observed_episode_count": len(records),
        "qualified_episode_count": len(qualified_records),
        "completion": completion,
        "metrics": metrics,
        "groups": groups,
    }
    _require_strict_json(report)
    return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize observed episode JSONL without running a policy."
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return success after writing an incomplete or non-ready report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    records = _read_jsonl(args.episodes)
    protocol = _read_json_object(args.protocol)
    report = build_report(records, protocol=protocol)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["completion"]["formal_ready"] is True or args.allow_incomplete:
        return 0
    return INCOMPLETE_EXIT_CODE


def _summarize(
    records: list[dict[str, Any]],
    *,
    protocol_action_dimension: int | None,
    protocol_action_scale: list[float] | None,
) -> dict[str, Any]:
    successes = sum(record["success"] for record in records)
    trials = len(records)
    inference_samples = [
        sample for record in records for sample in record["inference_latency_ms"]
    ]
    control_samples = [
        sample for record in records for sample in record["control_latency_ms"]
    ]

    action_dimensions = {
        len(record["actions"][0]) for record in records if record["actions"] is not None
    }
    if len(action_dimensions) > 1:
        raise ValueError("mixed action dimensions are not comparable")
    observed_action_dimension = next(iter(action_dimensions), None)
    if (
        observed_action_dimension is not None
        and protocol_action_dimension is not None
        and observed_action_dimension != protocol_action_dimension
    ):
        raise ValueError("action dimension conflicts with protocol")

    provided_record_scales = {
        tuple(record["action_scale"])
        for record in records
        if record["action_scale"] is not None
    }
    if len(provided_record_scales) > 1:
        raise ValueError("mixed action scales are not comparable")
    provided_record_scale = (
        list(next(iter(provided_record_scales))) if provided_record_scales else None
    )
    if (
        provided_record_scale is not None
        and protocol_action_scale is not None
        and provided_record_scale != protocol_action_scale
    ):
        raise ValueError("episode action scale conflicts with protocol action scale")

    action_dimension = protocol_action_dimension or observed_action_dimension
    if protocol_action_scale is not None:
        scale = protocol_action_scale
    else:
        effective_record_scales = {
            tuple(record["action_scale"])
            if record["action_scale"] is not None
            else (1.0,) * len(record["actions"][0])
            for record in records
            if record["actions"] is not None
        }
        if len(effective_record_scales) > 1:
            raise ValueError("mixed action scales are not comparable")
        scale = (
            list(next(iter(effective_record_scales)))
            if effective_record_scales
            else provided_record_scale
        )
    if action_dimension is None and scale is not None:
        action_dimension = len(scale)
    if action_dimension is not None:
        if scale is None:
            scale = [1.0] * action_dimension
        if len(scale) != action_dimension:
            raise ValueError("action_scale must match action dimension")

    smoothness_results = []
    for record in records:
        actions = record["actions"]
        if actions is None or len(actions) == 0:
            continue
        result = action_smoothness(actions, action_scale=scale)
        if result["transitions"]:
            smoothness_results.append(result)
    transitions = sum(result["transitions"] for result in smoothness_results)
    smoothness_value = (
        sum(result["value"] * result["transitions"] for result in smoothness_results)
        / transitions
        if transitions
        else None
    )

    return {
        "success": {
            "successes": successes,
            "trials": trials,
            "rate": successes / trials if trials else None,
            "wilson": wilson_interval(successes, trials) if trials else None,
        },
        "failures": failure_taxonomy(records),
        "latency_ms": {
            "inference": latency_percentiles(inference_samples),
            "control": latency_percentiles(control_samples),
        },
        "action_smoothness": {
            "metric": "mean_scaled_l2_delta",
            "formula": "mean_t(norm_2((action[t]-action[t-1])/action_scale))",
            "aggregation": "transition_weighted_mean_across_episodes",
            "lower_is_smoother": True,
            "action_dimension": action_dimension,
            "action_scale": scale,
            "episodes": len(smoothness_results),
            "transitions": transitions,
            "value": smoothness_value,
        },
    }


def _adapt_episode(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    success_value = _get(record, "success")
    if isinstance(success_value, np.bool_):
        success = bool(success_value)
    elif isinstance(success_value, bool):
        success = success_value
    else:
        raise TypeError("episode success must be bool")

    actions = _get(record, "actions", None)
    if actions is None:
        actions = _get(record, "action", None)
    if actions is None:
        steps = _get(record, "steps", ())
        extracted = [_get(step, "action") for step in steps]
        actions = extracted or None

    latency = _get(record, "latency_ms", None)
    inference_latency = _get(record, "inference_latency_ms", None)
    control_latency = _get(record, "control_latency_ms", None)
    if isinstance(latency, Mapping):
        if inference_latency is None:
            inference_latency = latency.get("inference")
        if control_latency is None:
            control_latency = latency.get("control")

    failure_stage, failure_type, outcome_errors = _failure_metadata(
        success=success,
        direct_stage=_get(record, "failure_stage", None),
        direct_type=_get(record, "failure_type", None),
        failure_payload=_get(record, "failure", None),
    )
    action_rows, action_errors = _action_rows(actions)
    action_scale, action_scale_errors = _action_scale_values(
        _get(record, "action_scale", None)
    )
    inference_latency, inference_latency_errors = _latency_values(
        inference_latency, "inference_latency_ms"
    )
    control_latency, control_latency_errors = _latency_values(
        control_latency, "control_latency_ms"
    )
    trial_id, trial_id_errors = _record_optional_label(
        _get(record, "trial_id", None), None, "trial_id"
    )
    condition_id, condition_id_errors = _record_optional_label(
        _get(record, "condition_id", None), "unspecified", "condition_id"
    )

    return {
        "episode_id": _get(record, "episode_id", None),
        "trial_id": trial_id,
        "seed": _optional_non_negative_int(_get(record, "seed", None), "seed"),
        "policy_id": _record_identity_label(_get(record, "policy_id", None)),
        "checkpoint_id": _record_identity_label(
            _get(record, "checkpoint_id", None), "unspecified"
        ),
        "condition_id": condition_id,
        "k_exec": _optional_positive_int(_get(record, "k_exec", None), "k_exec"),
        "success": success,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "_outcome_evidence_errors": outcome_errors,
        "actions": action_rows,
        "action_scale": action_scale,
        "_record_schema_errors": (
            action_errors
            + action_scale_errors
            + inference_latency_errors
            + control_latency_errors
            + trial_id_errors
            + condition_id_errors
        ),
        "inference_latency_ms": inference_latency,
        "control_latency_ms": control_latency,
        "reset_receipt": _get(record, "reset_receipt", None),
    }


def _failure_metadata(
    *,
    success: bool,
    direct_stage: Any,
    direct_type: Any,
    failure_payload: Any,
) -> tuple[Any, Any, list[str]]:
    errors: list[str] = []
    nested_stage = None
    nested_type = None
    if failure_payload is not None:
        if isinstance(failure_payload, Mapping):
            nested_stage = failure_payload.get("stage")
            nested_type = failure_payload.get("type")
        else:
            _append_unique(errors, "failure_payload_must_be_object")

    stage, stage_source = _resolve_failure_field(
        field_name="failure_stage",
        direct_value=direct_stage,
        nested_value=nested_stage,
        errors=errors,
    )
    failure_type, type_source = _resolve_failure_field(
        field_name="failure_type",
        direct_value=direct_type,
        nested_value=nested_type,
        errors=errors,
    )
    if (
        stage_source is not None
        and type_source is not None
        and stage_source != type_source
    ):
        _append_unique(errors, "failure_stage_type_source_mismatch")

    for field_name, value in (
        ("failure_stage", direct_stage),
        ("failure_stage", nested_stage),
        ("failure_type", direct_type),
        ("failure_type", nested_type),
    ):
        if _has_failure_label(value):
            _validate_failure_label_shape(field_name, value, errors)

    if success:
        if any(
            _has_failure_label(value)
            for value in (direct_stage, nested_stage, direct_type, nested_type)
        ):
            _append_unique(errors, "successful_episode_has_failure_metadata")
        return (
            _json_safe_value(stage, "episode.failure_stage", []),
            _json_safe_value(failure_type, "episode.failure_type", []),
            errors,
        )

    if not _has_failure_label(stage):
        _append_unique(errors, "failed_episode_missing_failure_stage")
    elif isinstance(stage, str) and stage == "unknown":
        _append_unique(errors, "failed_episode_unknown_failure_stage")

    if not _has_failure_label(failure_type):
        _append_unique(errors, "failed_episode_missing_failure_type")
    elif isinstance(failure_type, str) and failure_type == "unclassified":
        _append_unique(errors, "failed_episode_unclassified_failure_type")
    elif (
        isinstance(failure_type, str)
        and failure_type == failure_type.strip()
        and failure_type not in FORMAL_FAILURE_TYPES
    ):
        _append_unique(errors, "failed_episode_unknown_failure_type")
    return (
        _json_safe_value(stage, "episode.failure_stage", []),
        _json_safe_value(failure_type, "episode.failure_type", []),
        errors,
    )


def _resolve_failure_field(
    *,
    field_name: str,
    direct_value: Any,
    nested_value: Any,
    errors: list[str],
) -> tuple[Any, str | None]:
    has_direct = _has_failure_label(direct_value)
    has_nested = _has_failure_label(nested_value)
    if has_direct and has_nested:
        qualifier = "duplicate" if direct_value == nested_value else "conflicting"
        _append_unique(errors, f"{qualifier}_{field_name}_sources")
    if has_direct:
        return direct_value, "direct"
    if has_nested:
        return nested_value, "nested"
    return None, None


def _has_failure_label(value: Any) -> bool:
    return value is not None and value != ""


def _validate_failure_label_shape(
    field_name: str,
    value: Any,
    errors: list[str],
) -> None:
    if not isinstance(value, str):
        _append_unique(errors, f"{field_name}_must_be_string")
    elif _FAILURE_LABEL_PATTERN.fullmatch(value) is None:
        _append_unique(errors, f"{field_name}_must_be_canonical")


def _append_unique(errors: list[str], error: str) -> None:
    if error not in errors:
        errors.append(error)


def _action_rows(value: Any) -> tuple[list[list[float]] | None, list[str]]:
    if value is None:
        return None, []
    try:
        object_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return None, ["actions_must_be_numeric_matrix"]
    if object_array.size == 0:
        return None, ["actions_must_be_non_empty_matrix"]
    if object_array.ndim != 2:
        return None, ["actions_must_have_time_by_dimension_shape"]
    if any(isinstance(item, (bool, np.bool_)) for item in object_array.flat):
        return None, ["actions_boolean_is_not_number"]
    if any(not _is_real_number(item) for item in object_array.flat):
        return None, ["actions_must_be_numeric_matrix"]
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return None, ["actions_must_be_finite"]
    return array.tolist(), []


def _latency_values(value: Any, name: str) -> tuple[list[float], list[str]]:
    if value is None:
        return [], []
    try:
        object_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return [], [f"{name}_must_be_numeric_vector"]
    if object_array.ndim != 1:
        return [], [f"{name}_must_be_one_dimensional"]
    if object_array.size == 0:
        return [], []
    if any(isinstance(item, (bool, np.bool_)) for item in object_array.flat):
        return [], [f"{name}_boolean_is_not_number"]
    if any(not _is_real_number(item) for item in object_array.flat):
        return [], [f"{name}_must_be_numeric_vector"]
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return [], [f"{name}_must_be_finite"]
    if np.any(array < 0.0):
        return [], [f"{name}_must_be_non_negative"]
    return array.tolist(), []


def _action_scale_values(value: Any) -> tuple[list[float] | None, list[str]]:
    if value is None:
        return None, []
    try:
        object_array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return None, ["action_scale_must_be_numeric_vector"]
    if object_array.ndim != 1 or object_array.size == 0:
        return None, ["action_scale_must_be_non_empty_vector"]
    if any(isinstance(item, (bool, np.bool_)) for item in object_array.flat):
        return None, ["action_scale_boolean_is_not_number"]
    if any(not _is_real_number(item) for item in object_array.flat):
        return None, ["action_scale_must_be_numeric_vector"]
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        return None, ["action_scale_must_contain_positive_finite_values"]
    return array.tolist(), []


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _protocol_action_contract(
    protocol: Mapping[str, Any],
) -> tuple[int | None, list[float] | None, list[str]]:
    errors: list[str] = []
    action_spec = protocol.get("action_spec")
    if action_spec is None:
        return None, None, errors
    if not isinstance(action_spec, Mapping):
        return None, None, ["protocol.action_spec:must_be_object"]

    dimension = action_spec.get("dimension")
    if isinstance(dimension, bool):
        errors.append("protocol.action_spec.dimension:boolean_is_not_integer")
        dimension = None
    elif not isinstance(dimension, int):
        errors.append("protocol.action_spec.dimension:must_be_integer")
        dimension = None
    elif dimension <= 0:
        errors.append("protocol.action_spec.dimension:must_be_positive")
        dimension = None

    scale_value = action_spec.get("scale")
    if not isinstance(scale_value, list):
        errors.append("protocol.action_spec.scale:must_be_array")
        return None, None, errors
    if dimension is not None and len(scale_value) != dimension:
        errors.append("protocol.action_spec.scale:length_must_match_dimension")

    scale = []
    for index, value in enumerate(scale_value):
        path = f"protocol.action_spec.scale[{index}]"
        if isinstance(value, bool):
            errors.append(f"{path}:boolean_is_not_number")
        elif not isinstance(value, (int, float)):
            errors.append(f"{path}:must_be_number")
        elif not math.isfinite(value):
            errors.append(f"{path}:non_finite_number")
        elif value <= 0:
            errors.append(f"{path}:must_be_positive")
        else:
            scale.append(float(value))
    if errors:
        return None, None, errors
    return dimension, scale, errors


def _record_identity_label(value: Any, default: str = "unspecified") -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return default
    return value


def _optional_label(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record labels must be non-empty strings")
    return value.strip()


def _record_optional_label(
    value: Any,
    default: str | None,
    field_name: str,
) -> tuple[str | None, list[str]]:
    if value is None:
        return default, []
    if not isinstance(value, str) or not value.strip():
        return default, [f"{field_name}_must_be_non_empty_string"]
    if value != value.strip():
        return value, [f"{field_name}_must_be_canonical"]
    return value, []


def _completion(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    extra_protocol_errors: Sequence[str],
    protocol_action_dimension: int | None,
    protocol_action_scale: list[float] | None,
) -> tuple[dict[str, Any], list[int]]:
    protocol_validation = _protocol_validation(protocol, extra_protocol_errors)
    outcome_evidence = _outcome_evidence(records)
    reset_admission_evidence, reset_receipt_errors = _reset_admission_evidence(
        records, protocol
    )
    for record_index, errors in reset_receipt_errors.items():
        for error in errors:
            _append_unique(records[record_index]["_record_schema_errors"], error)
    declared_planned = _declared_planned_count(protocol)
    has_trial_plan = any(field in protocol for field in TRIAL_LIST_FIELDS)
    if not has_trial_plan:
        candidate_indices = set(range(len(records)))
        record_schema_evidence, record_schema_errors = _record_schema_evidence(
            records,
            comparable_indices=candidate_indices,
            protocol_action_dimension=protocol_action_dimension,
            protocol_action_scale=protocol_action_scale,
        )
        qualified_indices, record_qualification = _record_qualification(
            records,
            candidate_indices=candidate_indices,
            unexpected_indices=set(),
            duplicate_indices=set(),
            protocol_valid=protocol_validation["valid"],
            record_schema_errors=record_schema_errors,
        )
        trial_coverage_complete = (
            len(records) == declared_planned if declared_planned is not None else None
        )
        complete = trial_coverage_complete
        if (
            protocol_validation["valid"] is not True
            or outcome_evidence["complete"] is not True
            or record_schema_evidence["complete"] is not True
            or (
                reset_admission_evidence["required"] is True
                and reset_admission_evidence["complete"] is not True
            )
        ):
            complete = False
        completion = _with_formal_readiness(
            {
                "observed": len(records),
                "planned": declared_planned,
                "trial_coverage_complete": trial_coverage_complete,
                "complete": complete,
                "protocol_validation": protocol_validation,
                "outcome_evidence": outcome_evidence,
                "reset_admission_evidence": reset_admission_evidence,
                "record_schema_evidence": record_schema_evidence,
                "record_qualification": record_qualification,
            },
            protocol,
            records,
        )
        return completion, qualified_indices

    planned_trials = _planned_trial_identities(protocol)
    observed_trials = [_observed_trial_identity(record) for record in records]
    missing, unexpected, matched_indices, unexpected_indices = _reconcile_trials(
        planned_trials, observed_trials
    )
    duplicate, duplicate_indices = _duplicate_observed_trials(observed_trials)
    candidate_indices = matched_indices - duplicate_indices
    record_schema_evidence, record_schema_errors = _record_schema_evidence(
        records,
        comparable_indices=candidate_indices,
        protocol_action_dimension=protocol_action_dimension,
        protocol_action_scale=protocol_action_scale,
    )
    qualified_indices, record_qualification = _record_qualification(
        records,
        candidate_indices=candidate_indices,
        unexpected_indices=unexpected_indices,
        duplicate_indices=duplicate_indices,
        protocol_valid=protocol_validation["valid"],
        record_schema_errors=record_schema_errors,
    )
    plan_count_matches = declared_planned is None or declared_planned == len(
        planned_trials
    )
    trial_coverage_complete = (
        plan_count_matches and not missing and not unexpected and not duplicate
    )

    completion: dict[str, Any] = {
        "observed": len(records),
        "planned": declared_planned
        if declared_planned is not None
        else len(planned_trials),
        "trial_coverage_complete": trial_coverage_complete,
        "complete": (
            trial_coverage_complete
            and protocol_validation["valid"]
            and outcome_evidence["complete"]
            and record_schema_evidence["complete"]
            and (
                reset_admission_evidence["required"] is not True
                or reset_admission_evidence["complete"] is True
            )
        ),
        "protocol_validation": protocol_validation,
        "outcome_evidence": outcome_evidence,
        "reset_admission_evidence": reset_admission_evidence,
        "record_schema_evidence": record_schema_evidence,
        "record_qualification": record_qualification,
        "matching": "trial_identity",
        "missing": missing,
        "unexpected": unexpected,
        "duplicate": duplicate,
    }
    if not plan_count_matches:
        completion["plan_count_mismatch"] = {
            "declared": declared_planned,
            "listed": len(planned_trials),
        }
    completion = _with_formal_readiness(completion, protocol, records)
    return completion, qualified_indices


def _with_formal_readiness(
    completion: dict[str, Any],
    protocol: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_disjointness = _seed_disjointness_status(protocol)
    blockers = []
    if completion["trial_coverage_complete"] is not True:
        blockers.append("trial_coverage_incomplete")
    if completion.get("matching") != "trial_identity":
        blockers.append("trial_identity_plan_missing")
    if completion["outcome_evidence"]["complete"] is not True:
        blockers.append("outcome_evidence_invalid")
    if completion["record_schema_evidence"]["complete"] is not True:
        blockers.append("record_schema_invalid")
    protocol_validation = completion["protocol_validation"]
    if protocol_validation["valid"] is not True:
        blockers.append("protocol_invalid")
    elif protocol_validation["formal"] is not True:
        blockers.append("non_formal_protocol")
    blockers.append("seed_disjointness_unverified")
    reset_evidence = completion["reset_admission_evidence"]
    if reset_evidence["required"] is True:
        if reset_evidence["invalid_record_count"]:
            blockers.append("reset_admission_invalid")
        elif reset_evidence["complete"] is not True:
            blockers.append("reset_admission_unverified")
    has_planned_r1 = _has_planned_r1_trial(protocol)
    if protocol.get("protocol_id") == M4_PROTOCOL_ID and not has_planned_r1:
        blockers.append("r1_plan_missing")
    elif has_planned_r1:
        blockers.append("r1_perturbation_unverified")
    if any(record["actions"] is None for record in records):
        blockers.append("required_actions_missing")
    if any(not record["inference_latency_ms"] for record in records):
        blockers.append("required_inference_latency_missing")
    if any(not record["control_latency_ms"] for record in records):
        blockers.append("required_control_latency_missing")

    completion["seed_disjointness"] = seed_disjointness
    completion["formal_ready"] = False
    completion["formal_readiness_blockers"] = blockers
    return completion


def _reset_admission_evidence(
    records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    protocol_id = protocol.get("protocol_id")
    required = protocol_id == M4_PROTOCOL_ID or (
        isinstance(protocol_id, str)
        and protocol_id.startswith(CUSTOM_M4_PROTOCOL_PREFIX)
    )
    if not required:
        return (
            {
                "required": False,
                "complete": None,
                "content_recomputed": None,
                "runner_provenance_verified": None,
                "observed_record_count": len(records),
                "valid_record_count": 0,
                "missing_record_count": 0,
                "invalid_record_count": 0,
                "missing_trial_ids": [],
                "invalid_records": [],
            },
            {},
        )

    contract = protocol.get("reset_admission")
    missing_trial_ids: list[str] = []
    invalid_records = []
    errors_by_index: dict[int, list[str]] = {}
    valid_count = 0
    missing_count = 0
    for record_index, record in enumerate(records):
        receipt = record["reset_receipt"]
        if receipt is None:
            missing_count += 1
            trial_id = record["trial_id"]
            if isinstance(trial_id, str) and trial_id not in missing_trial_ids:
                missing_trial_ids.append(trial_id)
            errors_by_index[record_index] = ["reset_receipt:missing"]
            continue
        errors = _validate_reset_receipt(receipt, record, contract)
        if errors:
            errors_by_index[record_index] = errors
            invalid_records.append(
                {
                    "record_index": record_index,
                    "trial_id": record["trial_id"],
                    "errors": errors,
                }
            )
        else:
            valid_count += 1

    invalid_count = len(invalid_records)
    content_recomputed = missing_count == 0 and invalid_count == 0
    return (
        {
            "required": True,
            "complete": False,
            "content_recomputed": content_recomputed,
            "runner_provenance_verified": False,
            "observed_record_count": len(records),
            "valid_record_count": valid_count,
            "missing_record_count": missing_count,
            "invalid_record_count": invalid_count,
            "missing_trial_ids": missing_trial_ids,
            "invalid_records": invalid_records,
        },
        errors_by_index,
    )


def _validate_reset_receipt(
    receipt: Any,
    record: Mapping[str, Any],
    contract: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, Mapping):
        return ["reset_receipt:protocol_contract_missing"]
    if not isinstance(receipt, Mapping):
        return ["reset_receipt:must_be_object"]

    receipt_contract = contract.get("receipt")
    if not isinstance(receipt_contract, Mapping):
        return ["reset_receipt:protocol_schema_missing"]
    _validate_exact_fields(
        receipt,
        receipt_contract.get("top_level_fields"),
        "reset_receipt",
        errors,
    )

    _require_exact_value(
        receipt.get("schema_version"),
        contract.get("receipt_schema_version"),
        "reset_receipt.schema_version",
        errors,
    )
    _require_exact_value(
        receipt.get("sampler_version"),
        contract.get("sampler_version"),
        "reset_receipt.sampler_version",
        errors,
    )
    candidate_hash = contract.get("candidate_hash")
    expected_hash_version = (
        candidate_hash.get("version") if isinstance(candidate_hash, Mapping) else None
    )
    _require_exact_value(
        receipt.get("candidate_hash_version"),
        expected_hash_version,
        "reset_receipt.candidate_hash_version",
        errors,
    )

    seed = receipt.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _append_unique(errors, "reset_receipt.seed:must_be_integer")
    elif seed != record["seed"]:
        _append_unique(errors, "reset_receipt.seed:conflicts_with_trial")
    _require_exact_value(
        receipt.get("condition_id"),
        record["condition_id"],
        "reset_receipt.condition_id",
        errors,
    )

    ranges = contract.get("ranges")
    if not isinstance(ranges, Mapping):
        _append_unique(errors, "reset_receipt:protocol_ranges_missing")
        ranges = {}
    is_l1 = record["condition_id"] == "L1"
    expected_proposal = ranges.get("l1_proposal_id" if is_l1 else "default_proposal_id")
    _require_exact_value(
        receipt.get("proposal_id"),
        expected_proposal,
        "reset_receipt.proposal_id",
        errors,
        mismatch="conflicts_with_condition",
    )
    expected_ranges = {
        "cube_x_m": ranges.get("l1_outer_cube_x_m" if is_l1 else "training_cube_x_m"),
        "cube_y_m": ranges.get("l1_outer_cube_y_m" if is_l1 else "training_cube_y_m"),
        "box_x_m": ranges.get("box_x_m"),
        "box_y_m": ranges.get("box_y_m"),
    }
    _require_exact_value(
        receipt.get("effective_ranges"),
        expected_ranges,
        "reset_receipt.effective_ranges",
        errors,
        mismatch="conflicts_with_condition",
    )

    rng_contract = contract.get("rng")
    expected_rng = {
        "api": rng_contract.get("api") if isinstance(rng_contract, Mapping) else None,
        "bit_generator": (
            rng_contract.get("bit_generator")
            if isinstance(rng_contract, Mapping)
            else None
        ),
    }
    _require_exact_value(
        receipt.get("rng"),
        expected_rng,
        "reset_receipt.rng",
        errors,
    )
    if receipt.get("collision_free") is not True:
        _append_unique(errors, "reset_receipt.collision_free:must_be_true")

    admission = contract.get("admission")
    if not isinstance(admission, Mapping):
        _append_unique(errors, "reset_receipt:protocol_admission_missing")
        admission = {}
    target = _validate_target_sampling(
        receipt.get("target_sampling"),
        expected_fields=receipt_contract.get("target_fields"),
        is_l1=is_l1,
        ranges=ranges,
        errors=errors,
    )
    placed_objects: list[tuple[tuple[float, float], float]] = []
    if target is not None:
        placed_objects.append(target)

    distractors = receipt.get("distractor_sampling")
    expected_distractors = 2 if record["condition_id"] == "L2" else 0
    if not isinstance(distractors, list):
        _append_unique(errors, "reset_receipt.distractor_sampling:must_be_array")
    else:
        if len(distractors) != expected_distractors:
            _append_unique(
                errors,
                "reset_receipt.distractor_sampling:count_conflicts_with_condition",
            )
        for index, sample in enumerate(distractors):
            accepted = _validate_distractor_sampling(
                sample,
                expected_fields=receipt_contract.get("distractor_fields"),
                candidate_fields=receipt_contract.get("candidate_ledger_fields"),
                path=f"reset_receipt.distractor_sampling[{index}]",
                budget=contract.get("max_attempts_per_distractor"),
                minimum_separation=admission.get("distractor_minimum_separation_m"),
                x_bounds=ranges.get("training_cube_x_m"),
                y_bounds=ranges.get("training_cube_y_m"),
                placed_objects=placed_objects,
                errors=errors,
                expected_index=index,
            )
            if accepted is not None:
                placed_objects.append(accepted)

    _validate_box_sampling(
        receipt.get("box_sampling"),
        expected_fields=receipt_contract.get("box_fields"),
        candidate_fields=receipt_contract.get("candidate_ledger_fields"),
        path="reset_receipt.box_sampling",
        budget=contract.get("max_box_attempts"),
        placed_objects=placed_objects,
        cube_half_extent=admission.get("cube_half_extent_m"),
        box_outer_half_extent=admission.get("box_outer_half_extent_m"),
        minimum_clearance=admission.get("box_minimum_clearance_m"),
        x_bounds=ranges.get("box_x_m"),
        y_bounds=ranges.get("box_y_m"),
        errors=errors,
    )
    return errors


def _validate_exact_fields(
    value: Mapping[Any, Any],
    expected_fields: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(expected_fields, list) or not all(
        isinstance(field, str) for field in expected_fields
    ):
        _append_unique(errors, f"{path}:protocol_fields_invalid")
        return
    missing = [field for field in expected_fields if field not in value]
    unexpected = [key for key in value if key not in expected_fields]
    if missing:
        _append_unique(errors, f"{path}:missing_fields:{','.join(missing)}")
    if unexpected:
        rendered = ",".join(str(key) for key in unexpected)
        _append_unique(errors, f"{path}:unexpected_fields:{rendered}")


def _require_exact_value(
    value: Any,
    expected: Any,
    path: str,
    errors: list[str],
    *,
    mismatch: str = "conflicts_with_protocol",
) -> None:
    if not _strict_json_equal(value, expected):
        _append_unique(errors, f"{path}:{mismatch}")


def _validate_target_sampling(
    sample: Any,
    *,
    expected_fields: Any,
    is_l1: bool,
    ranges: Mapping[str, Any],
    errors: list[str],
) -> tuple[tuple[float, float], float] | None:
    path = "reset_receipt.target_sampling"
    if not isinstance(sample, Mapping):
        _append_unique(errors, f"{path}:must_be_object")
        return None
    _validate_exact_fields(sample, expected_fields, path, errors)
    xy = _validated_xy(sample.get("accepted_xy"), f"{path}.accepted_xy", errors)
    yaw = _validated_finite_number(
        sample.get("accepted_yaw_rad"), f"{path}.accepted_yaw_rad", errors
    )
    partition = sample.get("partition")
    if xy is None or yaw is None:
        return None
    if not (-math.pi / 4 <= yaw < math.pi / 4):
        _append_unique(errors, f"{path}.accepted_yaw_rad:outside_protocol_range")

    if is_l1:
        expected_partition = _l1_partition(xy, ranges)
        if expected_partition is None:
            _append_unique(errors, f"{path}.accepted_xy:not_in_l1_outer_shell")
        elif partition != expected_partition:
            _append_unique(errors, f"{path}.partition:conflicts_with_accepted_xy")
    else:
        if partition != "training_rectangle":
            _append_unique(errors, f"{path}.partition:must_be_training_rectangle")
        if not _xy_in_ranges(
            xy,
            ranges.get("training_cube_x_m"),
            ranges.get("training_cube_y_m"),
        ):
            _append_unique(errors, f"{path}.accepted_xy:outside_training_rectangle")
    return xy, yaw


def _l1_partition(xy: tuple[float, float], ranges: Mapping[str, Any]) -> str | None:
    outer_x = _numeric_bounds(ranges.get("l1_outer_cube_x_m"))
    outer_y = _numeric_bounds(ranges.get("l1_outer_cube_y_m"))
    training_x = _numeric_bounds(ranges.get("training_cube_x_m"))
    training_y = _numeric_bounds(ranges.get("training_cube_y_m"))
    if None in (outer_x, outer_y, training_x, training_y):
        return None
    assert outer_x is not None
    assert outer_y is not None
    assert training_x is not None
    assert training_y is not None
    x, y = xy
    if not (_in_half_open(x, outer_x) and _in_half_open(y, outer_y)):
        return None
    if x < training_x[0]:
        return "left"
    if x >= training_x[1]:
        return "right"
    if y < training_y[0]:
        return "bottom"
    if y >= training_y[1]:
        return "top"
    return None


def _validate_distractor_sampling(
    sample: Any,
    *,
    expected_fields: Any,
    candidate_fields: Any,
    path: str,
    budget: Any,
    minimum_separation: Any,
    x_bounds: Any,
    y_bounds: Any,
    placed_objects: list[tuple[tuple[float, float], float]],
    errors: list[str],
    expected_index: int,
) -> tuple[tuple[float, float], float] | None:
    if not isinstance(sample, Mapping):
        _append_unique(errors, f"{path}:must_be_object")
        return None
    _validate_exact_fields(sample, expected_fields, path, errors)
    index_value = sample.get("distractor_index")
    if (
        isinstance(index_value, bool)
        or not isinstance(index_value, int)
        or index_value != expected_index
    ):
        _append_unique(errors, f"{path}.distractor_index:must_match_order")
    threshold = _validated_finite_number(
        minimum_separation, f"{path}:protocol_threshold", errors
    )

    def collision_check(xy: tuple[float, float]) -> tuple[bool, float | None]:
        if not placed_objects or threshold is None:
            return False, None
        clearance = min(math.dist(xy, placed[0]) for placed in placed_objects)
        return clearance > threshold, clearance

    ledger = _validate_candidate_ledger(
        sample,
        candidate_fields=candidate_fields,
        path=path,
        budget=budget,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        collision_check=collision_check,
        errors=errors,
    )
    accepted_xy = _validated_xy(
        sample.get("accepted_xy"), f"{path}.accepted_xy", errors
    )
    yaw = _validated_finite_number(
        sample.get("accepted_yaw_rad"), f"{path}.accepted_yaw_rad", errors
    )
    if yaw is not None and not (-math.pi <= yaw < math.pi):
        _append_unique(errors, f"{path}.accepted_yaw_rad:outside_protocol_range")
    if ledger and accepted_xy is not None and accepted_xy != tuple(ledger[-1]["xy"]):
        _append_unique(errors, f"{path}.accepted_xy:must_equal_final_candidate")
    if accepted_xy is not None:
        expected_collision, expected_clearance = collision_check(accepted_xy)
        if not expected_collision:
            _append_unique(errors, f"{path}.accepted_xy:not_collision_free")
        _compare_recomputed_number(
            sample.get("accepted_min_center_separation_m"),
            expected_clearance,
            f"{path}.accepted_min_center_separation_m",
            errors,
        )
    if accepted_xy is None or yaw is None:
        return None
    return accepted_xy, yaw


def _validate_box_sampling(
    sample: Any,
    *,
    expected_fields: Any,
    candidate_fields: Any,
    path: str,
    budget: Any,
    placed_objects: list[tuple[tuple[float, float], float]],
    cube_half_extent: Any,
    box_outer_half_extent: Any,
    minimum_clearance: Any,
    x_bounds: Any,
    y_bounds: Any,
    errors: list[str],
) -> None:
    if not isinstance(sample, Mapping):
        _append_unique(errors, f"{path}:must_be_object")
        return
    _validate_exact_fields(sample, expected_fields, path, errors)
    cube_half = _validated_finite_number(
        cube_half_extent, f"{path}:protocol_cube_half_extent", errors
    )
    box_half = _validated_finite_number(
        box_outer_half_extent, f"{path}:protocol_box_half_extent", errors
    )
    threshold = _validated_finite_number(
        minimum_clearance, f"{path}:protocol_clearance", errors
    )

    def collision_check(xy: tuple[float, float]) -> tuple[bool, float | None]:
        if (
            not placed_objects
            or cube_half is None
            or box_half is None
            or threshold is None
        ):
            return False, None
        clearance = _box_clearance(xy, placed_objects, cube_half, box_half)
        return clearance > threshold, clearance

    ledger = _validate_candidate_ledger(
        sample,
        candidate_fields=candidate_fields,
        path=path,
        budget=budget,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        collision_check=collision_check,
        errors=errors,
    )
    accepted_xy = _validated_xy(
        sample.get("accepted_xy"), f"{path}.accepted_xy", errors
    )
    if ledger and accepted_xy is not None and accepted_xy != tuple(ledger[-1]["xy"]):
        _append_unique(errors, f"{path}.accepted_xy:must_equal_final_candidate")
    if accepted_xy is not None:
        expected_collision, expected_clearance = collision_check(accepted_xy)
        if not expected_collision:
            _append_unique(errors, f"{path}.accepted_xy:not_collision_free")
        _compare_recomputed_number(
            sample.get("accepted_min_clearance_m"),
            expected_clearance,
            f"{path}.accepted_min_clearance_m",
            errors,
        )


def _validate_candidate_ledger(
    sample: Mapping[str, Any],
    *,
    candidate_fields: Any,
    path: str,
    budget: Any,
    x_bounds: Any,
    y_bounds: Any,
    collision_check: Any,
    errors: list[str],
) -> list[dict[str, Any]]:
    attempts = sample.get("attempts")
    rejections = sample.get("rejections")
    accepted_index = sample.get("accepted_candidate_index")
    valid_attempts = (
        not isinstance(attempts, bool) and isinstance(attempts, int) and attempts > 0
    )
    valid_rejections = (
        not isinstance(rejections, bool)
        and isinstance(rejections, int)
        and rejections >= 0
    )
    if not valid_attempts:
        _append_unique(errors, f"{path}.attempts:must_be_positive_integer")
    if not valid_rejections:
        _append_unique(errors, f"{path}.rejections:must_be_non_negative_integer")
    if valid_attempts and valid_rejections and attempts != rejections + 1:
        _append_unique(errors, f"{path}:attempts_must_equal_rejections_plus_one")
    if (
        isinstance(accepted_index, bool)
        or not isinstance(accepted_index, int)
        or not valid_rejections
        or accepted_index != rejections
    ):
        _append_unique(errors, f"{path}.accepted_candidate_index:must_equal_rejections")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or (valid_attempts and attempts > budget)
    ):
        _append_unique(errors, f"{path}.attempts:exceeds_protocol_budget")
    if sample.get("collision_free") is not True:
        _append_unique(errors, f"{path}.collision_free:must_be_true")

    ledger_value = sample.get("candidate_ledger")
    if not isinstance(ledger_value, list) or not ledger_value:
        _append_unique(errors, f"{path}.candidate_ledger:must_be_non_empty_array")
        return []
    if (
        isinstance(budget, int)
        and not isinstance(budget, bool)
        and len(ledger_value) > budget
    ):
        _append_unique(errors, f"{path}.candidate_ledger:exceeds_protocol_budget")
    if valid_attempts and len(ledger_value) != attempts:
        _append_unique(errors, f"{path}.candidate_ledger:length_must_equal_attempts")

    canonical: list[dict[str, Any]] = []
    for index, candidate in enumerate(ledger_value):
        candidate_path = f"{path}.candidate_ledger[{index}]"
        if not isinstance(candidate, Mapping):
            _append_unique(errors, f"{candidate_path}:must_be_object")
            continue
        _validate_exact_fields(candidate, candidate_fields, candidate_path, errors)
        candidate_index = candidate.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index != index
        ):
            _append_unique(errors, f"{candidate_path}.candidate_index:must_match_order")
        xy = _validated_xy(candidate.get("xy"), f"{candidate_path}.xy", errors)
        collision_free = candidate.get("collision_free")
        if not isinstance(collision_free, bool):
            _append_unique(errors, f"{candidate_path}.collision_free:must_be_bool")
        if xy is None or not isinstance(collision_free, bool):
            continue
        if not _xy_in_ranges(xy, x_bounds, y_bounds):
            _append_unique(errors, f"{candidate_path}.xy:outside_protocol_range")
        expected_collision, _ = collision_check(xy)
        if collision_free is not expected_collision:
            _append_unique(errors, f"{candidate_path}.collision_free:geometry_mismatch")
        if index < len(ledger_value) - 1 and collision_free:
            _append_unique(errors, f"{candidate_path}:accepted_candidate_not_final")
        if index == len(ledger_value) - 1 and not collision_free:
            _append_unique(errors, f"{candidate_path}:final_candidate_not_accepted")
        canonical.append(
            {"candidate_index": index, "collision_free": collision_free, "xy": list(xy)}
        )

    digest = sample.get("candidate_sequence_sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        _append_unique(errors, f"{path}.candidate_sequence_sha256:must_be_lower_hex_64")
    elif len(canonical) == len(ledger_value):
        encoded = json.dumps(
            canonical,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        expected_digest = hashlib.sha256(encoded).hexdigest()
        if digest != expected_digest:
            _append_unique(errors, f"{path}.candidate_sequence_sha256:content_mismatch")
    return canonical


def _box_clearance(
    box_xy: tuple[float, float],
    placed_objects: list[tuple[tuple[float, float], float]],
    cube_half_extent: float,
    box_outer_half_extent: float,
) -> float:
    box_low = (box_xy[0] - box_outer_half_extent, box_xy[1] - box_outer_half_extent)
    box_high = (box_xy[0] + box_outer_half_extent, box_xy[1] + box_outer_half_extent)
    clearances = []
    for object_xy, yaw in placed_objects:
        half_extent = cube_half_extent * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        object_low = (object_xy[0] - half_extent, object_xy[1] - half_extent)
        object_high = (object_xy[0] + half_extent, object_xy[1] + half_extent)
        clearances.append(
            max(
                box_low[0] - object_high[0],
                object_low[0] - box_high[0],
                box_low[1] - object_high[1],
                object_low[1] - box_high[1],
            )
        )
    return min(clearances)


def _validated_xy(
    value: Any, path: str, errors: list[str]
) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        _append_unique(errors, f"{path}:must_be_two_number_array")
        return None
    values = []
    for item in value:
        if not _is_real_number(item) or not math.isfinite(float(item)):
            _append_unique(errors, f"{path}:must_be_finite")
            return None
        values.append(float(item))
    return values[0], values[1]


def _validated_finite_number(value: Any, path: str, errors: list[str]) -> float | None:
    if not _is_real_number(value) or not math.isfinite(float(value)):
        _append_unique(errors, f"{path}:must_be_finite_number")
        return None
    return float(value)


def _numeric_bounds(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(
        not _is_real_number(item) or not math.isfinite(float(item)) for item in value
    ):
        return None
    low, high = float(value[0]), float(value[1])
    if not low < high:
        return None
    return low, high


def _xy_in_ranges(xy: tuple[float, float], x_bounds: Any, y_bounds: Any) -> bool:
    normalized_x = _numeric_bounds(x_bounds)
    normalized_y = _numeric_bounds(y_bounds)
    return bool(
        normalized_x is not None
        and normalized_y is not None
        and _in_half_open(xy[0], normalized_x)
        and _in_half_open(xy[1], normalized_y)
    )


def _in_half_open(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value < bounds[1]


def _compare_recomputed_number(
    observed: Any,
    expected: float | None,
    path: str,
    errors: list[str],
) -> None:
    if (
        expected is None
        or not _is_real_number(observed)
        or not math.isfinite(float(observed))
        or not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
    ):
        _append_unique(errors, f"{path}:recomputed_value_mismatch")


def _record_schema_evidence(
    records: list[dict[str, Any]],
    *,
    comparable_indices: set[int],
    protocol_action_dimension: int | None,
    protocol_action_scale: list[float] | None,
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    errors_by_index = {
        index: list(record["_record_schema_errors"])
        for index, record in enumerate(records)
    }

    for index, record in enumerate(records):
        actions = record["actions"]
        scale = record["action_scale"]
        if actions is not None:
            action_dimension = len(actions[0])
            if (
                protocol_action_dimension is not None
                and action_dimension != protocol_action_dimension
            ):
                _append_unique(
                    errors_by_index[index],
                    "action_dimension_conflicts_with_protocol",
                )
            if scale is not None and len(scale) != action_dimension:
                _append_unique(
                    errors_by_index[index],
                    "action_scale_must_match_action_dimension",
                )
        elif scale is not None:
            _append_unique(errors_by_index[index], "action_scale_without_actions")
        if (
            scale is not None
            and protocol_action_scale is not None
            and not _strict_json_equal(scale, protocol_action_scale)
        ):
            _append_unique(
                errors_by_index[index],
                "episode_action_scale_conflicts_with_protocol_action_scale",
            )

    if protocol_action_dimension is None and protocol_action_scale is None:
        comparable_action_indices = [
            index
            for index in sorted(comparable_indices)
            if records[index]["actions"] is not None
        ]
        dimensions = {
            len(records[index]["actions"][0]) for index in comparable_action_indices
        }
        if len(dimensions) > 1:
            for index in comparable_action_indices:
                _append_unique(errors_by_index[index], "mixed_action_dimensions")

        effective_scales = {
            tuple(records[index]["action_scale"])
            if records[index]["action_scale"] is not None
            else (1.0,) * len(records[index]["actions"][0])
            for index in comparable_action_indices
        }
        if len(effective_scales) > 1:
            for index in comparable_action_indices:
                _append_unique(errors_by_index[index], "mixed_action_scales")

    invalid_records = []
    invalid_trial_ids = []
    seen_trial_ids = set()
    for record_index, record in enumerate(records):
        errors = errors_by_index[record_index]
        if not errors:
            continue
        trial_id = record["trial_id"]
        if isinstance(trial_id, str) and trial_id not in seen_trial_ids:
            invalid_trial_ids.append(trial_id)
            seen_trial_ids.add(trial_id)
        invalid_records.append(
            {
                "record_index": record_index,
                "trial_id": trial_id,
                "errors": errors,
            }
        )
    evidence = {
        "complete": not invalid_records,
        "invalid_record_count": len(invalid_records),
        "invalid_trial_ids": invalid_trial_ids,
        "invalid_records": invalid_records,
    }
    return evidence, errors_by_index


def _record_qualification(
    records: list[dict[str, Any]],
    *,
    candidate_indices: set[int],
    unexpected_indices: set[int],
    duplicate_indices: set[int],
    protocol_valid: bool,
    record_schema_errors: Mapping[int, Sequence[str]],
) -> tuple[list[int], dict[str, Any]]:
    qualified_indices = []
    excluded_records = []
    for record_index, record in enumerate(records):
        reasons = []
        if not protocol_valid:
            reasons.append("protocol_invalid")
        if record_index in unexpected_indices:
            reasons.append("unexpected_trial_identity")
        if record_index in duplicate_indices:
            reasons.append("duplicate_trial_observation")
        if record_index not in candidate_indices and not (
            record_index in unexpected_indices or record_index in duplicate_indices
        ):
            reasons.append("trial_identity_not_qualified")
        if record["_outcome_evidence_errors"]:
            reasons.append("outcome_evidence_invalid")
        if record_schema_errors[record_index]:
            reasons.append("record_schema_invalid")
        if reasons:
            excluded_records.append(
                {
                    "record_index": record_index,
                    "trial_id": record["trial_id"],
                    "reasons": reasons,
                }
            )
        else:
            qualified_indices.append(record_index)

    return qualified_indices, {
        "qualified_record_count": len(qualified_indices),
        "qualified_record_indices": qualified_indices,
        "excluded_record_count": len(excluded_records),
        "excluded_records": excluded_records,
    }


def _outcome_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_records = []
    invalid_trial_ids = []
    seen_trial_ids = set()
    for record_index, record in enumerate(records):
        errors = record["_outcome_evidence_errors"]
        if not errors:
            continue
        trial_id = record["trial_id"]
        if isinstance(trial_id, str) and trial_id not in seen_trial_ids:
            invalid_trial_ids.append(trial_id)
            seen_trial_ids.add(trial_id)
        invalid_records.append(
            {
                "record_index": record_index,
                "trial_id": trial_id,
                "success": record["success"],
                "failure_stage": record["failure_stage"],
                "failure_type": record["failure_type"],
                "errors": list(errors),
            }
        )
    return {
        "complete": not invalid_records,
        "invalid_record_count": len(invalid_records),
        "invalid_trial_ids": invalid_trial_ids,
        "invalid_records": invalid_records,
    }


def _protocol_validation(
    protocol: Mapping[str, Any],
    extra_errors: Sequence[str] = (),
) -> dict[str, Any]:
    errors = list(dict.fromkeys(extra_errors))
    protocol_id = _canonical_protocol_label(
        protocol.get("protocol_id"), "protocol.protocol_id", errors
    )
    has_trial_plan = any(field in protocol for field in TRIAL_LIST_FIELDS)
    is_m4_protocol = protocol_id == M4_PROTOCOL_ID or (
        isinstance(protocol_id, str)
        and protocol_id.startswith(CUSTOM_M4_PROTOCOL_PREFIX)
    )

    top_level_identity: dict[str, str | None] = {}
    if has_trial_plan or is_m4_protocol:
        for field in OPTIONAL_TRIAL_IDENTITY_FIELDS:
            top_level_identity[field] = _canonical_protocol_label(
                protocol.get(field), f"protocol.{field}", errors
            )
        _validate_trial_identity_freeze(protocol, top_level_identity, errors)

    if protocol_id == M4_PROTOCOL_ID:
        mode = "frozen"
        if all(
            top_level_identity.get(field) is not None
            for field in OPTIONAL_TRIAL_IDENTITY_FIELDS
        ):
            expected = build_m4_plan(
                policy_id=top_level_identity["policy_id"],
                checkpoint_id=top_level_identity["checkpoint_id"],
                selected_k_exec=M4_SELECTED_K_EXEC,
            ).to_dict()
            _compare_m4_protocol(expected, protocol, "frozen", errors)
        elif protocol.get("protocol_mode") != "frozen":
            errors.append("frozen_protocol_field_mismatch:protocol_mode")
    elif isinstance(protocol_id, str) and protocol_id.startswith(
        CUSTOM_M4_PROTOCOL_PREFIX
    ):
        mode = "custom_non_formal"
        _validate_custom_m4_protocol(protocol, top_level_identity, errors)
    else:
        mode = "generic_non_formal"
        declared_mode = protocol.get("protocol_mode")
        if declared_mode not in {None, "generic_non_formal"}:
            errors.append("protocol.protocol_mode:inconsistent_with_protocol_id")

    valid = not errors
    return {
        "valid": valid,
        "formal": mode == "frozen" and valid,
        "mode": mode,
        "errors": errors,
    }


def _canonical_protocol_label(
    value: Any,
    path: str,
    errors: list[str],
) -> str | None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        errors.append(f"{path}:required_canonical_non_empty_string")
        return None
    return value


def _validate_trial_identity_freeze(
    protocol: Mapping[str, Any],
    top_level_identity: Mapping[str, str | None],
    errors: list[str],
) -> None:
    for list_field in TRIAL_LIST_FIELDS:
        trials = protocol.get(list_field, [])
        if not isinstance(trials, list):
            errors.append(f"protocol.{list_field}:must_be_list")
            continue
        for index, trial in enumerate(trials):
            if not isinstance(trial, Mapping):
                errors.append(f"{list_field}[{index}]:must_be_object")
                continue
            for field in OPTIONAL_TRIAL_IDENTITY_FIELDS:
                if field not in trial:
                    errors.append(
                        f"{list_field}[{index}].{field}:"
                        "missing_required_frozen_identity"
                    )
                    continue
                path = f"{list_field}[{index}].{field}"
                value = trial[field]
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or value != value.strip()
                ):
                    errors.append(f"{path}:must_be_canonical_non_empty_string")
                    continue
                frozen_value = top_level_identity.get(field)
                if frozen_value is not None and value != frozen_value:
                    errors.append(f"{path}:conflicts_with_protocol")


def _validate_custom_m4_protocol(
    protocol: Mapping[str, Any],
    top_level_identity: Mapping[str, str | None],
    errors: list[str],
) -> None:
    if not all(
        top_level_identity.get(field) is not None
        for field in OPTIONAL_TRIAL_IDENTITY_FIELDS
    ):
        return
    seeds = protocol.get("seeds")
    reactivity_k_exec = protocol.get("reactivity_k_exec")
    selected_k_exec = protocol.get("selected_k_exec")
    if not isinstance(seeds, list) or not isinstance(reactivity_k_exec, list):
        errors.append("custom_protocol_parameters_invalid")
        return
    conditions = protocol.get("conditions")
    include_language = bool(
        isinstance(conditions, Mapping)
        and isinstance(conditions.get("L5"), Mapping)
        and conditions["L5"].get("enabled") is True
    )
    try:
        expected = build_m4_plan(
            policy_id=top_level_identity["policy_id"],
            checkpoint_id=top_level_identity["checkpoint_id"],
            seeds=tuple(seeds),
            reactivity_k_exec=tuple(reactivity_k_exec),
            selected_k_exec=selected_k_exec,
            include_language=include_language,
        ).to_dict()
    except (TypeError, ValueError):
        errors.append("custom_protocol_parameters_invalid")
        return
    _compare_m4_protocol(expected, protocol, "custom", errors)


def _compare_m4_protocol(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    error_prefix: str,
    errors: list[str],
) -> None:
    fields = (
        "schema_version",
        "protocol_id",
        "protocol_mode",
        "customization_reasons",
        "seeds",
        "reactivity_k_exec",
        "selected_k_exec",
        "seed_disjointness",
        "reset_admission",
        "action_spec",
        "conditions",
        "reactivity_trials",
        "robustness_trials",
        "planned_trials",
    )
    for field in fields:
        if not _strict_json_equal(actual.get(field), expected[field]):
            errors.append(f"{error_prefix}_protocol_field_mismatch:{field}")


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def _seed_disjointness_status(protocol: Mapping[str, Any]) -> dict[str, Any]:
    default = {
        "status": "unchecked",
        "collection_manifest_id": None,
        "reason": "collection_manifest_not_verified",
    }
    evidence = protocol.get("seed_disjointness")
    if evidence is None:
        return default
    if not isinstance(evidence, Mapping):
        raise TypeError("protocol seed_disjointness must be an object")

    declared_status = evidence.get("status", "unchecked")
    if declared_status not in {"checked", "unchecked"}:
        raise ValueError(
            "protocol seed_disjointness status must be checked or unchecked"
        )
    manifest_id = evidence.get("collection_manifest_id")
    if manifest_id is not None:
        if not isinstance(manifest_id, str) or not manifest_id.strip():
            raise ValueError(
                "protocol seed_disjointness collection_manifest_id must be "
                "a non-empty string or null"
            )
        manifest_id = manifest_id.strip()

    if declared_status == "checked" and manifest_id is None:
        return {
            **default,
            "declared_status": "checked",
            "reason": "missing_collection_manifest_id",
        }
    if declared_status == "checked":
        return {
            "status": "unchecked",
            "collection_manifest_id": manifest_id,
            "declared_status": "checked",
            "reason": "collection_manifest_not_verified",
        }
    return {
        "status": "unchecked",
        "collection_manifest_id": manifest_id,
        "reason": "collection_manifest_not_verified",
    }


def _has_planned_r1_trial(protocol: Mapping[str, Any]) -> bool:
    return any(
        trial.get("condition_id") == "R1"
        for field in TRIAL_LIST_FIELDS
        for trial in protocol.get(field, [])
        if isinstance(trial, Mapping)
    )


def _declared_planned_count(protocol: Mapping[str, Any]) -> int | None:
    planned = protocol.get("planned_trials")
    if isinstance(planned, bool) or (
        planned is not None and not isinstance(planned, int)
    ):
        raise TypeError("protocol planned_trials must be an integer or null")
    if isinstance(planned, int) and planned < 0:
        raise ValueError("protocol planned_trials must be non-negative")
    return planned


def _planned_trial_identities(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    identities = []
    for list_field in TRIAL_LIST_FIELDS:
        if list_field not in protocol:
            continue
        trials = protocol[list_field]
        if not isinstance(trials, list):
            raise TypeError(f"protocol {list_field} must be a list")
        for index, trial in enumerate(trials):
            if not isinstance(trial, Mapping):
                raise TypeError(f"protocol {list_field}[{index}] must be an object")
            identities.append(
                _planned_trial_identity(trial, protocol, list_field, index)
            )
    return identities


def _planned_trial_identity(
    trial: Mapping[str, Any],
    protocol: Mapping[str, Any],
    list_field: str,
    index: int,
) -> dict[str, Any]:
    prefix = f"protocol {list_field}[{index}]"
    trial_id = _optional_label(trial.get("trial_id"), None)
    condition_id = _optional_label(trial.get("condition_id"), None)
    seed = _optional_non_negative_int(trial.get("seed"), f"{prefix} seed")
    k_exec = _optional_positive_int(trial.get("k_exec"), f"{prefix} k_exec")
    required_values = {
        "trial_id": trial_id,
        "seed": seed,
        "condition_id": condition_id,
        "k_exec": k_exec,
    }
    missing_fields = [
        field for field, value in required_values.items() if value is None
    ]
    if missing_fields:
        joined = ", ".join(missing_fields)
        raise ValueError(f"{prefix} is missing required identity fields: {joined}")

    identity = dict(required_values)
    for field in OPTIONAL_TRIAL_IDENTITY_FIELDS:
        top_level_value = _canonical_label_or_none(protocol.get(field))
        trial_value = (
            _canonical_label_or_none(trial.get(field)) if field in trial else None
        )
        value = top_level_value if top_level_value is not None else trial_value
        if value is not None:
            identity[field] = value
    return identity


def _canonical_label_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _observed_trial_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = {field: record[field] for field in REQUIRED_TRIAL_IDENTITY_FIELDS}
    for field in OPTIONAL_TRIAL_IDENTITY_FIELDS:
        value = record[field]
        if value != "unspecified":
            identity[field] = value
    return identity


def _reconcile_trials(
    planned_trials: list[dict[str, Any]],
    observed_trials: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[int],
    set[int],
]:
    unmatched_planned = set(range(len(planned_trials)))
    unexpected = []
    matched_indices = set()
    unexpected_indices = set()
    for observed_index, observed in enumerate(observed_trials):
        match = next(
            (
                index
                for index, planned in enumerate(planned_trials)
                if index in unmatched_planned
                and _trial_identities_match(planned, observed)
            ),
            None,
        )
        if match is not None:
            unmatched_planned.remove(match)
            matched_indices.add(observed_index)
            continue
        if not any(
            _trial_identities_match(planned, observed) for planned in planned_trials
        ):
            unexpected.append(observed)
            unexpected_indices.add(observed_index)

    missing = [
        planned
        for index, planned in enumerate(planned_trials)
        if index in unmatched_planned
    ]
    return missing, unexpected, matched_indices, unexpected_indices


def _trial_identities_match(
    planned: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    if any(
        planned[field] != observed[field] for field in REQUIRED_TRIAL_IDENTITY_FIELDS
    ):
        return False
    return all(
        field not in planned
        or (field in observed and planned[field] == observed[field])
        for field in OPTIONAL_TRIAL_IDENTITY_FIELDS
    )


def _duplicate_observed_trials(
    observed_trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    grouped: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
    for observed_index, identity in enumerate(observed_trials):
        key = tuple(identity[field] for field in REQUIRED_TRIAL_IDENTITY_FIELDS)
        grouped.setdefault(key, []).append((observed_index, identity))

    duplicates = []
    duplicate_indices = set()
    for values in grouped.values():
        if len(values) < 2:
            continue
        duplicate_indices.update(index for index, _identity in values)
        duplicate = {
            field: values[0][1][field] for field in REQUIRED_TRIAL_IDENTITY_FIELDS
        }
        for field in OPTIONAL_TRIAL_IDENTITY_FIELDS:
            available_values = {identity.get(field) for _, identity in values}
            if len(available_values) == 1 and None not in available_values:
                duplicate[field] = available_values.pop()
        duplicate["observed_count"] = len(values)
        duplicates.append(duplicate)
    return duplicates, duplicate_indices


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_int(value: Any, name: str) -> int | None:
    result = _optional_non_negative_int(value, name)
    if result is not None and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _protocol_dict(
    protocol: Mapping[str, Any] | Any,
) -> tuple[dict[str, Any], list[str]]:
    if isinstance(protocol, Mapping):
        payload = dict(protocol)
    elif hasattr(protocol, "to_dict"):
        payload = protocol.to_dict()
    else:
        raise TypeError("protocol must be a mapping or expose to_dict()")
    errors: list[str] = []
    safe_payload = _json_safe_value(payload, "protocol", errors)
    if not isinstance(safe_payload, dict):
        raise TypeError("protocol must serialize to an object")
    _require_strict_json(safe_payload)
    return safe_payload, errors


def _json_safe_value(value: Any, path: str, errors: list[str]) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        errors.append(f"{path}:non_finite_number")
        if math.isnan(value):
            label = "NaN"
        elif value > 0:
            label = "Infinity"
        else:
            label = "-Infinity"
        return {"invalid_non_finite_number": label}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path}:non_string_object_key")
                key = f"invalid-key:{type(key).__name__}:{key!s}"
            result[key] = _json_safe_value(item, f"{path}.{key}", errors)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(item, f"{path}[{index}]", errors)
            for index, item in enumerate(value)
        ]
    errors.append(f"{path}:unsupported_json_type:{type(value).__name__}")
    return {"invalid_python_type": type(value).__name__}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: episode must be a JSON object")
            records.append(value)
    return records


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: protocol must be a JSON object")
    return value


def _get(record: Mapping[str, Any] | Any, name: str, default: Any = ...) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is ...:
        raise ValueError(f"episode record is missing required field: {name}")
    return default


def _sortable_group_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)


def _require_strict_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "report values must be finite and JSON serializable"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
