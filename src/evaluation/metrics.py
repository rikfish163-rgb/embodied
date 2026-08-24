"""Pure evaluation metrics that do not import an environment or policy."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

CANONICAL_FAILURE_TYPES = (
    "misalignment",
    "empty_grasp",
    "object_slip",
    "box_collision",
    "placement_failure",
    "timeout",
    "invalid_action",
    "inference_error",
    "environment_error",
    "unclassified",
)

UNKNOWN_STAGE = "unknown"


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a two-sided Wilson score interval for a binomial proportion.

    ``trials`` must be positive. An empty evaluation is deliberately rejected:
    it must not turn into a report that resembles an observed result.
    """

    _validate_count("successes", successes)
    _validate_count("trials", trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a real number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")

    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    low = max(0.0, center - radius)
    high = min(1.0, center + radius)
    # Avoid tiny floating-point residue at the two exact binomial boundaries.
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return {"confidence": confidence, "low": low, "high": high}


def latency_percentiles(values_ms: Iterable[float]) -> dict[str, int | float | None]:
    """Summarize non-negative latency samples with linear percentiles."""

    samples = np.asarray(list(values_ms), dtype=float)
    if samples.ndim != 1:
        raise ValueError("latency samples must be a one-dimensional sequence")
    if samples.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    if not np.all(np.isfinite(samples)):
        raise ValueError("latency samples must be finite")
    if np.any(samples < 0.0):
        raise ValueError("latency samples must be non-negative")

    p50, p95, p99 = np.percentile(samples, [50, 95, 99], method="linear")
    return {
        "count": int(samples.size),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
    }


def action_smoothness(
    actions: Sequence[Sequence[float]] | np.ndarray,
    *,
    action_scale: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    r"""Return mean scaled L2 first-difference magnitude (lower is smoother).

    For actions :math:`a_t \in R^D` and a positive, fixed scale vector
    :math:`s`, the metric is

    .. math::

       S = \frac{1}{T-1}\sum_{t=1}^{T-1}
           \left\|(a_t-a_{t-1}) / s\right\|_2.

    The default scale is all ones. Comparisons are valid only when policies use
    the same action representation and scale. A one-step sequence has no
    transition, so its value is ``None`` instead of a fabricated zero.
    """

    array = np.asarray(actions, dtype=float)
    if array.ndim != 2:
        raise ValueError("actions must have shape (time, action_dim)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("actions must contain at least one finite action")
    if not np.all(np.isfinite(array)):
        raise ValueError("actions must be finite")

    if action_scale is None:
        scale = np.ones(array.shape[1], dtype=float)
    else:
        scale = np.asarray(action_scale, dtype=float)
        if scale.shape != (array.shape[1],):
            raise ValueError("action_scale must match action_dim")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("action_scale must contain positive finite values")

    transitions = array.shape[0] - 1
    value: float | None
    if transitions == 0:
        value = None
    else:
        scaled_delta = np.diff(array, axis=0) / scale
        value = float(np.linalg.norm(scaled_delta, axis=1).mean())
    return {
        "metric": "mean_scaled_l2_delta",
        "formula": "mean_t(norm_2((action[t]-action[t-1])/action_scale))",
        "lower_is_smoother": True,
        "action_dim": int(array.shape[1]),
        "transitions": int(transitions),
        "action_scale": scale.tolist(),
        "value": value,
    }


def failure_taxonomy(records: Iterable[Mapping[str, Any] | Any]) -> dict[str, Any]:
    """Cross-tabulate failed episodes by execution stage and failure type.

    Stage is intentionally open vocabulary so the current expert stages and
    future BC/ACT runner stages can coexist. Failure type is controlled by
    :data:`CANONICAL_FAILURE_TYPES`; absent or unknown values are counted as
    ``unclassified`` and their raw labels are surfaced separately.
    """

    by_stage: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    raw_unclassified: Counter[str] = Counter()
    matrix: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        success = _required_bool(record, "success")
        if success:
            continue
        stage_value = _failure_field(record, "failure_stage", "stage")
        stage = _normalized_label(stage_value, default=UNKNOWN_STAGE)
        raw_type = _failure_field(record, "failure_type", "type")
        failure_type = _normalized_label(raw_type, default="unclassified")
        if failure_type not in CANONICAL_FAILURE_TYPES:
            raw_unclassified[failure_type] += 1
            failure_type = "unclassified"

        by_stage[stage] += 1
        by_type[failure_type] += 1
        matrix[stage][failure_type] += 1

    return {
        "total_failures": int(sum(by_stage.values())),
        "by_stage": dict(sorted(by_stage.items())),
        "by_type": dict(sorted(by_type.items())),
        "stage_by_type": {
            stage: dict(sorted(counts.items()))
            for stage, counts in sorted(matrix.items())
        },
        "raw_unclassified_types": dict(sorted(raw_unclassified.items())),
    }


def _validate_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _required_bool(record: Mapping[str, Any] | Any, name: str) -> bool:
    value = _field(record, name)
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, bool):
        raise TypeError(f"episode {name} must be bool")
    return value


def _failure_field(
    record: Mapping[str, Any] | Any,
    direct_name: str,
    nested_name: str,
) -> Any:
    value = _field(record, direct_name, None)
    if value is not None:
        return value
    failure = _field(record, "failure", None)
    if isinstance(failure, Mapping):
        return failure.get(nested_name)
    return None


def _field(record: Mapping[str, Any] | Any, name: str, default: Any = ...) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is ...:
        raise ValueError(f"episode record is missing required field: {name}")
    return default


def _normalized_label(value: Any, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    label = value.strip().lower()
    return label or default
