"""Model-independent evaluation protocol, metrics, and report helpers."""

from .metrics import (
    CANONICAL_FAILURE_TYPES,
    action_smoothness,
    failure_taxonomy,
    latency_percentiles,
    wilson_interval,
)
from .protocol import (
    M4_EVAL_SEEDS,
    REACTIVITY_K_EXEC,
    EvaluationPlan,
    assert_disjoint_seeds,
    build_m4_plan,
)


def build_report(*args, **kwargs):
    """Lazily import :mod:`evaluation.report` so its CLI runs without warnings."""

    from .report import build_report as _build_report

    return _build_report(*args, **kwargs)


__all__ = [
    "CANONICAL_FAILURE_TYPES",
    "M4_EVAL_SEEDS",
    "REACTIVITY_K_EXEC",
    "EvaluationPlan",
    "action_smoothness",
    "assert_disjoint_seeds",
    "build_m4_plan",
    "build_report",
    "failure_taxonomy",
    "latency_percentiles",
    "wilson_interval",
]
