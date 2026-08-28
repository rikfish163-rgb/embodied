"""Deterministic action-only replay for eligible M2 expert episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from env import scene as env_scene
from env.asset_provenance import collect_asset_provenance
from env.pick_place import PickPlace
from expert.evaluate import _git_state

from .lineage import validate_lineage_revalidation_receipt
from .manifest import (
    _validate_assets,
    _validate_environment,
    _validate_git,
    append_jsonl_fsync,
    assets_payload,
    atomic_write_json_no_clobber,
    canonical_json_bytes,
    compiled_model_fingerprint,
    environment_payload,
    initialize_jsonl_no_clobber,
    load_json_object_with_sha256,
    load_jsonl_relative_with_sha256,
    manifest_sha256,
    maximum_episode_steps,
    open_verified_episode,
    validate_collection_manifest,
    validate_relative_path,
)

REPLAY_SCHEMA = "m2-replay-summary.v1"
REPLAY_PLAN_SCHEMA = "m2-replay-plan.v1"
REPLAY_PLAN_FILENAME = "plan.json"
REPLAY_TRIALS_FILENAME = "trials.jsonl"
REPLAY_SUMMARY_FILENAME = "summary.json"
LINEAGE_RECEIPT_FILENAME = "lineage-revalidation.json"
FORMAL_REPLAY_COUNT = 20
FORMAL_REPLAY_REQUIRED = 18
FORMAL_SELECTION_SEED = 20260824
SELECTION_ALGORITHM = "numpy-pcg64-permutation.v1"
PAIR_SELECTION_ALGORITHM = "sha256-rank-v1"
PAIR_REPLAY_TRIAL_SCHEMA = "m2-replay-trial.v1"
PAIR_ACTION_HASH_ALGORITHM = "sha256-f32le-c-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SUMMARY_KEYS = {
    "schema_version",
    "generated_at",
    "formal",
    "source_manifest_sha256",
    "source_split",
    "source_manifest_formal",
    "replay_provenance",
    "selection",
    "selected_seeds",
    "trial_count",
    "success_count",
    "trials_path",
    "trials_sha256",
    "gate",
    "cli_config",
}
_TRIAL_KEYS = {
    "trial_index",
    "seed",
    "path",
    "num_actions",
    "status",
    "success",
    "error",
}
_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "generated_at",
    "formal",
    "selection",
    "manifests",
    "candidate_count",
    "candidate_set_sha256",
    "selected_trials",
    "cli_config",
}
_PLAN_MANIFEST_KEYS = {
    "split",
    "manifest_id",
    "file_sha256",
    "formal",
    "attempt_count",
    "eligible_success_count",
}
_PLAN_TRIAL_KEYS = {
    "trial_id",
    "rank",
    "manifest_id",
    "split",
    "seed",
    "source_relative_path",
    "source_file_sha256",
    "source_num_steps",
}
_PAIR_TRIAL_KEYS = {
    "schema_version",
    "trial_id",
    "rank",
    "plan_id",
    "manifest_id",
    "split",
    "seed",
    "source_relative_path",
    "source_file_sha256",
    "source_num_steps",
    "action_dataset_sha256",
    "action_hash_algorithm",
    "runner_config_id",
    "reset_seed",
    "expected_steps",
    "executed_steps",
    "success",
    "failure_stage",
    "exception_type",
    "exception_message",
    "final_hold_steps",
    "started_at_utc",
    "finished_at_utc",
}
_PAIR_SUMMARY_KEYS = {
    "schema_version",
    "summary_id",
    "generated_at",
    "formal",
    "plan_id",
    "plan_path",
    "plan_sha256",
    "trials_path",
    "trials_sha256",
    "replay_provenance",
    "lineage_receipt",
    "identity_reconciliation",
    "trial_count",
    "success_count",
    "failed_trial_ids",
    "gate",
    "cli_config",
}


class ReplayProvenanceError(RuntimeError):
    """The current runtime cannot reproduce a formal source collection."""


@dataclass(frozen=True)
class ReplayConfig:
    manifest_path: Path
    output_dir: Path
    selection_seed: int = FORMAL_SELECTION_SEED
    count: int = FORMAL_REPLAY_COUNT
    smoke: bool = False
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True)
class ReplayPlanConfig:
    train_manifest_path: Path
    validation_manifest_path: Path
    output_dir: Path
    selection_seed: int = FORMAL_SELECTION_SEED
    count: int = FORMAL_REPLAY_COUNT
    smoke: bool = False


@dataclass(frozen=True)
class PairReplayConfig:
    train_manifest_path: Path
    validation_manifest_path: Path
    replay_dir: Path
    smoke: bool = False
    lineage_receipt_path: Path | None = None
    project_root: Path = PROJECT_ROOT


@dataclass(frozen=True)
class ReplayValidationIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class ReplayValidationReport:
    path: Path
    errors: tuple[ReplayValidationIssue, ...]
    summary: dict[str, Any] | None = None
    sha256: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                f"invalid replay summary {self.path}:\n{self.format_errors()}"
            )


@dataclass(frozen=True)
class ReplayPlanValidationReport:
    path: Path
    errors: tuple[ReplayValidationIssue, ...]
    plan: dict[str, Any] | None = None
    sha256: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                f"invalid replay plan {self.path}:\n{self.format_errors()}"
            )


def select_replay_entries(
    manifest: dict[str, Any],
    *,
    count: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    if type(count) is not int or count <= 0:
        raise ValueError("replay count must be a positive integer")
    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("selection_seed must be a non-negative integer")
    eligible = manifest.get("eligible_successes")
    if not isinstance(eligible, list):
        raise ValueError("manifest eligible_successes must be an array")
    if count > len(eligible):
        raise ValueError(
            f"cannot select {count} successes without replacement from {len(eligible)}"
        )
    seeds = [entry.get("seed") for entry in eligible if isinstance(entry, dict)]
    if len(seeds) != len(eligible) or len(set(seeds)) != len(seeds):
        raise ValueError("eligible success seeds must be unique")
    rng = np.random.Generator(np.random.PCG64(selection_seed))
    indices = rng.permutation(len(eligible))[:count]
    return [dict(eligible[int(index)]) for index in indices]


def create_replay_plan(
    config: ReplayPlanConfig,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Publish the deterministic train+validation replay selection plan."""

    _validate_replay_plan_config(config)
    train_path = Path(config.train_manifest_path).expanduser().absolute()
    validation_path = Path(config.validation_manifest_path).expanduser().absolute()
    output_dir = Path(config.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"replay output directory already exists: {output_dir}")

    sources = _validated_pair_sources(train_path, validation_path)
    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    plan = _build_replay_plan_document(
        sources,
        selection_seed=config.selection_seed,
        count=config.count,
        smoke=config.smoke,
        generated_at=timestamp.astimezone(UTC),
    )
    _require_unchanged_source_manifests(sources)

    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / REPLAY_PLAN_FILENAME
    atomic_write_json_no_clobber(plan_path, plan)
    published, _ = load_json_object_with_sha256(plan_path)
    if published != plan:
        raise RuntimeError("published replay plan differs from the validated plan")
    return plan


def validate_replay_plan(
    path: Path,
    *,
    train_manifest_path: Path,
    validation_manifest_path: Path,
) -> ReplayPlanValidationReport:
    """Rebuild a replay plan from both manifests and reject any substitution."""

    plan_path = Path(path)
    errors: list[ReplayValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(ReplayValidationIssue(code, location, message))

    try:
        plan, initial_sha = load_json_object_with_sha256(plan_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.plan", "/", f"cannot read strict replay plan JSON: {error}")
        return ReplayPlanValidationReport(plan_path, tuple(errors), None, None)

    _validate_replay_plan_schema(plan, add)
    if errors:
        return ReplayPlanValidationReport(plan_path, tuple(errors), plan, initial_sha)

    try:
        sources = _validated_pair_sources(
            Path(train_manifest_path).expanduser().absolute(),
            Path(validation_manifest_path).expanduser().absolute(),
        )
        generated_at = datetime.fromisoformat(plan["generated_at"])
        expected = _build_replay_plan_document(
            sources,
            selection_seed=plan["selection"]["seed"],
            count=plan["selection"]["count"],
            smoke=plan["cli_config"]["smoke"],
            generated_at=generated_at,
        )
        if plan != expected:
            add(
                "replay.plan.identity",
                "/",
                "plan does not match deterministic reconstruction from both manifests",
            )
        _require_unchanged_source_manifests(sources)
    except (OSError, TypeError, ValueError) as error:
        add("replay.plan.source", "/manifests", str(error))

    try:
        _, final_sha = load_json_object_with_sha256(plan_path)
        if final_sha != initial_sha:
            add("replay.plan", "/", "plan changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.plan", "/", f"cannot complete stable validation: {error}")
    return ReplayPlanValidationReport(plan_path, tuple(errors), plan, initial_sha)


def _validated_pair_sources(
    train_path: Path,
    validation_path: Path,
) -> dict[str, tuple[Path, dict[str, Any], str]]:
    if train_path == validation_path:
        raise ValueError("train and validation manifests must be different files")
    sources: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for expected_split, path in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        report = validate_collection_manifest(path)
        report.raise_for_errors()
        if report.manifest is None or report.sha256 is None:
            raise ValueError(f"validated {expected_split} manifest is unavailable")
        if report.manifest["split"] != expected_split:
            raise ValueError(
                f"{expected_split} manifest path contains split "
                f"{report.manifest['split']!r}"
            )
        sources[expected_split] = (path, report.manifest, report.sha256)

    train_seeds = {attempt["seed"] for attempt in sources["train"][1]["attempts"]}
    validation_seeds = {
        attempt["seed"] for attempt in sources["validation"][1]["attempts"]
    }
    overlap = sorted(train_seeds.intersection(validation_seeds))
    if overlap:
        raise ValueError(f"train and validation attempted seeds overlap: {overlap}")
    return sources


def _build_replay_plan_document(
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
    *,
    selection_seed: int,
    count: int,
    smoke: bool,
    generated_at: datetime,
) -> dict[str, Any]:
    if set(sources) != {"train", "validation"}:
        raise ValueError("replay plan requires train and validation manifests")
    if smoke is False:
        train = sources["train"][1]
        validation = sources["validation"][1]
        if not (
            train["formal"] is True
            and validation["formal"] is True
            and len(train["eligible_successes"]) == 200
            and len(validation["eligible_successes"]) == 40
        ):
            raise ValueError(
                "formal replay plan requires exactly 200 train and 40 validation "
                "eligible successes from formal manifests"
            )

    manifest_refs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        _path, manifest, digest = sources[split]
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{split} manifest digest is not SHA-256")
        manifest_id = f"sha256:{digest}"
        eligible = manifest["eligible_successes"]
        manifest_refs.append(
            {
                "split": split,
                "manifest_id": manifest_id,
                "file_sha256": digest,
                "formal": manifest["formal"],
                "attempt_count": manifest["attempt_count"],
                "eligible_success_count": len(eligible),
            }
        )
        for entry in eligible:
            rank_payload = {
                "algorithm": PAIR_SELECTION_ALGORITHM,
                "manifest_id": manifest_id,
                "selection_seed": selection_seed,
                "split": split,
                "seed": entry["seed"],
                "relative_hdf5_path": entry["path"],
                "episode_sha256": entry["sha256"],
            }
            rank = hashlib.sha256(canonical_json_bytes(rank_payload)).hexdigest()
            candidates.append(
                {
                    "rank": rank,
                    "manifest_id": manifest_id,
                    "split": split,
                    "seed": entry["seed"],
                    "source_relative_path": entry["path"],
                    "source_file_sha256": entry["sha256"],
                    "source_num_steps": entry["num_steps"],
                }
            )

    if count > len(candidates):
        raise ValueError(
            f"cannot select {count} replay trials from {len(candidates)} candidates"
        )
    candidates.sort(
        key=lambda item: (
            item["rank"],
            item["split"],
            item["seed"],
            item["source_relative_path"],
        )
    )
    if len(
        {
            (
                item["split"],
                item["seed"],
                item["source_relative_path"],
                item["source_file_sha256"],
            )
            for item in candidates
        }
    ) != len(candidates):
        raise ValueError("replay candidate identities must be unique")

    candidate_set_sha256 = hashlib.sha256(canonical_json_bytes(candidates)).hexdigest()
    selected_trials: list[dict[str, Any]] = []
    for candidate in candidates[:count]:
        identity = {
            "schema_version": "m2-replay-trial-identity.v1",
            **candidate,
        }
        selected_trials.append(
            {
                "trial_id": _content_id(identity),
                **candidate,
            }
        )

    plan: dict[str, Any] = {
        "schema_version": REPLAY_PLAN_SCHEMA,
        "plan_id": None,
        "generated_at": generated_at.astimezone(UTC).isoformat(),
        "formal": not smoke,
        "selection": {
            "algorithm": PAIR_SELECTION_ALGORITHM,
            "seed": selection_seed,
            "count": count,
            "without_replacement": True,
        },
        "manifests": manifest_refs,
        "candidate_count": len(candidates),
        "candidate_set_sha256": candidate_set_sha256,
        "selected_trials": selected_trials,
        "cli_config": {
            "selection_seed": selection_seed,
            "count": count,
            "smoke": smoke,
        },
    }
    plan["plan_id"] = _content_id(_replay_plan_identity_payload(plan))
    return plan


def _replay_plan_identity_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "formal": plan["formal"],
        "selection": plan["selection"],
        "manifests": plan["manifests"],
        "candidate_count": plan["candidate_count"],
        "candidate_set_sha256": plan["candidate_set_sha256"],
        "selected_trials": plan["selected_trials"],
        "cli_config": plan["cli_config"],
    }


def _content_id(payload: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _require_unchanged_source_manifests(
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
) -> None:
    for split, (path, _manifest, expected_digest) in sources.items():
        _payload, current_digest = load_json_object_with_sha256(path)
        if current_digest != expected_digest:
            raise RuntimeError(f"{split} collection manifest changed during planning")


def _validate_replay_plan_config(config: ReplayPlanConfig) -> None:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if type(config.count) is not int or config.count <= 0:
        raise ValueError("count must be a positive integer")
    if config.count > FORMAL_REPLAY_COUNT:
        raise ValueError("replay count cannot exceed 20")
    if config.count != FORMAL_REPLAY_COUNT and not config.smoke:
        raise ValueError("a replay count smaller than 20 requires --smoke")
    if type(config.selection_seed) is not int or config.selection_seed < 0:
        raise ValueError("selection_seed must be a non-negative integer")
    if config.selection_seed != FORMAL_SELECTION_SEED and not config.smoke:
        raise ValueError("a noncanonical selection seed requires --smoke")


def _validate_replay_plan_schema(plan: dict[str, Any], add: Any) -> None:
    if set(plan) != _PLAN_KEYS:
        add("replay.plan.schema", "/", "replay plan has the wrong fields")
        return
    if plan["schema_version"] != REPLAY_PLAN_SCHEMA:
        add(
            "replay.plan.schema",
            "/schema_version",
            f"expected {REPLAY_PLAN_SCHEMA}",
        )
    try:
        generated_at = datetime.fromisoformat(plan["generated_at"])
        if generated_at.tzinfo is None:
            raise ValueError("timezone is missing")
    except (TypeError, ValueError):
        add(
            "replay.plan.schema",
            "/generated_at",
            "generated_at must be timezone-aware ISO-8601",
        )
    if type(plan["formal"]) is not bool:
        add("replay.plan.schema", "/formal", "formal must be boolean")
    if not _is_content_id(plan["plan_id"]):
        add("replay.plan.schema", "/plan_id", "plan_id must be a SHA-256 ID")

    selection = plan["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "algorithm",
        "seed",
        "count",
        "without_replacement",
    }:
        add("replay.plan.schema", "/selection", "selection has the wrong fields")
        return
    if selection["algorithm"] != PAIR_SELECTION_ALGORITHM:
        add(
            "replay.plan.selection",
            "/selection/algorithm",
            "selection algorithm is not frozen",
        )
    if type(selection["seed"]) is not int or selection["seed"] < 0:
        add("replay.plan.schema", "/selection/seed", "seed must be non-negative")
    if (
        type(selection["count"]) is not int
        or not 1 <= selection["count"] <= FORMAL_REPLAY_COUNT
    ):
        add("replay.plan.schema", "/selection/count", "count must be in 1..20")
    if selection["without_replacement"] is not True:
        add(
            "replay.plan.selection",
            "/selection/without_replacement",
            "selection must be without replacement",
        )

    manifests = plan["manifests"]
    if not isinstance(manifests, list) or len(manifests) != 2:
        add("replay.plan.schema", "/manifests", "plan requires two manifests")
        return
    seen_splits: set[str] = set()
    for index, manifest in enumerate(manifests):
        location = f"/manifests/{index}"
        if not isinstance(manifest, dict) or set(manifest) != _PLAN_MANIFEST_KEYS:
            add("replay.plan.schema", location, "manifest reference has wrong fields")
            continue
        split = manifest["split"]
        if split not in {"train", "validation"}:
            add("replay.plan.schema", f"{location}/split", "split is invalid")
        else:
            seen_splits.add(split)
        if not _is_content_id(manifest["manifest_id"]):
            add(
                "replay.plan.schema",
                f"{location}/manifest_id",
                "manifest_id must be a SHA-256 ID",
            )
        if not _is_sha256(manifest["file_sha256"]):
            add(
                "replay.plan.schema",
                f"{location}/file_sha256",
                "file digest must be SHA-256",
            )
        if manifest["manifest_id"] != f"sha256:{manifest['file_sha256']}":
            add(
                "replay.plan.identity",
                f"{location}/manifest_id",
                "manifest_id and file digest disagree",
            )
        if type(manifest["formal"]) is not bool:
            add("replay.plan.schema", f"{location}/formal", "formal must be boolean")
        for field in ("attempt_count", "eligible_success_count"):
            if type(manifest[field]) is not int or manifest[field] < 0:
                add(
                    "replay.plan.schema",
                    f"{location}/{field}",
                    f"{field} must be a non-negative integer",
                )
    if seen_splits != {"train", "validation"}:
        add(
            "replay.plan.identity",
            "/manifests",
            "manifest references must contain train and validation",
        )

    if type(plan["candidate_count"]) is not int or plan["candidate_count"] <= 0:
        add(
            "replay.plan.schema",
            "/candidate_count",
            "candidate_count must be positive",
        )
    if not _is_sha256(plan["candidate_set_sha256"]):
        add(
            "replay.plan.schema",
            "/candidate_set_sha256",
            "candidate set digest must be SHA-256",
        )

    trials = plan["selected_trials"]
    if not isinstance(trials, list) or len(trials) != selection["count"]:
        add(
            "replay.plan.identity",
            "/selected_trials",
            "selected trial count differs from selection",
        )
        return
    identities: set[tuple[Any, ...]] = set()
    for index, trial in enumerate(trials):
        location = f"/selected_trials/{index}"
        if not isinstance(trial, dict) or set(trial) != _PLAN_TRIAL_KEYS:
            add("replay.plan.schema", location, "trial has the wrong fields")
            continue
        if not _is_content_id(trial["trial_id"]):
            add(
                "replay.plan.schema",
                f"{location}/trial_id",
                "trial_id must be a SHA-256 ID",
            )
        if not _is_sha256(trial["rank"]):
            add("replay.plan.schema", f"{location}/rank", "rank must be SHA-256")
        if not _is_content_id(trial["manifest_id"]):
            add(
                "replay.plan.schema",
                f"{location}/manifest_id",
                "manifest_id must be a SHA-256 ID",
            )
        if trial["split"] not in {"train", "validation"}:
            add("replay.plan.schema", f"{location}/split", "split is invalid")
        if type(trial["seed"]) is not int or trial["seed"] < 0:
            add("replay.plan.schema", f"{location}/seed", "seed must be non-negative")
        if not isinstance(trial["source_relative_path"], str):
            add(
                "replay.plan.schema",
                f"{location}/source_relative_path",
                "source path must be text",
            )
        else:
            try:
                validate_relative_path(trial["source_relative_path"])
            except ValueError as error:
                add(
                    "replay.plan.schema", f"{location}/source_relative_path", str(error)
                )
        if not _is_sha256(trial["source_file_sha256"]):
            add(
                "replay.plan.schema",
                f"{location}/source_file_sha256",
                "source digest must be SHA-256",
            )
        if type(trial["source_num_steps"]) is not int or trial["source_num_steps"] <= 0:
            add(
                "replay.plan.schema",
                f"{location}/source_num_steps",
                "source_num_steps must be positive",
            )
        identity = (
            trial["trial_id"],
            trial["rank"],
            trial["split"],
            trial["seed"],
            trial["source_relative_path"],
            trial["source_file_sha256"],
        )
        if identity in identities:
            add("replay.plan.identity", location, "trial identity is duplicated")
        identities.add(identity)

    cli = plan["cli_config"]
    if not isinstance(cli, dict) or set(cli) != {"selection_seed", "count", "smoke"}:
        add("replay.plan.schema", "/cli_config", "CLI config has the wrong fields")
        return
    if (
        cli["selection_seed"] != selection["seed"]
        or cli["count"] != selection["count"]
        or type(cli["smoke"]) is not bool
    ):
        add("replay.plan.identity", "/cli_config", "CLI and selection disagree")
    if type(cli["smoke"]) is bool and plan["formal"] is cli["smoke"]:
        add("replay.plan.identity", "/formal", "formal and smoke flags disagree")
    if cli["smoke"] is False and (
        selection["seed"] != FORMAL_SELECTION_SEED
        or selection["count"] != FORMAL_REPLAY_COUNT
    ):
        add(
            "replay.plan.formal",
            "/selection",
            "formal plan requires the frozen seed and count",
        )
    try:
        expected_plan_id = _content_id(_replay_plan_identity_payload(plan))
    except (TypeError, ValueError):
        add("replay.plan.schema", "/plan_id", "plan identity is not canonical JSON")
    else:
        if plan["plan_id"] != expected_plan_id:
            add("replay.plan.identity", "/plan_id", "plan identity digest is wrong")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_content_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _SHA256_RE.fullmatch(value.removeprefix("sha256:")) is not None
    )


def _validated_lineage_reference(
    path: Path,
    *,
    replay_dir: Path,
    project_root: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    require_formal: bool,
) -> dict[str, Any]:
    lineage_path = Path(path).expanduser().absolute()
    expected_path = Path(replay_dir).expanduser().absolute() / LINEAGE_RECEIPT_FILENAME
    if lineage_path != expected_path or lineage_path.is_symlink():
        raise ReplayProvenanceError(
            f"lineage receipt must be {LINEAGE_RECEIPT_FILENAME} inside replay_dir"
        )
    report = validate_lineage_revalidation_receipt(
        lineage_path,
        repository_root=Path(project_root).expanduser().absolute(),
        train_manifest_path=Path(train_manifest_path).expanduser().absolute(),
        validation_manifest_path=Path(validation_manifest_path).expanduser().absolute(),
    )
    if not report.valid or not report.passed or report.receipt is None:
        detail = report.format_errors() or "lineage receipt did not pass"
        raise ReplayProvenanceError(f"invalid lineage receipt:\n{detail}")
    if require_formal and report.receipt["formal"] is not True:
        raise ReplayProvenanceError("formal lineage receipt is required")
    if report.sha256 is None:
        raise ReplayProvenanceError("lineage receipt digest is unavailable")
    return {
        "provided": True,
        "path": LINEAGE_RECEIPT_FILENAME,
        "sha256": report.sha256,
        "receipt_id": report.receipt["receipt_id"],
        "validated": True,
        "formal": report.receipt["formal"],
    }


def _empty_lineage_reference() -> dict[str, Any]:
    return {
        "provided": False,
        "path": None,
        "sha256": None,
        "receipt_id": None,
        "validated": False,
        "formal": False,
    }


def replay_manifest_pair(
    config: PairReplayConfig,
    *,
    env_factory: Callable[[], Any] = PickPlace,
    git_state_fn: Callable[[Path], dict[str, Any]] = _git_state,
    asset_provenance_fn: Callable[..., Any] = collect_asset_provenance,
    environment_fingerprint_fn: Callable[[Any], dict[str, Any]] = (
        compiled_model_fingerprint
    ),
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Execute every pre-registered paired replay trial exactly once."""

    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    train_path = Path(config.train_manifest_path).expanduser().absolute()
    validation_path = Path(config.validation_manifest_path).expanduser().absolute()
    replay_dir = Path(config.replay_dir).expanduser().absolute()
    if replay_dir.is_symlink() or not replay_dir.is_dir():
        raise ValueError("replay_dir must be an existing non-symlink directory")
    trials_path = replay_dir / REPLAY_TRIALS_FILENAME
    summary_path = replay_dir / REPLAY_SUMMARY_FILENAME
    if trials_path.exists() or trials_path.is_symlink():
        raise FileExistsError(f"replay trial ledger already exists: {trials_path}")
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f"replay summary already exists: {summary_path}")

    plan_path = replay_dir / REPLAY_PLAN_FILENAME
    plan_validation = validate_replay_plan(
        plan_path,
        train_manifest_path=train_path,
        validation_manifest_path=validation_path,
    )
    plan_validation.raise_for_errors()
    plan = plan_validation.plan
    plan_digest = plan_validation.sha256
    if plan is None or plan_digest is None:
        raise RuntimeError("validated replay plan is unavailable")
    if plan["cli_config"]["smoke"] is not config.smoke:
        raise ValueError("runner smoke flag must match the pre-registered plan")
    lineage_reference = _empty_lineage_reference()
    if plan["formal"] is True or config.lineage_receipt_path is not None:
        lineage_path = config.lineage_receipt_path or (
            replay_dir / LINEAGE_RECEIPT_FILENAME
        )
        lineage_reference = _validated_lineage_reference(
            lineage_path,
            replay_dir=replay_dir,
            project_root=config.project_root,
            train_manifest_path=train_path,
            validation_manifest_path=validation_path,
            require_formal=plan["formal"],
        )

    sources = _validated_pair_sources(train_path, validation_path)
    runner_config_id = _pair_runner_config_id()
    env = env_factory()
    try:
        replay_provenance = _capture_replay_provenance(
            config,
            env,
            git_state_fn=git_state_fn,
            asset_provenance_fn=asset_provenance_fn,
            environment_fingerprint_fn=environment_fingerprint_fn,
        )
        if plan["formal"] is True:
            _require_formal_pair_replay_provenance(
                sources["train"][1], replay_provenance
            )
        trials_state = initialize_jsonl_no_clobber(trials_path)
        trials: list[dict[str, Any]] = []
        for planned in plan["selected_trials"]:
            started_at = _utc_now(now_fn)
            executed_steps = 0
            action_digest: str | None = None
            success = False
            failure_stage: str | None = "exception"
            exception_type: str | None = None
            exception_message: str | None = None
            final_hold_steps = 0
            try:
                source_path, source_manifest, _source_digest = sources[planned["split"]]
                with open_verified_episode(
                    source_path.parent,
                    planned["source_relative_path"],
                    expected_sha256=planned["source_file_sha256"],
                    max_num_steps=maximum_episode_steps(source_manifest["controller"]),
                ) as episode:
                    episode.validation.raise_for_errors()
                    metadata = episode.metadata
                    if (
                        metadata is None
                        or metadata.seed != planned["seed"]
                        or metadata.success is not True
                        or metadata.num_steps != planned["source_num_steps"]
                    ):
                        raise RuntimeError(
                            "planned replay source metadata is inconsistent"
                        )
                    actions = episode.handle["action"]
                    action_digest = _action_dataset_sha256(actions)
                    env.reset(np.random.default_rng(planned["seed"]))
                    for action_index in range(planned["source_num_steps"]):
                        action = np.asarray(actions[action_index], dtype=np.float64)
                        env.step(action)
                        executed_steps += 1
                    success = bool(env.success())
                    final_hold_steps = _placement_hold_steps(env)
                    failure_stage = None if success else "final_placement"
            except Exception as error:  # noqa: BLE001 - trial exceptions are evidence
                success = False
                failure_stage = "exception"
                exception_type = type(error).__name__
                exception_message = str(error).replace("\x00", "\\0")[:2000]

            record = {
                "schema_version": PAIR_REPLAY_TRIAL_SCHEMA,
                "trial_id": planned["trial_id"],
                "rank": planned["rank"],
                "plan_id": plan["plan_id"],
                "manifest_id": planned["manifest_id"],
                "split": planned["split"],
                "seed": planned["seed"],
                "source_relative_path": planned["source_relative_path"],
                "source_file_sha256": planned["source_file_sha256"],
                "source_num_steps": planned["source_num_steps"],
                "action_dataset_sha256": action_digest,
                "action_hash_algorithm": PAIR_ACTION_HASH_ALGORITHM,
                "runner_config_id": runner_config_id,
                "reset_seed": planned["seed"],
                "expected_steps": planned["source_num_steps"],
                "executed_steps": executed_steps,
                "success": success,
                "failure_stage": failure_stage,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "final_hold_steps": final_hold_steps,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": _utc_now(now_fn).isoformat(),
            }
            trials_state = append_jsonl_fsync(
                trials_path,
                record,
                expected_snapshot=trials_state,
            )
            trials.append(record)

        _require_unchanged_source_manifests(sources)
        final_plan_validation = validate_replay_plan(
            plan_path,
            train_manifest_path=train_path,
            validation_manifest_path=validation_path,
        )
        final_plan_validation.raise_for_errors()
        if final_plan_validation.sha256 != plan_digest:
            raise RuntimeError("replay plan changed during execution")
        if plan["formal"] is True:
            final_provenance = _capture_replay_provenance(
                config,
                env,
                git_state_fn=git_state_fn,
                asset_provenance_fn=asset_provenance_fn,
                environment_fingerprint_fn=environment_fingerprint_fn,
            )
            _require_formal_pair_replay_provenance(
                sources["train"][1], final_provenance
            )
            if final_provenance != replay_provenance:
                raise ReplayProvenanceError(
                    "formal paired replay provenance changed during execution"
                )

        reconciliation = _identity_reconciliation(plan, trials)
        success_count = sum(record["success"] is True for record in trials)
        required = _required_successes(len(plan["selected_trials"]))
        summary: dict[str, Any] = {
            "schema_version": REPLAY_SCHEMA,
            "summary_id": None,
            "generated_at": _utc_now(now_fn).isoformat(),
            "formal": plan["formal"],
            "plan_id": plan["plan_id"],
            "plan_path": REPLAY_PLAN_FILENAME,
            "plan_sha256": plan_digest,
            "trials_path": REPLAY_TRIALS_FILENAME,
            "trials_sha256": manifest_sha256(trials_path),
            "replay_provenance": replay_provenance,
            "lineage_receipt": lineage_reference,
            "identity_reconciliation": reconciliation,
            "trial_count": len(trials),
            "success_count": success_count,
            "failed_trial_ids": [
                record["trial_id"] for record in trials if record["success"] is False
            ],
            "gate": {
                "required_successes": required,
                "passed": bool(
                    reconciliation["complete"] and success_count >= required
                ),
            },
            "cli_config": {"smoke": config.smoke},
        }
        summary["summary_id"] = _content_id(_pair_summary_identity_payload(summary))
        atomic_write_json_no_clobber(summary_path, summary)
        validation = validate_pair_replay_summary(
            summary_path,
            plan_path=plan_path,
            train_manifest_path=train_path,
            validation_manifest_path=validation_path,
            project_root=config.project_root,
        )
        validation.raise_for_errors()
        return summary
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def validate_pair_replay_summary(
    path: Path,
    *,
    plan_path: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> ReplayValidationReport:
    summary_path = Path(path)
    errors: list[ReplayValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(ReplayValidationIssue(code, location, message))

    try:
        summary, initial_summary_sha = load_json_object_with_sha256(summary_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.summary", "/", f"cannot read strict replay JSON: {error}")
        return ReplayValidationReport(summary_path, tuple(errors), None, None)
    _validate_pair_summary_schema(summary, add)
    if errors:
        return ReplayValidationReport(
            summary_path,
            tuple(errors),
            summary,
            initial_summary_sha,
        )

    plan_report = validate_replay_plan(
        plan_path,
        train_manifest_path=train_manifest_path,
        validation_manifest_path=validation_manifest_path,
    )
    if not plan_report.valid or plan_report.plan is None or plan_report.sha256 is None:
        add("replay.plan", "/plan_path", "referenced replay plan is invalid")
        plan = None
    else:
        plan = plan_report.plan
        if summary["plan_id"] != plan["plan_id"]:
            add("replay.plan", "/plan_id", "summary plan identity differs")
        if summary["plan_sha256"] != plan_report.sha256:
            add("replay.plan", "/plan_sha256", "summary plan digest differs")
        if summary["formal"] is not plan["formal"]:
            add("replay.formal", "/formal", "summary and plan formal flags differ")

    try:
        trials, trials_digest, _, _ = load_jsonl_relative_with_sha256(
            summary_path.parent,
            summary["trials_path"],
        )
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.trials", "/trials_path", f"cannot read trial ledger: {error}")
        trials = []
        trials_digest = None
    if trials_digest is not None and trials_digest != summary["trials_sha256"]:
        add("replay.trials", "/trials_sha256", "trial ledger digest does not match")

    sources: dict[str, tuple[Path, dict[str, Any], str]] = {}
    if plan is not None:
        try:
            sources = _load_pair_sources_after_plan(
                Path(train_manifest_path).expanduser().absolute(),
                Path(validation_manifest_path).expanduser().absolute(),
                plan,
            )
        except (OSError, TypeError, ValueError) as error:
            add("replay.source", "/", str(error))
    lineage_valid = (
        summary["formal"] is False and not summary["lineage_receipt"]["provided"]
    )
    if summary["lineage_receipt"]["provided"] is True:
        try:
            expected_lineage = _validated_lineage_reference(
                summary_path.parent / summary["lineage_receipt"]["path"],
                replay_dir=summary_path.parent,
                project_root=project_root,
                train_manifest_path=train_manifest_path,
                validation_manifest_path=validation_manifest_path,
                require_formal=summary["formal"],
            )
            if summary["lineage_receipt"] != expected_lineage:
                add(
                    "replay.provenance",
                    "/lineage_receipt",
                    "lineage receipt reference differs from validated evidence",
                )
            else:
                lineage_valid = True
        except (OSError, ReplayProvenanceError, TypeError, ValueError) as error:
            add("replay.provenance", "/lineage_receipt", str(error))
    if summary["formal"] is True and sources:
        try:
            _require_formal_pair_replay_provenance(
                sources["train"][1], summary["replay_provenance"]
            )
        except (KeyError, ReplayProvenanceError, TypeError, ValueError) as error:
            add("replay.provenance", "/replay_provenance", str(error))
    _validate_pair_trials(trials, plan, sources, add)

    if plan is not None:
        expected_reconciliation = _identity_reconciliation(plan, trials)
        if summary["identity_reconciliation"] != expected_reconciliation:
            add(
                "replay.identity",
                "/identity_reconciliation",
                "identity reconciliation does not match plan and ledger",
            )
    successful = sum(
        isinstance(trial, dict) and trial.get("success") is True for trial in trials
    )
    failed_ids = [
        trial.get("trial_id")
        for trial in trials
        if isinstance(trial, dict) and trial.get("success") is False
    ]
    if summary["trial_count"] != len(trials):
        add("replay.trials", "/trial_count", "trial_count does not match ledger")
    if summary["success_count"] != successful:
        add("replay.trials", "/success_count", "success_count does not match ledger")
    if summary["failed_trial_ids"] != failed_ids:
        add(
            "replay.trials",
            "/failed_trial_ids",
            "failed trial identities do not match ledger",
        )
    required = _required_successes(len(trials)) if trials else FORMAL_REPLAY_REQUIRED
    expected_passed = bool(
        summary["identity_reconciliation"]["complete"]
        and successful >= required
        and lineage_valid
    )
    if summary["gate"]["required_successes"] != required:
        add("replay.gate", "/gate/required_successes", "replay threshold is wrong")
    if summary["gate"]["passed"] is not expected_passed:
        add("replay.gate", "/gate/passed", "replay gate arithmetic is wrong")

    expected_summary_id = _content_id(_pair_summary_identity_payload(summary))
    if summary["summary_id"] != expected_summary_id:
        add("replay.summary.identity", "/summary_id", "summary identity is wrong")
    try:
        _, final_summary_sha = load_json_object_with_sha256(summary_path)
        if final_summary_sha != initial_summary_sha:
            add("replay.summary", "/", "summary changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.summary", "/", f"cannot complete stable validation: {error}")
    return ReplayValidationReport(
        summary_path,
        tuple(errors),
        summary,
        initial_summary_sha,
    )


def _load_pair_sources_after_plan(
    train_path: Path,
    validation_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, tuple[Path, dict[str, Any], str]]:
    references = {item["split"]: item for item in plan["manifests"]}
    sources: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        manifest, digest = load_json_object_with_sha256(path)
        reference = references[split]
        if digest != reference["file_sha256"]:
            raise ValueError(f"{split} manifest changed after plan validation")
        if manifest.get("split") != split:
            raise ValueError(f"{split} manifest split changed")
        sources[split] = (path, manifest, digest)
    return sources


def _validate_pair_trials(
    trials: list[Any],
    plan: Mapping[str, Any] | None,
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
    add: Any,
) -> None:
    planned = plan["selected_trials"] if plan is not None else []
    runner_config_id = _pair_runner_config_id()
    for index, trial in enumerate(trials):
        location = f"/trials/{index}"
        if not isinstance(trial, dict) or set(trial) != _PAIR_TRIAL_KEYS:
            add("replay.trials.schema", location, "trial has the wrong fields")
            continue
        expected = planned[index] if index < len(planned) else None
        if trial["schema_version"] != PAIR_REPLAY_TRIAL_SCHEMA:
            add(
                "replay.trials.schema",
                f"{location}/schema_version",
                f"expected {PAIR_REPLAY_TRIAL_SCHEMA}",
            )
        if expected is not None:
            comparisons = {
                "trial_id": expected["trial_id"],
                "rank": expected["rank"],
                "plan_id": plan["plan_id"],
                "manifest_id": expected["manifest_id"],
                "split": expected["split"],
                "seed": expected["seed"],
                "source_relative_path": expected["source_relative_path"],
                "source_file_sha256": expected["source_file_sha256"],
                "source_num_steps": expected["source_num_steps"],
                "reset_seed": expected["seed"],
                "expected_steps": expected["source_num_steps"],
            }
            for field, expected_value in comparisons.items():
                if trial[field] != expected_value or type(trial[field]) is not type(
                    expected_value
                ):
                    add(
                        "replay.trials.identity",
                        f"{location}/{field}",
                        "trial field differs from plan",
                    )
        if trial["action_hash_algorithm"] != PAIR_ACTION_HASH_ALGORITHM:
            add(
                "replay.trials.schema",
                f"{location}/action_hash_algorithm",
                "action hash algorithm is not frozen",
            )
        if not _is_sha256(trial["action_dataset_sha256"]):
            add(
                "replay.trials.schema",
                f"{location}/action_dataset_sha256",
                "action dataset digest must be SHA-256",
            )
        if trial["runner_config_id"] != runner_config_id:
            add(
                "replay.trials.identity",
                f"{location}/runner_config_id",
                "runner contract identity differs",
            )
        if (
            type(trial["executed_steps"]) is not int
            or trial["executed_steps"] < 0
            or type(trial["expected_steps"]) is not int
            or trial["executed_steps"] > trial["expected_steps"]
        ):
            add(
                "replay.trials.schema",
                f"{location}/executed_steps",
                "executed_steps must be within the planned range",
            )
        if type(trial["success"]) is not bool:
            add("replay.trials.schema", f"{location}/success", "success must be bool")
        if type(trial["final_hold_steps"]) is not int or trial["final_hold_steps"] < 0:
            add(
                "replay.trials.schema",
                f"{location}/final_hold_steps",
                "final_hold_steps must be non-negative",
            )
        _validate_trial_outcome(trial, location, add)
        for field in ("started_at_utc", "finished_at_utc"):
            try:
                parsed = datetime.fromisoformat(trial[field])
                if parsed.tzinfo is None:
                    raise ValueError("timezone is missing")
            except (TypeError, ValueError):
                add(
                    "replay.trials.schema",
                    f"{location}/{field}",
                    f"{field} must be timezone-aware ISO-8601",
                )

        if expected is not None and expected["split"] in sources:
            source_path, source_manifest, _digest = sources[expected["split"]]
            try:
                with open_verified_episode(
                    source_path.parent,
                    expected["source_relative_path"],
                    expected_sha256=expected["source_file_sha256"],
                    max_num_steps=maximum_episode_steps(source_manifest["controller"]),
                ) as episode:
                    episode.validation.raise_for_errors()
                    source_action_digest = _action_dataset_sha256(
                        episode.handle["action"]
                    )
                if trial["action_dataset_sha256"] != source_action_digest:
                    add(
                        "replay.trials.source",
                        f"{location}/action_dataset_sha256",
                        "action dataset digest differs from source episode",
                    )
            except (OSError, RuntimeError, ValueError) as error:
                add(
                    "replay.trials.source",
                    location,
                    f"cannot revalidate source episode: {error}",
                )


def _validate_trial_outcome(trial: Mapping[str, Any], location: str, add: Any) -> None:
    success = trial["success"]
    failure_stage = trial["failure_stage"]
    exception_type = trial["exception_type"]
    exception_message = trial["exception_message"]
    if success is True:
        if (
            failure_stage is not None
            or exception_type is not None
            or exception_message is not None
            or trial["executed_steps"] != trial["expected_steps"]
        ):
            add(
                "replay.trials.outcome",
                location,
                "successful trial has conflicting failure evidence",
            )
        return
    if not isinstance(failure_stage, str) or not failure_stage:
        add(
            "replay.trials.outcome",
            f"{location}/failure_stage",
            "failed trial requires a failure stage",
        )
    if exception_type is None:
        if exception_message is not None:
            add(
                "replay.trials.outcome",
                f"{location}/exception_message",
                "exception message requires an exception type",
            )
    elif not isinstance(exception_type, str) or not exception_type:
        add(
            "replay.trials.outcome",
            f"{location}/exception_type",
            "exception type must be non-empty text or null",
        )
    elif not isinstance(exception_message, str):
        add(
            "replay.trials.outcome",
            f"{location}/exception_message",
            "exception trial requires a text message",
        )


def _validate_pair_summary_schema(summary: dict[str, Any], add: Any) -> None:
    if set(summary) != _PAIR_SUMMARY_KEYS:
        add("replay.summary.schema", "/", "paired replay summary has wrong fields")
        return
    if summary["schema_version"] != REPLAY_SCHEMA:
        add(
            "replay.summary.schema",
            "/schema_version",
            f"expected {REPLAY_SCHEMA}",
        )
    try:
        parsed = datetime.fromisoformat(summary["generated_at"])
        if parsed.tzinfo is None:
            raise ValueError("timezone is missing")
    except (TypeError, ValueError):
        add(
            "replay.summary.schema",
            "/generated_at",
            "generated_at must be timezone-aware ISO-8601",
        )
    for field in ("summary_id", "plan_id"):
        if not _is_content_id(summary[field]):
            add(
                "replay.summary.schema",
                f"/{field}",
                f"{field} must be a SHA-256 ID",
            )
    if type(summary["formal"]) is not bool:
        add("replay.summary.schema", "/formal", "formal must be boolean")
    if summary["plan_path"] != REPLAY_PLAN_FILENAME:
        add("replay.summary.schema", "/plan_path", f"expected {REPLAY_PLAN_FILENAME}")
    if summary["trials_path"] != REPLAY_TRIALS_FILENAME:
        add(
            "replay.summary.schema",
            "/trials_path",
            f"expected {REPLAY_TRIALS_FILENAME}",
        )
    for field in ("plan_sha256", "trials_sha256"):
        if not _is_sha256(summary[field]):
            add(
                "replay.summary.schema",
                f"/{field}",
                f"{field} must be SHA-256",
            )
    provenance = summary["replay_provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"git", "assets", "environment"}
        or any(not isinstance(value, dict) for value in provenance.values())
    ):
        add(
            "replay.summary.schema",
            "/replay_provenance",
            "replay provenance has wrong fields",
        )
    else:

        def provenance_add(_code: str, location: str, message: str) -> None:
            add("replay.provenance", f"/replay_provenance{location}", message)

        _validate_git(provenance["git"], provenance_add)
        _validate_assets(provenance["assets"], provenance_add)
        _validate_environment(provenance["environment"], provenance_add)
    lineage = summary["lineage_receipt"]
    if not isinstance(lineage, dict) or set(lineage) != {
        "provided",
        "path",
        "sha256",
        "receipt_id",
        "validated",
        "formal",
    }:
        add(
            "replay.summary.schema",
            "/lineage_receipt",
            "lineage receipt has wrong fields",
        )
    elif not (
        type(lineage["provided"]) is bool
        and type(lineage["validated"]) is bool
        and type(lineage["formal"]) is bool
        and (
            (
                lineage["provided"] is False
                and lineage["path"] is None
                and lineage["sha256"] is None
                and lineage["receipt_id"] is None
                and lineage["formal"] is False
                and lineage["validated"] is False
            )
            or (
                lineage["provided"] is True
                and lineage["path"] == LINEAGE_RECEIPT_FILENAME
                and _is_sha256(lineage["sha256"])
                and _is_content_id(lineage["receipt_id"])
                and lineage["validated"] is True
            )
        )
    ):
        add(
            "replay.summary.schema",
            "/lineage_receipt",
            "lineage receipt fields are inconsistent",
        )
    reconciliation = summary["identity_reconciliation"]
    expected_reconciliation_keys = {
        "planned",
        "observed",
        "missing_trial_ids",
        "unexpected_trial_ids",
        "duplicate_trial_ids",
        "complete",
    }
    if (
        not isinstance(reconciliation, dict)
        or set(reconciliation) != expected_reconciliation_keys
    ):
        add(
            "replay.summary.schema",
            "/identity_reconciliation",
            "identity reconciliation has wrong fields",
        )
        return
    for field in ("planned", "observed"):
        if type(reconciliation[field]) is not int or reconciliation[field] < 0:
            add(
                "replay.summary.schema",
                f"/identity_reconciliation/{field}",
                f"{field} must be non-negative",
            )
    for field in ("missing_trial_ids", "unexpected_trial_ids", "duplicate_trial_ids"):
        if not isinstance(reconciliation[field], list) or any(
            not _is_content_id(value) for value in reconciliation[field]
        ):
            add(
                "replay.summary.schema",
                f"/identity_reconciliation/{field}",
                f"{field} must contain SHA-256 IDs",
            )
    if type(reconciliation["complete"]) is not bool:
        add(
            "replay.summary.schema",
            "/identity_reconciliation/complete",
            "complete must be boolean",
        )
    for field in ("trial_count", "success_count"):
        if type(summary[field]) is not int or summary[field] < 0:
            add(
                "replay.summary.schema",
                f"/{field}",
                f"{field} must be non-negative",
            )
    if not isinstance(summary["failed_trial_ids"], list) or any(
        not _is_content_id(value) for value in summary["failed_trial_ids"]
    ):
        add(
            "replay.summary.schema",
            "/failed_trial_ids",
            "failed trial identities must be SHA-256 IDs",
        )
    gate = summary["gate"]
    if not isinstance(gate, dict) or set(gate) != {"required_successes", "passed"}:
        add("replay.summary.schema", "/gate", "gate has wrong fields")
    elif not (
        type(gate["required_successes"]) is int
        and gate["required_successes"] > 0
        and type(gate["passed"]) is bool
    ):
        add("replay.summary.schema", "/gate", "gate fields have wrong types")
    cli = summary["cli_config"]
    if (
        not isinstance(cli, dict)
        or set(cli) != {"smoke"}
        or type(cli["smoke"]) is not bool
    ):
        add("replay.summary.schema", "/cli_config", "CLI config has wrong fields")
    elif summary["formal"] is cli["smoke"]:
        add("replay.formal", "/formal", "formal and smoke flags disagree")
    if (
        summary["formal"] is True
        and isinstance(lineage, dict)
        and (lineage.get("validated") is not True or lineage.get("formal") is not True)
    ):
        add(
            "replay.provenance",
            "/lineage_receipt",
            "formal replay requires validated lineage",
        )


def _pair_summary_identity_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in sorted(_PAIR_SUMMARY_KEYS - {"summary_id", "generated_at"})
    }


def _identity_reconciliation(
    plan: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned_ids = [trial["trial_id"] for trial in plan["selected_trials"]]
    observed_ids = [trial.get("trial_id") for trial in trials]
    duplicate_ids = sorted(
        {
            trial_id
            for trial_id in observed_ids
            if isinstance(trial_id, str) and observed_ids.count(trial_id) > 1
        }
    )
    missing = [trial_id for trial_id in planned_ids if trial_id not in observed_ids]
    unexpected = [
        trial_id
        for trial_id in observed_ids
        if isinstance(trial_id, str) and trial_id not in planned_ids
    ]
    complete = bool(
        not missing
        and not unexpected
        and not duplicate_ids
        and observed_ids == planned_ids
    )
    return {
        "planned": len(planned_ids),
        "observed": len(observed_ids),
        "missing_trial_ids": missing,
        "unexpected_trial_ids": unexpected,
        "duplicate_trial_ids": duplicate_ids,
        "complete": complete,
    }


def _pair_runner_config_id() -> str:
    return _content_id(
        {
            "schema_version": "m2-replay-runner-config.v1",
            "reset": "numpy.random.default_rng(trial.seed)",
            "actions": "execute_all_source_actions_once_in_index_order",
            "action_dataset_hash": PAIR_ACTION_HASH_ALGORITHM,
            "success": "env.success_after_final_action",
            "failure_stage": "final_placement_or_exception",
        }
    )


def _action_dataset_sha256(dataset: Any) -> str:
    if getattr(dataset, "ndim", None) != 2 or tuple(dataset.shape[1:]) != (8,):
        raise ValueError("action dataset must have shape [T,8]")
    digest = hashlib.sha256()
    for start in range(0, int(dataset.shape[0]), 256):
        stop = min(start + 256, int(dataset.shape[0]))
        block = np.ascontiguousarray(dataset[start:stop], dtype="<f4")
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _placement_hold_steps(env: Any) -> int:
    status_fn = getattr(env, "placement_status", None)
    if not callable(status_fn):
        return 0
    status = status_fn()
    if not isinstance(status, Mapping):
        raise ValueError("placement_status must return a mapping")
    hold_steps = status.get("hold_steps")
    if type(hold_steps) is not int or hold_steps < 0:
        raise ValueError("placement_status hold_steps must be non-negative")
    return hold_steps


def _utc_now(now_fn: Callable[[], datetime] | None) -> datetime:
    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def replay_collection(
    config: ReplayConfig,
    *,
    env_factory: Callable[[], Any] = PickPlace,
    git_state_fn: Callable[[Path], dict[str, Any]] = _git_state,
    asset_provenance_fn: Callable[..., Any] = collect_asset_provenance,
    environment_fingerprint_fn: Callable[[Any], dict[str, Any]] = (
        compiled_model_fingerprint
    ),
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    _validate_replay_config(config)
    manifest_path = Path(config.manifest_path).expanduser().absolute()
    output_dir = Path(config.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"replay output directory already exists: {output_dir}")

    initial_validation = validate_collection_manifest(manifest_path)
    initial_validation.raise_for_errors()
    manifest = initial_validation.manifest
    if manifest is None:
        raise RuntimeError("validated manifest payload is unavailable")
    formal_requested = bool(
        config.smoke is False
        and config.count == FORMAL_REPLAY_COUNT
        and config.selection_seed == FORMAL_SELECTION_SEED
    )
    if formal_requested and manifest["formal"] is not True:
        raise ValueError("formal replay requires a formal collection manifest")
    source_digest = initial_validation.sha256
    if source_digest is None:
        raise RuntimeError("validated manifest digest is unavailable")
    selected = select_replay_entries(
        manifest,
        count=config.count,
        selection_seed=config.selection_seed,
    )

    env = env_factory()
    try:
        replay_provenance = _capture_replay_provenance(
            config,
            env,
            git_state_fn=git_state_fn,
            asset_provenance_fn=asset_provenance_fn,
            environment_fingerprint_fn=environment_fingerprint_fn,
        )
        if formal_requested:
            _require_formal_replay_provenance(manifest, replay_provenance)

        output_dir.mkdir(parents=True, exist_ok=False)
        trials_path = output_dir / REPLAY_TRIALS_FILENAME
        trials_state = initialize_jsonl_no_clobber(trials_path)
        trials: list[dict[str, Any]] = []
        run_root = manifest_path.parent
        maximum_steps = maximum_episode_steps(manifest["controller"])
        for trial_index, entry in enumerate(selected):
            seed = entry["seed"]
            try:
                with open_verified_episode(
                    run_root,
                    entry["path"],
                    expected_sha256=entry["sha256"],
                    max_num_steps=maximum_steps,
                ) as episode:
                    episode.validation.raise_for_errors()
                    metadata = episode.metadata
                    if (
                        metadata is None
                        or metadata.seed != seed
                        or metadata.success is not True
                        or metadata.num_steps != entry["num_steps"]
                    ):
                        raise RuntimeError(
                            "eligible replay entry metadata is inconsistent"
                        )
                    actions = episode.handle["action"]
                    num_actions = int(actions.shape[0])
                    env.reset(np.random.default_rng(seed))
                    for action_index in range(num_actions):
                        action = np.asarray(
                            actions[action_index],
                            dtype=np.float64,
                        )
                        env.step(action)
                    success = bool(env.success())
                record = {
                    "trial_index": trial_index,
                    "seed": seed,
                    "path": entry["path"],
                    "num_actions": num_actions,
                    "status": "success" if success else "failure",
                    "success": success,
                    "error": None,
                }
                trials_state = append_jsonl_fsync(
                    trials_path,
                    record,
                    expected_snapshot=trials_state,
                )
                trials.append(record)
            except BaseException as error:
                record = {
                    "trial_index": trial_index,
                    "seed": seed,
                    "path": entry.get("path"),
                    "num_actions": None,
                    "status": "exception",
                    "success": None,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error).replace("\x00", "\\0")[:2000],
                    },
                }
                trials_state = append_jsonl_fsync(
                    trials_path,
                    record,
                    expected_snapshot=trials_state,
                )
                raise

        final_validation = validate_collection_manifest(manifest_path)
        final_validation.raise_for_errors()
        if final_validation.sha256 != source_digest:
            raise RuntimeError("collection manifest changed during replay")
        if len({trial["seed"] for trial in trials}) != len(trials):
            raise RuntimeError("replay selected a duplicate seed")

        success_count = sum(trial["success"] is True for trial in trials)
        required = _required_successes(config.count)
        timestamp = (now_fn or (lambda: datetime.now(UTC)))()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        summary: dict[str, Any] = {
            "schema_version": REPLAY_SCHEMA,
            "generated_at": timestamp.astimezone(UTC).isoformat(),
            "formal": bool(formal_requested and manifest["formal"] is True),
            "source_manifest_sha256": source_digest,
            "source_split": manifest["split"],
            "source_manifest_formal": manifest["formal"],
            "replay_provenance": replay_provenance,
            "selection": {
                "algorithm": SELECTION_ALGORITHM,
                "seed": config.selection_seed,
                "count": config.count,
                "without_replacement": True,
            },
            "selected_seeds": [entry["seed"] for entry in selected],
            "trial_count": len(trials),
            "success_count": success_count,
            "trials_path": REPLAY_TRIALS_FILENAME,
            "trials_sha256": manifest_sha256(trials_path),
            "gate": {
                "required_successes": required,
                "passed": success_count >= required,
            },
            "cli_config": {
                "selection_seed": config.selection_seed,
                "count": config.count,
                "smoke": config.smoke,
            },
        }
        atomic_write_json_no_clobber(output_dir / REPLAY_SUMMARY_FILENAME, summary)
        validation = validate_replay_summary(
            output_dir / REPLAY_SUMMARY_FILENAME,
            collection_manifest_path=manifest_path,
        )
        validation.raise_for_errors()
        return summary
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _capture_replay_provenance(
    config: ReplayConfig,
    env: Any,
    *,
    git_state_fn: Callable[[Path], dict[str, Any]],
    asset_provenance_fn: Callable[..., Any],
    environment_fingerprint_fn: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    if getattr(getattr(env, "cfg", None), "debug_viz", None) is not False:
        raise ReplayProvenanceError("replay requires env.cfg.debug_viz exactly false")
    git = git_state_fn(Path(config.project_root))
    provenance = asset_provenance_fn(
        Path(config.project_root),
        runtime_asset_root=env_scene.MENAGERIE,
    )
    environment = environment_payload(env, environment_fingerprint_fn(env))
    return {
        "git": git,
        "assets": assets_payload(provenance),
        "environment": environment,
    }


def _require_formal_replay_provenance(
    source_manifest: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    source_git = source_manifest["git"]
    current_git = current["git"]
    if not (
        current_git.get("provenance_complete") is True
        and current_git.get("source_provenance_clean") is True
        and current_git.get("tracked_worktree_clean") is True
    ):
        raise ReplayProvenanceError(
            "formal replay requires complete clean current source provenance"
        )
    current_commit = current_git.get("commit")
    if (
        not isinstance(current_commit, str)
        or _COMMIT_RE.fullmatch(current_commit) is None
        or current_commit != source_git.get("commit")
    ):
        raise ReplayProvenanceError(
            "formal replay current Git commit differs from the source collection"
        )
    if current["assets"] != source_manifest["assets"]:
        raise ReplayProvenanceError(
            "formal replay asset provenance differs from the source collection"
        )
    if current["environment"] != source_manifest["environment"]:
        raise ReplayProvenanceError(
            "formal replay environment fingerprint differs from the source collection"
        )


def _require_formal_pair_replay_provenance(
    source_manifest: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    current_git = current["git"]
    if not (
        current_git.get("provenance_complete") is True
        and current_git.get("source_provenance_clean") is True
        and current_git.get("tracked_worktree_clean") is True
    ):
        raise ReplayProvenanceError(
            "formal paired replay requires complete clean current source provenance"
        )
    current_commit = current_git.get("commit")
    if (
        not isinstance(current_commit, str)
        or _COMMIT_RE.fullmatch(current_commit) is None
    ):
        raise ReplayProvenanceError("formal paired replay Git commit is invalid")
    if current["assets"] != source_manifest["assets"]:
        raise ReplayProvenanceError(
            "formal paired replay asset provenance differs from the source collection"
        )
    if current["environment"] != source_manifest["environment"]:
        raise ReplayProvenanceError(
            "formal paired replay environment differs from the source collection"
        )


def validate_replay_summary(
    path: Path,
    *,
    collection_manifest_path: Path | None = None,
) -> ReplayValidationReport:
    summary_path = Path(path)
    errors: list[ReplayValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(ReplayValidationIssue(code, location, message))

    try:
        summary, initial_summary_sha = load_json_object_with_sha256(summary_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.summary", "/", f"cannot read strict replay JSON: {error}")
        return ReplayValidationReport(summary_path, tuple(errors), None, None)

    _validate_summary_schema(summary, add)
    if errors:
        return ReplayValidationReport(
            summary_path,
            tuple(errors),
            summary,
            initial_summary_sha,
        )

    source_manifest: dict[str, Any] | None = None
    source_digest: str | None = None
    if collection_manifest_path is not None:
        source_path = Path(collection_manifest_path)
        source_report = validate_collection_manifest(source_path)
        if not source_report.valid or source_report.manifest is None:
            add(
                "replay.source",
                "/source_manifest_sha256",
                "source collection manifest is not valid",
            )
        else:
            source_manifest = source_report.manifest
            source_digest = source_report.sha256
            if source_digest is None:
                add(
                    "replay.source",
                    "/source_manifest_sha256",
                    "source collection digest is unavailable",
                )
                source_digest = ""
            if summary["source_manifest_sha256"] != source_digest:
                add(
                    "replay.source",
                    "/source_manifest_sha256",
                    "source collection digest does not match",
                )
            if summary["source_split"] != source_manifest["split"]:
                add("replay.source", "/source_split", "source split does not match")
            if summary["source_manifest_formal"] is not source_manifest["formal"]:
                add(
                    "replay.source",
                    "/source_manifest_formal",
                    "source formal flag does not match",
                )

    try:
        trials, trials_digest, _, _ = load_jsonl_relative_with_sha256(
            summary_path.parent,
            summary["trials_path"],
        )
    except (OSError, UnicodeError, ValueError) as error:
        add("replay.trials", "/trials_path", f"cannot read trial ledger: {error}")
        trials = []
        trials_digest = None
    if trials_digest is not None and trials_digest != summary["trials_sha256"]:
        add("replay.trials", "/trials_sha256", "trial ledger digest does not match")

    selected_seeds = summary["selected_seeds"]
    if len(trials) != summary["trial_count"]:
        add("replay.trials", "/trial_count", "trial_count does not match ledger")
    successful = 0
    for index, trial in enumerate(trials):
        location = f"/trials/{index}"
        if not isinstance(trial, dict) or set(trial) != _TRIAL_KEYS:
            add("replay.trials", location, "trial record has the wrong fields")
            continue
        if type(trial["trial_index"]) is not int or trial["trial_index"] != index:
            add(
                "replay.trials",
                f"{location}/trial_index",
                "trial index is not contiguous",
            )
        expected_seed = selected_seeds[index] if index < len(selected_seeds) else None
        if type(trial["seed"]) is not int or trial["seed"] != expected_seed:
            add(
                "replay.trials",
                f"{location}/seed",
                "trial seed does not match selection",
            )
        if not isinstance(trial["path"], str):
            add("replay.trials", f"{location}/path", "trial path must be text")
        else:
            try:
                validate_relative_path(trial["path"])
            except ValueError as error:
                add("replay.trials", f"{location}/path", str(error))
        if type(trial["num_actions"]) is not int or trial["num_actions"] <= 0:
            add(
                "replay.trials",
                f"{location}/num_actions",
                "num_actions must be positive",
            )
        if type(trial["success"]) is not bool:
            add("replay.trials", f"{location}/success", "success must be boolean")
        else:
            successful += int(trial["success"])
            expected_status = "success" if trial["success"] else "failure"
            if trial["status"] != expected_status:
                add(
                    "replay.trials", f"{location}/status", "status and success disagree"
                )
        if trial["error"] is not None:
            add(
                "replay.trials",
                f"{location}/error",
                "final trial cannot contain an error",
            )

    if len(set(selected_seeds)) != len(selected_seeds):
        add("replay.selection", "/selected_seeds", "selected seeds are duplicated")
    if summary["success_count"] != successful:
        add("replay.trials", "/success_count", "success_count does not match ledger")
    required = _required_successes(summary["trial_count"])
    if summary["gate"]["required_successes"] != required:
        add("replay.gate", "/gate/required_successes", "wrong replay threshold")
    if summary["gate"]["passed"] is not (successful >= required):
        add("replay.gate", "/gate/passed", "gate arithmetic does not match trials")

    if source_manifest is not None:
        try:
            rebuilt = select_replay_entries(
                source_manifest,
                count=summary["selection"]["count"],
                selection_seed=summary["selection"]["seed"],
            )
        except ValueError as error:
            add("replay.selection", "/selection", str(error))
            rebuilt = []
        rebuilt_seeds = [entry["seed"] for entry in rebuilt]
        if selected_seeds != rebuilt_seeds:
            add(
                "replay.selection",
                "/selected_seeds",
                "selected seeds do not match deterministic reconstruction",
            )
        for index, (trial, entry) in enumerate(zip(trials, rebuilt, strict=False)):
            if not isinstance(trial, dict):
                continue
            if trial.get("path") != entry["path"]:
                add(
                    "replay.trials",
                    f"/trials/{index}/path",
                    "trial path differs from source",
                )
            if trial.get("num_actions") != entry["num_steps"]:
                add(
                    "replay.trials",
                    f"/trials/{index}/num_actions",
                    "trial action count differs from source",
                )
        if summary["formal"] is True:
            try:
                _require_formal_replay_provenance(
                    source_manifest,
                    summary["replay_provenance"],
                )
            except ReplayProvenanceError as error:
                add("replay.provenance", "/replay_provenance", str(error))

    try:
        _, final_summary_sha = load_json_object_with_sha256(summary_path)
        if final_summary_sha != initial_summary_sha:
            add("replay.summary", "/", "summary changed during validation")
        if collection_manifest_path is not None and source_digest is not None:
            final_source_report = validate_collection_manifest(collection_manifest_path)
            if (
                not final_source_report.valid
                or final_source_report.sha256 != source_digest
            ):
                add("replay.source", "/", "source collection changed during validation")
    except (OSError, ValueError) as error:
        add("replay.summary", "/", f"cannot complete stable validation: {error}")
    return ReplayValidationReport(
        summary_path,
        tuple(errors),
        summary,
        initial_summary_sha,
    )


def _validate_summary_schema(summary: dict[str, Any], add: Any) -> None:
    if set(summary) != _SUMMARY_KEYS:
        add("replay.schema", "/", "replay summary has the wrong fields")
        return
    if summary["schema_version"] != REPLAY_SCHEMA:
        add("replay.schema", "/schema_version", f"expected {REPLAY_SCHEMA}")
    try:
        parsed = datetime.fromisoformat(summary["generated_at"])
        if parsed.tzinfo is None:
            raise ValueError("timezone is missing")
    except (TypeError, ValueError):
        add(
            "replay.schema",
            "/generated_at",
            "generated_at must be timezone-aware ISO-8601",
        )
    for name in ("formal", "source_manifest_formal"):
        if type(summary[name]) is not bool:
            add("replay.schema", f"/{name}", f"{name} must be boolean")
    if (
        not isinstance(summary["source_manifest_sha256"], str)
        or _SHA256_RE.fullmatch(summary["source_manifest_sha256"]) is None
    ):
        add("replay.schema", "/source_manifest_sha256", "source digest must be SHA-256")
    if not isinstance(summary["source_split"], str) or summary["source_split"] not in {
        "train",
        "validation",
    }:
        add("replay.schema", "/source_split", "source split is invalid")
    provenance = summary["replay_provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "git",
            "assets",
            "environment",
        }
        or any(not isinstance(value, dict) for value in provenance.values())
    ):
        add(
            "replay.schema",
            "/replay_provenance",
            "replay provenance has the wrong shape",
        )
    else:

        def add_provenance_issue(
            _code: str,
            location: str,
            message: str,
        ) -> None:
            add(
                "replay.provenance",
                f"/replay_provenance{location}",
                message,
            )

        _validate_git(provenance["git"], add_provenance_issue)
        _validate_assets(provenance["assets"], add_provenance_issue)
        _validate_environment(provenance["environment"], add_provenance_issue)

    selection = summary["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "algorithm",
        "seed",
        "count",
        "without_replacement",
    }:
        add("replay.schema", "/selection", "selection has the wrong fields")
        return
    if selection["algorithm"] != SELECTION_ALGORITHM:
        add(
            "replay.selection",
            "/selection/algorithm",
            "selection algorithm is not frozen",
        )
    if type(selection["seed"]) is not int or selection["seed"] < 0:
        add("replay.schema", "/selection/seed", "selection seed must be non-negative")
    if (
        type(selection["count"]) is not int
        or not 1 <= selection["count"] <= FORMAL_REPLAY_COUNT
    ):
        add("replay.schema", "/selection/count", "selection count must be in 1..20")
    if selection["without_replacement"] is not True:
        add(
            "replay.selection",
            "/selection/without_replacement",
            "selection must be without replacement",
        )

    seeds = summary["selected_seeds"]
    if not isinstance(seeds, list) or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        add(
            "replay.schema",
            "/selected_seeds",
            "selected seeds must be non-negative integers",
        )
        return
    valid_counts = True
    for name in ("trial_count", "success_count"):
        if type(summary[name]) is not int or summary[name] < 0:
            add("replay.schema", f"/{name}", f"{name} must be a non-negative integer")
            valid_counts = False
    if valid_counts and (
        summary["trial_count"] != selection["count"] or len(seeds) != selection["count"]
    ):
        add(
            "replay.selection",
            "/selected_seeds",
            "selection, seed, and trial counts disagree",
        )
    if valid_counts and summary["success_count"] > summary["trial_count"]:
        add("replay.schema", "/success_count", "success_count exceeds trial_count")
    if summary["trials_path"] != REPLAY_TRIALS_FILENAME:
        add("replay.schema", "/trials_path", f"expected {REPLAY_TRIALS_FILENAME}")
    if (
        not isinstance(summary["trials_sha256"], str)
        or _SHA256_RE.fullmatch(summary["trials_sha256"]) is None
    ):
        add("replay.schema", "/trials_sha256", "trial digest must be SHA-256")

    gate = summary["gate"]
    if not isinstance(gate, dict) or set(gate) != {"required_successes", "passed"}:
        add("replay.schema", "/gate", "gate has the wrong fields")
        return
    if type(gate["required_successes"]) is not int or gate["required_successes"] <= 0:
        add("replay.schema", "/gate/required_successes", "threshold must be positive")
    if type(gate["passed"]) is not bool:
        add("replay.schema", "/gate/passed", "passed must be boolean")

    cli = summary["cli_config"]
    if not isinstance(cli, dict) or set(cli) != {"selection_seed", "count", "smoke"}:
        add("replay.schema", "/cli_config", "CLI config has the wrong fields")
        return
    if (
        cli["selection_seed"] != selection["seed"]
        or cli["count"] != selection["count"]
        or type(cli["smoke"]) is not bool
    ):
        add("replay.schema", "/cli_config", "CLI config and selection disagree")
    if cli["smoke"] is False and (
        selection["count"] != FORMAL_REPLAY_COUNT
        or selection["seed"] != FORMAL_SELECTION_SEED
        or summary["source_manifest_formal"] is not True
    ):
        add(
            "replay.formal",
            "/cli_config",
            "non-smoke replay requires a formal source and the frozen count/seed",
        )
    expected_formal = bool(
        summary["source_manifest_formal"] is True
        and cli["smoke"] is False
        and selection["count"] == FORMAL_REPLAY_COUNT
        and selection["seed"] == FORMAL_SELECTION_SEED
    )
    if type(summary["formal"]) is bool and summary["formal"] is not expected_formal:
        add(
            "replay.formal",
            "/formal",
            "formal flag is inconsistent with frozen replay protocol",
        )


def _required_successes(count: int) -> int:
    return (
        FORMAL_REPLAY_REQUIRED
        if count == FORMAL_REPLAY_COUNT
        else math.ceil(0.9 * count)
    )


def _validate_replay_config(config: ReplayConfig) -> None:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if type(config.count) is not int or config.count <= 0:
        raise ValueError("count must be a positive integer")
    if config.count > FORMAL_REPLAY_COUNT:
        raise ValueError("replay count cannot exceed 20")
    if config.count != FORMAL_REPLAY_COUNT and not config.smoke:
        raise ValueError("a replay count smaller than 20 requires --smoke")
    if type(config.selection_seed) is not int or config.selection_seed < 0:
        raise ValueError("selection_seed must be a non-negative integer")
    if config.selection_seed != FORMAL_SELECTION_SEED and not config.smoke:
        raise ValueError("a noncanonical selection seed requires --smoke")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay stored M2 expert actions with their original reset seeds."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pair-mode", choices=("plan", "run", "validate"))
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--lineage-receipt", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--selection-seed", type=int, default=FORMAL_SELECTION_SEED)
    parser.add_argument("--count", type=int, default=FORMAL_REPLAY_COUNT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.pair_mode is not None:
            if (
                args.train_manifest is None
                or args.validation_manifest is None
                or args.replay_dir is None
            ):
                parser.error(
                    "paired replay requires --train-manifest, "
                    "--validation-manifest, and --replay-dir"
                )
            if args.pair_mode == "plan":
                plan = create_replay_plan(
                    ReplayPlanConfig(
                        train_manifest_path=args.train_manifest,
                        validation_manifest_path=args.validation_manifest,
                        output_dir=args.replay_dir,
                        selection_seed=args.selection_seed,
                        count=args.count,
                        smoke=args.smoke,
                    )
                )
                print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
                return 0
            if args.pair_mode == "run":
                summary = replay_manifest_pair(
                    PairReplayConfig(
                        train_manifest_path=args.train_manifest,
                        validation_manifest_path=args.validation_manifest,
                        replay_dir=args.replay_dir,
                        smoke=args.smoke,
                        lineage_receipt_path=args.lineage_receipt,
                        project_root=args.project_root,
                    )
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 0 if summary["gate"]["passed"] else 1
            report = validate_pair_replay_summary(
                args.replay_dir / REPLAY_SUMMARY_FILENAME,
                plan_path=args.replay_dir / REPLAY_PLAN_FILENAME,
                train_manifest_path=args.train_manifest,
                validation_manifest_path=args.validation_manifest,
                project_root=args.project_root,
            )
            result = {
                "valid": report.valid,
                "errors": [str(error) for error in report.errors],
                "gate_passed": bool(
                    report.valid
                    and report.summary is not None
                    and report.summary["gate"]["passed"] is True
                ),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["gate_passed"] else 2
        if args.manifest is None or args.output_dir is None:
            parser.error("legacy replay requires --manifest and --output-dir")
        summary = replay_collection(
            ReplayConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                selection_seed=args.selection_seed,
                count=args.count,
                smoke=args.smoke,
            )
        )
    except (FileExistsError, ReplayProvenanceError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
