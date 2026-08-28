"""Executable M1/M2 lineage equality receipts for formal replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import (
    atomic_write_json_no_clobber,
    canonical_json_bytes,
    load_json_object_with_sha256,
    sha256_bytes,
    stable_read_relative_file,
    validate_collection_manifest,
    validate_manifest_pair,
    validate_relative_path,
)

LINEAGE_SCHEMA = "m1-m2-lineage-revalidation.v1"
SOURCE_SCOPE_SCHEMA = "m1-m2-collection-runtime-scope.v1"
PLACEMENT_PREDICATE_VERSION = "obb-full-containment-hold-v1"
FORMAL_ACCEPTED_COMMIT = "0d92035fcd345b21c8b6f8d8d6a626915a645ea9"
FORMAL_BASELINE_COMMIT = "90ffb5225b72b6bee3fc16f9c3622833cfd30808"
FORMAL_RUNTIME_PATHS = (
    "env.sh",
    "pyproject.toml",
    "src/data/__init__.py",
    "src/data/collection.py",
    "src/data/hdf5.py",
    "src/data/manifest.py",
    "src/env/__init__.py",
    "src/env/asset_provenance.py",
    "src/env/pick_place.py",
    "src/env/scene.py",
    "src/expert/__init__.py",
    "src/expert/evaluate.py",
    "src/expert/scripted.py",
    "uv.lock",
)
FORMAL_DOCUMENTATION_ALLOWLIST = ("README.md", "docs/DATA_SCHEMA.md")
FORMAL_HISTORICAL_EVIDENCE = (
    "runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/summary.json",
    "runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/episodes.jsonl",
)
HISTORICAL_OLD_PREDICATE_COMMIT = "a97977ad5267562f97b5c1a43623e0a2ce23d79c"

_GIT = "/usr/bin/git"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TOP_KEYS = {
    "schema_version",
    "receipt_id",
    "status",
    "formal",
    "generated_at",
    "creation_head",
    "accepted_commit",
    "baseline_commit",
    "source_scope",
    "runtime_equality",
    "repository_differences",
    "manifests",
    "controller",
    "environment",
    "assets",
    "placement_predicate",
    "writer_contract",
    "historical_evidence_denylist",
}


@dataclass(frozen=True)
class LineageRevalidationConfig:
    repository_root: Path
    train_manifest_path: Path
    validation_manifest_path: Path
    output_path: Path
    accepted_commit: str = FORMAL_ACCEPTED_COMMIT
    baseline_commit: str = FORMAL_BASELINE_COMMIT
    runtime_paths: tuple[str, ...] = FORMAL_RUNTIME_PATHS
    documentation_allowlist: tuple[str, ...] = FORMAL_DOCUMENTATION_ALLOWLIST
    smoke: bool = False


@dataclass(frozen=True)
class LineageValidationIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class LineageValidationReport:
    path: Path
    errors: tuple[LineageValidationIssue, ...]
    receipt: dict[str, Any] | None = None
    sha256: str | None = None
    passed: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(f"invalid lineage receipt:\n{self.format_errors()}")


def create_lineage_revalidation_receipt(
    config: LineageRevalidationConfig,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    normalized = _validate_config(config)
    payload = _build_payload(normalized, now_fn=now_fn)
    receipt = {**payload, "receipt_id": _content_id(payload)}
    atomic_write_json_no_clobber(normalized.output_path, receipt)
    report = validate_lineage_revalidation_receipt(
        normalized.output_path,
        repository_root=normalized.repository_root,
        train_manifest_path=normalized.train_manifest_path,
        validation_manifest_path=normalized.validation_manifest_path,
    )
    report.raise_for_errors()
    return receipt


def validate_lineage_revalidation_receipt(
    path: Path,
    *,
    repository_root: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
) -> LineageValidationReport:
    receipt_path = Path(path)
    errors: list[LineageValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(LineageValidationIssue(code, location, message))

    try:
        receipt, initial_sha = load_json_object_with_sha256(receipt_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("lineage.file", "/", str(error))
        return LineageValidationReport(receipt_path, tuple(errors))
    _validate_receipt_schema(receipt, add)
    if errors:
        return LineageValidationReport(
            receipt_path, tuple(errors), receipt, initial_sha
        )
    identity = {key: receipt[key] for key in _TOP_KEYS if key != "receipt_id"}
    if receipt["receipt_id"] != _content_id(identity):
        add("lineage.identity", "/receipt_id", "receipt identity is wrong")

    root = Path(repository_root).expanduser().absolute()
    try:
        runtime_paths = tuple(receipt["source_scope"]["runtime_paths"])
        documentation_allowlist = tuple(
            receipt["repository_differences"]["documentation_allowlist"]
        )
        if receipt["formal"] is True:
            if receipt["accepted_commit"] != FORMAL_ACCEPTED_COMMIT:
                add(
                    "lineage.formal",
                    "/accepted_commit",
                    "formal accepted commit differs",
                )
            if receipt["baseline_commit"] != FORMAL_BASELINE_COMMIT:
                add(
                    "lineage.formal",
                    "/baseline_commit",
                    "formal baseline commit differs",
                )
            if runtime_paths != FORMAL_RUNTIME_PATHS:
                add(
                    "lineage.formal",
                    "/source_scope/runtime_paths",
                    "formal runtime scope differs",
                )
            if documentation_allowlist != FORMAL_DOCUMENTATION_ALLOWLIST:
                add(
                    "lineage.formal",
                    "/repository_differences/documentation_allowlist",
                    "formal documentation allowlist differs",
                )
        accepted_entries = _commit_entries(
            root, receipt["accepted_commit"], runtime_paths
        )
        baseline_entries = _commit_entries(
            root, receipt["baseline_commit"], runtime_paths
        )
        current_entries = _worktree_entries(root, runtime_paths)
        source_scope = receipt["source_scope"]
        expected_scope = _source_scope(
            runtime_paths,
            accepted_entries,
            baseline_entries,
            current_entries,
        )
        if source_scope != expected_scope:
            add(
                "lineage.source",
                "/source_scope",
                "runtime source scope differs from commits or worktree",
            )
        expected_equality = _runtime_equality(
            accepted_entries, baseline_entries, current_entries
        )
        if receipt["runtime_equality"] != expected_equality:
            add(
                "lineage.source",
                "/runtime_equality",
                "runtime equality result differs",
            )
        expected_differences = _repository_differences(
            root,
            receipt["accepted_commit"],
            receipt["baseline_commit"],
            documentation_allowlist,
        )
        if receipt["repository_differences"] != expected_differences:
            add(
                "lineage.diff",
                "/repository_differences",
                "repository difference audit differs",
            )
        if not expected_equality["equal"]:
            add("lineage.source", "/runtime_equality", "runtime source differs")
        if not expected_differences["allowlisted_only"]:
            add("lineage.diff", "/repository_differences", "non-doc change found")
        if not _is_ancestor(root, receipt["baseline_commit"], _head(root)):
            add("lineage.git", "/creation_head", "baseline is not an ancestor")

        train, train_sha = load_json_object_with_sha256(Path(train_manifest_path))
        validation, validation_sha = load_json_object_with_sha256(
            Path(validation_manifest_path)
        )
        _validate_manifest_inputs(
            train,
            validation,
            accepted_commit=receipt["accepted_commit"],
            formal=receipt["formal"],
            train_path=Path(train_manifest_path),
            validation_path=Path(validation_manifest_path),
        )
        expected_facts = _manifest_facts(train, train_sha, validation, validation_sha)
        for name in ("manifests", "controller", "environment", "assets"):
            if receipt[name] != expected_facts[name]:
                add(
                    "lineage.manifest",
                    f"/{name}",
                    f"{name} differs from source manifests",
                )
        expected_predicate = _placement_predicate(expected_scope, train)
        if receipt["placement_predicate"] != expected_predicate:
            add(
                "lineage.predicate",
                "/placement_predicate",
                "strict predicate evidence differs",
            )
        expected_writer = _writer_contract(expected_scope)
        if receipt["writer_contract"] != expected_writer:
            add(
                "lineage.writer",
                "/writer_contract",
                "writer contract evidence differs",
            )
        expected_denylist = _historical_denylist(root, formal=receipt["formal"])
        if receipt["historical_evidence_denylist"] != expected_denylist:
            add(
                "lineage.historical",
                "/historical_evidence_denylist",
                "historical evidence denylist differs",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        add("lineage.source", "/", str(error))
    try:
        _, final_sha = load_json_object_with_sha256(receipt_path)
        if final_sha != initial_sha:
            add("lineage.file", "/", "receipt changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("lineage.file", "/", str(error))
    passed = not errors and receipt["status"] == "passed"
    return LineageValidationReport(
        receipt_path,
        tuple(errors),
        receipt,
        initial_sha,
        passed=passed,
    )


def _validate_config(config: LineageRevalidationConfig) -> LineageRevalidationConfig:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be boolean")
    if not config.smoke and (
        config.accepted_commit != FORMAL_ACCEPTED_COMMIT
        or config.baseline_commit != FORMAL_BASELINE_COMMIT
        or config.runtime_paths != FORMAL_RUNTIME_PATHS
        or config.documentation_allowlist != FORMAL_DOCUMENTATION_ALLOWLIST
    ):
        raise ValueError("formal lineage constants cannot be customized")
    _require_commit(config.accepted_commit, "accepted_commit")
    _require_commit(config.baseline_commit, "baseline_commit")
    runtime_paths = _validated_paths(config.runtime_paths, "runtime_paths")
    documentation = _validated_paths(
        config.documentation_allowlist, "documentation_allowlist"
    )
    if set(runtime_paths) & set(documentation):
        raise ValueError("runtime and documentation paths overlap")
    return LineageRevalidationConfig(
        repository_root=Path(config.repository_root).expanduser().absolute(),
        train_manifest_path=Path(config.train_manifest_path).expanduser().absolute(),
        validation_manifest_path=Path(config.validation_manifest_path)
        .expanduser()
        .absolute(),
        output_path=Path(config.output_path).expanduser().absolute(),
        accepted_commit=config.accepted_commit,
        baseline_commit=config.baseline_commit,
        runtime_paths=runtime_paths,
        documentation_allowlist=documentation,
        smoke=config.smoke,
    )


def _build_payload(
    config: LineageRevalidationConfig,
    *,
    now_fn: Callable[[], datetime] | None,
) -> dict[str, Any]:
    root = config.repository_root.resolve(strict=True)
    head = _head(root)
    if not _is_ancestor(root, config.accepted_commit, config.baseline_commit):
        raise ValueError("accepted lineage commit is not an ancestor of baseline")
    if not _is_ancestor(root, config.baseline_commit, head):
        raise ValueError("baseline lineage commit is not an ancestor of HEAD")
    accepted_entries = _commit_entries(
        root, config.accepted_commit, config.runtime_paths
    )
    baseline_entries = _commit_entries(
        root, config.baseline_commit, config.runtime_paths
    )
    current_entries = _worktree_entries(root, config.runtime_paths)
    equality = _runtime_equality(accepted_entries, baseline_entries, current_entries)
    if not equality["equal"]:
        raise ValueError("runtime source differs across accepted/baseline/worktree")
    differences = _repository_differences(
        root,
        config.accepted_commit,
        config.baseline_commit,
        config.documentation_allowlist,
    )
    if not differences["allowlisted_only"]:
        raise ValueError("repository has non-allowlisted lineage differences")
    train, train_sha = load_json_object_with_sha256(config.train_manifest_path)
    validation, validation_sha = load_json_object_with_sha256(
        config.validation_manifest_path
    )
    _validate_manifest_inputs(
        train,
        validation,
        accepted_commit=config.accepted_commit,
        formal=not config.smoke,
        train_path=config.train_manifest_path,
        validation_path=config.validation_manifest_path,
    )
    facts = _manifest_facts(train, train_sha, validation, validation_sha)
    source_scope = _source_scope(
        config.runtime_paths,
        accepted_entries,
        baseline_entries,
        current_entries,
    )
    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return {
        "schema_version": LINEAGE_SCHEMA,
        "status": "passed",
        "formal": not config.smoke,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "creation_head": head,
        "accepted_commit": config.accepted_commit,
        "baseline_commit": config.baseline_commit,
        "source_scope": source_scope,
        "runtime_equality": equality,
        "repository_differences": differences,
        **facts,
        "placement_predicate": _placement_predicate(source_scope, train),
        "writer_contract": _writer_contract(source_scope),
        "historical_evidence_denylist": _historical_denylist(
            root, formal=not config.smoke
        ),
    }


def _validate_manifest_inputs(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    accepted_commit: str,
    formal: bool,
    train_path: Path,
    validation_path: Path,
) -> None:
    for expected_split, manifest in (("train", train), ("validation", validation)):
        required = {
            "schema_version",
            "split",
            "formal",
            "git",
            "controller",
            "assets",
            "environment",
            "attempt_count",
        }
        if not required <= set(manifest):
            raise ValueError(f"{expected_split} manifest lacks lineage fields")
        if manifest["schema_version"] != "m2-collection-manifest.v1":
            raise ValueError(f"{expected_split} manifest has wrong schema")
        if manifest["split"] != expected_split:
            raise ValueError(f"{expected_split} manifest split differs")
        git = manifest["git"]
        if not isinstance(git, dict) or git.get("commit") != accepted_commit:
            raise ValueError(f"{expected_split} manifest commit differs")
        if formal and not (
            manifest["formal"] is True
            and git.get("source_provenance_clean") is True
            and git.get("provenance_complete") is True
        ):
            raise ValueError(f"{expected_split} formal provenance is incomplete")
    if train["controller"] != validation["controller"]:
        raise ValueError("train and validation controller contracts differ")
    if train["environment"] != validation["environment"]:
        raise ValueError("train and validation environment contracts differ")
    if train["assets"] != validation["assets"]:
        raise ValueError("train and validation asset contracts differ")
    if formal:
        train_report = validate_collection_manifest(train_path)
        validation_report = validate_collection_manifest(validation_path)
        if not train_report.valid or not validation_report.valid:
            raise ValueError("formal source collection manifest is invalid")
        pair = validate_manifest_pair(train_path, validation_path)
        if not pair.valid:
            raise ValueError("formal collection manifest pair is invalid")
        if not (
            train.get("success_count") == 200
            and len(train.get("eligible_successes", [])) == 200
            and validation.get("success_count") == 40
            and len(validation.get("eligible_successes", [])) == 40
        ):
            raise ValueError("formal collection population is not exactly 200+40")


def _manifest_facts(
    train: Mapping[str, Any],
    train_sha: str,
    validation: Mapping[str, Any],
    validation_sha: str,
) -> dict[str, Any]:
    manifests = [
        {
            "split": "train",
            "file_sha256": train_sha,
            "commit": train["git"]["commit"],
            "formal": train["formal"],
            "attempt_count": train["attempt_count"],
            "success_count": train.get("success_count"),
        },
        {
            "split": "validation",
            "file_sha256": validation_sha,
            "commit": validation["git"]["commit"],
            "formal": validation["formal"],
            "attempt_count": validation["attempt_count"],
            "success_count": validation.get("success_count"),
        },
    ]
    return {
        "manifests": manifests,
        "controller": {
            "sha256": sha256_bytes(canonical_json_bytes(train["controller"])),
            "contract": train["controller"],
        },
        "environment": {
            "sha256": sha256_bytes(canonical_json_bytes(train["environment"])),
            "contract": train["environment"],
        },
        "assets": {
            "sha256": sha256_bytes(canonical_json_bytes(train["assets"])),
            "aggregate_manifest_sha256": train["assets"]["aggregate_manifest_sha256"],
        },
    }


def _commit_entries(
    root: Path, commit: str, runtime_paths: Sequence[str]
) -> list[dict[str, Any]]:
    _require_commit(commit, "commit")
    entries: list[dict[str, Any]] = []
    for path in runtime_paths:
        content = _git_bytes(root, "show", f"{commit}:{path}")
        entries.append(
            {
                "path": path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return entries


def _worktree_entries(root: Path, runtime_paths: Sequence[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in runtime_paths:
        content, snapshot, _ = stable_read_relative_file(root, path)
        entries.append(
            {
                "path": path,
                "size_bytes": snapshot.size,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return entries


def _scope_sha(entries: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(entries)))


def _source_scope(
    runtime_paths: Sequence[str],
    accepted_entries: list[dict[str, Any]],
    baseline_entries: list[dict[str, Any]],
    current_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCOPE_SCHEMA,
        "runtime_paths": list(runtime_paths),
        "accepted": {
            "entries": accepted_entries,
            "aggregate_sha256": _scope_sha(accepted_entries),
        },
        "baseline": {
            "entries": baseline_entries,
            "aggregate_sha256": _scope_sha(baseline_entries),
        },
        "current_worktree": {
            "entries": current_entries,
            "aggregate_sha256": _scope_sha(current_entries),
        },
    }


def _runtime_equality(
    accepted_entries: Sequence[Mapping[str, Any]],
    baseline_entries: Sequence[Mapping[str, Any]],
    current_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    accepted = {entry["path"]: entry["sha256"] for entry in accepted_entries}
    baseline = {entry["path"]: entry["sha256"] for entry in baseline_entries}
    current = {entry["path"]: entry["sha256"] for entry in current_entries}
    differing = sorted(
        path
        for path in accepted
        if not (accepted[path] == baseline.get(path) == current.get(path))
    )
    return {
        "equal": not differing,
        "differing_runtime_paths": differing,
    }


def _repository_differences(
    root: Path,
    accepted_commit: str,
    baseline_commit: str,
    documentation_allowlist: Sequence[str],
) -> dict[str, Any]:
    raw = _git_bytes(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        f"{accepted_commit}..{baseline_commit}",
    )
    fields = raw.split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 2:
        raise ValueError("cannot parse Git lineage difference output")
    changes: list[dict[str, str]] = []
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        path = fields[index + 1].decode("utf-8")
        validate_relative_path(path)
        if status not in {"A", "M", "D"}:
            raise ValueError(f"unsupported lineage difference status: {status}")
        changes.append({"status": status, "path": path})
    changed_paths = [change["path"] for change in changes]
    allowlist = list(documentation_allowlist)
    return {
        "documentation_allowlist": allowlist,
        "changes": changes,
        "changed_paths": changed_paths,
        "allowlisted_only": changed_paths == allowlist,
    }


def _placement_predicate(
    source_scope: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    pick_place = next(
        entry
        for entry in source_scope["accepted"]["entries"]
        if entry["path"] == "src/env/pick_place.py"
        or entry["path"].endswith("runtime.py")
    )
    task_config = manifest.get("environment", {}).get("task_config", {})
    return {
        "version": PLACEMENT_PREDICATE_VERSION,
        "implementation_path": pick_place["path"],
        "implementation_sha256": pick_place["sha256"],
        "success_hold_s": task_config.get("success_hold_s"),
        "success_z_tolerance": task_config.get("success_z_tolerance"),
        "requirements": [
            "rotated_cube_obb_inside_all_box_inner_faces",
            "cube_above_box_bottom",
            "cube_below_wall_top",
            "near_bottom_within_tolerance",
            "continuous_hold_duration",
        ],
    }


def _writer_contract(source_scope: Mapping[str, Any]) -> dict[str, Any]:
    writer = next(
        entry
        for entry in source_scope["accepted"]["entries"]
        if entry["path"] == "src/data/hdf5.py" or entry["path"].endswith("writer.py")
    )
    return {
        "schema_version": 1,
        "implementation_path": writer["path"],
        "implementation_sha256": writer["sha256"],
        "time_alignment": "pre_action",
        "action_semantics": (
            "absolute_joint_position_targets_rad[7]+normalized_gripper_open[1]"
        ),
    }


def _historical_denylist(root: Path, *, formal: bool) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if formal:
        for path in FORMAL_HISTORICAL_EVIDENCE:
            content, snapshot, _ = stable_read_relative_file(root, path)
            files.append(
                {
                    "path": path,
                    "size_bytes": snapshot.size,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    return {
        "predicate": "historical-center-point-approximation",
        "commit": HISTORICAL_OLD_PREDICATE_COMMIT,
        "files": files,
        "admissible_as_current_lineage": False,
    }


def _validate_receipt_schema(receipt: Mapping[str, Any], add: Any) -> None:
    if not isinstance(receipt, dict) or set(receipt) != _TOP_KEYS:
        add("lineage.schema", "/", "receipt has the wrong fields")
        return
    if receipt["schema_version"] != LINEAGE_SCHEMA:
        add("lineage.schema", "/schema_version", "wrong lineage schema")
    if receipt["status"] != "passed":
        add("lineage.schema", "/status", "lineage status must be passed")
    if type(receipt["formal"]) is not bool:
        add("lineage.schema", "/formal", "formal must be boolean")
    if not _is_content_id(receipt["receipt_id"]):
        add("lineage.schema", "/receipt_id", "receipt ID is invalid")
    for key in ("creation_head", "accepted_commit", "baseline_commit"):
        if not isinstance(receipt[key], str) or _COMMIT.fullmatch(receipt[key]) is None:
            add("lineage.schema", f"/{key}", f"{key} is not a full Git SHA")
    try:
        generated = datetime.fromisoformat(receipt["generated_at"])
        if generated.tzinfo is None:
            raise ValueError("timezone missing")
    except (TypeError, ValueError):
        add("lineage.schema", "/generated_at", "generated_at is invalid")
    source = receipt["source_scope"]
    if not isinstance(source, dict) or set(source) != {
        "schema_version",
        "runtime_paths",
        "accepted",
        "baseline",
        "current_worktree",
    }:
        add("lineage.schema", "/source_scope", "source scope fields are invalid")
    else:
        if source["schema_version"] != SOURCE_SCOPE_SCHEMA:
            add("lineage.schema", "/source_scope/schema_version", "wrong scope schema")
        try:
            _validated_paths(tuple(source["runtime_paths"]), "runtime_paths")
            for name in ("accepted", "baseline", "current_worktree"):
                scope = source[name]
                if not isinstance(scope, dict) or set(scope) != {
                    "entries",
                    "aggregate_sha256",
                }:
                    raise ValueError(f"{name} scope has wrong fields")
                if not _is_sha256(scope["aggregate_sha256"]):
                    raise ValueError(f"{name} aggregate digest is invalid")
                if not isinstance(scope["entries"], list):
                    raise ValueError(f"{name} entries must be a list")
                for entry in scope["entries"]:
                    if not isinstance(entry, dict) or set(entry) != {
                        "path",
                        "size_bytes",
                        "sha256",
                    }:
                        raise ValueError(f"{name} entry has wrong fields")
                    validate_relative_path(entry["path"])
                    if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
                        raise ValueError(f"{name} entry size is invalid")
                    if not _is_sha256(entry["sha256"]):
                        raise ValueError(f"{name} entry digest is invalid")
        except (TypeError, ValueError) as error:
            add("lineage.schema", "/source_scope", str(error))
    equality = receipt["runtime_equality"]
    if not isinstance(equality, dict) or set(equality) != {
        "equal",
        "differing_runtime_paths",
    }:
        add("lineage.schema", "/runtime_equality", "runtime equality is invalid")
    elif not (
        type(equality["equal"]) is bool
        and isinstance(equality["differing_runtime_paths"], list)
    ):
        add("lineage.schema", "/runtime_equality", "runtime equality values invalid")
    differences = receipt["repository_differences"]
    if not isinstance(differences, dict) or set(differences) != {
        "documentation_allowlist",
        "changes",
        "changed_paths",
        "allowlisted_only",
    }:
        add("lineage.schema", "/repository_differences", "difference audit invalid")
    if not isinstance(receipt["manifests"], list) or len(receipt["manifests"]) != 2:
        add("lineage.schema", "/manifests", "exactly two manifests are required")
    for name in (
        "controller",
        "environment",
        "assets",
        "placement_predicate",
        "writer_contract",
        "historical_evidence_denylist",
    ):
        if not isinstance(receipt[name], dict):
            add("lineage.schema", f"/{name}", f"{name} must be an object")


def _validated_paths(values: Sequence[str], label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        if label == "documentation_allowlist" and values == ():
            return ()
        raise ValueError(f"{label} must be a non-empty tuple")
    for value in values:
        validate_relative_path(value)
    if len(set(values)) != len(values) or tuple(sorted(values)) != tuple(values):
        raise ValueError(f"{label} must be unique and sorted")
    return tuple(values)


def _require_commit(value: str, label: str) -> None:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA")


def _head(root: Path) -> str:
    head = _git_text(root, "rev-parse", "HEAD")
    _require_commit(head, "HEAD")
    return head


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [_GIT, "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    )
    return result.returncode == 0


def _git_text(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("utf-8").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        [_GIT, "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise ValueError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout


def _content_id(payload: Mapping[str, Any]) -> str:
    return f"sha256:{sha256_bytes(canonical_json_bytes(dict(payload)))}"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_content_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    validate = commands.add_parser("validate")
    for command in (create, validate):
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--train-manifest", type=Path, required=True)
        command.add_argument("--validation-manifest", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--smoke", action="store_true")
    create.add_argument("--accepted-commit", default=FORMAL_ACCEPTED_COMMIT)
    create.add_argument("--baseline-commit", default=FORMAL_BASELINE_COMMIT)
    create.add_argument("--runtime-path", action="append")
    create.add_argument("--documentation-path", action="append")
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            runtime_paths = (
                tuple(sorted(args.runtime_path))
                if args.runtime_path is not None
                else FORMAL_RUNTIME_PATHS
            )
            documentation = (
                tuple(sorted(args.documentation_path))
                if args.documentation_path is not None
                else FORMAL_DOCUMENTATION_ALLOWLIST
            )
            result = create_lineage_revalidation_receipt(
                LineageRevalidationConfig(
                    repository_root=args.repository_root,
                    train_manifest_path=args.train_manifest,
                    validation_manifest_path=args.validation_manifest,
                    output_path=args.output,
                    accepted_commit=args.accepted_commit,
                    baseline_commit=args.baseline_commit,
                    runtime_paths=runtime_paths,
                    documentation_allowlist=documentation,
                    smoke=args.smoke,
                )
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        report = validate_lineage_revalidation_receipt(
            args.receipt,
            repository_root=args.repository_root,
            train_manifest_path=args.train_manifest,
            validation_manifest_path=args.validation_manifest,
        )
        result = {
            "valid": report.valid,
            "passed": report.passed,
            "errors": [str(error) for error in report.errors],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.valid and report.passed else 2
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


__all__ = [
    "LineageRevalidationConfig",
    "LineageValidationIssue",
    "LineageValidationReport",
    "create_lineage_revalidation_receipt",
    "validate_lineage_revalidation_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
