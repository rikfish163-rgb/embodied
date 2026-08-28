"""Closed-world M2 acceptance gate over collection, replay, and human review."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from expert.evaluate import _git_state

from .lineage import validate_lineage_revalidation_receipt
from .manifest import (
    VALIDATION_REPORT_SCHEMA,
    atomic_write_json_no_clobber,
    canonical_json_bytes,
    load_json_object_with_sha256,
    sha256_bytes,
    stable_read_relative_file,
    validate_collection_manifest,
    validate_manifest_pair,
    validate_relative_path,
)
from .manual_review import validate_manual_review_pack
from .replay import (
    REPLAY_PLAN_FILENAME,
    validate_pair_replay_summary,
)
from .reporting import REPORT_SCHEMA, _aggregate_actions
from .review_attestation import validate_human_review_attestation

M2_GATE_SCHEMA = "m2-gate-receipt.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TOP_KEYS = {
    "schema_version",
    "gate_id",
    "status",
    "formal",
    "generated_at",
    "creation_git",
    "inputs",
    "collection",
    "data_reports",
    "replay",
    "human_review",
    "lineage",
    "checks",
}
_INPUT_NAMES = {
    "train_manifest",
    "validation_manifest",
    "train_validation_report",
    "pair_validation_report",
    "train_report",
    "validation_report",
    "replay_summary",
    "review_pack",
    "human_attestation",
    "reviewer_registry",
    "lineage_receipt",
}


@dataclass(frozen=True)
class M2GateConfig:
    project_root: Path
    train_manifest_path: Path
    validation_manifest_path: Path
    train_validation_report_path: Path
    pair_validation_report_path: Path
    train_report_path: Path
    validation_report_path: Path
    replay_summary_path: Path
    review_pack_path: Path
    human_attestation_path: Path
    reviewer_registry_path: Path
    lineage_receipt_path: Path
    output_path: Path
    smoke: bool = False


@dataclass(frozen=True)
class M2GateIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class M2GateValidationReport:
    path: Path
    errors: tuple[M2GateIssue, ...]
    receipt: dict[str, Any] | None = None
    sha256: str | None = None
    passed: bool = False
    formal: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(f"invalid M2 gate receipt:\n{self.format_errors()}")


def create_m2_gate_receipt(
    config: M2GateConfig,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    normalized = _normalize_config(config)
    if not normalized.human_attestation_path.is_file():
        raise ValueError("a signed human review attestation is required")
    facts = _evaluate_inputs(normalized)
    creation_git = _creation_git(normalized.project_root, formal=not normalized.smoke)
    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    payload = {
        "schema_version": M2_GATE_SCHEMA,
        "status": "passed",
        "formal": not normalized.smoke,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "creation_git": creation_git,
        **facts,
    }
    receipt = {**payload, "gate_id": _content_id(payload)}
    atomic_write_json_no_clobber(normalized.output_path, receipt)
    report = validate_m2_gate_receipt(
        normalized.output_path,
        project_root=normalized.project_root,
    )
    report.raise_for_errors()
    return receipt


def validate_m2_gate_receipt(
    path: Path,
    *,
    project_root: Path,
) -> M2GateValidationReport:
    receipt_path = Path(path)
    errors: list[M2GateIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(M2GateIssue(code, location, message))

    try:
        receipt, initial_sha = load_json_object_with_sha256(receipt_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("m2.gate.file", "/", str(error))
        return M2GateValidationReport(receipt_path, tuple(errors))
    _validate_schema(receipt, add)
    if errors:
        return M2GateValidationReport(receipt_path, tuple(errors), receipt, initial_sha)
    identity = {key: receipt[key] for key in _TOP_KEYS if key != "gate_id"}
    if receipt["gate_id"] != _content_id(identity):
        add("m2.gate.identity", "/gate_id", "gate identity is wrong")
    try:
        config = _config_from_receipt(
            receipt,
            project_root=Path(project_root).expanduser().absolute(),
            output_path=receipt_path,
        )
        current_facts = _evaluate_inputs(config)
        for field in (
            "inputs",
            "collection",
            "data_reports",
            "replay",
            "human_review",
            "lineage",
            "checks",
        ):
            if receipt[field] != current_facts[field]:
                add(
                    "m2.gate.input",
                    f"/{field}",
                    f"{field} differs from revalidated upstream evidence",
                )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        add("m2.gate.input", "/", str(error))
    try:
        _, final_sha = load_json_object_with_sha256(receipt_path)
        if final_sha != initial_sha:
            add("m2.gate.file", "/", "gate receipt changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("m2.gate.file", "/", str(error))
    passed = not errors and receipt["status"] == "passed"
    return M2GateValidationReport(
        receipt_path,
        tuple(errors),
        receipt,
        initial_sha,
        passed=passed,
        formal=receipt["formal"] is True,
    )


def _normalize_config(config: M2GateConfig) -> M2GateConfig:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be boolean")
    root = Path(config.project_root).expanduser().absolute()
    fields = {
        name: Path(getattr(config, name)).expanduser().absolute()
        for name in (
            "train_manifest_path",
            "validation_manifest_path",
            "train_validation_report_path",
            "pair_validation_report_path",
            "train_report_path",
            "validation_report_path",
            "replay_summary_path",
            "review_pack_path",
            "human_attestation_path",
            "reviewer_registry_path",
            "lineage_receipt_path",
            "output_path",
        )
    }
    for name, value in fields.items():
        try:
            relative = value.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"{name} must be inside project_root") from error
        validate_relative_path(relative)
    return M2GateConfig(project_root=root, smoke=config.smoke, **fields)


def _evaluate_inputs(config: M2GateConfig) -> dict[str, Any]:
    root = config.project_root
    input_paths = {
        "train_manifest": config.train_manifest_path,
        "validation_manifest": config.validation_manifest_path,
        "train_validation_report": config.train_validation_report_path,
        "pair_validation_report": config.pair_validation_report_path,
        "train_report": config.train_report_path,
        "validation_report": config.validation_report_path,
        "replay_summary": config.replay_summary_path,
        "review_pack": config.review_pack_path,
        "human_attestation": config.human_attestation_path,
        "reviewer_registry": config.reviewer_registry_path,
        "lineage_receipt": config.lineage_receipt_path,
    }
    initial_refs = {name: _input_ref(root, path) for name, path in input_paths.items()}

    train_report = validate_collection_manifest(config.train_manifest_path)
    validation_report = validate_collection_manifest(config.validation_manifest_path)
    pair_report = validate_manifest_pair(
        config.train_manifest_path, config.validation_manifest_path
    )
    train_report.raise_for_errors()
    validation_report.raise_for_errors()
    pair_report.raise_for_errors()
    if (
        train_report.manifest is None
        or validation_report.manifest is None
        or train_report.sha256 is None
        or validation_report.sha256 is None
    ):
        raise RuntimeError("validated collection manifest facts are unavailable")
    train = train_report.manifest
    validation = validation_report.manifest
    formal = not config.smoke
    if formal and not (
        train["formal"] is True
        and validation["formal"] is True
        and train["success_count"] == 200
        and len(train["eligible_successes"]) == 200
        and validation["success_count"] == 40
        and len(validation["eligible_successes"]) == 40
    ):
        raise ValueError("formal M2 gate requires exact valid 200+40 collections")

    _validate_published_validation_report(
        config.train_validation_report_path,
        expected_paths=(config.train_manifest_path,),
    )
    _validate_published_validation_report(
        config.pair_validation_report_path,
        expected_paths=(config.train_manifest_path, config.validation_manifest_path),
    )
    train_data_report = _validate_data_report(
        config.train_report_path,
        config.train_manifest_path,
        train,
        train_report.sha256,
        formal=formal,
    )
    validation_data_report = _validate_data_report(
        config.validation_report_path,
        config.validation_manifest_path,
        validation,
        validation_report.sha256,
        formal=formal,
    )

    lineage_report = validate_lineage_revalidation_receipt(
        config.lineage_receipt_path,
        repository_root=root,
        train_manifest_path=config.train_manifest_path,
        validation_manifest_path=config.validation_manifest_path,
    )
    lineage_report.raise_for_errors()
    if not lineage_report.passed or lineage_report.receipt is None:
        raise ValueError("lineage revalidation has not passed")
    if formal and lineage_report.receipt["formal"] is not True:
        raise ValueError("formal M2 gate requires formal lineage evidence")

    replay_report = validate_pair_replay_summary(
        config.replay_summary_path,
        plan_path=config.replay_summary_path.parent / REPLAY_PLAN_FILENAME,
        train_manifest_path=config.train_manifest_path,
        validation_manifest_path=config.validation_manifest_path,
        project_root=root,
    )
    replay_report.raise_for_errors()
    if replay_report.summary is None or replay_report.sha256 is None:
        raise RuntimeError("validated paired replay facts are unavailable")
    replay = replay_report.summary
    if replay["gate"]["passed"] is not True:
        raise ValueError("paired replay gate has not passed")
    if formal and replay["formal"] is not True:
        raise ValueError("formal M2 gate requires formal paired replay")

    pack_report = validate_manual_review_pack(
        config.review_pack_path,
        train_manifest_path=config.train_manifest_path,
        validation_manifest_path=config.validation_manifest_path,
    )
    pack_report.raise_for_errors()
    if pack_report.pack is None or pack_report.sha256 is None:
        raise RuntimeError("validated manual review pack facts are unavailable")
    attestation_report = validate_human_review_attestation(
        config.human_attestation_path,
        review_pack_path=config.review_pack_path,
        reviewer_registry_path=config.reviewer_registry_path,
        reviewer_repository_root=root,
        train_manifest_path=config.train_manifest_path,
        validation_manifest_path=config.validation_manifest_path,
    )
    attestation_report.raise_for_errors()
    if (
        not attestation_report.complete
        or attestation_report.attestation is None
        or attestation_report.sha256 is None
    ):
        raise ValueError("signed human review attestation is incomplete")
    attestation = attestation_report.attestation
    if formal and not (
        attestation_report.formal
        and len(attestation["reviews"]) == 20
        and all(review["verdict"] == "consistent" for review in attestation["reviews"])
    ):
        raise ValueError("formal M2 gate requires 20 consistent signed human reviews")

    final_refs = {name: _input_ref(root, path) for name, path in input_paths.items()}
    if final_refs != initial_refs:
        raise RuntimeError("M2 gate inputs changed during validation")
    collection = {
        "train": _collection_facts(train, train_report.sha256),
        "validation": _collection_facts(validation, validation_report.sha256),
        "pair_valid": True,
        "attempt_count": train["attempt_count"] + validation["attempt_count"],
        "success_count": train["success_count"] + validation["success_count"],
        "failed_attempt_count": (
            train["attempt_count"]
            + validation["attempt_count"]
            - train["success_count"]
            - validation["success_count"]
        ),
    }
    data_reports = {
        "train": train_data_report,
        "validation": validation_data_report,
    }
    replay_facts = {
        "summary_id": replay["summary_id"],
        "plan_id": replay["plan_id"],
        "formal": replay["formal"],
        "trial_count": replay["trial_count"],
        "success_count": replay["success_count"],
        "required_successes": replay["gate"]["required_successes"],
        "passed": replay["gate"]["passed"],
    }
    human_review = {
        "pack_id": pack_report.pack["pack_id"],
        "attestation_id": attestation["attestation_id"],
        "formal": attestation_report.formal,
        "review_count": len(attestation["reviews"]),
        "reviewer_id": attestation["reviewer"]["reviewer_id"],
        "signer_fingerprint": attestation["signature"]["signer_fingerprint"],
        "all_consistent": all(
            review["verdict"] == "consistent" for review in attestation["reviews"]
        ),
        "complete": attestation_report.complete,
    }
    lineage = {
        "receipt_id": lineage_report.receipt["receipt_id"],
        "formal": lineage_report.receipt["formal"],
        "accepted_commit": lineage_report.receipt["accepted_commit"],
        "baseline_commit": lineage_report.receipt["baseline_commit"],
        "runtime_equal": lineage_report.receipt["runtime_equality"]["equal"],
        "passed": lineage_report.passed,
    }
    checks = {
        "collection_pair_valid": True,
        "published_validation_reports_valid": True,
        "data_reports_recomputed": True,
        "paired_replay_passed": replay_facts["passed"],
        "human_review_signed_complete": human_review["complete"],
        "lineage_passed": lineage["passed"],
        "all_input_hashes_stable": True,
    }
    return {
        "inputs": initial_refs,
        "collection": collection,
        "data_reports": data_reports,
        "replay": replay_facts,
        "human_review": human_review,
        "lineage": lineage,
        "checks": checks,
    }


def _validate_published_validation_report(
    path: Path,
    *,
    expected_paths: Sequence[Path],
) -> None:
    report, _ = load_json_object_with_sha256(path)
    if set(report) != {"schema_version", "valid", "manifest_paths", "errors"}:
        raise ValueError("published collection validation report has wrong fields")
    if not (
        report["schema_version"] == VALIDATION_REPORT_SCHEMA
        and report["valid"] is True
        and report["errors"] == []
        and isinstance(report["manifest_paths"], list)
    ):
        raise ValueError("published collection validation report did not pass")
    actual = sorted(
        Path(value).expanduser().absolute() for value in report["manifest_paths"]
    )
    expected = sorted(Path(value).expanduser().absolute() for value in expected_paths)
    if actual != expected:
        raise ValueError("published collection validation paths differ")


def _validate_data_report(
    path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    *,
    formal: bool,
) -> dict[str, Any]:
    report, report_sha = load_json_object_with_sha256(path)
    required = {
        "schema_version",
        "formal",
        "source_manifest_sha256",
        "source_split",
        "source_manifest_formal",
        "counts",
        "episode_lengths",
        "actions",
        "schema_errors",
    }
    if not required <= set(report):
        raise ValueError("M2 data report lacks required fields")
    if not (
        report["schema_version"] == REPORT_SCHEMA
        and report["source_manifest_sha256"] == manifest_sha
        and report["source_split"] == manifest["split"]
        and report["source_manifest_formal"] is manifest["formal"]
        and report["schema_errors"] == []
    ):
        raise ValueError("M2 data report source identity differs")
    if formal and report["formal"] is not True:
        raise ValueError("formal M2 gate requires formal data reports")
    counts = {
        "attempts": manifest["attempt_count"],
        "successes": manifest["success_count"],
        "failures": manifest["attempt_count"] - manifest["success_count"],
    }
    if report["counts"] != counts:
        raise ValueError("M2 data report counts differ")
    temporary_parent = Path(os.environ.get("TMPDIR", path.parent))
    if not temporary_parent.is_dir():
        temporary_parent = path.parent
    _, lengths, actions, _, _ = _aggregate_actions(
        dict(manifest),
        manifest_path.parent,
        temporary_parent=temporary_parent,
    )
    if report["episode_lengths"] != lengths or report["actions"] != actions:
        raise ValueError("M2 data report statistics differ from source episodes")
    return {
        "file_sha256": report_sha,
        "formal": report["formal"],
        "source_manifest_sha256": manifest_sha,
        "counts": counts,
        "statistics_recomputed": True,
    }


def _collection_facts(manifest: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "manifest_sha256": digest,
        "split": manifest["split"],
        "formal": manifest["formal"],
        "attempt_count": manifest["attempt_count"],
        "success_count": manifest["success_count"],
        "eligible_success_count": len(manifest["eligible_successes"]),
        "ledger_path": manifest["ledger_path"],
        "controller_sha256": sha256_bytes(canonical_json_bytes(manifest["controller"])),
        "assets_sha256": sha256_bytes(canonical_json_bytes(manifest["assets"])),
        "environment_sha256": sha256_bytes(
            canonical_json_bytes(manifest["environment"])
        ),
    }


def _input_ref(root: Path, path: Path) -> dict[str, Any]:
    relative = Path(path).expanduser().absolute().relative_to(root).as_posix()
    validate_relative_path(relative)
    content, snapshot, _ = stable_read_relative_file(root, relative)
    return {
        "path": relative,
        "size_bytes": snapshot.size,
        "sha256": sha256_bytes(content),
    }


def _creation_git(root: Path, *, formal: bool) -> dict[str, Any]:
    if not formal:
        return {"formal_checked": False, "commit": None}
    git = _git_state(root)
    if not (
        git.get("provenance_complete") is True
        and git.get("source_provenance_clean") is True
        and git.get("tracked_worktree_clean") is True
        and isinstance(git.get("commit"), str)
        and _COMMIT.fullmatch(git["commit"]) is not None
    ):
        raise ValueError("formal M2 gate requires clean complete source provenance")
    return {"formal_checked": True, "commit": git["commit"], "git": git}


def _config_from_receipt(
    receipt: Mapping[str, Any],
    *,
    project_root: Path,
    output_path: Path,
) -> M2GateConfig:
    inputs = receipt["inputs"]
    mapping = {
        "train_manifest_path": "train_manifest",
        "validation_manifest_path": "validation_manifest",
        "train_validation_report_path": "train_validation_report",
        "pair_validation_report_path": "pair_validation_report",
        "train_report_path": "train_report",
        "validation_report_path": "validation_report",
        "replay_summary_path": "replay_summary",
        "review_pack_path": "review_pack",
        "human_attestation_path": "human_attestation",
        "reviewer_registry_path": "reviewer_registry",
        "lineage_receipt_path": "lineage_receipt",
    }
    paths = {
        field: project_root / validate_relative_path(inputs[name]["path"])
        for field, name in mapping.items()
    }
    return _normalize_config(
        M2GateConfig(
            project_root=project_root,
            output_path=output_path,
            smoke=receipt["formal"] is False,
            **paths,
        )
    )


def _validate_schema(receipt: Mapping[str, Any], add: Any) -> None:
    if not isinstance(receipt, dict) or set(receipt) != _TOP_KEYS:
        add("m2.gate.schema", "/", "gate receipt has wrong fields")
        return
    if receipt["schema_version"] != M2_GATE_SCHEMA:
        add("m2.gate.schema", "/schema_version", "wrong M2 gate schema")
    if receipt["status"] != "passed":
        add("m2.gate.schema", "/status", "gate status must be passed")
    if type(receipt["formal"]) is not bool:
        add("m2.gate.schema", "/formal", "formal must be boolean")
    if not _is_content_id(receipt["gate_id"]):
        add("m2.gate.schema", "/gate_id", "gate ID is invalid")
    try:
        generated = datetime.fromisoformat(receipt["generated_at"])
        if generated.tzinfo is None:
            raise ValueError("timezone missing")
    except (TypeError, ValueError):
        add("m2.gate.schema", "/generated_at", "generated_at is invalid")
    creation_git = receipt["creation_git"]
    if not isinstance(creation_git, dict):
        add("m2.gate.schema", "/creation_git", "creation_git must be an object")
    elif receipt["formal"] is True:
        if not (
            set(creation_git) == {"formal_checked", "commit", "git"}
            and creation_git["formal_checked"] is True
            and isinstance(creation_git["commit"], str)
            and _COMMIT.fullmatch(creation_git["commit"]) is not None
            and isinstance(creation_git["git"], dict)
        ):
            add(
                "m2.gate.schema",
                "/creation_git",
                "formal creation Git evidence is invalid",
            )
    elif creation_git != {"formal_checked": False, "commit": None}:
        add(
            "m2.gate.schema",
            "/creation_git",
            "smoke creation Git evidence is invalid",
        )
    inputs = receipt["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_NAMES:
        add("m2.gate.schema", "/inputs", "gate inputs have wrong fields")
    else:
        for name, reference in inputs.items():
            location = f"/inputs/{name}"
            if not isinstance(reference, dict) or set(reference) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                add("m2.gate.schema", location, "input reference has wrong fields")
                continue
            try:
                validate_relative_path(reference["path"])
            except (TypeError, ValueError) as error:
                add("m2.gate.schema", f"{location}/path", str(error))
            if (
                type(reference["size_bytes"]) is not int
                or reference["size_bytes"] < 0
                or not _is_sha256(reference["sha256"])
            ):
                add("m2.gate.schema", location, "input size or digest is invalid")
    for name in (
        "collection",
        "data_reports",
        "replay",
        "human_review",
        "lineage",
        "checks",
    ):
        if not isinstance(receipt[name], dict):
            add("m2.gate.schema", f"/{name}", f"{name} must be an object")
    checks = receipt["checks"]
    if isinstance(checks, dict) and (
        not checks or any(value is not True for value in checks.values())
    ):
        add("m2.gate.schema", "/checks", "every M2 gate check must be true")


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
    create.add_argument("--project-root", required=True, type=Path)
    create.add_argument("--train-manifest", required=True, type=Path)
    create.add_argument("--validation-manifest", required=True, type=Path)
    create.add_argument("--train-validation-report", required=True, type=Path)
    create.add_argument("--pair-validation-report", required=True, type=Path)
    create.add_argument("--train-report", required=True, type=Path)
    create.add_argument("--validation-report", required=True, type=Path)
    create.add_argument("--replay-summary", required=True, type=Path)
    create.add_argument("--review-pack", required=True, type=Path)
    create.add_argument("--human-attestation", required=True, type=Path)
    create.add_argument("--reviewer-registry", required=True, type=Path)
    create.add_argument("--lineage-receipt", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--smoke", action="store_true")
    validate.add_argument("--project-root", required=True, type=Path)
    validate.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            receipt = create_m2_gate_receipt(
                M2GateConfig(
                    project_root=args.project_root,
                    train_manifest_path=args.train_manifest,
                    validation_manifest_path=args.validation_manifest,
                    train_validation_report_path=args.train_validation_report,
                    pair_validation_report_path=args.pair_validation_report,
                    train_report_path=args.train_report,
                    validation_report_path=args.validation_report,
                    replay_summary_path=args.replay_summary,
                    review_pack_path=args.review_pack,
                    human_attestation_path=args.human_attestation,
                    reviewer_registry_path=args.reviewer_registry,
                    lineage_receipt_path=args.lineage_receipt,
                    output_path=args.output,
                    smoke=args.smoke,
                )
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        report = validate_m2_gate_receipt(
            args.receipt,
            project_root=args.project_root,
        )
        output = {
            "valid": report.valid,
            "passed": report.passed,
            "formal": report.formal,
            "errors": [str(error) for error in report.errors],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.valid and report.passed else 2
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


__all__ = [
    "M2GateConfig",
    "M2GateIssue",
    "M2GateValidationReport",
    "create_m2_gate_receipt",
    "validate_m2_gate_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
