"""Signed, human-authored M2 review attestations.

This module derives hashes and validates signatures.  It never generates a
reviewer identity, review timestamps, findings, verdicts, or a private-key
signature.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .manifest import (
    append_jsonl_fsync,
    atomic_write_json_no_clobber,
    canonical_json_bytes,
    initialize_jsonl_no_clobber,
    load_json_object_with_sha256,
    load_jsonl_relative_with_sha256,
    manifest_sha256,
    sha256_bytes,
    validate_relative_path,
)
from .manual_review import (
    FORMAL_REVIEW_COUNT,
    REVIEW_PACK_SCHEMA,
    REVIEW_TRIAL_SCHEMA,
    validate_manual_review_pack,
)

REVIEWER_REGISTRY_SCHEMA = "m2-reviewer-registry.v1"
ATTESTATION_REQUEST_SCHEMA = "m2-human-review-signing-request.v1"
ATTESTATION_SCHEMA = "m2-human-review-attestation.v1"
ATTESTATION_NAMESPACE = "embodied-m2-human-review-v1"
ATTESTATION_STATEMENT = (
    "I personally reviewed every referenced image, recorded each finding "
    "without automated judgment, and attest that these records are accurate."
)
SIGNING_MESSAGE_FILENAME = "attestation-message.jsonl"
ATTESTATION_REQUEST_FILENAME = "attestation-request.json"
SSH_SIGNATURE_ALGORITHM = "openssh-sshsig-v1"
_GIT_EXECUTABLE = "/usr/bin/git"

_REVIEWER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}\Z")
_COMMIT_ID = re.compile(r"[0-9a-f]{40}\Z")
_REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "declared_at_utc",
    "reviewers",
}
_REVIEWER_KEYS = {
    "reviewer_id",
    "display_name",
    "ssh_public_key",
    "ssh_fingerprint",
}
_REVIEW_KEYS = {
    "schema_version",
    "manual_review_id",
    "review_pack_id",
    "manifest_id",
    "split",
    "attempt_index",
    "seed",
    "source_relative_path",
    "source_file_sha256",
    "source_num_steps",
    "classification",
    "media",
    "reviewer_id",
    "review_started_at_utc",
    "review_completed_at_utc",
    "finding",
    "verdict",
}
_PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "formal",
    "review_pack",
    "reviewer_registry",
    "reviewer",
    "reviews",
    "attestation_statement",
}
_REQUEST_KEYS = {
    "schema_version",
    "attestation_id",
    "status",
    "signing_namespace",
    "payload",
    "signing_message",
}
_SIGNATURE_KEYS = {
    "algorithm",
    "namespace",
    "signer_fingerprint",
    "armored_signature",
}


@dataclass(frozen=True)
class HumanReviewAttestationConfig:
    review_pack_path: Path
    completed_reviews_path: Path
    reviewer_registry_path: Path
    reviewer_repository_root: Path
    reviewer_registry_commit: str
    reviewer_id: str
    train_manifest_path: Path
    validation_manifest_path: Path
    output_dir: Path
    smoke: bool = False


@dataclass(frozen=True)
class HumanReviewAttestationIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class HumanReviewAttestationReport:
    path: Path
    errors: tuple[HumanReviewAttestationIssue, ...]
    attestation: dict[str, Any] | None = None
    sha256: str | None = None
    status: str = "invalid"
    complete: bool = False
    formal: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                f"invalid human review attestation {self.path}:\n{self.format_errors()}"
            )


def create_reviewer_registry(
    path: Path,
    *,
    reviewer_id: str,
    display_name: str,
    ssh_public_key: str,
    declared_at_utc: str,
) -> dict[str, Any]:
    """Publish a reviewer-supplied public identity; no private key is read."""

    _require_reviewer_id(reviewer_id)
    _require_text(display_name, "display_name", maximum=256)
    declared = _parse_utc(declared_at_utc, "declared_at_utc")
    key_type, key_data = _parse_ssh_public_key(ssh_public_key)
    reviewer = {
        "reviewer_id": reviewer_id,
        "display_name": display_name,
        "ssh_public_key": f"{key_type} {key_data}",
        "ssh_fingerprint": _ssh_fingerprint(key_data),
    }
    identity = {
        "schema_version": REVIEWER_REGISTRY_SCHEMA,
        "declared_at_utc": declared.isoformat(),
        "reviewers": [reviewer],
    }
    registry = {**identity, "registry_id": _content_id(identity)}
    atomic_write_json_no_clobber(Path(path), registry)
    return registry


def create_human_review_attestation_request(
    config: HumanReviewAttestationConfig,
) -> dict[str, Any]:
    """Validate human-authored rows and publish the exact bytes to sign."""

    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    _require_reviewer_id(config.reviewer_id)
    output_dir = Path(config.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"attestation request output already exists: {output_dir}"
        )

    pack_path = Path(config.review_pack_path).expanduser().absolute()
    pack_report = validate_manual_review_pack(
        pack_path,
        train_manifest_path=Path(config.train_manifest_path).expanduser().absolute(),
        validation_manifest_path=Path(config.validation_manifest_path)
        .expanduser()
        .absolute(),
    )
    pack_report.raise_for_errors()
    if pack_report.pack is None or pack_report.sha256 is None:
        raise RuntimeError("validated review pack metadata is unavailable")
    pack = pack_report.pack
    if pack["schema_version"] != REVIEW_PACK_SCHEMA:
        raise ValueError("wrong manual review pack schema")
    if not config.smoke and not (
        pack["formal"] is True and len(pack["selected_reviews"]) == FORMAL_REVIEW_COUNT
    ):
        raise ValueError("formal attestation requires an exact 20-row formal pack")

    registry_path = Path(config.reviewer_registry_path).expanduser().absolute()
    registry, registry_sha = _load_reviewer_registry(registry_path)
    reviewer = _select_reviewer(registry, config.reviewer_id)
    registry_reference, commit_time = _bind_registry_commit(
        registry_path,
        Path(config.reviewer_repository_root).expanduser().absolute(),
        config.reviewer_registry_commit,
        registry,
        registry_sha,
    )
    rows, _rows_sha, _, _ = load_jsonl_relative_with_sha256(
        Path(config.completed_reviews_path).expanduser().absolute().parent,
        Path(config.completed_reviews_path).name,
    )
    reviews = _validated_human_rows(
        rows,
        pack,
        reviewer_id=config.reviewer_id,
        registry_declared_at=_parse_utc(
            registry["declared_at_utc"], "registry.declared_at_utc"
        ),
        registry_commit_time=commit_time,
    )
    payload = {
        "schema_version": ATTESTATION_SCHEMA,
        "status": "signed",
        "formal": pack["formal"],
        "review_pack": {
            "schema_version": pack["schema_version"],
            "pack_id": pack["pack_id"],
            "file_sha256": pack_report.sha256,
        },
        "reviewer_registry": registry_reference,
        "reviewer": reviewer,
        "reviews": reviews,
        "attestation_statement": ATTESTATION_STATEMENT,
    }
    attestation_id = _content_id(payload)
    message_bytes = canonical_json_bytes(payload) + b"\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    message_path = output_dir / SIGNING_MESSAGE_FILENAME
    state = initialize_jsonl_no_clobber(message_path)
    append_jsonl_fsync(message_path, payload, expected_snapshot=state)
    if manifest_sha256(message_path) != sha256_bytes(message_bytes):
        raise RuntimeError("published signing message differs from canonical payload")
    request = {
        "schema_version": ATTESTATION_REQUEST_SCHEMA,
        "attestation_id": attestation_id,
        "status": "awaiting_signature",
        "signing_namespace": ATTESTATION_NAMESPACE,
        "payload": payload,
        "signing_message": {
            "path": SIGNING_MESSAGE_FILENAME,
            "sha256": sha256_bytes(message_bytes),
        },
    }
    atomic_write_json_no_clobber(output_dir / ATTESTATION_REQUEST_FILENAME, request)
    return request


def finalize_human_review_attestation(
    request_path: Path,
    *,
    signature_path: Path,
    output_path: Path,
    review_pack_path: Path,
    reviewer_registry_path: Path,
    reviewer_repository_root: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
) -> dict[str, Any]:
    """Verify a detached SSH signature and publish the signed bundle."""

    request_path = Path(request_path).expanduser().absolute()
    request, _request_sha = load_json_object_with_sha256(request_path)
    _validate_request(request)
    message_path = request_path.parent / validate_relative_path(
        request["signing_message"]["path"]
    )
    message = _read_exact_signing_message(
        message_path,
        request["payload"],
        request["signing_message"]["sha256"],
    )
    signature = _read_signature(Path(signature_path).expanduser().absolute())
    reviewer = request["payload"]["reviewer"]
    _verify_ssh_signature(
        message,
        signature,
        reviewer_id=reviewer["reviewer_id"],
        public_key=reviewer["ssh_public_key"],
        namespace=request["signing_namespace"],
    )
    attestation = {
        **request["payload"],
        "attestation_id": request["attestation_id"],
        "signature": {
            "algorithm": SSH_SIGNATURE_ALGORITHM,
            "namespace": request["signing_namespace"],
            "signer_fingerprint": reviewer["ssh_fingerprint"],
            "armored_signature": signature,
        },
    }
    temporary_parent = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
        candidate_path = Path(directory) / "attestation.json"
        atomic_write_json_no_clobber(candidate_path, attestation)
        candidate_report = validate_human_review_attestation(
            candidate_path,
            review_pack_path=review_pack_path,
            reviewer_registry_path=reviewer_registry_path,
            reviewer_repository_root=reviewer_repository_root,
            train_manifest_path=train_manifest_path,
            validation_manifest_path=validation_manifest_path,
        )
        candidate_report.raise_for_errors()
    atomic_write_json_no_clobber(Path(output_path), attestation)
    return attestation


def validate_human_review_attestation(
    path: Path,
    *,
    review_pack_path: Path,
    reviewer_registry_path: Path,
    reviewer_repository_root: Path,
    train_manifest_path: Path,
    validation_manifest_path: Path,
) -> HumanReviewAttestationReport:
    """Validate closed-world review content, registry lineage, and SSH signature."""

    attestation_path = Path(path)
    issues: list[HumanReviewAttestationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        issues.append(HumanReviewAttestationIssue(code, location, message))

    try:
        attestation, initial_sha = load_json_object_with_sha256(attestation_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("attestation.file", "/", str(error))
        return HumanReviewAttestationReport(attestation_path, tuple(issues))
    _validate_attestation_schema(attestation, add)
    if issues:
        return HumanReviewAttestationReport(
            attestation_path,
            tuple(issues),
            attestation,
            initial_sha,
        )

    payload = {key: attestation[key] for key in _PAYLOAD_KEYS}
    expected_id = _content_id(payload)
    if attestation["attestation_id"] != expected_id:
        add(
            "attestation.identity",
            "/attestation_id",
            "attestation ID differs from the signed payload",
        )
    try:
        pack_report = validate_manual_review_pack(
            Path(review_pack_path).expanduser().absolute(),
            train_manifest_path=Path(train_manifest_path).expanduser().absolute(),
            validation_manifest_path=Path(validation_manifest_path)
            .expanduser()
            .absolute(),
        )
        pack_report.raise_for_errors()
        if pack_report.pack is None or pack_report.sha256 is None:
            raise RuntimeError("validated review pack metadata is unavailable")
        expected_pack_ref = {
            "schema_version": pack_report.pack["schema_version"],
            "pack_id": pack_report.pack["pack_id"],
            "file_sha256": pack_report.sha256,
        }
        if attestation["review_pack"] != expected_pack_ref:
            add(
                "attestation.pack",
                "/review_pack",
                "review pack reference differs from the validated pack",
            )
        registry_path = Path(reviewer_registry_path).expanduser().absolute()
        registry, registry_sha = _load_reviewer_registry(registry_path)
        registry_ref, commit_time = _bind_registry_commit(
            registry_path,
            Path(reviewer_repository_root).expanduser().absolute(),
            attestation["reviewer_registry"]["source_commit"],
            registry,
            registry_sha,
        )
        if attestation["reviewer_registry"] != registry_ref:
            add(
                "attestation.registry",
                "/reviewer_registry",
                "reviewer registry reference differs",
            )
        reviewer = _select_reviewer(registry, attestation["reviewer"]["reviewer_id"])
        if attestation["reviewer"] != reviewer:
            add(
                "attestation.registry",
                "/reviewer",
                "embedded reviewer differs from the registry",
            )
        expected_reviews = _validated_human_rows(
            attestation["reviews"],
            pack_report.pack,
            reviewer_id=reviewer["reviewer_id"],
            registry_declared_at=_parse_utc(
                registry["declared_at_utc"], "registry.declared_at_utc"
            ),
            registry_commit_time=commit_time,
        )
        if attestation["reviews"] != expected_reviews:
            add(
                "attestation.reviews",
                "/reviews",
                "review rows are not canonical",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        add("attestation.source", "/", str(error))

    try:
        signature = attestation["signature"]
        _verify_ssh_signature(
            canonical_json_bytes(payload) + b"\n",
            signature["armored_signature"],
            reviewer_id=attestation["reviewer"]["reviewer_id"],
            public_key=attestation["reviewer"]["ssh_public_key"],
            namespace=signature["namespace"],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        add("attestation.signature", "/signature", str(error))

    try:
        _, final_sha = load_json_object_with_sha256(attestation_path)
        if final_sha != initial_sha:
            add("attestation.file", "/", "attestation changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("attestation.file", "/", str(error))
    all_consistent = all(
        review.get("verdict") == "consistent" for review in attestation["reviews"]
    )
    complete = not issues and all_consistent
    status = "signed" if complete else "review_issue_found" if not issues else "invalid"
    return HumanReviewAttestationReport(
        attestation_path,
        tuple(issues),
        attestation,
        initial_sha,
        status=status,
        complete=complete,
        formal=attestation["formal"] is True,
    )


def _load_reviewer_registry(path: Path) -> tuple[dict[str, Any], str]:
    registry, digest = load_json_object_with_sha256(path)
    if set(registry) != _REGISTRY_KEYS:
        raise ValueError("reviewer registry has the wrong fields")
    if registry["schema_version"] != REVIEWER_REGISTRY_SCHEMA:
        raise ValueError("wrong reviewer registry schema")
    declared = _parse_utc(registry["declared_at_utc"], "declared_at_utc")
    if registry["declared_at_utc"] != declared.isoformat():
        raise ValueError("registry timestamp is not canonical UTC ISO-8601")
    reviewers = registry["reviewers"]
    if not isinstance(reviewers, list) or not reviewers:
        raise ValueError("reviewer registry must contain at least one reviewer")
    identities: set[str] = set()
    for reviewer in reviewers:
        _validate_reviewer(reviewer)
        if reviewer["reviewer_id"] in identities:
            raise ValueError("reviewer registry contains duplicate identities")
        identities.add(reviewer["reviewer_id"])
    identity = {
        "schema_version": registry["schema_version"],
        "declared_at_utc": registry["declared_at_utc"],
        "reviewers": reviewers,
    }
    if registry["registry_id"] != _content_id(identity):
        raise ValueError("reviewer registry identity is wrong")
    return registry, digest


def _select_reviewer(registry: Mapping[str, Any], reviewer_id: str) -> dict[str, Any]:
    matches = [
        reviewer
        for reviewer in registry["reviewers"]
        if reviewer["reviewer_id"] == reviewer_id
    ]
    if len(matches) != 1:
        raise ValueError("selected reviewer is not uniquely registered")
    return dict(matches[0])


def _bind_registry_commit(
    registry_path: Path,
    repository_root: Path,
    commit: str,
    registry: Mapping[str, Any],
    registry_sha: str,
) -> tuple[dict[str, Any], datetime]:
    if not isinstance(commit, str) or _COMMIT_ID.fullmatch(commit) is None:
        raise ValueError("reviewer registry commit must be a full lowercase Git SHA")
    root = repository_root.resolve(strict=True)
    lexical_registry = registry_path.expanduser().absolute()
    try:
        relative = lexical_registry.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("reviewer registry is outside its repository") from error
    validate_relative_path(relative)
    if ":" in relative:
        raise ValueError("reviewer registry path contains a forbidden colon")
    _reject_symlink_components(root, relative)
    current = lexical_registry.resolve(strict=True)
    if current != root / relative:
        raise ValueError("reviewer registry path is not canonical")
    head = _git(root, "rev-parse", "HEAD")
    if _COMMIT_ID.fullmatch(head) is None:
        raise ValueError("cannot resolve repository HEAD")
    ancestor = subprocess.run(
        [
            _GIT_EXECUTABLE,
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            commit,
            head,
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )
    if ancestor.returncode != 0:
        raise ValueError("reviewer registry commit is not an ancestor of HEAD")
    committed = subprocess.run(
        [_GIT_EXECUTABLE, "-C", str(root), "show", f"{commit}:{relative}"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    if committed.returncode != 0:
        raise ValueError("reviewer registry is absent from the declared commit")
    current_bytes = registry_path.read_bytes()
    if (
        committed.stdout != current_bytes
        or sha256_bytes(committed.stdout) != registry_sha
    ):
        raise ValueError("reviewer registry bytes differ from the declared commit")
    committed_registry = json.loads(committed.stdout)
    if committed_registry != registry:
        raise ValueError("committed reviewer registry JSON differs")
    commit_time = _parse_utc(
        _git(root, "show", "-s", "--format=%cI", commit),
        "reviewer registry commit time",
    )
    return (
        {
            "schema_version": REVIEWER_REGISTRY_SCHEMA,
            "registry_id": registry["registry_id"],
            "file_sha256": registry_sha,
            "source_commit": commit,
            "source_path": relative,
        },
        commit_time,
    )


def _validated_human_rows(
    rows: Sequence[Any],
    pack: Mapping[str, Any],
    *,
    reviewer_id: str,
    registry_declared_at: datetime,
    registry_commit_time: datetime,
) -> list[dict[str, Any]]:
    selected = pack["selected_reviews"]
    if not isinstance(rows, list) or len(rows) != len(selected):
        raise ValueError("completed reviews must contain exactly the selected rows")
    if pack["formal"] is True and len(rows) != FORMAL_REVIEW_COUNT:
        raise ValueError("formal human attestation requires exactly 20 reviews")
    completed: list[dict[str, Any]] = []
    for index, (row, review) in enumerate(zip(rows, selected, strict=True)):
        if not isinstance(row, dict) or set(row) != _REVIEW_KEYS:
            raise ValueError(f"human review fields are invalid at row {index}")
        if any(
            row[field] is None
            for field in (
                "reviewer_id",
                "review_started_at_utc",
                "review_completed_at_utc",
                "finding",
                "verdict",
            )
        ):
            raise ValueError(f"human review fields are incomplete at row {index}")
        expected = {
            "schema_version": REVIEW_TRIAL_SCHEMA,
            "manual_review_id": review["manual_review_id"],
            "review_pack_id": pack["pack_id"],
            "manifest_id": review["manifest_id"],
            "split": review["split"],
            "attempt_index": review["attempt_index"],
            "seed": review["seed"],
            "source_relative_path": review["source_relative_path"],
            "source_file_sha256": review["source_file_sha256"],
            "source_num_steps": review["source_num_steps"],
            "classification": review["classification"],
            "media": review["media"],
        }
        for field, value in expected.items():
            if row[field] != value or type(row[field]) is not type(value):
                raise ValueError(
                    f"human review immutable field {field} differs at row {index}"
                )
        if row["reviewer_id"] != reviewer_id:
            raise ValueError(f"human review reviewer differs at row {index}")
        started = _parse_utc(
            row["review_started_at_utc"], f"reviews[{index}].review_started_at_utc"
        )
        completed_at = _parse_utc(
            row["review_completed_at_utc"],
            f"reviews[{index}].review_completed_at_utc",
        )
        if (
            row["review_started_at_utc"] != started.isoformat()
            or row["review_completed_at_utc"] != completed_at.isoformat()
        ):
            raise ValueError(
                f"human review timestamps are not canonical at row {index}"
            )
        if started < registry_declared_at or started < registry_commit_time:
            raise ValueError(
                f"human review started before reviewer predeclaration at row {index}"
            )
        if completed_at < started:
            raise ValueError(f"human review completed before it started at row {index}")
        _require_text(row["finding"], f"reviews[{index}].finding", maximum=2000)
        if row["verdict"] not in {"consistent", "inconsistent"}:
            raise ValueError(f"human review verdict is invalid at row {index}")
        completed.append(dict(row))
    return completed


def _validate_request(request: Mapping[str, Any]) -> None:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        raise ValueError("attestation request has the wrong fields")
    if request["schema_version"] != ATTESTATION_REQUEST_SCHEMA:
        raise ValueError("wrong attestation request schema")
    if request["status"] != "awaiting_signature":
        raise ValueError("attestation request has wrong status")
    if request["signing_namespace"] != ATTESTATION_NAMESPACE:
        raise ValueError("attestation signing namespace differs")
    payload = request["payload"]
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("attestation payload has the wrong fields")
    if request["attestation_id"] != _content_id(payload):
        raise ValueError("attestation request identity is wrong")
    message = request["signing_message"]
    if not isinstance(message, dict) or set(message) != {"path", "sha256"}:
        raise ValueError("signing message reference has the wrong fields")
    if message["path"] != SIGNING_MESSAGE_FILENAME or not _is_sha256(message["sha256"]):
        raise ValueError("signing message reference is invalid")


def _read_exact_signing_message(
    path: Path,
    payload: Mapping[str, Any],
    expected_sha256: str,
) -> bytes:
    content = path.read_bytes()
    expected = canonical_json_bytes(payload) + b"\n"
    if content != expected or sha256_bytes(content) != expected_sha256:
        raise ValueError("signing message differs from the request payload")
    return content


def _read_signature(path: Path) -> str:
    content = path.read_bytes()
    if not content or len(content) > 65536 or b"\x00" in content:
        raise ValueError("SSH signature size or encoding is invalid")
    try:
        signature = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("SSH signature must be ASCII armored") from error
    if not (
        signature.startswith("-----BEGIN SSH SIGNATURE-----\n")
        and signature.rstrip().endswith("-----END SSH SIGNATURE-----")
    ):
        raise ValueError("SSH signature armor is invalid")
    return signature


def _validate_attestation_schema(attestation: Mapping[str, Any], add: Any) -> None:
    expected_keys = _PAYLOAD_KEYS | {"attestation_id", "signature"}
    if not isinstance(attestation, dict) or set(attestation) != expected_keys:
        add("attestation.schema", "/", "attestation has the wrong fields")
        return
    if attestation["schema_version"] != ATTESTATION_SCHEMA:
        add("attestation.schema", "/schema_version", "wrong attestation schema")
    if attestation["status"] != "signed":
        add("attestation.schema", "/status", "attestation status must be signed")
    if type(attestation["formal"]) is not bool:
        add("attestation.schema", "/formal", "formal must be boolean")
    if not _is_content_id(attestation["attestation_id"]):
        add("attestation.schema", "/attestation_id", "attestation ID is invalid")
    pack = attestation["review_pack"]
    if not isinstance(pack, dict) or set(pack) != {
        "schema_version",
        "pack_id",
        "file_sha256",
    }:
        add("attestation.schema", "/review_pack", "pack reference is invalid")
    elif not (
        pack["schema_version"] == REVIEW_PACK_SCHEMA
        and _is_content_id(pack["pack_id"])
        and _is_sha256(pack["file_sha256"])
    ):
        add("attestation.schema", "/review_pack", "pack reference values are invalid")
    registry = attestation["reviewer_registry"]
    if not isinstance(registry, dict) or set(registry) != {
        "schema_version",
        "registry_id",
        "file_sha256",
        "source_commit",
        "source_path",
    }:
        add(
            "attestation.schema",
            "/reviewer_registry",
            "registry reference is invalid",
        )
    else:
        try:
            validate_relative_path(registry["source_path"])
        except (TypeError, ValueError) as error:
            add("attestation.schema", "/reviewer_registry/source_path", str(error))
        if not (
            registry["schema_version"] == REVIEWER_REGISTRY_SCHEMA
            and _is_content_id(registry["registry_id"])
            and _is_sha256(registry["file_sha256"])
            and isinstance(registry["source_commit"], str)
            and _COMMIT_ID.fullmatch(registry["source_commit"]) is not None
        ):
            add(
                "attestation.schema",
                "/reviewer_registry",
                "registry reference values are invalid",
            )
    try:
        _validate_reviewer(attestation["reviewer"])
    except (TypeError, ValueError) as error:
        add("attestation.schema", "/reviewer", str(error))
    reviews = attestation["reviews"]
    if not isinstance(reviews, list) or not reviews:
        add("attestation.schema", "/reviews", "reviews must be a non-empty list")
    elif any(
        not isinstance(review, dict) or set(review) != _REVIEW_KEYS
        for review in reviews
    ):
        add("attestation.schema", "/reviews", "review rows have wrong fields")
    if attestation["formal"] is True and len(reviews) != FORMAL_REVIEW_COUNT:
        add(
            "attestation.formal",
            "/reviews",
            f"formal attestation requires exactly {FORMAL_REVIEW_COUNT} reviews",
        )
    if attestation["attestation_statement"] != ATTESTATION_STATEMENT:
        add(
            "attestation.schema",
            "/attestation_statement",
            "attestation statement differs",
        )
    signature = attestation["signature"]
    if not isinstance(signature, dict) or set(signature) != _SIGNATURE_KEYS:
        add("attestation.schema", "/signature", "signature has the wrong fields")
    elif not (
        signature["algorithm"] == SSH_SIGNATURE_ALGORITHM
        and signature["namespace"] == ATTESTATION_NAMESPACE
        and signature["signer_fingerprint"]
        == attestation["reviewer"].get("ssh_fingerprint")
        and isinstance(signature["armored_signature"], str)
    ):
        add("attestation.schema", "/signature", "signature values are invalid")


def _validate_reviewer(reviewer: Any) -> None:
    if not isinstance(reviewer, dict) or set(reviewer) != _REVIEWER_KEYS:
        raise ValueError("reviewer entry has the wrong fields")
    _require_reviewer_id(reviewer["reviewer_id"])
    _require_text(reviewer["display_name"], "display_name", maximum=256)
    key_type, key_data = _parse_ssh_public_key(reviewer["ssh_public_key"])
    if reviewer["ssh_public_key"] != f"{key_type} {key_data}":
        raise ValueError("SSH public key must omit comments and extra whitespace")
    if reviewer["ssh_fingerprint"] != _ssh_fingerprint(key_data):
        raise ValueError("SSH public-key fingerprint is wrong")


def _verify_ssh_signature(
    message: bytes,
    signature: str,
    *,
    reviewer_id: str,
    public_key: str,
    namespace: str,
) -> None:
    _require_reviewer_id(reviewer_id)
    if namespace != ATTESTATION_NAMESPACE:
        raise ValueError("wrong SSH signature namespace")
    key_type, key_data = _parse_ssh_public_key(public_key)
    executable = Path("/usr/bin/ssh-keygen")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("ssh-keygen is required to verify the human signature")
    temporary_parent = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
        root = Path(directory)
        allowed_signers = root / "allowed_signers"
        signature_path = root / "attestation.sig"
        allowed_signers.write_text(
            f"{reviewer_id} {key_type} {key_data}\n",
            encoding="ascii",
        )
        signature_path.write_text(signature, encoding="ascii")
        os.chmod(allowed_signers, 0o600)
        os.chmod(signature_path, 0o600)
        result = subprocess.run(
            [
                str(executable),
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                reviewer_id,
                "-n",
                namespace,
                "-s",
                str(signature_path),
            ],
            input=message,
            check=False,
            capture_output=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"SSH signature verification failed: {stderr[:400]}")


def _parse_ssh_public_key(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ValueError("SSH public key must be one line")
    fields = value.strip().split()
    if len(fields) < 2:
        raise ValueError("SSH public key is incomplete")
    key_type, key_data = fields[:2]
    if key_type not in {
        "ssh-ed25519",
        "sk-ssh-ed25519@openssh.com",
        "rsa-sha2-512",
        "rsa-sha2-256",
        "ssh-rsa",
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
    }:
        raise ValueError("SSH public-key algorithm is not allowed")
    try:
        decoded = base64.b64decode(key_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("SSH public-key payload is not valid base64") from error
    if len(decoded) < 32 or len(decoded) > 16384:
        raise ValueError("SSH public-key payload size is invalid")
    return key_type, key_data


def _ssh_fingerprint(key_data: str) -> str:
    decoded = base64.b64decode(key_data, validate=True)
    digest = base64.b64encode(hashlib.sha256(decoded).digest())
    return f"SHA256:{digest.decode('ascii').rstrip('=')}"


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC")
    return parsed.astimezone(UTC)


def _require_reviewer_id(value: Any) -> None:
    if not isinstance(value, str) or _REVIEWER_ID.fullmatch(value) is None:
        raise ValueError("reviewer_id has invalid characters or length")


def _require_text(value: Any, label: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be non-empty canonical text")


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


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        [_GIT_EXECUTABLE, "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise ValueError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def _reject_symlink_components(root: Path, relative_path: str) -> None:
    current = root
    for component in validate_relative_path(relative_path).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("reviewer registry path contains a symbolic link")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("registry", help="publish a public reviewer entry")
    registry.add_argument("--output", required=True, type=Path)
    registry.add_argument("--reviewer-id", required=True)
    registry.add_argument("--display-name", required=True)
    registry.add_argument("--ssh-public-key-file", required=True, type=Path)
    registry.add_argument("--declared-at-utc", required=True)

    prepare = commands.add_parser(
        "prepare", help="publish exact bytes for human signing"
    )
    prepare.add_argument("--review-pack", required=True, type=Path)
    prepare.add_argument("--completed-reviews", required=True, type=Path)
    prepare.add_argument("--reviewer-registry", required=True, type=Path)
    prepare.add_argument("--reviewer-repository-root", required=True, type=Path)
    prepare.add_argument("--reviewer-registry-commit", required=True)
    prepare.add_argument("--reviewer-id", required=True)
    prepare.add_argument("--train-manifest", required=True, type=Path)
    prepare.add_argument("--validation-manifest", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--smoke", action="store_true")

    finalize = commands.add_parser(
        "finalize", help="verify an SSH signature and publish the attestation"
    )
    finalize.add_argument("--request", required=True, type=Path)
    finalize.add_argument("--signature", required=True, type=Path)
    finalize.add_argument("--output", required=True, type=Path)
    _add_validation_paths(finalize)

    validate = commands.add_parser("validate", help="validate a signed attestation")
    validate.add_argument("--attestation", required=True, type=Path)
    _add_validation_paths(validate)
    return parser


def _add_validation_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--review-pack", required=True, type=Path)
    parser.add_argument("--reviewer-registry", required=True, type=Path)
    parser.add_argument("--reviewer-repository-root", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "registry":
            result = create_reviewer_registry(
                args.output,
                reviewer_id=args.reviewer_id,
                display_name=args.display_name,
                ssh_public_key=args.ssh_public_key_file.read_text(encoding="utf-8"),
                declared_at_utc=args.declared_at_utc,
            )
        elif args.command == "prepare":
            result = create_human_review_attestation_request(
                HumanReviewAttestationConfig(
                    review_pack_path=args.review_pack,
                    completed_reviews_path=args.completed_reviews,
                    reviewer_registry_path=args.reviewer_registry,
                    reviewer_repository_root=args.reviewer_repository_root,
                    reviewer_registry_commit=args.reviewer_registry_commit,
                    reviewer_id=args.reviewer_id,
                    train_manifest_path=args.train_manifest,
                    validation_manifest_path=args.validation_manifest,
                    output_dir=args.output_dir,
                    smoke=args.smoke,
                )
            )
        elif args.command == "finalize":
            result = finalize_human_review_attestation(
                args.request,
                signature_path=args.signature,
                output_path=args.output,
                review_pack_path=args.review_pack,
                reviewer_registry_path=args.reviewer_registry,
                reviewer_repository_root=args.reviewer_repository_root,
                train_manifest_path=args.train_manifest,
                validation_manifest_path=args.validation_manifest,
            )
        else:
            report = validate_human_review_attestation(
                args.attestation,
                review_pack_path=args.review_pack,
                reviewer_registry_path=args.reviewer_registry,
                reviewer_repository_root=args.reviewer_repository_root,
                train_manifest_path=args.train_manifest,
                validation_manifest_path=args.validation_manifest,
            )
            result = {
                "valid": report.valid,
                "complete": report.complete,
                "formal": report.formal,
                "status": report.status,
                "errors": [str(error) for error in report.errors],
            }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report.valid and report.complete else 2
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ATTESTATION_NAMESPACE",
    "HumanReviewAttestationConfig",
    "HumanReviewAttestationIssue",
    "HumanReviewAttestationReport",
    "create_human_review_attestation_request",
    "create_reviewer_registry",
    "finalize_human_review_attestation",
    "validate_human_review_attestation",
]


if __name__ == "__main__":
    raise SystemExit(main())
