"""Fail-closed collection manifests and safe episode file consumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import h5py
import mujoco
import numpy as np

from env.pick_place import TaskConfig
from expert.scripted import ExpertConfig

from .hdf5 import (
    EpisodeMetadata,
    ValidationReport,
    _metadata_from_handle,
    _validate_episode_handle,
)

COLLECTION_SCHEMA = "m2-collection-manifest.v1"
LEDGER_SCHEMA = "m2-attempt-ledger.v1"
SPLIT_PROTOCOL = "m3-m4-seed-protocol.v2"
ENVIRONMENT_SCHEMA = "m2-environment.v1"
COMPILED_MODEL_SCHEMA = "mujoco-compiled-model.v1"
MANIFEST_FILENAME = "manifest.json"
LEDGER_FILENAME = "attempts.jsonl"
EPISODES_ROOT = "episodes"
MAX_CONSUMER_EPISODE_STEPS = 10_000
VALIDATION_REPORT_SCHEMA = "m2-collection-validation-report.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SPLITS: dict[str, dict[str, int]] = {
    "train": {
        "candidate_seed_min": 0,
        "candidate_seed_max": 999,
        "scan_start": 0,
        "formal_target_successes": 200,
    },
    "validation": {
        "candidate_seed_min": 1000,
        "candidate_seed_max": 9999,
        "scan_start": 1000,
        "formal_target_successes": 40,
    },
}
_ATTEMPT_KEYS = {
    "attempt_index",
    "seed",
    "status",
    "success",
    "failure_stage",
    "path",
    "num_steps",
    "sha256",
    "error",
}
_ELIGIBLE_KEYS = {"seed", "path", "num_steps", "sha256"}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "split_protocol",
    "split",
    "formal",
    "target_successes",
    "attempt_count",
    "success_count",
    "attempts",
    "eligible_successes",
    "controller",
    "git",
    "assets",
    "environment",
    "generated_at",
    "cli_config",
    "ledger_path",
    "episodes_root",
}
_GIT_KEYS = {
    "commit",
    "tracked_worktree_clean",
    "worktree_clean",
    "source_provenance_clean",
    "provenance_complete",
    "worktree_status_sha256",
    "untracked_file_count",
    "untracked_paths_sha256",
    "relevant_untracked_files",
}
_GIT_FILE_KEYS = {"git_status", "path", "sha256", "size_bytes"}
_TASK_CONFIG_KEYS = set(asdict(TaskConfig()))
_CONTROLLER_DEFAULTS = asdict(ExpertConfig())
_COMPILED_MODEL_KEYS = {
    "fingerprint_schema",
    "mjb_sha256",
    "fingerprint_sha256",
    "nq",
    "nv",
    "nu",
    "nbody",
    "ngeom",
    "nsite",
    "policy_sites",
}


@dataclass(frozen=True)
class CollectionValidationIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class CollectionValidationReport:
    path: Path
    errors: tuple[CollectionValidationIssue, ...]
    manifest: dict[str, Any] | None = None
    sha256: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise CollectionManifestError(self)


class CollectionManifestError(ValueError):
    def __init__(self, report: CollectionValidationReport):
        self.report = report
        super().__init__(
            f"invalid collection manifest {report.path}:\n{report.format_errors()}"
        )


class UnsafeCollectionPath(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class EpisodeDigestMismatch(ValueError):
    """The anchored episode bytes do not match the frozen manifest digest."""

    def __init__(self, expected: str, observed: str):
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"episode SHA-256 mismatch: expected {expected}, observed {observed}"
        )


class EpisodeSizeLimitError(ValueError):
    """An episode declares more transitions than its controller can produce."""


class AtomicPublicationError(RuntimeError):
    """A JSON target exists but publication durability is indeterminate."""

    def __init__(
        self,
        *,
        target_path: Path,
        partial_path: Path,
        target_matches_source: bool,
        target_valid: bool,
        target_sha256: str | None,
    ):
        self.target_path = target_path
        self.partial_path = partial_path
        self.published = True
        self.state = "publication_indeterminate"
        self.target_matches_source = target_matches_source
        self.target_valid = target_valid
        self.target_sha256 = target_sha256
        super().__init__(
            "JSON target was linked but publication durability is indeterminate; "
            f"reconcile {target_path} before retrying"
        )


class LedgerIntegrityError(RuntimeError):
    """A durable JSONL ledger path or inode changed between appends."""


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True)
class LedgerState:
    snapshot: FileSnapshot
    sha256: str


@dataclass(frozen=True)
class EpisodeInspection:
    path: Path
    sha256: str
    snapshot: FileSnapshot
    validation: ValidationReport
    metadata: EpisodeMetadata | None


@dataclass(frozen=True)
class VerifiedEpisode:
    """One validated episode kept open on its original anchored file handle."""

    path: Path
    sha256: str
    snapshot: FileSnapshot
    validation: ValidationReport
    metadata: EpisodeMetadata | None
    handle: h5py.File


def split_contract(split: str) -> dict[str, int | str]:
    if split not in _SPLITS:
        raise ValueError("split must be 'train' or 'validation'")
    contract = _SPLITS[split]
    return {
        "name": SPLIT_PROTOCOL,
        "candidate_seed_min": contract["candidate_seed_min"],
        "candidate_seed_max": contract["candidate_seed_max"],
        "scan_start": contract["scan_start"],
        "reserved_seed_min": 10000,
    }


def formal_target_successes(split: str) -> int:
    if split not in _SPLITS:
        raise ValueError("split must be 'train' or 'validation'")
    return _SPLITS[split]["formal_target_successes"]


def maximum_episode_steps(controller: Mapping[str, Any]) -> int:
    """Return the exact maximum number of callbacks reachable by run_episode."""

    required = (
        "max_move_steps",
        "max_attempts",
        "close_steps",
        "open_steps",
        "settle_steps",
        "recovery_wait_steps",
    )
    values: dict[str, int] = {}
    for name in required:
        value = controller.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"controller {name} must be a positive integer")
        values[name] = value
    move = values["max_move_steps"]
    attempts = values["max_attempts"]
    return (
        attempts * (4 * move + values["close_steps"])
        + (attempts - 1) * values["recovery_wait_steps"]
        + 2 * move
        + values["open_steps"]
        + values["settle_steps"]
    )


def validate_split_seed(split: str, seed: int) -> None:
    if split not in _SPLITS:
        raise ValueError("split must be 'train' or 'validation'")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    contract = _SPLITS[split]
    if not contract["candidate_seed_min"] <= seed <= contract["candidate_seed_max"]:
        raise ValueError(
            f"seed {seed} is outside the {split} M2 namespace "
            f"{contract['candidate_seed_min']}..{contract['candidate_seed_max']}"
        )
    if seed >= 10000:
        raise ValueError("M2 collection rejects every seed >=10000")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def initialize_jsonl_no_clobber(path: Path) -> LedgerState:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        os.fsync(descriptor)
        snapshot = _snapshot_from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(snapshot.mode):
            raise LedgerIntegrityError("JSONL ledger must be a regular file")
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _regular_path_snapshot(path) != snapshot:
        raise LedgerIntegrityError("JSONL ledger changed during initialization")
    return LedgerState(snapshot=snapshot, sha256=hashlib.sha256(b"").hexdigest())


def append_jsonl_fsync(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_snapshot: LedgerState,
) -> LedgerState:
    if not isinstance(expected_snapshot, LedgerState):
        raise TypeError("expected_snapshot must be a LedgerState")
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    flags = os.O_RDWR | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LedgerIntegrityError("cannot open the anchored JSONL ledger") from error
    try:
        before = _snapshot_from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before != expected_snapshot.snapshot:
            raise LedgerIntegrityError("JSONL ledger changed before append")
        if _regular_path_snapshot(Path(path)) != before:
            raise LedgerIntegrityError("JSONL ledger path changed before append")
        before_digest, expected_digest = _descriptor_digest_with_suffix(
            descriptor, encoded
        )
        if before_digest != expected_snapshot.sha256:
            raise LedgerIntegrityError("JSONL ledger content changed before append")

        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSONL append")
            view = view[written:]
        os.fsync(descriptor)
        after = _snapshot_from_stat(os.fstat(descriptor))
        after_digest = _sha256_descriptor(descriptor)
        if (
            after.identity != before.identity
            or not stat.S_ISREG(after.mode)
            or after.size != before.size + len(encoded)
            or after_digest != expected_digest
        ):
            raise LedgerIntegrityError("JSONL ledger changed during append")
        if _regular_path_snapshot(Path(path)) != after:
            raise LedgerIntegrityError("JSONL ledger path changed during append")
        return LedgerState(snapshot=after, sha256=after_digest)
    finally:
        os.close(descriptor)


def atomic_write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"file already exists: {path}")
    partial = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    descriptor: int | None = None
    target_descriptor: int | None = None
    linked = False
    expected_sha256 = sha256_bytes(encoded)
    try:
        descriptor = os.open(
            partial,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSON write")
            view = view[written:]
        os.fsync(descriptor)
        validated_snapshot = _snapshot_from_stat(os.fstat(descriptor))
        if (
            not stat.S_ISREG(validated_snapshot.mode)
            or _sha256_descriptor(descriptor) != expected_sha256
            or _snapshot_from_stat(os.fstat(descriptor)) != validated_snapshot
            or _regular_path_snapshot(partial) != validated_snapshot
        ):
            raise OSError("JSON partial changed before publication")
        os.link(partial, path)
        linked = True
        target_descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        linked_snapshot = _snapshot_from_stat(os.fstat(target_descriptor))
        if (
            not _same_file_content_identity(validated_snapshot, linked_snapshot)
            or _sha256_descriptor(target_descriptor) != expected_sha256
            or _snapshot_from_stat(os.fstat(target_descriptor)) != linked_snapshot
            or _regular_path_snapshot(path) != linked_snapshot
            or _regular_path_snapshot(partial) != linked_snapshot
        ):
            raise OSError("linked JSON target differs from the validated partial")

        partial.unlink()
        unlinked_snapshot = _snapshot_from_stat(os.fstat(target_descriptor))
        if (
            not _same_file_content_identity(linked_snapshot, unlinked_snapshot)
            or _sha256_descriptor(target_descriptor) != expected_sha256
            or _snapshot_from_stat(os.fstat(target_descriptor)) != unlinked_snapshot
            or _regular_path_snapshot(path) != unlinked_snapshot
            or partial.exists()
            or partial.is_symlink()
        ):
            raise OSError("JSON target changed while removing the partial")

        _fsync_directory(path.parent)
        durable_snapshot = _snapshot_from_stat(os.fstat(target_descriptor))
        if (
            durable_snapshot != unlinked_snapshot
            or _sha256_descriptor(target_descriptor) != expected_sha256
            or _snapshot_from_stat(os.fstat(target_descriptor)) != durable_snapshot
            or _regular_path_snapshot(path) != durable_snapshot
        ):
            raise OSError("JSON target changed before durable publication completed")
        return path
    except BaseException as error:
        if linked:
            target_sha256, target_valid = _json_target_evidence(path)
            raise AtomicPublicationError(
                target_path=path,
                partial_path=partial,
                target_matches_source=target_sha256 == expected_sha256,
                target_valid=target_valid,
                target_sha256=target_sha256,
            ) from error
        if isinstance(error, FileExistsError):
            raise FileExistsError(f"file already exists: {path}") from error
        raise
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        if not linked:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass


def _regular_path_snapshot(path: Path) -> FileSnapshot:
    try:
        status = Path(path).lstat()
    except OSError as error:
        raise LedgerIntegrityError(f"regular file is unavailable: {path}") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LedgerIntegrityError(f"path is not a non-symlink regular file: {path}")
    return _snapshot_from_stat(status)


def _same_file_content_identity(first: FileSnapshot, second: FileSnapshot) -> bool:
    return (
        first.device,
        first.inode,
        first.mode,
        first.size,
        first.mtime_ns,
    ) == (
        second.device,
        second.inode,
        second.mode,
        second.size,
        second.mtime_ns,
    )


def _descriptor_digest_with_suffix(descriptor: int, suffix: bytes) -> tuple[str, str]:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    before = digest.hexdigest()
    digest.update(suffix)
    return before, digest.hexdigest()


def _json_target_evidence(path: Path) -> tuple[str | None, bool]:
    try:
        content = _stable_read_absolute(path)
    except (OSError, ValueError):
        return None, False
    digest = sha256_bytes(content)
    try:
        payload = _strict_json_loads(content.decode("utf-8"))
    except (UnicodeError, ValueError):
        return digest, False
    return digest, isinstance(payload, dict)


def load_collection_manifest(path: Path) -> dict[str, Any]:
    payload, _ = load_json_object_with_sha256(path)
    return payload


def load_json_object_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    content = _stable_read_absolute(Path(path))
    payload = _strict_json_loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload, sha256_bytes(content)


def load_jsonl_relative(root: Path, relative_path: str) -> list[Any]:
    records, _, _, _ = load_jsonl_relative_with_sha256(root, relative_path)
    return records


def load_jsonl_relative_with_sha256(
    root: Path,
    relative_path: str,
) -> tuple[list[Any], str, FileSnapshot, Path]:
    content, snapshot, path = stable_read_relative_file(root, relative_path)
    records = _strict_jsonl_loads(content.decode("utf-8"))
    return records, sha256_bytes(content), snapshot, path


def manifest_sha256(path: Path) -> str:
    return sha256_bytes(_stable_read_absolute(Path(path)))


def fsync_directory(path: Path) -> None:
    _fsync_directory(Path(path))


def compiled_model_fingerprint(env: Any) -> dict[str, Any]:
    model = env.model
    # ``PickPlace.reset`` intentionally randomizes the static box body's
    # position in ``model.body_pos`` for every episode.  That is episode
    # provenance (and is recorded in the reset receipt), not a change to the
    # compiled scene.  Canonicalize that one runtime placement while saving
    # the model so formal collection can re-check the same static fingerprint
    # after an episode has run.
    dynamic_body_ids: list[int] = []
    box_id = getattr(env, "bid_box", None)
    if isinstance(box_id, (int, np.integer)) and 0 <= int(box_id) < int(model.nbody):
        dynamic_body_ids.append(int(box_id))
    original_body_positions = {
        body_id: np.asarray(model.body_pos[body_id], dtype=np.float64).copy()
        for body_id in dynamic_body_ids
    }
    try:
        for body_id in dynamic_body_ids:
            model.body_pos[body_id] = 0.0
        buffer = np.empty(int(model.nbuffer), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        site_names = [model.site(index).name for index in range(model.nsite)]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("cannot fingerprint the compiled MuJoCo model") from error
    finally:
        for body_id, position in original_body_positions.items():
            model.body_pos[body_id] = position
    policy_sites: list[dict[str, Any]] = []
    for name in ("flange", "tcp"):
        if site_names.count(name) != 1:
            raise ValueError("compiled model must contain one flange and one tcp site")
        rgba = np.asarray(model.site(name).rgba, dtype=np.float64)
        if rgba.shape != (4,) or not np.all(np.isfinite(rgba)) or rgba[3] != 0.0:
            raise ValueError("compiled policy sites must have finite invisible RGBA")
        policy_sites.append({"name": name, "rgba": rgba.tolist()})
    base: dict[str, Any] = {
        "fingerprint_schema": COMPILED_MODEL_SCHEMA,
        "mjb_sha256": hashlib.sha256(buffer).hexdigest(),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "policy_sites": policy_sites,
    }
    return {
        **base,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(base)),
    }


def environment_payload(env: Any, compiled_model: Mapping[str, Any]) -> dict[str, Any]:
    task_config = asdict(env.cfg)
    task_config["colors"] = list(task_config["colors"])
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "mujoco_version": mujoco.__version__,
        "task_config": task_config,
        "debug_viz": env.cfg.debug_viz,
        "compiled_model": dict(compiled_model),
    }


def assets_payload(provenance: Any) -> dict[str, Any]:
    return {
        "summary": provenance.summary_dict(),
        "content_manifest": provenance.manifest_dict(),
        "aggregate_manifest_sha256": provenance.aggregate_manifest_sha256,
    }


def validate_collection_manifest(path: Path) -> CollectionValidationReport:
    path = Path(path)
    errors: list[CollectionValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(CollectionValidationIssue(code, location, message))

    try:
        manifest, manifest_digest = load_json_object_with_sha256(path)
    except (OSError, UnicodeError, ValueError) as error:
        add("manifest.json", "/", f"cannot read strict manifest JSON: {error}")
        return CollectionValidationReport(path, tuple(errors), None, None)

    _validate_manifest_schema(manifest, add)
    if errors:
        return CollectionValidationReport(
            path, tuple(errors), manifest, manifest_digest
        )

    run_root = path.parent
    attempts = manifest["attempts"]
    expected_eligible = [
        {
            "seed": attempt["seed"],
            "path": attempt["path"],
            "num_steps": attempt["num_steps"],
            "sha256": attempt["sha256"],
        }
        for attempt in attempts
        if attempt["success"]
    ]
    if manifest["eligible_successes"] != expected_eligible:
        add(
            "manifest.eligible",
            "/eligible_successes",
            "eligible_successes must exactly equal successful attempts",
        )

    try:
        ledger_content, _, _ = stable_read_relative_file(
            run_root,
            manifest["ledger_path"],
        )
        ledger = _strict_jsonl_loads(ledger_content.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, UnsafeCollectionPath) as error:
        code = error.code if isinstance(error, UnsafeCollectionPath) else "ledger.read"
        add(code, f"/{manifest['ledger_path']}", f"cannot read ledger: {error}")
    else:
        if ledger != attempts:
            add(
                "ledger.mismatch",
                f"/{manifest['ledger_path']}",
                "ledger records must agree one-for-one with manifest attempts",
            )

    seen_paths: set[str] = set()
    seen_inodes: dict[tuple[int, int], str] = {}
    episode_step_limit = min(
        maximum_episode_steps(manifest["controller"]),
        MAX_CONSUMER_EPISODE_STEPS,
    )
    for index, attempt in enumerate(attempts):
        location = f"/attempts/{index}"
        relative_path = attempt["path"]
        if relative_path in seen_paths:
            add("path.duplicate", f"{location}/path", "episode path is duplicated")
            continue
        seen_paths.add(relative_path)
        try:
            inspection = inspect_episode(
                run_root,
                relative_path,
                expected_sha256=attempt["sha256"],
                max_num_steps=episode_step_limit,
            )
        except UnsafeCollectionPath as error:
            add(error.code, f"{location}/path", str(error))
            continue
        except EpisodeDigestMismatch as error:
            add("episode.sha256", f"{location}/sha256", str(error))
            continue
        except EpisodeSizeLimitError as error:
            add("episode.num_steps_limit", location, str(error))
            continue
        except (OSError, RuntimeError, ValueError) as error:
            add("episode.read", f"{location}/path", f"cannot consume episode: {error}")
            continue

        first_path = seen_inodes.get(inspection.snapshot.identity)
        if first_path is not None:
            add(
                "path.inode_duplicate",
                f"{location}/path",
                f"episode inode is already named by {first_path}",
            )
        else:
            seen_inodes[inspection.snapshot.identity] = relative_path
        if inspection.sha256 != attempt["sha256"]:
            add(
                "episode.sha256",
                f"{location}/sha256",
                "episode SHA-256 does not match the manifest",
            )
        for issue in inspection.validation.errors:
            add(
                f"episode.schema.{issue.code}",
                f"{location}{issue.location}",
                issue.message,
            )
        metadata = inspection.metadata
        if metadata is None:
            continue
        if metadata.seed != attempt["seed"]:
            add("episode.seed", location, "HDF5 seed does not match the manifest")
        if metadata.success is not attempt["success"]:
            add("episode.outcome", location, "HDF5 outcome does not match the manifest")
        if metadata.failure_stage != attempt["failure_stage"]:
            add(
                "episode.failure_stage",
                location,
                "HDF5 failure_stage does not match the manifest",
            )
        if metadata.num_steps != attempt["num_steps"]:
            add("episode.num_steps", location, "HDF5 num_steps does not match")

    return CollectionValidationReport(path, tuple(errors), manifest, manifest_digest)


def validate_manifest_pair(
    first_path: Path,
    second_path: Path,
) -> CollectionValidationReport:
    first = validate_collection_manifest(first_path)
    second = validate_collection_manifest(second_path)
    errors: list[CollectionValidationIssue] = []
    errors.extend(
        CollectionValidationIssue(error.code, f"/first{error.location}", error.message)
        for error in first.errors
    )
    errors.extend(
        CollectionValidationIssue(error.code, f"/second{error.location}", error.message)
        for error in second.errors
    )
    manifests = [first.manifest, second.manifest]
    if not errors and all(manifest is not None for manifest in manifests):
        typed = [manifest for manifest in manifests if manifest is not None]
        if {manifest["split"] for manifest in typed} != {"train", "validation"}:
            errors.append(
                CollectionValidationIssue(
                    "pair.splits",
                    "/",
                    "pair must contain exactly one train and one validation manifest",
                )
            )
        first_seeds = {attempt["seed"] for attempt in typed[0]["attempts"]}
        second_seeds = {attempt["seed"] for attempt in typed[1]["attempts"]}
        overlap = sorted(first_seeds.intersection(second_seeds))
        if overlap:
            errors.append(
                CollectionValidationIssue(
                    "pair.seed_overlap",
                    "/",
                    f"train/validation attempted seeds overlap: {overlap}",
                )
            )
    return CollectionValidationReport(
        Path(first_path),
        tuple(errors),
        first.manifest,
        first.sha256,
    )


def inspect_episode(
    run_root: Path,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
    max_num_steps: int = MAX_CONSUMER_EPISODE_STEPS,
) -> EpisodeInspection:
    with open_verified_episode(
        run_root,
        relative_path,
        expected_sha256=expected_sha256,
        max_num_steps=max_num_steps,
    ) as episode:
        return EpisodeInspection(
            path=episode.path,
            sha256=episode.sha256,
            snapshot=episode.snapshot,
            validation=episode.validation,
            metadata=episode.metadata,
        )


@contextmanager
def open_verified_episode(
    run_root: Path,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
    max_num_steps: int = MAX_CONSUMER_EPISODE_STEPS,
) -> Iterator[VerifiedEpisode]:
    """Open, validate, consume, and recheck one immutable anchored HDF5 inode."""

    if type(max_num_steps) is not int or max_num_steps <= 0:
        raise ValueError("max_num_steps must be a positive integer")
    pure = validate_relative_path(relative_path)
    root = Path(run_root)
    path, lexical_snapshot = _safe_regular_path(root, relative_path)
    directory_snapshots = _directory_chain_snapshots(root, pure)
    descriptor = _open_anchored_regular(root, pure)
    stream: Any | None = None
    handle: h5py.File | None = None
    before_digest: str | None = None
    try:
        before_snapshot = _snapshot_from_stat(os.fstat(descriptor))
        if before_snapshot != lexical_snapshot:
            raise UnsafeCollectionPath(
                "path.changed",
                "episode changed before its anchored handle was consumed",
            )
        before_digest = _sha256_descriptor(descriptor)
        if expected_sha256 is not None and before_digest != expected_sha256:
            raise EpisodeDigestMismatch(expected_sha256, before_digest)

        stream = os.fdopen(os.dup(descriptor), "rb", buffering=0)
        handle = h5py.File(stream, "r")
        _preflight_episode_size(handle, max_num_steps=max_num_steps)
        validation = _validate_episode_handle(handle, path)
        metadata = _metadata_from_handle(handle) if validation.valid else None
        yield VerifiedEpisode(
            path=path,
            sha256=before_digest,
            snapshot=before_snapshot,
            validation=validation,
            metadata=metadata,
            handle=handle,
        )
    finally:
        if handle is not None:
            handle.close()
        if stream is not None:
            stream.close()
        try:
            after_snapshot = _snapshot_from_stat(os.fstat(descriptor))
            after_digest = _sha256_descriptor(descriptor)
        finally:
            os.close(descriptor)
        final_path, final_snapshot = _safe_regular_path(root, relative_path)
        final_directories = _directory_chain_snapshots(root, pure)
        if (
            final_path != path
            or after_snapshot != lexical_snapshot
            or (before_digest is not None and after_digest != before_digest)
            or final_snapshot != lexical_snapshot
            or final_directories != directory_snapshots
        ):
            raise RuntimeError("episode changed while its anchored handle was consumed")


def _preflight_episode_size(handle: h5py.File, *, max_num_steps: int) -> None:
    raw_num_steps = handle.attrs.get("num_steps")
    if isinstance(raw_num_steps, (int, np.integer)) and not isinstance(
        raw_num_steps, (bool, np.bool_)
    ):
        if int(raw_num_steps) > max_num_steps:
            raise EpisodeSizeLimitError(
                f"declared num_steps {int(raw_num_steps)} exceeds limit {max_num_steps}"
            )
    for key in (
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
        "action",
        "timestamp",
        "stage",
    ):
        try:
            dataset = handle.get(key)
            rows = dataset.shape[0] if isinstance(dataset, h5py.Dataset) else None
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if type(rows) is int and rows > max_num_steps:
            raise EpisodeSizeLimitError(
                f"dataset {key!r} length {rows} exceeds limit {max_num_steps}"
            )


def stable_sha256_relative_file(
    root: Path,
    relative_path: str,
) -> tuple[str, FileSnapshot, Path]:
    path, lexical_snapshot = _safe_regular_path(root, relative_path)
    descriptor = _open_anchored_regular(
        Path(root), validate_relative_path(relative_path)
    )
    digest = hashlib.sha256()
    try:
        before = _snapshot_from_stat(os.fstat(descriptor))
        if before != lexical_snapshot:
            raise UnsafeCollectionPath("path.changed", "file changed before hashing")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = _snapshot_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    final_path, final_snapshot = _safe_regular_path(root, relative_path)
    if path != final_path or before != after or after != final_snapshot:
        raise UnsafeCollectionPath("path.changed", "file changed while hashing")
    return digest.hexdigest(), after, path


def stable_read_relative_file(
    root: Path,
    relative_path: str,
) -> tuple[bytes, FileSnapshot, Path]:
    path, lexical_snapshot = _safe_regular_path(root, relative_path)
    descriptor = _open_anchored_regular(
        Path(root), validate_relative_path(relative_path)
    )
    chunks: list[bytes] = []
    try:
        before = _snapshot_from_stat(os.fstat(descriptor))
        if before != lexical_snapshot:
            raise UnsafeCollectionPath("path.changed", "file changed before reading")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = _snapshot_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    final_path, final_snapshot = _safe_regular_path(root, relative_path)
    if path != final_path or before != after or after != final_snapshot:
        raise UnsafeCollectionPath("path.changed", "file changed while reading")
    return b"".join(chunks), after, path


def validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise UnsafeCollectionPath("path.canonical", "path must be a non-empty string")
    if "\x00" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise UnsafeCollectionPath(
            "path.canonical", "path contains forbidden characters"
        )
    if value.startswith("/") or "//" in value:
        raise UnsafeCollectionPath(
            "path.canonical", "path must be canonical and relative"
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeCollectionPath(
            "path.canonical", "path contains an unsafe component"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value:
        raise UnsafeCollectionPath("path.canonical", "path is not canonical POSIX")
    return pure


def _validate_manifest_schema(manifest: dict[str, Any], add: Any) -> None:
    _exact_keys(manifest, _TOP_LEVEL_KEYS, "/", add)
    if set(manifest) != _TOP_LEVEL_KEYS:
        return
    if manifest["schema_version"] != COLLECTION_SCHEMA:
        add("manifest.value", "/schema_version", f"expected {COLLECTION_SCHEMA}")
    split = manifest["split"]
    if not isinstance(split, str) or split not in _SPLITS:
        add("manifest.value", "/split", "split must be train or validation")
        return
    expected_protocol = split_contract(split)
    protocol = manifest["split_protocol"]
    if not isinstance(protocol, dict):
        add("manifest.type", "/split_protocol", "split protocol must be an object")
    else:
        _exact_keys(protocol, set(expected_protocol), "/split_protocol", add)
        if set(protocol) == set(expected_protocol):
            if protocol["name"] != SPLIT_PROTOCOL:
                add(
                    "manifest.value",
                    "/split_protocol/name",
                    f"expected {SPLIT_PROTOCOL}",
                )
            for name in (
                "candidate_seed_min",
                "candidate_seed_max",
                "scan_start",
                "reserved_seed_min",
            ):
                _require_int(protocol[name], f"/split_protocol/{name}", add, minimum=0)
    if protocol != expected_protocol:
        add(
            "manifest.value",
            "/split_protocol",
            "split protocol must exactly match m3-m4-seed-protocol.v2",
        )
    _require_bool(manifest["formal"], "/formal", add)
    for name in ("target_successes", "attempt_count", "success_count"):
        _require_int(manifest[name], f"/{name}", add, minimum=1)
    if not isinstance(manifest["attempts"], list):
        add("manifest.type", "/attempts", "attempts must be an array")
        return
    if not isinstance(manifest["eligible_successes"], list):
        add(
            "manifest.type",
            "/eligible_successes",
            "eligible_successes must be an array",
        )
        return
    if not isinstance(manifest["ledger_path"], str):
        add("manifest.type", "/ledger_path", "ledger_path must be a string")
    elif manifest["ledger_path"] != LEDGER_FILENAME:
        add("manifest.value", "/ledger_path", f"expected {LEDGER_FILENAME}")
    if not isinstance(manifest["episodes_root"], str):
        add("manifest.type", "/episodes_root", "episodes_root must be a string")
    elif manifest["episodes_root"] != EPISODES_ROOT:
        add("manifest.value", "/episodes_root", f"expected {EPISODES_ROOT}")
    _validate_timestamp(manifest["generated_at"], "/generated_at", add)
    _validate_controller(manifest["controller"], add)
    _validate_git(manifest["git"], add)
    _validate_assets(manifest["assets"], add)
    _validate_environment(manifest["environment"], add)
    _validate_cli_config(manifest["cli_config"], split, add)

    try:
        episode_step_limit = maximum_episode_steps(manifest["controller"])
    except (AttributeError, TypeError, ValueError):
        episode_step_limit = 0
    else:
        if episode_step_limit > MAX_CONSUMER_EPISODE_STEPS:
            add(
                "manifest.controller_limit",
                "/controller",
                "controller permits more transitions than the consumer hard limit",
            )
            episode_step_limit = MAX_CONSUMER_EPISODE_STEPS

    contract = _SPLITS[split]
    seeds: list[int] = []
    successes = 0
    for index, attempt in enumerate(manifest["attempts"]):
        location = f"/attempts/{index}"
        if not isinstance(attempt, dict):
            add("manifest.type", location, "attempt must be an object")
            continue
        _exact_keys(attempt, _ATTEMPT_KEYS, location, add)
        if set(attempt) != _ATTEMPT_KEYS:
            continue
        _require_int(
            attempt["attempt_index"], f"{location}/attempt_index", add, minimum=0
        )
        if attempt["attempt_index"] != index:
            add(
                "manifest.value",
                f"{location}/attempt_index",
                "attempt indices must be contiguous",
            )
        _require_int(attempt["seed"], f"{location}/seed", add, minimum=0)
        if type(attempt["seed"]) is int:
            seeds.append(attempt["seed"])
            try:
                validate_split_seed(split, attempt["seed"])
            except (TypeError, ValueError) as error:
                add("manifest.seed", f"{location}/seed", str(error))
            expected_seed = contract["scan_start"] + index
            if attempt["seed"] != expected_seed:
                add(
                    "manifest.seed_sequence",
                    f"{location}/seed",
                    f"expected monotonically scanned seed {expected_seed}",
                )
        status = attempt["status"]
        if not isinstance(status, str) or status not in {"success", "failure"}:
            add(
                "manifest.value",
                f"{location}/status",
                "final attempts must be success or failure",
            )
        _require_bool(attempt["success"], f"{location}/success", add)
        if type(attempt["success"]) is bool:
            successes += int(attempt["success"])
            if (status == "success") is not attempt["success"]:
                add("manifest.outcome", location, "status and success disagree")
        failure_stage = attempt["failure_stage"]
        if attempt["success"] is True and failure_stage is not None:
            add(
                "manifest.outcome",
                f"{location}/failure_stage",
                "success must use null failure_stage",
            )
        if attempt["success"] is False and (
            not isinstance(failure_stage, str) or not failure_stage.strip()
        ):
            add(
                "manifest.outcome",
                f"{location}/failure_stage",
                "failure must name its stage",
            )
        if attempt["error"] is not None:
            add(
                "manifest.value",
                f"{location}/error",
                "completed manifest attempts cannot have errors",
            )
        _validate_episode_reference(
            attempt,
            location,
            add,
            maximum_steps=episode_step_limit,
        )

    if len(seeds) != len(set(seeds)):
        add("manifest.seed_duplicate", "/attempts", "attempt seeds must be unique")
    if type(manifest["attempt_count"]) is int and manifest["attempt_count"] != len(
        manifest["attempts"]
    ):
        add("manifest.count", "/attempt_count", "attempt_count does not match attempts")
    if (
        type(manifest["success_count"]) is int
        and manifest["success_count"] != successes
    ):
        add("manifest.count", "/success_count", "success_count does not match attempts")
    if (
        type(manifest["success_count"]) is int
        and type(manifest["target_successes"]) is int
        and manifest["success_count"] != manifest["target_successes"]
    ):
        add(
            "manifest.target",
            "/success_count",
            "final collection must reach its target",
        )

    for index, eligible in enumerate(manifest["eligible_successes"]):
        location = f"/eligible_successes/{index}"
        if not isinstance(eligible, dict):
            add("manifest.type", location, "eligible item must be an object")
            continue
        _exact_keys(eligible, _ELIGIBLE_KEYS, location, add)
        if set(eligible) == _ELIGIBLE_KEYS:
            _require_int(eligible["seed"], f"{location}/seed", add, minimum=0)
            _require_int(eligible["num_steps"], f"{location}/num_steps", add, minimum=1)
            if (
                type(eligible["num_steps"]) is int
                and episode_step_limit > 0
                and eligible["num_steps"] > episode_step_limit
            ):
                add(
                    "manifest.num_steps_limit",
                    f"{location}/num_steps",
                    "eligible num_steps exceeds the controller-reachable maximum",
                )
            _require_sha(eligible["sha256"], f"{location}/sha256", add)
            _require_relative_path(eligible["path"], f"{location}/path", add)

    target = manifest["target_successes"]
    cli = manifest["cli_config"]
    if isinstance(cli, dict) and type(target) is int:
        formal_target = formal_target_successes(split)
        smoke = cli.get("smoke")
        diagnostic = cli.get("diagnostic_allow_dirty")
        if smoke is False and target != formal_target:
            add(
                "manifest.target",
                "/target_successes",
                "non-smoke target must be formal 200/40",
            )
        if smoke is True and not 1 <= target <= formal_target:
            add(
                "manifest.target",
                "/target_successes",
                "smoke target must be bounded by the formal target",
            )
        expected_formal = smoke is False and diagnostic is False
        if (
            type(manifest["formal"]) is bool
            and manifest["formal"] is not expected_formal
        ):
            add(
                "manifest.formal",
                "/formal",
                "formal flag is inconsistent with CLI config",
            )
        if manifest["formal"] is True:
            git = manifest["git"]
            if not isinstance(git, dict) or not (
                git.get("provenance_complete") is True
                and git.get("source_provenance_clean") is True
                and git.get("tracked_worktree_clean") is True
                and isinstance(git.get("commit"), str)
                and _COMMIT_RE.fullmatch(git["commit"])
            ):
                add(
                    "manifest.formal",
                    "/git",
                    "formal manifest requires fixed clean source provenance",
                )


def _validate_episode_reference(
    attempt: dict[str, Any],
    location: str,
    add: Any,
    *,
    maximum_steps: int,
) -> None:
    _require_relative_path(attempt["path"], f"{location}/path", add)
    if type(attempt["seed"]) is int and isinstance(attempt["path"], str):
        expected = f"{EPISODES_ROOT}/seed_{attempt['seed']:06d}.h5"
        if attempt["path"] != expected:
            add(
                "path.canonical",
                f"{location}/path",
                f"expected canonical episode path {expected}",
            )
    _require_int(attempt["num_steps"], f"{location}/num_steps", add, minimum=1)
    if (
        type(attempt["num_steps"]) is int
        and maximum_steps > 0
        and attempt["num_steps"] > maximum_steps
    ):
        add(
            "manifest.num_steps_limit",
            f"{location}/num_steps",
            f"num_steps exceeds controller-reachable maximum {maximum_steps}",
        )
    _require_sha(attempt["sha256"], f"{location}/sha256", add)


def _validate_controller(value: Any, add: Any) -> None:
    location = "/controller"
    if not isinstance(value, dict):
        add("manifest.type", location, "controller must be an object")
        return
    _exact_keys(value, set(_CONTROLLER_DEFAULTS), location, add)
    for name, default in _CONTROLLER_DEFAULTS.items():
        if name not in value:
            continue
        if type(default) is int:
            _require_int(value[name], f"{location}/{name}", add, minimum=1)
        else:
            _require_number(value[name], f"{location}/{name}", add)


def _validate_git(value: Any, add: Any) -> None:
    location = "/git"
    if not isinstance(value, dict):
        add("manifest.type", location, "git must be an object")
        return
    _exact_keys(value, _GIT_KEYS, location, add)
    if set(value) != _GIT_KEYS:
        return
    if value["commit"] is not None and (
        not isinstance(value["commit"], str)
        or _COMMIT_RE.fullmatch(value["commit"]) is None
    ):
        add(
            "manifest.type",
            f"{location}/commit",
            "commit must be null or a full lowercase SHA",
        )
    for name in (
        "tracked_worktree_clean",
        "worktree_clean",
        "source_provenance_clean",
        "provenance_complete",
    ):
        _require_bool(value[name], f"{location}/{name}", add)
    for name in ("worktree_status_sha256", "untracked_paths_sha256"):
        if value[name] is not None:
            _require_sha(value[name], f"{location}/{name}", add)
    if value["untracked_file_count"] is not None:
        _require_int(
            value["untracked_file_count"],
            f"{location}/untracked_file_count",
            add,
            minimum=0,
        )
    files = value["relevant_untracked_files"]
    if not isinstance(files, list):
        add("manifest.type", f"{location}/relevant_untracked_files", "must be an array")
        return
    for index, item in enumerate(files):
        item_location = f"{location}/relevant_untracked_files/{index}"
        if not isinstance(item, dict):
            add(
                "manifest.type", item_location, "Git file fingerprint must be an object"
            )
            continue
        _exact_keys(item, _GIT_FILE_KEYS, item_location, add)
        if set(item) != _GIT_FILE_KEYS:
            continue
        if not isinstance(item["git_status"], str) or item["git_status"] not in {
            "untracked",
            "ignored",
        }:
            add(
                "manifest.value",
                f"{item_location}/git_status",
                "unsupported git status",
            )
        if not isinstance(item["path"], str) or not item["path"]:
            add(
                "manifest.type",
                f"{item_location}/path",
                "Git path must be non-empty text",
            )
        if item["sha256"] is not None:
            _require_sha(item["sha256"], f"{item_location}/sha256", add)
        if item["size_bytes"] is not None:
            _require_int(
                item["size_bytes"], f"{item_location}/size_bytes", add, minimum=0
            )


def _validate_assets(value: Any, add: Any) -> None:
    location = "/assets"
    expected_keys = {"summary", "content_manifest", "aggregate_manifest_sha256"}
    if not isinstance(value, dict):
        add("manifest.type", location, "assets must be an object")
        return
    _exact_keys(value, expected_keys, location, add)
    if set(value) != expected_keys:
        return
    _require_sha(
        value["aggregate_manifest_sha256"], f"{location}/aggregate_manifest_sha256", add
    )
    if not isinstance(value["summary"], dict) or not isinstance(
        value["content_manifest"], dict
    ):
        add("manifest.type", location, "asset summary and manifest must be objects")
        return
    summary = value["summary"]
    summary_keys = {
        "canonical_root",
        "revision",
        "file_count",
        "aggregate_manifest_sha256",
    }
    _exact_keys(summary, summary_keys, f"{location}/summary", add)
    if set(summary) == summary_keys:
        if summary["canonical_root"] != "menagerie/franka_emika_panda":
            add(
                "manifest.value",
                f"{location}/summary/canonical_root",
                "asset root is not the canonical vendored Panda snapshot",
            )
        _require_int(
            summary["file_count"], f"{location}/summary/file_count", add, minimum=1
        )
        _require_sha(
            summary["aggregate_manifest_sha256"],
            f"{location}/summary/aggregate_manifest_sha256",
            add,
        )
        revision = summary["revision"]
        revision_keys = {"repository", "revision", "vendored_path", "license"}
        if not isinstance(revision, dict):
            add(
                "manifest.type",
                f"{location}/summary/revision",
                "revision must be an object",
            )
        else:
            _exact_keys(revision, revision_keys, f"{location}/summary/revision", add)
            if set(revision) == revision_keys:
                expected_revision_values = {
                    "repository": "https://github.com/google-deepmind/mujoco_menagerie",
                    "vendored_path": "franka_emika_panda",
                    "license": "Apache-2.0",
                }
                for name, expected in expected_revision_values.items():
                    if revision[name] != expected:
                        add(
                            "manifest.value",
                            f"{location}/summary/revision/{name}",
                            f"expected canonical value {expected!r}",
                        )
                if (
                    not isinstance(revision["revision"], str)
                    or _COMMIT_RE.fullmatch(revision["revision"]) is None
                ):
                    add(
                        "manifest.type",
                        f"{location}/summary/revision/revision",
                        "asset revision must be a full lowercase commit SHA",
                    )
    content_manifest = value["content_manifest"]
    content_keys = {"schema_version", "files"}
    _exact_keys(content_manifest, content_keys, f"{location}/content_manifest", add)
    if set(content_manifest) == content_keys:
        if content_manifest["schema_version"] != "asset-manifest.v1":
            add(
                "manifest.value",
                f"{location}/content_manifest/schema_version",
                "unsupported asset manifest schema",
            )
        files = content_manifest["files"]
        if not isinstance(files, list) or not files:
            add(
                "manifest.type",
                f"{location}/content_manifest/files",
                "asset files must be a non-empty array",
            )
        else:
            paths: list[str] = []
            for index, item in enumerate(files):
                item_location = f"{location}/content_manifest/files/{index}"
                if not isinstance(item, dict):
                    add(
                        "manifest.type",
                        item_location,
                        "asset fingerprint must be an object",
                    )
                    continue
                file_keys = {"path", "sha256", "size_bytes"}
                _exact_keys(item, file_keys, item_location, add)
                if set(item) != file_keys:
                    continue
                _require_relative_path(item["path"], f"{item_location}/path", add)
                if isinstance(item["path"], str):
                    paths.append(item["path"])
                _require_sha(item["sha256"], f"{item_location}/sha256", add)
                _require_int(
                    item["size_bytes"], f"{item_location}/size_bytes", add, minimum=0
                )
            if len(paths) != len(set(paths)):
                add(
                    "manifest.value",
                    f"{location}/content_manifest/files",
                    "asset paths must be unique",
                )
            if paths != sorted(paths):
                add(
                    "manifest.value",
                    f"{location}/content_manifest/files",
                    "asset paths must use canonical sorted order",
                )
            if type(summary.get("file_count")) is int and summary["file_count"] != len(
                files
            ):
                add(
                    "manifest.value",
                    f"{location}/summary/file_count",
                    "asset file_count does not match content manifest",
                )
    try:
        observed = sha256_bytes(canonical_json_bytes(content_manifest))
    except (TypeError, ValueError):
        add(
            "manifest.type",
            f"{location}/content_manifest",
            "asset manifest must be finite JSON",
        )
        return
    if observed != value["aggregate_manifest_sha256"]:
        add("manifest.value", location, "asset content manifest hash does not match")
    summary_hash = value["summary"].get("aggregate_manifest_sha256")
    if summary_hash != value["aggregate_manifest_sha256"]:
        add(
            "manifest.value", f"{location}/summary", "asset summary hash does not match"
        )


def _validate_environment(value: Any, add: Any) -> None:
    location = "/environment"
    keys = {
        "schema_version",
        "mujoco_version",
        "task_config",
        "debug_viz",
        "compiled_model",
    }
    if not isinstance(value, dict):
        add("manifest.type", location, "environment must be an object")
        return
    _exact_keys(value, keys, location, add)
    if set(value) != keys:
        return
    if value["schema_version"] != ENVIRONMENT_SCHEMA:
        add(
            "manifest.value",
            f"{location}/schema_version",
            f"expected {ENVIRONMENT_SCHEMA}",
        )
    if not isinstance(value["mujoco_version"], str) or not value["mujoco_version"]:
        add(
            "manifest.type", f"{location}/mujoco_version", "MuJoCo version must be text"
        )
    if value["debug_viz"] is not False:
        add(
            "manifest.value", f"{location}/debug_viz", "debug_viz must be exactly false"
        )
    task = value["task_config"]
    if not isinstance(task, dict):
        add("manifest.type", f"{location}/task_config", "task_config must be an object")
    else:
        _exact_keys(task, _TASK_CONFIG_KEYS, f"{location}/task_config", add)
        if task.get("debug_viz") is not False:
            add(
                "manifest.value",
                f"{location}/task_config/debug_viz",
                "debug_viz must be false",
            )
        for name in ("n_distractors", "img_size"):
            if name in task:
                _require_int(
                    task[name], f"{location}/task_config/{name}", add, minimum=0
                )
        for name in ("control_hz", "success_hold_s", "success_z_tolerance"):
            if name in task:
                _require_number(task[name], f"{location}/task_config/{name}", add)
        if "colors" in task and (
            not isinstance(task["colors"], list)
            or any(not isinstance(item, str) or not item for item in task["colors"])
        ):
            add(
                "manifest.type",
                f"{location}/task_config/colors",
                "colors must be a string array",
            )
    compiled = value["compiled_model"]
    if not isinstance(compiled, dict):
        add(
            "manifest.type",
            f"{location}/compiled_model",
            "compiled_model must be an object",
        )
        return
    _exact_keys(compiled, _COMPILED_MODEL_KEYS, f"{location}/compiled_model", add)
    if set(compiled) != _COMPILED_MODEL_KEYS:
        return
    if compiled["fingerprint_schema"] != COMPILED_MODEL_SCHEMA:
        add(
            "manifest.value",
            f"{location}/compiled_model/fingerprint_schema",
            "wrong model fingerprint schema",
        )
    for name in ("mjb_sha256", "fingerprint_sha256"):
        _require_sha(compiled[name], f"{location}/compiled_model/{name}", add)
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "nsite"):
        _require_int(
            compiled[name], f"{location}/compiled_model/{name}", add, minimum=1
        )
    sites = compiled["policy_sites"]
    if not isinstance(sites, list) or len(sites) != 2:
        add(
            "manifest.type",
            f"{location}/compiled_model/policy_sites",
            "exactly two policy sites required",
        )
    else:
        names: list[str] = []
        for index, site in enumerate(sites):
            site_location = f"{location}/compiled_model/policy_sites/{index}"
            if not isinstance(site, dict) or set(site) != {"name", "rgba"}:
                add(
                    "manifest.type",
                    site_location,
                    "policy site must contain name and rgba",
                )
                continue
            names.append(site["name"])
            rgba = site["rgba"]
            if (
                not isinstance(rgba, list)
                or len(rgba) != 4
                or any(
                    type(item) not in {int, float}
                    or isinstance(item, bool)
                    or not np.isfinite(item)
                    for item in rgba
                )
                or rgba[3] != 0.0
            ):
                add(
                    "manifest.value",
                    f"{site_location}/rgba",
                    "policy site RGBA must be finite and invisible",
                )
        if names != ["flange", "tcp"]:
            add(
                "manifest.value",
                f"{location}/compiled_model/policy_sites",
                "sites must be canonical flange,tcp order",
            )
    if set(compiled) == _COMPILED_MODEL_KEYS:
        base = {
            key: value for key, value in compiled.items() if key != "fingerprint_sha256"
        }
        if sha256_bytes(canonical_json_bytes(base)) != compiled["fingerprint_sha256"]:
            add(
                "manifest.value",
                f"{location}/compiled_model/fingerprint_sha256",
                "compiled fingerprint hash does not match",
            )


def _validate_cli_config(value: Any, split: str, add: Any) -> None:
    location = "/cli_config"
    keys = {"split", "target_successes", "smoke", "diagnostic_allow_dirty"}
    if not isinstance(value, dict):
        add("manifest.type", location, "cli_config must be an object")
        return
    _exact_keys(value, keys, location, add)
    if set(value) != keys:
        return
    if value["split"] != split:
        add("manifest.value", f"{location}/split", "CLI split disagrees with manifest")
    _require_int(
        value["target_successes"], f"{location}/target_successes", add, minimum=1
    )
    for name in ("smoke", "diagnostic_allow_dirty"):
        _require_bool(value[name], f"{location}/{name}", add)


def _exact_keys(
    value: dict[str, Any], expected: set[str], location: str, add: Any
) -> None:
    received = set(value)
    if received != expected:
        add(
            "manifest.keys",
            location,
            f"expected exact keys; missing={sorted(expected - received)}, extra={sorted(received - expected)}",
        )


def _require_bool(value: Any, location: str, add: Any) -> None:
    if type(value) is not bool:
        add("manifest.type", location, "value must be a JSON boolean")


def _require_int(value: Any, location: str, add: Any, *, minimum: int) -> None:
    if type(value) is not int:
        add("manifest.type", location, "value must be a JSON integer, not boolean")
    elif value < minimum:
        add("manifest.value", location, f"value must be >= {minimum}")


def _require_number(value: Any, location: str, add: Any) -> None:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        add("manifest.type", location, "value must be a finite JSON number")


def _require_sha(value: Any, location: str, add: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        add("manifest.type", location, "value must be a lowercase SHA-256")


def _require_relative_path(value: Any, location: str, add: Any) -> None:
    if not isinstance(value, str):
        add("manifest.type", location, "path must be a string")
        return
    try:
        validate_relative_path(value)
    except UnsafeCollectionPath as error:
        add(error.code, location, str(error))


def _validate_timestamp(value: Any, location: str, add: Any) -> None:
    if not isinstance(value, str):
        add("manifest.type", location, "timestamp must be ISO-8601 text")
        return
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        add("manifest.value", location, "timestamp is not valid ISO-8601")
        return
    if parsed.tzinfo is None:
        add("manifest.value", location, "timestamp must include a timezone")


def _safe_regular_path(root: Path, relative_path: str) -> tuple[Path, FileSnapshot]:
    pure = validate_relative_path(relative_path)
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        root_status = root_absolute.lstat()
    except OSError as error:
        raise UnsafeCollectionPath("path.root", "run root is unavailable") from error
    if stat.S_ISLNK(root_status.st_mode):
        raise UnsafeCollectionPath("path.symlink", "run root must not be a symlink")
    if not stat.S_ISDIR(root_status.st_mode):
        raise UnsafeCollectionPath("path.root", "run root must be a directory")
    resolved_root = root_absolute.resolve(strict=True)
    current = root_absolute
    leaf_snapshot: FileSnapshot | None = None
    for index, component in enumerate(pure.parts):
        current = current / component
        try:
            status = current.lstat()
        except OSError as error:
            raise UnsafeCollectionPath(
                "path.missing", f"path component is missing: {component}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise UnsafeCollectionPath(
                "path.symlink", f"path contains a symlink: {component}"
            )
        if index < len(pure.parts) - 1:
            if not stat.S_ISDIR(status.st_mode):
                raise UnsafeCollectionPath(
                    "path.non_directory",
                    f"path component is not a directory: {component}",
                )
        elif not stat.S_ISREG(status.st_mode):
            raise UnsafeCollectionPath(
                "path.non_regular", "path leaf must be a regular file"
            )
        else:
            leaf_snapshot = _snapshot_from_stat(status)
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise UnsafeCollectionPath(
            "path.escape", "path does not resolve beneath the run root"
        ) from error
    if leaf_snapshot is None:
        raise UnsafeCollectionPath("path.non_regular", "path has no regular leaf")
    return current, leaf_snapshot


def _open_anchored_regular(root: Path, relative: PurePosixPath) -> int:
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        directory_fd = os.open(root_absolute, directory_flags)
    except OSError as error:
        raise UnsafeCollectionPath("path.root", "cannot anchor run root") from error
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise UnsafeCollectionPath(
                    "path.symlink", "unsafe directory component"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            descriptor = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise UnsafeCollectionPath(
                "path.symlink", "unsafe or missing file leaf"
            ) from error
    finally:
        os.close(directory_fd)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise UnsafeCollectionPath("path.non_regular", "file leaf is not regular")
    return descriptor


def _snapshot_from_stat(status: os.stat_result) -> FileSnapshot:
    return FileSnapshot(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        ctime_ns=status.st_ctime_ns,
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _directory_chain_snapshots(
    root: Path,
    relative: PurePosixPath,
) -> tuple[tuple[str, FileSnapshot], ...]:
    current = Path(os.path.abspath(os.fspath(root)))
    output: list[tuple[str, FileSnapshot]] = []
    for component in (None, *relative.parts[:-1]):
        if component is not None:
            current = current / component
        status = current.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise UnsafeCollectionPath(
                "path.symlink",
                "episode directory chain is no longer a real directory",
            )
        output.append((os.fspath(current), _snapshot_from_stat(status)))
    return tuple(output)


def _stable_read_absolute(path: Path) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.lstat()
    except OSError as error:
        raise ValueError("file is unavailable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("file must be a non-symlink regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        opened_before = _snapshot_from_stat(os.fstat(descriptor))
        if opened_before != _snapshot_from_stat(before):
            raise ValueError("file changed before read")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = _snapshot_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after = _snapshot_from_stat(absolute.lstat())
    if opened_before != opened_after or opened_after != after:
        raise ValueError("file changed while read")
    return b"".join(chunks)


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _strict_jsonl_loads(text: str) -> list[Any]:
    if text and not text.endswith("\n"):
        raise ValueError("JSONL ledger must end with a newline")
    lines = text.splitlines()
    if any(not line for line in lines):
        raise ValueError("JSONL ledger contains an empty line")
    return [_strict_json_loads(line) for line in lines]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for M2 collection manifests."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--other-manifest",
        type=Path,
        help="also enforce the train/validation cross-manifest contract",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="atomically publish a JSON validation report without clobbering",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.other_manifest is None:
        validation = validate_collection_manifest(args.manifest)
        manifest_paths = [args.manifest]
    else:
        validation = validate_manifest_pair(args.manifest, args.other_manifest)
        manifest_paths = [args.manifest, args.other_manifest]

    payload = {
        "schema_version": VALIDATION_REPORT_SCHEMA,
        "valid": validation.valid,
        "manifest_paths": [
            Path(os.path.abspath(os.fspath(path))).as_posix() for path in manifest_paths
        ],
        "errors": [asdict(error) for error in validation.errors],
    }
    if args.report is not None:
        atomic_write_json_no_clobber(args.report, payload)
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if validation.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
