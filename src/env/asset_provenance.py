"""Fail-closed provenance for the repository-vendored Panda assets.

Formal collectors and evaluators must use the Panda snapshot tracked by this
repository.  The interactive native viewer intentionally keeps its existing
``MENAGERIE`` override behavior; callers opt into this formal boundary by
calling :func:`collect_asset_provenance` before creating output artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ASSET_ROOT = PurePosixPath("menagerie/franka_emika_panda")
_REVISION_PATH = PurePosixPath("menagerie/VENDORED_REVISION")
_REQUIRED_ASSET_PATHS = frozenset(
    {
        "menagerie/franka_emika_panda/panda.xml",
        "menagerie/franka_emika_panda/scene.xml",
    }
)
_EXPECTED_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
_EXPECTED_VENDORED_PATH = "franka_emika_panda"
_EXPECTED_LICENSE = "Apache-2.0"
_REVISION_KEYS = ("repository", "revision", "vendored_path", "license")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_MAX_REVISION_BYTES = 4096
_MANIFEST_SCHEMA = "asset-manifest.v1"


class AssetProvenanceError(RuntimeError):
    """The formal Panda asset boundary could not be established."""


@dataclass(frozen=True)
class VendoredRevision:
    repository: str
    revision: str
    vendored_path: str
    license: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "vendored_path": self.vendored_path,
            "license": self.license,
        }


@dataclass(frozen=True)
class AssetFileFingerprint:
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class AssetProvenance:
    canonical_root: str
    revision: VendoredRevision
    files: tuple[AssetFileFingerprint, ...]
    aggregate_manifest_sha256: str

    def manifest_dict(self) -> dict[str, Any]:
        """Return the content-free, project-relative file manifest."""

        return {
            "schema_version": _MANIFEST_SCHEMA,
            "files": [fingerprint.to_dict() for fingerprint in self.files],
        }

    def summary_dict(self) -> dict[str, Any]:
        """Return the compact mapping embedded in future formal summaries."""

        return {
            "canonical_root": self.canonical_root,
            "revision": self.revision.to_dict(),
            "file_count": len(self.files),
            "aggregate_manifest_sha256": self.aggregate_manifest_sha256,
        }


def collect_asset_provenance(
    project_root: Path = PROJECT_ROOT,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_asset_root: Path | None = None,
) -> AssetProvenance:
    """Validate and fingerprint the canonical tracked Panda asset snapshot.

    ``project_root`` and ``environ`` are injectable so future M2 collection and
    fixture tests can reuse exactly the same boundary.  Formal runtime callers
    should also pass the asset root already captured by their scene module;
    this catches environment changes made before or after module import.
    """

    root = _resolved_project_root(project_root)
    canonical_asset_path = root.joinpath(*_ASSET_ROOT.parts)
    _path_snapshot(root, _ASSET_ROOT.parts, leaf_kind="directory")
    try:
        resolved_asset_path = canonical_asset_path.resolve(strict=True)
    except OSError as error:
        raise AssetProvenanceError("canonical Panda asset root is unavailable") from error
    if resolved_asset_path != canonical_asset_path:
        raise AssetProvenanceError("canonical Panda asset root contains a symlink")

    environment = os.environ if environ is None else environ
    configured_menagerie = environment.get("MENAGERIE")
    if configured_menagerie is not None:
        if not configured_menagerie:
            raise AssetProvenanceError("MENAGERIE must name the canonical repository root")
        configured_asset_path = Path(configured_menagerie) / _EXPECTED_VENDORED_PATH
        if _absolute_lexical_path(configured_asset_path) != canonical_asset_path:
            raise AssetProvenanceError(
                "external MENAGERIE overrides are forbidden for formal runs"
            )

    if runtime_asset_root is not None:
        if _absolute_lexical_path(runtime_asset_root) != canonical_asset_path:
            raise AssetProvenanceError(
                "runtime Panda asset root is not the canonical repository snapshot"
            )

    tracked_paths = _git_tracked_asset_paths(root)
    missing_required = sorted(_REQUIRED_ASSET_PATHS.difference(tracked_paths))
    if str(_REVISION_PATH) not in tracked_paths:
        missing_required.insert(0, str(_REVISION_PATH))
    if missing_required:
        raise AssetProvenanceError(
            "required tracked Panda asset is missing: " + ", ".join(missing_required)
        )
    fingerprints: list[AssetFileFingerprint] = []
    revision_bytes: bytes | None = None
    for tracked_path in tracked_paths:
        content, fingerprint = _fingerprint_tracked_file(
            root,
            tracked_path,
            capture_content=tracked_path == str(_REVISION_PATH),
        )
        fingerprints.append(fingerprint)
        if content is not None:
            revision_bytes = content

    if revision_bytes is None:
        raise AssetProvenanceError("tracked VENDORED_REVISION could not be read")
    revision = _parse_vendored_revision(revision_bytes)
    _require_committed_asset_snapshot(root, tracked_paths)
    files = tuple(fingerprints)
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "files": [fingerprint.to_dict() for fingerprint in files],
    }
    aggregate = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    return AssetProvenance(
        canonical_root=str(_ASSET_ROOT),
        revision=revision,
        files=files,
        aggregate_manifest_sha256=aggregate,
    )


def _resolved_project_root(project_root: Path) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
        root_status = root.stat(follow_symlinks=False)
    except OSError as error:
        raise AssetProvenanceError("project root is unavailable") from error
    if not stat.S_ISDIR(root_status.st_mode):
        raise AssetProvenanceError("project root is not a directory")
    return root


def _absolute_lexical_path(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as error:
        raise AssetProvenanceError("asset root path is invalid") from error


def _git_tracked_asset_paths(project_root: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            str(_REVISION_PATH),
            str(_ASSET_ROOT),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssetProvenanceError("cannot enumerate tracked Panda assets with Git")
    return _parse_git_tracked_paths(completed.stdout)


def _require_committed_asset_snapshot(
    project_root: Path,
    tracked_paths: list[str],
) -> None:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--no-ext-diff",
            "HEAD",
            "--",
            *tracked_paths,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 1:
        raise AssetProvenanceError(
            "tracked Panda assets differ from the committed snapshot"
        )
    if completed.returncode != 0:
        raise AssetProvenanceError(
            "cannot compare tracked Panda assets with the committed snapshot"
        )


def _parse_git_tracked_paths(output: bytes) -> list[str]:
    """Strictly parse ``git ls-files -z`` output for the allowed asset scope."""

    if not output:
        return []
    if not output.endswith(b"\0"):
        raise AssetProvenanceError("Git tracked-path output is not NUL terminated")

    parsed: list[str] = []
    seen: set[str] = set()
    for encoded_path in output[:-1].split(b"\0"):
        if not encoded_path:
            raise AssetProvenanceError("Git tracked-path output contains an empty path")
        try:
            path = encoded_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AssetProvenanceError("tracked Panda asset path is not UTF-8") from error
        if path in seen:
            raise AssetProvenanceError("duplicate tracked Panda asset path")
        seen.add(path)

        pure_path = PurePosixPath(path)
        parts = pure_path.parts
        if (
            not parts
            or pure_path.is_absolute()
            or pure_path.as_posix() != path
            or any(part in {"", ".", ".."} for part in parts)
            or "\\" in path
            or any(ord(character) < 32 for character in path)
        ):
            raise AssetProvenanceError("tracked Panda asset path is not canonical")
        if path != str(_REVISION_PATH) and not path.startswith(f"{_ASSET_ROOT}/"):
            raise AssetProvenanceError("tracked Panda asset path escapes the allowed scope")
        parsed.append(path)
    return sorted(parsed)


def _fingerprint_tracked_file(
    project_root: Path,
    tracked_path: str,
    *,
    capture_content: bool,
) -> tuple[bytes | None, AssetFileFingerprint]:
    relative_path = PurePosixPath(tracked_path)
    before_path = _path_snapshot(
        project_root,
        relative_path.parts,
        leaf_kind="regular",
    )
    descriptor = _open_anchored_regular_file(project_root, relative_path.parts)
    captured = bytearray() if capture_content else None
    digest = hashlib.sha256()
    try:
        before_open = os.fstat(descriptor)
        if not stat.S_ISREG(before_open.st_mode):
            raise AssetProvenanceError(f"tracked Panda asset is not regular: {tracked_path}")
        if _stat_identity(before_open) != before_path[-1]:
            raise AssetProvenanceError(f"tracked Panda asset changed before read: {tracked_path}")

        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                if len(captured) + len(chunk) > _MAX_REVISION_BYTES:
                    raise AssetProvenanceError("VENDORED_REVISION is too large")
                captured.extend(chunk)
        after_open = os.fstat(descriptor)
    except OSError as error:
        raise AssetProvenanceError(f"cannot read tracked Panda asset: {tracked_path}") from error
    finally:
        os.close(descriptor)

    after_path = _path_snapshot(
        project_root,
        relative_path.parts,
        leaf_kind="regular",
    )
    if _stat_identity(before_open) != _stat_identity(after_open):
        raise AssetProvenanceError(f"tracked Panda asset changed during read: {tracked_path}")
    if before_path != after_path or _stat_identity(after_open) != after_path[-1]:
        raise AssetProvenanceError(f"tracked Panda asset path changed during read: {tracked_path}")

    return (
        bytes(captured) if captured is not None else None,
        AssetFileFingerprint(
            path=tracked_path,
            size_bytes=before_open.st_size,
            sha256=digest.hexdigest(),
        ),
    )


def _open_anchored_regular_file(project_root: Path, parts: tuple[str, ...]) -> int:
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(project_root, directory_flags)
    except OSError as error:
        raise AssetProvenanceError("cannot anchor the project root") from error
    try:
        for component in parts[:-1]:
            try:
                next_directory_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise AssetProvenanceError(
                    "tracked Panda asset contains a symlink or invalid component"
                ) from error
            os.close(directory_fd)
            directory_fd = next_directory_fd
        try:
            return os.open(parts[-1], file_flags, dir_fd=directory_fd)
        except OSError as error:
            raise AssetProvenanceError(
                "tracked Panda asset is missing, non-regular, or a symlink"
            ) from error
    finally:
        os.close(directory_fd)


def _path_snapshot(
    project_root: Path,
    parts: tuple[str, ...],
    *,
    leaf_kind: str,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    current = project_root
    snapshots: list[tuple[int, int, int, int, int, int]] = []
    for index, component in enumerate(parts):
        current = current / component
        relative = PurePosixPath(*parts[: index + 1])
        try:
            status = current.lstat()
        except OSError as error:
            raise AssetProvenanceError(f"asset path is missing: {relative}") from error
        if stat.S_ISLNK(status.st_mode):
            raise AssetProvenanceError(f"asset path contains a symlink: {relative}")
        expected_kind = leaf_kind if index == len(parts) - 1 else "directory"
        if expected_kind == "directory" and not stat.S_ISDIR(status.st_mode):
            raise AssetProvenanceError(f"asset path component is not a directory: {relative}")
        if expected_kind == "regular" and not stat.S_ISREG(status.st_mode):
            raise AssetProvenanceError(f"tracked Panda asset is not regular: {relative}")
        snapshots.append(_stat_identity(status))
    return tuple(snapshots)


def _stat_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _parse_vendored_revision(content: bytes) -> VendoredRevision:
    if len(content) > _MAX_REVISION_BYTES:
        raise AssetProvenanceError("VENDORED_REVISION is too large")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AssetProvenanceError("VENDORED_REVISION is not UTF-8") from error
    if "\r" in text or not text.endswith("\n"):
        raise AssetProvenanceError("VENDORED_REVISION must use LF-terminated lines")

    lines = text[:-1].split("\n")
    if len(lines) != len(_REVISION_KEYS):
        raise AssetProvenanceError("VENDORED_REVISION must contain exactly four fields")
    values: dict[str, str] = {}
    for expected_key, line in zip(_REVISION_KEYS, lines, strict=True):
        if line.count("=") != 1:
            raise AssetProvenanceError("VENDORED_REVISION field syntax is invalid")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            raise AssetProvenanceError("VENDORED_REVISION fields are missing or out of order")
        values[key] = value

    if values["repository"] != _EXPECTED_REPOSITORY:
        raise AssetProvenanceError("VENDORED_REVISION repository is not canonical")
    if _COMMIT_PATTERN.fullmatch(values["revision"]) is None:
        raise AssetProvenanceError("VENDORED_REVISION revision is not a full commit SHA")
    if values["vendored_path"] != _EXPECTED_VENDORED_PATH:
        raise AssetProvenanceError("VENDORED_REVISION vendored_path is not canonical")
    if values["license"] != _EXPECTED_LICENSE:
        raise AssetProvenanceError("VENDORED_REVISION license is not canonical")
    return VendoredRevision(**values)


def _canonical_json_bytes(payload: Any) -> bytes:
    """Apply the project's stable JSON hashing rule (no trailing LF)."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
