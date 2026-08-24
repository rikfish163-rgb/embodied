"""Fail-closed Git/source provenance manifests for the LeRobot boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

try:  # Package import in the editable project.
    from . import canonical_json as _DEFAULT_CANONICAL
except ImportError:  # Standalone immutable snapshot used by the launcher.
    try:
        import canonical_json as _DEFAULT_CANONICAL  # type: ignore[no-redef]
    except ImportError:
        _DEFAULT_CANONICAL = None  # type: ignore[assignment]


MANIFEST_SCHEMA = "lerobot-source-input-manifest.v1"
SCOPE_SCHEMA = "lerobot-source-input-scope.v1"
OBSERVATION_SCHEMA = "worktree-observation.v1"
RECHECK_SCHEMA = "lerobot-source-recheck.v1"


class SourceProvenanceError(RuntimeError):
    """Raised when the source state cannot be audited completely."""


def _canonical_module(module: ModuleType | None = None) -> ModuleType:
    selected = module or _DEFAULT_CANONICAL
    if selected is None:
        raise SourceProvenanceError("a trusted canonical helper is required")
    return selected


def _load_canonical_helper(path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise SourceProvenanceError("canonical helper must be a regular file")
    spec = importlib.util.spec_from_file_location("trusted_canonical_json", path)
    if spec is None or spec.loader is None:
        raise SourceProvenanceError("cannot load canonical helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SourceProvenanceError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise SourceProvenanceError(f"invalid scope path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise SourceProvenanceError(f"scope path is not canonical: {value!r}")
    return value


def _safe_hash_path(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise SourceProvenanceError(f"symlink is not allowed: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SourceProvenanceError(f"non-regular file is not allowed: {path.name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise SourceProvenanceError(f"file changed while hashing: {path.name}")
        return before.st_size, digest.hexdigest()
    finally:
        os.close(fd)


def _safe_repo_path(repo: Path, relative: str) -> Path:
    current = repo
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise SourceProvenanceError(f"symlink component in scope path: {relative}")
    return current


def _load_scope(
    scope_config: Path, canonical: ModuleType
) -> tuple[list[str], list[tuple[str, tuple[str, ...]]]]:
    document = canonical.load_json_strict(scope_config)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "files",
        "trees",
    }:
        raise SourceProvenanceError("scope config has unexpected fields")
    if document["schema_version"] != SCOPE_SCHEMA:
        raise SourceProvenanceError("unsupported source scope schema")
    files_raw = document["files"]
    trees_raw = document["trees"]
    if not isinstance(files_raw, list) or not isinstance(trees_raw, list):
        raise SourceProvenanceError("scope files and trees must be arrays")

    files = [
        _normalized_relative_path(value)
        for value in files_raw
        if isinstance(value, str)
    ]
    if len(files) != len(files_raw) or len(set(files)) != len(files):
        raise SourceProvenanceError("scope files must be unique strings")

    trees: list[tuple[str, tuple[str, ...]]] = []
    for item in trees_raw:
        if not isinstance(item, dict) or set(item) != {"path", "extensions"}:
            raise SourceProvenanceError("scope tree has unexpected fields")
        path = _normalized_relative_path(item["path"])
        extensions = item["extensions"]
        if not isinstance(extensions, list) or not all(
            isinstance(extension, str) and extension.startswith(".")
            for extension in extensions
        ):
            raise SourceProvenanceError("tree extensions must be dotted strings")
        if len(set(extensions)) != len(extensions):
            raise SourceProvenanceError("tree extensions must be unique")
        trees.append((path, tuple(sorted(extensions))))
    if len({path for path, _ in trees}) != len(trees):
        raise SourceProvenanceError("scope tree paths must be unique")
    return sorted(files), sorted(trees)


def _matches_extensions(path: str, extensions: tuple[str, ...]) -> bool:
    return not extensions or PurePosixPath(path).suffix in extensions


def _tracked_paths(repo: Path, tree: str, extensions: tuple[str, ...]) -> set[str]:
    raw = _git(repo, "ls-files", "-z", "--", tree)
    paths = {
        _normalized_relative_path(item.decode("utf-8", errors="strict"))
        for item in raw.split(b"\x00")
        if item
    }
    return {path for path in paths if _matches_extensions(path, extensions)}


def _working_tree_paths(repo: Path, tree: str, extensions: tuple[str, ...]) -> set[str]:
    root = _safe_repo_path(repo, tree)
    if not root.exists():
        return set()
    if not root.is_dir():
        raise SourceProvenanceError(f"scope tree is not a directory: {tree}")
    result: set[str] = set()
    for directory, subdirs, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        safe_subdirs: list[str] = []
        for subdir in subdirs:
            candidate = directory_path / subdir
            relative = candidate.relative_to(repo).as_posix()
            if candidate.is_symlink():
                result.add(relative)
            else:
                safe_subdirs.append(subdir)
        subdirs[:] = safe_subdirs
        for filename in files:
            candidate = directory_path / filename
            relative = candidate.relative_to(repo).as_posix()
            if _matches_extensions(relative, extensions):
                result.add(relative)
    return result


def _index_entry(repo: Path, relative: str) -> tuple[str, str]:
    raw = _git(repo, "ls-files", "--stage", "-z", "--", relative)
    records = [record for record in raw.split(b"\x00") if record]
    if len(records) != 1:
        raise SourceProvenanceError(f"index entry is missing or conflicted: {relative}")
    header, encoded_path = records[0].split(b"\t", 1)
    mode, oid, stage = header.decode("ascii").split()
    if stage != "0" or encoded_path.decode("utf-8", errors="strict") != relative:
        raise SourceProvenanceError(f"invalid index entry: {relative}")
    return mode, oid


def _head_entry(repo: Path, relative: str) -> tuple[str, str]:
    raw = _git(repo, "ls-tree", "-z", "HEAD", "--", relative)
    records = [record for record in raw.split(b"\x00") if record]
    if len(records) != 1:
        raise SourceProvenanceError(f"HEAD entry is missing: {relative}")
    header, encoded_path = records[0].split(b"\t", 1)
    mode, object_type, oid = header.decode("ascii").split()
    if (
        object_type != "blob"
        or encoded_path.decode("utf-8", errors="strict") != relative
    ):
        raise SourceProvenanceError(f"HEAD entry is not a blob: {relative}")
    return mode, oid


def _worktree_observation(repo: Path, git_commit: str) -> dict[str, Any]:
    raw = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    records = raw.split(b"\x00")
    untracked: list[bytes] = []
    tracked_dirty_count = 0
    index = 0
    while index < len(records):
        record = records[index]
        if not record:
            index += 1
            continue
        status_code = record[:2]
        if status_code == b"??":
            untracked.append(record[3:])
        else:
            tracked_dirty_count += 1
        index += 2 if b"R" in status_code or b"C" in status_code else 1
    path_bytes = b"".join(path + b"\x00" for path in sorted(untracked))
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "git_commit": git_commit,
        "porcelain_v1_z_sha256": _sha256(raw),
        "tracked_dirty_count": tracked_dirty_count,
        "untracked_count": len(untracked),
        "untracked_path_list_sha256": _sha256(path_bytes),
        "worktree_clean": not raw,
    }


def build_source_snapshot(
    repo_root: str | os.PathLike[str],
    scope_config: str | os.PathLike[str],
    *,
    scope_config_repo_path: str,
    canonical_module: ModuleType | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic source and privacy-preserving worktree observations."""

    canonical = _canonical_module(canonical_module)
    repo = Path(repo_root).resolve(strict=True)
    if not (repo / ".git").exists():
        raise SourceProvenanceError("repo root is not a Git working tree")
    scope_path = Path(scope_config)
    if not scope_path.is_absolute():
        scope_path = (Path.cwd() / scope_path).resolve(strict=True)
    scope_repo_path = _normalized_relative_path(scope_config_repo_path)
    files, trees = _load_scope(scope_path, canonical)
    if scope_repo_path not in files:
        raise SourceProvenanceError("scope config must include its repository path")
    _, scope_config_sha256 = _safe_hash_path(scope_path)

    git_commit = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    head_tree = _git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    tracked_status = _git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=no"
    )

    expected_paths = set(files)
    working_paths = set(files)
    for tree, extensions in trees:
        expected_paths.update(_tracked_paths(repo, tree, extensions))
        working_paths.update(_working_tree_paths(repo, tree, extensions))

    all_tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git(repo, "ls-files", "-z").split(b"\x00")
        if item
    }
    tracked_scope = sorted(expected_paths & all_tracked)
    relevant_untracked: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for relative in sorted(working_paths - all_tracked):
        path = _safe_repo_path(repo, relative)
        status_name = (
            "ignored"
            if subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", "--", relative],
                check=False,
            ).returncode
            == 0
            else "untracked"
        )
        try:
            size, content_sha256 = _safe_hash_path(path)
        except (OSError, SourceProvenanceError):
            errors.append({"code": "unsafe_relevant_source", "path": relative})
            size, content_sha256 = 0, ""
        relevant_untracked.append(
            {
                "path": relative,
                "status": status_name,
                "size_bytes": size,
                "content_sha256": content_sha256,
            }
        )
    for relative in sorted(expected_paths - all_tracked):
        if relative not in working_paths:
            errors.append({"code": "missing_scope_path", "path": relative})

    entries: list[dict[str, Any]] = []
    entries_exact = True
    for relative in tracked_scope:
        try:
            index_mode, index_oid = _index_entry(repo, relative)
            head_mode, head_oid = _head_entry(repo, relative)
            working = _safe_repo_path(repo, relative)
            size, content_sha256 = _safe_hash_path(working)
            head_content_sha256 = _sha256(_git(repo, "show", f"HEAD:{relative}"))
            exact = (
                index_mode in {"100644", "100755"}
                and index_mode == head_mode
                and index_oid == head_oid
                and content_sha256 == head_content_sha256
            )
        except (OSError, SourceProvenanceError, UnicodeError) as exc:
            errors.append({"code": "scope_entry_unreadable", "path": relative})
            index_mode = ""
            index_oid = ""
            head_oid = ""
            size = 0
            content_sha256 = ""
            exact = False
            if isinstance(exc, SourceProvenanceError) and "symlink" in str(exc):
                errors[-1]["code"] = "scope_entry_symlink"
        entries_exact = entries_exact and exact
        entries.append(
            {
                "path": relative,
                "mode": index_mode,
                "index_blob_oid": index_oid,
                "head_blob_oid": head_oid,
                "size_bytes": size,
                "content_sha256": content_sha256,
            }
        )

    scope_entry = next(
        (entry for entry in entries if entry["path"] == scope_repo_path), None
    )
    if scope_entry is None or scope_entry["content_sha256"] != scope_config_sha256:
        errors.append({"code": "scope_snapshot_mismatch", "path": scope_repo_path})
        entries_exact = False

    path_set_sha256 = _sha256(canonical.canonical_bytes([e["path"] for e in entries]))
    index_entries_sha256 = _sha256(
        canonical.canonical_bytes(
            [
                {
                    "path": entry["path"],
                    "mode": entry["mode"],
                    "index_blob_oid": entry["index_blob_oid"],
                    "head_blob_oid": entry["head_blob_oid"],
                }
                for entry in entries
            ]
        )
    )
    content_entries_sha256 = _sha256(
        canonical.canonical_bytes(
            [
                {
                    "path": entry["path"],
                    "size_bytes": entry["size_bytes"],
                    "content_sha256": entry["content_sha256"],
                }
                for entry in entries
            ]
        )
    )
    provenance_complete = not errors
    source_clean = (
        provenance_complete
        and not tracked_status
        and entries_exact
        and not relevant_untracked
    )
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "git_commit": git_commit,
        "head_tree": head_tree,
        "scope_config_sha256": scope_config_sha256,
        "path_set_sha256": path_set_sha256,
        "index_entries_sha256": index_entries_sha256,
        "content_entries_sha256": content_entries_sha256,
        "tracked_status_sha256": _sha256(tracked_status),
        "tracked_worktree_clean": not tracked_status,
        "provenance_complete": provenance_complete,
        "source_provenance_clean": source_clean,
        "entries": entries,
        "relevant_untracked_files": relevant_untracked,
        "errors": errors,
    }
    manifest = canonical.materialize_identity(payload, "manifest_id")
    return manifest, _worktree_observation(repo, git_commit)


def publish_source_snapshot(
    repo_root: str | os.PathLike[str],
    scope_config: str | os.PathLike[str],
    *,
    scope_config_repo_path: str,
    manifest_output: str | os.PathLike[str],
    observation_output: str | os.PathLike[str],
    canonical_module: ModuleType | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = _canonical_module(canonical_module)
    manifest_path = Path(manifest_output)
    observation_path = Path(observation_output)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)
    if observation_path.exists() or observation_path.is_symlink():
        raise FileExistsError(observation_path)
    manifest, observation = build_source_snapshot(
        repo_root,
        scope_config,
        scope_config_repo_path=scope_config_repo_path,
        canonical_module=canonical,
    )
    canonical.publish_canonical_no_clobber(observation_path, observation)
    canonical.publish_canonical_no_clobber(manifest_path, manifest)
    return manifest, observation


def create_source_manifest(
    *args: Any, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility name for the source snapshot library API."""

    return publish_source_snapshot(*args, **kwargs)


def validate_manifest(
    document: Any,
    *,
    require_schema: str = MANIFEST_SCHEMA,
    canonical_module: ModuleType | None = None,
) -> bool:
    canonical = _canonical_module(canonical_module)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != require_schema
    ):
        return False
    required = {
        "schema_version",
        "manifest_id",
        "git_commit",
        "head_tree",
        "scope_config_sha256",
        "path_set_sha256",
        "index_entries_sha256",
        "content_entries_sha256",
        "tracked_status_sha256",
        "source_provenance_clean",
        "entries",
    }
    entries = document.get("entries")
    if not required.issubset(document) or not isinstance(entries, list):
        return False
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(paths) != len(entries) or paths != sorted(set(paths)):
        return False
    return canonical.validate_identity(document, "manifest_id")


def verify_current(
    repo_root: str | os.PathLike[str],
    scope_config: str | os.PathLike[str],
    *,
    scope_config_repo_path: str,
    baseline: dict[str, Any] | str | os.PathLike[str],
    evidence_output: str | os.PathLike[str],
    current_manifest_output: str | os.PathLike[str] | None = None,
    current_observation_output: str | os.PathLike[str] | None = None,
    canonical_module: ModuleType | None = None,
) -> dict[str, Any]:
    canonical = _canonical_module(canonical_module)
    baseline_document = (
        canonical.load_json_strict(baseline)
        if isinstance(baseline, (str, os.PathLike))
        else baseline
    )
    if not validate_manifest(baseline_document, canonical_module=canonical):
        raise SourceProvenanceError("baseline source manifest is invalid")
    current, observation = build_source_snapshot(
        repo_root,
        scope_config,
        scope_config_repo_path=scope_config_repo_path,
        canonical_module=canonical,
    )
    if current_manifest_output is not None:
        canonical.publish_canonical_no_clobber(current_manifest_output, current)
    if current_observation_output is not None:
        canonical.publish_canonical_no_clobber(current_observation_output, observation)
    evidence = {
        "schema_version": RECHECK_SCHEMA,
        "complete": current.get("provenance_complete") is True,
        "exact_match": current == baseline_document,
        "git_commit_match": current["git_commit"] == baseline_document["git_commit"],
        "path_set_match": current["path_set_sha256"]
        == baseline_document["path_set_sha256"],
        "index_entries_match": current["index_entries_sha256"]
        == baseline_document["index_entries_sha256"],
        "content_entries_match": current["content_entries_sha256"]
        == baseline_document["content_entries_sha256"],
        "current_source_provenance_clean": current["source_provenance_clean"],
        "baseline_manifest_id": baseline_document["manifest_id"],
        "current_manifest_id": current["manifest_id"],
        "current_manifest": current,
        "worktree_observation": observation,
    }
    canonical.publish_canonical_no_clobber(evidence_output, evidence)
    return evidence


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", "--repo-root", dest="repo", required=True, type=Path)
    parser.add_argument(
        "--scope-config", "--scope", dest="scope", required=True, type=Path
    )
    parser.add_argument("--scope-config-repo-path", required=True)
    parser.add_argument("--canonical-helper", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", aliases=["create"])
    _add_source_arguments(snapshot)
    snapshot.add_argument(
        "--output", "--manifest-out", dest="manifest", required=True, type=Path
    )
    snapshot.add_argument(
        "--worktree-observation-output",
        "--observation-out",
        dest="observation",
        required=True,
        type=Path,
    )

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--canonical-helper", required=True, type=Path)
    validate.add_argument("--require-schema", default=MANIFEST_SCHEMA)

    verify = subparsers.add_parser("verify-current", aliases=["verify"])
    _add_source_arguments(verify)
    verify.add_argument("--baseline", required=True, type=Path)
    verify.add_argument("--evidence-output", required=True, type=Path)
    verify.add_argument("--current-manifest-output", type=Path)
    verify.add_argument("--current-worktree-observation-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        canonical = _load_canonical_helper(args.canonical_helper.resolve(strict=True))
        if args.command == "validate-manifest":
            document = canonical.load_json_strict(args.manifest)
            return (
                0
                if validate_manifest(
                    document,
                    require_schema=args.require_schema,
                    canonical_module=canonical,
                )
                else 2
            )
        if args.command in {"snapshot", "create"}:
            publish_source_snapshot(
                args.repo,
                args.scope,
                scope_config_repo_path=args.scope_config_repo_path,
                manifest_output=args.manifest,
                observation_output=args.observation,
                canonical_module=canonical,
            )
            return 0
        evidence = verify_current(
            args.repo,
            args.scope,
            scope_config_repo_path=args.scope_config_repo_path,
            baseline=args.baseline,
            evidence_output=args.evidence_output,
            current_manifest_output=args.current_manifest_output,
            current_observation_output=args.current_worktree_observation_output,
            canonical_module=canonical,
        )
        return (
            0
            if evidence["complete"]
            and evidence["exact_match"]
            and evidence["current_source_provenance_clean"]
            else 3
        )
    except (FileExistsError, OSError, SourceProvenanceError, ValueError) as exc:
        print(f"source-provenance: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
