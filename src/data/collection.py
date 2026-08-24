"""Auditable M2 split collection with durable per-attempt accounting."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from env import scene as env_scene
from env.asset_provenance import collect_asset_provenance
from env.pick_place import PickPlace
from expert.evaluate import _git_state
from expert.scripted import ExpertConfig, config_dict, run_episode

from .hdf5 import EpisodePublicationError, HDF5EpisodeWriter
from .manifest import (
    COLLECTION_SCHEMA,
    EPISODES_ROOT,
    LEDGER_FILENAME,
    MANIFEST_FILENAME,
    append_jsonl_fsync,
    assets_payload,
    atomic_write_json_no_clobber,
    compiled_model_fingerprint,
    environment_payload,
    formal_target_successes,
    fsync_directory,
    initialize_jsonl_no_clobber,
    inspect_episode,
    load_jsonl_relative,
    maximum_episode_steps,
    split_contract,
    stable_sha256_relative_file,
    validate_collection_manifest,
    validate_relative_path,
    validate_split_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class FormalCollectionError(RuntimeError):
    """Formal collection provenance is not clean and fixed before output."""


class CollectionExhaustedError(RuntimeError):
    """A split namespace ended before reaching the requested successes."""


@dataclass(frozen=True)
class CollectionConfig:
    output_dir: Path
    split: str
    target_successes: int | None = None
    smoke: bool = False
    diagnostic_allow_dirty: bool = False
    project_root: Path = PROJECT_ROOT
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)


def collect_split(
    config: CollectionConfig,
    *,
    env_factory: Callable[[], Any] = PickPlace,
    episode_runner: Callable[..., Any] = run_episode,
    git_state_fn: Callable[[Path], dict[str, Any]] = _git_state,
    asset_provenance_fn: Callable[..., Any] = collect_asset_provenance,
    environment_fingerprint_fn: Callable[[Any], dict[str, Any]] = (
        compiled_model_fingerprint
    ),
    writer_factory: Callable[..., HDF5EpisodeWriter] = HDF5EpisodeWriter,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Collect one frozen M2 split into a brand-new directory.

    Expert failures are valid, retained HDF5 episodes. Exceptions are durable
    ledger outcomes and stop the run without publishing a final manifest.
    """

    normalized = _validated_config(config)
    output_dir = Path(normalized.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    git = git_state_fn(Path(normalized.project_root))
    formal = not normalized.smoke and not normalized.diagnostic_allow_dirty
    if formal:
        _require_formal_git_state(git)

    provenance = asset_provenance_fn(
        Path(normalized.project_root),
        runtime_asset_root=env_scene.MENAGERIE,
    )
    assets = assets_payload(provenance)
    env = env_factory()
    try:
        if getattr(getattr(env, "cfg", None), "debug_viz", None) is not False:
            raise ValueError("M2 collection requires env.cfg.debug_viz exactly false")
        model_fingerprint = environment_fingerprint_fn(env)
        environment = environment_payload(env, model_fingerprint)

        output_dir.mkdir(parents=True, exist_ok=False)
        episodes_dir = output_dir / EPISODES_ROOT
        episodes_dir.mkdir()
        ledger_path = output_dir / LEDGER_FILENAME
        ledger_state = initialize_jsonl_no_clobber(ledger_path)

        attempts: list[dict[str, Any]] = []
        successes = 0
        contract = split_contract(normalized.split)
        seed = int(contract["scan_start"])
        maximum = int(contract["candidate_seed_max"])

        while successes < normalized.target_successes:
            if seed > maximum:
                raise CollectionExhaustedError(
                    f"{normalized.split} namespace exhausted at {maximum} with "
                    f"{successes}/{normalized.target_successes} successes"
                )
            validate_split_seed(normalized.split, seed)
            relative_path = f"{EPISODES_ROOT}/seed_{seed:06d}.h5"
            episode_path = output_dir / relative_path
            attempt_index = len(attempts)
            result: Any | None = None
            try:
                with writer_factory(episode_path, seed=seed) as writer:
                    result = episode_runner(
                        env,
                        seed=seed,
                        config=normalized.expert_config,
                        step_callback=writer.capture,
                    )
                    writer.finalize(
                        success=result.success,
                        failure_stage=result.failure_stage,
                    )
            except EpisodePublicationError as error:
                record = _publication_indeterminate_record(
                    attempt_index=attempt_index,
                    seed=seed,
                    output_dir=output_dir,
                    relative_path=relative_path,
                    result=result,
                    error=error,
                )
                ledger_state = append_jsonl_fsync(
                    ledger_path,
                    record,
                    expected_snapshot=ledger_state,
                )
                raise
            except BaseException as error:
                record = _exception_record(
                    attempt_index=attempt_index,
                    seed=seed,
                    error=error,
                )
                ledger_state = append_jsonl_fsync(
                    ledger_path,
                    record,
                    expected_snapshot=ledger_state,
                )
                raise

            try:
                inspection = inspect_episode(
                    output_dir,
                    relative_path,
                    max_num_steps=maximum_episode_steps(
                        config_dict(normalized.expert_config)
                    ),
                )
                inspection.validation.raise_for_errors()
                metadata = inspection.metadata
                if metadata is None:
                    raise RuntimeError("published episode has no valid metadata")
                if result is None:
                    raise RuntimeError("episode runner returned no result")
                if (
                    metadata.seed != seed
                    or metadata.success is not bool(result.success)
                    or metadata.failure_stage != result.failure_stage
                    or metadata.num_steps != int(result.control_steps)
                ):
                    raise RuntimeError(
                        "published HDF5 metadata disagrees with expert result"
                    )
                success = bool(result.success)
                record = {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "status": "success" if success else "failure",
                    "success": success,
                    "failure_stage": result.failure_stage,
                    "path": relative_path,
                    "num_steps": metadata.num_steps,
                    "sha256": inspection.sha256,
                    "error": None,
                }
            except BaseException as error:
                record = _postpublication_exception_record(
                    attempt_index=attempt_index,
                    seed=seed,
                    relative_path=relative_path,
                    result=result,
                    error=error,
                )
                ledger_state = append_jsonl_fsync(
                    ledger_path,
                    record,
                    expected_snapshot=ledger_state,
                )
                raise
            ledger_state = append_jsonl_fsync(
                ledger_path,
                record,
                expected_snapshot=ledger_state,
            )
            attempts.append(record)
            successes += int(success)
            seed += 1

        eligible = [
            {
                "seed": attempt["seed"],
                "path": attempt["path"],
                "num_steps": attempt["num_steps"],
                "sha256": attempt["sha256"],
            }
            for attempt in attempts
            if attempt["success"]
        ]
        if formal:
            current_git = git_state_fn(Path(normalized.project_root))
            _require_formal_git_state(current_git)
            current_provenance = asset_provenance_fn(
                Path(normalized.project_root),
                runtime_asset_root=env_scene.MENAGERIE,
            )
            current_assets = assets_payload(current_provenance)
            current_environment = environment_payload(
                env,
                environment_fingerprint_fn(env),
            )
            if (
                current_git != git
                or current_assets != assets
                or current_environment != environment
            ):
                raise FormalCollectionError(
                    "formal runtime provenance changed during collection; "
                    "manifest was not published"
                )
        timestamp = (now_fn or (lambda: datetime.now(UTC)))()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        manifest: dict[str, Any] = {
            "schema_version": COLLECTION_SCHEMA,
            "split_protocol": split_contract(normalized.split),
            "split": normalized.split,
            "formal": formal,
            "target_successes": normalized.target_successes,
            "attempt_count": len(attempts),
            "success_count": successes,
            "attempts": attempts,
            "eligible_successes": eligible,
            "controller": config_dict(normalized.expert_config),
            "git": git,
            "assets": assets,
            "environment": environment,
            "generated_at": timestamp.astimezone(UTC).isoformat(),
            "cli_config": {
                "split": normalized.split,
                "target_successes": normalized.target_successes,
                "smoke": normalized.smoke,
                "diagnostic_allow_dirty": normalized.diagnostic_allow_dirty,
            },
            "ledger_path": LEDGER_FILENAME,
            "episodes_root": EPISODES_ROOT,
        }
        atomic_write_json_no_clobber(output_dir / MANIFEST_FILENAME, manifest)
        report = validate_collection_manifest(output_dir / MANIFEST_FILENAME)
        if not report.valid:
            raise RuntimeError(
                "new collection manifest failed self-validation:\n"
                + report.format_errors()
            )
        return manifest
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def reconcile_indeterminate_run(
    run_root: Path,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Verify one indeterminate publication and remove only its same-inode partial.

    Reconciliation never resumes collection, retries a seed, deletes the target,
    or publishes a manifest.  It writes a separate no-clobber receipt after the
    durable ledger record has been checked against the target HDF5 and SHA-256.
    """

    root = Path(run_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reconciliation run root must be a real directory")
    if (root / MANIFEST_FILENAME).exists() or (root / MANIFEST_FILENAME).is_symlink():
        raise ValueError("a finalized manifest cannot be publication-reconciled")
    records = load_jsonl_relative(root, LEDGER_FILENAME)
    if not records or not isinstance(records[-1], dict):
        raise ValueError("ledger has no final publication-indeterminate record")
    record = records[-1]
    if record.get("status") != "publication_indeterminate":
        raise ValueError("ledger does not end in publication_indeterminate")
    if (
        sum(
            isinstance(item, dict) and item.get("status") == "publication_indeterminate"
            for item in records
        )
        != 1
    ):
        raise ValueError("ledger must contain exactly one indeterminate publication")
    error = record.get("error")
    if not isinstance(error, dict) or error.get("state") != "publication_indeterminate":
        raise ValueError("ledger publication evidence is incomplete")
    if not (
        error.get("target_matches_source") is True
        and error.get("target_valid") is True
        and error.get("target_path") == record.get("path")
        and error.get("target_sha256") == record.get("sha256")
    ):
        raise ValueError("ledger target evidence is not a verified publication")
    relative_target = record.get("path")
    if not isinstance(relative_target, str):
        raise ValueError("indeterminate record has no target path")
    validate_relative_path(relative_target)
    inspection = inspect_episode(
        root,
        relative_target,
        expected_sha256=record.get("sha256"),
    )
    inspection.validation.raise_for_errors()
    if inspection.sha256 != record.get("sha256"):
        raise ValueError("indeterminate target digest does not match the ledger")
    metadata = inspection.metadata
    if metadata is None or not (
        metadata.seed == record.get("seed")
        and metadata.success is record.get("success")
        and metadata.failure_stage == record.get("failure_stage")
        and metadata.num_steps == record.get("num_steps")
    ):
        raise ValueError("indeterminate target metadata does not match the ledger")

    partial_removed = False
    relative_partial = error.get("partial_path")
    if relative_partial is not None:
        if not isinstance(relative_partial, str):
            raise ValueError("partial_path must be canonical relative text")
        validate_relative_path(relative_partial)
        try:
            partial_digest, partial_snapshot, partial_path = (
                stable_sha256_relative_file(root, relative_partial)
            )
        except Exception as partial_error:
            if getattr(partial_error, "code", None) != "path.missing":
                raise
        else:
            if (
                partial_snapshot.identity != inspection.snapshot.identity
                or partial_digest != inspection.sha256
            ):
                raise ValueError("partial is not the verified target hard link")
            partial_path.unlink()
            fsync_directory(partial_path.parent)
            partial_removed = True

    final_inspection = inspect_episode(
        root,
        relative_target,
        expected_sha256=record["sha256"],
    )
    if (
        final_inspection.sha256 != record["sha256"]
        or not final_inspection.validation.valid
    ):
        raise RuntimeError("target changed during publication reconciliation")
    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    receipt = {
        "schema_version": "m2-publication-reconciliation.v1",
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "status": "verified_publication",
        "attempt_index": record["attempt_index"],
        "seed": record["seed"],
        "target_path": relative_target,
        "target_sha256": record["sha256"],
        "partial_path": relative_partial,
        "partial_hardlink_removed": partial_removed,
        "collection_resumed": False,
        "manifest_published": False,
    }
    atomic_write_json_no_clobber(root / "reconciliation.json", receipt)
    return receipt


def _validated_config(config: CollectionConfig) -> CollectionConfig:
    if config.split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if type(config.diagnostic_allow_dirty) is not bool:
        raise TypeError("diagnostic_allow_dirty must be a boolean")
    formal_target = formal_target_successes(config.split)
    target = (
        formal_target if config.target_successes is None else config.target_successes
    )
    if type(target) is not int or target <= 0:
        raise ValueError("target_successes must be a positive integer")
    if target > formal_target:
        raise ValueError("target_successes cannot exceed the formal split target")
    if target != formal_target and not config.smoke:
        raise ValueError("a smaller target requires the explicit smoke flag")
    return CollectionConfig(
        output_dir=config.output_dir,
        split=config.split,
        target_successes=target,
        smoke=config.smoke,
        diagnostic_allow_dirty=config.diagnostic_allow_dirty,
        project_root=config.project_root,
        expert_config=config.expert_config,
    )


def _require_formal_git_state(git: dict[str, Any]) -> None:
    commit = git.get("commit")
    if not (
        git.get("provenance_complete") is True
        and git.get("source_provenance_clean") is True
        and git.get("tracked_worktree_clean") is True
    ):
        raise FormalCollectionError(
            "formal collection requires complete, clean tracked runtime source provenance"
        )
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise FormalCollectionError("formal collection requires a fixed full Git HEAD")


def _safe_error(error: BaseException) -> dict[str, str]:
    message = str(error).replace("\x00", "\\0")[:2000]
    return {"type": type(error).__name__, "message": message}


def _exception_record(
    *,
    attempt_index: int,
    seed: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "seed": seed,
        "status": "exception",
        "success": None,
        "failure_stage": None,
        "path": None,
        "num_steps": None,
        "sha256": None,
        "error": _safe_error(error),
    }


def _publication_indeterminate_record(
    *,
    attempt_index: int,
    seed: int,
    output_dir: Path,
    relative_path: str,
    result: Any | None,
    error: EpisodePublicationError,
) -> dict[str, Any]:
    published_valid = error.target_matches_source and error.target_valid
    try:
        partial_path = error.partial_path.relative_to(output_dir).as_posix()
    except ValueError:
        partial_path = None
    publication_error: dict[str, Any] = {
        **_safe_error(error),
        "published": error.published,
        "state": error.state,
        "target_matches_source": error.target_matches_source,
        "target_valid": error.target_valid,
        "target_sha256": error.target_sha256,
        "target_path": relative_path,
        "partial_path": partial_path,
    }
    return {
        "attempt_index": attempt_index,
        "seed": seed,
        "status": "publication_indeterminate",
        "success": bool(result.success) if result is not None else None,
        "failure_stage": result.failure_stage if result is not None else None,
        "path": relative_path if published_valid else None,
        "num_steps": int(result.control_steps) if result is not None else None,
        "sha256": error.target_sha256 if published_valid else None,
        "error": publication_error,
    }


def _postpublication_exception_record(
    *,
    attempt_index: int,
    seed: int,
    relative_path: str,
    result: Any | None,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "seed": seed,
        "status": "postpublication_exception",
        "success": bool(result.success) if result is not None else None,
        "failure_stage": result.failure_stage if result is not None else None,
        "path": relative_path,
        "num_steps": int(result.control_steps) if result is not None else None,
        "sha256": None,
        "error": _safe_error(error),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one auditable M2 train or validation split."
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-successes", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow a smaller diagnostic target; manifest is permanently non-formal",
    )
    parser.add_argument(
        "--diagnostic-allow-dirty",
        action="store_true",
        help="allow dirty runtime provenance; manifest is permanently non-formal",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        manifest = collect_split(
            CollectionConfig(
                output_dir=args.output_dir,
                split=args.split,
                target_successes=args.target_successes,
                smoke=args.smoke,
                diagnostic_allow_dirty=args.diagnostic_allow_dirty,
            )
        )
    except (FileExistsError, FormalCollectionError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
