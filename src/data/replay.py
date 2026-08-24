"""Deterministic action-only replay for eligible M2 expert episodes."""

from __future__ import annotations

import argparse
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

from .manifest import (
    _validate_assets,
    _validate_environment,
    _validate_git,
    append_jsonl_fsync,
    assets_payload,
    atomic_write_json_no_clobber,
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
REPLAY_TRIALS_FILENAME = "trials.jsonl"
REPLAY_SUMMARY_FILENAME = "summary.json"
FORMAL_REPLAY_COUNT = 20
FORMAL_REPLAY_REQUIRED = 18
FORMAL_SELECTION_SEED = 20260824
SELECTION_ALGORITHM = "numpy-pcg64-permutation.v1"
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=FORMAL_SELECTION_SEED)
    parser.add_argument("--count", type=int, default=FORMAL_REPLAY_COUNT)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
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
