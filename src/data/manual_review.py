"""Deterministic M2 human-review packs; this module never invents judgments."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .manifest import (
    append_jsonl_fsync,
    atomic_write_json_no_clobber,
    initialize_jsonl_no_clobber,
    load_json_object_with_sha256,
    load_jsonl_relative_with_sha256,
    manifest_sha256,
    maximum_episode_steps,
    open_verified_episode,
    validate_relative_path,
)
from .replay import (
    FORMAL_REPLAY_COUNT,
    FORMAL_SELECTION_SEED,
    _content_id,
    _is_content_id,
    _is_sha256,
    _require_unchanged_source_manifests,
    _validated_pair_sources,
)
from .reporting import _contact_sheet, _review_frame_indices, _select_manual_candidates

REVIEW_PACK_SCHEMA = "m2-human-review-pack.v1"
REVIEW_TRIAL_SCHEMA = "m2-manual-review-trial.v1"
REVIEW_PACK_FILENAME = "review-pack.json"
REVIEW_TEMPLATE_FILENAME = "manual-review-template.jsonl"
CONTACT_SHEETS_DIR = "contact_sheets"
REVIEW_SELECTION_ALGORITHM = "failure-first-length-action-outlier.v1"
FORMAL_REVIEW_COUNT = FORMAL_REPLAY_COUNT
FORMAL_REVIEW_SEED = FORMAL_SELECTION_SEED

_PACK_KEYS = {
    "schema_version",
    "pack_id",
    "generated_at",
    "formal",
    "status",
    "manifests",
    "selection",
    "candidate_count",
    "selected_reviews",
    "template",
    "cli_config",
}
_MANIFEST_REF_KEYS = {
    "split",
    "manifest_id",
    "file_sha256",
    "formal",
    "attempt_count",
}
_SELECTED_REVIEW_KEYS = {
    "manual_review_id",
    "manifest_id",
    "split",
    "attempt_index",
    "seed",
    "source_relative_path",
    "source_file_sha256",
    "source_num_steps",
    "outcome",
    "failure_stage",
    "classification",
    "reasons",
    "outlier_score",
    "frame_indices",
    "media",
}
_TEMPLATE_KEYS = {
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


@dataclass(frozen=True)
class ManualReviewPackConfig:
    train_manifest_path: Path
    validation_manifest_path: Path
    output_dir: Path
    count: int = FORMAL_REVIEW_COUNT
    selection_seed: int = FORMAL_REVIEW_SEED
    smoke: bool = False


@dataclass(frozen=True)
class ManualReviewValidationIssue:
    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class ManualReviewPackValidationReport:
    path: Path
    errors: tuple[ManualReviewValidationIssue, ...]
    pack: dict[str, Any] | None = None
    sha256: str | None = None
    status: str = "invalid"
    complete: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(error) for error in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                f"invalid manual review pack {self.path}:\n{self.format_errors()}"
            )


def create_manual_review_pack(
    config: ManualReviewPackConfig,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Create immutable media and an unsigned, empty human-review template."""

    _validate_pack_config(config)
    train_path = Path(config.train_manifest_path).expanduser().absolute()
    validation_path = Path(config.validation_manifest_path).expanduser().absolute()
    output_dir = Path(config.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"manual review output already exists: {output_dir}")
    sources = _validated_pair_sources(train_path, validation_path)
    _require_formal_population(sources, smoke=config.smoke)
    candidate_count = sum(len(source[1]["attempts"]) for source in sources.values())
    if config.count > candidate_count:
        raise ValueError("manual review count exceeds collected attempts")

    selected = _select_pair_reviews(
        sources,
        count=config.count,
        selection_seed=config.selection_seed,
    )
    prepared: list[tuple[dict[str, Any], np.ndarray]] = []
    for selected_index, item in enumerate(selected):
        split = item["attempt"]["split"]
        source_path, source_manifest, _digest = sources[split]
        attempt = item["attempt"]
        with open_verified_episode(
            source_path.parent,
            attempt["path"],
            expected_sha256=attempt["sha256"],
            max_num_steps=maximum_episode_steps(source_manifest["controller"]),
        ) as episode:
            episode.validation.raise_for_errors()
            if episode.metadata is None:
                raise RuntimeError("episode metadata unavailable for human review")
            frame_indices = _review_frame_indices(episode.metadata.num_steps)
            front = np.stack(
                [
                    episode.handle["observation.images.front"][index]
                    for index in frame_indices
                ]
            )
            wrist = np.stack(
                [
                    episode.handle["observation.images.wrist"][index]
                    for index in frame_indices
                ]
            )
            sheet = _contact_sheet(front, wrist)
        media_path = (
            f"{CONTACT_SHEETS_DIR}/{split}_attempt_{attempt['attempt_index']:06d}_"
            f"seed_{attempt['seed']:06d}.png"
        )
        review_identity = {
            "schema_version": "m2-manual-review-identity.v1",
            "selection_algorithm": REVIEW_SELECTION_ALGORITHM,
            "selection_seed": config.selection_seed,
            "selected_index": selected_index,
            "manifest_id": attempt["manifest_id"],
            "split": split,
            "attempt_index": attempt["attempt_index"],
            "seed": attempt["seed"],
            "source_relative_path": attempt["path"],
            "source_file_sha256": attempt["sha256"],
            "source_num_steps": attempt["num_steps"],
            "classification": "failure" if not attempt["success"] else "anomaly",
        }
        prepared.append(
            (
                {
                    "manual_review_id": _content_id(review_identity),
                    "manifest_id": attempt["manifest_id"],
                    "split": split,
                    "attempt_index": attempt["attempt_index"],
                    "seed": attempt["seed"],
                    "source_relative_path": attempt["path"],
                    "source_file_sha256": attempt["sha256"],
                    "source_num_steps": attempt["num_steps"],
                    "outcome": attempt["status"],
                    "failure_stage": attempt["failure_stage"],
                    "classification": review_identity["classification"],
                    "reasons": item["reasons"],
                    "outlier_score": item["outlier_score"],
                    "frame_indices": frame_indices,
                    "media": {"path": media_path, "sha256": None},
                },
                sheet,
            )
        )

    _require_unchanged_source_manifests(sources)
    output_dir.mkdir(parents=True, exist_ok=False)
    media_dir = output_dir / CONTACT_SHEETS_DIR
    media_dir.mkdir()
    selected_reviews: list[dict[str, Any]] = []
    for review, sheet in prepared:
        media_path = output_dir / review["media"]["path"]
        if media_path.exists() or media_path.is_symlink():
            raise FileExistsError(f"manual review media already exists: {media_path}")
        imageio.imwrite(media_path, sheet, format="png")
        review["media"]["sha256"] = manifest_sha256(media_path)
        selected_reviews.append(review)

    manifest_refs = _manifest_references(sources)
    selection = {
        "algorithm": REVIEW_SELECTION_ALGORITHM,
        "seed": config.selection_seed,
        "count": config.count,
    }
    pack_identity = {
        "schema_version": REVIEW_PACK_SCHEMA,
        "formal": not config.smoke,
        "manifests": manifest_refs,
        "selection": selection,
        "candidate_count": candidate_count,
        "selected_reviews": selected_reviews,
        "cli_config": {
            "count": config.count,
            "selection_seed": config.selection_seed,
            "smoke": config.smoke,
        },
    }
    pack_id = _content_id(pack_identity)
    template_path = output_dir / REVIEW_TEMPLATE_FILENAME
    template_state = initialize_jsonl_no_clobber(template_path)
    for review in selected_reviews:
        template_state = append_jsonl_fsync(
            template_path,
            _template_row(pack_id, review),
            expected_snapshot=template_state,
        )

    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    pack = {
        **pack_identity,
        "pack_id": pack_id,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "status": "awaiting_human_review",
        "template": {
            "path": REVIEW_TEMPLATE_FILENAME,
            "sha256": manifest_sha256(template_path),
            "row_count": len(selected_reviews),
        },
    }
    atomic_write_json_no_clobber(output_dir / REVIEW_PACK_FILENAME, pack)
    return pack


def validate_manual_review_pack(
    path: Path,
    *,
    train_manifest_path: Path,
    validation_manifest_path: Path,
) -> ManualReviewPackValidationReport:
    pack_path = Path(path)
    errors: list[ManualReviewValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        errors.append(ManualReviewValidationIssue(code, location, message))

    try:
        pack, initial_sha = load_json_object_with_sha256(pack_path)
    except (OSError, UnicodeError, ValueError) as error:
        add("manual.pack", "/", f"cannot read strict pack JSON: {error}")
        return ManualReviewPackValidationReport(pack_path, tuple(errors))
    _validate_pack_schema(pack, add)
    if errors:
        return ManualReviewPackValidationReport(
            pack_path,
            tuple(errors),
            pack,
            initial_sha,
        )

    try:
        sources = _validated_pair_sources(
            Path(train_manifest_path).expanduser().absolute(),
            Path(validation_manifest_path).expanduser().absolute(),
        )
        _require_formal_population(sources, smoke=pack["cli_config"]["smoke"])
        if pack["manifests"] != _manifest_references(sources):
            add(
                "manual.pack.source",
                "/manifests",
                "manifest references differ from source files",
            )
        expected_selected = _select_pair_reviews(
            sources,
            count=pack["selection"]["count"],
            selection_seed=pack["selection"]["seed"],
        )
        _validate_selected_media(
            pack_path.parent, pack, expected_selected, sources, add
        )
        _require_unchanged_source_manifests(sources)
    except (OSError, TypeError, ValueError) as error:
        add("manual.pack.source", "/", str(error))

    try:
        rows, rows_digest, _, _ = load_jsonl_relative_with_sha256(
            pack_path.parent,
            pack["template"]["path"],
        )
    except (OSError, UnicodeError, ValueError) as error:
        add("manual.pack.template", "/template/path", str(error))
        rows = []
        rows_digest = None
    if rows_digest is not None and rows_digest != pack["template"]["sha256"]:
        add(
            "manual.pack.template",
            "/template/sha256",
            "template digest differs",
        )
    if len(rows) != pack["template"]["row_count"]:
        add(
            "manual.pack.template",
            "/template/row_count",
            "template row count differs",
        )
    for index, (row, review) in enumerate(
        zip(rows, pack["selected_reviews"], strict=False)
    ):
        expected = _template_row(pack["pack_id"], review)
        if row != expected:
            add(
                "manual.pack.template",
                f"/template/rows/{index}",
                "template row is not the exact empty review template",
            )

    expected_pack_id = _content_id(_pack_identity_payload(pack))
    if pack["pack_id"] != expected_pack_id:
        add("manual.pack.identity", "/pack_id", "pack identity is wrong")
    try:
        _, final_sha = load_json_object_with_sha256(pack_path)
        if final_sha != initial_sha:
            add("manual.pack", "/", "pack changed during validation")
    except (OSError, UnicodeError, ValueError) as error:
        add("manual.pack", "/", f"cannot complete stable validation: {error}")
    status = "awaiting_human_review" if not errors else "invalid"
    return ManualReviewPackValidationReport(
        pack_path,
        tuple(errors),
        pack,
        initial_sha,
        status=status,
        complete=False,
    )


def _select_pair_reviews(
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
    *,
    count: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_steps = 0
    action_sum = np.zeros(8, dtype=np.float64)
    action_sum_squares = np.zeros(8, dtype=np.float64)
    for split in ("train", "validation"):
        source_path, manifest, digest = sources[split]
        manifest_id = f"sha256:{digest}"
        max_steps = maximum_episode_steps(manifest["controller"])
        for source_attempt in manifest["attempts"]:
            attempt = {**source_attempt, "split": split, "manifest_id": manifest_id}
            with open_verified_episode(
                source_path.parent,
                attempt["path"],
                expected_sha256=attempt["sha256"],
                max_num_steps=max_steps,
            ) as episode:
                dataset = episode.handle["action"]
                num_steps = int(dataset.shape[0])
                episode_sum = np.zeros(8, dtype=np.float64)
                for start in range(0, num_steps, 256):
                    stop = min(start + 256, num_steps)
                    block = np.asarray(dataset[start:stop], dtype=np.float64)
                    block_sum = np.sum(block, axis=0, dtype=np.float64)
                    episode_sum += block_sum
                    action_sum += block_sum
                    action_sum_squares += np.sum(
                        np.square(block),
                        axis=0,
                        dtype=np.float64,
                    )
                total_steps += num_steps
                rows.append(
                    {
                        "attempt": attempt,
                        "num_steps": num_steps,
                        "action_mean": episode_sum / num_steps,
                    }
                )
    if total_steps <= 0 or not rows:
        raise ValueError("manual review requires non-empty collected episodes")
    lengths = np.asarray([row["num_steps"] for row in rows], dtype=np.float64)
    mean = action_sum / total_steps
    variance = np.maximum(action_sum_squares / total_steps - np.square(mean), 0.0)
    return _select_manual_candidates(
        rows,
        count=count,
        selection_seed=selection_seed,
        global_action_mean=mean,
        global_action_std=np.sqrt(variance),
        length_mean=float(np.mean(lengths)),
        length_std=float(np.std(lengths)),
    )


def _validate_selected_media(
    output_root: Path,
    pack: Mapping[str, Any],
    expected_selected: Sequence[Mapping[str, Any]],
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
    add: Any,
) -> None:
    if len(expected_selected) != len(pack["selected_reviews"]):
        add(
            "manual.pack.selection",
            "/selected_reviews",
            "selected review count differs from deterministic reconstruction",
        )
        return
    for index, (actual, selected) in enumerate(
        zip(pack["selected_reviews"], expected_selected, strict=True)
    ):
        location = f"/selected_reviews/{index}"
        attempt = selected["attempt"]
        expected_fields = {
            "manifest_id": attempt["manifest_id"],
            "split": attempt["split"],
            "attempt_index": attempt["attempt_index"],
            "seed": attempt["seed"],
            "source_relative_path": attempt["path"],
            "source_file_sha256": attempt["sha256"],
            "source_num_steps": attempt["num_steps"],
            "outcome": attempt["status"],
            "failure_stage": attempt["failure_stage"],
            "classification": "failure" if not attempt["success"] else "anomaly",
            "reasons": selected["reasons"],
            "outlier_score": selected["outlier_score"],
            "frame_indices": _review_frame_indices(attempt["num_steps"]),
        }
        for field, expected in expected_fields.items():
            if actual[field] != expected or type(actual[field]) is not type(expected):
                add(
                    "manual.pack.selection",
                    f"{location}/{field}",
                    "selected review differs from deterministic reconstruction",
                )
        expected_identity = {
            "schema_version": "m2-manual-review-identity.v1",
            "selection_algorithm": REVIEW_SELECTION_ALGORITHM,
            "selection_seed": pack["selection"]["seed"],
            "selected_index": index,
            **{
                key: expected_fields[key]
                for key in (
                    "manifest_id",
                    "split",
                    "attempt_index",
                    "seed",
                    "source_relative_path",
                    "source_file_sha256",
                    "source_num_steps",
                    "classification",
                )
            },
        }
        if actual["manual_review_id"] != _content_id(expected_identity):
            add(
                "manual.pack.identity",
                f"{location}/manual_review_id",
                "manual review identity is wrong",
            )
        media = actual["media"]
        try:
            media_path = output_root / validate_relative_path(media["path"])
            digest = manifest_sha256(media_path)
            if digest != media["sha256"]:
                add(
                    "manual.pack.media",
                    f"{location}/media/sha256",
                    "media digest differs",
                )
            source_path, source_manifest, _digest = sources[attempt["split"]]
            with open_verified_episode(
                source_path.parent,
                attempt["path"],
                expected_sha256=attempt["sha256"],
                max_num_steps=maximum_episode_steps(source_manifest["controller"]),
            ) as episode:
                indices = expected_fields["frame_indices"]
                expected_sheet = _contact_sheet(
                    np.stack(
                        [
                            episode.handle["observation.images.front"][frame]
                            for frame in indices
                        ]
                    ),
                    np.stack(
                        [
                            episode.handle["observation.images.wrist"][frame]
                            for frame in indices
                        ]
                    ),
                )
            decoded = np.asarray(imageio.imread(media_path), dtype=np.uint8)
            if not np.array_equal(decoded, expected_sheet):
                add(
                    "manual.pack.media",
                    f"{location}/media/path",
                    "decoded media pixels differ from source frames",
                )
        except (OSError, TypeError, ValueError) as error:
            add("manual.pack.media", f"{location}/media", str(error))


def _manifest_references(
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "manifest_id": f"sha256:{sources[split][2]}",
            "file_sha256": sources[split][2],
            "formal": sources[split][1]["formal"],
            "attempt_count": sources[split][1]["attempt_count"],
        }
        for split in ("train", "validation")
    ]


def _template_row(pack_id: str, review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_TRIAL_SCHEMA,
        "manual_review_id": review["manual_review_id"],
        "review_pack_id": pack_id,
        "manifest_id": review["manifest_id"],
        "split": review["split"],
        "attempt_index": review["attempt_index"],
        "seed": review["seed"],
        "source_relative_path": review["source_relative_path"],
        "source_file_sha256": review["source_file_sha256"],
        "source_num_steps": review["source_num_steps"],
        "classification": review["classification"],
        "media": review["media"],
        "reviewer_id": None,
        "review_started_at_utc": None,
        "review_completed_at_utc": None,
        "finding": None,
        "verdict": None,
    }


def _pack_identity_payload(pack: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": pack["schema_version"],
        "formal": pack["formal"],
        "manifests": pack["manifests"],
        "selection": pack["selection"],
        "candidate_count": pack["candidate_count"],
        "selected_reviews": pack["selected_reviews"],
        "cli_config": pack["cli_config"],
    }


def _require_formal_population(
    sources: Mapping[str, tuple[Path, dict[str, Any], str]],
    *,
    smoke: bool,
) -> None:
    if smoke:
        return
    train = sources["train"][1]
    validation = sources["validation"][1]
    if not (
        train["formal"] is True
        and validation["formal"] is True
        and len(train["eligible_successes"]) == 200
        and len(validation["eligible_successes"]) == 40
    ):
        raise ValueError(
            "formal manual review requires exactly 200 train and 40 validation "
            "eligible successes from formal manifests"
        )


def _validate_pack_config(config: ManualReviewPackConfig) -> None:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if type(config.count) is not int or not 1 <= config.count <= FORMAL_REVIEW_COUNT:
        raise ValueError("manual review count must be in 1..20")
    if config.count != FORMAL_REVIEW_COUNT and not config.smoke:
        raise ValueError("a manual review count smaller than 20 requires --smoke")
    if type(config.selection_seed) is not int or config.selection_seed < 0:
        raise ValueError("manual review selection seed must be non-negative")
    if config.selection_seed != FORMAL_REVIEW_SEED and not config.smoke:
        raise ValueError("a noncanonical manual review seed requires --smoke")


def _validate_pack_schema(pack: dict[str, Any], add: Any) -> None:
    if set(pack) != _PACK_KEYS:
        add("manual.pack.schema", "/", "review pack has the wrong fields")
        return
    if pack["schema_version"] != REVIEW_PACK_SCHEMA:
        add(
            "manual.pack.schema",
            "/schema_version",
            f"expected {REVIEW_PACK_SCHEMA}",
        )
    if not _is_content_id(pack["pack_id"]):
        add("manual.pack.schema", "/pack_id", "pack_id must be a SHA-256 ID")
    try:
        parsed = datetime.fromisoformat(pack["generated_at"])
        if parsed.tzinfo is None:
            raise ValueError("timezone is missing")
    except (TypeError, ValueError):
        add(
            "manual.pack.schema",
            "/generated_at",
            "generated_at must be timezone-aware ISO-8601",
        )
    if type(pack["formal"]) is not bool:
        add("manual.pack.schema", "/formal", "formal must be boolean")
    if pack["status"] != "awaiting_human_review":
        add(
            "manual.pack.schema",
            "/status",
            "unsigned pack status must be awaiting_human_review",
        )
    manifests = pack["manifests"]
    if not isinstance(manifests, list) or len(manifests) != 2:
        add("manual.pack.schema", "/manifests", "pack requires two manifests")
        return
    for index, manifest in enumerate(manifests):
        location = f"/manifests/{index}"
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_REF_KEYS:
            add("manual.pack.schema", location, "manifest reference has wrong fields")
            continue
        if manifest["split"] not in {"train", "validation"}:
            add("manual.pack.schema", f"{location}/split", "split is invalid")
        if not _is_content_id(manifest["manifest_id"]):
            add(
                "manual.pack.schema",
                f"{location}/manifest_id",
                "manifest ID is invalid",
            )
        if not _is_sha256(manifest["file_sha256"]):
            add(
                "manual.pack.schema",
                f"{location}/file_sha256",
                "manifest digest is invalid",
            )
        if manifest["manifest_id"] != f"sha256:{manifest['file_sha256']}":
            add(
                "manual.pack.identity",
                f"{location}/manifest_id",
                "manifest identity and digest differ",
            )
        if type(manifest["formal"]) is not bool:
            add("manual.pack.schema", f"{location}/formal", "formal must be bool")
        if type(manifest["attempt_count"]) is not int or manifest["attempt_count"] < 0:
            add(
                "manual.pack.schema",
                f"{location}/attempt_count",
                "attempt count must be non-negative",
            )
    selection = pack["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "algorithm",
        "seed",
        "count",
    }:
        add("manual.pack.schema", "/selection", "selection has wrong fields")
        return
    if selection["algorithm"] != REVIEW_SELECTION_ALGORITHM:
        add(
            "manual.pack.selection",
            "/selection/algorithm",
            "selection algorithm is not frozen",
        )
    if type(selection["seed"]) is not int or selection["seed"] < 0:
        add("manual.pack.schema", "/selection/seed", "seed must be non-negative")
    if type(selection["count"]) is not int or not 1 <= selection["count"] <= 20:
        add("manual.pack.schema", "/selection/count", "count must be in 1..20")
    if type(pack["candidate_count"]) is not int or pack["candidate_count"] <= 0:
        add(
            "manual.pack.schema",
            "/candidate_count",
            "candidate count must be positive",
        )
    reviews = pack["selected_reviews"]
    if not isinstance(reviews, list) or len(reviews) != selection["count"]:
        add(
            "manual.pack.schema",
            "/selected_reviews",
            "selected review count differs",
        )
        return
    for index, review in enumerate(reviews):
        location = f"/selected_reviews/{index}"
        if not isinstance(review, dict) or set(review) != _SELECTED_REVIEW_KEYS:
            add("manual.pack.schema", location, "selected review has wrong fields")
            continue
        if not _is_content_id(review["manual_review_id"]):
            add(
                "manual.pack.schema",
                f"{location}/manual_review_id",
                "review ID is invalid",
            )
        if not _is_content_id(review["manifest_id"]):
            add(
                "manual.pack.schema",
                f"{location}/manifest_id",
                "manifest ID is invalid",
            )
        if review["split"] not in {"train", "validation"}:
            add("manual.pack.schema", f"{location}/split", "split is invalid")
        for field in ("attempt_index", "seed"):
            if type(review[field]) is not int or review[field] < 0:
                add(
                    "manual.pack.schema",
                    f"{location}/{field}",
                    f"{field} must be non-negative",
                )
        if not isinstance(review["source_relative_path"], str):
            add(
                "manual.pack.schema",
                f"{location}/source_relative_path",
                "source path must be text",
            )
        else:
            try:
                validate_relative_path(review["source_relative_path"])
            except ValueError as error:
                add(
                    "manual.pack.schema",
                    f"{location}/source_relative_path",
                    str(error),
                )
        if not _is_sha256(review["source_file_sha256"]):
            add(
                "manual.pack.schema",
                f"{location}/source_file_sha256",
                "source digest is invalid",
            )
        if (
            type(review["source_num_steps"]) is not int
            or review["source_num_steps"] <= 0
        ):
            add(
                "manual.pack.schema",
                f"{location}/source_num_steps",
                "source_num_steps must be positive",
            )
        if review["classification"] not in {"failure", "anomaly"}:
            add(
                "manual.pack.schema",
                f"{location}/classification",
                "classification is invalid",
            )
        if not isinstance(review["reasons"], list) or not review["reasons"]:
            add(
                "manual.pack.schema",
                f"{location}/reasons",
                "reasons must be non-empty",
            )
        if not isinstance(review["outlier_score"], float) or not math.isfinite(
            review["outlier_score"]
        ):
            add(
                "manual.pack.schema",
                f"{location}/outlier_score",
                "outlier score must be finite float",
            )
        if not isinstance(review["frame_indices"], list) or any(
            type(frame) is not int or frame < 0 for frame in review["frame_indices"]
        ):
            add(
                "manual.pack.schema",
                f"{location}/frame_indices",
                "frame indices are invalid",
            )
        media = review["media"]
        if (
            not isinstance(media, dict)
            or set(media) != {"path", "sha256"}
            or not isinstance(media["path"], str)
            or not _is_sha256(media["sha256"])
        ):
            add("manual.pack.schema", f"{location}/media", "media is invalid")
    template = pack["template"]
    if not isinstance(template, dict) or set(template) != {
        "path",
        "sha256",
        "row_count",
    }:
        add("manual.pack.schema", "/template", "template has wrong fields")
    elif not (
        template["path"] == REVIEW_TEMPLATE_FILENAME
        and _is_sha256(template["sha256"])
        and type(template["row_count"]) is int
        and template["row_count"] == selection["count"]
    ):
        add("manual.pack.schema", "/template", "template fields are invalid")
    cli = pack["cli_config"]
    if not isinstance(cli, dict) or set(cli) != {"count", "selection_seed", "smoke"}:
        add("manual.pack.schema", "/cli_config", "CLI config has wrong fields")
    elif not (
        cli["count"] == selection["count"]
        and cli["selection_seed"] == selection["seed"]
        and type(cli["smoke"]) is bool
        and pack["formal"] is not cli["smoke"]
    ):
        add("manual.pack.identity", "/cli_config", "CLI config is inconsistent")
    if pack["formal"] is True:
        if selection["count"] != FORMAL_REVIEW_COUNT:
            add(
                "manual.pack.formal",
                "/selection/count",
                f"formal review count must be {FORMAL_REVIEW_COUNT}",
            )
        if selection["seed"] != FORMAL_REVIEW_SEED:
            add(
                "manual.pack.formal",
                "/selection/seed",
                f"formal review seed must be {FORMAL_REVIEW_SEED}",
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=FORMAL_REVIEW_COUNT)
    parser.add_argument("--selection-seed", type=int, default=FORMAL_REVIEW_SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        pack = create_manual_review_pack(
            ManualReviewPackConfig(
                train_manifest_path=args.train_manifest,
                validation_manifest_path=args.validation_manifest,
                output_dir=args.output_dir,
                count=args.count,
                selection_seed=args.selection_seed,
                smoke=args.smoke,
            )
        )
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
