"""Raw M2 collection statistics and pending human-review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
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
    manifest_sha256,
    maximum_episode_steps,
    open_verified_episode,
    validate_collection_manifest,
)
from .replay import FORMAL_SELECTION_SEED, validate_replay_summary

REPORT_SCHEMA = "m2-data-report.v1"
REPORT_FILENAME = "report.json"
MANUAL_REVIEW_FILENAME = "manual_review.jsonl"
CONTACT_SHEETS_DIR = "contact_sheets"
FORMAL_MANUAL_REVIEW_COUNT = 20
FORMAL_MANUAL_SELECTION_SEED = FORMAL_SELECTION_SEED
MAX_REPORT_ACTION_BYTES = 2 * 1024**3
REPORT_FREE_SPACE_MARGIN_BYTES = 64 * 1024**2
_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


@dataclass(frozen=True)
class ReportConfig:
    manifest_path: Path
    output_dir: Path
    replay_summary_path: Path | None = None
    manual_review_count: int = FORMAL_MANUAL_REVIEW_COUNT
    manual_selection_seed: int = FORMAL_MANUAL_SELECTION_SEED
    smoke: bool = False


def generate_collection_report(
    config: ReportConfig,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    _validate_report_config(config)
    manifest_path = Path(config.manifest_path).expanduser().absolute()
    output_dir = Path(config.output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"report output directory already exists: {output_dir}")

    validation = validate_collection_manifest(manifest_path)
    validation.raise_for_errors()
    manifest = validation.manifest
    if manifest is None:
        raise RuntimeError("validated collection manifest is unavailable")
    if config.manual_review_count > len(manifest["attempts"]):
        raise ValueError("manual review count exceeds attempted episodes")
    source_digest = validation.sha256
    if source_digest is None:
        raise RuntimeError("validated collection digest is unavailable")
    run_root = manifest_path.parent

    (
        episode_rows,
        length_stats,
        action_stats,
        global_action_mean,
        global_action_std,
    ) = _aggregate_actions(
        manifest,
        run_root,
        temporary_parent=(
            output_dir.parent if output_dir.parent.is_dir() else run_root
        ),
    )
    schema_errors: list[dict[str, str]] = []
    lengths = np.asarray([row["num_steps"] for row in episode_rows], dtype=np.float64)
    candidates = _select_manual_candidates(
        episode_rows,
        count=config.manual_review_count,
        selection_seed=config.manual_selection_seed,
        global_action_mean=global_action_mean,
        global_action_std=global_action_std,
        length_mean=float(np.mean(lengths)),
        length_std=float(np.std(lengths)),
    )

    prepared_reviews: list[tuple[dict[str, Any], np.ndarray]] = []
    for review_index, candidate in enumerate(candidates):
        attempt = candidate["attempt"]
        with open_verified_episode(
            run_root,
            attempt["path"],
            expected_sha256=attempt["sha256"],
            max_num_steps=maximum_episode_steps(manifest["controller"]),
        ) as episode:
            episode.validation.raise_for_errors()
            if episode.metadata is None:
                raise RuntimeError("episode metadata disappeared during review")
            frame_indices = _review_frame_indices(episode.metadata.num_steps)
            front_dataset = episode.handle["observation.images.front"]
            wrist_dataset = episode.handle["observation.images.wrist"]
            front = np.stack([front_dataset[index] for index in frame_indices])
            wrist = np.stack([wrist_dataset[index] for index in frame_indices])
            sheet = _contact_sheet(front, wrist)
        contact_sheet_path = f"{CONTACT_SHEETS_DIR}/seed_{attempt['seed']:06d}.png"
        review = {
            "review_index": review_index,
            "seed": attempt["seed"],
            "path": attempt["path"],
            "outcome": attempt["status"],
            "failure_stage": attempt["failure_stage"],
            "reasons": candidate["reasons"],
            "outlier_score": candidate["outlier_score"],
            "frame_indices": frame_indices,
            "contact_sheet": contact_sheet_path,
            "verdict": "PENDING",
            "notes": "",
        }
        prepared_reviews.append((review, sheet))

    final_validation = validate_collection_manifest(manifest_path)
    final_validation.raise_for_errors()
    if final_validation.sha256 != source_digest:
        raise RuntimeError("collection manifest changed during report generation")

    output_dir.mkdir(parents=True, exist_ok=False)
    contact_dir = output_dir / CONTACT_SHEETS_DIR
    contact_dir.mkdir()
    manual_path = output_dir / MANUAL_REVIEW_FILENAME
    manual_state = initialize_jsonl_no_clobber(manual_path)
    manual_candidates: list[dict[str, Any]] = []
    for review, sheet in prepared_reviews:
        image_path = output_dir / review["contact_sheet"]
        if image_path.exists() or image_path.is_symlink():
            raise FileExistsError(f"contact sheet already exists: {image_path}")
        imageio.imwrite(image_path, sheet, format="png")
        manual_state = append_jsonl_fsync(
            manual_path,
            review,
            expected_snapshot=manual_state,
        )
        manual_candidates.append(
            {
                key: value
                for key, value in review.items()
                if key not in {"verdict", "notes"}
            }
        )

    timestamp = (now_fn or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    success_count = manifest["success_count"]
    attempt_count = manifest["attempt_count"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": timestamp.astimezone(UTC).isoformat(),
        "formal": bool(
            manifest["formal"] is True
            and config.smoke is False
            and config.manual_review_count == FORMAL_MANUAL_REVIEW_COUNT
            and config.manual_selection_seed == FORMAL_MANUAL_SELECTION_SEED
        ),
        "source_manifest_sha256": source_digest,
        "source_split": manifest["split"],
        "source_manifest_formal": manifest["formal"],
        "counts": {
            "attempts": attempt_count,
            "successes": success_count,
            "failures": attempt_count - success_count,
        },
        "episode_lengths": length_stats,
        "actions": action_stats,
        "schema_errors": schema_errors,
        "replay_linkage": _replay_linkage(
            config.replay_summary_path,
            source_manifest_path=manifest_path,
            source_digest=source_digest,
        ),
        "manual_review_complete": False,
        "manual_review_count": len(manual_candidates),
        "manual_review_candidates": manual_candidates,
        "manual_review_path": MANUAL_REVIEW_FILENAME,
        "manual_review_selection": {
            "algorithm": "failure-first-length-action-outlier.v1",
            "seed": config.manual_selection_seed,
        },
        "cli_config": {
            "manual_review_count": config.manual_review_count,
            "manual_selection_seed": config.manual_selection_seed,
            "smoke": config.smoke,
        },
    }
    atomic_write_json_no_clobber(output_dir / REPORT_FILENAME, report)
    return report


def _aggregate_actions(
    manifest: dict[str, Any],
    run_root: Path,
    *,
    temporary_parent: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    total_frames = sum(attempt["num_steps"] for attempt in manifest["attempts"])
    if total_frames <= 0:
        raise ValueError("report requires at least one valid episode")
    required_bytes = total_frames * 8 * np.dtype(np.float64).itemsize
    if required_bytes > MAX_REPORT_ACTION_BYTES:
        raise ValueError(
            "report action spool exceeds the frozen byte budget "
            f"({required_bytes} > {MAX_REPORT_ACTION_BYTES})"
        )
    temporary_parent = Path(temporary_parent)
    if temporary_parent.is_symlink() or not temporary_parent.is_dir():
        raise ValueError("report temporary parent must be a real existing directory")
    required_free = required_bytes + REPORT_FREE_SPACE_MARGIN_BYTES
    available = shutil.disk_usage(temporary_parent).free
    if available < required_free:
        raise OSError(
            "insufficient free space for bounded report action spool: "
            f"need {required_free} bytes, have {available}"
        )
    maximum_steps = maximum_episode_steps(manifest["controller"])
    episode_rows: list[dict[str, Any]] = []
    action_sum = np.zeros(8, dtype=np.float64)
    action_sum_squares = np.zeros(8, dtype=np.float64)
    offset = 0
    with tempfile.TemporaryDirectory(
        prefix=".m2-action-stats-",
        dir=temporary_parent,
    ) as temporary:
        action_path = Path(temporary) / "actions.f64"
        actions = np.memmap(
            action_path,
            mode="w+",
            dtype=np.float64,
            shape=(total_frames, 8),
        )
        for attempt in manifest["attempts"]:
            with open_verified_episode(
                run_root,
                attempt["path"],
                expected_sha256=attempt["sha256"],
                max_num_steps=maximum_steps,
            ) as episode:
                episode.validation.raise_for_errors()
                if episode.metadata is None:
                    raise RuntimeError(
                        "episode metadata disappeared during aggregation"
                    )
                dataset = episode.handle["action"]
                num_steps = int(dataset.shape[0])
                if num_steps != attempt["num_steps"]:
                    raise RuntimeError("episode length changed during aggregation")
                episode_sum = np.zeros(8, dtype=np.float64)
                for start in range(0, num_steps, 256):
                    stop = min(start + 256, num_steps)
                    block = np.asarray(dataset[start:stop], dtype=np.float64)
                    actions[offset : offset + block.shape[0]] = block
                    offset += block.shape[0]
                    block_sum = np.sum(block, axis=0, dtype=np.float64)
                    episode_sum += block_sum
                    action_sum += block_sum
                    action_sum_squares += np.sum(
                        np.square(block),
                        axis=0,
                        dtype=np.float64,
                    )
                episode_rows.append(
                    {
                        "attempt": attempt,
                        "num_steps": num_steps,
                        "action_mean": episode_sum / num_steps,
                    }
                )
        if offset != total_frames:
            raise RuntimeError("manifest frame count changed during aggregation")
        actions.flush()
        action_stats = _action_distribution(actions)
        del actions
    lengths = np.asarray(
        [row["num_steps"] for row in episode_rows],
        dtype=np.float64,
    )
    mean = action_sum / total_frames
    variance = np.maximum(action_sum_squares / total_frames - np.square(mean), 0.0)
    return (
        episode_rows,
        _distribution(lengths),
        action_stats,
        mean,
        np.sqrt(variance),
    )


def _distribution(values: np.ndarray) -> dict[str, Any]:
    percentiles = np.percentile(values, _PERCENTILES)
    return {
        "count": int(values.size),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {
            f"p{percentile:02d}": float(value)
            for percentile, value in zip(_PERCENTILES, percentiles, strict=True)
        },
    }


def _action_distribution(actions: np.ndarray) -> dict[str, Any]:
    dimensions: list[dict[str, Any]] = []
    for dimension in range(actions.shape[1]):
        column = actions[:, dimension]
        percentiles = np.percentile(column, _PERCENTILES)
        dimensions.append(
            {
                "index": dimension,
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "percentiles": {
                    f"p{percentile:02d}": float(percentiles[index])
                    for index, percentile in enumerate(_PERCENTILES)
                },
            }
        )
    return {"frame_count": int(actions.shape[0]), "dimensions": dimensions}


def _select_manual_candidates(
    rows: list[dict[str, Any]],
    *,
    count: int,
    selection_seed: int,
    global_action_mean: np.ndarray,
    global_action_std: np.ndarray,
    length_mean: float,
    length_std: float,
) -> list[dict[str, Any]]:
    safe_action_std = np.where(global_action_std > 1e-12, global_action_std, 1.0)
    safe_length_std = length_std if length_std > 1e-12 else 1.0
    ranked: list[dict[str, Any]] = []
    for row in rows:
        attempt = row["attempt"]
        length_score = abs(row["num_steps"] - length_mean) / safe_length_std
        action_score = float(
            np.max(np.abs(row["action_mean"] - global_action_mean) / safe_action_std)
        )
        score = float(length_score + action_score)
        tie = hashlib.sha256(
            f"{selection_seed}:{attempt['seed']}".encode("ascii")
        ).hexdigest()
        reasons = (
            [f"expert_failure:{attempt['failure_stage']}"]
            if not attempt["success"]
            else ["deterministic_length_action_outlier"]
        )
        ranked.append(
            {
                "attempt": attempt,
                "outlier_score": score,
                "reasons": reasons,
                "tie": tie,
            }
        )
    ranked.sort(
        key=lambda item: (
            0 if not item["attempt"]["success"] else 1,
            -item["outlier_score"],
            item["tie"],
        )
    )
    return ranked[:count]


def _review_frame_indices(num_steps: int) -> list[int]:
    if num_steps <= 0:
        raise ValueError("manual review requires a non-empty episode")
    return sorted({0, num_steps // 2, num_steps - 1})


def _contact_sheet(
    front: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray:
    if front.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError("contact sheets require uint8 camera frames")
    gutter = np.full((front.shape[1], 4, 3), 255, dtype=np.uint8)

    def row(frames: np.ndarray) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            if index:
                pieces.append(gutter)
            pieces.append(frame)
        return np.concatenate(pieces, axis=1)

    top = row(front)
    bottom = row(wrist)
    horizontal = np.full((4, top.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([top, horizontal, bottom], axis=0)


def _replay_linkage(
    path: Path | None,
    *,
    source_manifest_path: Path,
    source_digest: str,
) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "linked": False}
    replay_path = Path(path).expanduser().absolute()
    try:
        validation = validate_replay_summary(
            replay_path,
            collection_manifest_path=source_manifest_path,
        )
        summary_digest = validation.sha256 or manifest_sha256(replay_path)
    except (OSError, UnicodeError, ValueError) as error:
        return {
            "provided": True,
            "linked": False,
            "error": f"cannot read replay summary: {error}",
        }
    summary = validation.summary or {}
    linked = bool(
        validation.valid and summary.get("source_manifest_sha256") == source_digest
    )
    error = None if validation.valid else validation.format_errors()
    return {
        "provided": True,
        "linked": linked,
        "summary_sha256": summary_digest,
        "source_manifest_sha256": summary.get("source_manifest_sha256"),
        "trial_count": summary.get("trial_count"),
        "success_count": summary.get("success_count"),
        "gate": summary.get("gate"),
        "validation_error": error,
    }


def _validate_report_config(config: ReportConfig) -> None:
    if type(config.smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if type(config.manual_review_count) is not int or config.manual_review_count <= 0:
        raise ValueError("manual_review_count must be a positive integer")
    if config.manual_review_count > FORMAL_MANUAL_REVIEW_COUNT:
        raise ValueError("manual_review_count cannot exceed 20")
    if config.manual_review_count != FORMAL_MANUAL_REVIEW_COUNT and not config.smoke:
        raise ValueError("a manual review count smaller than 20 requires --smoke")
    if (
        type(config.manual_selection_seed) is not int
        or config.manual_selection_seed < 0
    ):
        raise ValueError("manual_selection_seed must be a non-negative integer")
    if (
        config.manual_selection_seed != FORMAL_MANUAL_SELECTION_SEED
        and not config.smoke
    ):
        raise ValueError("a noncanonical manual selection seed requires --smoke")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw M2 statistics and pending manual-review sheets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--manual-review-count", type=int, default=20)
    parser.add_argument(
        "--manual-selection-seed",
        type=int,
        default=FORMAL_MANUAL_SELECTION_SEED,
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        report = generate_collection_report(
            ReportConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                replay_summary_path=args.replay_summary,
                manual_review_count=args.manual_review_count,
                manual_selection_seed=args.manual_selection_seed,
                smoke=args.smoke,
            )
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
