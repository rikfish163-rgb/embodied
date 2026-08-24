"""Batch evaluation and video export for the scripted M1 expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from env import scene as env_scene
from env.asset_provenance import AssetProvenanceError, collect_asset_provenance
from env.pick_place import PickPlace

from .scripted import EpisodeResult, ExpertConfig, config_dict, run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RUNTIME_INPUT_ROOTS = frozenset({"config", "configs", "menagerie", "scripts", "src"})
_ROOT_INPUT_FILES = frozenset(
    {
        ".python-version",
        "env.sh",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "uv.lock",
    }
)
_IGNORED_INPUT_PATHS = (
    "src",
    "scripts",
    "menagerie",
    "config",
    "configs",
    ".python-version",
    "env.sh",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "uv.lock",
    "requirements*.txt",
    ":(top,glob)*.cfg",
    ":(top,glob)*.ini",
    ":(top,glob)*.py",
    ":(top,glob)*.toml",
    ":(top,glob)*.yaml",
    ":(top,glob)*.yml",
)


class VideoDependencyError(RuntimeError):
    """MP4 recording support is unavailable before a run starts."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the privileged Panda DLS expert on fixed seeds."
    )
    parser.add_argument("--seed-start", type=_non_negative_int, default=0)
    parser.add_argument("--num-seeds", type=_positive_int, default=100)
    parser.add_argument("--required-successes", type=_non_negative_int, default=90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new output directory; existing directories are never overwritten",
    )
    parser.add_argument(
        "--record",
        choices=("none", "failures", "all"),
        default="none",
        help="write side-by-side front/wrist MP4 rollouts",
    )
    parser.add_argument(
        "--video-stride",
        type=_positive_int,
        default=2,
        help="record every Nth 20 Hz control step (default: 2 -> 10 FPS)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.required_successes > args.num_seeds:
        parser.error("--required-successes cannot exceed --num-seeds")

    output_dir = args.output_dir or _default_output_dir()
    try:
        summary = evaluate(
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            required_successes=args.required_successes,
            output_dir=output_dir,
            record=args.record,
            video_stride=args.video_stride,
        )
    except FileExistsError:
        parser.error(f"output directory already exists: {output_dir}")
    except AssetProvenanceError as error:
        parser.error(f"asset provenance check failed: {error}")
    except VideoDependencyError as error:
        parser.error(str(error))

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["gate"]["passed"] else 1


def evaluate(
    *,
    seed_start: int,
    num_seeds: int,
    required_successes: int,
    output_dir: Path,
    record: str = "none",
    video_stride: int = 2,
    config: ExpertConfig | None = None,
) -> dict[str, Any]:
    _validate_evaluation_request(
        seed_start=seed_start,
        num_seeds=num_seeds,
        required_successes=required_successes,
        record=record,
        video_stride=video_stride,
    )
    asset_provenance = collect_asset_provenance(
        PROJECT_ROOT,
        runtime_asset_root=env_scene.MENAGERIE,
    )
    _require_video_support(record)
    cfg = config or ExpertConfig()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    videos_dir = output_dir / "videos"
    if record != "none":
        videos_dir.mkdir()

    env = PickPlace()
    results: list[EpisodeResult] = []
    video_paths: dict[int, Path] = {}
    try:
        for index, seed in enumerate(range(seed_start, seed_start + num_seeds), 1):
            if record == "all":
                video_path = videos_dir / f"seed_{seed:04d}.mp4"
                result = _run_with_video(
                    env,
                    seed=seed,
                    config=cfg,
                    path=video_path,
                    stride=video_stride,
                )
                video_paths[seed] = video_path
            else:
                result = run_episode(env, seed=seed, config=cfg)
            results.append(result)
            print(
                f"seed={seed:04d} success={result.success} "
                f"stage={result.failure_stage or '-':10s} "
                f"attempts={result.attempts} sim={result.sim_time_s:.2f}s",
                flush=True,
            )
            if index % 10 == 0 or index == num_seeds:
                successes = sum(item.success for item in results)
                print(
                    f"PROGRESS {index}/{num_seeds} success={successes}/{index}",
                    flush=True,
                )

        if record == "failures":
            for result in results:
                if result.success:
                    continue
                video_path = videos_dir / f"seed_{result.seed:04d}.mp4"
                replay = _run_with_video(
                    env,
                    seed=result.seed,
                    config=cfg,
                    path=video_path,
                    stride=video_stride,
                )
                if replay.success != result.success:
                    raise RuntimeError(
                        f"video replay changed deterministic result for seed {result.seed}"
                    )
                video_paths[result.seed] = video_path
    finally:
        env.close()

    episode_payloads = []
    for result in results:
        payload = result.to_dict()
        if result.seed in video_paths:
            payload["video"] = str(video_paths[result.seed].relative_to(output_dir))
        episode_payloads.append(payload)

    episodes_path = output_dir / "episodes.jsonl"
    with episodes_path.open("x", encoding="utf-8") as handle:
        for payload in episode_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    success_count = sum(result.success for result in results)
    failure_counts = Counter(
        result.failure_stage for result in results if not result.success
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "mujoco_version": mujoco.__version__,
        "git": _git_state(),
        "asset_provenance": asset_provenance.summary_dict(),
        "seed_start": seed_start,
        "num_episodes": num_seeds,
        "successes": success_count,
        "success_rate": success_count / num_seeds,
        "recovered_successes": sum(result.recovered for result in results),
        "failure_counts": dict(sorted(failure_counts.items())),
        "mean_sim_time_s": float(np.mean([result.sim_time_s for result in results])),
        "controller": config_dict(cfg),
        "record": record,
        "video_stride": video_stride,
        "gate": {
            "required_successes": required_successes,
            "passed": success_count >= required_successes,
        },
        "episodes_file": episodes_path.name,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _validate_evaluation_request(
    *,
    seed_start: int,
    num_seeds: int,
    required_successes: int,
    record: str,
    video_stride: int,
) -> None:
    for name, value in (
        ("seed_start", seed_start),
        ("num_seeds", num_seeds),
        ("required_successes", required_successes),
        ("video_stride", video_stride),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if required_successes < 0:
        raise ValueError("required_successes must be non-negative")
    if required_successes > num_seeds:
        raise ValueError("required_successes cannot exceed num_seeds")
    if record not in {"none", "failures", "all"}:
        raise ValueError("record must be one of: none, failures, all")
    if video_stride <= 0:
        raise ValueError("video_stride must be positive")


def _run_with_video(
    env: PickPlace,
    *,
    seed: int,
    config: ExpertConfig,
    path: Path,
    stride: int,
) -> EpisodeResult:
    recorder = _VideoRecorder(
        path,
        fps=env.cfg.control_hz / stride,
        stride=stride,
    )
    try:
        return run_episode(
            env,
            seed=seed,
            config=config,
            step_callback=recorder.capture,
        )
    finally:
        recorder.close()


class _VideoRecorder:
    def __init__(self, path: Path, *, fps: float, stride: int):
        try:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(
                path,
                fps=fps,
                codec="libx264",
                quality=7,
                macro_block_size=2,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            raise VideoDependencyError(_VIDEO_DEPENDENCY_MESSAGE) from error
        self._stride = stride
        self._step_index = 0

    def capture(self, env: PickPlace, stage: str, action: np.ndarray) -> None:
        del stage, action
        if self._step_index % self._stride == 0:
            front = env.render("front")
            wrist = env.render("wrist")
            self._writer.append_data(np.concatenate([front, wrist], axis=1))
        self._step_index += 1

    def close(self) -> None:
        self._writer.close()


_VIDEO_DEPENDENCY_MESSAGE = (
    "MP4 recording requires the video dependency group: uv sync --locked --group video"
)


def _require_video_support(record: str) -> None:
    """Resolve the packaged FFmpeg binary before creating run artifacts."""

    if record == "none":
        return
    try:
        import imageio.v2  # noqa: F401
        import imageio_ffmpeg

        version = imageio_ffmpeg.get_ffmpeg_version()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise VideoDependencyError(_VIDEO_DEPENDENCY_MESSAGE) from error
    if not isinstance(version, str) or not version.strip():
        raise VideoDependencyError(_VIDEO_DEPENDENCY_MESSAGE)


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / "m1" / timestamp


def _git_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    worktree_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    ignored_inputs = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *_IGNORED_INPUT_PATHS,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )

    untracked_paths = _git_paths(untracked.stdout) if untracked.returncode == 0 else []
    relevant_paths = {
        (path, "untracked") for path in untracked_paths if _is_runtime_input(path)
    }
    if ignored_inputs.returncode == 0:
        relevant_paths.update(
            (path, "ignored")
            for path in _git_paths(ignored_inputs.stdout)
            if _is_runtime_input(path)
        )

    relevant_files: list[dict[str, Any]] = []
    fingerprints_complete = True
    for path, git_status in sorted(relevant_paths):
        fingerprint, complete = _fingerprint_file(
            project_root,
            path=path,
            git_status=git_status,
        )
        relevant_files.append(fingerprint)
        fingerprints_complete &= complete

    commands_complete = all(
        result.returncode == 0
        for result in (
            commit,
            tracked_status,
            worktree_status,
            untracked,
            ignored_inputs,
        )
    )
    provenance_complete = commands_complete and fingerprints_complete
    tracked_worktree_clean = (
        tracked_status.returncode == 0 and not tracked_status.stdout.strip()
    )
    return {
        "commit": commit.stdout.decode("ascii").strip()
        if commit.returncode == 0
        else None,
        # Historical M1 summaries define this as tracked files only. Keep that
        # meaning stable; use the new fields below for fail-closed provenance.
        "tracked_worktree_clean": tracked_worktree_clean,
        "worktree_clean": worktree_status.returncode == 0
        and not worktree_status.stdout,
        "source_provenance_clean": provenance_complete
        and tracked_worktree_clean
        and not relevant_files,
        "provenance_complete": provenance_complete,
        "worktree_status_sha256": _output_sha256(worktree_status),
        "untracked_file_count": len(untracked_paths)
        if untracked.returncode == 0
        else None,
        "untracked_paths_sha256": _output_sha256(untracked),
        "relevant_untracked_files": relevant_files,
    }


def _git_paths(output: bytes) -> list[str]:
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def _is_runtime_input(path: str) -> bool:
    parts = Path(path).parts
    if not parts:
        return False
    if any(
        part == "__pycache__" or part.endswith((".dist-info", ".egg-info"))
        for part in parts
    ):
        return False
    if parts[0] in _RUNTIME_INPUT_ROOTS:
        return True
    if len(parts) != 1:
        return False
    name = parts[0]
    return (
        name in _ROOT_INPUT_FILES
        or name.startswith("requirements")
        and name.endswith(".txt")
        or Path(name).suffix in {".cfg", ".ini", ".py", ".toml", ".yaml", ".yml"}
    )


def _fingerprint_file(
    project_root: Path,
    *,
    path: str,
    git_status: str,
) -> tuple[dict[str, Any], bool]:
    absolute_path = project_root / path
    try:
        before = absolute_path.lstat()
        digest = hashlib.sha256()
        if stat.S_ISREG(before.st_mode):
            with absolute_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif stat.S_ISLNK(before.st_mode):
            digest.update(os.fsencode(os.readlink(absolute_path)))
        else:
            return _incomplete_fingerprint(path, git_status, before.st_size), False
        after = absolute_path.lstat()
    except OSError:
        return _incomplete_fingerprint(path, git_status, None), False

    stable = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    return (
        {
            "git_status": git_status,
            "path": path,
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        },
        stable,
    )


def _incomplete_fingerprint(
    path: str,
    git_status: str,
    size_bytes: int | None,
) -> dict[str, Any]:
    return {
        "git_status": git_status,
        "path": path,
        "sha256": None,
        "size_bytes": size_bytes,
    }


def _output_sha256(result: subprocess.CompletedProcess[bytes]) -> str | None:
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
