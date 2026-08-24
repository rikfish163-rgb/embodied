"""Streaming HDF5 episodes with pre-action observation alignment.

One file contains one complete episode.  A writer appends one transition at a
time to resizable datasets, validates the completed temporary file, and only
then publishes it at the requested path without replacing an existing file.
"""

from __future__ import annotations

import errno
import hashlib
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

SCHEMA_VERSION = 1
IMAGE_SHAPE = (128, 128, 3)
STATE_DIM = 8
ACTION_DIM = 8
DEFAULT_CONTROL_DT_S = 0.05
VALIDATION_CHUNK_ROWS = 256
STAGE_MAX_UTF8_BYTES = 64
STAGE_CHUNKS = (VALIDATION_CHUNK_ROWS,)
MAX_SCHEMA_CHUNK_BYTES = 16 * 1024 * 1024
TIME_ALIGNMENT = "pre_action"
ACTION_SEMANTICS = "absolute_joint_position_targets_rad[7]+normalized_gripper_open[1]"
CANONICAL_ACTION_MIN = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
    0.0,
)
CANONICAL_ACTION_MAX = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
    1.0,
)

OBSERVATION_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "observation.state",
)
DATASET_KEYS = (*OBSERVATION_KEYS, "action", "timestamp", "stage")
REQUIRED_ATTRIBUTES = (
    "schema_version",
    "seed",
    "success",
    "failure_stage",
    "num_steps",
    "control_dt_s",
    "time_alignment",
    "action_semantics",
    "action_min",
    "action_max",
    "complete",
)


@dataclass(frozen=True)
class EpisodeMetadata:
    """Metadata needed to audit and split an episode."""

    schema_version: int
    seed: int
    success: bool
    failure_stage: str | None
    num_steps: int
    control_dt_s: float
    time_alignment: str
    action_semantics: str
    action_min: NDArray[np.float64]
    action_max: NDArray[np.float64]


@dataclass(frozen=True)
class Transition:
    """One observation and the action chosen from that observation."""

    observation: dict[str, NDArray[Any]]
    action: NDArray[np.float32]
    timestamp: float
    stage: str


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable validation error with a precise HDF5 location."""

    code: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one episode file."""

    path: Path
    num_steps: int | None
    errors: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(str(issue) for issue in self.errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DatasetValidationError(self)


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


class DatasetValidationError(ValueError):
    """Raised when a completed episode violates the frozen schema."""

    def __init__(self, report: ValidationReport):
        self.report = report
        details = report.format_errors() or "unknown validation error"
        super().__init__(f"invalid HDF5 episode {report.path}:\n{details}")


class EpisodePublicationError(RuntimeError):
    """Raised when publication completed but durability is indeterminate."""

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
            "episode target was linked but publication durability is "
            f"indeterminate; reconcile {target_path} before retrying"
        )


class HDF5EpisodeWriter:
    """Append transitions to a temporary HDF5 file and publish atomically.

    The instance itself can be passed to ``expert.scripted.run_episode`` as its
    ``step_callback``; ``capture`` is also provided explicitly.  The callback
    calls only ``env.observe()``, reads ``env.data.time`` and the environment's
    action/control contract, and therefore never reads or writes object qpos.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        seed: int,
        action_bounds: tuple[ArrayLike, ArrayLike] | None = None,
        control_dt_s: float = DEFAULT_CONTROL_DT_S,
        flush_every: int = 32,
    ):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not _is_real_scalar(control_dt_s):
            raise TypeError("control_dt_s must be a real number")
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be finite and positive")
        if not np.isclose(
            control_dt_s,
            DEFAULT_CONTROL_DT_S,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("control_dt_s must be 0.05 seconds (20 Hz)")
        if (
            not isinstance(flush_every, int)
            or isinstance(flush_every, bool)
            or flush_every <= 0
        ):
            raise ValueError("flush_every must be a positive integer")

        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"episode already exists: {target}")

        self.path = target
        self.seed = seed
        self.control_dt_s = float(control_dt_s)
        self.flush_every = flush_every
        self._partial_path = target.parent / (
            f".{target.name}.partial-{uuid.uuid4().hex}"
        )
        self._state = "open"
        self._poisoned = False
        self._num_steps = 0
        self._last_timestamp: float | None = None
        self._action_min: NDArray[np.float64] | None = None
        self._action_max: NDArray[np.float64] | None = None
        self._environment_contract_checked = False

        try:
            self._file = h5py.File(self._partial_path, "x")
            self._initialize_file()
            if action_bounds is not None:
                self._set_action_bounds(action_bounds)
        except Exception:
            file_handle = getattr(self, "_file", None)
            if file_handle is not None and file_handle.id.valid:
                try:
                    file_handle.close()
                except Exception:
                    pass
            self._cleanup_partial()
            raise

    def _initialize_file(self) -> None:
        attrs = self._file.attrs
        attrs["schema_version"] = np.int64(SCHEMA_VERSION)
        attrs["seed"] = np.int64(self.seed)
        attrs["control_dt_s"] = np.float64(self.control_dt_s)
        attrs["time_alignment"] = TIME_ALIGNMENT
        attrs["action_semantics"] = ACTION_SEMANTICS
        attrs["complete"] = np.bool_(False)

        for key in ("observation.images.front", "observation.images.wrist"):
            self._file.create_dataset(
                key,
                shape=(0, *IMAGE_SHAPE),
                maxshape=(None, *IMAGE_SHAPE),
                chunks=(1, *IMAGE_SHAPE),
                dtype=np.uint8,
                compression="lzf",
                shuffle=True,
            )
        self._file.create_dataset(
            "observation.state",
            shape=(0, STATE_DIM),
            maxshape=(None, STATE_DIM),
            chunks=(256, STATE_DIM),
            dtype=np.float32,
        )
        self._file.create_dataset(
            "action",
            shape=(0, ACTION_DIM),
            maxshape=(None, ACTION_DIM),
            chunks=(256, ACTION_DIM),
            dtype=np.float32,
        )
        self._file.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=np.float64,
        )
        self._file.create_dataset(
            "stage",
            shape=(0,),
            maxshape=(None,),
            chunks=STAGE_CHUNKS,
            dtype=h5py.string_dtype(
                encoding="utf-8",
                length=STAGE_MAX_UTF8_BYTES,
            ),
        )

    def _set_action_bounds(self, action_bounds: tuple[ArrayLike, ArrayLike]) -> None:
        lower, upper = _normalize_action_bounds(action_bounds)
        if self._action_min is not None:
            if not (
                np.array_equal(lower, self._action_min)
                and np.array_equal(upper, self._action_max)
            ):
                raise ValueError("action bounds changed during an episode")
            return
        self._action_min = lower
        self._action_max = upper
        self._file.attrs["action_min"] = lower
        self._file.attrs["action_max"] = upper

    def capture(self, env: Any, stage: str, action: ArrayLike) -> None:
        """Capture the pre-action transition supplied by the M1 expert."""

        self._ensure_open()
        _validate_policy_visual_contract(env)
        if not self._environment_contract_checked:
            try:
                control_hz = float(env.cfg.control_hz)
                arm_ranges = np.asarray(
                    env.model.actuator_ctrlrange[:7], dtype=np.float64
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise TypeError(
                    "callback env must expose cfg.control_hz and 7 arm actuator ranges"
                ) from error
            if arm_ranges.shape != (7, 2):
                raise ValueError(
                    "callback env must expose seven [min, max] arm action ranges"
                )
            observed_dt = 1.0 / control_hz if control_hz > 0 else np.inf
            if not np.isclose(
                observed_dt,
                self.control_dt_s,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "writer control_dt_s does not match env.cfg.control_hz"
                )
            environment_bounds = (
                np.concatenate([arm_ranges[:, 0], [0.0]]),
                np.concatenate([arm_ranges[:, 1], [1.0]]),
            )
            self._set_action_bounds(environment_bounds)
            self._environment_contract_checked = True

        timestamp = float(env.data.time)
        observation = env.observe()
        self.append(
            observation,
            action,
            timestamp=timestamp,
            stage=stage,
        )

    __call__ = capture

    def append(
        self,
        observation: Mapping[str, ArrayLike],
        action: ArrayLike,
        *,
        timestamp: float,
        stage: str,
    ) -> None:
        """Validate and append exactly one pre-action transition."""

        self._ensure_open()
        if self._poisoned:
            raise RuntimeError("writer is unusable after a failed HDF5 append")
        if self._action_min is None or self._action_max is None:
            raise RuntimeError(
                "action bounds are not configured; pass them to the writer or use capture"
            )
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        received_keys = set(observation)
        expected_keys = set(OBSERVATION_KEYS)
        if received_keys != expected_keys:
            missing = sorted(expected_keys - received_keys)
            extra = sorted(received_keys - expected_keys)
            raise ValueError(
                f"observation keys must match the policy contract; missing={missing}, "
                f"extra={extra}"
            )

        front = _validate_image(
            observation["observation.images.front"],
            "observation.images.front",
        )
        wrist = _validate_image(
            observation["observation.images.wrist"],
            "observation.images.wrist",
        )
        state = _finite_vector(
            observation["observation.state"],
            STATE_DIM,
            "observation.state",
        ).astype(np.float32)
        if not 0.0 <= state[7] <= 1.0:
            raise ValueError("observation.state gripper must be in [0, 1]")
        action_array = _finite_vector(action, ACTION_DIM, "action")
        if np.any(action_array < self._action_min) or np.any(
            action_array > self._action_max
        ):
            raise ValueError("action must stay within the recorded action bounds")
        stored_action = action_array.astype(np.float32)

        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError("timestamp must be finite") from error
        if not np.isfinite(timestamp_value):
            raise ValueError("timestamp must be finite")
        if timestamp_value < 0:
            raise ValueError("timestamp must be non-negative")
        if self._last_timestamp is not None and timestamp_value <= self._last_timestamp:
            raise ValueError("timestamp must be strictly increasing")
        stage = _validate_stage_text(stage, "stage")

        index = self._num_steps
        try:
            for key in DATASET_KEYS:
                self._file[key].resize(index + 1, axis=0)
            self._file["observation.images.front"][index] = front
            self._file["observation.images.wrist"][index] = wrist
            self._file["observation.state"][index] = state
            self._file["action"][index] = stored_action
            self._file["timestamp"][index] = timestamp_value
            self._file["stage"][index] = stage
        except Exception:
            self._poisoned = True
            raise

        self._num_steps += 1
        self._last_timestamp = timestamp_value
        if self._num_steps % self.flush_every == 0:
            self._file.flush()

    def finalize(self, *, success: bool, failure_stage: str | None) -> Path:
        """Validate, fsync and publish the complete episode without clobbering."""

        self._ensure_open()
        if self._poisoned:
            raise RuntimeError("cannot finalize a writer after a failed append")
        if not isinstance(success, (bool, np.bool_)):
            raise TypeError("success must be a boolean")
        success_value = bool(success)
        if success_value:
            if failure_stage is not None:
                raise ValueError("successful episode must have failure_stage=None")
            stored_failure_stage = ""
        else:
            if not isinstance(failure_stage, str) or not failure_stage.strip():
                raise ValueError("failed episode must have a non-empty failure_stage")
            stored_failure_stage = _validate_stage_text(
                failure_stage,
                "failure_stage",
            )
        if self._num_steps == 0:
            raise ValueError("episode must contain at least one transition")
        if self._action_min is None or self._action_max is None:
            raise RuntimeError("cannot finalize without action bounds")

        try:
            attrs = self._file.attrs
            attrs["success"] = np.bool_(success_value)
            attrs["failure_stage"] = stored_failure_stage
            attrs["num_steps"] = np.int64(self._num_steps)
            attrs["complete"] = np.bool_(True)
            self._file.flush()
            self._file.close()
            self._state = "closed"
            _fsync_file(self._partial_path)

            validated_snapshot = _file_snapshot(self._partial_path)
            report = validate_episode(self._partial_path)
            report.raise_for_errors()
            if _file_snapshot(self._partial_path) != validated_snapshot:
                raise RuntimeError("episode partial changed after validation")
            validated_sha256, _ = _stable_sha256(
                self._partial_path,
                expected_snapshot=validated_snapshot,
            )
            _publish_no_clobber(
                self._partial_path,
                self.path,
                validated_snapshot=validated_snapshot,
                validated_sha256=validated_sha256,
            )
            self._state = "finalized"
            return self.path
        except EpisodePublicationError:
            self._state = "publication_indeterminate"
            raise
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close the writer and remove its unpublished temporary file."""

        if self._state in {"finalized", "publication_indeterminate", "aborted"}:
            return
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            try:
                if file_handle.id.valid:
                    file_handle.close()
            finally:
                self._cleanup_partial()
        else:
            self._cleanup_partial()
        self._state = "aborted"

    def _cleanup_partial(self) -> None:
        partial = getattr(self, "_partial_path", None)
        if partial is not None:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"writer is not open (state={self._state})")

    def __enter__(self) -> HDF5EpisodeWriter:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state not in {"finalized", "publication_indeterminate"}:
            self.abort()


class HDF5EpisodeReader:
    """Read one transition at a time without materializing an episode."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        try:
            self._file = h5py.File(self.path, "r")
        except (OSError, ValueError) as error:
            report = ValidationReport(
                self.path,
                None,
                (
                    ValidationIssue(
                        code="file.open",
                        location="/",
                        message=f"cannot open HDF5 file: {error}",
                    ),
                ),
            )
            raise DatasetValidationError(report) from error
        try:
            _validate_episode_handle(self._file, self.path).raise_for_errors()
            self.metadata = _metadata_from_handle(self._file)
        except Exception:
            self._file.close()
            raise

    def __len__(self) -> int:
        return self.metadata.num_steps

    def __getitem__(self, index: int) -> Transition:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("episode indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        stage = self._file["stage"].asstr()[index]
        return Transition(
            observation={
                "observation.images.front": self._file["observation.images.front"][
                    index
                ],
                "observation.images.wrist": self._file["observation.images.wrist"][
                    index
                ],
                "observation.state": self._file["observation.state"][index],
            },
            action=self._file["action"][index],
            timestamp=float(self._file["timestamp"][index]),
            stage=str(stage),
        )

    def __iter__(self) -> Iterator[Transition]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    def __enter__(self) -> HDF5EpisodeReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def validate_episode(path: str | os.PathLike[str]) -> ValidationReport:
    """Validate shapes, dtypes, alignment, values, bounds and metadata."""

    episode_path = Path(path).expanduser().resolve()
    try:
        handle = h5py.File(episode_path, "r")
    except (OSError, ValueError) as error:
        issue = ValidationIssue(
            code="file.open",
            location="/",
            message=f"cannot open HDF5 file: {error}",
        )
        return ValidationReport(episode_path, None, (issue,))

    with handle:
        return _validate_episode_handle(handle, episode_path)


def _validate_episode_handle(
    handle: h5py.File,
    episode_path: Path,
) -> ValidationReport:
    """Validate the exact open file handle without taking ownership of it."""

    issues: list[ValidationIssue] = []

    def add(code: str, location: str, message: str) -> None:
        issues.append(ValidationIssue(code=code, location=location, message=message))

    num_steps: int | None = None
    datasets: dict[str, h5py.Dataset] = {}
    dataset_object_addresses: dict[int, str] = {}
    with nullcontext():
        root_keys = set(handle.keys())
        expected_keys = set(DATASET_KEYS)
        for key in sorted(expected_keys - root_keys):
            add(f"{key}.missing", f"/{key}", "required dataset is missing")
        for key in sorted(root_keys - expected_keys):
            add(
                "dataset.unexpected",
                f"/{key}",
                "unexpected dataset or group is not part of schema version 1",
            )

        for key in DATASET_KEYS:
            if key not in root_keys:
                continue
            try:
                link = handle.get(key, getlink=True)
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                add(
                    f"{key}.link",
                    f"/{key}",
                    f"cannot inspect schema link: {error}",
                )
                continue
            if not isinstance(link, h5py.HardLink):
                add(
                    f"{key}.link",
                    f"/{key}",
                    "schema entry must be a direct hard link in this episode file",
                )
                continue
            try:
                dataset = handle[key]
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                add(
                    f"{key}.object",
                    f"/{key}",
                    f"cannot open schema object: {error}",
                )
                continue
            if not isinstance(dataset, h5py.Dataset):
                add(f"{key}.type", f"/{key}", "schema entry must be a dataset")
                continue
            try:
                is_virtual = bool(dataset.is_virtual)
                external_storage = dataset.external
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                add(
                    f"{key}.storage",
                    f"/{key}",
                    f"cannot inspect dataset storage: {error}",
                )
                continue
            if is_virtual:
                add(
                    f"{key}.storage.virtual",
                    f"/{key}",
                    "virtual datasets are not self-contained episode payloads",
                )
                continue
            if external_storage:
                add(
                    f"{key}.storage.external",
                    f"/{key}",
                    "external raw storage is not a self-contained episode payload",
                )
                continue
            try:
                dataset_attributes = tuple(dataset.attrs.keys())
            except (KeyError, OSError, RuntimeError, UnicodeError, ValueError) as error:
                add(
                    f"{key}.attributes",
                    f"/{key}",
                    f"cannot inspect dataset attribute names: {error}",
                )
                continue
            for attribute in sorted(dataset_attributes, key=str):
                add(
                    f"{key}.attribute.unexpected",
                    f"/{key}/@{attribute}",
                    "schema datasets must not have attributes",
                )
            try:
                object_address = int(h5py.h5o.get_info(dataset.id).addr)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                add(
                    f"{key}.object_identity",
                    f"/{key}",
                    f"cannot inspect dataset object identity: {error}",
                )
                continue
            first_key = dataset_object_addresses.get(object_address)
            if first_key is not None:
                add(
                    "dataset.object_alias",
                    f"/{key}",
                    f"schema datasets /{first_key} and /{key} must be distinct objects",
                )
            else:
                dataset_object_addresses[object_address] = key
            datasets[key] = dataset

        try:
            root_attributes = set(handle.attrs.keys())
        except (KeyError, OSError, RuntimeError, UnicodeError, ValueError) as error:
            add(
                "metadata.attributes",
                "/",
                f"cannot inspect root attribute names: {error}",
            )
            root_attributes = set()
        expected_attributes = set(REQUIRED_ATTRIBUTES)
        for name in sorted(root_attributes - expected_attributes, key=str):
            add(
                "metadata.attribute.unexpected",
                f"/@{name}",
                "unexpected root attribute is not part of schema version 1",
            )
        for name in REQUIRED_ATTRIBUTES:
            if name not in root_attributes:
                add(
                    f"metadata.{name}.missing",
                    f"/@{name}",
                    "required episode attribute is missing",
                )

        schema = handle.attrs.get("schema_version")
        if schema is not None and not _is_integer(schema):
            add(
                "metadata.schema_version.type",
                "/@schema_version",
                "schema_version must be an integer",
            )
        elif schema is not None and int(schema) != SCHEMA_VERSION:
            add(
                "metadata.schema_version.value",
                "/@schema_version",
                f"expected schema version {SCHEMA_VERSION}, got {schema}",
            )

        seed = handle.attrs.get("seed")
        if seed is not None and (not _is_integer(seed) or int(seed) < 0):
            add(
                "metadata.seed.value",
                "/@seed",
                "seed must be a non-negative integer",
            )

        raw_num_steps = handle.attrs.get("num_steps")
        if raw_num_steps is not None:
            if not _is_integer(raw_num_steps) or int(raw_num_steps) <= 0:
                add(
                    "metadata.num_steps.value",
                    "/@num_steps",
                    "num_steps must be a positive integer",
                )
            else:
                num_steps = int(raw_num_steps)

        control_dt = handle.attrs.get("control_dt_s")
        control_dt_value = np.nan
        valid_control_dt = False
        if control_dt is not None:
            if not _is_real_scalar(control_dt):
                add(
                    "metadata.control_dt_s.type",
                    "/@control_dt_s",
                    "control_dt_s must be a real numeric scalar",
                )
            else:
                control_dt_value = float(control_dt)
                valid_control_dt = bool(
                    np.isfinite(control_dt_value)
                    and np.isclose(
                        control_dt_value,
                        DEFAULT_CONTROL_DT_S,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                if not valid_control_dt:
                    add(
                        "metadata.control_dt_s.value",
                        "/@control_dt_s",
                        "control_dt_s must be 0.05 seconds (20 Hz)",
                    )

        raw_time_alignment = handle.attrs.get("time_alignment")
        time_alignment = _optional_text(raw_time_alignment)
        if raw_time_alignment is not None:
            if time_alignment is None:
                add(
                    "metadata.time_alignment.type",
                    "/@time_alignment",
                    "time_alignment must be a UTF-8 string",
                )
            elif time_alignment != TIME_ALIGNMENT:
                add(
                    "metadata.time_alignment.value",
                    "/@time_alignment",
                    f"expected {TIME_ALIGNMENT!r}, got {time_alignment!r}",
                )

        raw_action_semantics = handle.attrs.get("action_semantics")
        action_semantics = _optional_text(raw_action_semantics)
        if raw_action_semantics is not None:
            if action_semantics is None:
                add(
                    "metadata.action_semantics.type",
                    "/@action_semantics",
                    "action_semantics must be a UTF-8 string",
                )
            elif action_semantics != ACTION_SEMANTICS:
                add(
                    "metadata.action_semantics.value",
                    "/@action_semantics",
                    f"expected {ACTION_SEMANTICS!r}",
                )

        complete = handle.attrs.get("complete")
        if complete is not None and (
            not isinstance(complete, (bool, np.bool_)) or not bool(complete)
        ):
            add(
                "metadata.complete.value",
                "/@complete",
                "complete must be true for a published episode",
            )

        success = handle.attrs.get("success")
        success_value: bool | None = None
        if success is not None:
            if not isinstance(success, (bool, np.bool_)):
                add(
                    "metadata.success.type",
                    "/@success",
                    "success must be a boolean",
                )
            else:
                success_value = bool(success)

        failure_stage = _optional_text(handle.attrs.get("failure_stage"))
        if "failure_stage" in handle.attrs and failure_stage is None:
            add(
                "metadata.failure_stage.type",
                "/@failure_stage",
                "failure_stage must be a UTF-8 string",
            )
        if success_value is True and failure_stage:
            add(
                "metadata.failure_stage.success",
                "/@failure_stage",
                "successful episode must use an empty failure_stage",
            )
        if success_value is False and not failure_stage:
            add(
                "metadata.failure_stage.failure",
                "/@failure_stage",
                "failed episode must name its failure stage",
            )

        action_min, action_max = _validated_bounds_from_attrs(handle, add)

        expected_tails: dict[str, tuple[int, ...]] = {
            "observation.images.front": IMAGE_SHAPE,
            "observation.images.wrist": IMAGE_SHAPE,
            "observation.state": (STATE_DIM,),
            "action": (ACTION_DIM,),
            "timestamp": (),
            "stage": (),
        }
        expected_dtypes: dict[str, np.dtype[Any] | str] = {
            "observation.images.front": np.dtype(np.uint8),
            "observation.images.wrist": np.dtype(np.uint8),
            "observation.state": np.dtype(np.float32),
            "action": np.dtype(np.float32),
            "timestamp": np.dtype(np.float64),
            "stage": "utf-8",
        }
        lengths: dict[str, int] = {}
        value_scan_keys: set[str] = set()
        payload_structure_valid = set(datasets) == set(DATASET_KEYS) and len(
            dataset_object_addresses
        ) == len(DATASET_KEYS)
        for key in DATASET_KEYS:
            dataset = datasets.get(key)
            if dataset is None:
                continue
            expected_tail = expected_tails[key]
            shape_valid = (
                dataset.ndim == len(expected_tail) + 1
                and tuple(dataset.shape[1:]) == expected_tail
            )
            if not shape_valid:
                payload_structure_valid = False
                add(
                    f"{key}.shape",
                    f"/{key}",
                    f"expected shape [T, {', '.join(map(str, expected_tail))}], "
                    f"got {dataset.shape}",
                )
            lengths[key] = int(dataset.shape[0]) if dataset.ndim else -1
            if num_steps is not None and lengths[key] != num_steps:
                payload_structure_valid = False
                add(
                    f"{key}.length",
                    f"/{key}",
                    f"length {lengths[key]} does not match num_steps={num_steps}",
                )

            expected_dtype = expected_dtypes[key]
            dtype_valid = False
            if expected_dtype == "utf-8":
                string_info = h5py.check_string_dtype(dataset.dtype)
                if (
                    string_info is None
                    or string_info.encoding != "utf-8"
                    or string_info.length != STAGE_MAX_UTF8_BYTES
                ):
                    payload_structure_valid = False
                    add(
                        f"{key}.dtype",
                        f"/{key}",
                        "expected fixed UTF-8 string dtype with "
                        f"{STAGE_MAX_UTF8_BYTES}-byte items, got {dataset.dtype}",
                    )
                else:
                    dtype_valid = True
            elif dataset.dtype != expected_dtype:
                payload_structure_valid = False
                add(
                    f"{key}.dtype",
                    f"/{key}",
                    f"expected dtype {expected_dtype}, got {dataset.dtype}",
                )
            else:
                dtype_valid = True

            layout_valid = True
            try:
                chunks = dataset.chunks
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                layout_valid = False
                payload_structure_valid = False
                add(
                    f"{key}.layout",
                    f"/{key}",
                    f"cannot inspect dataset chunk layout: {error}",
                )
            else:
                if key == "stage" and chunks != STAGE_CHUNKS:
                    layout_valid = False
                    payload_structure_valid = False
                    add(
                        "stage.layout",
                        "/stage",
                        f"expected chunk layout {STAGE_CHUNKS}, got {chunks}",
                    )
                if chunks is not None:
                    chunk_elements = 1
                    for extent in chunks:
                        chunk_elements *= int(extent)
                    chunk_bytes = chunk_elements * int(dataset.dtype.itemsize)
                    if chunk_bytes > MAX_SCHEMA_CHUNK_BYTES:
                        layout_valid = False
                        payload_structure_valid = False
                        add(
                            f"{key}.chunk_bytes",
                            f"/{key}",
                            f"chunk requires {chunk_bytes} decoded bytes; maximum is "
                            f"{MAX_SCHEMA_CHUNK_BYTES}",
                        )

            if (
                shape_valid
                and dtype_valid
                and layout_valid
                and num_steps is not None
                and lengths[key] == num_steps
            ):
                value_scan_keys.add(key)

        if lengths and len(set(lengths.values())) > 1:
            add(
                "dataset.length_mismatch",
                "/",
                f"dataset lengths differ: {lengths}",
            )
        if num_steps == 0 or (
            num_steps is None and lengths and max(lengths.values()) == 0
        ):
            add("episode.empty", "/", "episode must contain at least one transition")

        if not payload_structure_valid:
            value_scan_keys.clear()

        for key in ("observation.state", "action"):
            dataset = datasets.get(key)
            if isinstance(dataset, h5py.Dataset) and key in value_scan_keys:
                if _dataset_has_non_finite(dataset):
                    add(
                        f"{key}.non_finite",
                        f"/{key}",
                        "dataset contains NaN or infinity",
                    )

        state_dataset = datasets.get("observation.state")
        if (
            isinstance(state_dataset, h5py.Dataset)
            and "observation.state" in value_scan_keys
            and _state_gripper_out_of_bounds(state_dataset)
        ):
            add(
                "observation.state.gripper_out_of_bounds",
                "/observation.state",
                "one or more gripper state values are outside [0, 1]",
            )

        action_dataset = datasets.get("action")
        if (
            isinstance(action_dataset, h5py.Dataset)
            and action_min is not None
            and action_max is not None
            and "action" in value_scan_keys
            and _action_out_of_bounds(action_dataset, action_min, action_max)
        ):
            add(
                "action.out_of_bounds",
                "/action",
                "one or more actions exceed /@action_min or /@action_max",
            )

        timestamp_dataset = datasets.get("timestamp")
        if (
            isinstance(timestamp_dataset, h5py.Dataset)
            and "timestamp" in value_scan_keys
        ):
            (
                has_non_finite,
                has_negative,
                has_non_increasing,
                has_wrong_control_dt,
            ) = _scan_timestamps(
                timestamp_dataset,
                control_dt_value=control_dt_value,
                check_control_dt=valid_control_dt,
            )
            if has_non_finite:
                add(
                    "timestamp.non_finite",
                    "/timestamp",
                    "timestamps must all be finite",
                )
            elif has_negative:
                add(
                    "timestamp.negative",
                    "/timestamp",
                    "timestamps must be non-negative",
                )
            if has_non_increasing:
                add(
                    "timestamp.not_strictly_increasing",
                    "/timestamp",
                    "timestamps must be strictly increasing",
                )
            elif has_wrong_control_dt:
                add(
                    "timestamp.control_dt",
                    "/timestamp",
                    "timestamp deltas do not match /@control_dt_s",
                )

        failure_stage_observed = False
        stage_dataset = datasets.get("stage")
        if isinstance(stage_dataset, h5py.Dataset) and "stage" in value_scan_keys:
            try:
                has_empty_stage, failure_stage_observed = _scan_stages(
                    stage_dataset,
                    failure_stage=failure_stage,
                )
            except (OSError, UnicodeError, ValueError) as error:
                add("stage.encoding", "/stage", f"cannot decode UTF-8 stages: {error}")
            else:
                if has_empty_stage:
                    add(
                        "stage.empty",
                        "/stage",
                        "stage labels must be non-empty strings",
                    )
        if (
            success_value is False
            and failure_stage
            and "stage" in value_scan_keys
            and not failure_stage_observed
        ):
            add(
                "metadata.failure_stage.not_observed",
                "/@failure_stage",
                "failure_stage does not appear in the transition stage labels",
            )

    return ValidationReport(episode_path, num_steps, tuple(issues))


def _metadata_from_handle(handle: h5py.File) -> EpisodeMetadata:
    missing = [name for name in REQUIRED_ATTRIBUTES if name not in handle.attrs]
    missing.extend(key for key in DATASET_KEYS if key not in handle)
    if missing:
        raise ValueError(f"episode is missing required fields: {sorted(missing)}")
    if int(handle.attrs["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={handle.attrs['schema_version']}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not bool(handle.attrs["complete"]):
        raise ValueError("episode is not complete")
    failure_stage = _optional_text(handle.attrs["failure_stage"])
    if failure_stage is None:
        raise ValueError("failure_stage metadata is not UTF-8 text")
    time_alignment = _optional_text(handle.attrs["time_alignment"])
    if time_alignment is None:
        raise ValueError("time_alignment metadata is not UTF-8 text")
    action_semantics = _optional_text(handle.attrs["action_semantics"])
    if action_semantics is None:
        raise ValueError("action_semantics metadata is not UTF-8 text")
    return EpisodeMetadata(
        schema_version=int(handle.attrs["schema_version"]),
        seed=int(handle.attrs["seed"]),
        success=bool(handle.attrs["success"]),
        failure_stage=failure_stage or None,
        num_steps=int(handle.attrs["num_steps"]),
        control_dt_s=float(handle.attrs["control_dt_s"]),
        time_alignment=time_alignment,
        action_semantics=action_semantics,
        action_min=np.asarray(handle.attrs["action_min"], dtype=np.float64).copy(),
        action_max=np.asarray(handle.attrs["action_max"], dtype=np.float64).copy(),
    )


def _normalize_action_bounds(
    action_bounds: tuple[ArrayLike, ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(action_bounds, tuple) or len(action_bounds) != 2:
        raise TypeError("action_bounds must be a (minimum, maximum) tuple")
    lower = _finite_vector(action_bounds[0], ACTION_DIM, "action minimum")
    upper = _finite_vector(action_bounds[1], ACTION_DIM, "action maximum")
    if np.any(lower >= upper):
        raise ValueError("each action minimum must be smaller than its maximum")
    expected_lower, expected_upper = _canonical_action_bounds()
    if not (
        np.array_equal(lower, expected_lower) and np.array_equal(upper, expected_upper)
    ):
        raise ValueError("action bounds must match the Panda environment contract")
    return lower.copy(), upper.copy()


def _finite_vector(value: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite {length}-vector") from error
    if array.shape != (length,):
        raise ValueError(f"{name} shape must be ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_image(value: ArrayLike, name: str) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if array.shape != IMAGE_SHAPE:
        raise ValueError(f"{name} shape must be {IMAGE_SHAPE}, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} dtype must be uint8, got {array.dtype}")
    return array


def _validate_stage_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text") from error
    if len(encoded) > STAGE_MAX_UTF8_BYTES:
        raise ValueError(f"{name} must be at most {STAGE_MAX_UTF8_BYTES} UTF-8 bytes")
    return value


def _validate_policy_visual_contract(env: Any) -> None:
    try:
        debug_viz = env.cfg.debug_viz
        site_names = [env.model.site(index).name for index in range(env.model.nsite)]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise TypeError(
            "callback env must expose debug_viz and named policy-observation sites"
        ) from error
    if debug_viz is not False:
        raise ValueError("callback env cfg.debug_viz must be exactly False")
    if site_names.count("tcp") != 1 or site_names.count("flange") != 1:
        raise ValueError(
            "environment model must contain exactly one tcp and one flange site"
        )

    site_alphas: list[float] = []
    for site_name in ("tcp", "flange"):
        site = env.model.site(site_names.index(site_name))
        rgba = np.asarray(site.rgba, dtype=np.float64)
        if rgba.shape != (4,):
            raise ValueError(
                f"{site_name} policy-observation site RGBA must be length 4"
            )
        site_alphas.append(float(rgba[3]))
    if not np.all(np.isfinite(site_alphas)) or any(
        alpha != 0.0 for alpha in site_alphas
    ):
        raise ValueError(
            "tcp and flange policy-observation site alpha must be finite and exactly 0"
        )


def _validated_bounds_from_attrs(
    handle: h5py.File,
    add: Any,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    expected_lower, expected_upper = _canonical_action_bounds()
    expected_by_name = {
        "action_min": expected_lower,
        "action_max": expected_upper,
    }
    values: list[NDArray[np.float64] | None] = []
    contract_mismatch = False
    for name in ("action_min", "action_max"):
        if name not in handle.attrs:
            values.append(None)
            continue
        array = np.asarray(handle.attrs[name])
        if array.dtype != np.dtype(np.float64):
            add(
                f"metadata.{name}.type",
                f"/@{name}",
                f"{name} must have dtype float64",
            )
            values.append(None)
            continue
        if array.shape != (ACTION_DIM,) or not np.all(np.isfinite(array)):
            add(
                f"metadata.{name}.value",
                f"/@{name}",
                f"{name} must be a finite {ACTION_DIM}-vector",
            )
            values.append(None)
        else:
            values.append(array)
            if not np.array_equal(array, expected_by_name[name]):
                contract_mismatch = True
    lower, upper = values
    if lower is not None and upper is not None and np.any(lower >= upper):
        add(
            "metadata.action_bounds.order",
            "/@action_min",
            "every action_min value must be smaller than action_max",
        )
    if contract_mismatch:
        add(
            "metadata.action_bounds.contract",
            "/@action_min",
            "action bounds must match the Panda environment contract",
        )
    return expected_lower, expected_upper


def _dataset_has_non_finite(dataset: h5py.Dataset) -> bool:
    for start in range(0, dataset.shape[0], VALIDATION_CHUNK_ROWS):
        values = dataset[start : start + VALIDATION_CHUNK_ROWS]
        if not np.all(np.isfinite(values)):
            return True
    return False


def _state_gripper_out_of_bounds(dataset: h5py.Dataset) -> bool:
    for start in range(0, dataset.shape[0], VALIDATION_CHUNK_ROWS):
        values = np.asarray(
            dataset[start : start + VALIDATION_CHUNK_ROWS, 7],
            dtype=np.float64,
        )
        if np.any(values < 0.0) or np.any(values > 1.0):
            return True
    return False


def _action_out_of_bounds(
    dataset: h5py.Dataset,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> bool:
    tolerance = 1e-6
    for start in range(0, dataset.shape[0], VALIDATION_CHUNK_ROWS):
        values = np.asarray(
            dataset[start : start + VALIDATION_CHUNK_ROWS],
            dtype=np.float64,
        )
        arm_values = values[:, :7]
        if np.any(arm_values < lower[:7] - tolerance) or np.any(
            arm_values > upper[:7] + tolerance
        ):
            return True
        gripper_values = values[:, 7]
        if np.any(gripper_values < 0.0) or np.any(gripper_values > 1.0):
            return True
    return False


def _scan_timestamps(
    dataset: h5py.Dataset,
    *,
    control_dt_value: float,
    check_control_dt: bool,
) -> tuple[bool, bool, bool, bool]:
    has_non_finite = False
    has_negative = False
    has_non_increasing = False
    has_wrong_control_dt = False
    previous: float | None = None

    for start in range(0, dataset.shape[0], VALIDATION_CHUNK_ROWS):
        values = np.asarray(
            dataset[start : start + VALIDATION_CHUNK_ROWS],
            dtype=np.float64,
        )
        has_non_finite = has_non_finite or not bool(np.all(np.isfinite(values)))
        has_negative = has_negative or bool(np.any(values < 0.0))
        if values.size == 0:
            continue

        if previous is not None:
            boundary_delta = float(values[0]) - previous
            has_non_increasing = has_non_increasing or boundary_delta <= 0.0
            if check_control_dt:
                has_wrong_control_dt = has_wrong_control_dt or not bool(
                    np.isclose(
                        boundary_delta,
                        control_dt_value,
                        rtol=1e-6,
                        atol=1e-9,
                    )
                )

        deltas = np.diff(values)
        if deltas.size:
            has_non_increasing = has_non_increasing or bool(np.any(deltas <= 0.0))
            if check_control_dt:
                has_wrong_control_dt = has_wrong_control_dt or not bool(
                    np.allclose(
                        deltas,
                        control_dt_value,
                        rtol=1e-6,
                        atol=1e-9,
                    )
                )
        previous = float(values[-1])

    return (
        has_non_finite,
        has_negative,
        has_non_increasing,
        has_wrong_control_dt,
    )


def _scan_stages(
    dataset: h5py.Dataset,
    *,
    failure_stage: str | None,
) -> tuple[bool, bool]:
    has_empty_stage = False
    failure_stage_observed = False
    decoded = dataset.asstr()
    for start in range(0, dataset.shape[0], VALIDATION_CHUNK_ROWS):
        values = decoded[start : start + VALIDATION_CHUNK_ROWS]
        for value in values:
            stage = str(value)
            has_empty_stage = has_empty_stage or not bool(stage.strip())
            failure_stage_observed = failure_stage_observed or stage == failure_stage
    return has_empty_stage, failure_stage_observed


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _canonical_action_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.asarray(CANONICAL_ACTION_MIN, dtype=np.float64),
        np.asarray(CANONICAL_ACTION_MAX, dtype=np.float64),
    )


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def _publish_no_clobber(
    partial_path: Path,
    target_path: Path,
    *,
    validated_snapshot: _FileSnapshot,
    validated_sha256: str,
) -> None:
    """Atomically expose a complete same-directory file without replacement."""

    _stable_sha256(
        partial_path,
        expected_snapshot=validated_snapshot,
        expected_digest=validated_sha256,
    )
    try:
        # A same-filesystem hard link is an atomic no-replace publication: the
        # destination either did not exist and now names the complete inode, or
        # link() fails with FileExistsError.  os.replace/rename would clobber.
        os.link(partial_path, target_path)
    except FileExistsError as error:
        raise FileExistsError(f"episode already exists: {target_path}") from error

    try:
        linked_partial_snapshot = _file_snapshot(partial_path)
        linked_target_snapshot = _file_snapshot(target_path)
        if linked_partial_snapshot != linked_target_snapshot or not (
            _matches_validated_content_metadata(
                linked_target_snapshot,
                validated_snapshot,
            )
        ):
            raise RuntimeError(
                "published target does not match the validated episode inode"
            )
        _, linked_snapshot = _stable_sha256(
            target_path,
            expected_snapshot=linked_target_snapshot,
            expected_digest=validated_sha256,
        )
        if _file_snapshot(partial_path) != linked_snapshot:
            raise RuntimeError("published links changed while verifying digest")

        validate_episode(target_path).raise_for_errors()
        _, pre_unlink_snapshot = _stable_sha256(
            target_path,
            expected_snapshot=linked_snapshot,
            expected_digest=validated_sha256,
        )
        if _file_snapshot(partial_path) != pre_unlink_snapshot:
            raise RuntimeError("published target changed during post-link validation")

        partial_path.unlink()
        unlinked_snapshot = _file_snapshot(target_path)
        if not _matches_validated_content_metadata(
            unlinked_snapshot,
            validated_snapshot,
        ):
            raise RuntimeError("published target metadata changed before fsync")
        _, pre_fsync_snapshot = _stable_sha256(
            target_path,
            expected_snapshot=unlinked_snapshot,
            expected_digest=validated_sha256,
        )

        _fsync_directory(target_path.parent)
        _stable_sha256(
            target_path,
            expected_snapshot=pre_fsync_snapshot,
            expected_digest=validated_sha256,
        )
    except Exception as error:
        target_matches_source, target_valid, target_sha256 = _inspect_published_target(
            target_path, validated_snapshot.identity
        )
        raise EpisodePublicationError(
            target_path=target_path,
            partial_path=partial_path,
            target_matches_source=target_matches_source,
            target_valid=target_valid,
            target_sha256=target_sha256,
        ) from error


def _file_identity(path: Path) -> tuple[int, int]:
    return _file_snapshot(path).identity


def _file_snapshot(path: Path) -> _FileSnapshot:
    status = os.stat(path, follow_symlinks=False)
    return _FileSnapshot(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        ctime_ns=status.st_ctime_ns,
    )


def _matches_validated_content_metadata(
    candidate: _FileSnapshot,
    validated: _FileSnapshot,
) -> bool:
    """Match fields that a hard-link count change must leave untouched.

    The caller separately proves each complete pre/post-link snapshot is stable
    while hashing it.  ctime is intentionally re-baselined only after our own
    link/unlink because those operations update inode metadata by definition.
    """

    return (
        candidate.identity == validated.identity
        and candidate.size == validated.size
        and candidate.mtime_ns == validated.mtime_ns
    )


def _stable_sha256(
    path: Path,
    *,
    expected_snapshot: _FileSnapshot | None = None,
    expected_digest: str | None = None,
) -> tuple[str, _FileSnapshot]:
    before = _file_snapshot(path)
    if expected_snapshot is not None and before != expected_snapshot:
        raise RuntimeError("episode file snapshot changed before hashing")
    digest = _sha256_file(path)
    after = _file_snapshot(path)
    if after != before:
        raise RuntimeError("episode file snapshot changed while hashing")
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError("episode file digest changed")
    return digest, after


def _inspect_published_target(
    target_path: Path,
    validated_identity: tuple[int, int],
) -> tuple[bool, bool, str | None]:
    try:
        initial_snapshot = _file_snapshot(target_path)
        if initial_snapshot.identity != validated_identity:
            return False, False, None
        digest, stable_snapshot = _stable_sha256(
            target_path,
            expected_snapshot=initial_snapshot,
        )
        report = validate_episode(target_path)
        if _file_snapshot(target_path) != stable_snapshot:
            return False, False, None
    except (KeyError, OSError, RuntimeError, ValueError):
        return False, False, None
    return True, report.valid, digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
