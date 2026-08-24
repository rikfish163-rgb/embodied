"""Pure mapping boundary for LeRobotDataset v3 at pinned LeRobot 0.6.1.

This module neither imports nor installs LeRobot.  In the pinned official
writer, ``add_frame`` receives user features plus ``task`` and creates
``timestamp``/``frame_index`` itself; ``save_episode`` creates ``index``,
``episode_index`` and ``task_index``.  Those writer-managed keys therefore do
not appear in :func:`map_lerobot_frame`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .hdf5 import HDF5EpisodeReader, OBSERVATION_KEYS, Transition

LEROBOT_VERSION = "0.6.1"
LEROBOT_DATASET_CODEBASE_VERSION = "v3.0"
LEROBOT_FRAME_FEATURE_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "observation.state",
    "action",
)
LEROBOT_ADD_FRAME_KEYS = (*LEROBOT_FRAME_FEATURE_KEYS, "task")
LEROBOT_WRITER_MANAGED_KEYS = (
    "timestamp",
    "frame_index",
    "index",
    "episode_index",
    "task_index",
)
CONTROL_FPS = 20


class LeRobotVersionError(RuntimeError):
    """The optional LeRobot runtime is outside the audited boundary."""


def require_lerobot_version(version: str) -> str:
    if version != LEROBOT_VERSION:
        raise LeRobotVersionError(
            f"adapter supports exactly LeRobot {LEROBOT_VERSION}, got {version!r}"
        )
    return version


def lerobot_feature_spec(*, use_videos: bool = True) -> dict[str, dict[str, Any]]:
    """Return user-defined features accepted by ``LeRobotDataset.create``."""

    visual_dtype = "video" if use_videos else "image"
    joints = [f"joint_{index}.position" for index in range(7)] + ["gripper.open"]
    return {
        "observation.images.front": {
            "dtype": visual_dtype,
            "shape": [128, 128, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": visual_dtype,
            "shape": [128, 128, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": joints.copy(),
        },
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": joints.copy(),
        },
    }


def map_lerobot_frame(transition: Transition, *, task: str) -> dict[str, Any]:
    """Map one validated HDF5 row to official ``add_frame`` user keys."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be non-empty text")
    if set(transition.observation) != set(OBSERVATION_KEYS):
        raise ValueError("transition observation must match the policy allowlist")
    front = _image(transition.observation["observation.images.front"], "front")
    wrist = _image(transition.observation["observation.images.wrist"], "wrist")
    state = _float32_vector(
        transition.observation["observation.state"],
        "observation.state",
    )
    action = _float32_vector(transition.action, "action")
    frame = {
        "observation.images.front": front.copy(),
        "observation.images.wrist": wrist.copy(),
        "observation.state": state.copy(),
        "action": action.copy(),
        "task": task,
    }
    if set(frame) != set(LEROBOT_ADD_FRAME_KEYS):
        raise RuntimeError("adapter emitted a non-allowlisted LeRobot frame key")
    return frame


def iter_lerobot_episode(
    reader: HDF5EpisodeReader,
    *,
    task: str,
) -> Iterator[dict[str, Any]]:
    """Yield an episode in order without exposing audit metadata as features."""

    first_timestamp: float | None = None
    for frame_index, transition in enumerate(reader):
        if first_timestamp is None:
            first_timestamp = transition.timestamp
        relative_timestamp = transition.timestamp - first_timestamp
        expected = frame_index / CONTROL_FPS
        if not np.isclose(relative_timestamp, expected, rtol=1e-6, atol=1e-9):
            raise ValueError("source timestamps do not match the 20 Hz writer boundary")
        yield map_lerobot_frame(transition, task=task)


def map_lerobot_episode(
    reader: HDF5EpisodeReader,
    *,
    task: str,
) -> dict[str, Any]:
    """Describe one optional conversion call without invoking LeRobot."""

    frames = tuple(iter_lerobot_episode(reader, task=task))
    return {
        "adapter_version": f"lerobot-{LEROBOT_VERSION}-dataset-{LEROBOT_DATASET_CODEBASE_VERSION}",
        "fps": CONTROL_FPS,
        "length": len(frames),
        "frames": frames,
        "writer_managed_keys": LEROBOT_WRITER_MANAGED_KEYS,
    }


def _image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (128, 128, 3) or array.dtype != np.uint8:
        raise ValueError(f"{name} image must be uint8 [128,128,3] HWC RGB")
    return array


def _float32_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (8,) or array.dtype != np.float32:
        raise ValueError(f"{name} must be float32 [8]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
