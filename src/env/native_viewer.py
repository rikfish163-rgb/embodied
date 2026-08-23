"""Launch the project scene in MuJoCo's native Simulate GUI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from .pick_place import PickPlace, TaskConfig


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the Panda pick-and-place task in MuJoCo's native viewer."
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=0,
        help="deterministic scene seed (default: 0)",
    )
    parser.add_argument(
        "--debug-sites",
        action="store_true",
        help="show the TCP and flange sites; policy observations keep them hidden",
    )
    return parser


def create_environment(*, seed: int, debug_sites: bool = False) -> PickPlace:
    """Build and reset the exact task model that will be shown in the GUI."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    environment = PickPlace(TaskConfig(debug_viz=debug_sites))
    environment.reset(np.random.default_rng(seed))
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    environment = create_environment(seed=args.seed, debug_sites=args.debug_sites)

    print(f"MuJoCo {mujoco.__version__} native viewer | scene seed={args.seed}")
    print("Close the MuJoCo window to return to the terminal.")
    try:
        mujoco.viewer.launch(
            environment.model,
            environment.data,
            show_left_ui=True,
            show_right_ui=True,
        )
    except mujoco.FatalError as error:
        parser.exit(
            1,
            "Unable to open MuJoCo's native GLFW window. "
            "Run this command from a graphical desktop session "
            f"(DISPLAY/WAYLAND_DISPLAY must be available).\nDetails: {error}\n",
        )
    finally:
        environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
