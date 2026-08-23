"""Run a deterministic, read-only smoke check of the project foundation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MENAGERIE", str(PROJECT_ROOT / "menagerie"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(PROJECT_ROOT / "cache" / "pycache"))

import mujoco
import numpy as np

from env.pick_place import PickPlace


def main() -> int:
    env = PickPlace()
    try:
        first = env.reset(np.random.default_rng(0))
        second = env.reset(np.random.default_rng(0))
        deterministic_reset = first == second

        cameras: dict[str, dict[str, object]] = {}
        for camera in ("front", "wrist"):
            image = env.render(camera)
            cameras[camera] = {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "non_blank": bool(float(image.std()) > 1.0),
            }

        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env.sid_tcp)
        arm_jacobian = np.vstack([jacp[:, :7], jacr[:, :7]])

        report = {
            "status": "ok",
            "mujoco": mujoco.__version__,
            "model": {
                "nq": env.model.nq,
                "nv": env.model.nv,
                "nu": env.model.nu,
                "ncam": env.model.ncam,
                "nsite": env.model.nsite,
                "physics_timestep_s": env.model.opt.timestep,
                "control_hz": env.cfg.control_hz,
            },
            "deterministic_reset": deterministic_reset,
            "cameras": cameras,
            "tcp_jacobian": {
                "shape": list(arm_jacobian.shape),
                "rank": int(np.linalg.matrix_rank(arm_jacobian)),
                "finite": bool(np.isfinite(arm_jacobian).all()),
            },
            "success_contract": {
                "hold_s": env.cfg.success_hold_s,
                "required_physics_steps": env.required_success_steps,
                "initial_success": env.success(),
            },
        }

        checks = [
            deterministic_reset,
            all(item["shape"] == [128, 128, 3] for item in cameras.values()),
            all(item["dtype"] == "uint8" for item in cameras.values()),
            all(item["non_blank"] for item in cameras.values()),
            arm_jacobian.shape == (6, 7),
            np.isfinite(arm_jacobian).all(),
            np.linalg.matrix_rank(arm_jacobian) == 6,
            not env.success(),
        ]
        if not all(checks):
            report["status"] = "failed"

        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
