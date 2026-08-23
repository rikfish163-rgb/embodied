from __future__ import annotations

import numpy as np


def test_native_viewer_launches_the_seeded_pick_place_scene(monkeypatch) -> None:
    from env import native_viewer

    launched: dict[str, object] = {}

    def fake_launch(model, data, **kwargs) -> None:
        launched["model"] = model
        launched["data"] = data
        launched["kwargs"] = kwargs
        launched["qpos"] = data.qpos.copy()
        launched["ctrl"] = data.ctrl.copy()

    monkeypatch.setattr(native_viewer.mujoco.viewer, "launch", fake_launch)

    result = native_viewer.main(["--seed", "17", "--debug-sites"])

    assert result == 0
    assert launched["model"].ncam == 2
    assert launched["model"].nsite == 2
    assert launched["kwargs"] == {"show_left_ui": True, "show_right_ui": True}
    np.testing.assert_allclose(launched["ctrl"][:7], launched["qpos"][:7])
    assert launched["ctrl"][7] == 255.0

    # The same seed must reproduce the exact scene shown in the native viewer.
    expected = native_viewer.create_environment(seed=17, debug_sites=True)
    try:
        np.testing.assert_allclose(launched["qpos"], expected.data.qpos)
    finally:
        expected.close()


def test_native_viewer_rejects_negative_seed() -> None:
    from env import native_viewer

    parser = native_viewer.create_parser()

    try:
        parser.parse_args(["--seed", "-1"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("negative seed should be rejected")
