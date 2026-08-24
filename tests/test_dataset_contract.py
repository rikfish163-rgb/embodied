from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import data.hdf5 as hdf5_module
from data import (
    DatasetValidationError,
    HDF5EpisodeReader,
    HDF5EpisodeWriter,
    validate_episode,
)
from env.pick_place import PickPlace, TaskConfig
from expert.scripted import ExpertConfig, run_episode


def _observation(step: int = 0) -> dict[str, np.ndarray]:
    return {
        "observation.images.front": np.full((128, 128, 3), step, dtype=np.uint8),
        "observation.images.wrist": np.full((128, 128, 3), step + 1, dtype=np.uint8),
        "observation.state": np.array(
            [0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785, step / 10],
            dtype=np.float64,
        ),
    }


def _action_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0],
            dtype=np.float64,
        ),
        np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0],
            dtype=np.float64,
        ),
    )


def _valid_action(*, gripper: float = 0.5) -> np.ndarray:
    return np.array(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, gripper],
        dtype=np.float64,
    )


def _write_valid_episode(path: Path, *, steps: int = 2) -> None:
    with HDF5EpisodeWriter(path, seed=7, action_bounds=_action_bounds()) as writer:
        for index in range(steps):
            writer.append(
                _observation(index),
                _valid_action(),
                timestamp=index * 0.05,
                stage="pregrasp",
            )
        writer.finalize(success=True, failure_stage=None)


def _mutate_valid_timestamp_and_restore_mtime(path: Path, value: float) -> None:
    before = os.stat(path, follow_symlinks=False)
    with h5py.File(path, "r+") as handle:
        handle["timestamp"][0] = np.float64(value)
    os.utime(
        path,
        ns=(before.st_atime_ns, before.st_mtime_ns),
        follow_symlinks=False,
    )
    after = os.stat(path, follow_symlinks=False)
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns


def test_writer_reader_and_validator_freeze_the_streaming_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode_000007.h5"

    _write_valid_episode(path)

    report = validate_episode(path)
    assert report.valid, report.format_errors()
    assert report.num_steps == 2

    with h5py.File(path, "r") as handle:
        assert set(handle) == {
            "action",
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
            "stage",
            "timestamp",
        }
        assert handle["observation.images.front"].shape == (2, 128, 128, 3)
        assert handle["observation.images.front"].dtype == np.dtype("uint8")
        assert handle["observation.state"].dtype == np.dtype("float32")
        assert handle["action"].dtype == np.dtype("float32")
        assert handle["timestamp"].dtype == np.dtype("float64")
        assert handle.attrs["schema_version"] == 1
        assert handle.attrs["seed"] == 7
        assert bool(handle.attrs["success"])
        assert handle.attrs["failure_stage"] == ""
        assert handle.attrs["time_alignment"] == "pre_action"
        assert handle.attrs["control_dt_s"] == pytest.approx(0.05)
        assert handle.attrs["action_semantics"] == (
            "absolute_joint_position_targets_rad[7]+normalized_gripper_open[1]"
        )
        assert set(handle.attrs) == {
            "action_max",
            "action_min",
            "action_semantics",
            "complete",
            "control_dt_s",
            "failure_stage",
            "num_steps",
            "schema_version",
            "seed",
            "success",
            "time_alignment",
        }
        for key in (
            "action",
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
            "stage",
            "timestamp",
        ):
            assert not handle[key].attrs
        object_addresses = {
            h5py.h5o.get_info(handle[key].id).addr for key in hdf5_module.DATASET_KEYS
        }
        assert len(object_addresses) == len(hdf5_module.DATASET_KEYS)

    with HDF5EpisodeReader(path) as reader:
        assert len(reader) == 2
        assert reader.metadata.seed == 7
        assert reader.metadata.success
        assert reader.metadata.failure_stage is None
        transition = reader[1]
        assert transition.timestamp == pytest.approx(0.05)
        assert transition.stage == "pregrasp"
        assert set(transition.observation) == {
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
        }
        assert transition.observation["observation.images.front"].dtype == np.uint8
        assert transition.observation["observation.state"].shape == (8,)
        np.testing.assert_allclose(transition.action, _valid_action())


def test_writer_freezes_bounded_fixed_stage_and_chunk_layout(tmp_path: Path) -> None:
    path = tmp_path / "bounded-layout.h5"

    _write_valid_episode(path, steps=1)

    with h5py.File(path, "r") as handle:
        stage = handle["stage"]
        string_info = h5py.check_string_dtype(stage.dtype)
        assert string_info is not None
        assert string_info.encoding == "utf-8"
        assert string_info.length == 64
        assert stage.chunks == (256,)
        for key in hdf5_module.DATASET_KEYS:
            dataset = handle[key]
            assert dataset.chunks is not None
            chunk_bytes = int(np.prod(dataset.chunks)) * dataset.dtype.itemsize
            assert chunk_bytes <= 16 * 1024 * 1024, key

    report = validate_episode(path)
    assert report.valid, report.format_errors()
    with HDF5EpisodeReader(path) as reader:
        assert reader[0].stage == "pregrasp"


def test_writer_rejects_stage_label_over_fixed_utf8_capacity(tmp_path: Path) -> None:
    path = tmp_path / "oversized-stage-label.h5"

    with HDF5EpisodeWriter(path, seed=7, action_bounds=_action_bounds()) as writer:
        with pytest.raises(ValueError, match="64 UTF-8 bytes"):
            writer.append(
                _observation(),
                _valid_action(),
                timestamp=0.0,
                stage="界" * 22,
            )
        writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
        writer.finalize(success=True, failure_stage=None)

    assert validate_episode(path).valid


def test_writer_rejects_failure_stage_over_fixed_utf8_capacity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-failure-stage.h5"

    with HDF5EpisodeWriter(path, seed=7, action_bounds=_action_bounds()) as writer:
        writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
        with pytest.raises(ValueError, match="failure_stage.*64 UTF-8 bytes"):
            writer.finalize(success=False, failure_stage="界" * 22)
        writer.finalize(success=False, failure_stage="pregrasp")

    assert validate_episode(path).valid


def test_writer_is_a_pre_action_callback_for_the_m1_expert(tmp_path: Path) -> None:
    path = tmp_path / "expert_failure.h5"
    env = PickPlace()
    try:
        assert env.cfg.debug_viz is False
        assert [float(env.model.site(name).rgba[3]) for name in ("tcp", "flange")] == [
            0.0,
            0.0,
        ]
        with HDF5EpisodeWriter(path, seed=0) as writer:
            result = run_episode(
                env,
                seed=0,
                config=ExpertConfig(max_move_steps=1, max_attempts=1),
                record_steps=True,
                step_callback=writer.capture,
            )
            writer.finalize(
                success=result.success,
                failure_stage=result.failure_stage,
            )
    finally:
        env.close()

    assert not result.success
    assert result.failure_stage == "pregrasp"
    with HDF5EpisodeReader(path) as reader:
        assert len(reader) == result.control_steps == 1
        assert reader.metadata.failure_stage == "pregrasp"
        assert reader[0].timestamp == pytest.approx(0.0)
        np.testing.assert_allclose(reader[0].action, result.steps[0].action)


def test_capture_rejects_real_environment_debug_markers_before_first_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "debug-markers.h5"
    env = PickPlace(TaskConfig(debug_viz=True))
    writer = HDF5EpisodeWriter(path, seed=0)
    try:
        with writer:
            with pytest.raises(ValueError, match="debug_viz"):
                writer.capture(env, "pregrasp", _valid_action())
            assert writer._num_steps == 0
    finally:
        env.close()

    assert not path.exists()
    assert not writer._partial_path.exists()


def test_capture_requires_debug_viz_to_be_exactly_false(tmp_path: Path) -> None:
    path = tmp_path / "non-boolean-debug-flag.h5"
    env = PickPlace(TaskConfig(debug_viz=False))
    env.cfg.debug_viz = np.bool_(False)
    writer = HDF5EpisodeWriter(path, seed=0)
    try:
        with writer:
            with pytest.raises(ValueError, match="debug_viz"):
                writer.capture(env, "pregrasp", _valid_action())
            assert writer._num_steps == 0
    finally:
        env.close()

    assert not path.exists()


@pytest.mark.parametrize(
    ("site_name", "alpha"),
    [("tcp", 0.6), ("flange", 0.6), ("tcp", np.nan)],
)
def test_capture_rejects_visible_or_non_finite_policy_sites(
    tmp_path: Path,
    site_name: str,
    alpha: float,
) -> None:
    path = tmp_path / f"bad-site-alpha-{site_name}-{alpha}.h5"
    env = PickPlace(TaskConfig(debug_viz=False))
    env.model.site(site_name).rgba[3] = alpha
    writer = HDF5EpisodeWriter(path, seed=0)
    try:
        with writer:
            with pytest.raises(ValueError, match="site alpha"):
                writer.capture(env, "pregrasp", _valid_action())
            assert writer._num_steps == 0
    finally:
        env.close()

    assert not path.exists()


def test_capture_rechecks_visual_leakage_before_every_frame(tmp_path: Path) -> None:
    path = tmp_path / "site-became-visible.h5"
    env = PickPlace(TaskConfig(debug_viz=False))
    writer = HDF5EpisodeWriter(path, seed=0)
    try:
        with writer:
            writer.capture(env, "pregrasp", _valid_action())
            assert writer._num_steps == 1
            env.model.site("tcp").rgba[3] = 0.6
            env.data.time = 0.05
            with pytest.raises(ValueError, match="site alpha"):
                writer.capture(env, "pregrasp", _valid_action())
            assert writer._num_steps == 1
    finally:
        env.close()

    assert not path.exists()


@pytest.mark.parametrize(
    "site_names",
    [("tcp",), ("tcp", "flange", "tcp")],
    ids=["missing", "duplicate"],
)
def test_capture_requires_exactly_one_tcp_and_flange_site(
    tmp_path: Path,
    site_names: tuple[str, ...],
) -> None:
    path = tmp_path / f"bad-site-names-{len(site_names)}.h5"
    env = PickPlace(TaskConfig(debug_viz=False))
    base_model = env.model

    class SiteModelProxy:
        actuator_ctrlrange = base_model.actuator_ctrlrange
        nsite = len(site_names)

        def site(self, index: int) -> object:
            source = base_model.site("tcp" if site_names[index] == "tcp" else "flange")

            class SiteProxy:
                name = site_names[index]
                rgba = source.rgba

            return SiteProxy()

    class EnvironmentProxy:
        cfg = env.cfg
        data = env.data
        model = SiteModelProxy()

        @staticmethod
        def observe() -> dict[str, np.ndarray]:
            return _observation()

    environment = EnvironmentProxy()
    writer = HDF5EpisodeWriter(path, seed=0)
    try:
        with writer:
            with pytest.raises(ValueError, match="exactly one.*tcp.*flange"):
                writer.capture(environment, "pregrasp", _valid_action())
            assert writer._num_steps == 0
    finally:
        env.close()

    assert not path.exists()


def test_writer_never_overwrites_and_abort_removes_partial_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.h5"
    existing.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        HDF5EpisodeWriter(existing, seed=0, action_bounds=_action_bounds())
    assert existing.read_bytes() == b"keep me"

    aborted = tmp_path / "aborted.h5"
    with pytest.raises(RuntimeError, match="stop collection"):
        with HDF5EpisodeWriter(
            aborted, seed=0, action_bounds=_action_bounds()
        ) as writer:
            writer.append(
                _observation(),
                _valid_action(),
                timestamp=0.0,
                stage="pregrasp",
            )
            raise RuntimeError("stop collection")

    assert not aborted.exists()
    assert not list(tmp_path.glob(".*.partial-*"))

    raced = tmp_path / "raced.h5"
    writer = HDF5EpisodeWriter(raced, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    raced.write_bytes(b"winner")
    with pytest.raises(FileExistsError):
        writer.finalize(success=False, failure_stage="pregrasp")
    assert raced.read_bytes() == b"winner"
    assert not list(tmp_path.glob(".*.partial-*"))


@pytest.mark.parametrize("failure_point", ["partial_unlink", "directory_fsync"])
def test_post_link_publication_fault_has_a_reconcilable_typed_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    path = tmp_path / f"post-link-{failure_point}.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")

    if failure_point == "partial_unlink":
        real_unlink = Path.unlink

        def fail_partial_unlink(
            candidate: Path,
            *args: object,
            **kwargs: object,
        ) -> None:
            if candidate == writer._partial_path:
                raise OSError("injected partial unlink failure")
            real_unlink(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_partial_unlink)
    else:

        def fail_directory_fsync(path: Path) -> None:
            del path
            raise OSError("injected directory fsync failure")

        monkeypatch.setattr(hdf5_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(hdf5_module.EpisodePublicationError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    error = caught.value
    target_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert error.published is True
    assert error.state == "publication_indeterminate"
    assert error.target_path == path
    assert error.partial_path == writer._partial_path
    assert error.target_matches_source is True
    assert error.target_valid is True
    assert error.target_sha256 == target_digest
    assert writer._state == "publication_indeterminate"
    assert validate_episode(path).valid
    if writer._partial_path.exists():
        assert writer._partial_path.samefile(path)

    with pytest.raises(FileExistsError):
        HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == target_digest


@pytest.mark.parametrize("mutation", ["replace", "overwrite"])
def test_writer_does_not_publish_a_partial_changed_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    path = tmp_path / f"pre-link-{mutation}.h5"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"not-an-hdf5-episode")
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_validate_episode = hdf5_module.validate_episode

    def validate_then_mutate(candidate: object) -> object:
        report = real_validate_episode(candidate)
        if Path(candidate) == writer._partial_path and report.valid:
            if mutation == "replace":
                replacement.replace(writer._partial_path)
            else:
                writer._partial_path.write_bytes(b"not-an-hdf5-episode")
        return report

    monkeypatch.setattr(hdf5_module, "validate_episode", validate_then_mutate)

    with pytest.raises(RuntimeError, match="changed after validation"):
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    assert writer._state == "aborted"
    assert not path.exists()
    assert not writer._partial_path.exists()


def test_writer_binds_full_snapshot_and_digest_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pre-link-restored-mtime.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_publish = hdf5_module._publish_no_clobber

    def mutate_before_publish(
        partial_path: Path,
        target_path: Path,
        **kwargs: object,
    ) -> None:
        _mutate_valid_timestamp_and_restore_mtime(partial_path, 0.01)
        assert validate_episode(partial_path).valid
        real_publish(partial_path, target_path, **kwargs)

    monkeypatch.setattr(
        hdf5_module,
        "_publish_no_clobber",
        mutate_before_publish,
    )

    with pytest.raises(RuntimeError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    assert not isinstance(caught.value, hdf5_module.EpisodePublicationError)
    assert writer._state == "aborted"
    assert not path.exists()
    assert not writer._partial_path.exists()


def test_writer_rejects_a_partial_changed_while_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pre-link-hash-race.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_sha256_file = hdf5_module._sha256_file
    mutated = False

    def mutate_during_hash(candidate: Path) -> str:
        nonlocal mutated
        digest = real_sha256_file(candidate)
        if candidate == writer._partial_path and not mutated:
            mutated = True
            _mutate_valid_timestamp_and_restore_mtime(candidate, 0.01)
        return digest

    monkeypatch.setattr(hdf5_module, "_sha256_file", mutate_during_hash)

    with pytest.raises(RuntimeError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    assert not isinstance(caught.value, hdf5_module.EpisodePublicationError)
    assert writer._state == "aborted"
    assert not path.exists()
    assert not writer._partial_path.exists()


def test_post_link_schema_corruption_cannot_return_publication_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "post-link-corruption.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_link = hdf5_module.os.link

    def link_then_corrupt(source: object, target: object) -> None:
        real_link(source, target)
        Path(target).write_bytes(b"not-an-hdf5-episode")

    monkeypatch.setattr(hdf5_module.os, "link", link_then_corrupt)

    with pytest.raises(hdf5_module.EpisodePublicationError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    error = caught.value
    target_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert error.published is True
    assert error.state == "publication_indeterminate"
    assert error.target_path == path
    assert error.target_matches_source is True
    assert error.target_valid is False
    assert error.target_sha256 == target_digest
    assert writer._state == "publication_indeterminate"
    assert not validate_episode(path).valid


@pytest.mark.parametrize("mutation", ["replace", "overwrite"])
def test_target_changed_after_post_link_validation_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    path = tmp_path / f"post-validation-target-{mutation}.h5"
    replacement = tmp_path / "post-validation-replacement.bin"
    replacement.write_bytes(b"not-an-hdf5-episode")
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_validate_episode = hdf5_module.validate_episode

    def validate_then_mutate_target(candidate: object) -> object:
        report = real_validate_episode(candidate)
        if Path(candidate) == path and report.valid:
            if mutation == "replace":
                replacement.replace(path)
            else:
                path.write_bytes(b"not-an-hdf5-episode")
        return report

    monkeypatch.setattr(
        hdf5_module,
        "validate_episode",
        validate_then_mutate_target,
    )

    with pytest.raises(hdf5_module.EpisodePublicationError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    error = caught.value
    assert error.published is True
    assert error.state == "publication_indeterminate"
    assert error.target_path == path
    assert error.target_matches_source is (mutation == "overwrite")
    assert error.target_valid is False
    if mutation == "overwrite":
        assert error.target_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        assert error.target_sha256 is None
    assert writer._state == "publication_indeterminate"
    assert not real_validate_episode(path).valid


def test_post_link_valid_mutation_with_restored_mtime_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "post-link-restored-mtime.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_unlink = Path.unlink
    digests: list[str] = []

    def mutate_before_partial_unlink(
        candidate: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if candidate == writer._partial_path and not digests:
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            _mutate_valid_timestamp_and_restore_mtime(path, 0.01)
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            assert validate_episode(path).valid
        real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", mutate_before_partial_unlink)

    with pytest.raises(hdf5_module.EpisodePublicationError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    error = caught.value
    assert digests[0] != digests[1]
    assert error.target_matches_source is True
    assert error.target_valid is True
    assert error.target_sha256 == digests[1]
    assert writer._state == "publication_indeterminate"
    assert validate_episode(path).valid


def test_post_link_change_while_hashing_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "post-link-hash-race.h5"
    writer = HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds())
    writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
    real_sha256_file = hdf5_module._sha256_file
    mutated = False

    def mutate_during_target_hash(candidate: Path) -> str:
        nonlocal mutated
        digest = real_sha256_file(candidate)
        if candidate == path and not mutated:
            mutated = True
            _mutate_valid_timestamp_and_restore_mtime(candidate, 0.01)
        return digest

    monkeypatch.setattr(
        hdf5_module,
        "_sha256_file",
        mutate_during_target_hash,
    )

    with pytest.raises(hdf5_module.EpisodePublicationError) as caught:
        with writer:
            writer.finalize(success=False, failure_stage="pregrasp")

    error = caught.value
    assert error.target_matches_source is True
    assert error.target_valid is True
    assert error.target_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert writer._state == "publication_indeterminate"
    assert validate_episode(path).valid


def test_writer_rejects_a_control_period_other_than_20_hz(tmp_path: Path) -> None:
    path = tmp_path / "wrong-rate.h5"

    with pytest.raises(ValueError, match="20 Hz"):
        with HDF5EpisodeWriter(
            path,
            seed=0,
            action_bounds=_action_bounds(),
            control_dt_s=0.1,
        ):
            pass

    assert not path.exists()
    assert not list(tmp_path.glob(".*.partial-*"))


def test_validator_rejects_an_episode_recorded_at_a_rate_other_than_20_hz(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-rate.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle.attrs["control_dt_s"] = np.float64(0.1)
        handle["timestamp"][1] = np.float64(0.1)

    report = validate_episode(path)

    assert "metadata.control_dt_s.value" in {issue.code for issue in report.errors}


@pytest.mark.parametrize("attribute", ["time_alignment", "action_semantics"])
def test_validator_rejects_non_text_contract_metadata(
    tmp_path: Path,
    attribute: str,
) -> None:
    path = tmp_path / f"bad-{attribute}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle.attrs[attribute] = np.int64(7)

    report = validate_episode(path)

    assert f"metadata.{attribute}.type" in {issue.code for issue in report.errors}


@pytest.mark.parametrize("attribute", ["time_alignment", "action_semantics"])
def test_reader_rejects_non_text_contract_metadata(
    tmp_path: Path,
    attribute: str,
) -> None:
    path = tmp_path / f"bad-reader-{attribute}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle.attrs[attribute] = np.int64(7)

    with pytest.raises(ValueError, match=attribute):
        HDF5EpisodeReader(path)


@pytest.mark.parametrize(
    "value_kind",
    ["scalar", "text", "object_reference", "region_reference"],
)
def test_validator_and_reader_reject_unexpected_root_attributes(
    tmp_path: Path,
    value_kind: str,
) -> None:
    path = tmp_path / f"unexpected-root-attribute-{value_kind}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        if value_kind == "scalar":
            value: object = np.float64(0.4)
        elif value_kind == "text":
            value = "privileged"
        elif value_kind == "object_reference":
            value = handle["action"].ref
        else:
            value = handle["action"].regionref[0:1]
        handle.attrs["privileged.payload"] = value

    report = validate_episode(path)

    assert "metadata.attribute.unexpected" in {issue.code for issue in report.errors}
    with pytest.raises(DatasetValidationError) as caught:
        HDF5EpisodeReader(path)
    assert "metadata.attribute.unexpected" in {
        issue.code for issue in caught.value.report.errors
    }


@pytest.mark.parametrize(
    "value_kind",
    ["scalar", "text", "object_reference", "region_reference"],
)
def test_validator_and_reader_reject_schema_dataset_attributes(
    tmp_path: Path,
    value_kind: str,
) -> None:
    path = tmp_path / f"unexpected-dataset-attribute-{value_kind}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        if value_kind == "scalar":
            value: object = np.float64(0.4)
        elif value_kind == "text":
            value = "opaque"
        elif value_kind == "object_reference":
            value = handle["observation.state"].ref
        else:
            value = handle["observation.state"].regionref[0:1]
        handle["action"].attrs["opaque"] = value

    report = validate_episode(path)

    assert "action.attribute.unexpected" in {issue.code for issue in report.errors}
    with pytest.raises(DatasetValidationError) as caught:
        HDF5EpisodeReader(path)
    assert "action.attribute.unexpected" in {
        issue.code for issue in caught.value.report.errors
    }


@pytest.mark.parametrize("key", ["observation.state", "action"])
def test_validator_reports_shape_for_scalar_numeric_dataset(
    tmp_path: Path,
    key: str,
) -> None:
    path = tmp_path / f"scalar-{key}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        del handle[key]
        handle.create_dataset(key, data=np.float32(0.0))

    report = validate_episode(path)

    assert not report.valid
    assert f"{key}.shape" in {issue.code for issue in report.errors}


@pytest.mark.parametrize(
    ("key", "corruption", "expected_code"),
    [
        ("observation.state", "huge_tail", "observation.state.shape"),
        ("action", "huge_tail", "action.shape"),
        ("observation.state", "wrong_dtype", "observation.state.dtype"),
        ("action", "wrong_dtype", "action.dtype"),
        ("observation.state", "huge_length", "observation.state.length"),
        ("action", "huge_length", "action.length"),
    ],
)
def test_validator_never_reads_malformed_numeric_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    corruption: str,
    expected_code: str,
) -> None:
    path = tmp_path / f"malformed-{key}-{corruption}.h5"
    _write_valid_episode(path, steps=1)

    with h5py.File(path, "r+") as handle:
        del handle[key]
        if corruption == "huge_tail":
            shape = (1, 100_000_000)
            dtype = np.float32
            chunks = (1, 1024)
        elif corruption == "huge_length":
            shape = (100_000_000, 8)
            dtype = np.float32
            chunks = (256, 8)
        else:
            shape = (1, 8)
            dtype = np.float64
            chunks = (1, 8)
        handle.create_dataset(
            key,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            fillvalue=0,
        )

    real_getitem = h5py.Dataset.__getitem__

    def reject_malformed_payload_read(
        dataset: h5py.Dataset,
        selection: object,
    ) -> object:
        if dataset.name == f"/{key}":
            raise AssertionError(
                f"validator read malformed payload {dataset.name} {selection!r}"
            )
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_malformed_payload_read)

    report = validate_episode(path)

    assert expected_code in {issue.code for issue in report.errors}


def test_validator_rejects_oversized_fixed_stage_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized-fixed-stage.h5"
    _write_valid_episode(path, steps=1)

    with h5py.File(path, "r+") as handle:
        del handle["stage"]
        handle.create_dataset(
            "stage",
            shape=(1,),
            maxshape=(None,),
            chunks=(1,),
            dtype=h5py.string_dtype(encoding="utf-8", length=16 * 1024 * 1024),
        )

    real_getitem = h5py.Dataset.__getitem__

    def reject_stage_payload_read(
        dataset: h5py.Dataset,
        selection: object,
    ) -> object:
        if dataset.name in {f"/{name}" for name in hdf5_module.DATASET_KEYS}:
            raise AssertionError(
                f"validator read payload before rejecting /stage: "
                f"{dataset.name} {selection!r}"
            )
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_stage_payload_read)

    report = validate_episode(path)

    assert "stage.dtype" in {issue.code for issue in report.errors}


def test_validator_rejects_vlen_stage_before_reading_single_huge_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized-vlen-stage.h5"
    _write_valid_episode(path, steps=1)

    with h5py.File(path, "r+") as handle:
        del handle["stage"]
        stage = handle.create_dataset(
            "stage",
            shape=(1,),
            maxshape=(None,),
            chunks=(1,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        stage[0] = "x" * (4 * 1024 * 1024)

    real_getitem = h5py.Dataset.__getitem__

    def reject_stage_payload_read(
        dataset: h5py.Dataset,
        selection: object,
    ) -> object:
        if dataset.name in {f"/{name}" for name in hdf5_module.DATASET_KEYS}:
            raise AssertionError(
                f"validator read payload before rejecting vlen /stage: "
                f"{dataset.name} {selection!r}"
            )
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_stage_payload_read)

    report = validate_episode(path)

    assert "stage.dtype" in {issue.code for issue in report.errors}


@pytest.mark.parametrize("key", hdf5_module.DATASET_KEYS)
def test_validator_rejects_oversized_storage_chunks_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    path = tmp_path / f"oversized-chunk-{key}.h5"
    _write_valid_episode(path, steps=1)

    if key.startswith("observation.images"):
        shape = (1, 128, 128, 3)
        chunks = (512, 128, 128, 3)
        dtype = np.dtype(np.uint8)
    elif key in {"observation.state", "action"}:
        shape = (1, 8)
        chunks = (1_000_000, 8)
        dtype = np.dtype(np.float32)
    elif key == "timestamp":
        shape = (1,)
        chunks = (3_000_000,)
        dtype = np.dtype(np.float64)
    else:
        shape = (1,)
        chunks = (300_000,)
        dtype = h5py.string_dtype(encoding="utf-8", length=64)

    with h5py.File(path, "r+") as handle:
        del handle[key]
        handle.create_dataset(
            key,
            shape=shape,
            maxshape=(None, *shape[1:]),
            chunks=chunks,
            dtype=dtype,
        )

    real_getitem = h5py.Dataset.__getitem__

    def reject_attacked_payload_read(
        dataset: h5py.Dataset,
        selection: object,
    ) -> object:
        if dataset.name in {f"/{name}" for name in hdf5_module.DATASET_KEYS}:
            raise AssertionError(
                f"validator read payload before rejecting oversized chunk "
                f"{dataset.name} {selection!r}"
            )
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_attacked_payload_read)

    report = validate_episode(path)

    assert f"{key}.chunk_bytes" in {issue.code for issue in report.errors}


def test_validator_reads_long_timestamps_and_stages_in_fixed_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "long-audit-columns.h5"
    _write_valid_episode(path, steps=1)
    num_steps = 100_000

    with h5py.File(path, "r+") as handle:
        for key in ("observation.images.front", "observation.images.wrist"):
            del handle[key]
            handle.create_dataset(
                key,
                shape=(num_steps, 128, 128, 3),
                maxshape=(None, 128, 128, 3),
                chunks=(1, 128, 128, 3),
                dtype=np.uint8,
                compression="lzf",
                shuffle=True,
            )
        for key in ("observation.state", "action"):
            del handle[key]
            handle.create_dataset(
                key,
                shape=(num_steps, 8),
                maxshape=(None, 8),
                chunks=(256, 8),
                dtype=np.float32,
            )
        del handle["timestamp"]
        handle.create_dataset(
            "timestamp",
            data=np.arange(num_steps, dtype=np.float64) * 0.05,
            chunks=(256,),
        )
        del handle["stage"]
        handle.create_dataset(
            "stage",
            shape=(num_steps,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8", length=64),
            chunks=(256,),
            fillvalue="pregrasp",
        )
        handle.attrs["num_steps"] = np.int64(num_steps)
        handle.attrs["success"] = np.bool_(False)
        handle.attrs["failure_stage"] = "pregrasp"

    reads: dict[str, list[object]] = {"/timestamp": [], "/stage": []}
    real_getitem = h5py.Dataset.__getitem__

    def record_audit_column_read(
        dataset: h5py.Dataset,
        selection: object,
    ) -> object:
        if dataset.name in reads:
            reads[dataset.name].append(selection)
        return real_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_audit_column_read)

    report = validate_episode(path)

    assert "timestamp.non_finite" not in {issue.code for issue in report.errors}
    assert "timestamp.not_strictly_increasing" not in {
        issue.code for issue in report.errors
    }
    assert "timestamp.control_dt" not in {issue.code for issue in report.errors}
    assert "stage.empty" not in {issue.code for issue in report.errors}
    for dataset_name, selections in reads.items():
        assert len(selections) > 1, dataset_name
        for selection in selections:
            assert isinstance(selection, slice), (dataset_name, selection)
            start = 0 if selection.start is None else selection.start
            stop = num_steps if selection.stop is None else selection.stop
            assert stop - start <= 256, (dataset_name, selection)


def test_validator_rejects_string_control_period(tmp_path: Path) -> None:
    path = tmp_path / "string-control-dt.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle.attrs["control_dt_s"] = "0.05"

    report = validate_episode(path)

    assert "metadata.control_dt_s.type" in {issue.code for issue in report.errors}


@pytest.mark.parametrize("attribute", ["action_min", "action_max"])
def test_validator_rejects_string_action_bounds(
    tmp_path: Path,
    attribute: str,
) -> None:
    path = tmp_path / f"string-{attribute}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        values = np.asarray(handle.attrs[attribute], dtype=np.float64)
        handle.attrs[attribute] = np.asarray(
            [str(value).encode("ascii") for value in values],
            dtype="S32",
        )

    report = validate_episode(path)

    assert f"metadata.{attribute}.type" in {issue.code for issue in report.errors}


def test_validator_rejects_tampered_bounds_and_checks_actions_against_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tampered-bounds.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        action_min = np.asarray(handle.attrs["action_min"], dtype=np.float64)
        action_max = np.asarray(handle.attrs["action_max"], dtype=np.float64)
        action_min[0] -= 0.1
        action_max[7] = 2.0
        handle.attrs["action_min"] = action_min
        handle.attrs["action_max"] = action_max
        handle["action"][0, 7] = np.float32(1.5)

    report = validate_episode(path)
    codes = {issue.code for issue in report.errors}

    assert "metadata.action_bounds.contract" in codes
    assert "action.out_of_bounds" in codes


@pytest.mark.parametrize("gripper", [-1e-7, 1.0000002])
def test_validator_strictly_checks_normalized_gripper_action(
    tmp_path: Path,
    gripper: float,
) -> None:
    path = tmp_path / f"validator-action-{gripper}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle["action"][0, 7] = np.float32(gripper)

    report = validate_episode(path)

    assert "action.out_of_bounds" in {issue.code for issue in report.errors}


def test_writer_rejects_bounds_that_differ_from_environment_contract(
    tmp_path: Path,
) -> None:
    lower, upper = _action_bounds()
    upper[0] += 0.1

    with pytest.raises(ValueError, match="environment contract"):
        with HDF5EpisodeWriter(
            tmp_path / "wrong-action-bounds.h5",
            seed=0,
            action_bounds=(lower, upper),
        ):
            pass

    assert not list(tmp_path.glob(".*.partial-*"))


@pytest.mark.parametrize("gripper", [-1e-7, 1.0000002])
def test_writer_rejects_out_of_range_gripper_state(
    tmp_path: Path,
    gripper: float,
) -> None:
    path = tmp_path / f"writer-state-{gripper}.h5"
    observation = _observation()
    observation["observation.state"][7] = gripper

    with HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds()) as writer:
        with pytest.raises(ValueError, match=r"gripper.*\[0, 1\]"):
            writer.append(
                observation,
                _valid_action(),
                timestamp=0.0,
                stage="pregrasp",
            )


@pytest.mark.parametrize("gripper", [-1e-7, 1.0000002])
def test_validator_rejects_out_of_range_gripper_state(
    tmp_path: Path,
    gripper: float,
) -> None:
    path = tmp_path / f"validator-state-{gripper}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle["observation.state"][0, 7] = np.float32(gripper)

    report = validate_episode(path)

    assert "observation.state.gripper_out_of_bounds" in {
        issue.code for issue in report.errors
    }


@pytest.mark.parametrize(
    ("attribute", "value", "code"),
    [
        ("time_alignment", "post_action", "metadata.time_alignment.value"),
        ("action_semantics", "relative_joint_delta", "metadata.action_semantics.value"),
        ("control_dt_s", np.float64(0.1), "metadata.control_dt_s.value"),
        ("control_dt_s", "0.05", "metadata.control_dt_s.type"),
    ],
)
def test_reader_runs_full_validation_before_opening_episode(
    tmp_path: Path,
    attribute: str,
    value: object,
    code: str,
) -> None:
    path = tmp_path / f"reader-invalid-{attribute}-{code}.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        handle.attrs[attribute] = value

    with pytest.raises(DatasetValidationError) as error:
        HDF5EpisodeReader(path)

    assert code in {issue.code for issue in error.value.report.errors}


def test_external_link_payload_changes_never_pass_episode_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external-link.h5"
    payload_path = tmp_path / "external-payload.h5"
    _write_valid_episode(path)
    actions = np.stack([_valid_action(), _valid_action()]).astype(np.float32)

    with h5py.File(payload_path, "w") as payload:
        payload.create_dataset("action", data=actions)
    with h5py.File(path, "r+") as handle:
        del handle["action"]
        handle["action"] = h5py.ExternalLink(str(payload_path), "/action")

    episode_digest = hashlib.sha256(path.read_bytes()).digest()
    first_report = validate_episode(path)

    assert not first_report.valid
    assert "action.link" in {issue.code for issue in first_report.errors}
    with pytest.raises(DatasetValidationError):
        HDF5EpisodeReader(path)

    with h5py.File(payload_path, "r+") as payload:
        payload["action"][0, 0] = np.float32(0.25)

    assert hashlib.sha256(path.read_bytes()).digest() == episode_digest
    with h5py.File(path, "r") as handle:
        assert handle["action"][0, 0] == pytest.approx(0.25)
    second_report = validate_episode(path)
    assert not second_report.valid
    assert "action.link" in {issue.code for issue in second_report.errors}


@pytest.mark.parametrize(
    ("source_key", "alias_key"),
    [
        (source_key, alias_key)
        for index, source_key in enumerate(hdf5_module.DATASET_KEYS)
        for alias_key in hdf5_module.DATASET_KEYS[index + 1 :]
    ],
)
def test_validator_and_reader_reject_aliased_schema_dataset_objects(
    tmp_path: Path,
    source_key: str,
    alias_key: str,
) -> None:
    path = tmp_path / "aliased-schema-object.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        del handle[alias_key]
        handle[alias_key] = handle[source_key]
        assert (
            h5py.h5o.get_info(handle[source_key].id).addr
            == h5py.h5o.get_info(handle[alias_key].id).addr
        )

    report = validate_episode(path)

    assert "dataset.object_alias" in {issue.code for issue in report.errors}
    with pytest.raises(DatasetValidationError) as caught:
        HDF5EpisodeReader(path)
    assert "dataset.object_alias" in {
        issue.code for issue in caught.value.report.errors
    }


def test_anonymous_reference_graph_is_rejected_without_traversal_or_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "anonymous-reference-graph.h5"
    payload_path = tmp_path / "privileged-payload.h5"
    _write_valid_episode(path)
    with h5py.File(payload_path, "w") as payload:
        payload.create_dataset("privileged", data=np.array([0.4], dtype=np.float32))

    with h5py.File(path, "r+") as handle:
        hidden = handle.create_group(None)
        hidden["self"] = hidden
        hidden["payload"] = h5py.ExternalLink(str(payload_path), "/privileged")
        handle["action"].attrs["opaque_ref"] = hidden.ref
        assert set(handle) == {
            "action",
            "observation.images.front",
            "observation.images.wrist",
            "observation.state",
            "stage",
            "timestamp",
        }

    episode_digest = hashlib.sha256(path.read_bytes()).digest()
    first_report = validate_episode(path)
    assert "action.attribute.unexpected" in {
        issue.code for issue in first_report.errors
    }

    with h5py.File(payload_path, "r+") as payload:
        payload["privileged"][0] = np.float32(0.9)
    assert hashlib.sha256(path.read_bytes()).digest() == episode_digest
    with h5py.File(path, "r") as handle:
        reference = handle["action"].attrs["opaque_ref"]
        hidden = handle[reference]
        assert hidden["payload"][0] == pytest.approx(0.9)
    second_report = validate_episode(path)
    assert "action.attribute.unexpected" in {
        issue.code for issue in second_report.errors
    }

    payload_path.unlink()
    broken_report = validate_episode(path)
    assert "action.attribute.unexpected" in {
        issue.code for issue in broken_report.errors
    }

    opened: list[h5py.File] = []
    real_file = hdf5_module.h5py.File

    def tracking_file(*args: object, **kwargs: object) -> h5py.File:
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(hdf5_module.h5py, "File", tracking_file)
    with pytest.raises(DatasetValidationError) as caught:
        HDF5EpisodeReader(path)

    assert "action.attribute.unexpected" in {
        issue.code for issue in caught.value.report.errors
    }
    assert len(opened) == 1
    assert not opened[0].id.valid


def test_validator_rejects_soft_linked_schema_dataset(tmp_path: Path) -> None:
    path = tmp_path / "soft-link.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        del handle["action"]
        handle["action"] = h5py.SoftLink("/observation.state")

    report = validate_episode(path)

    assert "action.link" in {issue.code for issue in report.errors}


def test_validator_reports_broken_external_link_without_raising(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken-external-link.h5"
    _write_valid_episode(path)

    with h5py.File(path, "r+") as handle:
        del handle["action"]
        handle["action"] = h5py.ExternalLink(
            str(tmp_path / "missing-payload.h5"), "/action"
        )

    report = validate_episode(path)

    assert "action.link" in {issue.code for issue in report.errors}


def test_validator_rejects_virtual_dataset_storage(tmp_path: Path) -> None:
    path = tmp_path / "virtual-dataset.h5"
    payload_path = tmp_path / "virtual-payload.h5"
    _write_valid_episode(path)
    actions = np.stack([_valid_action(), _valid_action()]).astype(np.float32)

    with h5py.File(payload_path, "w") as payload:
        payload.create_dataset("action", data=actions)
    layout = h5py.VirtualLayout(shape=actions.shape, dtype=np.float32)
    layout[:] = h5py.VirtualSource(str(payload_path), "/action", shape=actions.shape)
    with h5py.File(path, "r+") as handle:
        del handle["action"]
        handle.create_virtual_dataset("action", layout)

    report = validate_episode(path)

    assert "action.storage.virtual" in {issue.code for issue in report.errors}


def test_validator_rejects_external_raw_dataset_storage(tmp_path: Path) -> None:
    path = tmp_path / "external-raw.h5"
    raw_path = tmp_path / "action.raw"
    _write_valid_episode(path)
    actions = np.stack([_valid_action(), _valid_action()]).astype(np.float32)

    with h5py.File(path, "r+") as handle:
        del handle["action"]
        dataset = handle.create_dataset(
            "action",
            shape=actions.shape,
            dtype=np.float32,
            external=[(str(raw_path), 0, h5py.h5f.UNLIMITED)],
        )
        dataset[:] = actions

    report = validate_episode(path)

    assert "action.storage.external" in {issue.code for issue in report.errors}


def test_reader_validates_and_reads_the_same_single_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reader-target.h5"
    replacement = tmp_path / "reader-replacement.h5"
    _write_valid_episode(path)
    _write_valid_episode(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle.attrs["time_alignment"] = "post_action"

    real_file = hdf5_module.h5py.File
    read_opens = 0

    def replace_on_second_read(
        file_name: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> h5py.File:
        nonlocal read_opens
        if Path(file_name) == path and mode == "r":
            read_opens += 1
            if read_opens == 2:
                replacement.replace(path)
        return real_file(file_name, mode, *args, **kwargs)

    monkeypatch.setattr(hdf5_module.h5py, "File", replace_on_second_read)

    with HDF5EpisodeReader(path) as reader:
        assert reader.metadata.time_alignment == "pre_action"

    assert read_opens == 1


def test_reader_closes_its_only_handle_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reader-invalid.h5"
    _write_valid_episode(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["time_alignment"] = "post_action"

    opened: list[h5py.File] = []
    real_file = hdf5_module.h5py.File

    def tracking_file(*args: object, **kwargs: object) -> h5py.File:
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(hdf5_module.h5py, "File", tracking_file)

    with pytest.raises(DatasetValidationError):
        HDF5EpisodeReader(path)

    assert len(opened) == 1
    assert not opened[0].id.valid


def test_writer_closes_partial_handle_before_cleanup_on_init_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[h5py.File] = []
    real_file = hdf5_module.h5py.File

    def tracking_file(*args: object, **kwargs: object) -> h5py.File:
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(hdf5_module.h5py, "File", tracking_file)

    with pytest.raises(ValueError, match="minimum"):
        HDF5EpisodeWriter(
            tmp_path / "bad-init.h5",
            seed=0,
            action_bounds=(np.zeros(8), np.zeros(8)),
        )

    assert len(opened) == 1
    assert not opened[0].id.valid
    assert not list(tmp_path.glob(".*.partial-*"))


def test_validator_returns_locatable_errors_for_corrupt_data(tmp_path: Path) -> None:
    value_path = tmp_path / "corrupt-values.h5"
    _write_valid_episode(value_path)

    with h5py.File(value_path, "r+") as handle:
        del handle.attrs["seed"]
        handle["observation.state"][0, 0] = np.nan
        handle["action"][0, 7] = 1.5
        handle["timestamp"][1] = handle["timestamp"][0]

    value_report = validate_episode(value_path)
    value_codes = {issue.code for issue in value_report.errors}

    assert not value_report.valid
    assert {
        "metadata.seed.missing",
        "observation.state.non_finite",
        "action.out_of_bounds",
        "timestamp.not_strictly_increasing",
    } <= value_codes

    structure_path = tmp_path / "corrupt-structure.h5"
    _write_valid_episode(structure_path)
    with h5py.File(structure_path, "r+") as handle:
        handle["stage"].resize(1, axis=0)
        front = handle["observation.images.front"][:]
        del handle["observation.images.front"]
        handle.create_dataset("observation.images.front", data=front.astype(np.int16))

    structure_report = validate_episode(structure_path)
    structure_codes = {issue.code for issue in structure_report.errors}

    assert not structure_report.valid
    assert {
        "stage.length",
        "observation.images.front.dtype",
    } <= structure_codes


@pytest.mark.parametrize(
    ("success", "failure_stage", "message"),
    [
        (True, "pregrasp", "successful episode"),
        (False, None, "failed episode"),
        (False, "", "failed episode"),
    ],
)
def test_finalize_rejects_inconsistent_episode_metadata(
    tmp_path: Path,
    success: bool,
    failure_stage: str | None,
    message: str,
) -> None:
    path = tmp_path / f"bad-{success}-{failure_stage!s}.h5"

    with HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds()) as writer:
        writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
        with pytest.raises(ValueError, match=message):
            writer.finalize(success=success, failure_stage=failure_stage)

    assert not path.exists()


def test_writer_rejects_invalid_transition_before_it_reaches_disk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.h5"

    with HDF5EpisodeWriter(path, seed=0, action_bounds=_action_bounds()) as writer:
        privileged_observation = _observation()
        privileged_observation["privileged.cube_position"] = np.zeros(3)
        with pytest.raises(ValueError, match=r"extra=.*privileged\.cube_position"):
            writer.append(
                privileged_observation,
                _valid_action(),
                timestamp=0.0,
                stage="pregrasp",
            )

        bad_state = _observation()
        bad_state["observation.state"][0] = np.inf
        with pytest.raises(ValueError, match="observation.state.*finite"):
            writer.append(bad_state, _valid_action(), timestamp=0.0, stage="pregrasp")

        with pytest.raises(ValueError, match="action.*bounds"):
            writer.append(
                _observation(),
                _valid_action(gripper=1.1),
                timestamp=0.0,
                stage="pregrasp",
            )

        writer.append(_observation(), _valid_action(), timestamp=0.0, stage="pregrasp")
        with pytest.raises(ValueError, match="strictly increasing"):
            writer.append(
                _observation(1),
                _valid_action(),
                timestamp=0.0,
                stage="pregrasp",
            )
        writer.finalize(success=False, failure_stage="pregrasp")

    assert validate_episode(path).valid
