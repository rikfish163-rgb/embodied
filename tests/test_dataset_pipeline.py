from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import h5py

import data.collection as collection_module
import data.manifest as manifest_module
import data.replay as replay_module
import data.reporting as reporting_module
from data import HDF5EpisodeReader
from data.collection import (
    CollectionConfig,
    CollectionExhaustedError,
    FormalCollectionError,
    collect_split,
    reconcile_indeterminate_run,
)
from data.hdf5 import EpisodePublicationError, HDF5EpisodeWriter
from data.lerobot_adapter import (
    LEROBOT_VERSION,
    LeRobotVersionError,
    iter_lerobot_episode,
    lerobot_feature_spec,
    map_lerobot_frame,
    require_lerobot_version,
)
from data.manifest import (
    AtomicPublicationError,
    LedgerIntegrityError,
    append_jsonl_fsync,
    atomic_write_json_no_clobber,
    initialize_jsonl_no_clobber,
    load_collection_manifest,
    maximum_episode_steps,
    validate_collection_manifest,
    validate_manifest_pair,
    validate_split_seed,
)
from data.replay import (
    ReplayConfig,
    ReplayProvenanceError,
    replay_collection,
    select_replay_entries,
    validate_replay_summary,
)
from data.reporting import ReportConfig, generate_collection_report
from env.pick_place import TaskConfig


_ACTION_MIN = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0],
    dtype=np.float64,
)
_ACTION_MAX = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0],
    dtype=np.float64,
)


class _Site:
    def __init__(self, name: str):
        self.name = name
        self.rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


class _CollectionModel:
    def __init__(self) -> None:
        self.actuator_ctrlrange = np.column_stack([_ACTION_MIN, _ACTION_MAX])
        self.nsite = 2
        self._sites = (_Site("tcp"), _Site("flange"))

    def site(self, index_or_name: int | str) -> _Site:
        if isinstance(index_or_name, str):
            return next(site for site in self._sites if site.name == index_or_name)
        return self._sites[index_or_name]


class _CollectionEnv:
    def __init__(self) -> None:
        self.cfg = TaskConfig()
        self.model = _CollectionModel()
        self.data = SimpleNamespace(time=0.0)
        self.frame_value = 0
        self.closed = False

    def observe(self) -> dict[str, np.ndarray]:
        value = self.frame_value
        return {
            "observation.images.front": np.full((128, 128, 3), value, dtype=np.uint8),
            "observation.images.wrist": np.full(
                (128, 128, 3), value + 1, dtype=np.uint8
            ),
            "observation.state": np.array(
                [0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785, 0.5],
                dtype=np.float64,
            ),
        }

    def close(self) -> None:
        self.closed = True


def _action(seed: int, step: int) -> np.ndarray:
    return np.array(
        [
            (seed % 7) * 0.01,
            0.0,
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            min(0.2 + step * 0.1, 1.0),
        ],
        dtype=np.float64,
    )


def _runner(
    *,
    outcomes: dict[int, bool] | None = None,
    lengths: dict[int, int] | None = None,
    interrupt_seed: int | None = None,
):
    configured_outcomes = outcomes or {}
    configured_lengths = lengths or {}

    def run(
        env: _CollectionEnv,
        *,
        seed: int,
        config: object,
        step_callback: Any,
    ) -> SimpleNamespace:
        del config
        for step in range(configured_lengths.get(seed, 2)):
            env.frame_value = (seed + step) % 200
            env.data.time = step * 0.05
            step_callback(env, "pregrasp", _action(seed, step))
            if interrupt_seed == seed:
                raise RuntimeError("injected collection interruption")
        success = configured_outcomes.get(seed, True)
        return SimpleNamespace(
            seed=seed,
            success=success,
            failure_stage=None if success else "pregrasp",
            control_steps=configured_lengths.get(seed, 2),
        )

    return run


def _git_state(*, clean: bool = True) -> dict[str, Any]:
    return {
        "commit": "1" * 40,
        "tracked_worktree_clean": clean,
        "worktree_clean": clean,
        "source_provenance_clean": clean,
        "provenance_complete": True,
        "worktree_status_sha256": "2" * 64,
        "untracked_file_count": 0,
        "untracked_paths_sha256": "3" * 64,
        "relevant_untracked_files": [],
    }


class _AssetProvenance:
    def __init__(self) -> None:
        self._manifest = {
            "schema_version": "asset-manifest.v1",
            "files": [
                {
                    "path": "menagerie/franka_emika_panda/panda.xml",
                    "sha256": "4" * 64,
                    "size_bytes": 123,
                }
            ],
        }
        self.aggregate_manifest_sha256 = hashlib.sha256(
            json.dumps(
                self._manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def manifest_dict(self) -> dict[str, Any]:
        return self._manifest

    def summary_dict(self) -> dict[str, Any]:
        return {
            "canonical_root": "menagerie/franka_emika_panda",
            "revision": {
                "repository": "https://github.com/google-deepmind/mujoco_menagerie",
                "revision": "5" * 40,
                "vendored_path": "franka_emika_panda",
                "license": "Apache-2.0",
            },
            "file_count": 1,
            "aggregate_manifest_sha256": self.aggregate_manifest_sha256,
        }


def _environment_fingerprint(env: _CollectionEnv) -> dict[str, Any]:
    assert asdict(env.cfg)["debug_viz"] is False
    base = {
        "fingerprint_schema": "mujoco-compiled-model.v1",
        "mjb_sha256": "6" * 64,
        "nq": 16,
        "nv": 15,
        "nu": 8,
        "nbody": 20,
        "ngeom": 30,
        "nsite": 2,
        "policy_sites": [
            {"name": "flange", "rgba": [0.0, 0.0, 0.0, 0.0]},
            {"name": "tcp", "rgba": [0.0, 0.0, 0.0, 0.0]},
        ],
    }
    return {
        **base,
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                base,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def _collect(
    tmp_path: Path,
    *,
    name: str = "run",
    split: str = "train",
    target_successes: int = 2,
    outcomes: dict[int, bool] | None = None,
    lengths: dict[int, int] | None = None,
    interrupt_seed: int | None = None,
    clean_git: bool = True,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / name
    manifest = collect_split(
        CollectionConfig(
            output_dir=root,
            split=split,
            target_successes=target_successes,
            smoke=True,
            diagnostic_allow_dirty=not clean_git,
            project_root=tmp_path,
        ),
        env_factory=_CollectionEnv,
        episode_runner=_runner(
            outcomes=outcomes,
            lengths=lengths,
            interrupt_seed=interrupt_seed,
        ),
        git_state_fn=lambda _: _git_state(clean=clean_git),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )
    return root, manifest


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_ledger(root: Path, attempts: list[dict[str, Any]]) -> None:
    (root / "attempts.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in attempts
        ),
        encoding="utf-8",
    )


def _codes(report: Any) -> set[str]:
    return {error.code for error in report.errors}


def test_collection_is_no_clobber_and_manifest_is_self_validating(
    tmp_path: Path,
) -> None:
    root, manifest = _collect(tmp_path)
    original = (root / "manifest.json").read_bytes()

    assert manifest["success_count"] == 2
    assert manifest["attempt_count"] == 2
    assert manifest["formal"] is False
    assert [item["seed"] for item in manifest["attempts"]] == [0, 1]
    assert validate_collection_manifest(root / "manifest.json").valid

    with pytest.raises(FileExistsError):
        _collect(tmp_path)
    assert (root / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("fault", ["partial_unlink", "directory_fsync"])
def test_atomic_json_postlink_failure_is_typed_and_preserves_target_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    target = tmp_path / "published.json"
    payload = {"schema_version": "test.v1", "ok": True}
    real_unlink = Path.unlink
    real_fsync = manifest_module._fsync_directory

    if fault == "partial_unlink":

        def fail_partial_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path.name.startswith(".published.json.partial-"):
                raise OSError("injected partial unlink failure")
            real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_partial_unlink)
    else:

        def fail_directory_fsync(path: Path) -> None:
            del path
            raise OSError("injected directory fsync failure")

        monkeypatch.setattr(manifest_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(AtomicPublicationError) as captured:
        atomic_write_json_no_clobber(target, payload)

    error = captured.value
    assert error.state == "publication_indeterminate"
    assert error.published is True
    assert error.target_path == target
    assert error.target_matches_source is True
    assert error.target_valid is True
    assert error.target_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    if fault == "partial_unlink":
        assert error.partial_path.exists()
    else:
        assert not error.partial_path.exists()

    monkeypatch.setattr(manifest_module, "_fsync_directory", real_fsync)


def test_atomic_json_rejects_in_place_partial_tampering_before_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "published.json"
    payload = {"schema_version": "test.v1", "ok": True}
    real_link = os.link

    def tamper_then_link(source: Path, destination: Path) -> None:
        source_path = Path(source)
        before = source_path.stat()
        source_path.write_text('{"schema_version":"evil.v1"}\n', encoding="utf-8")
        os.utime(
            source_path,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        real_link(source, destination)

    monkeypatch.setattr(os, "link", tamper_then_link)

    with pytest.raises(AtomicPublicationError) as captured:
        atomic_write_json_no_clobber(target, payload)

    assert captured.value.published is True
    assert captured.value.target_matches_source is False
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "schema_version": "evil.v1"
    }


def test_atomic_json_rechecks_target_digest_after_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "published.json"
    payload = {"schema_version": "test.v1", "ok": True}
    real_fsync = manifest_module._fsync_directory

    def tamper_then_fsync(directory: Path) -> None:
        target.write_text('{"schema_version":"post-link-evil.v1"}\n', encoding="utf-8")
        real_fsync(directory)

    monkeypatch.setattr(manifest_module, "_fsync_directory", tamper_then_fsync)

    with pytest.raises(AtomicPublicationError) as captured:
        atomic_write_json_no_clobber(target, payload)

    assert captured.value.published is True
    assert captured.value.target_matches_source is False
    assert captured.value.target_valid is True


@pytest.mark.parametrize("replacement", ["symlink", "regular", "same_inode"])
def test_jsonl_append_rejects_ledger_replacement_before_writing(
    tmp_path: Path,
    replacement: str,
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    expected_snapshot = initialize_jsonl_no_clobber(ledger)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"outside-before\n")

    if replacement == "symlink":
        ledger.unlink()
        ledger.symlink_to(outside)
    elif replacement == "regular":
        ledger.unlink()
        ledger.write_bytes(b"replacement-before\n")
    else:
        before = ledger.stat()
        ledger.write_bytes(b"same-inode-before\n")
        os.utime(ledger, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(LedgerIntegrityError):
        append_jsonl_fsync(
            ledger,
            {"schema_version": "attempt.v1", "attempt": 1},
            expected_snapshot=expected_snapshot,
        )

    assert outside.read_bytes() == b"outside-before\n"
    if replacement == "symlink":
        assert ledger.is_symlink()
    elif replacement == "regular":
        assert ledger.read_bytes() == b"replacement-before\n"
    else:
        assert ledger.read_bytes() == b"same-inode-before\n"


def test_interrupted_attempt_is_fsynced_to_the_ledger_without_a_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "interrupted"

    with pytest.raises(RuntimeError, match="injected collection interruption"):
        _collect(tmp_path, name="interrupted", target_successes=1, interrupt_seed=0)

    assert not (root / "manifest.json").exists()
    assert not list((root / "episodes").glob("*.h5"))
    records = [
        json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
    ]
    assert records == [
        {
            "attempt_index": 0,
            "error": {
                "message": "injected collection interruption",
                "type": "RuntimeError",
            },
            "failure_stage": None,
            "num_steps": None,
            "path": None,
            "seed": 0,
            "sha256": None,
            "status": "exception",
            "success": None,
        }
    ]


def test_postpublication_audit_exception_is_not_omitted_from_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "postpublication"

    def fail_inspection(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected postpublication audit failure")

    monkeypatch.setattr(collection_module, "inspect_episode", fail_inspection)
    with pytest.raises(RuntimeError, match="postpublication audit failure"):
        _collect(tmp_path, name="postpublication", target_successes=1)

    [record] = [
        json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
    ]
    assert record["status"] == "postpublication_exception"
    assert record["path"] == "episodes/seed_000000.h5"
    assert (root / record["path"]).exists()
    assert record["error"]["type"] == "RuntimeError"
    assert not (root / "manifest.json").exists()


def test_publication_indeterminate_is_durable_and_reconciled_without_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "indeterminate"

    class IndeterminateWriter(HDF5EpisodeWriter):
        def finalize(self, *, success: bool, failure_stage: str | None) -> Path:
            target = super().finalize(
                success=success,
                failure_stage=failure_stage,
            )
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            os.link(target, self._partial_path)
            self._state = "publication_indeterminate"
            raise EpisodePublicationError(
                target_path=target,
                partial_path=self._partial_path,
                target_matches_source=True,
                target_valid=True,
                target_sha256=digest,
            )

    with pytest.raises(EpisodePublicationError):
        collect_split(
            CollectionConfig(
                output_dir=root,
                split="train",
                target_successes=1,
                smoke=True,
                project_root=tmp_path,
            ),
            env_factory=_CollectionEnv,
            episode_runner=_runner(),
            git_state_fn=lambda _: _git_state(),
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
            writer_factory=IndeterminateWriter,
        )

    [record] = [
        json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
    ]
    assert record["status"] == "publication_indeterminate"
    assert record["error"]["state"] == "publication_indeterminate"
    assert record["error"]["target_matches_source"] is True
    assert record["error"]["target_valid"] is True
    assert record["error"]["target_path"] == "episodes/seed_000000.h5"
    partial = root / record["error"]["partial_path"]
    target = root / record["error"]["target_path"]
    assert partial.samefile(target)

    receipt = reconcile_indeterminate_run(root)

    assert receipt["status"] == "verified_publication"
    assert receipt["target_sha256"] == record["sha256"]
    assert receipt["partial_hardlink_removed"] is True
    assert target.exists()
    assert not partial.exists()
    assert not (root / "manifest.json").exists()
    assert (root / "reconciliation.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "seed", "path", "sha256"])
def test_manifest_rejects_any_ledger_manifest_disagreement(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=2)
    ledger = [dict(item) for item in manifest["attempts"]]
    if mutation == "missing":
        ledger.pop()
    elif mutation == "extra":
        ledger.append(dict(ledger[-1]))
    elif mutation == "seed":
        ledger[0]["seed"] = 9
    elif mutation == "path":
        ledger[0]["path"] = "episodes/seed_000009.h5"
    else:
        ledger[0]["sha256"] = "0" * 64
    _write_ledger(root, ledger)

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "ledger.mismatch" in _codes(report)


def test_manifest_rejects_checksum_tampering(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    episode = root / manifest["attempts"][0]["path"]
    with episode.open("ab") as stream:
        stream.write(b"tamper")

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "episode.sha256" in _codes(report)


def test_manifest_validator_cli_returns_status_and_never_clobbers_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    valid_report = tmp_path / "valid-validation.json"

    assert (
        manifest_module.main(
            [
                "--manifest",
                str(root / "manifest.json"),
                "--report",
                str(valid_report),
            ]
        )
        == 0
    )
    assert json.loads(valid_report.read_text(encoding="utf-8"))["valid"] is True
    with pytest.raises(FileExistsError):
        manifest_module.main(
            [
                "--manifest",
                str(root / "manifest.json"),
                "--report",
                str(valid_report),
            ]
        )

    episode = root / manifest["attempts"][0]["path"]
    with episode.open("ab") as stream:
        stream.write(b"tamper")
    invalid_report = tmp_path / "invalid-validation.json"
    assert (
        manifest_module.main(
            [
                "--manifest",
                str(root / "manifest.json"),
                "--report",
                str(invalid_report),
            ]
        )
        == 1
    )
    payload = json.loads(invalid_report.read_text(encoding="utf-8"))
    assert payload["valid"] is False
    assert any(error["code"] == "episode.sha256" for error in payload["errors"])
    capsys.readouterr()


@pytest.mark.parametrize(
    "unsafe_path",
    ["", "/tmp/episode.h5", "../episode.h5", "episodes\\x.h5", "episodes//x.h5"],
)
def test_manifest_rejects_noncanonical_episode_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    manifest["attempts"][0]["path"] = unsafe_path
    manifest["eligible_successes"][0]["path"] = unsafe_path
    _write_manifest(root / "manifest.json", manifest)
    _write_ledger(root, manifest["attempts"])

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "path.canonical" in _codes(report)


def test_manifest_rejects_symlinked_episode_components(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    episode = root / manifest["attempts"][0]["path"]
    real_episode = episode.with_name("real.h5")
    episode.rename(real_episode)
    episode.symlink_to(real_episode.name)

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "path.symlink" in _codes(report)


def test_manifest_rejects_two_paths_to_the_same_inode(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=2)
    first = root / manifest["attempts"][0]["path"]
    second = root / manifest["attempts"][1]["path"]
    second.unlink()
    os.link(first, second)
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    manifest["attempts"][1]["sha256"] = first_sha
    manifest["eligible_successes"][1]["sha256"] = first_sha
    _write_manifest(root / "manifest.json", manifest)
    _write_ledger(root, manifest["attempts"])

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "path.inode_duplicate" in _codes(report)


def test_manifest_rejects_path_replacement_during_hdf5_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    path = root / manifest["attempts"][0]["path"]
    held = path.with_name("held-original.h5")
    replacement = path.with_name("replacement.h5")
    shutil.copy2(path, replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["action"][0, 0] = handle["action"][0, 0] + np.float32(0.01)
    with h5py.File(path, "r") as handle:
        original_action = float(handle["action"][0, 0])
    consumed_actions: list[float] = []
    real_validate = manifest_module._validate_episode_handle

    def replace_while_validating(handle: h5py.File, logical_path: Path):
        path.rename(held)
        replacement.rename(path)
        try:
            report = real_validate(handle, logical_path)
            consumed_actions.append(float(handle["action"][0, 0]))
            return report
        finally:
            path.unlink()
            held.rename(path)

    monkeypatch.setattr(
        manifest_module,
        "_validate_episode_handle",
        replace_while_validating,
    )

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "episode.read" in _codes(report)
    assert consumed_actions == [original_action]


def test_manifest_rejects_same_inode_metadata_change_during_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    path = root / manifest["attempts"][0]["path"]
    real_validate = manifest_module._validate_episode_handle

    def touch_while_validating(handle: h5py.File, logical_path: Path):
        report = real_validate(handle, logical_path)
        before = path.stat()
        os.utime(
            path,
            ns=(before.st_atime_ns, before.st_mtime_ns),
            follow_symlinks=False,
        )
        return report

    monkeypatch.setattr(
        manifest_module,
        "_validate_episode_handle",
        touch_while_validating,
    )

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "episode.read" in _codes(report)


def test_manifest_rejects_unreachable_episode_length_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    unreachable = maximum_episode_steps(manifest["controller"]) + 1
    manifest["attempts"][0]["num_steps"] = unreachable
    manifest["eligible_successes"][0]["num_steps"] = unreachable
    _write_manifest(root / "manifest.json", manifest)
    _write_ledger(root, manifest["attempts"])

    def should_not_open_episode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("episode payload was opened before the length gate")

    monkeypatch.setattr(manifest_module, "inspect_episode", should_not_open_episode)

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "manifest.num_steps_limit" in _codes(report)


def test_manifest_rejects_json_bool_as_integer_and_nan(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    manifest["target_successes"] = True
    _write_manifest(root / "manifest.json", manifest)
    assert "manifest.type" in _codes(
        validate_collection_manifest(root / "manifest.json")
    )

    manifest["target_successes"] = 1
    manifest["split_protocol"]["candidate_seed_min"] = False
    _write_manifest(root / "manifest.json", manifest)
    assert "manifest.type" in _codes(
        validate_collection_manifest(root / "manifest.json")
    )

    manifest["split_protocol"]["candidate_seed_min"] = 0
    manifest["environment"]["compiled_model"]["nq"] = float("nan")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "manifest.json" in _codes(
        validate_collection_manifest(root / "manifest.json")
    )


def test_manifest_asset_provenance_is_closed_world_and_strictly_typed(
    tmp_path: Path,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    manifest["assets"]["summary"]["unexpected"] = "not allowed"
    manifest["assets"]["content_manifest"]["files"][0]["size_bytes"] = True
    _write_manifest(root / "manifest.json", manifest)

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert {"manifest.keys", "manifest.type"}.issubset(_codes(report))


def test_seed_contract_rejects_cross_split_and_evaluation_namespaces() -> None:
    validate_split_seed("train", 0)
    validate_split_seed("train", 999)
    validate_split_seed("validation", 1000)
    validate_split_seed("validation", 9999)

    for split, seed in (
        ("train", 1000),
        ("validation", 999),
        ("train", 10000),
        ("validation", 10050),
    ):
        with pytest.raises(ValueError):
            validate_split_seed(split, seed)
    with pytest.raises((TypeError, ValueError)):
        validate_split_seed("train", True)


def test_cross_manifest_validation_requires_train_validation_disjointness(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="train",
        split="train",
        target_successes=1,
    )
    validation_root, _ = _collect(
        tmp_path,
        name="validation",
        split="validation",
        target_successes=1,
    )

    report = validate_manifest_pair(
        train_root / "manifest.json",
        validation_root / "manifest.json",
    )

    assert report.valid, report.format_errors()


def test_replay_selection_is_deterministic_unique_and_exact(tmp_path: Path) -> None:
    root, _ = _collect(tmp_path, target_successes=5)
    manifest = load_collection_manifest(root / "manifest.json")

    selected_once = select_replay_entries(
        manifest,
        count=3,
        selection_seed=20260824,
    )
    selected_twice = select_replay_entries(
        manifest,
        count=3,
        selection_seed=20260824,
    )

    assert selected_once == selected_twice
    assert len(selected_once) == 3
    assert len({item["seed"] for item in selected_once}) == 3
    with pytest.raises(ValueError):
        select_replay_entries(manifest, count=6, selection_seed=0)


class _ReplayEnv:
    def __init__(self) -> None:
        self.cfg = TaskConfig()
        self.model = _CollectionModel()
        self.reset_seeds: list[int] = []
        self.trial_actions: list[list[np.ndarray]] = []
        self._actions: list[np.ndarray] | None = None

    def reset(self, rng: np.random.Generator) -> dict[str, object]:
        seed = int(rng.bit_generator._seed_seq.entropy)
        self.reset_seeds.append(seed)
        self._actions = []
        self.trial_actions.append(self._actions)
        return {}

    def step(self, action: np.ndarray) -> dict[str, object]:
        assert self._actions is not None
        self._actions.append(np.asarray(action).copy())
        return {"success": False}

    def success(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_replay_resets_the_source_seed_and_steps_each_stored_action_once(
    tmp_path: Path,
) -> None:
    root, manifest = _collect(
        tmp_path,
        target_successes=2,
        lengths={0: 2, 1: 3},
    )
    replay_env = _ReplayEnv()
    replay_root = tmp_path / "replay"

    summary = replay_collection(
        ReplayConfig(
            manifest_path=root / "manifest.json",
            output_dir=replay_root,
            selection_seed=9,
            count=2,
            smoke=True,
        ),
        env_factory=lambda: replay_env,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )

    selected_seeds = summary["selected_seeds"]
    assert replay_env.reset_seeds == selected_seeds
    by_seed = {item["seed"]: item for item in manifest["eligible_successes"]}
    for seed, actions in zip(selected_seeds, replay_env.trial_actions, strict=True):
        with HDF5EpisodeReader(root / by_seed[seed]["path"]) as reader:
            expected = [transition.action for transition in reader]
        assert len(actions) == len(expected)
        for actual, wanted in zip(actions, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)
    assert summary["success_count"] == 2
    assert summary["gate"]["passed"] is True
    assert summary["formal"] is False
    assert len((replay_root / "trials.jsonl").read_text().splitlines()) == 2
    replay_validation = validate_replay_summary(
        replay_root / "summary.json",
        collection_manifest_path=root / "manifest.json",
    )
    assert replay_validation.valid, replay_validation.format_errors()


def test_formal_replay_rejects_noncanonical_selection_seed_before_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "noncanonical-replay"

    with pytest.raises(ValueError, match="selection seed"):
        replay_collection(
            ReplayConfig(
                manifest_path=tmp_path / "unused-manifest.json",
                output_dir=output,
                selection_seed=1,
            )
        )

    assert not output.exists()


def test_formal_replay_provenance_mismatch_fails_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "formal-source.json"
    source_path.write_text("{}\n", encoding="utf-8")
    source_env = manifest_module.environment_payload(
        _CollectionEnv(),
        _environment_fingerprint(_CollectionEnv()),
    )
    source_manifest = {
        "formal": True,
        "split": "train",
        "git": _git_state(),
        "assets": manifest_module.assets_payload(_AssetProvenance()),
        "environment": source_env,
        "eligible_successes": [
            {
                "seed": seed,
                "path": f"episodes/seed_{seed:06d}.h5",
                "num_steps": 1,
                "sha256": "7" * 64,
            }
            for seed in range(20)
        ],
    }
    fake_report = SimpleNamespace(
        manifest=source_manifest,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        raise_for_errors=lambda: None,
    )
    monkeypatch.setattr(
        replay_module,
        "validate_collection_manifest",
        lambda _path: fake_report,
    )
    changed_git = _git_state()
    changed_git["commit"] = "8" * 40
    output = tmp_path / "formal-replay"

    with pytest.raises(ReplayProvenanceError, match="Git commit"):
        replay_collection(
            ReplayConfig(manifest_path=source_path, output_dir=output),
            env_factory=_CollectionEnv,
            git_state_fn=lambda _: changed_git,
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
        )

    assert not output.exists()


def test_replay_validator_rebuilds_selection_and_report_rejects_forged_linkage(
    tmp_path: Path,
) -> None:
    root, _ = _collect(tmp_path, target_successes=2)
    replay_root = tmp_path / "verified-replay"
    replay_collection(
        ReplayConfig(
            manifest_path=root / "manifest.json",
            output_dir=replay_root,
            selection_seed=9,
            count=2,
            smoke=True,
        ),
        env_factory=_ReplayEnv,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )
    summary_path = replay_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trials_path = replay_root / summary["trials_path"]
    original_trials = trials_path.read_bytes()
    trials = [json.loads(line) for line in original_trials.decode().splitlines()]

    extra_trial = dict(trials[-1])
    extra_trial["trial_index"] = len(trials)
    extra_trial["seed"] = 999
    trials_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n" for item in [*trials, extra_trial]
        ),
        encoding="utf-8",
    )
    summary["trials_sha256"] = hashlib.sha256(trials_path.read_bytes()).hexdigest()
    _write_manifest(summary_path, summary)
    extra_validation = validate_replay_summary(
        summary_path,
        collection_manifest_path=root / "manifest.json",
    )
    assert not extra_validation.valid
    assert "replay.trials" in _codes(extra_validation)

    trials_path.write_bytes(original_trials)
    summary["trials_sha256"] = hashlib.sha256(original_trials).hexdigest()
    trials[0]["seed"] = 999
    trials_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in trials),
        encoding="utf-8",
    )
    summary["trials_sha256"] = hashlib.sha256(trials_path.read_bytes()).hexdigest()
    _write_manifest(summary_path, summary)
    ledger_validation = validate_replay_summary(
        summary_path,
        collection_manifest_path=root / "manifest.json",
    )
    assert not ledger_validation.valid
    assert "replay.trials" in _codes(ledger_validation)

    trials_path.write_bytes(original_trials)
    summary["trials_sha256"] = hashlib.sha256(original_trials).hexdigest()
    summary["selected_seeds"] = list(reversed(summary["selected_seeds"]))
    _write_manifest(summary_path, summary)

    validation = validate_replay_summary(
        summary_path,
        collection_manifest_path=root / "manifest.json",
    )
    assert not validation.valid
    assert "replay.selection" in _codes(validation)

    report = generate_collection_report(
        ReportConfig(
            manifest_path=root / "manifest.json",
            output_dir=tmp_path / "report-with-forged-replay",
            replay_summary_path=summary_path,
            manual_review_count=2,
            smoke=True,
        )
    )
    assert report["replay_linkage"]["provided"] is True
    assert report["replay_linkage"]["linked"] is False


@pytest.mark.parametrize("field", ["trial_count", "success_count"])
def test_replay_validator_rejects_wrong_count_types_without_crashing(
    tmp_path: Path,
    field: str,
) -> None:
    root, _ = _collect(tmp_path, target_successes=1)
    replay_root = tmp_path / f"typed-replay-{field}"
    replay_collection(
        ReplayConfig(
            manifest_path=root / "manifest.json",
            output_dir=replay_root,
            count=1,
            smoke=True,
        ),
        env_factory=_ReplayEnv,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )
    summary_path = replay_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = "not-an-integer"
    _write_manifest(summary_path, summary)

    validation = validate_replay_summary(
        summary_path,
        collection_manifest_path=root / "manifest.json",
    )

    assert not validation.valid
    assert "replay.schema" in _codes(validation)


def test_replay_validator_rejects_malformed_runtime_provenance(
    tmp_path: Path,
) -> None:
    root, _ = _collect(tmp_path, target_successes=1)
    replay_root = tmp_path / "malformed-provenance-replay"
    replay_collection(
        ReplayConfig(
            manifest_path=root / "manifest.json",
            output_dir=replay_root,
            count=1,
            smoke=True,
        ),
        env_factory=_ReplayEnv,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )
    summary_path = replay_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["replay_provenance"]["git"] = {}
    _write_manifest(summary_path, summary)

    validation = validate_replay_summary(
        summary_path,
        collection_manifest_path=root / "manifest.json",
    )

    assert not validation.valid
    assert "replay.provenance" in _codes(validation)


def test_formal_report_rejects_noncanonical_manual_selection_seed_before_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "noncanonical-report"

    with pytest.raises(ValueError, match="selection seed"):
        generate_collection_report(
            ReportConfig(
                manifest_path=tmp_path / "unused-manifest.json",
                output_dir=output,
                manual_selection_seed=1,
            )
        )

    assert not output.exists()


def test_report_aggregates_raw_actions_and_leaves_manual_review_pending(
    tmp_path: Path,
) -> None:
    root, _ = _collect(
        tmp_path,
        target_successes=2,
        outcomes={0: False, 1: True, 2: True},
        lengths={0: 1, 1: 2, 2: 3},
    )
    report_root = tmp_path / "report"

    report = generate_collection_report(
        ReportConfig(
            manifest_path=root / "manifest.json",
            output_dir=report_root,
            manual_review_count=3,
            manual_selection_seed=17,
            smoke=True,
        )
    )

    assert report["counts"] == {
        "attempts": 3,
        "failures": 1,
        "successes": 2,
    }
    assert report["episode_lengths"]["count"] == 3
    assert report["episode_lengths"]["min"] == 1
    assert report["episode_lengths"]["max"] == 3
    assert report["actions"]["frame_count"] == 6
    assert len(report["actions"]["dimensions"]) == 8
    assert report["schema_errors"] == []
    assert report["formal"] is False
    assert report["cli_config"] == {
        "manual_review_count": 3,
        "manual_selection_seed": 17,
        "smoke": True,
    }
    assert report["manual_review_complete"] is False
    assert len(report["manual_review_candidates"]) == 3
    assert report["manual_review_candidates"][0]["seed"] == 0

    pending = [
        json.loads(line)
        for line in (report_root / "manual_review.jsonl").read_text().splitlines()
    ]
    assert len(pending) == 3
    assert all(item["verdict"] == "PENDING" for item in pending)
    assert all(item["notes"] == "" for item in pending)
    for item in pending:
        contact_sheet = report_root / item["contact_sheet"]
        assert contact_sheet.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_report_rejects_action_spool_over_budget_before_opening_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    monkeypatch.setattr(reporting_module, "MAX_REPORT_ACTION_BYTES", 1)

    def should_not_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("payload opened before report byte budget")

    monkeypatch.setattr(reporting_module, "open_verified_episode", should_not_open)

    with pytest.raises(ValueError, match="byte budget"):
        reporting_module._aggregate_actions(
            manifest,
            root,
            temporary_parent=tmp_path,
        )


def test_lerobot_adapter_exposes_only_official_frame_features(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    episode = root / manifest["eligible_successes"][0]["path"]

    with HDF5EpisodeReader(episode) as reader:
        frame = map_lerobot_frame(reader[0], task="pick place cube into box")
        all_frames = list(iter_lerobot_episode(reader, task="pick place cube into box"))

    assert set(frame) == {
        "action",
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
        "task",
    }
    assert not {"seed", "stage", "outcome", "failure_stage", "timestamp"}.intersection(
        frame
    )
    assert len(all_frames) == manifest["eligible_successes"][0]["num_steps"]
    assert set(lerobot_feature_spec()) == {
        "action",
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
    }
    assert require_lerobot_version(LEROBOT_VERSION) == LEROBOT_VERSION
    with pytest.raises(LeRobotVersionError):
        require_lerobot_version("0.6.0")


def test_formal_collection_fails_dirty_before_creating_output(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    env_created = False

    def env_factory() -> _CollectionEnv:
        nonlocal env_created
        env_created = True
        return _CollectionEnv()

    with pytest.raises(FormalCollectionError, match="source provenance"):
        collect_split(
            CollectionConfig(
                output_dir=root,
                split="train",
                project_root=tmp_path,
            ),
            env_factory=env_factory,
            episode_runner=_runner(),
            git_state_fn=lambda _: _git_state(clean=False),
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
        )

    assert env_created is False
    assert not root.exists()


def test_formal_collection_refuses_source_change_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        manifest_module._SPLITS["train"],
        "formal_target_successes",
        1,
    )
    git_states = [_git_state(), {**_git_state(), "commit": "9" * 40}]
    root = tmp_path / "source-changed"

    with pytest.raises(FormalCollectionError, match="changed during collection"):
        collect_split(
            CollectionConfig(output_dir=root, split="train", project_root=tmp_path),
            env_factory=_CollectionEnv,
            episode_runner=_runner(),
            git_state_fn=lambda _: git_states.pop(0),
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
        )

    assert (root / "attempts.jsonl").exists()
    assert len(list((root / "episodes").glob("*.h5"))) == 1
    assert not (root / "manifest.json").exists()


def test_collection_namespace_exhaustion_is_bounded_and_auditable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exhausted"
    real_contract = collection_module.split_contract

    def one_seed_contract(split: str) -> dict[str, int | str]:
        contract = dict(real_contract(split))
        contract["candidate_seed_max"] = 0
        return contract

    monkeypatch.setattr(collection_module, "split_contract", one_seed_contract)
    with pytest.raises(CollectionExhaustedError, match="0/1 successes"):
        collect_split(
            CollectionConfig(
                output_dir=root,
                split="train",
                target_successes=1,
                smoke=True,
                project_root=tmp_path,
            ),
            env_factory=_CollectionEnv,
            episode_runner=_runner(outcomes={0: False}),
            git_state_fn=lambda _: _git_state(),
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
        )

    records = [
        json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
    ]
    assert [(record["seed"], record["status"]) for record in records] == [
        (0, "failure")
    ]
    assert len(list((root / "episodes").glob("*.h5"))) == 1
    assert not (root / "manifest.json").exists()


def test_smoke_target_must_be_explicit_and_is_permanently_nonformal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="smoke"):
        collect_split(
            CollectionConfig(
                output_dir=tmp_path / "invalid",
                split="train",
                target_successes=1,
                project_root=tmp_path,
            ),
            env_factory=_CollectionEnv,
            episode_runner=_runner(),
            git_state_fn=lambda _: _git_state(),
            asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
            environment_fingerprint_fn=_environment_fingerprint,
        )

    root, manifest = _collect(tmp_path, name="diagnostic", target_successes=1)
    assert manifest["formal"] is False
    assert load_collection_manifest(root / "manifest.json")["formal"] is False


def test_manifest_cannot_relabel_a_smoke_target_as_formal(tmp_path: Path) -> None:
    root, manifest = _collect(tmp_path, target_successes=1)
    manifest["formal"] = True
    manifest["cli_config"]["smoke"] = False
    _write_manifest(root / "manifest.json", manifest)

    report = validate_collection_manifest(root / "manifest.json")

    assert not report.valid
    assert "manifest.target" in _codes(report)
