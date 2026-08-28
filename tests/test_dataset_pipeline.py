from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
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
from data.gate import (
    M2GateConfig,
    create_m2_gate_receipt,
    validate_m2_gate_receipt,
)
from data.lerobot_adapter import (
    LEROBOT_VERSION,
    LeRobotVersionError,
    iter_lerobot_episode,
    lerobot_feature_spec,
    map_lerobot_frame,
    require_lerobot_version,
)
from data.lineage import (
    LineageRevalidationConfig,
    create_lineage_revalidation_receipt,
    validate_lineage_revalidation_receipt,
)
from data.manual_review import (
    ManualReviewPackConfig,
    create_manual_review_pack,
    validate_manual_review_pack,
)
from data.review_attestation import (
    ATTESTATION_NAMESPACE,
    HumanReviewAttestationConfig,
    create_human_review_attestation_request,
    create_reviewer_registry,
    finalize_human_review_attestation,
    validate_human_review_attestation,
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
    PAIR_SELECTION_ALGORITHM,
    PairReplayConfig,
    ReplayConfig,
    ReplayPlanConfig,
    ReplayProvenanceError,
    create_replay_plan,
    replay_manifest_pair,
    replay_collection,
    select_replay_entries,
    validate_pair_replay_summary,
    validate_replay_plan,
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


def test_pair_replay_plan_uses_sha256_rank_and_rebuilds_exact_selection(
    tmp_path: Path,
) -> None:
    train_root, train_manifest = _collect(
        tmp_path,
        name="pair-train",
        split="train",
        target_successes=4,
    )
    validation_root, validation_manifest = _collect(
        tmp_path,
        name="pair-validation",
        split="validation",
        target_successes=3,
    )
    output = tmp_path / "pair-replay"

    plan = create_replay_plan(
        ReplayPlanConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=output,
            count=4,
            smoke=True,
        ),
        now_fn=lambda: datetime(2026, 8, 27, 0, 0, tzinfo=UTC),
    )

    expected: list[tuple[str, str, int, str, str, int, str]] = []
    for root, manifest in (
        (train_root, train_manifest),
        (validation_root, validation_manifest),
    ):
        digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        manifest_id = f"sha256:{digest}"
        for entry in manifest["eligible_successes"]:
            rank_payload = {
                "algorithm": "sha256-rank-v1",
                "manifest_id": manifest_id,
                "selection_seed": 20260824,
                "split": manifest["split"],
                "seed": entry["seed"],
                "relative_hdf5_path": entry["path"],
                "episode_sha256": entry["sha256"],
            }
            rank = hashlib.sha256(
                json.dumps(
                    rank_payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            expected.append(
                (
                    rank,
                    manifest["split"],
                    entry["seed"],
                    entry["path"],
                    entry["sha256"],
                    entry["num_steps"],
                    manifest_id,
                )
            )
    expected.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    assert plan["schema_version"] == "m2-replay-plan.v1"
    assert plan["selection"] == {
        "algorithm": PAIR_SELECTION_ALGORITHM,
        "seed": 20260824,
        "count": 4,
        "without_replacement": True,
    }
    assert plan["candidate_count"] == 7
    assert [
        (
            trial["rank"],
            trial["split"],
            trial["seed"],
            trial["source_relative_path"],
            trial["source_file_sha256"],
            trial["source_num_steps"],
            trial["manifest_id"],
        )
        for trial in plan["selected_trials"]
    ] == expected[:4]
    assert len({trial["trial_id"] for trial in plan["selected_trials"]}) == 4
    validation = validate_replay_plan(
        output / "plan.json",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert validation.valid, validation.format_errors()


def test_pair_replay_plan_is_closed_world_and_formal_requires_200_plus_40(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="small-train",
        split="train",
        target_successes=2,
    )
    validation_root, _ = _collect(
        tmp_path,
        name="small-validation",
        split="validation",
        target_successes=2,
    )
    formal_output = tmp_path / "invalid-formal-plan"
    with pytest.raises(ValueError, match="200 train and 40 validation"):
        create_replay_plan(
            ReplayPlanConfig(
                train_manifest_path=train_root / "manifest.json",
                validation_manifest_path=validation_root / "manifest.json",
                output_dir=formal_output,
            )
        )
    assert not formal_output.exists()

    smoke_output = tmp_path / "tamper-plan"
    create_replay_plan(
        ReplayPlanConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=smoke_output,
            count=2,
            smoke=True,
        )
    )
    plan_path = smoke_output / "plan.json"
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["selected_trials"][0]["seed"] += 1
    tampered_path = tmp_path / "tampered-plan.json"
    _write_manifest(tampered_path, tampered)
    validation = validate_replay_plan(
        tampered_path,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert not validation.valid
    assert "replay.plan.identity" in _codes(validation)


def test_pair_replay_runner_writes_plan_bound_trial_facts_and_valid_summary(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="runner-train",
        split="train",
        target_successes=3,
        lengths={0: 2, 1: 3, 2: 4},
    )
    validation_root, _ = _collect(
        tmp_path,
        name="runner-validation",
        split="validation",
        target_successes=2,
        lengths={1000: 3, 1001: 2},
    )
    replay_root = tmp_path / "paired-runner"
    plan = create_replay_plan(
        ReplayPlanConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=replay_root,
            count=4,
            smoke=True,
        )
    )
    replay_env = _ReplayEnv()

    summary = replay_manifest_pair(
        PairReplayConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            replay_dir=replay_root,
            smoke=True,
        ),
        env_factory=lambda: replay_env,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
        now_fn=lambda: datetime(2026, 8, 27, 1, 0, tzinfo=UTC),
    )

    trials = [
        json.loads(line)
        for line in (replay_root / "trials.jsonl").read_text().splitlines()
    ]
    assert [trial["trial_id"] for trial in trials] == [
        trial["trial_id"] for trial in plan["selected_trials"]
    ]
    assert [trial["seed"] for trial in trials] == replay_env.reset_seeds
    assert all(trial["schema_version"] == "m2-replay-trial.v1" for trial in trials)
    assert all(trial["expected_steps"] == trial["executed_steps"] for trial in trials)
    assert all(trial["success"] is True for trial in trials)
    for trial in trials:
        source_root = train_root if trial["split"] == "train" else validation_root
        with h5py.File(source_root / trial["source_relative_path"], "r") as episode:
            actions = np.asarray(episode["action"][:], dtype="<f4")
        assert (
            trial["action_dataset_sha256"]
            == hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
        )
    assert summary["identity_reconciliation"] == {
        "planned": 4,
        "observed": 4,
        "missing_trial_ids": [],
        "unexpected_trial_ids": [],
        "duplicate_trial_ids": [],
        "complete": True,
    }
    assert summary["success_count"] == 4
    assert summary["gate"]["passed"] is True
    validation = validate_pair_replay_summary(
        replay_root / "summary.json",
        plan_path=replay_root / "plan.json",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert validation.valid, validation.format_errors()

    tampered_root = tmp_path / "paired-runner-tampered"
    shutil.copytree(replay_root, tampered_root)
    tampered_trials = [
        json.loads(line)
        for line in (tampered_root / "trials.jsonl").read_text().splitlines()
    ]
    tampered_trials[0]["action_dataset_sha256"] = "f" * 64
    (tampered_root / "trials.jsonl").write_text(
        "".join(
            json.dumps(
                trial,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for trial in tampered_trials
        ),
        encoding="utf-8",
    )
    tampered_summary = json.loads(
        (tampered_root / "summary.json").read_text(encoding="utf-8")
    )
    tampered_summary["trials_sha256"] = hashlib.sha256(
        (tampered_root / "trials.jsonl").read_bytes()
    ).hexdigest()
    summary_identity = {
        key: tampered_summary[key]
        for key in sorted(set(tampered_summary) - {"summary_id", "generated_at"})
    }
    tampered_summary["summary_id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                summary_identity,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    _write_manifest(tampered_root / "summary.json", tampered_summary)
    tamper_validation = validate_pair_replay_summary(
        tampered_root / "summary.json",
        plan_path=tampered_root / "plan.json",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert not tamper_validation.valid
    assert "replay.trials.source" in _codes(tamper_validation)


def test_pair_replay_runner_keeps_exception_row_and_never_replaces_trial(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="exception-train",
        split="train",
        target_successes=2,
    )
    validation_root, _ = _collect(
        tmp_path,
        name="exception-validation",
        split="validation",
        target_successes=2,
    )
    replay_root = tmp_path / "exception-replay"
    plan = create_replay_plan(
        ReplayPlanConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=replay_root,
            count=4,
            smoke=True,
        )
    )
    exception_seed = plan["selected_trials"][0]["seed"]

    summary = replay_manifest_pair(
        PairReplayConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            replay_dir=replay_root,
            smoke=True,
        ),
        env_factory=lambda: _ExceptionReplayEnv(exception_seed),
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )

    trials = [
        json.loads(line)
        for line in (replay_root / "trials.jsonl").read_text().splitlines()
    ]
    assert len(trials) == 4
    assert [trial["trial_id"] for trial in trials] == [
        trial["trial_id"] for trial in plan["selected_trials"]
    ]
    failed = trials[0]
    assert failed["seed"] == exception_seed
    assert failed["success"] is False
    assert failed["exception_type"] == "RuntimeError"
    assert failed["executed_steps"] == 0
    assert summary["success_count"] == 3
    assert summary["failed_trial_ids"] == [failed["trial_id"]]
    assert summary["gate"]["passed"] is False
    validation = validate_pair_replay_summary(
        replay_root / "summary.json",
        plan_path=replay_root / "plan.json",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert validation.valid, validation.format_errors()


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


class _ExceptionReplayEnv(_ReplayEnv):
    def __init__(self, exception_seed: int) -> None:
        super().__init__()
        self.exception_seed = exception_seed

    def step(self, action: np.ndarray) -> dict[str, object]:
        if self.reset_seeds[-1] == self.exception_seed and not self._actions:
            raise RuntimeError("injected paired replay failure")
        return super().step(action)


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


def test_pair_manual_review_pack_is_failure_first_and_leaves_judgments_empty(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="review-train",
        split="train",
        target_successes=3,
        outcomes={0: False},
        lengths={0: 5, 1: 2, 2: 3, 3: 4},
    )
    validation_root, _ = _collect(
        tmp_path,
        name="review-validation",
        split="validation",
        target_successes=2,
        lengths={1000: 2, 1001: 5},
    )
    output = tmp_path / "review-pack"

    pack = create_manual_review_pack(
        ManualReviewPackConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=output,
            count=4,
            smoke=True,
        ),
        now_fn=lambda: datetime(2026, 8, 27, 2, 0, tzinfo=UTC),
    )

    assert pack["schema_version"] == "m2-human-review-pack.v1"
    assert pack["status"] == "awaiting_human_review"
    assert pack["candidate_count"] == 6
    assert len(pack["selected_reviews"]) == 4
    assert pack["selected_reviews"][0]["seed"] == 0
    assert pack["selected_reviews"][0]["classification"] == "failure"
    assert all(
        review["classification"] == "anomaly" for review in pack["selected_reviews"][1:]
    )
    templates = [
        json.loads(line)
        for line in (output / "manual-review-template.jsonl").read_text().splitlines()
    ]
    assert len(templates) == 4
    for template, selected in zip(templates, pack["selected_reviews"], strict=True):
        assert template["schema_version"] == "m2-manual-review-trial.v1"
        assert template["manual_review_id"] == selected["manual_review_id"]
        assert template["reviewer_id"] is None
        assert template["review_started_at_utc"] is None
        assert template["review_completed_at_utc"] is None
        assert template["finding"] is None
        assert template["verdict"] is None
        media_path = output / template["media"]["path"]
        assert media_path.is_file()
        assert (
            hashlib.sha256(media_path.read_bytes()).hexdigest()
            == template["media"]["sha256"]
        )
    validation = validate_manual_review_pack(
        output / "review-pack.json",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert validation.valid, validation.format_errors()
    assert validation.complete is False
    assert validation.status == "awaiting_human_review"

    tampered = json.loads(json.dumps(pack))
    tampered["formal"] = True
    tampered["selection"]["seed"] = 7
    tampered["cli_config"]["selection_seed"] = 7
    tampered["cli_config"]["smoke"] = False
    tampered_path = output / "tampered-formal-pack.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_validation = validate_manual_review_pack(
        tampered_path,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert any(
        issue.code == "manual.pack.formal" and issue.location == "/selection/seed"
        for issue in tampered_validation.errors
    )


def test_formal_manual_review_pack_requires_exact_population_before_output(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="formal-review-train",
        split="train",
        target_successes=2,
    )
    validation_root, _ = _collect(
        tmp_path,
        name="formal-review-validation",
        split="validation",
        target_successes=2,
    )
    output = tmp_path / "invalid-formal-review"

    with pytest.raises(ValueError, match="200 train and 40 validation"):
        create_manual_review_pack(
            ManualReviewPackConfig(
                train_manifest_path=train_root / "manifest.json",
                validation_manifest_path=validation_root / "manifest.json",
                output_dir=output,
            )
        )
    assert not output.exists()


def test_human_review_attestation_requires_human_fields_and_valid_ssh_signature(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="attestation-train",
        split="train",
        target_successes=3,
        outcomes={0: False},
    )
    validation_root, _ = _collect(
        tmp_path,
        name="attestation-validation",
        split="validation",
        target_successes=2,
    )
    pack_root = tmp_path / "attestation-pack"
    pack = create_manual_review_pack(
        ManualReviewPackConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=pack_root,
            count=4,
            smoke=True,
        ),
        now_fn=lambda: datetime(2026, 8, 27, 1, 30, tzinfo=UTC),
    )

    key_root = tmp_path / "keys"
    key_root.mkdir()
    private_key = key_root / "reviewer_ed25519"
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "m2-test-reviewer",
            "-f",
            str(private_key),
        ],
        check=True,
    )
    reviewer_repo = tmp_path / "reviewer-repo"
    reviewer_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(reviewer_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(reviewer_repo), "config", "user.name", "M2 Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(reviewer_repo),
            "config",
            "user.email",
            "m2-test@example.invalid",
        ],
        check=True,
    )
    registry_path = reviewer_repo / "reviewers.json"
    with pytest.raises(ValueError, match="reviewer_id"):
        create_reviewer_registry(
            reviewer_repo / "invalid-reviewers.json",
            reviewer_id="human-reviewer-1\nattacker",
            display_name="Human Reviewer",
            ssh_public_key=private_key.with_suffix(".pub").read_text().strip(),
            declared_at_utc="2026-08-27T01:00:00+00:00",
        )
    assert not (reviewer_repo / "invalid-reviewers.json").exists()
    registry = create_reviewer_registry(
        registry_path,
        reviewer_id="human-reviewer-1",
        display_name="Human Reviewer",
        ssh_public_key=private_key.with_suffix(".pub").read_text().strip(),
        declared_at_utc="2026-08-27T01:00:00+00:00",
    )
    subprocess.run(
        ["git", "-C", str(reviewer_repo), "add", "reviewers.json"],
        check=True,
    )
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-08-27T01:05:00+00:00",
        "GIT_COMMITTER_DATE": "2026-08-27T01:05:00+00:00",
    }
    subprocess.run(
        ["git", "-C", str(reviewer_repo), "commit", "-q", "-m", "registry"],
        check=True,
        env=commit_env,
    )
    registry_commit = subprocess.run(
        ["git", "-C", str(reviewer_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    base_config = dict(
        review_pack_path=pack_root / "review-pack.json",
        completed_reviews_path=pack_root / "manual-review-template.jsonl",
        reviewer_registry_path=registry_path,
        reviewer_repository_root=reviewer_repo,
        reviewer_registry_commit=registry_commit,
        reviewer_id="human-reviewer-1",
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
        smoke=True,
    )
    unsigned_root = tmp_path / "unsigned-request"
    with pytest.raises(ValueError, match="human review fields"):
        create_human_review_attestation_request(
            HumanReviewAttestationConfig(
                **base_config,
                output_dir=unsigned_root,
            )
        )
    assert not unsigned_root.exists()

    completed_path = pack_root / "manual-review-completed.jsonl"
    completed_rows = []
    for index, line in enumerate(
        (pack_root / "manual-review-template.jsonl").read_text().splitlines()
    ):
        row = json.loads(line)
        row["reviewer_id"] = "human-reviewer-1"
        row["review_started_at_utc"] = f"2026-08-27T02:00:{index:02d}+00:00"
        row["review_completed_at_utc"] = f"2026-08-27T02:01:{index:02d}+00:00"
        row["finding"] = f"Human checked review row {index}; source and media agree."
        row["verdict"] = "consistent"
        completed_rows.append(row)
    completed_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in completed_rows
        ),
        encoding="utf-8",
    )
    request_root = tmp_path / "signed-request"
    request = create_human_review_attestation_request(
        HumanReviewAttestationConfig(
            **{
                **base_config,
                "completed_reviews_path": completed_path,
            },
            output_dir=request_root,
        )
    )
    assert request["status"] == "awaiting_signature"
    assert request["payload"]["review_pack"]["pack_id"] == pack["pack_id"]
    assert (
        request["payload"]["reviewer_registry"]["registry_id"]
        == registry["registry_id"]
    )
    signing_message = request_root / "attestation-message.jsonl"
    wrong_key = key_root / "wrong_ed25519"
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(wrong_key),
        ],
        check=True,
    )
    wrong_message = request_root / "wrong-message.jsonl"
    shutil.copyfile(signing_message, wrong_message)
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(wrong_key),
            "-n",
            ATTESTATION_NAMESPACE,
            str(wrong_message),
        ],
        check=True,
        capture_output=True,
    )
    wrong_output = request_root / "wrong-key-attestation.json"
    with pytest.raises(ValueError, match="signature verification failed"):
        finalize_human_review_attestation(
            request_root / "attestation-request.json",
            signature_path=Path(f"{wrong_message}.sig"),
            output_path=wrong_output,
            review_pack_path=pack_root / "review-pack.json",
            reviewer_registry_path=registry_path,
            reviewer_repository_root=reviewer_repo,
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
        )
    assert not wrong_output.exists()
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(private_key),
            "-n",
            ATTESTATION_NAMESPACE,
            str(signing_message),
        ],
        check=True,
        capture_output=True,
    )
    signature_path = Path(f"{signing_message}.sig")
    attestation_path = request_root / "attestation.json"
    attestation = finalize_human_review_attestation(
        request_root / "attestation-request.json",
        signature_path=signature_path,
        output_path=attestation_path,
        review_pack_path=pack_root / "review-pack.json",
        reviewer_registry_path=registry_path,
        reviewer_repository_root=reviewer_repo,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert attestation["status"] == "signed"
    validation = validate_human_review_attestation(
        attestation_path,
        review_pack_path=pack_root / "review-pack.json",
        reviewer_registry_path=registry_path,
        reviewer_repository_root=reviewer_repo,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert validation.valid, validation.format_errors()
    assert validation.complete is True
    assert validation.formal is False
    assert validation.status == "signed"

    tampered = json.loads(attestation_path.read_text())
    tampered["reviews"][0]["finding"] = "Attacker changed the human finding."
    tampered_path = request_root / "tampered-attestation.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_validation = validate_human_review_attestation(
        tampered_path,
        review_pack_path=pack_root / "review-pack.json",
        reviewer_registry_path=registry_path,
        reviewer_repository_root=reviewer_repo,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert not tampered_validation.valid
    assert tampered_validation.complete is False

    drifted_registry = json.loads(registry_path.read_text())
    drifted_registry["reviewers"][0]["display_name"] = "Changed Reviewer"
    registry_path.write_text(json.dumps(drifted_registry), encoding="utf-8")
    drift_validation = validate_human_review_attestation(
        attestation_path,
        review_pack_path=pack_root / "review-pack.json",
        reviewer_registry_path=registry_path,
        reviewer_repository_root=reviewer_repo,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )
    assert not drift_validation.valid
    assert drift_validation.complete is False


def test_lineage_receipt_accepts_only_equal_runtime_and_allowlisted_docs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "lineage-repo"
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "config", "user.name", "Lineage"],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "lineage@example.invalid",
        ],
        check=True,
    )
    (repository / "src").mkdir()
    (repository / "docs").mkdir()
    (repository / "src/runtime.py").write_text("STRICT_PREDICATE = True\n")
    (repository / "src/writer.py").write_text("SCHEMA_VERSION = 1\n")
    (repository / "docs/data.md").write_text("old docs\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "add", "src", "docs"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "commit", "-q", "-m", "source"],
        check=True,
    )
    accepted_commit = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repository / "docs/data.md").write_text("new docs only\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "add", "docs/data.md"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "commit", "-q", "-m", "docs"],
        check=True,
    )
    baseline_commit = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    common = {
        "schema_version": "m2-collection-manifest.v1",
        "formal": False,
        "git": {
            "commit": accepted_commit,
            "source_provenance_clean": True,
            "provenance_complete": True,
        },
        "controller": {"damping": 0.025, "max_attempts": 2},
        "assets": {"aggregate_manifest_sha256": "a" * 64},
        "environment": {
            "schema_version": "m2-environment.v1",
            "mujoco_version": "3.11.0",
            "task_config": {"success_hold_s": 1.0},
            "compiled_model": {"fingerprint_sha256": "b" * 64},
        },
    }
    train_manifest = tmp_path / "lineage-train.json"
    validation_manifest = tmp_path / "lineage-validation.json"
    train_manifest.write_text(
        json.dumps({**common, "split": "train", "attempt_count": 2}),
        encoding="utf-8",
    )
    validation_manifest.write_text(
        json.dumps({**common, "split": "validation", "attempt_count": 1}),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "lineage-revalidation.json"
    receipt = create_lineage_revalidation_receipt(
        LineageRevalidationConfig(
            repository_root=repository,
            train_manifest_path=train_manifest,
            validation_manifest_path=validation_manifest,
            output_path=receipt_path,
            accepted_commit=accepted_commit,
            baseline_commit=baseline_commit,
            runtime_paths=("src/runtime.py", "src/writer.py"),
            documentation_allowlist=("docs/data.md",),
            smoke=True,
        ),
        now_fn=lambda: datetime(2026, 8, 27, 3, 0, tzinfo=UTC),
    )
    assert receipt["schema_version"] == "m1-m2-lineage-revalidation.v1"
    assert receipt["status"] == "passed"
    assert receipt["runtime_equality"]["equal"] is True
    assert receipt["repository_differences"]["changed_paths"] == ["docs/data.md"]
    report = validate_lineage_revalidation_receipt(
        receipt_path,
        repository_root=repository,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
    )
    assert report.valid, report.format_errors()
    assert report.passed is True
    reference = replay_module._validated_lineage_reference(
        receipt_path,
        replay_dir=tmp_path,
        project_root=repository,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        require_formal=False,
    )
    assert reference["provided"] is True
    assert reference["validated"] is True
    assert reference["receipt_id"] == receipt["receipt_id"]
    with pytest.raises(ReplayProvenanceError, match="formal lineage"):
        replay_module._validated_lineage_reference(
            receipt_path,
            replay_dir=tmp_path,
            project_root=repository,
            train_manifest_path=train_manifest,
            validation_manifest_path=validation_manifest,
            require_formal=True,
        )

    (repository / "src/runtime.py").write_text("STRICT_PREDICATE = False\n")
    drifted = validate_lineage_revalidation_receipt(
        receipt_path,
        repository_root=repository,
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
    )
    assert not drifted.valid
    assert drifted.passed is False
    assert any(issue.code == "lineage.source" for issue in drifted.errors)


def test_lineage_receipt_rejects_runtime_change_between_commits(tmp_path: Path) -> None:
    repository = tmp_path / "lineage-runtime-drift"
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "config", "user.name", "Lineage"],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "lineage@example.invalid",
        ],
        check=True,
    )
    runtime = repository / "runtime.py"
    runtime.write_text("version = 1\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "add", "runtime.py"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "commit", "-q", "-m", "v1"],
        check=True,
    )
    accepted = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime.write_text("version = 2\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "add", "runtime.py"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "commit", "-q", "-m", "v2"],
        check=True,
    )
    baseline = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common = {
        "schema_version": "m2-collection-manifest.v1",
        "formal": False,
        "git": {
            "commit": accepted,
            "source_provenance_clean": True,
            "provenance_complete": True,
        },
        "controller": {},
        "assets": {"aggregate_manifest_sha256": "a" * 64},
        "environment": {},
        "attempt_count": 1,
    }
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text(json.dumps({**common, "split": "train"}), encoding="utf-8")
    validation.write_text(
        json.dumps({**common, "split": "validation"}), encoding="utf-8"
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="runtime source differs"):
        create_lineage_revalidation_receipt(
            LineageRevalidationConfig(
                repository_root=repository,
                train_manifest_path=train,
                validation_manifest_path=validation,
                output_path=output,
                accepted_commit=accepted,
                baseline_commit=baseline,
                runtime_paths=("runtime.py",),
                documentation_allowlist=(),
                smoke=True,
            )
        )
    assert not output.exists()


def test_m2_gate_cannot_publish_without_signed_human_attestation(
    tmp_path: Path,
) -> None:
    train_root, _ = _collect(
        tmp_path,
        name="gate-train",
        split="train",
        target_successes=2,
    )
    validation_root, _ = _collect(
        tmp_path,
        name="gate-validation",
        split="validation",
        target_successes=2,
    )
    pack_root = tmp_path / "gate-review-pack"
    create_manual_review_pack(
        ManualReviewPackConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=pack_root,
            count=2,
            smoke=True,
        )
    )
    output = tmp_path / "m2-gate.json"
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="signed human review attestation"):
        create_m2_gate_receipt(
            M2GateConfig(
                project_root=tmp_path,
                train_manifest_path=train_root / "manifest.json",
                validation_manifest_path=validation_root / "manifest.json",
                train_validation_report_path=missing,
                pair_validation_report_path=missing,
                train_report_path=missing,
                validation_report_path=missing,
                replay_summary_path=missing,
                review_pack_path=pack_root / "review-pack.json",
                human_attestation_path=missing,
                reviewer_registry_path=missing,
                lineage_receipt_path=missing,
                output_path=output,
                smoke=True,
            )
        )
    assert not output.exists()


def test_m2_gate_smoke_revalidates_every_bound_artifact(tmp_path: Path) -> None:
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "config", "user.name", "M2 Gate"],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(tmp_path),
            "config",
            "user.email",
            "m2-gate@example.invalid",
        ],
        check=True,
    )
    key = tmp_path / "reviewer-key"
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(key),
        ],
        check=True,
    )
    registry = tmp_path / "reviewers.json"
    create_reviewer_registry(
        registry,
        reviewer_id="m2-gate-reviewer",
        display_name="M2 Gate Reviewer",
        ssh_public_key=key.with_suffix(".pub").read_text().strip(),
        declared_at_utc="2026-08-28T01:00:00+00:00",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/runtime.py").write_text("STRICT_PREDICATE = True\n")
    (tmp_path / "src/writer.py").write_text("SCHEMA_VERSION = 1\n")
    (tmp_path / "docs/data.md").write_text("v1\n")
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(tmp_path),
            "add",
            "reviewers.json",
            "src/runtime.py",
            "src/writer.py",
            "docs/data.md",
        ],
        check=True,
    )
    accepted_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-08-28T01:05:00+00:00",
        "GIT_COMMITTER_DATE": "2026-08-28T01:05:00+00:00",
    }
    subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "commit", "-q", "-m", "source"],
        check=True,
        env=accepted_env,
    )
    accepted = subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "docs/data.md").write_text("v2 docs only\n")
    subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "add", "docs/data.md"],
        check=True,
    )
    baseline_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-08-28T01:10:00+00:00",
        "GIT_COMMITTER_DATE": "2026-08-28T01:10:00+00:00",
    }
    subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "commit", "-q", "-m", "docs"],
        check=True,
        env=baseline_env,
    )
    baseline = subprocess.run(
        ["/usr/bin/git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    train_root, train_manifest = _collect(
        tmp_path,
        name="gate-full-train",
        split="train",
        target_successes=2,
    )
    validation_root, validation_manifest = _collect(
        tmp_path,
        name="gate-full-validation",
        split="validation",
        target_successes=2,
    )
    for root, manifest in (
        (train_root, train_manifest),
        (validation_root, validation_manifest),
    ):
        manifest["git"]["commit"] = accepted
        _write_manifest(root / "manifest.json", manifest)

    train_validation_path = tmp_path / "train-validation.json"
    pair_validation_path = tmp_path / "pair-validation.json"
    train_validation_path.write_text(
        json.dumps(
            {
                "schema_version": "m2-collection-validation-report.v1",
                "valid": True,
                "manifest_paths": [(train_root / "manifest.json").as_posix()],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    pair_validation_path.write_text(
        json.dumps(
            {
                "schema_version": "m2-collection-validation-report.v1",
                "valid": True,
                "manifest_paths": [
                    (validation_root / "manifest.json").as_posix(),
                    (train_root / "manifest.json").as_posix(),
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    train_report_root = tmp_path / "gate-train-report"
    validation_report_root = tmp_path / "gate-validation-report"
    generate_collection_report(
        ReportConfig(
            manifest_path=train_root / "manifest.json",
            output_dir=train_report_root,
            manual_review_count=2,
            smoke=True,
        )
    )
    generate_collection_report(
        ReportConfig(
            manifest_path=validation_root / "manifest.json",
            output_dir=validation_report_root,
            manual_review_count=2,
            smoke=True,
        )
    )

    replay_root = tmp_path / "gate-pair-replay"
    create_replay_plan(
        ReplayPlanConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=replay_root,
            count=2,
            smoke=True,
        )
    )
    lineage_path = replay_root / "lineage-revalidation.json"
    create_lineage_revalidation_receipt(
        LineageRevalidationConfig(
            repository_root=tmp_path,
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_path=lineage_path,
            accepted_commit=accepted,
            baseline_commit=baseline,
            runtime_paths=("src/runtime.py", "src/writer.py"),
            documentation_allowlist=("docs/data.md",),
            smoke=True,
        )
    )
    replay_manifest_pair(
        PairReplayConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            replay_dir=replay_root,
            smoke=True,
            lineage_receipt_path=lineage_path,
            project_root=tmp_path,
        ),
        env_factory=_ReplayEnv,
        git_state_fn=lambda _: _git_state(),
        asset_provenance_fn=lambda *_args, **_kwargs: _AssetProvenance(),
        environment_fingerprint_fn=_environment_fingerprint,
    )

    pack_root = tmp_path / "gate-full-review-pack"
    create_manual_review_pack(
        ManualReviewPackConfig(
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=pack_root,
            count=2,
            smoke=True,
        )
    )
    completed = pack_root / "completed.jsonl"
    completed.write_text(
        "".join(
            json.dumps(
                {
                    **json.loads(line),
                    "reviewer_id": "m2-gate-reviewer",
                    "review_started_at_utc": f"2026-08-28T02:00:{index:02d}+00:00",
                    "review_completed_at_utc": f"2026-08-28T02:01:{index:02d}+00:00",
                    "finding": f"Human reviewed row {index}; source and media agree.",
                    "verdict": "consistent",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for index, line in enumerate(
                (pack_root / "manual-review-template.jsonl").read_text().splitlines()
            )
        ),
        encoding="utf-8",
    )
    attestation_root = tmp_path / "gate-attestation"
    create_human_review_attestation_request(
        HumanReviewAttestationConfig(
            review_pack_path=pack_root / "review-pack.json",
            completed_reviews_path=completed,
            reviewer_registry_path=registry,
            reviewer_repository_root=tmp_path,
            reviewer_registry_commit=accepted,
            reviewer_id="m2-gate-reviewer",
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            output_dir=attestation_root,
            smoke=True,
        )
    )
    signing_message = attestation_root / "attestation-message.jsonl"
    subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(key),
            "-n",
            ATTESTATION_NAMESPACE,
            str(signing_message),
        ],
        check=True,
        capture_output=True,
    )
    attestation_path = attestation_root / "attestation.json"
    finalize_human_review_attestation(
        attestation_root / "attestation-request.json",
        signature_path=Path(f"{signing_message}.sig"),
        output_path=attestation_path,
        review_pack_path=pack_root / "review-pack.json",
        reviewer_registry_path=registry,
        reviewer_repository_root=tmp_path,
        train_manifest_path=train_root / "manifest.json",
        validation_manifest_path=validation_root / "manifest.json",
    )

    gate_path = tmp_path / "m2-gate-smoke.json"
    gate = create_m2_gate_receipt(
        M2GateConfig(
            project_root=tmp_path,
            train_manifest_path=train_root / "manifest.json",
            validation_manifest_path=validation_root / "manifest.json",
            train_validation_report_path=train_validation_path,
            pair_validation_report_path=pair_validation_path,
            train_report_path=train_report_root / "report.json",
            validation_report_path=validation_report_root / "report.json",
            replay_summary_path=replay_root / "summary.json",
            review_pack_path=pack_root / "review-pack.json",
            human_attestation_path=attestation_path,
            reviewer_registry_path=registry,
            lineage_receipt_path=lineage_path,
            output_path=gate_path,
            smoke=True,
        )
    )
    assert gate["status"] == "passed"
    assert gate["formal"] is False
    assert gate["checks"] == {
        "collection_pair_valid": True,
        "published_validation_reports_valid": True,
        "data_reports_recomputed": True,
        "paired_replay_passed": True,
        "human_review_signed_complete": True,
        "lineage_passed": True,
        "all_input_hashes_stable": True,
    }
    gate_validation = validate_m2_gate_receipt(gate_path, project_root=tmp_path)
    assert gate_validation.valid, gate_validation.format_errors()
    assert gate_validation.passed is True

    train_report_payload = json.loads((train_report_root / "report.json").read_text())
    train_report_payload["actions"]["dimensions"][0]["mean"] += 1.0
    _write_manifest(train_report_root / "report.json", train_report_payload)
    tampered = validate_m2_gate_receipt(gate_path, project_root=tmp_path)
    assert not tampered.valid
    assert tampered.passed is False


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
