from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _source_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src/policy").mkdir(parents=True)
    (repo / "configs/m3").mkdir(parents=True)
    (repo / "src/policy/adapter.py").write_text("VALUE = 1\n", encoding="utf-8")
    scope = repo / "configs/m3/lerobot-source-inputs.json"
    scope.write_text(
        json.dumps(
            {
                "schema_version": "lerobot-source-input-scope.v1",
                "files": ["configs/m3/lerobot-source-inputs.json"],
                "trees": [{"path": "src/policy", "extensions": [".py"]}],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Policy Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, scope


def _formal_source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "formal-repo"
    (repo / "src/policy").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/policy/__init__.py").write_text("\n", encoding="utf-8")
    (repo / "scripts/check_training_environment.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Policy Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def test_canonical_json_golden_identity_and_no_clobber(tmp_path: Path) -> None:
    canonical = importlib.import_module("policy.canonical_json")
    payload = {"float": 1.0, "unicode": "机器人", "false": False, "null": None}
    expected = b'{"false":false,"float":1.0,"null":null,"unicode":"\xe6\x9c\xba\xe5\x99\xa8\xe4\xba\xba"}'

    assert canonical.canonical_bytes(payload) == expected
    assert hashlib.sha256(expected).hexdigest() == (
        "bde5262f0df421331496b60737cea47a92cd8fb49fa23694126b85921afcbc20"
    )

    document = canonical.materialize_identity(payload, "manifest_id")
    assert document["manifest_id"] == "sha256:" + hashlib.sha256(expected).hexdigest()
    assert canonical.validate_identity(document, "manifest_id")

    output = tmp_path / "receipt.json"
    canonical.publish_canonical_no_clobber(output, document)
    assert output.read_bytes() == canonical.canonical_bytes(document) + b"\n"
    with pytest.raises(FileExistsError):
        canonical.publish_canonical_no_clobber(output, {"replacement": True})
    assert output.read_bytes() == canonical.canonical_bytes(document) + b"\n"


def test_source_manifest_ignores_names_outside_scope_but_detects_source_drift(
    tmp_path: Path,
) -> None:
    source = importlib.import_module("policy.source_provenance")
    repo, scope = _source_repo(tmp_path)

    baseline, observation = source.build_source_snapshot(
        repo,
        scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    assert baseline["schema_version"] == "lerobot-source-input-manifest.v1"
    assert baseline["source_provenance_clean"] is True
    assert baseline["manifest_id"].startswith("sha256:")
    assert [entry["path"] for entry in baseline["entries"]] == [
        "configs/m3/lerobot-source-inputs.json",
        "src/policy/adapter.py",
    ]
    assert set(baseline) >= {
        "schema_version",
        "manifest_id",
        "git_commit",
        "head_tree",
        "scope_config_sha256",
        "path_set_sha256",
        "index_entries_sha256",
        "content_entries_sha256",
        "tracked_status_sha256",
        "source_provenance_clean",
        "entries",
    }

    (repo / "notes").mkdir()
    (repo / "notes/private-name.md").write_text("private\n", encoding="utf-8")
    same_manifest, changed_observation = source.build_source_snapshot(
        repo,
        scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    assert same_manifest == baseline
    assert changed_observation != observation
    assert changed_observation["untracked_count"] == 1
    assert "private-name" not in json.dumps(changed_observation)

    (repo / "src/policy/adapter.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty_manifest, _ = source.build_source_snapshot(
        repo,
        scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    assert dirty_manifest["source_provenance_clean"] is False
    assert (
        dirty_manifest["content_entries_sha256"] != baseline["content_entries_sha256"]
    )
    assert dirty_manifest["manifest_id"] != baseline["manifest_id"]


def test_source_manifest_and_recheck_outputs_are_no_clobber(tmp_path: Path) -> None:
    source = importlib.import_module("policy.source_provenance")
    repo, scope = _source_repo(tmp_path)
    manifest_path = tmp_path / "source-input-manifest.json"
    observation_path = tmp_path / "worktree-observation.json"

    baseline, _ = source.publish_source_snapshot(
        repo,
        scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
        manifest_output=manifest_path,
        observation_output=observation_path,
    )
    original_manifest = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        source.publish_source_snapshot(
            repo,
            scope,
            scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
            manifest_output=manifest_path,
            observation_output=observation_path,
        )
    assert manifest_path.read_bytes() == original_manifest

    evidence_path = tmp_path / "recheck.json"
    evidence = source.verify_current(
        repo,
        scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
        baseline=baseline,
        evidence_output=evidence_path,
    )
    assert evidence["schema_version"] == "lerobot-source-recheck.v1"
    assert evidence["complete"] is True
    assert evidence["exact_match"] is True
    assert evidence["git_commit_match"] is True
    assert evidence["path_set_match"] is True
    assert evidence["index_entries_match"] is True
    assert evidence["content_entries_match"] is True

    (repo / "src/policy/adapter.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        source.verify_current(
            repo,
            scope,
            scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
            baseline=baseline,
            evidence_output=evidence_path,
        )


def test_source_manifest_blocks_index_symlinks_and_same_tree_new_commit(
    tmp_path: Path,
) -> None:
    source = importlib.import_module("policy.source_provenance")

    index_repo, index_scope = _source_repo(tmp_path / "index")
    (index_repo / "src/policy/adapter.py").write_text("VALUE = 9\n", encoding="utf-8")
    _git(index_repo, "add", "src/policy/adapter.py")
    indexed, _ = source.build_source_snapshot(
        index_repo,
        index_scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    adapter_entry = next(
        entry
        for entry in indexed["entries"]
        if entry["path"] == "src/policy/adapter.py"
    )
    assert indexed["source_provenance_clean"] is False
    assert adapter_entry["index_blob_oid"] != adapter_entry["head_blob_oid"]

    symlink_repo, symlink_scope = _source_repo(tmp_path / "symlink")
    adapter = symlink_repo / "src/policy/adapter.py"
    adapter.unlink()
    adapter.symlink_to(symlink_repo / "outside.py")
    symlinked, _ = source.build_source_snapshot(
        symlink_repo,
        symlink_scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    assert symlinked["source_provenance_clean"] is False
    assert {error["code"] for error in symlinked["errors"]} >= {"scope_entry_symlink"}

    commit_repo, commit_scope = _source_repo(tmp_path / "commit")
    baseline, _ = source.build_source_snapshot(
        commit_repo,
        commit_scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    _git(commit_repo, "commit", "--allow-empty", "-qm", "same tree, new commit")
    superseding, _ = source.build_source_snapshot(
        commit_repo,
        commit_scope,
        scope_config_repo_path="configs/m3/lerobot-source-inputs.json",
    )
    assert superseding["source_provenance_clean"] is True
    assert superseding["head_tree"] == baseline["head_tree"]
    assert superseding["git_commit"] != baseline["git_commit"]
    assert superseding["manifest_id"] != baseline["manifest_id"]


def test_environment_contract_audit_and_failure_classification() -> None:
    verifier = importlib.import_module("policy.verify_project_training_environment")

    audit = verifier.audit_dependency_contract(
        REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"
    )
    assert audit["passed"] is True
    assert audit["direct_pins"] == {
        "safetensors": "0.8.0",
        "torch": "2.13.0",
        "torchvision": "0.28.0",
    }
    assert audit["locked_versions"] == {
        "safetensors": "0.8.0",
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
    }
    assert (
        verifier.classify_failure("Temporary failure in name resolution", 1)
        == "network"
    )
    assert verifier.classify_failure("No space left on device", 1) == "resource"
    assert (
        verifier.classify_failure("unsupported compute capability", 1)
        == "compatibility"
    )


def test_formal_environment_verification_fails_closed_on_inherited_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = importlib.import_module("policy.verify_project_training_environment")
    monkeypatch.setenv("PYTHONPATH", "/tmp/contamination")

    evidence = verifier.verify_project_training_environment(
        repo_root=REPO_ROOT,
        expected_prefix=REPO_ROOT / ".venv",
        expected_policy_root=REPO_ROOT / "src/policy",
        expected_editable_root=REPO_ROOT,
        pyproject=REPO_ROOT / "pyproject.toml",
        lock=REPO_ROOT / "uv.lock",
        require_cuda_smoke=False,
        formal=True,
    )

    assert evidence["schema_version"] == "project-train-env-receipt.v1"
    assert evidence["status"] == "blocked"
    assert evidence["passed"] is None
    assert evidence["formal"] is True
    assert evidence["checks"]["pythonpath_unset"]["passed"] is False
    # The source gate and interpreter-environment gate are independent.  The
    # working tree is HEAD-identical here; inherited PYTHONPATH alone must
    # still keep the formal receipt blocked.
    assert evidence["checks"]["formal_source_head_identical"]["passed"] is True
    # The CI foundation job intentionally installs only the test group; the
    # optional train group is checked by the project-environment lane.  This
    # regression must therefore assert the contamination gate independently
    # of whether train-only packages are installed.


def test_formal_source_gate_rejects_symlink_and_startup_hook_bypasses(
    tmp_path: Path,
) -> None:
    verifier = importlib.import_module("policy.verify_project_training_environment")

    symlink_repo = _formal_source_repo(tmp_path / "symlink")
    assert verifier.inspect_formal_source_state(symlink_repo)["passed"] is True
    outside = tmp_path / "outside.py"
    outside.write_text("RAISED = True\n", encoding="utf-8")
    (symlink_repo / "src/policy/evil.py").symlink_to(outside)
    symlink_state = verifier.inspect_formal_source_state(symlink_repo)
    assert symlink_state["passed"] is False
    assert "src/policy/evil.py" in symlink_state["unsafe_paths"]

    startup_repo = _formal_source_repo(tmp_path / "startup")
    (startup_repo / "scripts/sitecustomize.py").write_text(
        "RAISED = True\n", encoding="utf-8"
    )
    startup_state = verifier.inspect_formal_source_state(startup_repo)
    assert startup_state["passed"] is False
    startup_entry = next(
        entry
        for entry in startup_state["entries"]
        if entry["path"] == "scripts/sitecustomize.py"
    )
    assert startup_entry["tracked"] is False


def test_environment_result_publish_is_canonical_and_no_clobber(tmp_path: Path) -> None:
    verifier = importlib.import_module("policy.verify_project_training_environment")
    canonical = importlib.import_module("policy.canonical_json")
    output = tmp_path / "project-receipt.json"
    evidence = {
        "schema_version": "project-train-env-receipt.v1",
        "status": "blocked",
        "passed": None,
    }

    document = verifier.publish_result_no_clobber(output, evidence)
    assert canonical.validate_identity(document, "receipt_id")
    assert output.read_bytes() == canonical.canonical_bytes(document) + b"\n"
    with pytest.raises(FileExistsError):
        verifier.publish_result_no_clobber(output, evidence)


def test_training_environment_script_uses_the_editable_policy_package() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "scripts/check_training_environment.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Verify the locked editable project training environment" in completed.stdout


def test_train_group_and_policy_discovery_are_exact_and_non_default() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["build-system"]["requires"] == ["setuptools==84.0.0"]
    assert project["dependency-groups"]["train"] == [
        "torch==2.13.0",
        "torchvision==0.28.0",
        "safetensors==0.8.0",
    ]
    assert "torch==2.13.0" not in project["project"]["dependencies"]
    assert "torchvision==0.28.0" not in project["project"]["dependencies"]
    assert "safetensors==0.8.0" not in project["project"]["dependencies"]
    assert "h5py==3.16.0" in project["project"]["dependencies"]
    assert "ruff==0.15.10" in project["dependency-groups"]["test"]

    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "data*",
        "env*",
        "evaluation*",
        "expert*",
        "policy*",
        "robotics*",
    ]
    assert project["tool"]["setuptools"]["packages"]["find"]["exclude"] == [
        "robotics.tests*"
    ]
    assert project["tool"]["uv"]["sources"] == {
        "torch": {"index": "pytorch-cu130"},
        "torchvision": {"index": "pytorch-cu130"},
    }
    assert project["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
        }
    ]


def test_lock_records_exact_train_group_and_reviewable_sources() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}
    root = packages["panda-reactive-il"]

    assert root["dev-dependencies"]["train"] == [
        {"name": "safetensors"},
        {"name": "torch"},
        {"name": "torchvision"},
    ]
    assert root["metadata"]["requires-dev"]["train"] == [
        {"name": "safetensors", "specifier": "==0.8.0"},
        {
            "name": "torch",
            "specifier": "==2.13.0",
            "index": "https://download.pytorch.org/whl/cu130",
        },
        {
            "name": "torchvision",
            "specifier": "==0.28.0",
            "index": "https://download.pytorch.org/whl/cu130",
        },
    ]

    expected = {
        "safetensors": ("0.8.0", "https://pypi.org/simple"),
        "torch": ("2.13.0+cu130", "https://download.pytorch.org/whl/cu130"),
        "torchvision": ("0.28.0+cu130", "https://download.pytorch.org/whl/cu130"),
    }
    for name, (version, registry) in expected.items():
        package = packages[name]
        assert package["version"] == version
        assert package["source"] == {"registry": registry}
        assert package.get("sdist") or package.get("wheels")
        for artifact in [
            *package.get("wheels", []),
            *([package["sdist"]] if package.get("sdist") else []),
        ]:
            assert artifact["url"].startswith("https://")
            assert artifact["hash"].startswith("sha256:")
