from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from expert.evaluate import _git_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_foundation_check_uses_the_installed_package_without_mutating_sys_path() -> (
    None
):
    original_path = list(sys.path)
    environment_names = ("MENAGERIE", "MUJOCO_GL", "PYTHONPYCACHEPREFIX")
    original_environment = {name: os.environ.get(name) for name in environment_names}
    try:
        runpy.run_path(
            str(PROJECT_ROOT / "scripts" / "check_foundation.py"),
            run_name="foundation_import_probe",
        )
        observed_path = list(sys.path)
    finally:
        sys.path[:] = original_path
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert observed_path == original_path


def _write_activation(root: Path, environment: str) -> None:
    activation = root / environment / "bin" / "activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(f"export SELECTED_ENV={environment}\n", encoding="utf-8")


def _source_env(root: Path, command: str, *, pythonpath: str | None) -> str:
    shutil.copy2(PROJECT_ROOT / "env.sh", root / "env.sh")
    environment = os.environ.copy()
    if pythonpath is None:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1" && eval "$2"',
            "env-test",
            str(root / "env.sh"),
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _new_repository(tmp_path: Path, *, ignore: str = "") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (repo / ".gitignore").write_text(ignore, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def test_env_sh_prefers_the_uv_environment_when_both_exist(tmp_path: Path) -> None:
    _write_activation(tmp_path, "venv")
    _write_activation(tmp_path, ".venv")

    output = _source_env(
        tmp_path,
        "printf '%s\\n' \"$SELECTED_ENV\"",
        pythonpath=None,
    )

    assert output == ".venv\n"


def test_env_sh_rejects_an_unlocked_legacy_environment(tmp_path: Path) -> None:
    _write_activation(tmp_path, "venv")
    shutil.copy2(PROJECT_ROOT / "env.sh", tmp_path / "env.sh")

    completed = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1" && printf "%s\\n" "${SELECTED_ENV-unset}"',
            "env-test",
            str(tmp_path / "env.sh"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "uv sync --locked --group test" in completed.stderr


def test_env_sh_does_not_inject_the_source_tree_into_pythonpath(
    tmp_path: Path,
) -> None:
    _write_activation(tmp_path, ".venv")

    output = _source_env(
        tmp_path,
        "printf '%s\\n' \"${PYTHONPATH-unset}\"",
        pythonpath=None,
    )

    assert output == "unset\n"


def test_env_sh_clears_an_inherited_pythonpath(tmp_path: Path) -> None:
    _write_activation(tmp_path, ".venv")

    output = _source_env(
        tmp_path,
        "printf '%s\\n' \"${PYTHONPATH-unset}\"",
        pythonpath="/tmp/untrusted-pythonpath",
    )

    assert output == "unset\n"


def test_git_state_fails_closed_for_tracked_changes(tmp_path: Path) -> None:
    repo = _new_repository(tmp_path)
    (repo / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")

    state = _git_state(repo)

    assert state["provenance_complete"] is True
    assert state["tracked_worktree_clean"] is False
    assert state["worktree_clean"] is False
    assert state["source_provenance_clean"] is False


def test_git_state_hashes_relevant_untracked_source_without_exposing_contents(
    tmp_path: Path,
) -> None:
    repo = _new_repository(tmp_path)
    source_content = b"PRIVATE_MARKER = 'do-not-log-this'\n"
    source_path = repo / "src" / "local_override.py"
    source_path.write_bytes(source_content)

    state = _git_state(repo)

    assert state["tracked_worktree_clean"] is True
    assert state["worktree_clean"] is False
    assert state["source_provenance_clean"] is False
    assert state["untracked_file_count"] == 1
    assert state["relevant_untracked_files"] == [
        {
            "git_status": "untracked",
            "path": "src/local_override.py",
            "sha256": hashlib.sha256(source_content).hexdigest(),
            "size_bytes": len(source_content),
        }
    ]
    assert "do-not-log-this" not in json.dumps(state)


def test_git_state_treats_root_python_startup_hooks_as_runtime_inputs(
    tmp_path: Path,
) -> None:
    repo = _new_repository(tmp_path)
    hook_content = b"RUNTIME_SIDE_EFFECT = True\n"
    (repo / "sitecustomize.py").write_bytes(hook_content)

    state = _git_state(repo)

    assert state["source_provenance_clean"] is False
    assert state["relevant_untracked_files"] == [
        {
            "git_status": "untracked",
            "path": "sitecustomize.py",
            "sha256": hashlib.sha256(hook_content).hexdigest(),
            "size_bytes": len(hook_content),
        }
    ]


@pytest.mark.parametrize("exclusion_source", ["gitignore", "info", "global"])
def test_git_state_fails_closed_for_ignored_root_python_startup_hooks(
    tmp_path: Path,
    exclusion_source: str,
) -> None:
    patterns = "sitecustomize.py\nusercustomize.py\n"
    repo = _new_repository(
        tmp_path, ignore=patterns if exclusion_source == "gitignore" else ""
    )
    if exclusion_source == "info":
        (repo / ".git" / "info" / "exclude").write_text(patterns, encoding="utf-8")
    elif exclusion_source == "global":
        excludes = tmp_path / "global-excludes"
        excludes.write_text(patterns, encoding="utf-8")
        _git(repo, "config", "core.excludesFile", str(excludes))
    expected = []
    for name in ("sitecustomize.py", "usercustomize.py"):
        content = f"HOOK_NAME = {name!r}\n".encode()
        (repo / name).write_bytes(content)
        expected.append(
            {
                "git_status": "ignored",
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )

    state = _git_state(repo)

    assert state["provenance_complete"] is True
    assert state["tracked_worktree_clean"] is True
    assert state["worktree_clean"] is True
    assert state["source_provenance_clean"] is False
    assert state["relevant_untracked_files"] == expected


def test_git_state_ignores_large_ignored_run_artifacts(tmp_path: Path) -> None:
    repo = _new_repository(tmp_path, ignore="runs/\n")
    artifact = repo / "runs" / "m1" / "video.mp4"
    artifact.parent.mkdir(parents=True)
    with artifact.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024)

    state = _git_state(repo)

    assert state["provenance_complete"] is True
    assert state["tracked_worktree_clean"] is True
    assert state["worktree_clean"] is True
    assert state["source_provenance_clean"] is True
    assert state["untracked_file_count"] == 0
    assert state["relevant_untracked_files"] == []


def test_git_state_ignores_generated_bytecode_and_package_metadata(
    tmp_path: Path,
) -> None:
    repo = _new_repository(
        tmp_path,
        ignore="src/**/__pycache__/\nsrc/*.egg-info/\n",
    )
    bytecode = repo / "src" / "__pycache__" / "module.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"generated bytecode")
    metadata = repo / "src" / "fixture.egg-info" / "PKG-INFO"
    metadata.parent.mkdir()
    metadata.write_text("generated metadata\n", encoding="utf-8")

    state = _git_state(repo)

    assert state["provenance_complete"] is True
    assert state["source_provenance_clean"] is True
    assert state["relevant_untracked_files"] == []


def test_git_state_fails_closed_for_ignored_source_files(tmp_path: Path) -> None:
    repo = _new_repository(tmp_path, ignore="src/local_override.py\n")
    source_content = b"VALUE = 99\n"
    source_path = repo / "src" / "local_override.py"
    source_path.write_bytes(source_content)

    state = _git_state(repo)

    assert state["provenance_complete"] is True
    assert state["tracked_worktree_clean"] is True
    assert state["worktree_clean"] is True
    assert state["source_provenance_clean"] is False
    assert state["relevant_untracked_files"] == [
        {
            "git_status": "ignored",
            "path": "src/local_override.py",
            "sha256": hashlib.sha256(source_content).hexdigest(),
            "size_bytes": len(source_content),
        }
    ]
