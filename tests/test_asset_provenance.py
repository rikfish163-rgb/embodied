from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from env.asset_provenance import (
    AssetProvenanceError,
    _parse_git_tracked_paths,
    collect_asset_provenance,
)
from expert.evaluate import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REVISION = (
    b"repository=https://github.com/google-deepmind/mujoco_menagerie\n"
    b"revision=0123456789abcdef0123456789abcdef01234567\n"
    b"vendored_path=franka_emika_panda\n"
    b"license=Apache-2.0\n"
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _new_asset_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    asset_root = repo / "menagerie" / "franka_emika_panda"
    (asset_root / "assets").mkdir(parents=True)
    (repo / "menagerie" / "VENDORED_REVISION").write_bytes(FIXTURE_REVISION)
    (asset_root / "assets" / "mesh.bin").write_bytes(b"mesh\0data\n")
    (asset_root / "panda.xml").write_bytes(b'<mujoco model="panda"/>\n')
    (asset_root / "scene.xml").write_bytes(b'<mujoco model="scene"/>\n')
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def test_collect_asset_provenance_builds_a_canonical_tracked_manifest(
    tmp_path: Path,
) -> None:
    repo = _new_asset_repository(tmp_path)

    provenance = collect_asset_provenance(repo, environ={})

    assert provenance.summary_dict() == {
        "aggregate_manifest_sha256": (
            "b60d6d2a0e682f00c0d83f1266739449b51345461957053e9b1c0508908793bf"
        ),
        "canonical_root": "menagerie/franka_emika_panda",
        "file_count": 4,
        "revision": {
            "license": "Apache-2.0",
            "repository": "https://github.com/google-deepmind/mujoco_menagerie",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "vendored_path": "franka_emika_panda",
        },
    }
    assert provenance.manifest_dict() == {
        "files": [
            {
                "path": "menagerie/VENDORED_REVISION",
                "sha256": (
                    "6887a8e5606b5ce664ce0c10615b102babfd29e2349d198135bcabbb92f83b3d"
                ),
                "size_bytes": 165,
            },
            {
                "path": "menagerie/franka_emika_panda/assets/mesh.bin",
                "sha256": (
                    "b70f135b45eb81e24c644ad8bdf7507598193db1ecfaa91c735e0f712825b190"
                ),
                "size_bytes": 10,
            },
            {
                "path": "menagerie/franka_emika_panda/panda.xml",
                "sha256": (
                    "ec7e7ae40776d85c9757c8e013b8ab1079615b52c53b21649954c135b8931a98"
                ),
                "size_bytes": 24,
            },
            {
                "path": "menagerie/franka_emika_panda/scene.xml",
                "sha256": (
                    "426b1b3db3ea6517c28001a785010110708bd5e9eaebf9b974b93eebf59b38ea"
                ),
                "size_bytes": 24,
            },
        ],
        "schema_version": "asset-manifest.v1",
    }
    serialized = json.dumps(provenance.summary_dict(), sort_keys=True)
    assert "<mujoco" not in serialized
    assert "mesh\\u0000data" not in serialized


def test_collect_asset_provenance_rejects_tracked_assets_changed_from_head(
    tmp_path: Path,
) -> None:
    repo = _new_asset_repository(tmp_path)
    scene_path = repo / "menagerie" / "franka_emika_panda" / "scene.xml"
    scene_path.write_bytes(b'<mujoco model="locally-modified"/>\n')

    with pytest.raises(AssetProvenanceError, match="committed snapshot"):
        collect_asset_provenance(repo, environ={})


@pytest.mark.parametrize(
    "revision_bytes",
    [
        pytest.param(FIXTURE_REVISION.replace(b"\n", b"\r\n"), id="crlf"),
        pytest.param(FIXTURE_REVISION + b"license=Apache-2.0\n", id="duplicate"),
        pytest.param(FIXTURE_REVISION + b"note=untrusted\n", id="unknown-key"),
        pytest.param(
            FIXTURE_REVISION.replace(
                b"repository=https://github.com/google-deepmind/mujoco_menagerie",
                b"repository=https://example.invalid/fork",
            ),
            id="wrong-repository",
        ),
        pytest.param(
            FIXTURE_REVISION.replace(
                b"revision=0123456789abcdef0123456789abcdef01234567",
                b"revision=main",
            ),
            id="non-commit-revision",
        ),
        pytest.param(
            FIXTURE_REVISION.replace(
                b"vendored_path=franka_emika_panda",
                b"vendored_path=../external",
            ),
            id="wrong-vendored-path",
        ),
        pytest.param(
            FIXTURE_REVISION.replace(b"license=Apache-2.0", b"license=unknown"),
            id="wrong-license",
        ),
        pytest.param(FIXTURE_REVISION + b"\xff", id="non-utf8"),
    ],
)
def test_collect_asset_provenance_rejects_malformed_revision_metadata(
    tmp_path: Path,
    revision_bytes: bytes,
) -> None:
    repo = _new_asset_repository(tmp_path)
    (repo / "menagerie" / "VENDORED_REVISION").write_bytes(revision_bytes)

    with pytest.raises(AssetProvenanceError):
        collect_asset_provenance(repo, environ={})


def test_collect_asset_provenance_rejects_an_external_menagerie_override(
    tmp_path: Path,
) -> None:
    repo = _new_asset_repository(tmp_path)
    external = tmp_path / "external-menagerie"

    with pytest.raises(AssetProvenanceError, match="MENAGERIE"):
        collect_asset_provenance(repo, environ={"MENAGERIE": str(external)})


@pytest.mark.parametrize("symlink_kind", ["root", "component", "leaf"])
def test_collect_asset_provenance_rejects_symlinked_asset_paths(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    repo = _new_asset_repository(tmp_path)
    asset_root = repo / "menagerie" / "franka_emika_panda"
    if symlink_kind == "root":
        target = tmp_path / "stored-root"
        asset_root.rename(target)
        asset_root.symlink_to(target, target_is_directory=True)
    elif symlink_kind == "component":
        component = asset_root / "assets"
        target = tmp_path / "stored-assets"
        component.rename(target)
        component.symlink_to(target, target_is_directory=True)
    else:
        leaf = asset_root / "scene.xml"
        target = tmp_path / "stored-scene.xml"
        leaf.rename(target)
        leaf.symlink_to(target)

    with pytest.raises(AssetProvenanceError, match="symlink"):
        collect_asset_provenance(repo, environ={})


@pytest.mark.parametrize("replacement", ["missing", "directory"])
def test_collect_asset_provenance_rejects_missing_or_non_regular_assets(
    tmp_path: Path,
    replacement: str,
) -> None:
    repo = _new_asset_repository(tmp_path)
    panda_xml = repo / "menagerie" / "franka_emika_panda" / "panda.xml"
    panda_xml.unlink()
    if replacement == "directory":
        panda_xml.mkdir()

    with pytest.raises(AssetProvenanceError):
        collect_asset_provenance(repo, environ={})


def test_collect_asset_provenance_requires_tracked_scene_and_panda_xml(
    tmp_path: Path,
) -> None:
    repo = _new_asset_repository(tmp_path)
    _git(
        repo,
        "rm",
        "--cached",
        "--quiet",
        "menagerie/franka_emika_panda/scene.xml",
    )

    with pytest.raises(AssetProvenanceError, match="scene.xml"):
        collect_asset_provenance(repo, environ={})


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(
            b"menagerie/VENDORED_REVISION\0"
            b"menagerie/franka_emika_panda/../escape.xml\0",
            id="path-escape",
        ),
        pytest.param(
            b"menagerie/VENDORED_REVISION\0menagerie/VENDORED_REVISION\0",
            id="duplicate",
        ),
    ],
)
def test_git_tracked_path_parser_rejects_escape_and_duplicate_paths(
    output: bytes,
) -> None:
    with pytest.raises(AssetProvenanceError):
        _parse_git_tracked_paths(output)


def test_real_vendored_panda_assets_pass_the_provenance_smoke() -> None:
    provenance = collect_asset_provenance(PROJECT_ROOT, environ={})

    assert provenance.canonical_root == "menagerie/franka_emika_panda"
    assert provenance.revision.revision == "da76818e269b82289eba39808e2fb91d679d6994"
    assert len(provenance.files) == 81
    assert {
        "menagerie/VENDORED_REVISION",
        "menagerie/franka_emika_panda/panda.xml",
        "menagerie/franka_emika_panda/scene.xml",
    } <= {fingerprint.path for fingerprint in provenance.files}


def test_evaluate_rejects_external_assets_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setenv("MENAGERIE", str(tmp_path / "external-menagerie"))

    with pytest.raises(AssetProvenanceError, match="MENAGERIE"):
        evaluate(
            seed_start=0,
            num_seeds=1,
            required_successes=0,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
