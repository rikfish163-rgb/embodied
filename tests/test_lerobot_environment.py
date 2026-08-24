from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = PROJECT_ROOT / "scripts" / "verify_lerobot_environment.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_lerobot_environment", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requirement_input_accepts_only_the_frozen_package_anchors(tmp_path: Path) -> None:
    verifier = _load_verifier()
    requirements = tmp_path / "requirements.lerobot-act.in"
    requirements.write_text(
        "\n".join(
            [
                "lerobot[training] @ git+https://github.com/huggingface/lerobot.git@"
                "7e241bd630a3719a56157a497ce5d08f244784f1",
                "torch==2.11.0+cu128",
                "torchvision==0.26.0+cu128",
                "numpy==2.2.6",
                "h5py==3.16.0",
                "mujoco==3.11.0",
                "imageio==2.37.4",
                "safetensors==0.8.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert verifier.read_requirement_anchors(requirements) == verifier.REQUIRED_ANCHORS

    requirements.write_text("Python==3.12.3\n", encoding="utf-8")
    with pytest.raises(verifier.ContractError, match="package anchors"):
        verifier.read_requirement_anchors(requirements)


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [
        pytest.param(
            [],
            {
                "status": "continue",
                "passed": None,
                "next_microbatch": 4,
                "next_accumulation_steps": 1,
            },
            id="start-at-four",
        ),
        pytest.param(
            [{"microbatch": 4, "result": "oom", "evidence_complete": True}],
            {
                "status": "continue",
                "passed": None,
                "next_microbatch": 2,
                "next_accumulation_steps": 2,
            },
            id="four-oom-continues-at-two",
        ),
        pytest.param(
            [
                {"microbatch": 4, "result": "oom", "evidence_complete": True},
                {"microbatch": 2, "result": "oom", "evidence_complete": True},
                {"microbatch": 1, "result": "passed", "evidence_complete": True},
            ],
            {
                "status": "passed",
                "passed": True,
                "selected_microbatch": 1,
                "gradient_accumulation_steps": 4,
                "effective_batch": 4,
            },
            id="first-complete-lower-batch-passes",
        ),
        pytest.param(
            [
                {"microbatch": 4, "result": "oom", "evidence_complete": True},
                {"microbatch": 2, "result": "oom", "evidence_complete": True},
                {"microbatch": 1, "result": "oom", "evidence_complete": True},
            ],
            {
                "status": "failed",
                "passed": False,
                "failure_class": "cuda_oom_batch1",
            },
            id="batch-one-oom-is-compatible-failure",
        ),
        pytest.param(
            [
                {
                    "microbatch": 4,
                    "result": "compatibility_failed",
                    "failure_class": "non_finite",
                    "evidence_complete": True,
                }
            ],
            {
                "status": "failed",
                "passed": False,
                "failure_class": "non_finite",
            },
            id="non-oom-does-not-descend",
        ),
        pytest.param(
            [{"microbatch": 4, "result": "unknown", "evidence_complete": True}],
            {
                "status": "blocked",
                "passed": None,
                "failure_class": "unknown_failure",
            },
            id="unknown-is-blocked",
        ),
        pytest.param(
            [{"microbatch": 4, "result": "oom", "evidence_complete": False}],
            {
                "status": "blocked",
                "passed": None,
                "failure_class": "incomplete_evidence",
            },
            id="partial-evidence-is-blocked",
        ),
    ],
)
def test_cuda_ladder_only_descends_after_complete_oom_evidence(
    attempts: list[dict[str, object]],
    expected: dict[str, object],
) -> None:
    verifier = _load_verifier()

    assert verifier.reconcile_cuda_attempts(attempts) == expected


def test_cuda_ladder_rejects_skipped_or_repeated_microbatches() -> None:
    verifier = _load_verifier()

    with pytest.raises(verifier.ContractError, match="4, 2, 1"):
        verifier.reconcile_cuda_attempts(
            [{"microbatch": 2, "result": "oom", "evidence_complete": True}]
        )


def test_repository_requirement_input_matches_the_frozen_contract() -> None:
    verifier = _load_verifier()

    assert verifier.read_requirement_anchors(
        PROJECT_ROOT / "requirements.lerobot-act.in"
    ) == verifier.REQUIRED_ANCHORS


def test_isolate_environment_removes_project_activation_and_routes_growth_to_data(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    inherited = {
        "HOME": "/home/example",
        "PATH": "/untrusted/bin",
        "PYTHONPATH": "/project/src",
        "VIRTUAL_ENV": "/project/venv",
    }

    environment = verifier.build_isolate_environment(tmp_path, inherited)

    assert environment["PATH"] == (
        f"{tmp_path}/.venv-lerobot/bin:/usr/local/bin:/usr/bin:/bin"
    )
    assert "VIRTUAL_ENV" not in environment
    assert environment["PYTHONPATH"] == f"{tmp_path}/src"
    assert environment["UV_CACHE_DIR"] == f"{tmp_path}/cache/uv-lerobot"
    assert environment["PIP_CACHE_DIR"] == f"{tmp_path}/cache/pip-lerobot"
    assert environment["TMPDIR"] == f"{tmp_path}/cache/tmp-lerobot"
    assert environment["XDG_CACHE_HOME"] == f"{tmp_path}/cache/xdg-lerobot"
    assert environment["HF_HOME"] == f"{tmp_path}/hf"
    assert environment["TORCH_HOME"] == f"{tmp_path}/cache/torch"
    assert environment["CUDA_CACHE_PATH"] == f"{tmp_path}/cache/cuda-lerobot"
    assert environment["CCACHE_DIR"] == f"{tmp_path}/cache/ccache-lerobot"
    assert environment["PYTHONPYCACHEPREFIX"] == f"{tmp_path}/cache/pycache-lerobot"
    assert environment["UV_PYTHON_DOWNLOADS"] == "never"


@pytest.mark.parametrize(
    ("phase", "data_free_kib", "root_free_kib", "expected"),
    [
        pytest.param("initial", 20 * 1024**2, 3 * 1024**2, None, id="initial-boundary"),
        pytest.param(
            "post-create", 15 * 1024**2, 3 * 1024**2, None, id="steady-boundary"
        ),
        pytest.param(
            "initial",
            20 * 1024**2 - 1,
            3 * 1024**2,
            "insufficient_data_space",
            id="initial-data-low",
        ),
        pytest.param(
            "post-sync",
            15 * 1024**2,
            3 * 1024**2 - 1,
            "insufficient_root_space",
            id="root-low",
        ),
    ],
)
def test_resource_gate_uses_the_frozen_pre_and_post_thresholds(
    phase: str,
    data_free_kib: int,
    root_free_kib: int,
    expected: str | None,
) -> None:
    verifier = _load_verifier()

    assert verifier.resource_blocker(phase, data_free_kib, root_free_kib) == expected


def test_compiled_lock_pins_and_hashes_the_complete_cu128_resolution() -> None:
    verifier = _load_verifier()

    summary = verifier.validate_compiled_lock(
        PROJECT_ROOT / "requirements.lerobot-act.lock.txt"
    )

    assert summary["key_requirements"] == {
        "h5py": "3.16.0",
        "imageio": "2.37.4",
        "mujoco": "3.11.0",
        "numpy": "2.2.6",
        "safetensors": "0.8.0",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
    }
    assert summary["lerobot_commit"] == verifier.LEROBOT_COMMIT
    assert summary["package_count"] == 103
    assert summary["all_registry_requirements_hashed"] is True


def test_compiled_lock_rejects_wrong_cuda_source_or_missing_distribution_hash(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    lock_text = (PROJECT_ROOT / "requirements.lerobot-act.lock.txt").read_text(
        encoding="utf-8"
    )
    wrong_source = tmp_path / "wrong-source.txt"
    wrong_source.write_text(
        lock_text.replace(
            "# from https://download.pytorch.org/whl/cu128",
            "# from https://pypi.org/simple",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.ContractError, match="cu128 index"):
        verifier.validate_compiled_lock(wrong_source)

    missing_hash = tmp_path / "missing-hash.txt"
    missing_hash.write_text(
        re.sub(
            r"(?ms)^zipp==4\.1\.0 \\\n(?:    --hash=sha256:[0-9a-f]+(?: \\\n|\n))+",
            "zipp==4.1.0\n",
            lock_text,
            count=1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.ContractError, match="hash"):
        verifier.validate_compiled_lock(missing_hash)


def test_known_vulnerable_lock_is_a_hard_dependency_security_blocker() -> None:
    verifier = _load_verifier()

    assert verifier.known_dependency_security_gate(
        PROJECT_ROOT / "requirements.lerobot-act.lock.txt"
    ) == {
        "status": "blocked",
        "passed": None,
        "reason_code": "dependency_security",
        "issues": [
            {
                "package": "datasets",
                "installed": "4.8.5",
                "safe_floor": "5.0.1",
                "advisories": ["PYSEC-2026-3716"],
            },
            {
                "package": "setuptools",
                "installed": "81.0.0",
                "safe_floor": "83.0.0",
                "advisories": ["GHSA-h35f-9h28-mq5c", "PYSEC-2026-3447"],
            },
            {
                "package": "torch",
                "installed": "2.11.0+cu128",
                "safe_floor": "2.13.0",
                "advisories": ["GHSA-rrmf-rvhw-rf47"],
            },
        ],
        "environment_creation_allowed": False,
        "smoke_allowed": False,
    }


def _source_manifest(*, clean: bool = True) -> dict[str, object]:
    return {
        "schema_version": "lerobot-source-input-manifest.v1",
        "manifest_id": "sha256:" + "1" * 64,
        "git_commit": "2" * 40,
        "head_tree": "3" * 40,
        "scope_config_sha256": "4" * 64,
        "path_set_sha256": "5" * 64,
        "index_entries_sha256": "6" * 64,
        "content_entries_sha256": "7" * 64,
        "tracked_status_sha256": "8" * 64,
        "source_provenance_clean": clean,
        "entries": [
            {
                "path": "requirements.lerobot-act.in",
                "mode": "100644",
                "index_blob_oid": "9" * 40,
                "head_blob_oid": "9" * 40,
                "size_bytes": 123,
                "content_sha256": "a" * 64,
            }
        ],
    }


def test_source_lineage_requires_an_exact_clean_start_to_terminal_recheck() -> None:
    verifier = _load_verifier()
    start = _source_manifest()

    assert verifier.source_lineage_gate(start, dict(start)) == {
        "formal_ready": True,
        "non_acceptance": False,
        "source_input_manifest_id": start["manifest_id"],
        "source_commit": start["git_commit"],
    }

    drifted = dict(start)
    drifted["content_entries_sha256"] = "b" * 64
    assert verifier.source_lineage_gate(start, drifted) == {
        "formal_ready": False,
        "non_acceptance": True,
        "blocker": "source_input_manifest_drift",
    }

    dirty = _source_manifest(clean=False)
    assert verifier.source_lineage_gate(dirty, dict(dirty)) == {
        "formal_ready": False,
        "non_acceptance": True,
        "blocker": "source_provenance_not_clean",
    }


def test_terminal_decision_cannot_promote_dirty_or_partial_development_smoke() -> None:
    verifier = _load_verifier()
    clean_lineage = verifier.source_lineage_gate(
        _source_manifest(), _source_manifest()
    )
    dirty_lineage = verifier.source_lineage_gate(
        _source_manifest(clean=False), _source_manifest(clean=False)
    )

    assert verifier.terminal_decision(
        requested_status="passed",
        failure_class=None,
        evidence_complete=True,
        resource_blocker_code=None,
        source_lineage=dirty_lineage,
    ) == {
        "status": "blocked",
        "passed": None,
        "reason_code": "source_provenance_not_clean",
        "non_acceptance": True,
    }
    assert verifier.terminal_decision(
        requested_status="failed",
        failure_class="non_finite",
        evidence_complete=False,
        resource_blocker_code=None,
        source_lineage=clean_lineage,
    )["status"] == "blocked"
    assert verifier.terminal_decision(
        requested_status="failed",
        failure_class="mystery",
        evidence_complete=True,
        resource_blocker_code=None,
        source_lineage=clean_lineage,
    )["status"] == "blocked"
    assert verifier.terminal_decision(
        requested_status="failed",
        failure_class="non_finite",
        evidence_complete=True,
        resource_blocker_code=None,
        source_lineage=clean_lineage,
    ) == {
        "status": "failed",
        "passed": False,
        "failure_class": "non_finite",
        "non_acceptance": False,
    }


def test_act_config_spec_matches_the_frozen_official_backend_contract() -> None:
    verifier = _load_verifier()

    assert verifier.act_config_spec(device="cpu") == {
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": (3, 128, 128)},
            "observation.images.wrist": {"type": "VISUAL", "shape": (3, 128, 128)},
            "observation.state": {"type": "STATE", "shape": (8,)},
        },
        "output_features": {"action": {"type": "ACTION", "shape": (8,)}},
        "n_obs_steps": 1,
        "chunk_size": 16,
        "n_action_steps": 1,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "use_vae": True,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "dropout": 0.1,
        "kl_weight": 10.0,
        "temporal_ensemble_coeff": None,
        "optimizer_lr": 1e-5,
        "optimizer_lr_backbone": 1e-5,
        "optimizer_weight_decay": 1e-4,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "IDENTITY",
            "ACTION": "IDENTITY",
        },
        "device": "cpu",
        "use_amp": False,
        "push_to_hub": False,
    }

    cuda_spec = verifier.act_config_spec(device="cuda")
    assert cuda_spec["device"] == "cuda"
    assert cuda_spec["use_amp"] is True


def test_synthetic_batch_contract_uses_only_allowed_training_keys() -> None:
    torch = pytest.importorskip("torch")
    verifier = _load_verifier()

    batch = verifier.make_synthetic_batch(torch, batch_size=2, device="cpu")

    assert set(batch) == {
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
        "action",
        "action_is_pad",
    }
    assert batch["observation.images.front"].shape == (2, 3, 128, 128)
    assert batch["observation.images.wrist"].shape == (2, 3, 128, 128)
    assert batch["observation.state"].shape == (2, 8)
    assert batch["action"].shape == (2, 16, 8)
    assert batch["action_is_pad"].shape == (2, 16)
    assert batch["action_is_pad"].dtype == torch.bool
    assert all(torch.isfinite(value).all() for value in batch.values())


def test_runtime_identity_requires_the_unique_prefix_and_isolated_modules(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    prefix = tmp_path / ".venv-lerobot"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python executable fixture")
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    module_paths = {
        name: site_packages / name / "__init__.py"
        for name in (
            "lerobot",
            "torch",
            "torchvision",
            "numpy",
            "h5py",
            "mujoco",
            "imageio",
            "safetensors",
        )
    }
    environment = verifier.build_isolate_environment(tmp_path, {})

    identity = verifier.verify_runtime_identity(
        repository=tmp_path,
        expected_prefix=prefix,
        executable=executable,
        sys_prefix=prefix,
        python_version=(3, 12, 3),
        environment=environment,
        module_paths=module_paths,
    )

    assert identity["python_version"] == "3.12.3"
    assert identity["prefix"] == str(prefix.resolve())
    assert identity["executable_sha256"] == (
        "sha256:" + verifier.sha256_file(executable)
    )

    module_paths["torch"] = tmp_path / "outside" / "torch" / "__init__.py"
    with pytest.raises(verifier.ContractError, match="torch.*prefix"):
        verifier.verify_runtime_identity(
            repository=tmp_path,
            expected_prefix=prefix,
            executable=executable,
            sys_prefix=prefix,
            python_version=(3, 12, 3),
            environment=environment,
            module_paths=module_paths,
        )


def test_policy_exercise_runs_the_complete_train_and_predict_chain() -> None:
    torch = pytest.importorskip("torch")
    verifier = _load_verifier()

    class TinyChunkPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, batch):
            prediction = self.scale * torch.ones_like(batch["action"])
            loss = torch.nn.functional.l1_loss(prediction, batch["action"])
            return loss, {"l1_loss": float(loss.detach()), "kld_loss": 0.0}

        def predict_action_chunk(self, batch):
            return self.scale * torch.ones(
                batch["observation.state"].shape[0], 16, 8
            )

    policy = TinyChunkPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    batch = verifier.make_synthetic_batch(torch, batch_size=2, device="cpu")
    initial_scale = policy.scale.detach().clone()

    evidence = verifier.exercise_policy(
        torch,
        policy=policy,
        optimizer=optimizer,
        batch=batch,
        device="cpu",
        accumulation_steps=2,
    )

    assert evidence["optimizer_step_completed"] is True
    assert evidence["prediction_shape"] == [2, 16, 8]
    assert evidence["loss_finite"] is True
    assert evidence["loss_dict_finite"] is True
    assert evidence["gradients_finite"] is True
    assert not torch.equal(policy.scale.detach(), initial_scale)


def test_policy_exercise_rejects_nonfinite_loss_as_compatibility_failure() -> None:
    torch = pytest.importorskip("torch")
    verifier = _load_verifier()

    class NonFinitePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            loss = self.weight * torch.tensor(float("nan"))
            return loss, {"l1_loss": float("nan")}

        def predict_action_chunk(self, batch):
            raise AssertionError("predict must not run after a non-finite loss")

    policy = NonFinitePolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    batch = verifier.make_synthetic_batch(torch, batch_size=1, device="cpu")

    with pytest.raises(verifier.CompatibilityFailure) as error:
        verifier.exercise_policy(
            torch,
            policy=policy,
            optimizer=optimizer,
            batch=batch,
            device="cpu",
            accumulation_steps=1,
        )
    assert error.value.failure_class == "non_finite"


def test_dependency_audit_cli_blocks_before_environment_creation_and_is_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dependency-security-evidence.json"
    command = [
        "/usr/bin/python3.12",
        str(VERIFIER_PATH),
        "audit-lock",
        "--requirements",
        str(PROJECT_ROOT / "requirements.lerobot-act.in"),
        "--lock",
        str(PROJECT_ROOT / "requirements.lerobot-act.lock.txt"),
        "--output",
        str(output),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "lerobot-dependency-security-evidence.v1"
    assert payload["status"] == "blocked"
    assert payload["passed"] is None
    assert payload["lock_classification"] == "development-blocked"
    assert payload["environment_creation_allowed"] is False
    assert payload["smoke_allowed"] is False
    assert payload["requirements_sha256"] == verifier_hash(
        PROJECT_ROOT / "requirements.lerobot-act.in"
    )
    assert payload["lock_sha256"] == verifier_hash(
        PROJECT_ROOT / "requirements.lerobot-act.lock.txt"
    )

    original = output.read_bytes()
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert output.read_bytes() == original


def verifier_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_blocked_receipt_records_hashed_evidence_without_enabling_fallback(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    dependency_evidence = tmp_path / "dependency-security-evidence.json"
    verifier.audit_dependency_lock(
        requirements=PROJECT_ROOT / "requirements.lerobot-act.in",
        lock=PROJECT_ROOT / "requirements.lerobot-act.lock.txt",
        output=dependency_evidence,
    )
    supporting = tmp_path / "osv-query.stdout.json"
    supporting.write_text('{"results":[]}\n', encoding="utf-8")
    prefix = tmp_path / ".venv-lerobot"
    draft_path = tmp_path / "receipt.draft.json"

    draft = verifier.build_dependency_blocked_receipt(
        repository=tmp_path,
        expected_prefix=prefix,
        dependency_evidence=dependency_evidence,
        supporting_evidence=[supporting],
        data_free_kib=20 * 1024**2,
        root_free_kib=3 * 1024**2,
        output=draft_path,
    )

    assert draft["schema_version"] == "lerobot-smoke-receipt.v1"
    assert draft["status"] == "blocked"
    assert draft["passed"] is None
    assert draft["reason_code"] == "dependency_security"
    assert draft["failure_stage"] == "dependency_security_audit"
    assert draft["fallback_allowed"] is False
    assert draft["backend_selected"] is None
    assert draft["non_acceptance"] is True
    assert draft["environment"]["expected_prefix"] == str(prefix.resolve())
    assert draft["environment"]["prefix_exists"] is False
    assert draft["environment"]["creation_attempted"] is False
    assert draft["environment"]["sync_attempted"] is False
    assert draft["environment"]["smoke_attempted"] is False
    assert draft["source_lineage"]["complete"] is False
    assert draft["evidence"] == [
        {
            "path": "osv-query.stdout.json",
            "sha256": verifier_hash(supporting),
            "size_bytes": supporting.stat().st_size,
        }
    ]


def test_resource_blocked_receipt_is_terminal_before_any_environment_attempt(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    output = tmp_path / "resource-blocked.draft.json"
    supporting = tmp_path / "security-floor-resolver.stderr.log"
    supporting.write_text("resolver floor is unsatisfiable\n", encoding="utf-8")

    draft = verifier.build_preflight_blocked_receipt(
        repository=tmp_path,
        expected_prefix=tmp_path / ".venv-lerobot",
        failure_stage="initial_space_preflight",
        reason_code="insufficient_root_space",
        data_free_kib=20 * 1024**2,
        root_free_kib=3 * 1024**2 - 1,
        supporting_evidence=[supporting],
        output=output,
    )

    assert draft["status"] == "blocked"
    assert draft["passed"] is None
    assert draft["failure_stage"] == "initial_space_preflight"
    assert draft["reason_code"] == "insufficient_root_space"
    assert draft["fallback_allowed"] is False
    assert draft["environment"] == {
        "expected_prefix": str((tmp_path / ".venv-lerobot").resolve()),
        "prefix_exists": False,
        "creation_attempted": False,
        "sync_attempted": False,
        "smoke_attempted": False,
    }
    assert draft["resources"]["root_free_kib"] == 3 * 1024**2 - 1
    assert draft["evidence"] == [
        {
            "path": supporting.name,
            "sha256": verifier_hash(supporting),
            "size_bytes": supporting.stat().st_size,
        }
    ]


def test_bootstrap_launcher_publishes_a_canonical_receipt_on_low_space() -> None:
    bootstrap = PROJECT_ROOT / "scripts" / "bootstrap_lerobot_act.sh"
    # /dev/shm is intentionally smaller than the frozen 20 GiB data preflight.
    # It gives this negative launcher test a real df result without a test-only
    # override in the production launcher.
    if not Path("/dev/shm").is_dir():
        pytest.skip("/dev/shm is unavailable")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        repository = Path(directory) / "repo"
        (repository / "scripts").mkdir(parents=True)
        (repository / "src" / "policy").mkdir(parents=True)
        (repository / "README.md").write_text("fixture\n", encoding="utf-8")
        (repository / ".gitignore").write_text(
            "/.venv-lerobot/\n/cache/\nhf/\nruns/\n", encoding="utf-8"
        )
        shutil.copy2(PROJECT_ROOT / "requirements.lerobot-act.in", repository)
        shutil.copy2(PROJECT_ROOT / "requirements.lerobot-act.lock.txt", repository)
        shutil.copy2(VERIFIER_PATH, repository / "scripts")
        shutil.copy2(
            PROJECT_ROOT / "src" / "policy" / "canonical_json.py",
            repository / "src" / "policy",
        )
        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        run_dir = repository / "runs" / "m3" / "resource-blocked"
        completed = subprocess.run(
            [
                "bash",
                str(bootstrap),
                "audit",
                "--repo-root",
                str(repository),
                "--run-dir",
                str(run_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 3, completed.stderr
        assert not (repository / ".venv-lerobot").exists()
        receipt = run_dir / "lerobot_smoke_receipt.json"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["status"] == "blocked"
        assert payload["passed"] is None
        assert payload["reason_code"] == "insufficient_data_space"
        assert payload["fallback_allowed"] is False
        assert payload["environment"]["creation_attempted"] is False
        assert subprocess.run(
            [
                "/usr/bin/python3.12",
                str(repository / "src" / "policy" / "canonical_json.py"),
                "validate-id",
                "--input",
                str(receipt),
                "--identity-field",
                "receipt_id",
                "--require-schema",
                "lerobot-smoke-receipt.v1",
            ],
            check=False,
        ).returncode == 0


@pytest.mark.parametrize("mode", ["audit", "sync", "smoke"])
def test_bootstrap_launcher_publishes_blocked_receipt_before_creating_the_venv(
    mode: str,
) -> None:
    bootstrap = PROJECT_ROOT / "scripts" / "bootstrap_lerobot_act.sh"
    data_tmp = PROJECT_ROOT / "cache" / "tmp-lerobot"
    data_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_tmp) as directory:
        repository = Path(directory) / "repo"
        (repository / "scripts").mkdir(parents=True)
        (repository / "src" / "policy").mkdir(parents=True)
        (repository / "runs" / "m3").mkdir(parents=True)
        (repository / "README.md").write_text("fixture\n", encoding="utf-8")
        (repository / ".gitignore").write_text(
            "/.venv-lerobot/\n/cache/\nhf/\nruns/\n", encoding="utf-8"
        )
        shutil.copy2(PROJECT_ROOT / "requirements.lerobot-act.in", repository)
        shutil.copy2(PROJECT_ROOT / "requirements.lerobot-act.lock.txt", repository)
        shutil.copy2(VERIFIER_PATH, repository / "scripts")
        shutil.copy2(
            PROJECT_ROOT / "src" / "policy" / "canonical_json.py",
            repository / "src" / "policy",
        )
        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        run_dir = repository / "runs" / "m3" / f"security-blocked-{mode}"
        command = [
            "bash",
            str(bootstrap),
            mode,
            "--repo-root",
            str(repository),
            "--run-dir",
            str(run_dir),
        ]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        assert completed.returncode == 3, completed.stderr
        assert not (repository / ".venv-lerobot").exists()
        receipt = run_dir / "lerobot_smoke_receipt.json"
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["status"] == "blocked"
        assert payload["passed"] is None
        assert payload["reason_code"] == "dependency_security"
        assert payload["fallback_allowed"] is False
        assert payload["environment"]["creation_attempted"] is False
        assert (run_dir / "dependency-audit.stdout.log").is_file()
        assert (run_dir / "dependency-audit.stderr.log").is_file()
        assert (run_dir / "dependency-audit.logs.sha256").is_file()
        assert subprocess.run(
            [
                "/usr/bin/python3.12",
                str(repository / "src" / "policy" / "canonical_json.py"),
                "validate-id",
                "--input",
                str(receipt),
                "--identity-field",
                "receipt_id",
                "--require-schema",
                "lerobot-smoke-receipt.v1",
            ],
            check=False,
        ).returncode == 0

        original = receipt.read_bytes()
        repeated = subprocess.run(command, check=False, capture_output=True, text=True)
        assert repeated.returncode == 2
        assert receipt.read_bytes() == original
