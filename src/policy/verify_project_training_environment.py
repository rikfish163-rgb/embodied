"""Verify the locked editable project training environment and CUDA smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
from pathlib import Path
from typing import Any

from . import canonical_json


EXPECTED_BASE_VERSIONS = {
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
}
EXPECTED_LOCKED_VERSIONS = {
    "safetensors": "0.8.0",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
}
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": passed, **evidence}


def audit_dependency_contract(
    pyproject: str | os.PathLike[str], lock: str | os.PathLike[str]
) -> dict[str, Any]:
    """Verify exact direct pins, indexes, package discovery, and lock artifacts."""

    pyproject_path = Path(pyproject).resolve(strict=True)
    lock_path = Path(lock).resolve(strict=True)
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    locked = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    train_group = project.get("dependency-groups", {}).get("train")
    expected_group = [
        "torch==2.13.0",
        "torchvision==0.28.0",
        "safetensors==0.8.0",
    ]
    if train_group != expected_group:
        errors.append("train_group_mismatch")
    if project.get("build-system", {}).get("requires") != ["setuptools==84.0.0"]:
        errors.append("build_backend_not_exact")
    discovery = (
        project.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
    )
    if discovery.get("include") != [
        "data*",
        "env*",
        "evaluation*",
        "expert*",
        "policy*",
        "robotics*",
    ] or discovery.get("exclude") != ["robotics.tests*"]:
        errors.append("package_discovery_mismatch")
    sources = project.get("tool", {}).get("uv", {}).get("sources")
    if sources != {
        "torch": {"index": "pytorch-cu130"},
        "torchvision": {"index": "pytorch-cu130"},
    }:
        errors.append("pytorch_source_mismatch")
    indexes = project.get("tool", {}).get("uv", {}).get("index")
    if indexes != [{"name": "pytorch-cu130", "url": PYTORCH_INDEX, "explicit": True}]:
        errors.append("pytorch_index_mismatch")

    packages_raw = locked.get("package", [])
    if not isinstance(packages_raw, list):
        raise ValueError("uv.lock package table is invalid")
    package_names = [package.get("name") for package in packages_raw]
    if len(set(package_names)) != len(package_names):
        errors.append("duplicate_lock_package")
    packages = {package["name"]: package for package in packages_raw}
    locked_versions: dict[str, str | None] = {}
    for name, expected_version in EXPECTED_LOCKED_VERSIONS.items():
        package = packages.get(name)
        if not isinstance(package, dict):
            errors.append(f"missing_lock_package:{name}")
            locked_versions[name] = None
            continue
        locked_versions[name] = package.get("version")
        if package.get("version") != expected_version:
            errors.append(f"locked_version_mismatch:{name}")
        expected_registry = (
            PYTORCH_INDEX
            if name in {"torch", "torchvision"}
            else "https://pypi.org/simple"
        )
        if package.get("source") != {"registry": expected_registry}:
            errors.append(f"locked_source_mismatch:{name}")
        artifacts = list(package.get("wheels", []))
        if package.get("sdist") is not None:
            artifacts.append(package["sdist"])
        if not artifacts:
            errors.append(f"missing_lock_artifacts:{name}")
        for artifact in artifacts:
            if not isinstance(artifact.get("url"), str) or not artifact[
                "url"
            ].startswith("https://"):
                errors.append(f"invalid_artifact_url:{name}")
            if not isinstance(artifact.get("hash"), str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", artifact["hash"]
            ):
                errors.append(f"invalid_artifact_hash:{name}")

    root = packages.get("panda-reactive-il", {})
    root_train = root.get("metadata", {}).get("requires-dev", {}).get("train")
    expected_root_train = [
        {"name": "safetensors", "specifier": "==0.8.0"},
        {"name": "torch", "specifier": "==2.13.0", "index": PYTORCH_INDEX},
        {
            "name": "torchvision",
            "specifier": "==0.28.0",
            "index": PYTORCH_INDEX,
        },
    ]
    if root_train != expected_root_train:
        errors.append("root_lock_train_group_mismatch")
    return {
        "passed": not errors,
        "direct_pins": dict(EXPECTED_BASE_VERSIONS),
        "locked_versions": locked_versions,
        "pytorch_index": PYTORCH_INDEX,
        "pyproject_sha256": _sha256_file(pyproject_path),
        "uv_lock_sha256": _sha256_file(lock_path),
        "errors": errors,
    }


def classify_failure(message: str, return_code: int | None = None) -> str:
    lowered = message.lower()
    if return_code is not None and return_code >= 128:
        return "resource"
    if re.search(
        r"no space left|enospc|disk quota|input/output error|i/o error|out of memory|oom",
        lowered,
    ):
        return "resource"
    if re.search(
        r"timed? out|temporary failure in name resolution|could not resolve|connection reset|network",
        lowered,
    ):
        return "network"
    return "compatibility"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def inspect_formal_source_state(repo: Path) -> dict[str, Any]:
    """Audit a closed Python/runtime scope without following unsafe file types."""

    repo = repo.resolve(strict=True)
    source_paths = {
        "scripts/check_training_environment.py",
        "pyproject.toml",
        "uv.lock",
    }
    unsafe_paths: list[str] = []
    for root_relative in ("src", "scripts"):
        root = repo / root_relative
        if not root.is_dir() or root.is_symlink():
            unsafe_paths.append(root_relative)
            continue
        for directory, subdirectories, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            retained_subdirectories: list[str] = []
            for name in subdirectories:
                candidate = directory_path / name
                relative = candidate.relative_to(repo).as_posix()
                mode = candidate.lstat().st_mode
                if name == "__pycache__":
                    continue
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    unsafe_paths.append(relative)
                else:
                    retained_subdirectories.append(name)
            subdirectories[:] = retained_subdirectories
            for name in filenames:
                candidate = directory_path / name
                if candidate.suffix != ".py":
                    continue
                relative = candidate.relative_to(repo).as_posix()
                source_paths.add(relative)
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    unsafe_paths.append(relative)
    for startup_hook in ("sitecustomize.py", "usercustomize.py"):
        candidate = repo / startup_hook
        if candidate.exists() or candidate.is_symlink():
            source_paths.add(startup_hook)
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                unsafe_paths.append(startup_hook)

    entries: list[dict[str, Any]] = []
    clean = not unsafe_paths
    for relative in sorted(source_paths):
        working = repo / relative
        tracked = (
            _git(repo, "ls-files", "--error-unmatch", "--", relative).returncode == 0
        )
        head = _git(repo, "show", f"HEAD:{relative}") if tracked else None
        if working.is_symlink() or not working.is_file():
            current_hash = None
        else:
            current_hash = _sha256_file(working)
        head_hash = (
            hashlib.sha256(head.stdout).hexdigest()
            if head and head.returncode == 0
            else None
        )
        identical = tracked and current_hash is not None and current_hash == head_hash
        clean = clean and identical
        entries.append(
            {
                "path": relative,
                "tracked": tracked,
                "current_sha256": current_hash,
                "head_sha256": head_hash,
                "head_identical": identical,
            }
        )
    return {
        "passed": clean,
        "closed_scope": ["src/**/*.py", "scripts/**/*.py", "root startup hooks"],
        "unsafe_paths": sorted(set(unsafe_paths)),
        "entries": entries,
    }


def _installed_package_evidence() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    missing: list[str] = []
    passed = True
    for name, expected in EXPECTED_LOCKED_VERSIONS.items():
        try:
            distribution = importlib.metadata.distribution(name)
            version = distribution.version
            record = distribution.read_text("RECORD")
            packages[name] = {
                "version": version,
                "expected_version": expected,
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest()
                if record is not None
                else None,
                "installer": distribution.read_text("INSTALLER").strip()
                if distribution.read_text("INSTALLER")
                else None,
            }
            passed = passed and version == expected and record is not None
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            passed = False
    return {"passed": passed, "packages": packages, "missing": missing}


def _editable_evidence(expected_root: Path) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("panda-reactive-il")
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            return _check(False, reason="direct_url_missing")
        direct_url = json.loads(direct_url_text)
        parsed = urllib.parse.urlparse(direct_url.get("url", ""))
        actual_root = Path(urllib.parse.unquote(parsed.path)).resolve()
        editable = direct_url.get("dir_info", {}).get("editable") is True
        return _check(
            parsed.scheme == "file" and editable and actual_root == expected_root,
            editable=editable,
            actual_root=str(actual_root),
            expected_root=str(expected_root),
            direct_url_sha256=hashlib.sha256(
                direct_url_text.encode("utf-8")
            ).hexdigest(),
        )
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as exc:
        return _check(False, reason=type(exc).__name__)


def run_cuda_smoke(tmpdir: Path) -> dict[str, Any]:
    """Run sm_120 bf16, dual-view ResNet18, and cross-process safetensors checks."""

    try:
        import torch
        import torchvision
        from safetensors.torch import save_file

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(device)
        capability = tuple(torch.cuda.get_device_capability(device))
        if not re.search(r"RTX\s*5070", name, flags=re.IGNORECASE):
            raise RuntimeError(f"unexpected GPU: {name}")
        if capability != (12, 0):
            raise RuntimeError(f"unexpected compute capability: {capability}")
        if torch.version.cuda != "13.0":
            raise RuntimeError(f"unexpected CUDA runtime: {torch.version.cuda}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is not supported")

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 16),
        ).to(device=device, dtype=torch.bfloat16)
        inputs = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
        outputs = model(inputs)
        loss = outputs.float().square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        if not gradients or any(gradient is None for gradient in gradients):
            raise RuntimeError("bf16 backward produced missing gradients")
        if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError("bf16 backward produced non-finite gradients")

        backbone = torchvision.models.resnet18(weights=None).to(
            device=device, dtype=torch.bfloat16
        )
        backbone.eval()
        two_views = torch.randn(2, 3, 128, 128, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            logits = backbone(two_views)
        if tuple(logits.shape) != (2, 1000) or not torch.isfinite(logits).all().item():
            raise RuntimeError("ResNet18 dual-view forward failed")

        tmpdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=tmpdir, prefix="m3-safetensors-"
        ) as directory:
            tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
            expected_tensor_hash = hashlib.sha256(
                tensor.contiguous().numpy().tobytes()
            ).hexdigest()
            tensor_path = Path(directory) / "probe.safetensors"
            save_file({"probe": tensor}, tensor_path)
            file_hash = _sha256_file(tensor_path)
            child_code = (
                "import hashlib,json,sys;"
                "from safetensors.torch import load_file;"
                "t=load_file(sys.argv[1],device='cpu')['probe'];"
                "print(json.dumps({'shape':list(t.shape),'dtype':str(t.dtype),"
                "'tensor_sha256':hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()},"
                "sort_keys=True,separators=(',',':')))"
            )
            child_env = os.environ.copy()
            child_env.pop("PYTHONPATH", None)
            child_env.pop("VIRTUAL_ENV", None)
            completed = subprocess.run(
                [sys.executable, "-c", child_code, str(tensor_path)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                timeout=60,
            )
            if completed.returncode != 0:
                raise RuntimeError("safetensors child load failed")
            loaded = json.loads(completed.stdout)
            if loaded != {
                "shape": [2, 3, 4],
                "dtype": "torch.float32",
                "tensor_sha256": expected_tensor_hash,
            }:
                raise RuntimeError("safetensors child content mismatch")

        torch.cuda.synchronize(device)
        return _check(
            True,
            device_name=name,
            compute_capability=list(capability),
            torch_cuda=torch.version.cuda,
            bf16_forward_shape=list(outputs.shape),
            bf16_backward_gradient_count=len(gradients),
            resnet18_weights=None,
            resnet18_input_shape=list(two_views.shape),
            resnet18_output_shape=list(logits.shape),
            safetensors_file_sha256=file_hash,
            safetensors_tensor_sha256=expected_tensor_hash,
            safetensors_shape=[2, 3, 4],
            safetensors_new_process=True,
        )
    except Exception as exc:
        return _check(
            False,
            failure_class=classify_failure(str(exc)),
            error_type=type(exc).__name__,
            reason=str(exc),
        )


def verify_project_training_environment(
    *,
    repo_root: str | os.PathLike[str],
    expected_prefix: str | os.PathLike[str],
    expected_policy_root: str | os.PathLike[str],
    expected_editable_root: str | os.PathLike[str],
    pyproject: str | os.PathLike[str],
    lock: str | os.PathLike[str],
    require_cuda_smoke: bool,
    formal: bool,
    expected_pyproject_sha256: str | None = None,
    expected_lock_sha256: str | None = None,
    data_min_free_gib: int = 15,
    root_min_free_gib: int = 3,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    prefix = Path(expected_prefix)
    policy_root = Path(expected_policy_root).resolve(strict=True)
    editable_root = Path(expected_editable_root).resolve(strict=True)
    pyproject_path = Path(pyproject).resolve(strict=True)
    lock_path = Path(lock).resolve(strict=True)
    dependency_audit = audit_dependency_contract(pyproject_path, lock_path)
    pyproject_sha256 = dependency_audit["pyproject_sha256"]
    lock_sha256 = dependency_audit["uv_lock_sha256"]
    checks: dict[str, Any] = {}

    checks["absolute_prefix"] = _check(prefix.is_absolute(), prefix=str(prefix))
    resolved_prefix = prefix.resolve(strict=True)
    checks["interpreter_prefix"] = _check(
        Path(sys.executable).resolve() == (resolved_prefix / "bin/python").resolve()
        and Path(sys.prefix).resolve() == resolved_prefix,
        executable=str(Path(sys.executable).resolve()),
        sys_prefix=str(Path(sys.prefix).resolve()),
        expected_prefix=str(resolved_prefix),
    )
    checks["pythonpath_unset"] = _check("PYTHONPATH" not in os.environ)
    checks["virtual_env_unset"] = _check("VIRTUAL_ENV" not in os.environ)
    checks["data_cache_environment"] = _check(
        os.environ.get("UV_CACHE_DIR") == str(repo / "cache/uv")
        and os.environ.get("TMPDIR") == str(repo / "cache/tmp"),
        uv_cache_dir=os.environ.get("UV_CACHE_DIR"),
        tmpdir=os.environ.get("TMPDIR"),
    )
    checks["dependency_contract"] = dependency_audit
    checks["expected_input_hashes"] = _check(
        (
            expected_pyproject_sha256 is None
            or pyproject_sha256 == expected_pyproject_sha256
        )
        and (expected_lock_sha256 is None or lock_sha256 == expected_lock_sha256),
        pyproject_sha256=pyproject_sha256,
        expected_pyproject_sha256=expected_pyproject_sha256,
        uv_lock_sha256=lock_sha256,
        expected_uv_lock_sha256=expected_lock_sha256,
    )
    checks["installed_packages"] = _installed_package_evidence()
    checks["editable_distribution"] = _editable_evidence(editable_root)
    try:
        policy = importlib.import_module("policy")
        policy_file = Path(policy.__file__).resolve()
        policy_import_passed = policy_file.is_relative_to(policy_root)
    except (ImportError, OSError, TypeError):
        policy_file = None
        policy_import_passed = False
    checks["policy_import"] = _check(
        policy_import_passed,
        policy_file=str(policy_file) if policy_file is not None else None,
        expected_policy_root=str(policy_root),
    )

    data_usage = os.statvfs(repo)
    root_usage = os.statvfs("/")
    data_free = data_usage.f_bavail * data_usage.f_frsize
    root_free = root_usage.f_bavail * root_usage.f_frsize
    checks["free_space"] = _check(
        data_free >= data_min_free_gib * 1024**3
        and root_free >= root_min_free_gib * 1024**3,
        data_available_bytes=data_free,
        root_available_bytes=root_free,
        data_min_free_gib=data_min_free_gib,
        root_min_free_gib=root_min_free_gib,
    )
    checks["formal_source_head_identical"] = (
        inspect_formal_source_state(repo) if formal else _check(True, not_applicable=True)
    )
    checks["cuda_smoke"] = (
        run_cuda_smoke(Path(os.environ.get("TMPDIR", "")))
        if require_cuda_smoke
        else _check(True, skipped=True)
    )

    required_checks = [
        "absolute_prefix",
        "interpreter_prefix",
        "pythonpath_unset",
        "virtual_env_unset",
        "data_cache_environment",
        "dependency_contract",
        "expected_input_hashes",
        "installed_packages",
        "editable_distribution",
        "policy_import",
        "free_space",
    ]
    if formal:
        required_checks.append("formal_source_head_identical")
    if require_cuda_smoke:
        required_checks.append("cuda_smoke")
    passed = all(checks[name]["passed"] for name in required_checks)
    blocked_checks = [name for name in required_checks if not checks[name]["passed"]]
    if formal:
        return {
            "schema_version": "project-train-env-receipt.v1",
            "formal": True,
            "non_acceptance": not passed,
            "status": "passed" if passed else "blocked",
            "passed": True if passed else None,
            "failure_class": None if passed else "compatibility",
            "reason_code": None if passed else "required_check_failed",
            "blocked_checks": blocked_checks,
            "checks": checks,
        }
    return {
        "schema_version": "project-train-env-development-smoke.v1",
        "formal": False,
        "non_acceptance": True,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "failure_class": None if passed else "compatibility",
        "reason_code": None if passed else "required_check_failed",
        "blocked_checks": blocked_checks,
        "checks": checks,
    }


def publish_result_no_clobber(
    output: str | os.PathLike[str], evidence: dict[str, Any]
) -> dict[str, Any]:
    identity_field = (
        "receipt_id"
        if evidence.get("schema_version") == "project-train-env-receipt.v1"
        else "smoke_id"
    )
    document = canonical_json.materialize_identity(evidence, identity_field)
    canonical_json.publish_canonical_no_clobber(output, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-prefix", required=True, type=Path)
    parser.add_argument("--pyproject", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--required-group", choices=["train"], default="train")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--expected-policy-root", required=True, type=Path)
    parser.add_argument("--expected-editable-root", required=True, type=Path)
    parser.add_argument("--require-pythonpath-unset", action="store_true")
    parser.add_argument("--expected-pyproject-sha256")
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--data-min-free-gib", type=int, default=15)
    parser.add_argument("--root-min-free-gib", type=int, default=3)
    parser.add_argument("--mode", choices=["formal", "development"], default="formal")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sync-stdout", type=Path)
    parser.add_argument("--sync-stderr", type=Path)
    parser.add_argument("--sync-check-stdout", type=Path)
    parser.add_argument("--sync-check-stderr", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = (args.repo_root or args.pyproject.parent).resolve(strict=True)
    try:
        evidence = verify_project_training_environment(
            repo_root=repo,
            expected_prefix=args.expected_prefix,
            expected_policy_root=args.expected_policy_root,
            expected_editable_root=args.expected_editable_root,
            pyproject=args.pyproject,
            lock=args.lock,
            require_cuda_smoke=args.cuda_smoke,
            formal=args.mode == "formal",
            expected_pyproject_sha256=args.expected_pyproject_sha256,
            expected_lock_sha256=args.expected_lock_sha256,
            data_min_free_gib=args.data_min_free_gib,
            root_min_free_gib=args.root_min_free_gib,
        )
        for label in (
            "sync_stdout",
            "sync_stderr",
            "sync_check_stdout",
            "sync_check_stderr",
        ):
            path = getattr(args, label)
            if path is not None:
                evidence.setdefault("logs", {})[label] = {
                    "path": str(path),
                    "sha256": _sha256_file(path) if path.is_file() else None,
                }
        document = publish_result_no_clobber(args.output, evidence)
        print(canonical_json.canonical_bytes(document).decode("utf-8"))
        return 0 if document["status"] == "passed" else 3
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"project-environment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
