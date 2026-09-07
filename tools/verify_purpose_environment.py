"""Validate and clean-install committed purpose-specific D08 environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "12-6.purpose-environment-profile.v1"
INDEX_SCHEMA = "12-6.purpose-environment-index.v1"
SPECS_SCHEMA = "12-6.purpose-environment-specs.v1"
EVIDENCE_SCHEMA = "12-6.purpose-environment-evidence.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@/\\]+(?: --hash=sha256:[0-9a-f]{64})+$"
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")


class PurposeEnvironmentError(RuntimeError):
    """Raised when a purpose environment is stale, ambiguous, or not reproducible."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PurposeEnvironmentError(f"cannot read JSON {path}") from exc
    if not isinstance(value, dict):
        raise PurposeEnvironmentError(f"JSON {path} must contain an object")
    return value


def _canonical_name(name: str) -> str:
    value = _NAME_NORMALIZER.sub("-", name.strip()).lower()
    if not value:
        raise PurposeEnvironmentError("distribution name must not be empty")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PurposeEnvironmentError(f"unsafe profile path: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PurposeEnvironmentError(f"profile path escapes root: {relative}") from exc
    return resolved


def _validate_self_hash(document: dict[str, Any], field: str, label: str) -> None:
    claimed = document.get(field)
    if _SHA256.fullmatch(str(claimed or "")) is None:
        raise PurposeEnvironmentError(f"{label} has invalid {field}")
    payload = dict(document)
    payload.pop(field, None)
    if _sha256_bytes(_canonical_bytes(payload)) != claimed:
        raise PurposeEnvironmentError(f"{label} self-hash mismatch")


def _validate_lock(path: Path, expected_count: int) -> dict[str, str]:
    packages: dict[str, str] = {}
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _LOCK_LINE.fullmatch(line) is None:
            raise PurposeEnvironmentError(f"non-exact or unhashed lock line {path}:{number}")
        name, tail = line.split("==", 1)
        version = tail.split(" ", 1)[0]
        canonical = _canonical_name(name)
        if canonical in packages:
            raise PurposeEnvironmentError(f"duplicate locked distribution {canonical} in {path}")
        packages[canonical] = version
        count += 1
    if count != expected_count:
        raise PurposeEnvironmentError(
            f"lock package count mismatch for {path}: manifest={expected_count} actual={count}"
        )
    return packages


def _validate_canonical_index(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / "requirements/locks/index.json"
    index = _load_json(path)
    if index.get("schema_version") != "12-6.dependency-lock-index.v1":
        raise PurposeEnvironmentError("canonical lock index schema mismatch")
    _validate_self_hash(index, "index_sha256", "canonical lock index")
    return index, {
        "path": "requirements/locks/index.json",
        "file_sha256": _sha256_file(path),
        "index_sha256": str(index["index_sha256"]),
    }


def _validate_canonical_profile(
    root: Path, index: dict[str, Any], reference: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    profile_id = str(reference.get("profile_id", ""))
    record = index.get("profiles", {}).get(profile_id)
    if not isinstance(record, dict):
        raise PurposeEnvironmentError(f"canonical base profile {profile_id!r} is missing")
    expected_ref = {
        "profile_id": profile_id,
        "path": str(record.get("path", "")),
        "file_sha256": str(record.get("sha256", "")),
        "manifest_sha256": str(record.get("manifest_sha256", "")),
    }
    if reference != expected_ref:
        raise PurposeEnvironmentError("purpose profile canonical-base reference drift")
    path = _safe_path(root, expected_ref["path"])
    if _sha256_file(path) != expected_ref["file_sha256"]:
        raise PurposeEnvironmentError("canonical base profile file hash mismatch")
    profile = _load_json(path)
    if profile.get("manifest_sha256") != expected_ref["manifest_sha256"]:
        raise PurposeEnvironmentError("canonical base profile semantic hash mismatch")
    locks = profile.get("locks")
    if not isinstance(locks, dict):
        raise PurposeEnvironmentError("canonical base profile lacks locks")
    validated: dict[str, dict[str, Any]] = {}
    for group, lock_record in locks.items():
        if not isinstance(lock_record, dict):
            raise PurposeEnvironmentError(f"malformed canonical {group} lock record")
        lock_path = _safe_path(root, str(lock_record.get("path", "")))
        if _sha256_file(lock_path) != lock_record.get("sha256"):
            raise PurposeEnvironmentError(f"canonical {group} lock hash mismatch")
        packages = _validate_lock(lock_path, int(lock_record.get("package_count", -1)))
        validated[group] = {"path": lock_path, "packages": packages}
    return profile, validated


def validate_registry(root: Path, profile_id: str) -> dict[str, Any]:
    root = root.resolve()
    specs = _load_json(root / "requirements/profiles/specs.json")
    if specs.get("schema_version") != SPECS_SCHEMA:
        raise PurposeEnvironmentError("purpose environment specs schema mismatch")
    expected_ids = set(specs.get("profiles", {}))
    if not expected_ids:
        raise PurposeEnvironmentError("purpose environment specs are empty")

    canonical_index, canonical_ref = _validate_canonical_index(root)
    index_path = root / "requirements/profiles/index.json"
    index = _load_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise PurposeEnvironmentError("purpose environment index schema mismatch")
    _validate_self_hash(index, "index_sha256", "purpose environment index")
    if index.get("canonical_lock") != canonical_ref:
        raise PurposeEnvironmentError("purpose index canonical-lock identity drift")
    records = index.get("profiles")
    if not isinstance(records, dict) or set(records) != expected_ids:
        raise PurposeEnvironmentError("purpose environment index profile set mismatch")
    if profile_id not in records:
        raise PurposeEnvironmentError(f"unknown purpose environment profile: {profile_id}")

    record = records[profile_id]
    if not isinstance(record, dict):
        raise PurposeEnvironmentError("purpose environment profile record is malformed")
    profile_path = _safe_path(root, str(record.get("path", "")))
    if _sha256_file(profile_path) != record.get("sha256"):
        raise PurposeEnvironmentError("purpose environment profile file hash mismatch")
    profile = _load_json(profile_path)
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise PurposeEnvironmentError("purpose environment profile schema mismatch")
    _validate_self_hash(profile, "profile_sha256", f"purpose profile {profile_id}")
    if profile.get("profile_sha256") != record.get("profile_sha256"):
        raise PurposeEnvironmentError("purpose environment profile semantic hash mismatch")
    if profile.get("profile_id") != profile_id:
        raise PurposeEnvironmentError("purpose environment profile identity mismatch")
    if profile.get("canonical_lock") != canonical_ref:
        raise PurposeEnvironmentError("purpose profile canonical-lock identity drift")

    spec = specs["profiles"].get(profile_id)
    if not isinstance(spec, dict):
        raise PurposeEnvironmentError("purpose environment spec is missing")
    if profile.get("kind") != spec.get("kind") or profile.get("purpose") != spec.get("purpose"):
        raise PurposeEnvironmentError("purpose environment spec/profile semantic drift")
    if profile.get("python", {}).get("version") != spec.get("python_version"):
        raise PurposeEnvironmentError("purpose environment Python version drift")

    locks = profile.get("locks")
    if not isinstance(locks, dict):
        raise PurposeEnvironmentError("purpose environment profile locks must be an object")
    purpose_locks: dict[str, dict[str, Any]] = {}
    for group, lock_record in locks.items():
        if not isinstance(lock_record, dict):
            raise PurposeEnvironmentError(f"purpose {group} lock record is malformed")
        lock_path = _safe_path(root, str(lock_record.get("path", "")))
        if _sha256_file(lock_path) != lock_record.get("sha256"):
            raise PurposeEnvironmentError(f"purpose {group} lock hash mismatch")
        packages = _validate_lock(lock_path, int(lock_record.get("package_count", -1)))
        purpose_locks[group] = {"path": lock_path, "packages": packages}

    base_profile: dict[str, Any] | None = None
    base_locks: dict[str, dict[str, Any]] = {}
    if profile.get("kind") in {"linux-overlay", "linux-base-role"}:
        reference = profile.get("base_profile")
        if not isinstance(reference, dict):
            raise PurposeEnvironmentError("Linux purpose profile lacks canonical base reference")
        base_profile, base_locks = _validate_canonical_profile(root, canonical_index, reference)
    elif profile.get("kind") == "windows-runtime":
        reference = profile.get("version_source_profile")
        if not isinstance(reference, dict):
            raise PurposeEnvironmentError("Windows purpose profile lacks version-source reference")
        _validate_canonical_profile(root, canonical_index, reference)
    else:
        raise PurposeEnvironmentError(f"unsupported purpose profile kind: {profile.get('kind')}")

    return {
        "root": root,
        "specs": specs,
        "index": index,
        "index_path": index_path,
        "canonical_index": canonical_index,
        "canonical_ref": canonical_ref,
        "record": record,
        "profile": profile,
        "profile_path": profile_path,
        "purpose_locks": purpose_locks,
        "base_profile": base_profile,
        "base_locks": base_locks,
    }


def _normalized_machine() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    return machine


def _assert_platform(profile: dict[str, Any]) -> None:
    expected = profile.get("platform")
    if not isinstance(expected, dict):
        raise PurposeEnvironmentError("profile platform contract is missing")
    if platform.system() != expected.get("system") or _normalized_machine() != expected.get("machine"):
        raise PurposeEnvironmentError(
            f"profile platform mismatch: expected {expected}, got {platform.system()}/{_normalized_machine()}"
        )
    expected_python = profile.get("python", {})
    actual_python = platform.python_version()
    if sys_implementation() != expected_python.get("implementation") or actual_python != expected_python.get("version"):
        raise PurposeEnvironmentError(
            f"profile Python mismatch: expected {expected_python}, got {sys_implementation()} {actual_python}"
        )


def sys_implementation() -> str:
    import sys

    return sys.implementation.name.lower()


def _venv_python(directory: Path) -> Path:
    if platform.system() == "Windows":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _run(command: list[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def _output(command: list[str | Path], *, cwd: Path) -> str:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _install_lock(python: Path, lock_path: Path, *, cwd: Path) -> None:
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--require-hashes",
            "--no-deps",
            "-r",
            lock_path,
        ],
        cwd=cwd,
    )


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONUTF8": "1",
            "SOURCE_DATE_EPOCH": "315532800",
        }
    )
    return env


def _build_project_wheel(python: Path, root: Path, wheel_dir: Path) -> Path:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            wheel_dir,
            root,
        ],
        cwd=root,
        env=_offline_env(),
    )
    wheels = sorted(wheel_dir.glob("twelve_six_ai-*.whl"))
    if len(wheels) != 1:
        raise PurposeEnvironmentError(f"expected one project wheel, found {len(wheels)}")
    return wheels[0]


def _installed_distributions(python: Path, *, cwd: Path) -> list[dict[str, str]]:
    code = (
        "import importlib.metadata as m,json; "
        "rows=sorted((d.metadata.get('Name') or '',d.version) for d in m.distributions()); "
        "print(json.dumps([{'name':n,'version':v} for n,v in rows if n],sort_keys=True))"
    )
    value = json.loads(_output([python, "-c", code], cwd=cwd))
    if not isinstance(value, list):
        raise PurposeEnvironmentError("installed distribution inventory must be a list")
    return value


def _probe_code(profile_id: str) -> str:
    base = [
        "import importlib.metadata as m, json, torch",
        "import twelve_six",
        "result={'project_version':m.version('twelve-six-ai'),'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}",
    ]
    if profile_id == "linux-x86_64-tokenizer-experiment":
        base.extend(
            [
                "import tokenizers",
                "from tokenizers import Tokenizer, models",
                "Tokenizer(models.BPE())",
                "result['tokenizers_version']=tokenizers.__version__",
            ]
        )
    elif profile_id == "linux-x86_64-transformers-interop":
        base.extend(
            [
                "import transformers",
                "from transformers import LlamaConfig, LlamaForCausalLM",
                "cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=16)",
                "model=LlamaForCausalLM(cfg).eval()",
                "out=model(input_ids=torch.tensor([[1,2,3]]))",
                "assert tuple(out.logits.shape)==(1,3,32)",
                "result['transformers_version']=transformers.__version__",
                "result['llama_random_init_forward']='PASS'",
            ]
        )
    elif profile_id == "linux-x86_64-cuda-training":
        base.extend(
            [
                "result['gpu_execution']='NOT_RUN_NO_GPU'",
                "if result['cuda_available']:",
                "    x=torch.ones(1,device='cuda')+1",
                "    assert float(x.cpu().item())==2.0",
                "    result['gpu_execution']='PASS'",
            ]
        )
    elif profile_id == "windows-x86_64-runtime":
        base.extend(
            [
                "import numpy, safetensors",
                "import twelve_six.checkpoint",
                "import twelve_six.inference.first_party",
                "result['numpy_version']=m.version('numpy')",
                "result['safetensors_version']=m.version('safetensors')",
                "result['checkpoint_import']='PASS'",
                "result['first_party_inference_import']='PASS'",
            ]
        )
    else:
        raise PurposeEnvironmentError(f"no runtime probe for {profile_id}")
    base.append("print(json.dumps(result,sort_keys=True))")
    return "\n".join(base)


def verify_install(
    *,
    root: Path,
    profile_id: str,
    source_sha: str,
    evidence_out: Path | None,
    project_wheel: Path | None,
) -> dict[str, Any]:
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise PurposeEnvironmentError("source SHA must be a full lowercase 40-character Git SHA")
    state = validate_registry(root, profile_id)
    profile = state["profile"]
    _assert_platform(profile)

    with tempfile.TemporaryDirectory(prefix="twelve-six-purpose-env-") as temp_name:
        temp = Path(temp_name)
        environment = temp / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _venv_python(environment)
        if not python.exists():
            raise PurposeEnvironmentError("virtualenv Python is missing")

        if profile["kind"] in {"linux-overlay", "linux-base-role"}:
            base_locks = state["base_locks"]
            _install_lock(python, base_locks["toolchain"]["path"], cwd=state["root"])
            _install_lock(python, base_locks["runtime"]["path"], cwd=state["root"])
            overlay = state["purpose_locks"].get("overlay")
            if overlay is not None:
                _install_lock(python, overlay["path"], cwd=state["root"])
        elif profile["kind"] == "windows-runtime":
            _install_lock(python, state["purpose_locks"]["toolchain"]["path"], cwd=state["root"])
            _install_lock(python, state["purpose_locks"]["runtime"]["path"], cwd=state["root"])
        else:
            raise PurposeEnvironmentError("unsupported purpose profile kind")

        pip_version = _output([python, "-m", "pip", "--version"], cwd=state["root"])
        if not pip_version.startswith("pip 26.2.1 "):
            raise PurposeEnvironmentError(f"locked pip version mismatch: {pip_version}")

        if project_wheel is None:
            if not (state["root"] / "pyproject.toml").is_file():
                raise PurposeEnvironmentError("project source is unavailable; --project-wheel is required")
            wheel = _build_project_wheel(python, state["root"], temp / "dist")
        else:
            wheel = project_wheel.resolve()
            if not wheel.is_file() or wheel.suffix != ".whl":
                raise PurposeEnvironmentError("--project-wheel must name an existing wheel")

        _run(
            [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", wheel],
            cwd=state["root"],
            env=_offline_env(),
        )
        probe = json.loads(_output([python, "-c", _probe_code(profile_id)], cwd=state["root"]))
        if not isinstance(probe, dict):
            raise PurposeEnvironmentError("runtime probe did not emit an object")

        expected_direct = {
            item.split("==", 1)[0].replace("_", "-").lower(): item.split("==", 1)[1]
            for item in profile.get("direct_requirements", [])
            if "==" in item
        }
        if profile_id == "linux-x86_64-tokenizer-experiment" and probe.get("tokenizers_version") != expected_direct.get("tokenizers"):
            raise PurposeEnvironmentError("tokenizers runtime version mismatch")
        if profile_id == "linux-x86_64-transformers-interop" and probe.get("transformers_version") != expected_direct.get("transformers"):
            raise PurposeEnvironmentError("Transformers runtime version mismatch")
        if profile_id == "linux-x86_64-cuda-training":
            expectations = profile.get("runtime_expectations", {})
            if not str(probe.get("torch_version", "")).startswith(str(expectations.get("torch", ""))):
                raise PurposeEnvironmentError("CUDA-role Torch version mismatch")
            if probe.get("torch_cuda") != expectations.get("torch_cuda"):
                raise PurposeEnvironmentError("CUDA-role Torch CUDA build mismatch")
        if profile_id == "windows-x86_64-runtime":
            direct_versions = profile.get("direct_versions", {})
            for name in ("numpy", "safetensors"):
                if probe.get(f"{name}_version") != direct_versions.get(name):
                    raise PurposeEnvironmentError(f"Windows {name} version mismatch")
            if not str(probe.get("torch_version", "")).startswith(str(direct_versions.get("torch", ""))):
                raise PurposeEnvironmentError("Windows Torch version mismatch")

        installed = _installed_distributions(python, cwd=state["root"])
        evidence: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA,
            "authority": "EXACT_HASH_LOCKED_PURPOSE_ENVIRONMENT",
            "source_sha": source_sha,
            "profile_id": profile_id,
            "kind": profile["kind"],
            "purpose": profile["purpose"],
            "python": profile["python"],
            "platform": profile["platform"],
            "purpose_index": {
                "path": "requirements/profiles/index.json",
                "file_sha256": _sha256_file(state["index_path"]),
                "index_sha256": state["index"]["index_sha256"],
            },
            "profile": {
                "path": state["record"]["path"],
                "file_sha256": state["record"]["sha256"],
                "profile_sha256": profile["profile_sha256"],
            },
            "canonical_lock": state["canonical_ref"],
            "project_wheel": {"filename": wheel.name, "sha256": _sha256_file(wheel)},
            "installed_distributions": installed,
            "installed_distributions_sha256": _sha256_bytes(_canonical_bytes(installed)),
            "runtime_probe": probe,
            "verification": {
                "registry_validation": "PASS",
                "exact_hash_install": "PASS",
                "project_wheel_install": "PASS",
                "runtime_probe": "PASS",
                "floating_resolution_in_authority_path": False,
            },
        }
        evidence["evidence_sha256"] = _sha256_bytes(_canonical_bytes(evidence))
        if evidence_out is not None:
            destination = evidence_out if evidence_out.is_absolute() else state["root"] / evidence_out
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-sha")
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--project-wheel", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    state = validate_registry(args.root, args.profile)
    if args.validate_only:
        print(f"profile={args.profile}")
        print(f"profile_sha256={state['profile']['profile_sha256']}")
        print(f"purpose_index_sha256={state['index']['index_sha256']}")
        return 0
    if args.source_sha is None:
        parser.error("--source-sha is required unless --validate-only is used")
    evidence = verify_install(
        root=args.root,
        profile_id=args.profile,
        source_sha=args.source_sha,
        evidence_out=args.evidence_out,
        project_wheel=args.project_wheel,
    )
    print(f"profile={evidence['profile_id']}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    print(f"gpu_execution={evidence['runtime_probe'].get('gpu_execution', 'NOT_APPLICABLE')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
