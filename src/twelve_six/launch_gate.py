"""Cheap, fail-closed launch gate for long learned-model experiments.

The gate deliberately checks cheap environment/tooling invariants before importing
model/runtime modules. It never instantiates model weights, creates an optimizer,
or performs training. A successful run emits a hash-signed launch envelope bound
to the exact Git SHA, launch request, environment profile and lock identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "12-6.expensive-run-launch-request.v1"
ENVELOPE_SCHEMA = "12-6.expensive-run-launch-envelope.v1"


class LaunchGateError(RuntimeError):
    """Raised when a long-run launch invariant is not satisfied."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchGateError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LaunchGateError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _repo_path(repo: Path, raw: str, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise LaunchGateError(f"{label} path is missing")
    candidate = (repo / raw).resolve()
    root = repo.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LaunchGateError(f"{label} path escapes repository: {raw}") from exc
    return candidate


def _relative(repo: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _git_head(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchGateError("cannot resolve exact source SHA") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise LaunchGateError("git HEAD is not a lowercase full 40-character SHA")
    return value


def _python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


def _check_tools(repo: Path, request: dict[str, Any]) -> list[dict[str, str]]:
    """Check pytest/Ruff/other tools before any heavyweight project import."""
    checks: list[dict[str, str]] = []
    modules = request.get("required_modules", [])
    commands = request.get("required_commands", [])
    if not isinstance(modules, list):
        raise LaunchGateError("required_modules must be a list")
    if not isinstance(commands, list):
        raise LaunchGateError("required_commands must be a list")

    for module in modules:
        if not isinstance(module, str) or not module:
            raise LaunchGateError("required_modules must contain non-empty strings")
        if importlib.util.find_spec(module) is None:
            raise LaunchGateError(f"required module unavailable: {module}")
        checks.append({"kind": "module", "name": module, "status": "PASS"})

    for command in commands:
        if not isinstance(command, str) or not command:
            raise LaunchGateError("required_commands must contain non-empty strings")
        if "/" in command or "\\" in command:
            candidate = _repo_path(repo, command, label="required command")
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise LaunchGateError(f"required command unavailable: {command}")
            resolved = str(candidate)
        else:
            found = shutil.which(command)
            if found is None:
                raise LaunchGateError(f"required command unavailable: {command}")
            resolved = found
        checks.append(
            {"kind": "command", "name": command, "path": resolved, "status": "PASS"}
        )
    return checks


def _validate_dependency_index(repo: Path, base_path: Path, base: dict[str, Any]) -> dict[str, str]:
    index_path = repo / "requirements" / "locks" / "index.json"
    index = _read_json(index_path)
    profiles = index.get("profiles")
    profile_id = base.get("profile_id")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise LaunchGateError(f"dependency lock index missing profile: {profile_id}")
    record = profiles[profile_id]
    if not isinstance(record, dict):
        raise LaunchGateError(f"invalid dependency lock index record: {profile_id}")
    if _repo_path(repo, str(record.get("path", "")), label="indexed dependency profile") != base_path:
        raise LaunchGateError("dependency lock index/profile path mismatch")
    if record.get("sha256") != _hash_file(base_path):
        raise LaunchGateError("dependency lock index/profile file hash mismatch")
    if record.get("manifest_sha256") != base.get("manifest_sha256"):
        raise LaunchGateError("dependency lock index/profile manifest mismatch")
    if index.get("python_version") != _python_version():
        raise LaunchGateError("dependency lock index Python identity mismatch")
    return {
        "path": _relative(repo, index_path),
        "file_sha256": _hash_file(index_path),
        "index_sha256": str(index.get("index_sha256")),
    }


def _validate_purpose_index(
    repo: Path,
    profile_path: Path,
    profile: dict[str, Any],
) -> dict[str, str] | None:
    try:
        profile_path.relative_to((repo / "requirements" / "profiles").resolve())
    except ValueError:
        return None
    index_path = repo / "requirements" / "profiles" / "index.json"
    index = _read_json(index_path)
    profiles = index.get("profiles")
    profile_id = profile.get("profile_id")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise LaunchGateError(f"purpose environment index missing profile: {profile_id}")
    record = profiles[profile_id]
    if not isinstance(record, dict):
        raise LaunchGateError(f"invalid purpose environment index record: {profile_id}")
    if _repo_path(repo, str(record.get("path", "")), label="indexed purpose profile") != profile_path:
        raise LaunchGateError("purpose environment index/profile path mismatch")
    if record.get("sha256") != _hash_file(profile_path):
        raise LaunchGateError("purpose environment index/profile file hash mismatch")
    if record.get("profile_sha256") != profile.get("profile_sha256"):
        raise LaunchGateError("purpose environment index/profile semantic mismatch")
    return {
        "path": _relative(repo, index_path),
        "file_sha256": _hash_file(index_path),
        "index_sha256": str(index.get("index_sha256")),
    }


def _verify_lock_profile(repo: Path, request: dict[str, Any]) -> dict[str, Any]:
    purpose = request.get("purpose_profile")
    if not isinstance(purpose, dict):
        raise LaunchGateError("purpose_profile must be an object")
    expected_id = purpose.get("profile_id")
    profile_path = _repo_path(repo, str(purpose.get("path", "")), label="purpose profile")
    profile = _read_json(profile_path)
    if profile.get("profile_id") != expected_id:
        raise LaunchGateError("purpose profile identity mismatch")

    expected_declared = purpose.get("profile_sha256")
    if expected_declared is not None and profile.get("profile_sha256") != expected_declared:
        raise LaunchGateError("purpose profile semantic SHA mismatch")

    purpose_index = _validate_purpose_index(repo, profile_path, profile)
    profile_file_sha = _hash_file(profile_path)
    base = profile
    base_path = profile_path
    base_pointer = profile.get("base_profile")
    if base_pointer is not None:
        if not isinstance(base_pointer, dict):
            raise LaunchGateError("purpose profile base_profile must be an object")
        base_path = _repo_path(repo, str(base_pointer.get("path", "")), label="base profile")
        if _hash_file(base_path) != base_pointer.get("file_sha256"):
            raise LaunchGateError("purpose/base profile file hash mismatch")
        base = _read_json(base_path)
        if base.get("profile_id") != base_pointer.get("profile_id"):
            raise LaunchGateError("purpose/base profile identity mismatch")
        if base.get("manifest_sha256") != base_pointer.get("manifest_sha256"):
            raise LaunchGateError("purpose/base profile manifest identity mismatch")

    dependency_index = _validate_dependency_index(repo, base_path, base)
    expected_python = profile.get("python", {}).get("version") or base.get("python", {}).get("version")
    if expected_python != _python_version():
        raise LaunchGateError(
            f"Python version mismatch: {_python_version()} != {expected_python}"
        )
    if platform.python_implementation().lower() != str(
        profile.get("python", {}).get("implementation", "cpython")
    ).lower():
        raise LaunchGateError("Python implementation mismatch")

    lock_result: dict[str, dict[str, str]] = {}
    locks = base.get("locks")
    if not isinstance(locks, dict) or not locks:
        raise LaunchGateError("base dependency profile has no locks")
    for role, item in sorted(locks.items()):
        if not isinstance(item, dict):
            raise LaunchGateError(f"invalid lock declaration: {role}")
        lock_path = _repo_path(repo, str(item.get("path", "")), label=f"{role} lock")
        actual = _hash_file(lock_path)
        if actual != item.get("sha256"):
            raise LaunchGateError(f"{role} lock hash mismatch")
        lock_result[str(role)] = {
            "path": _relative(repo, lock_path),
            "sha256": actual,
        }

    return {
        "profile_id": expected_id,
        "profile_path": _relative(repo, profile_path),
        "profile_file_sha256": profile_file_sha,
        "profile_semantic_sha256": profile.get("profile_sha256"),
        "purpose_index": purpose_index,
        "base_profile_id": base.get("profile_id"),
        "base_profile_path": _relative(repo, base_path),
        "base_profile_file_sha256": _hash_file(base_path),
        "base_manifest_sha256": base.get("manifest_sha256"),
        "dependency_index": dependency_index,
        "locks": lock_result,
    }


def _verify_test_selectors(repo: Path, request: dict[str, Any]) -> list[str]:
    selectors = request.get("test_selectors")
    if not isinstance(selectors, list) or not selectors:
        raise LaunchGateError("at least one expected test selector is required")
    checked: list[str] = []
    for selector in selectors:
        if not isinstance(selector, str) or not selector:
            raise LaunchGateError("test selectors must be non-empty strings")
        file_part, _, node_part = selector.partition("::")
        path = _repo_path(repo, file_part, label="test selector")
        if not path.is_file():
            raise LaunchGateError(f"test selector file missing: {file_part}")
        if node_part:
            text = path.read_text(encoding="utf-8")
            leaf = node_part.split("::")[-1].split("[")[0]
            if f"def {leaf}(" not in text and f"class {leaf}" not in text:
                raise LaunchGateError(f"test selector node missing: {selector}")
        checked.append(selector)
    return checked


def _verify_budget(request: dict[str, Any]) -> dict[str, int]:
    budget = request.get("budget")
    if not isinstance(budget, dict):
        raise LaunchGateError("budget must be an object")
    allowed = ("optimizer_steps", "target_optimized_tokens", "max_wall_minutes")
    normalized: dict[str, int] = {}
    for key in allowed:
        if key in budget:
            value = budget[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LaunchGateError(f"invalid run budget: {key} must be a positive integer")
            normalized[key] = value
    if "optimizer_steps" not in normalized and "target_optimized_tokens" not in normalized:
        raise LaunchGateError("invalid run budget: missing steps/token target")
    return normalized


def _verify_storage(repo: Path, request: dict[str, Any]) -> dict[str, Any]:
    output = _repo_path(repo, str(request.get("checkpoint_output", "")), label="checkpoint output")
    probe_root = output if output.exists() and output.is_dir() else output.parent
    probe_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe_root).free
    minimum = request.get("disk_min_free_bytes", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise LaunchGateError("disk_min_free_bytes must be a non-negative integer")
    if free < minimum:
        raise LaunchGateError(f"insufficient disk headroom: {free} < {minimum}")
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=probe_root, prefix=".launch-gate-", delete=True
        ) as handle:
            handle.write(b"12-6-launch-gate")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise LaunchGateError(f"checkpoint output is not writable: {output}") from exc
    return {
        "checkpoint_output": _relative(repo, output),
        "free_bytes": free,
        "minimum_free_bytes": minimum,
        "writable": True,
    }


def _verify_corpus(repo: Path, request: dict[str, Any]) -> dict[str, str]:
    corpus = request.get("corpus")
    if not isinstance(corpus, dict):
        raise LaunchGateError("corpus must be an object")
    manifest_path = _repo_path(repo, str(corpus.get("manifest_path", "")), label="corpus manifest")
    manifest = _read_json(manifest_path)
    identity_key = str(corpus.get("identity_key", "corpus_identity_sha256"))
    actual = manifest.get(identity_key)
    expected = corpus.get("expected_identity_sha256")
    if not isinstance(actual, str) or actual != expected:
        raise LaunchGateError("corpus identity mismatch")
    return {
        "manifest_path": _relative(repo, manifest_path),
        "manifest_file_sha256": _hash_file(manifest_path),
        "identity_key": identity_key,
        "identity_sha256": actual,
    }


def _verify_project_contracts(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Import project runtime only after cheap tooling/profile checks passed."""
    imports = request.get("critical_imports")
    if not isinstance(imports, list) or not imports:
        raise LaunchGateError("critical_imports must be a non-empty list")
    imported: list[str] = []
    for name in imports:
        if not isinstance(name, str) or not name:
            raise LaunchGateError("critical_imports must contain non-empty strings")
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - gate must fail closed on import defects
            raise LaunchGateError(f"critical import failed: {name}") from exc
        imported.append(name)

    model_cfg = request.get("model_spec")
    if not isinstance(model_cfg, dict):
        raise LaunchGateError("model_spec must be an object")
    from twelve_six.model import ModelSpec

    spec = ModelSpec.from_dict(dict(model_cfg))
    actual_spec_sha = spec.identity_sha256()
    if actual_spec_sha != request.get("model_spec_sha256"):
        raise LaunchGateError("ModelSpec semantic identity mismatch")
    params = spec.parameter_count()
    if params != request.get("parameter_count"):
        raise LaunchGateError("ModelSpec parameter count mismatch")
    model_result = {
        "model_spec_sha256": actual_spec_sha,
        "parameter_count": params,
    }

    tokenizer_req = request.get("tokenizer")
    if not isinstance(tokenizer_req, dict) or tokenizer_req.get("kind") != "byte":
        raise LaunchGateError("unsupported or missing tokenizer contract")
    from twelve_six.tokenization import ByteTokenizer

    tok = ByteTokenizer()
    expected_fields = {
        "version": tok.identity.version,
        "config_sha256": tok.identity.config_sha256,
        "vocab_sha256": tok.identity.vocab_sha256,
        "vocab_size": tok.identity.vocab_size,
    }
    for key, actual in expected_fields.items():
        if tokenizer_req.get(key) != actual:
            raise LaunchGateError(f"tokenizer identity mismatch: {key}")
    tokenizer_result = dict(expected_fields)
    tokenizer_result["special_tokens"] = dict(tok.identity.special_tokens)
    return model_result, tokenizer_result, imported


def _verify_gpu_if_required(request: dict[str, Any]) -> dict[str, Any]:
    requires_gpu = request.get("requires_gpu", False)
    if not isinstance(requires_gpu, bool):
        raise LaunchGateError("requires_gpu must be boolean")
    if not requires_gpu:
        return {"required": False, "checked": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise LaunchGateError("GPU required but torch cannot be imported") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise LaunchGateError("GPU required but no CUDA device is visible")
    return {
        "required": True,
        "checked": True,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def create_launch_envelope(repo: Path, request_path: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    request_file = _repo_path(repo, str(request_path), label="launch request")
    output_file = _repo_path(repo, str(output_path), label="launch envelope")
    request = _read_json(request_file)
    if request.get("schema") != REQUEST_SCHEMA:
        raise LaunchGateError("launch request schema mismatch")

    # Historical failure boundary: pytest/Ruff availability is checked first,
    # before importing twelve_six.model / torch or constructing a ModelSpec.
    tool_checks = _check_tools(repo, request)
    source_sha = _git_head(repo)
    profile = _verify_lock_profile(repo, request)
    selectors = _verify_test_selectors(repo, request)
    budget = _verify_budget(request)
    storage = _verify_storage(repo, request)
    corpus = _verify_corpus(repo, request)
    model, tokenizer, imports = _verify_project_contracts(request)
    gpu = _verify_gpu_if_required(request)

    unsigned = {
        "schema": ENVELOPE_SCHEMA,
        "source_sha": source_sha,
        "request_path": _relative(repo, request_file),
        "request_sha256": _hash_json(request),
        "binding": request.get("binding"),
        "python": {
            "implementation": platform.python_implementation().lower(),
            "version": _python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "purpose_profile": profile,
        "checks": {
            "tool_availability": tool_checks,
            "test_selectors": selectors,
            "budget": budget,
            "storage": storage,
            "corpus": corpus,
            "model": model,
            "tokenizer": tokenizer,
            "critical_imports": imports,
            "gpu": gpu,
        },
        "check_order": [
            "tool_availability",
            "source_sha",
            "purpose_profile_and_locks",
            "test_selectors",
            "run_budget",
            "disk_and_checkpoint_writability",
            "corpus_identity",
            "critical_imports_modelspec_tokenizer",
            "gpu_if_required",
        ],
        "training_performed": False,
    }
    envelope = dict(unsigned)
    envelope["envelope_sha256"] = _hash_json(unsigned)
    _write_json(output_file, envelope)
    return envelope


def verify_launch_envelope(
    repo: Path,
    request_path: Path,
    envelope_path: Path,
    *,
    expected_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    request_file = _repo_path(repo, str(request_path), label="launch request")
    envelope_file = _repo_path(repo, str(envelope_path), label="launch envelope")
    request = _read_json(request_file)
    envelope = _read_json(envelope_file)
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise LaunchGateError("launch envelope schema mismatch")
    unsigned = dict(envelope)
    signature = unsigned.pop("envelope_sha256", None)
    if not isinstance(signature, str) or signature != _hash_json(unsigned):
        raise LaunchGateError("launch envelope hash signature mismatch")
    if envelope.get("source_sha") != _git_head(repo):
        raise LaunchGateError("stale launch envelope: source SHA mismatch")
    if envelope.get("request_path") != _relative(repo, request_file):
        raise LaunchGateError("stale launch envelope: launch request path mismatch")
    if envelope.get("request_sha256") != _hash_json(request):
        raise LaunchGateError("stale launch envelope: launch config mismatch")
    if expected_binding is not None and envelope.get("binding") != expected_binding:
        raise LaunchGateError("launch envelope bound to another workflow/config")
    current_profile = _verify_lock_profile(repo, request)
    if current_profile != envelope.get("purpose_profile"):
        raise LaunchGateError("stale launch envelope: profile/lock identity changed")
    if envelope.get("python", {}).get("version") != _python_version():
        raise LaunchGateError("stale launch envelope: Python version changed")
    if envelope.get("training_performed") is not False:
        raise LaunchGateError("invalid launch envelope training marker")
    return envelope


def require_launch_envelope_from_env(
    repo: Path, *, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    request_raw = os.environ.get("TWELVE_SIX_LAUNCH_REQUEST")
    envelope_raw = os.environ.get("TWELVE_SIX_LAUNCH_ENVELOPE")
    if not request_raw or not envelope_raw:
        raise LaunchGateError(
            "long experiment refused: TWELVE_SIX_LAUNCH_REQUEST and "
            "TWELVE_SIX_LAUNCH_ENVELOPE are mandatory"
        )
    return verify_launch_envelope(
        repo,
        Path(request_raw),
        Path(envelope_raw),
        expected_binding=expected_binding,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    create = sub.add_parser("create")
    create.add_argument("--repo-root", type=Path, default=Path("."))
    create.add_argument("--request", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path("."))
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--envelope", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "create":
        value = create_launch_envelope(args.repo_root, args.request, args.output)
        print(
            json.dumps(
                {
                    "launch_gate": "PASS",
                    "source_sha": value["source_sha"],
                    "request_sha256": value["request_sha256"],
                    "envelope_sha256": value["envelope_sha256"],
                    "training_performed": False,
                },
                sort_keys=True,
            )
        )
    else:
        value = verify_launch_envelope(args.repo_root, args.request, args.envelope)
        print(
            json.dumps(
                {
                    "launch_envelope": "VALID",
                    "source_sha": value["source_sha"],
                    "envelope_sha256": value["envelope_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
