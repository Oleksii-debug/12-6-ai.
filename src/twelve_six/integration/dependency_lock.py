"""Fail-closed dependency lock and packaging environment contract."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.dependency-lock-profile.v1"
INDEX_SCHEMA_VERSION = "12-6.dependency-lock-index.v1"
EXACT_PYTHON_VERSION = "3.11.16"
PYTHON_IMPLEMENTATION = "cpython"
SUPPORTED_REQUIRES_PYTHON = ">=3.11,<3.12"
PROJECT_DISTRIBUTION = "twelve-six-ai"
CONSOLE_SCRIPTS = {"twelve-six-generate": "twelve_six.inference.cli:main"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME_NORMALIZER = re.compile(r"[-_.]+")


class DependencyLockError(ValueError):
    """Raised when lock evidence is missing, stale, ambiguous, or tampered."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_distribution_name(name: str) -> str:
    value = _NAME_NORMALIZER.sub("-", name.strip()).lower()
    if not value:
        raise DependencyLockError("distribution name must not be empty")
    return value


def current_profile_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    if system == "linux" and machine == "x86_64":
        return "linux-x86_64"
    if system == "windows" and machine == "x86_64":
        return "windows-x86_64"
    raise DependencyLockError(f"unsupported lock platform: {system}/{machine}")


def assert_exact_python() -> None:
    actual = platform.python_version()
    implementation = sys.implementation.name.lower()
    if implementation != PYTHON_IMPLEMENTATION or actual != EXACT_PYTHON_VERSION:
        raise DependencyLockError(
            "dependency lock requires "
            f"{PYTHON_IMPLEMENTATION} {EXACT_PYTHON_VERSION}; got {implementation} {actual}"
        )


def _project_metadata(pyproject_path: Path) -> dict[str, Any]:
    document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise DependencyLockError("pyproject [project] table is missing")
    if canonical_distribution_name(str(project.get("name", ""))) != PROJECT_DISTRIBUTION:
        raise DependencyLockError("unexpected project distribution name")
    if project.get("requires-python") != SUPPORTED_REQUIRES_PYTHON:
        raise DependencyLockError(
            f"requires-python must be exactly {SUPPORTED_REQUIRES_PYTHON!r}"
        )
    scripts = project.get("scripts") or {}
    if scripts != CONSOLE_SCRIPTS:
        raise DependencyLockError("console-script metadata drift")
    dependencies = project.get("dependencies") or []
    optional = project.get("optional-dependencies") or {}
    dev = optional.get("dev", []) if isinstance(optional, dict) else []
    if not isinstance(dependencies, list) or not isinstance(dev, list):
        raise DependencyLockError("runtime/dev dependency metadata is malformed")
    return {
        "runtime_requirements": sorted(str(item) for item in dependencies),
        "dev_requirements": sorted(str(item) for item in dev),
    }


def build_profile_manifest(
    *,
    root: str | Path,
    profile_id: str,
    lock_files: Mapping[str, str | Path],
    package_counts: Mapping[str, int],
    platform_system: str | None = None,
    platform_machine: str | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    pyproject = root_path / "pyproject.toml"
    metadata = _project_metadata(pyproject)
    locks: dict[str, dict[str, Any]] = {}
    for group, relative in sorted(lock_files.items()):
        path = root_path / relative
        count = int(package_counts[group])
        if count < 0:
            raise DependencyLockError("package count cannot be negative")
        locks[group] = {
            "path": Path(relative).as_posix(),
            "sha256": sha256_file(path),
            "package_count": count,
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "project": PROJECT_DISTRIBUTION,
        "python": {
            "implementation": PYTHON_IMPLEMENTATION,
            "version": EXACT_PYTHON_VERSION,
            "requires_python": SUPPORTED_REQUIRES_PYTHON,
        },
        "platform": {
            "system": platform_system or platform.system(),
            "machine": platform_machine or platform.machine(),
        },
        "pyproject_sha256": sha256_file(pyproject),
        "declared_requirements": metadata,
        "console_scripts": CONSOLE_SCRIPTS,
        "locks": locks,
    }
    payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyLockError(f"cannot read lock JSON {path}") from exc
    if not isinstance(value, dict):
        raise DependencyLockError(f"lock JSON {path} must contain an object")
    return value


def validate_profile_manifest(
    *, root: str | Path, manifest_path: str | Path, enforce_current_platform: bool = True
) -> dict[str, Any]:
    root_path = Path(root)
    path = root_path / manifest_path
    manifest = _load_json(path)
    claimed = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DependencyLockError("unsupported dependency-lock profile schema")
    if _SHA256.fullmatch(str(claimed or "")) is None:
        raise DependencyLockError("invalid profile manifest SHA-256")
    if sha256_bytes(canonical_json_bytes(payload)) != claimed:
        raise DependencyLockError("profile manifest self-hash mismatch")
    if manifest.get("project") != PROJECT_DISTRIBUTION:
        raise DependencyLockError("profile project identity mismatch")
    python_info = manifest.get("python")
    expected_python = {
        "implementation": PYTHON_IMPLEMENTATION,
        "version": EXACT_PYTHON_VERSION,
        "requires_python": SUPPORTED_REQUIRES_PYTHON,
    }
    if python_info != expected_python:
        raise DependencyLockError("profile Python policy mismatch")
    _project_metadata(root_path / "pyproject.toml")
    if manifest.get("pyproject_sha256") != sha256_file(root_path / "pyproject.toml"):
        raise DependencyLockError("profile is stale for current pyproject.toml")
    if manifest.get("console_scripts") != CONSOLE_SCRIPTS:
        raise DependencyLockError("profile console-script binding mismatch")
    if enforce_current_platform and manifest.get("profile_id") != current_profile_id():
        raise DependencyLockError("profile does not match current platform")
    locks = manifest.get("locks")
    if not isinstance(locks, dict) or set(locks) != {"toolchain", "runtime", "dev"}:
        raise DependencyLockError("profile must bind toolchain/runtime/dev locks")
    for group, record in locks.items():
        if not isinstance(record, dict):
            raise DependencyLockError(f"lock record {group!r} must be an object")
        digest = record.get("sha256")
        if _SHA256.fullmatch(str(digest or "")) is None:
            raise DependencyLockError(f"invalid SHA-256 for {group} lock")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise DependencyLockError(f"missing path for {group} lock")
        lock_path = root_path / relative
        if sha256_file(lock_path) != digest:
            raise DependencyLockError(f"{group} lock hash mismatch")
        count = record.get("package_count")
        if not isinstance(count, int) or count < 0:
            raise DependencyLockError(f"invalid package count for {group} lock")
    return manifest


def build_lock_index(*, root: str | Path, manifests: Mapping[str, str | Path]) -> dict[str, Any]:
    root_path = Path(root)
    profiles: dict[str, dict[str, str]] = {}
    for profile_id, relative in sorted(manifests.items()):
        manifest = validate_profile_manifest(
            root=root_path, manifest_path=relative, enforce_current_platform=False
        )
        if manifest["profile_id"] != profile_id:
            raise DependencyLockError("profile index key/id mismatch")
        profiles[profile_id] = {
            "path": Path(relative).as_posix(),
            "sha256": sha256_file(root_path / relative),
            "manifest_sha256": manifest["manifest_sha256"],
        }
    payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "project": PROJECT_DISTRIBUTION,
        "python_version": EXACT_PYTHON_VERSION,
        "profiles": profiles,
    }
    payload["index_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_lock_index(*, root: str | Path, index_path: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    index = _load_json(root_path / index_path)
    claimed = index.get("index_sha256")
    payload = dict(index)
    payload.pop("index_sha256", None)
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise DependencyLockError("unsupported dependency-lock index schema")
    if _SHA256.fullmatch(str(claimed or "")) is None:
        raise DependencyLockError("invalid dependency-lock index SHA-256")
    if sha256_bytes(canonical_json_bytes(payload)) != claimed:
        raise DependencyLockError("dependency-lock index self-hash mismatch")
    if index.get("project") != PROJECT_DISTRIBUTION:
        raise DependencyLockError("dependency-lock index project mismatch")
    if index.get("python_version") != EXACT_PYTHON_VERSION:
        raise DependencyLockError("dependency-lock index Python mismatch")
    profiles = index.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"linux-x86_64", "windows-x86_64"}:
        raise DependencyLockError("dependency-lock index must bind Linux and Windows x86_64")
    for profile_id, record in profiles.items():
        if not isinstance(record, dict):
            raise DependencyLockError("dependency-lock index profile must be an object")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise DependencyLockError("dependency-lock index profile path is invalid")
        if sha256_file(root_path / relative) != record.get("sha256"):
            raise DependencyLockError("dependency-lock profile file hash mismatch")
        manifest = validate_profile_manifest(
            root=root_path, manifest_path=relative, enforce_current_platform=False
        )
        if manifest["profile_id"] != profile_id:
            raise DependencyLockError("dependency-lock profile identity mismatch")
        if manifest["manifest_sha256"] != record.get("manifest_sha256"):
            raise DependencyLockError("dependency-lock profile semantic hash mismatch")
    return index
