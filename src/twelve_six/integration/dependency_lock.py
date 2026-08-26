"""Fail-closed dependency lock and packaging environment contract."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.dependency-lock-profile.v1"
INDEX_SCHEMA_VERSION = "12-6.dependency-lock-index.v1"
EXACT_PYTHON_VERSION = "3.11.16"
PYTHON_IMPLEMENTATION = "cpython"
SUPPORTED_REQUIRES_PYTHON = ">=3.11,<3.12"
PROJECT_DISTRIBUTION = "twelve-six-ai"
CONSOLE_SCRIPTS = {"twelve-six-generate": "twelve_six.inference.cli:main"}
SUPPORTED_PROFILES = {"linux-x86_64", "linux-aarch64"}
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
    elif machine in {"arm64", "aarch64"}:
        machine = "aarch64"
    profile_id = f"{system}-{machine}"
    if profile_id not in SUPPORTED_PROFILES:
        raise DependencyLockError(f"unsupported lock platform: {system}/{machine}")
    return profile_id


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
        raise DependencyLockError(f"requires-python must be exactly {SUPPORTED_REQUIRES_PYTHON!r}")
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
    if profile_id not in SUPPORTED_PROFILES:
        raise DependencyLockError(f"unsupported profile id: {profile_id}")
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
    profile_id = str(manifest.get("profile_id", "<unknown>"))
    label = f"profile {profile_id}"
    claimed = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DependencyLockError(f"{label}: unsupported dependency-lock profile schema")
    if _SHA256.fullmatch(str(claimed or "")) is None:
        raise DependencyLockError(f"{label}: invalid manifest SHA-256")
    if sha256_bytes(canonical_json_bytes(payload)) != claimed:
        raise DependencyLockError(f"{label}: manifest self-hash mismatch")
    if manifest.get("project") != PROJECT_DISTRIBUTION:
        raise DependencyLockError(f"{label}: project identity mismatch")
    if manifest.get("profile_id") not in SUPPORTED_PROFILES:
        raise DependencyLockError(f"{label}: unsupported profile identity")
    expected_python = {
        "implementation": PYTHON_IMPLEMENTATION,
        "version": EXACT_PYTHON_VERSION,
        "requires_python": SUPPORTED_REQUIRES_PYTHON,
    }
    if manifest.get("python") != expected_python:
        raise DependencyLockError(f"{label}: Python policy mismatch")
    metadata = _project_metadata(root_path / "pyproject.toml")
    pyproject_sha = sha256_file(root_path / "pyproject.toml")
    if manifest.get("pyproject_sha256") != pyproject_sha:
        raise DependencyLockError(
            f"{label}: component pyproject.toml stale: "
            f"expected={manifest.get('pyproject_sha256')} actual={pyproject_sha}"
        )
    if manifest.get("declared_requirements") != metadata:
        raise DependencyLockError(f"{label}: component pyproject requirements stale")
    if manifest.get("console_scripts") != CONSOLE_SCRIPTS:
        raise DependencyLockError(f"{label}: console-script binding mismatch")
    if enforce_current_platform and manifest.get("profile_id") != current_profile_id():
        raise DependencyLockError(f"{label}: does not match current platform")
    locks = manifest.get("locks")
    if not isinstance(locks, dict) or set(locks) != {"toolchain", "runtime", "dev"}:
        raise DependencyLockError(f"{label}: must bind toolchain/runtime/dev locks")
    for group, record in locks.items():
        if not isinstance(record, dict):
            raise DependencyLockError(f"{label}: lock record {group!r} must be an object")
        digest = record.get("sha256")
        if _SHA256.fullmatch(str(digest or "")) is None:
            raise DependencyLockError(f"{label}: invalid SHA-256 for {group} lock")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise DependencyLockError(f"{label}: missing path for {group} lock")
        lock_path = root_path / relative
        actual = sha256_file(lock_path)
        if actual != digest:
            raise DependencyLockError(
                f"{label}: component {group} stale: expected={digest} actual={actual} path={relative}"
            )
        count = record.get("package_count")
        if not isinstance(count, int) or count < 0:
            raise DependencyLockError(f"{label}: invalid package count for {group} lock")
    return manifest


def build_lock_index(*, root: str | Path, manifests: Mapping[str, str | Path]) -> dict[str, Any]:
    root_path = Path(root)
    if set(manifests) != SUPPORTED_PROFILES:
        raise DependencyLockError("lock index must bind every supported profile")
    profiles: dict[str, dict[str, str]] = {}
    for profile_id, relative in sorted(manifests.items()):
        manifest = validate_profile_manifest(
            root=root_path, manifest_path=relative, enforce_current_platform=False
        )
        if manifest["profile_id"] != profile_id:
            raise DependencyLockError(f"profile {profile_id}: index key/id mismatch")
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


def _selected_profiles(profile_ids: Iterable[str] | None) -> tuple[str, ...]:
    if profile_ids is None:
        return tuple(sorted(SUPPORTED_PROFILES))
    selected = tuple(sorted(set(profile_ids)))
    if not selected:
        raise DependencyLockError("dependency-lock validation scope must not be empty")
    unknown = set(selected) - SUPPORTED_PROFILES
    if unknown:
        raise DependencyLockError(f"unknown dependency-lock profiles: {sorted(unknown)}")
    return selected


def validate_lock_index(
    *,
    root: str | Path,
    index_path: str | Path,
    profile_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate aggregate metadata and either all or an explicit profile subset.

    The aggregate document itself always remains fail-closed. ``profile_ids=None`` is
    the global/release check and validates every profile. A narrow experiment passes
    its exact profile id so unrelated optional/other-platform profile bytes cannot
    prevent that experiment from reaching its own exact-lock validation.
    """

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
    if not isinstance(profiles, dict) or set(profiles) != SUPPORTED_PROFILES:
        raise DependencyLockError("dependency-lock index profile set mismatch")
    for profile_id in _selected_profiles(profile_ids):
        record = profiles[profile_id]
        if not isinstance(record, dict):
            raise DependencyLockError(f"profile {profile_id}: aggregate record must be an object")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise DependencyLockError(f"profile {profile_id}: aggregate path is invalid")
        actual_file_sha = sha256_file(root_path / relative)
        if actual_file_sha != record.get("sha256"):
            raise DependencyLockError(
                f"profile {profile_id}: aggregate profile bytes stale: "
                f"expected={record.get('sha256')} actual={actual_file_sha} path={relative}"
            )
        manifest = validate_profile_manifest(
            root=root_path, manifest_path=relative, enforce_current_platform=False
        )
        if manifest["profile_id"] != profile_id:
            raise DependencyLockError(f"profile {profile_id}: aggregate identity mismatch")
        if manifest["manifest_sha256"] != record.get("manifest_sha256"):
            raise DependencyLockError(f"profile {profile_id}: aggregate semantic hash stale")
    return index
