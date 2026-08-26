"""Safe exact-lock wheelhouse caching for CI dependency setup.

The cache is acceleration only.  Correctness remains anchored by committed purpose
profiles and exact-hash lock files; installs must still use pip --require-hashes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CACHE_SCHEMA = "12-6.hashed-dependency-cache.v1"
MANIFEST_SCHEMA = "12-6.hashed-dependency-cache-manifest.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;@/\\]+)(.*)$")
_NAME_NORMALIZER = re.compile(r"[-_.]+")
_FORBIDDEN_CPU_PREFIXES = ("nvidia-", "cuda-", "triton", "pytorch-triton")


class DependencyCacheError(RuntimeError):
    """Raised when cache identity or wheel content is not exact-lock safe."""


@dataclass(frozen=True)
class LockIdentity:
    path: str
    sha256: str
    package_count: int
    allowed_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ProfileResolution:
    profile_id: str
    profile_path: str
    profile_file_sha256: str
    profile_semantic_sha256: str | None
    support_profiles: tuple[tuple[str, str], ...]
    locks: tuple[LockIdentity, ...]
    system: str
    machine: str
    python_implementation: str
    python_version: str


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
        raise DependencyCacheError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DependencyCacheError(f"JSON must contain an object: {path}")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DependencyCacheError(f"unsafe repository-relative path: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise DependencyCacheError(f"path escapes repository root: {relative}") from exc
    return resolved


def _canonical_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name).lower()


def _normalize_machine(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"amd64", "x86_64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "aarch64"
    return value


def _parse_lock(path: Path, relative: str) -> LockIdentity:
    package_count = 0
    hashes: set[str] = set()
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            raise DependencyCacheError(f"non-exact lock line {relative}:{number}")
        name = _canonical_name(match.group(1))
        if name in seen:
            raise DependencyCacheError(f"duplicate distribution {name} in {relative}")
        seen.add(name)
        line_hashes = _HASH.findall(match.group(3))
        if not line_hashes:
            raise DependencyCacheError(f"unhashed lock line {relative}:{number}")
        hashes.update(line_hashes)
        package_count += 1
    if package_count == 0:
        raise DependencyCacheError(f"empty dependency lock: {relative}")
    return LockIdentity(
        path=relative,
        sha256=_sha256_file(path),
        package_count=package_count,
        allowed_hashes=tuple(sorted(hashes)),
    )


def _validate_recorded_file_hash(path: Path, record: dict[str, Any], label: str) -> None:
    expected = record.get("sha256")
    if expected is None:
        return
    if _SHA256.fullmatch(str(expected)) is None or _sha256_file(path) != expected:
        raise DependencyCacheError(f"{label} recorded file hash mismatch")


def _collect_locks(
    root: Path,
    document: dict[str, Any],
    *,
    seen_paths: set[str],
    identities: list[LockIdentity],
) -> None:
    locks = document.get("locks", {})
    if not isinstance(locks, dict):
        raise DependencyCacheError("profile locks must be an object")
    for name in sorted(locks):
        record = locks[name]
        if not isinstance(record, dict) or not record.get("path"):
            raise DependencyCacheError(f"malformed lock record: {name}")
        relative = str(record["path"])
        if relative in seen_paths:
            continue
        path = _safe_path(root, relative)
        if not path.is_file():
            raise DependencyCacheError(f"lock file is missing: {relative}")
        _validate_recorded_file_hash(path, record, f"lock {relative}")
        identity = _parse_lock(path, relative)
        expected_count = record.get("package_count")
        if expected_count is not None and int(expected_count) != identity.package_count:
            raise DependencyCacheError(f"lock package count mismatch: {relative}")
        identities.append(identity)
        seen_paths.add(relative)


def resolve_profile(root: Path, profile_path: str) -> ProfileResolution:
    """Resolve one purpose profile and every exact component lock it consumes."""

    root = root.resolve()
    path = _safe_path(root, profile_path)
    profile = _load_json(path)
    profile_id = str(profile.get("profile_id", ""))
    if not profile_id:
        raise DependencyCacheError("purpose profile lacks profile_id")

    semantic = profile.get("profile_sha256")
    if semantic is not None and _SHA256.fullmatch(str(semantic)) is None:
        raise DependencyCacheError("purpose profile has malformed profile_sha256")

    platform_contract = profile.get("platform", {})
    python_contract = profile.get("python", {})
    if not isinstance(platform_contract, dict) or not isinstance(python_contract, dict):
        raise DependencyCacheError("purpose profile lacks platform/Python contract")

    locks: list[LockIdentity] = []
    seen_paths: set[str] = set()
    support_profiles: list[tuple[str, str]] = []

    base_ref = profile.get("base_profile")
    if base_ref is not None:
        if not isinstance(base_ref, dict) or not base_ref.get("path"):
            raise DependencyCacheError("malformed base_profile reference")
        base_relative = str(base_ref["path"])
        base_path = _safe_path(root, base_relative)
        if not base_path.is_file():
            raise DependencyCacheError(f"base profile is missing: {base_relative}")
        expected = base_ref.get("file_sha256")
        base_sha = _sha256_file(base_path)
        if expected is not None and base_sha != expected:
            raise DependencyCacheError("base profile file hash mismatch")
        base = _load_json(base_path)
        support_profiles.append((base_relative, base_sha))
        _collect_locks(root, base, seen_paths=seen_paths, identities=locks)

    _collect_locks(root, profile, seen_paths=seen_paths, identities=locks)
    if not locks:
        raise DependencyCacheError("purpose profile resolves to no dependency locks")

    return ProfileResolution(
        profile_id=profile_id,
        profile_path=profile_path,
        profile_file_sha256=_sha256_file(path),
        profile_semantic_sha256=str(semantic) if semantic is not None else None,
        support_profiles=tuple(sorted(support_profiles)),
        locks=tuple(sorted(locks, key=lambda item: item.path)),
        system=str(platform_contract.get("system", "")),
        machine=_normalize_machine(str(platform_contract.get("machine", ""))),
        python_implementation=str(python_contract.get("implementation", "")).lower(),
        python_version=str(python_contract.get("version", "")),
    )


def _locked_package_names(root: Path, locks: Iterable[LockIdentity]) -> set[str]:
    names: set[str] = set()
    for lock in locks:
        path = _safe_path(root, lock.path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _REQUIREMENT.fullmatch(line)
            if match is None:
                raise DependencyCacheError(f"non-exact lock line in {lock.path}")
            names.add(_canonical_name(match.group(1)))
    return names


def assert_no_accidental_cuda(root: Path, resolution: ProfileResolution) -> None:
    """Fail closed if a non-CUDA purpose profile would cache CUDA vendor packages."""

    if "cuda" in resolution.profile_id.lower():
        return
    forbidden = sorted(
        name
        for name in _locked_package_names(root, resolution.locks)
        if name.startswith(_FORBIDDEN_CPU_PREFIXES)
    )
    if forbidden:
        raise DependencyCacheError(
            f"non-CUDA purpose profile {resolution.profile_id} includes CUDA packages: {forbidden}"
        )


def build_manifest(
    root: Path,
    profile_path: str,
    *,
    actual_system: str | None = None,
    actual_machine: str | None = None,
    actual_python_implementation: str | None = None,
    actual_python_version: str | None = None,
    reject_accidental_cuda: bool = True,
) -> dict[str, Any]:
    """Build the deterministic cache identity from exact environment authority inputs."""

    resolution = resolve_profile(root, profile_path)
    system = actual_system or platform.system()
    machine = _normalize_machine(actual_machine or platform.machine())
    implementation = (actual_python_implementation or sys.implementation.name).lower()
    version = actual_python_version or platform.python_version()

    expected = (
        resolution.system,
        resolution.machine,
        resolution.python_implementation,
        resolution.python_version,
    )
    actual = (system, machine, implementation, version)
    if expected != actual:
        raise DependencyCacheError(f"profile platform/Python mismatch: expected={expected} actual={actual}")
    if reject_accidental_cuda:
        assert_no_accidental_cuda(root.resolve(), resolution)

    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "platform": {"system": system, "machine": machine},
        "python": {"implementation": implementation, "version": version},
        "purpose_profile": {
            "profile_id": resolution.profile_id,
            "path": resolution.profile_path,
            "file_sha256": resolution.profile_file_sha256,
            "semantic_sha256": resolution.profile_semantic_sha256,
        },
        "support_profiles": [
            {"path": path, "file_sha256": digest}
            for path, digest in resolution.support_profiles
        ],
        "component_locks": [
            {
                "path": lock.path,
                "file_sha256": lock.sha256,
                "package_count": lock.package_count,
            }
            for lock in resolution.locks
        ],
    }
    identity = _sha256_bytes(_canonical_bytes(payload))
    safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolution.profile_id)
    key = f"ci164-v1-{system.lower()}-{machine}-py{version}-{safe_profile}-{identity}"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "cache_key": key,
        "identity_sha256": identity,
        **payload,
    }
    return manifest


def validate_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
    """Re-hash the current profile and locks before any cached wheel can be trusted."""

    root = root.resolve()
    profile = manifest.get("purpose_profile", {})
    profile_path = _safe_path(root, str(profile.get("path", "")))
    if _sha256_file(profile_path) != profile.get("file_sha256"):
        raise DependencyCacheError("purpose profile changed after cache key generation")
    for record in manifest.get("support_profiles", []):
        path = _safe_path(root, str(record.get("path", "")))
        if _sha256_file(path) != record.get("file_sha256"):
            raise DependencyCacheError("support profile changed after cache key generation")
    for record in manifest.get("component_locks", []):
        path = _safe_path(root, str(record.get("path", "")))
        if _sha256_file(path) != record.get("file_sha256"):
            raise DependencyCacheError("component lock changed after cache key generation")


def verify_wheelhouse(root: Path, manifest: dict[str, Any], wheelhouse: Path) -> dict[str, Any]:
    """Verify every cached wheel is authorized by at least one exact component-lock hash."""

    validate_manifest_files(root, manifest)
    allowed: set[str] = set()
    for record in manifest.get("component_locks", []):
        identity = _parse_lock(
            _safe_path(root.resolve(), str(record["path"])), str(record["path"])
        )
        allowed.update(identity.allowed_hashes)

    if not wheelhouse.exists():
        return {"status": "CACHE_MISS", "wheel_count": 0, "verified_bytes": 0}
    if not wheelhouse.is_dir():
        raise DependencyCacheError("wheelhouse path is not a directory")

    wheels = sorted(wheelhouse.glob("*.whl"))
    unexpected = sorted(path.name for path in wheelhouse.iterdir() if path.is_file() and path.suffix != ".whl")
    if unexpected:
        raise DependencyCacheError(f"wheelhouse contains non-wheel files: {unexpected}")
    total = 0
    rows = []
    for wheel in wheels:
        digest = _sha256_file(wheel)
        if digest not in allowed:
            raise DependencyCacheError(f"wheel hash is absent from selected exact locks: {wheel.name}")
        size = wheel.stat().st_size
        total += size
        rows.append({"filename": wheel.name, "sha256": digest, "bytes": size})
    return {
        "status": "VERIFIED" if wheels else "CACHE_MISS",
        "wheel_count": len(wheels),
        "verified_bytes": total,
        "wheels": rows,
    }
