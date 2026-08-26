from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.integration.hashed_dependency_cache import (
    DependencyCacheError,
    build_manifest,
    validate_manifest_files,
    verify_wheelhouse,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fixture(root: Path, *, package: str = "demo", wheel_hash: str | None = None) -> str:
    lock_path = root / "requirements/profiles/test/runtime.lock.txt"
    lock_path.parent.mkdir(parents=True)
    digest = wheel_hash or ("1" * 64)
    lock_path.write_text(f"{package}==1.0.0 --hash=sha256:{digest}\n", encoding="utf-8")
    profile_path = root / "requirements/profiles/test/profile.json"
    profile = {
        "schema_version": "12-6.purpose-environment-profile.v1",
        "profile_id": "linux-x86_64-cpu-training",
        "profile_sha256": "2" * 64,
        "purpose": "fixture",
        "platform": {"system": "Linux", "machine": "x86_64"},
        "python": {"implementation": "cpython", "version": "3.11.16"},
        "locks": {
            "runtime": {
                "path": "requirements/profiles/test/runtime.lock.txt",
                "package_count": 1,
                "sha256": _sha(lock_path.read_bytes()),
            }
        },
    }
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return "requirements/profiles/test/profile.json"


def _manifest(root: Path, profile_path: str) -> dict:
    return build_manifest(
        root,
        profile_path,
        actual_system="Linux",
        actual_machine="x86_64",
        actual_python_implementation="cpython",
        actual_python_version="3.11.16",
    )


def test_cache_key_is_deterministic_and_contains_authority_dimensions(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path)
    first = _manifest(tmp_path, profile_path)
    second = _manifest(tmp_path, profile_path)
    assert first == second
    assert first["cache_key"].startswith(
        "ci164-v1-linux-x86_64-py3.11.16-linux-x86_64-cpu-training-"
    )
    assert first["platform"] == {"system": "Linux", "machine": "x86_64"}
    assert first["python"] == {"implementation": "cpython", "version": "3.11.16"}
    assert first["purpose_profile"]["profile_id"] == "linux-x86_64-cpu-training"
    assert len(first["component_locks"]) == 1


def test_one_byte_profile_change_invalidates_and_rotates_cache_key(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path)
    manifest = _manifest(tmp_path, profile_path)
    path = tmp_path / profile_path
    original = path.read_bytes()
    data = bytearray(original)
    index = data.index(ord("e"), data.index(b'"purpose"'))
    data[index] = ord("f")
    path.write_bytes(bytes(data))

    with pytest.raises(DependencyCacheError, match="profile changed"):
        validate_manifest_files(tmp_path, manifest)

    replacement = _manifest(tmp_path, profile_path)
    assert replacement["cache_key"] != manifest["cache_key"]
    assert replacement["purpose_profile"]["file_sha256"] != manifest["purpose_profile"]["file_sha256"]


def test_one_byte_lock_change_invalidates_and_rotates_cache_key(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path)
    manifest = _manifest(tmp_path, profile_path)
    lock = tmp_path / "requirements/profiles/test/runtime.lock.txt"
    data = bytearray(lock.read_bytes())
    index = data.index(b"1.0.0") + len("1.0.")
    data[index] = ord("1")
    lock.write_bytes(bytes(data))

    with pytest.raises(DependencyCacheError, match="component lock changed"):
        validate_manifest_files(tmp_path, manifest)

    profile_file = tmp_path / profile_path
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    profile["locks"]["runtime"]["sha256"] = _sha(lock.read_bytes())
    profile_file.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replacement = _manifest(tmp_path, profile_path)
    assert replacement["cache_key"] != manifest["cache_key"]
    assert replacement["component_locks"][0]["file_sha256"] != manifest["component_locks"][0]["file_sha256"]


def test_platform_or_python_drift_cannot_reuse_key(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path)
    with pytest.raises(DependencyCacheError, match="platform/Python mismatch"):
        build_manifest(
            tmp_path,
            profile_path,
            actual_system="Linux",
            actual_machine="aarch64",
            actual_python_implementation="cpython",
            actual_python_version="3.11.16",
        )
    with pytest.raises(DependencyCacheError, match="platform/Python mismatch"):
        build_manifest(
            tmp_path,
            profile_path,
            actual_system="Linux",
            actual_machine="x86_64",
            actual_python_implementation="cpython",
            actual_python_version="3.11.15",
        )


def test_cpu_cache_rejects_cuda_vendor_packages(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path, package="nvidia-cublas-cu13")
    with pytest.raises(DependencyCacheError, match="includes CUDA packages"):
        _manifest(tmp_path, profile_path)


def test_empty_or_missing_wheelhouse_is_only_a_cache_miss(tmp_path: Path) -> None:
    profile_path = _write_fixture(tmp_path)
    manifest = _manifest(tmp_path, profile_path)
    result = verify_wheelhouse(tmp_path, manifest, tmp_path / "not-created")
    assert result == {"status": "CACHE_MISS", "wheel_count": 0, "verified_bytes": 0}


def test_every_cached_wheel_must_match_a_selected_lock_hash(tmp_path: Path) -> None:
    wheel_bytes = b"exact locked wheel fixture"
    profile_path = _write_fixture(tmp_path, wheel_hash=_sha(wheel_bytes))
    manifest = _manifest(tmp_path, profile_path)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "demo-1.0.0-py3-none-any.whl"
    wheel.write_bytes(wheel_bytes)
    result = verify_wheelhouse(tmp_path, manifest, wheelhouse)
    assert result["status"] == "VERIFIED"
    assert result["wheel_count"] == 1
    assert result["wheels"][0]["sha256"] == _sha(wheel_bytes)

    wheel.write_bytes(wheel_bytes + b"x")
    with pytest.raises(DependencyCacheError, match="absent from selected exact locks"):
        verify_wheelhouse(tmp_path, manifest, wheelhouse)
