from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from twelve_six.integration.dependency_lock import (
    PROFILE_PYTHON_VERSIONS,
    SUPPORTED_PROFILES,
    DependencyLockError,
    validate_lock_index,
    validate_profile_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_lock_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copytree(ROOT / "requirements", root / "requirements")
    return root


def _load_verifier():
    path = ROOT / "tools" / "verify_locked_environment.py"
    spec = importlib.util.spec_from_file_location("verify_locked_environment_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_index_binds_complete_profile_set() -> None:
    index = validate_lock_index(root=ROOT, index_path="requirements/locks/index.json")
    assert set(index["profiles"]) == SUPPORTED_PROFILES
    assert index["python_versions"] == PROFILE_PYTHON_VERSIONS
    assert index["index_sha256"] == (
        "8e21fdddfc4001ca3a1016764bbe910fca478022051435fbb0ad89ed9940a1a8"
    )


def test_windows_profile_binds_exact_python_and_runtime_lock() -> None:
    profile = validate_profile_manifest(
        root=ROOT,
        manifest_path="requirements/locks/windows-x86_64/profile.json",
        enforce_current_platform=False,
    )
    assert profile["python"]["version"] == "3.11.9"
    assert profile["platform"] == {"machine": "AMD64", "system": "Windows"}
    assert profile["locks"]["runtime"]["package_count"] == 12
    assert profile["locks"]["runtime"]["sha256"] == (
        "a94337bafdfc74ad2779d3c05b2f315048c1c39200a3011825ac6ebdff7b628f"
    )


def test_tampered_lock_is_rejected_before_install(tmp_path: Path) -> None:
    root = _copy_lock_fixture(tmp_path)
    lock_path = root / "requirements/locks/linux-x86_64/runtime.lock.txt"
    lock_path.write_text(lock_path.read_text(encoding="utf-8") + "tampered==1.0\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="profile file hash mismatch|lock hash mismatch"):
        validate_lock_index(root=root, index_path="requirements/locks/index.json")


def test_stale_pyproject_is_rejected(tmp_path: Path) -> None:
    root = _copy_lock_fixture(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("numpy>=1.26", "numpy>=1.27"),
        encoding="utf-8",
    )

    with pytest.raises(DependencyLockError, match="stale for current pyproject"):
        validate_profile_manifest(
            root=root,
            manifest_path="requirements/locks/linux-x86_64/profile.json",
            enforce_current_platform=False,
        )


def test_tampered_index_self_hash_is_rejected(tmp_path: Path) -> None:
    root = _copy_lock_fixture(tmp_path)
    path = root / "requirements/locks/index.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["python_versions"]["windows-x86_64"] = "3.11.8"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="self-hash mismatch"):
        validate_lock_index(root=root, index_path="requirements/locks/index.json")


def test_lock_text_rejects_floating_or_unhashed_requirements(tmp_path: Path) -> None:
    verifier = _load_verifier()
    floating = tmp_path / "floating.lock.txt"
    floating.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(verifier.LOCK.DependencyLockError, match="non-exact or unhashed"):
        verifier._validate_lock_text(floating, 1)

    unhashed = tmp_path / "unhashed.lock.txt"
    unhashed.write_text("numpy==2.4.6\n", encoding="utf-8")
    with pytest.raises(verifier.LOCK.DependencyLockError, match="non-exact or unhashed"):
        verifier._validate_lock_text(unhashed, 1)


def test_lock_text_rejects_duplicate_distribution(tmp_path: Path) -> None:
    verifier = _load_verifier()
    path = tmp_path / "duplicate.lock.txt"
    digest = "0" * 64
    path.write_text(
        f"numpy==2.4.6 --hash=sha256:{digest}\n"
        f"NumPy==2.4.6 --hash=sha256:{digest}\n",
        encoding="utf-8",
    )
    with pytest.raises(verifier.LOCK.DependencyLockError, match="duplicate locked distribution"):
        verifier._validate_lock_text(path, 2)
