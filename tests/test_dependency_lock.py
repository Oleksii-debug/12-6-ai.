from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from twelve_six.integration.dependency_lock import (
    EXACT_PYTHON_VERSION,
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
    assert index["python_version"] == EXACT_PYTHON_VERSION
    assert index["index_sha256"] == (
        "5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341"
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
    document["python_version"] = "3.11.15"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="self-hash mismatch"):
        validate_lock_index(root=root, index_path="requirements/locks/index.json")


def test_lock_text_rejects_floating_or_unhashed_requirements(tmp_path: Path) -> None:
    verifier = _load_verifier()
    floating = tmp_path / "floating.lock.txt"
    floating.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="non-exact or unhashed"):
        verifier._validate_lock_text(floating, 1)

    unhashed = tmp_path / "unhashed.lock.txt"
    unhashed.write_text("numpy==2.4.6\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="non-exact or unhashed"):
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
    with pytest.raises(DependencyLockError, match="duplicate locked distribution"):
        verifier._validate_lock_text(path, 2)
