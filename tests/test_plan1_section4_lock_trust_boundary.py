"""Plan-1 S4: locked environment files must be regular, repository-contained files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from twelve_six.integration.dependency_lock import (
    DependencyLockError,
    canonical_json_bytes,
    sha256_bytes,
    validate_lock_index,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = "requirements/locks/index.json"


def _copy_fixture(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", checkout / "pyproject.toml")
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    return checkout


def _replace_with_identical_symlink(path: Path, target: Path) -> None:
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem symlinks unavailable: {exc}")


def test_rejects_symlinked_lock_with_matching_bytes(tmp_path: Path) -> None:
    checkout = _copy_fixture(tmp_path)
    lock = checkout / "requirements/locks/linux-x86_64/runtime.lock.txt"
    _replace_with_identical_symlink(lock, tmp_path / "outside-runtime.lock.txt")
    with pytest.raises(DependencyLockError, match="unsafe|symlink|regular"):
        validate_lock_index(root=checkout, index_path=INDEX)


def test_rejects_symlinked_profile_with_matching_bytes(tmp_path: Path) -> None:
    checkout = _copy_fixture(tmp_path)
    profile = checkout / "requirements/locks/linux-aarch64/profile.json"
    _replace_with_identical_symlink(profile, tmp_path / "outside-profile.json")
    with pytest.raises(DependencyLockError, match="unsafe|symlink|regular"):
        validate_lock_index(root=checkout, index_path=INDEX)


def test_rejects_rehashed_absolute_profile_path(tmp_path: Path) -> None:
    checkout = _copy_fixture(tmp_path)
    outside = tmp_path / "outside-profile.json"
    outside.write_bytes((checkout / "requirements/locks/linux-aarch64/profile.json").read_bytes())
    path = checkout / INDEX
    index = json.loads(path.read_text(encoding="utf-8"))
    index["profiles"]["linux-aarch64"]["path"] = str(outside)
    index.pop("index_sha256")
    index["index_sha256"] = sha256_bytes(canonical_json_bytes(index))
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="unsafe|absolute|escape"):
        validate_lock_index(root=checkout, index_path=INDEX)


def test_rejects_rehashed_backslash_profile_path(tmp_path: Path) -> None:
    checkout = _copy_fixture(tmp_path)
    path = checkout / INDEX
    index = json.loads(path.read_text(encoding="utf-8"))
    record = index["profiles"]["windows-x86_64"]
    record["path"] = record["path"].replace("/", "\\")
    index.pop("index_sha256")
    index["index_sha256"] = sha256_bytes(canonical_json_bytes(index))
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DependencyLockError, match="unsafe lock path"):
        validate_lock_index(root=checkout, index_path=INDEX)


def _load_verifier():
    import importlib.util

    path = ROOT / "tools/verify_locked_environment.py"
    spec = importlib.util.spec_from_file_location("plan1_s4_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_preserves_canonical_paths_for_the_lock_contract(monkeypatch) -> None:
    verifier = _load_verifier()
    profile_id = verifier.LOCK.current_profile_id()
    validate_index = verifier.LOCK.validate_lock_index
    validate_profile = verifier.LOCK.validate_profile_manifest

    def strict_index(*, root, index_path):
        assert type(index_path) is str and index_path == INDEX
        return validate_index(root=root, index_path=index_path)

    def strict_profile(*, root, manifest_path, enforce_current_platform):
        assert type(manifest_path) is str and "\\" not in manifest_path
        return validate_profile(
            root=root, manifest_path=manifest_path, enforce_current_platform=enforce_current_platform,
        )

    monkeypatch.setattr(verifier.LOCK, "assert_exact_python", lambda: None)
    monkeypatch.setattr(verifier.LOCK, "validate_lock_index", strict_index)
    monkeypatch.setattr(verifier.LOCK, "validate_profile_manifest", strict_profile)
    index, profile = verifier.validate_committed_profile(profile_id)
    assert profile["profile_id"] == profile_id
    assert index["profiles"][profile_id]["manifest_sha256"] == profile["manifest_sha256"]


def test_windows_venv_python_layout(tmp_path: Path) -> None:
    verifier = _load_verifier()
    environment = tmp_path / "Windows Unicode Україна" / "venv"
    executable = environment / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    assert verifier._venv_python(environment) == executable


def test_zip_safe_reproducible_epoch() -> None:
    verifier = _load_verifier()
    environment = verifier._offline_env()
    assert environment["PIP_NO_INDEX"] == "1"
    assert int(environment["SOURCE_DATE_EPOCH"]) >= 315532800
