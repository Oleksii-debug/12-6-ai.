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


def _load_verifier():
    import importlib.util

    path = ROOT / "tools/verify_locked_environment.py"
    spec = importlib.util.spec_from_file_location("plan1_s4_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
