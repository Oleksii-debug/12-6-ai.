from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from twelve_six.integration import dependency_lock as LOCK

ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERGE = _load_script(ROOT / "tools/converge_environment_metadata.py", "env154_converge")
PURPOSE = _load_script(ROOT / "tools/verify_purpose_environment.py", "env154_purpose")


@pytest.fixture()
def env_root(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(ROOT / "requirements", tmp_path / "requirements")
    return tmp_path


def _append_comment(path: Path, label: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + f"# ENV-154 {label}\n", encoding="utf-8")


def _canonical_index(root: Path) -> Path:
    return Path("requirements/locks/index.json")


def test_committed_metadata_is_deterministically_derived(env_root: Path) -> None:
    first = CONVERGE.converge(env_root, write=False)
    second = CONVERGE.converge(env_root, write=False)
    assert first == second == {"status": "PASS", "stale": []}


def test_historical_pyproject_style_staleness_is_caught_and_repaired(env_root: Path) -> None:
    # TRAIN-41 run 32862102098 failed before environment creation because a
    # committed profile still bound old pyproject.toml bytes. Reproduce that
    # metadata-only drift without changing dependency semantics.
    _append_comment(env_root / "pyproject.toml", "historical-pyproject-drift")
    with pytest.raises(LOCK.DependencyLockError, match=r"linux-x86_64.*pyproject\.toml stale"):
        LOCK.validate_lock_index(
            root=env_root,
            index_path=_canonical_index(env_root),
            profile_ids={"linux-x86_64"},
        )

    repaired = CONVERGE.converge(env_root, write=True)
    assert repaired["status"] == "REPAIRED"
    assert any(item["path"].endswith("linux-x86_64/profile.json") for item in repaired["stale"])
    assert CONVERGE.converge(env_root, write=False) == {"status": "PASS", "stale": []}
    LOCK.validate_global_lock_index(root=env_root, index_path=_canonical_index(env_root))


@pytest.mark.parametrize("group", ["toolchain", "runtime", "dev"])
def test_exact_component_drift_names_group_and_repairs(env_root: Path, group: str) -> None:
    lock_path = env_root / f"requirements/locks/linux-x86_64/{group}.lock.txt"
    _append_comment(lock_path, f"{group}-bytes-drift")
    with pytest.raises(LOCK.DependencyLockError, match=rf"component {group} stale"):
        LOCK.validate_lock_index(
            root=env_root,
            index_path=_canonical_index(env_root),
            profile_ids={"linux-x86_64"},
        )
    CONVERGE.converge(env_root, write=True)
    LOCK.validate_lock_index(
        root=env_root,
        index_path=_canonical_index(env_root),
        profile_ids={"linux-x86_64"},
    )


def test_unused_other_platform_drift_does_not_block_scoped_experiment(env_root: Path) -> None:
    _append_comment(
        env_root / "requirements/locks/linux-aarch64/runtime.lock.txt",
        "unused-aarch64-runtime-drift",
    )

    # Narrow linux-x86_64 work remains bound to and verifies its own exact locks.
    LOCK.validate_lock_index(
        root=env_root,
        index_path=_canonical_index(env_root),
        profile_ids={"linux-x86_64"},
    )
    with pytest.raises(LOCK.DependencyLockError, match=r"linux-aarch64.*component runtime stale"):
        LOCK.validate_global_lock_index(root=env_root, index_path=_canonical_index(env_root))

    CONVERGE.converge(env_root, write=True)
    LOCK.validate_global_lock_index(root=env_root, index_path=_canonical_index(env_root))


def test_tokenizer_overlay_bytes_invalidate_profile_until_derived(env_root: Path) -> None:
    profile_id = "linux-x86_64-tokenizer-experiment"
    overlay = env_root / f"requirements/profiles/{profile_id}/overlay.lock.txt"
    _append_comment(overlay, "tokenizer-overlay-drift")
    with pytest.raises(PURPOSE.PurposeEnvironmentError, match=r"purpose overlay lock hash mismatch"):
        PURPOSE.validate_registry(env_root, profile_id)

    CONVERGE.converge(env_root, write=True)
    registry = PURPOSE.validate_registry(env_root, profile_id)
    assert registry["profile"]["locks"]["overlay"]["sha256"] == CONVERGE._sha_file(overlay)


def test_cuda_base_role_tracks_canonical_runtime_identity(env_root: Path) -> None:
    # D08's current CUDA purpose is a base-role over the canonical CUDA-enabled
    # runtime closure, not a separate overlay file. Drift in that runtime must
    # therefore invalidate the CUDA purpose's referenced base profile.
    profile_id = "linux-x86_64-cuda-training"
    runtime = env_root / "requirements/locks/linux-x86_64/runtime.lock.txt"
    _append_comment(runtime, "cuda-base-runtime-drift")
    with pytest.raises(PURPOSE.PurposeEnvironmentError, match=r"canonical runtime lock hash mismatch"):
        PURPOSE.validate_registry(env_root, profile_id)

    CONVERGE.converge(env_root, write=True)
    registry = PURPOSE.validate_registry(env_root, profile_id)
    assert registry["profile"]["base_profile"]["manifest_sha256"] == registry["base_profile"]["manifest_sha256"]


def test_global_release_check_still_covers_every_profile(env_root: Path) -> None:
    LOCK.validate_global_lock_index(root=env_root, index_path=_canonical_index(env_root))
    _append_comment(env_root / "requirements/locks/linux-aarch64/toolchain.lock.txt", "global-only-drift")
    with pytest.raises(LOCK.DependencyLockError, match=r"linux-aarch64.*component toolchain stale"):
        LOCK.validate_global_lock_index(root=env_root, index_path=_canonical_index(env_root))
