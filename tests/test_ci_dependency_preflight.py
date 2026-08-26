from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ci_dependency_preflight", ROOT / "tools" / "ci_dependency_preflight.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_linux_x86_lock_profile_hashes_validate_without_install() -> None:
    profile = MODULE._validate_profile("linux-x86_64")
    assert profile["profile_id"] == "linux-x86_64"
    assert set(profile["locks"]) == {"toolchain", "runtime", "dev"}
    assert all(len(item["sha256"]) == 64 for item in profile["locks"].values())
