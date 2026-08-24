"""Load dependency contracts without executing the twelve_six package root."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "twelve_six"
INTEGRATION_ROOT = PACKAGE_ROOT / "integration"


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency contract {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependency_contracts() -> tuple[Any, Any]:
    """Return dependency_lock and dependency_security without importing package __init__."""
    _namespace("twelve_six", PACKAGE_ROOT)
    _namespace("twelve_six.integration", INTEGRATION_ROOT)
    lock = _load(
        "twelve_six.integration.dependency_lock",
        INTEGRATION_ROOT / "dependency_lock.py",
    )
    security = _load(
        "twelve_six.integration.dependency_security",
        INTEGRATION_ROOT / "dependency_security.py",
    )
    return lock, security
