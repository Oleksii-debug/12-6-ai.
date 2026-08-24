"""Load stdlib-only integration tooling without executing twelve_six package imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _ensure_namespace(root: Path) -> None:
    source_root = root / "src" / "twelve_six"
    integration_root = source_root / "integration"
    if not integration_root.is_dir():
        raise RuntimeError(f"integration source directory is missing: {integration_root}")

    package = sys.modules.get("twelve_six")
    if package is None:
        package = ModuleType("twelve_six")
        package.__package__ = "twelve_six"
        package.__path__ = [str(source_root)]
        sys.modules["twelve_six"] = package

    integration = sys.modules.get("twelve_six.integration")
    if integration is None:
        integration = ModuleType("twelve_six.integration")
        integration.__package__ = "twelve_six.integration"
        integration.__path__ = [str(integration_root)]
        sys.modules["twelve_six.integration"] = integration


def load_integration_module(root: str | Path, module_name: str) -> Any:
    """Import one integration utility without importing ``twelve_six.__init__``."""

    root_path = Path(root).resolve()
    _ensure_namespace(root_path)
    return importlib.import_module(f"twelve_six.integration.{module_name}")
