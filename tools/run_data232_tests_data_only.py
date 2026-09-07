"""Run DATA-232 focused tests without importing the model-runtime package initializer."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE = ROOT / "src" / "twelve_six"
DATA_PACKAGE = SRC_PACKAGE / "data"


def _install_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def main() -> int:
    _install_namespace("twelve_six", SRC_PACKAGE)
    _install_namespace("twelve_six.data", DATA_PACKAGE)
    return pytest.main(["-q", "tests/test_data232_decontamination_authority_v2.py"])


if __name__ == "__main__":
    raise SystemExit(main())
