#!/usr/bin/env python3
"""Run DATA-297 without importing the training/model package root.

The repository's top-level ``twelve_six`` package eagerly imports torch. DATA-297 is
pure data-policy work and must remain LOCAL_FREE/lightweight, so this launcher
creates only the package namespace needed to load ``twelve_six.data`` modules.
It does not replace or copy DATA-33 privacy logic.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_data297_module():
    repository_root = Path(__file__).resolve().parents[1]
    package_root = repository_root / "src" / "twelve_six"
    data_root = package_root / "data"

    twelve_six = types.ModuleType("twelve_six")
    twelve_six.__path__ = [str(package_root)]
    twelve_six.__package__ = "twelve_six"
    sys.modules.setdefault("twelve_six", twelve_six)

    data_package = types.ModuleType("twelve_six.data")
    data_package.__path__ = [str(data_root)]
    data_package.__package__ = "twelve_six.data"
    sys.modules.setdefault("twelve_six.data", data_package)

    module_name = "twelve_six.data297_privacy_external_audit"
    module_path = package_root / "data297_privacy_external_audit.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load DATA-297 audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_data297_module()
    return int(module._main())


if __name__ == "__main__":
    raise SystemExit(main())
