#!/usr/bin/env python3
"""Data-only launcher for NEXT100-065F V8.

The historical terminal-V7 tree has a top-level ``twelve_six.__init__`` that imports
model runtime/torch.  Global dedup is a data-only operation and must not acquire a
model-runtime dependency merely to import the incumbent matcher.  This launcher
installs only the ``twelve_six`` package namespace for the exact V7 checkout, so
normal Python import resolution can load ``twelve_six.data`` without executing the
historical top-level package initializer.  Matcher code and semantics are unchanged.
"""
from __future__ import annotations

import importlib.machinery
import runpy
import sys
import types
from pathlib import Path


class DataOnlyBootstrapError(RuntimeError):
    """Raised when the exact data-only V7 namespace cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataOnlyBootstrapError(message)


def _arg_value(flag: str, argv: list[str]) -> str | None:
    for index, value in enumerate(argv):
        if value == flag:
            _require(index + 1 < len(argv), f"missing value for {flag}")
            return argv[index + 1]
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def install_twelve_six_namespace(v7_root: Path) -> None:
    package_root = (v7_root / "src" / "twelve_six").resolve()
    _require(package_root.is_dir(), f"missing terminal V7 package root: {package_root}")

    existing = sys.modules.get("twelve_six")
    if existing is not None:
        paths = [str(Path(value).resolve()) for value in getattr(existing, "__path__", [])]
        _require(str(package_root) in paths, "twelve_six already loaded from a different authority")
        return

    package = types.ModuleType("twelve_six")
    package.__package__ = "twelve_six"
    package.__path__ = [str(package_root)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "twelve_six", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_root)]
    sys.modules["twelve_six"] = package


def main() -> int:
    argv = sys.argv[1:]
    v7_value = _arg_value("--v7-root", argv)
    if v7_value is not None:
        install_twelve_six_namespace(Path(v7_value))

    target = Path(__file__).with_name("run_next100_065f_global_dedup_v8.py")
    _require(target.is_file(), f"missing V8 runner: {target}")
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
