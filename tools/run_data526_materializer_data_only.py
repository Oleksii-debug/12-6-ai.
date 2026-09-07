from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


def _arg_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise SystemExit(f"missing required argument: {flag}") from exc
    try:
        return sys.argv[index + 1]
    except IndexError as exc:
        raise SystemExit(f"missing value for argument: {flag}") from exc


def _install_data_only_namespace(v7_root: Path) -> None:
    package_root = (v7_root / "src" / "twelve_six").resolve()
    data_root = package_root / "data"
    if not (package_root / "__init__.py").is_file() or not data_root.is_dir():
        raise SystemExit(f"invalid V7 checkout layout: {v7_root}")

    twelve_six = types.ModuleType("twelve_six")
    twelve_six.__package__ = "twelve_six"
    twelve_six.__path__ = [str(package_root)]

    data = types.ModuleType("twelve_six.data")
    data.__package__ = "twelve_six.data"
    data.__path__ = [str(data_root)]

    twelve_six.data = data
    sys.modules["twelve_six"] = twelve_six
    sys.modules["twelve_six.data"] = data


def main() -> int:
    v7_root = Path(_arg_value("--v7-root"))
    _install_data_only_namespace(v7_root)
    runpy.run_path("tools/materialize_data526_records_from_v7.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
