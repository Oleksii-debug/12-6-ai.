#!/usr/bin/env python3
"""Namespace-isolated entry point for the #2746 post-dedup/post-QP carrier."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "tools/run_d03_code4_rich_fastapi_postqp_v1.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "d03_code4_rich_fastapi_postqp_runner",
        RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load post-QP runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    historical = tuple(
        module.reserve.code4.incumbent._HISTORICAL_MATCHER_MODULES
    )
    for module_name in historical:
        sys.modules.pop(module_name, None)
    importlib.invalidate_caches()

    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
