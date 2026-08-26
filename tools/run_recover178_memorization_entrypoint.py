#!/usr/bin/env python3
"""Stable entrypoint for the native RECOVER-178 exact-head runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    path = Path(__file__).with_name("run_recover178_memorization.py")
    spec = importlib.util.spec_from_file_location("recover178_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RECOVER-178 runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
