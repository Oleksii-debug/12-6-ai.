#!/usr/bin/env python3
"""Run or verify NEXT100-065D global cross-source dedup V6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.cross_source_capacity_audit_v6 import (
    audit_live,
    load_inputs,
    verify_report,
    write_report,
)
from twelve_six.data.cross_source_capacity_guard_v6 import verify_exact_terminal_vector


def _load(path: str) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: JSON root must be an object")
    return value


def _verify(report: dict[str, object]) -> None:
    verify_report(report)
    verify_exact_terminal_vector(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--base-inventory", required=True)
    run.add_argument("--v4-extension", required=True)
    run.add_argument("--v5-config", required=True)
    run.add_argument("--v6-config", required=True)
    run.add_argument("--report", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "run":
        base, v4_extension, v5_config, v6_config = load_inputs(
            args.base_inventory,
            args.v4_extension,
            args.v5_config,
            args.v6_config,
        )
        report = audit_live(base, v4_extension, v5_config, v6_config)
        _verify(report)
        write_report(report, args.report)
        return 0

    _verify(_load(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
