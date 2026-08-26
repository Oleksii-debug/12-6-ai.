#!/usr/bin/env python3
"""Emit the deterministic R01 FLOPs/vocabulary planning report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.scaling_accounting import build_planning_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/research/r01_flops_vocab_accounting_v1.json",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = build_planning_report(config)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
