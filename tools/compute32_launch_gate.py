#!/usr/bin/env python3
"""Evaluate a COMPUTE-32 paid-training launch gate without launching compute."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from twelve_six.training.scale_launch_gate import evaluate_launch_gate


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--authorization",
        default=None,
        help="explicit owner token; only literal COMPUTE_AUTHORIZED can open the final gate",
    )
    args = parser.parse_args()

    report = evaluate_launch_gate(
        _read_json(args.plan),
        _read_json(args.evidence),
        authorization=args.authorization,
    )
    print(json.dumps(asdict(report), sort_keys=True, indent=2))
    return 0 if report.launch_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
