"""Prepare or validate EVAL-237 source-family-generalization V2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.training.source_family_generalization_v2 import (
    assess_diversity,
    blocked_missing_data230_report,
    build_matched_arm_plan,
    load_family_projection,
    validate_blocked_report,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--blocked-missing-data230", action="store_true")
    group.add_argument("--preflight", type=Path)
    group.add_argument("--validate-blocked", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.validate_blocked is not None:
        validate_blocked_report(_read_json(args.validate_blocked))
        print(f"validated {args.validate_blocked}")
        return 0

    if args.output is None:
        parser.error("--output is required when producing evidence")

    if args.blocked_missing_data230:
        report = blocked_missing_data230_report()
        _write(args.output, report)
        print(args.output)
        return 0

    projection = _read_json(args.preflight)
    families = load_family_projection(projection)
    assessment = assess_diversity(families)
    result = {
        "projection": {
            "producer_worker_id": projection["producer_worker_id"],
            "data230_registry_identity": projection["data230_registry_identity"],
        },
        "assessment": assessment,
    }
    if assessment["ready"]:
        result["matched_arm_plan"] = build_matched_arm_plan(families)
    _write(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
