#!/usr/bin/env python3
"""Materialize/verify deterministic retained-source inventory from terminal V8."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.postdedup_inventory_v1 import (
    materialize_postdedup_inventory,
    verify_postdedup_inventory,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("materialize", "verify"))
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--expected-v8-report-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()

    report = _load(args.v8_report)
    if args.action == "materialize":
        inventory = materialize_postdedup_inventory(
            report,
            expected_v8_report_sha256=args.expected_v8_report_sha256,
        )
        _write(args.inventory, inventory)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                    "retained_source_count": inventory["retained_source_count"],
                    "retained_unique_capacity_bytes": inventory[
                        "retained_unique_capacity_bytes"
                    ],
                    "authorized_training_exposure": 0,
                },
                sort_keys=True,
            )
        )
        return 0

    inventory = _load(args.inventory)
    verify_postdedup_inventory(
        report,
        inventory,
        expected_v8_report_sha256=args.expected_v8_report_sha256,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                "authorized_training_exposure": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
