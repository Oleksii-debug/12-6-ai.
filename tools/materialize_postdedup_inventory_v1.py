#!/usr/bin/env python3
"""Materialize/verify survivor-bound retained inventory from terminal NEXT100-065F."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.postdedup_inventory_v1 import materialize_postdedup_inventory
from twelve_six.data.postdedup_survivor_binding_v1 import (
    build_survivor_inventory_binding,
    verify_survivor_inventory_binding,
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
    parser.add_argument("--survivor-authority", type=Path, required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--binding-evidence", type=Path, required=True)
    args = parser.parse_args()

    report = _load(args.v8_report)
    survivor_authority = _load(args.survivor_authority)
    if args.action == "materialize":
        inventory = materialize_postdedup_inventory(
            report,
            expected_v8_report_sha256=args.expected_v8_report_sha256,
        )
        binding = build_survivor_inventory_binding(
            report,
            survivor_authority,
            inventory,
            expected_v8_report_sha256=args.expected_v8_report_sha256,
            expected_survivor_authority_sha256=(
                args.expected_survivor_authority_sha256
            ),
        )
        _write(args.inventory, inventory)
        _write(args.binding_evidence, binding)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "survivor_authority_sha256": binding[
                        "survivor_authority_sha256"
                    ],
                    "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                    "binding_identity_sha256": binding["binding_identity_sha256"],
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
    binding = _load(args.binding_evidence)
    verify_survivor_inventory_binding(
        report,
        survivor_authority,
        inventory,
        binding,
        expected_v8_report_sha256=args.expected_v8_report_sha256,
        expected_survivor_authority_sha256=(args.expected_survivor_authority_sha256),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "survivor_authority_sha256": binding[
                    "survivor_authority_sha256"
                ],
                "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                "binding_identity_sha256": binding["binding_identity_sha256"],
                "authorized_training_exposure": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
