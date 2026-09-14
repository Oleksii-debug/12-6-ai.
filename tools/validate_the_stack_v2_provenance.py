#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.the_stack_v2_policy import evaluate_records, load_policy


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("records JSON must be an array of objects")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate The Stack v2 fail-closed provenance/rights feasibility policy."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/research/the_stack_v2_provenance_feasibility_v1.json"),
    )
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    output: dict[str, Any] = {
        "policy": "VALID",
        "schema": policy["schema"],
        "dataset_id": policy["upstream"]["dataset_id"],
        "dataset_revision": policy["upstream"]["dataset_revision"],
        "current_project_state": policy["admission"]["current_project_state"],
        "canonical_training_authorized": False,
        "bulk_download_authorized": False,
    }
    exit_code = 0
    if args.records is not None:
        evaluation = evaluate_records(policy, _load_records(args.records))
        output["evaluation"] = evaluation
        if evaluation["blocked"]:
            exit_code = 2

    print(json.dumps(output, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
