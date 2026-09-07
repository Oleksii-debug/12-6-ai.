#!/usr/bin/env python3
"""Validate and assess the R01 accelerated scaling roadmap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from twelve_six.accelerated_scaling import assess_roadmap

DEFAULT_PATH = Path("configs/research/r01_accelerated_scaling_roadmap_v2.json")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"contract_valid": False, "error": str(exc)}, sort_keys=True))
        return 2
    if not isinstance(payload, dict):
        print(
            json.dumps(
                {"contract_valid": False, "error": "roadmap root must be an object"},
                sort_keys=True,
            )
        )
        return 2
    result = assess_roadmap(payload).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["contract_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
