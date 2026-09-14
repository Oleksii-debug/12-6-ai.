#!/usr/bin/env python3
"""Assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from twelve_six.learned20m_readiness import assess_learned20m_readiness

DEFAULT_PATH = Path("configs/research/r01_learned20m_launch_readiness_v1.json")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(json.dumps({"error": "launch packet root must be an object"}, sort_keys=True))
        return 2
    result = assess_learned20m_readiness(payload).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["material_training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
