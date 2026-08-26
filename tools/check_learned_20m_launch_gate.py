#!/usr/bin/env python3
"""Evaluate a learned-20M launch packet without launching anything."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from twelve_six.learned_20m_launch_gate import assess_learned_20m_launch


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_learned_20m_launch_gate.py <packet.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    packet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        print("launch packet root must be an object", file=sys.stderr)
        return 2
    result = assess_learned_20m_launch(packet)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
