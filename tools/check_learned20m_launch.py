from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.learned20m_launch_gate import LaunchGateError, assess_launch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive the fail-closed learned-20M launch authorization state."
    )
    parser.add_argument("packet", type=Path)
    parser.add_argument(
        "--require-state",
        choices=["BLOCKED", "READY_FOR_AUTHORIZATION_REQUEST", "TRAINING_AUTHORIZED"],
        default=None,
    )
    args = parser.parse_args()

    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        result = assess_launch(packet)
    except (OSError, json.JSONDecodeError, LaunchGateError) as exc:
        print(json.dumps({"state": "INVALID", "error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps(result.as_dict(), sort_keys=True))
    if args.require_state is not None and result.state != args.require_state:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
