"""Validate the prepared S0 train/checkpoint/evaluation handoff evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.integration.s0_handoff import HandoffValidationError, validate_s0_handoff


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to an S0 handoff JSON document")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail unless the handoff has complete exact-green D01-D08 execution evidence",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        document: Any = json.loads(args.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise HandoffValidationError("handoff root must be an object")
        result = validate_s0_handoff(document)
    except (OSError, json.JSONDecodeError, HandoffValidationError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2

    payload = {"valid": True, **result}
    print(json.dumps(payload, sort_keys=True))
    if args.require_ready and not result["execution_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
