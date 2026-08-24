"""CLI for the D10 live release-governance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integration.release_governance import (
    ReleaseGovernanceError,
    verify_release_governance_dict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify protected live GitHub authority for a gated release state."
    )
    parser.add_argument("governance", type=Path, help="release-governance JSON document")
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw = json.loads(args.governance.read_text(encoding="utf-8"))
    try:
        result = verify_release_governance_dict(raw)
    except ReleaseGovernanceError as exc:
        print(json.dumps({"governance_gate": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
