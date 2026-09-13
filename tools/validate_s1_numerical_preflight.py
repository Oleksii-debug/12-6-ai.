"""Validate D02 S1 numerical preflight evidence without promotion authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s1_preflight_contract import validate_s1_preflight_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    validate_s1_preflight_bundle(payload, locked)
    print(
        json.dumps(
            {
                "status": "PASS",
                "authority": payload["authority"],
                "source_sha": payload["identity"]["source_sha"],
                "parameter_count": payload["identity"]["parameter_count"],
                "evidence_sha256": payload["evidence_sha256"],
                "environment_evidence_sha256": locked["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
