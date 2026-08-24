#!/usr/bin/env python3
"""Validate D02 S1 numerical preflight evidence without promotion authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s1_preflight import validate_s1_numerical_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_s1_numerical_preflight(payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "authority": payload["authority"],
                "source_sha": payload["identity"]["source_sha"],
                "parameter_count": payload["identity"]["parameter_count"],
                "evidence_sha256": payload["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
