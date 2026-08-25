"""Validate next-scale optimization experiment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.optimization_experiments import validate_optimization_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_optimization_evidence(payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "authority": payload["authority"],
                "source_sha": payload["identity"]["source_sha"],
                "evidence_sha256": payload["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
