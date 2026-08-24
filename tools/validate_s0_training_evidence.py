"""Validate candidate-bound D02 S0 training evidence for downstream consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_evidence_contract import validate_s0_training_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_s0_training_evidence(payload, require_locked_environment=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha": payload["identity"]["source_sha"],
                "identity_sha256": payload["identity_sha256"],
                "evidence_sha256": payload["evidence_sha256"],
                "optimized_tokens": payload["training"]["optimized_tokens"],
                "validation_optimized_tokens": payload["split_isolation"]["validation_optimized_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
