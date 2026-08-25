"""Validate S2/S3 scale evidence and its D08 environment binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_evidence_contract import (
    validate_locked_environment_evidence,
)
from twelve_six.training.s2_s3_scale_execution import (
    ScaleExecutionError,
    validate_scale_execution_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--expected-stage", choices=("S2", "S3"), required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_scale_execution_evidence(
        evidence,
        expected_stage=args.expected_stage,
    )
    identity = evidence["identity"]
    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    binding = validate_locked_environment_evidence(
        locked,
        source_sha=identity["source_sha"],
    )
    if identity["environment"] != binding:
        raise ScaleExecutionError("evidence environment binding mismatch")
    print(evidence["evidence_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
