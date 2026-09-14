"""Validate one TRAIN-29 S1 observability evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_evidence_contract import validate_locked_environment_evidence
from twelve_six.training.s1_observability_probe import validate_s1_observability_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    validate_s1_observability_probe(evidence)
    validate_locked_environment_evidence(
        locked,
        source_sha=evidence["identity"]["source_sha"],
    )
    expected_environment = validate_locked_environment_evidence(
        locked,
        source_sha=evidence["identity"]["source_sha"],
    )
    if evidence["identity"]["environment"] != expected_environment:
        raise ValueError("observability identity does not match locked-environment binding")
    print(
        json.dumps(
            {
                "status": "PASS",
                "identity_sha256": evidence["identity_sha256"],
                "bottleneck": evidence["telemetry"]["summary"]["bottleneck"][
                    "classification"
                ],
                "euro_2000_gate": evidence["paid_compute_decision_support"][
                    "euro_2000_gate"
                ],
                "euro_10000_gate": evidence["paid_compute_decision_support"][
                    "euro_10000_gate"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
