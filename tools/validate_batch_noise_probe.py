"""Validate one TRAIN-53 batch-noise evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.batch_noise_probe import validate_batch_noise_probe
from twelve_six.training.s0_evidence_contract import validate_locked_environment_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    validate_batch_noise_probe(evidence)
    expected_environment = validate_locked_environment_evidence(
        locked,
        source_sha=evidence["identity"]["source_sha"],
    )
    if evidence["identity"]["environment"] != expected_environment:
        raise ValueError("TRAIN-53 identity does not match locked-environment evidence")

    recommendation = evidence["recommendation"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "identity_sha256": evidence["identity_sha256"],
                "report_sha256": evidence["report_sha256"],
                "recommended_gradient_accumulation_steps": recommendation[
                    "recommended_gradient_accumulation_steps"
                ],
                "recommended_effective_loss_tokens_per_update": recommendation[
                    "recommended_effective_loss_tokens_per_update"
                ],
                "exact_critical_batch_size_claim": recommendation[
                    "exact_critical_batch_size_claim"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
