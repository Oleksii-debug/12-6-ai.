"""Validate self-contained D02 S0 repeatability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_repeatability import validate_s0_repeatability_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    validate_s0_repeatability_evidence(evidence)
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha": evidence["identity"]["source_sha"],
                "evidence_sha256": evidence["evidence_sha256"],
                "same_seed_exact_equivalence": evidence["proof"]["same_seed_exact_equivalence"],
                "different_seed_initialization_diverges": evidence["proof"]["different_seed_initialization_diverges"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
