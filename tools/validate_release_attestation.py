"""Validate a release attestation without mutating repository or promotion state."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.integration.release_attestation import (
    load_release_attestation,
    validate_release_attestation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("attestation", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()

    attestation = load_release_attestation(args.attestation)
    stage_manifest = validate_release_attestation(
        attestation,
        repo_root=args.repo_root,
        artifact_root=args.artifact_root,
    )

    print(f"repository={attestation.repository}")
    print(f"stage={attestation.stage}")
    print(f"status={attestation.status.value}")
    print(f"candidate_sha={attestation.candidate_sha or '-'}")
    print(
        "environment_profiles="
        + (",".join(sorted(item.profile_id for item in attestation.environment_evidence)) or "-")
    )
    print(
        "checkpoint_kinds="
        + (",".join(sorted(item.kind for item in attestation.checkpoint_artifacts)) or "-")
    )
    print(
        "supply_chain_kinds="
        + (",".join(sorted(item.kind for item in attestation.supply_chain_artifacts)) or "-")
    )
    print(f"attestation_sha256={attestation.attestation_sha256}")
    print(f"candidate_manifest_loaded={str(stage_manifest is not None).lower()}")
    print("promotion_decision=external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
