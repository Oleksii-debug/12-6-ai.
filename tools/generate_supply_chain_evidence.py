"""Generate deterministic lock-bound SBOM and supply-chain evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from _integration_bootstrap import load_integration_module

ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = load_integration_module(ROOT, "dependency_evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--sbom-out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--vulnerability-adjudication", type=Path)
    parser.add_argument("--license-adjudication", type=Path)
    args = parser.parse_args()

    sbom, evidence = _EVIDENCE.build_supply_chain_documents(
        root=ROOT,
        profile_id=args.profile,
        source_sha=args.source_sha,
        vulnerability_adjudication=args.vulnerability_adjudication,
        license_adjudication=args.license_adjudication,
    )
    _EVIDENCE.write_supply_chain_documents(
        sbom=sbom,
        evidence=evidence,
        sbom_path=args.sbom_out,
        evidence_path=args.evidence_out,
    )
    print(f"profile={evidence['profile_id']}")
    print(f"components={evidence['component_count']}")
    print(f"vulnerability={evidence['vulnerability']['status']}")
    print(f"license={evidence['license']['status']}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
