"""Local and release preflight commands for CI/supply-chain gates."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from twelve_six.integration.dependency_evidence import (
    SupplyChainEvidenceError,
    build_supply_chain_documents,
    validate_supply_chain_evidence,
)
from twelve_six.integration.dependency_lock import DependencyLockError, validate_lock_index
from twelve_six.integration.repo_policy import RepositoryPolicyError, validate_repository_policy
from twelve_six.integration.workflow_policy import WorkflowPolicyError, validate_repository_workflows

ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def local_preflight(profile: str, source_sha: str) -> dict[str, str]:
    validate_repository_policy(ROOT)
    validate_repository_workflows(ROOT)
    validate_lock_index(root=ROOT, index_path="requirements/locks/index.json")
    _, evidence = build_supply_chain_documents(
        root=ROOT,
        profile_id=profile,
        source_sha=source_sha,
    )
    return {
        "repository_policy": "PASS",
        "workflow_policy": "PASS",
        "dependency_lock": "PASS",
        "vulnerability": evidence["vulnerability"]["status"],
        "license": evidence["license"]["status"],
    }


def release_preflight(
    profile: str,
    source_sha: str,
    sbom_path: Path,
    evidence_path: Path,
) -> dict[str, str]:
    head = _git_head()
    if head != source_sha:
        raise SupplyChainEvidenceError(
            f"release source SHA must equal checkout HEAD: expected {source_sha} got {head}"
        )
    evidence = validate_supply_chain_evidence(
        root=ROOT,
        sbom_path=sbom_path,
        evidence_path=evidence_path,
        expected_source_sha=source_sha,
        require_resolved=True,
    )
    if evidence.get("profile_id") != profile:
        raise SupplyChainEvidenceError("release profile/evidence mismatch")
    return {
        "source_sha": source_sha,
        "profile": profile,
        "vulnerability": evidence["vulnerability"]["status"],
        "license": evidence["license"]["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local")
    local.add_argument("--profile", required=True)
    local.add_argument("--source-sha", default="UNBOUND_LOCAL")

    release = subparsers.add_parser("release")
    release.add_argument("--profile", required=True)
    release.add_argument("--source-sha", required=True)
    release.add_argument("--sbom", type=Path, required=True)
    release.add_argument("--evidence", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "local":
            results = local_preflight(args.profile, args.source_sha)
        else:
            results = release_preflight(
                args.profile,
                args.source_sha,
                args.sbom,
                args.evidence,
            )
    except (
        DependencyLockError,
        RepositoryPolicyError,
        SupplyChainEvidenceError,
        WorkflowPolicyError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"preflight=FAIL: {exc}")
        return 1

    print("preflight=PASS")
    for key, value in results.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
