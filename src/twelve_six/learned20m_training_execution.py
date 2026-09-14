"""Fail-closed GitHub-hosted carrier for the first learned-20M optimizer step.

This module owns orchestration admission only.  It deliberately does not implement
optimizer math, global leasing, or scientific authorization.  Until those incumbent
lineages publish terminal, machine-verifiable interfaces, execution stays blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from twelve_six.accelerated_scaling import REPOSITORY
from twelve_six.learned20m_training_lease import validate_launch_manifest

CANONICAL_REF = "refs/heads/main"
GITHUB_HOSTED_FREE = "GITHUB_HOSTED_FREE"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# These are integration barriers, not user-overridable switches.  Their owners must
# replace the barrier with an authenticated adapter to the already-canonical lineage.
_REQUIRED_UPSTREAM_INTEGRATIONS = (
    "global_cross_runner_training_lease_not_integrated",
    "terminal_prestep_pass_not_integrated",
    "real_target_training_worker_not_integrated",
)


@dataclass(frozen=True)
class ExecutionContext:
    event_name: str
    repository: str
    ref: str
    github_sha: str
    requested_source_sha: str


@dataclass(frozen=True)
class TrainingExecutionAssessment:
    context_valid: bool
    manifest_valid: bool
    github_hosted_free_bound: bool
    optimizer_start_permitted: bool
    contract_errors: tuple[str, ...]
    blockers: tuple[str, ...]
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_valid": self.context_valid,
            "manifest_valid": self.manifest_valid,
            "github_hosted_free_bound": self.github_hosted_free_bound,
            "optimizer_start_permitted": self.optimizer_start_permitted,
            "contract_errors": list(self.contract_errors),
            "blockers": list(self.blockers),
            "scientific_truth_changed": self.scientific_truth_changed,
        }


def _valid_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def assess_training_execution(
    manifest: Mapping[str, Any], context: ExecutionContext
) -> TrainingExecutionAssessment:
    """Assess carrier admission without converting readiness into training authority."""
    contract_errors = tuple(validate_launch_manifest(manifest))
    blockers: list[str] = []

    if context.event_name != "workflow_dispatch":
        blockers.append("event_must_be_workflow_dispatch")
    if context.repository != REPOSITORY:
        blockers.append("repository_mismatch")
    if context.ref != CANONICAL_REF:
        blockers.append("ref_must_be_main")
    if not _valid_git_sha(context.github_sha):
        blockers.append("github_sha_invalid")
    if not _valid_git_sha(context.requested_source_sha):
        blockers.append("requested_source_sha_invalid")
    if (
        _valid_git_sha(context.github_sha)
        and _valid_git_sha(context.requested_source_sha)
        and context.github_sha != context.requested_source_sha
    ):
        blockers.append("requested_source_sha_not_current_checkout")

    identities = manifest.get("identities")
    source_git_sha = identities.get("source_git_sha") if isinstance(identities, Mapping) else None
    if _valid_git_sha(context.github_sha) and source_git_sha != context.github_sha:
        blockers.append("manifest_source_git_sha_not_current_checkout")

    resource = manifest.get("resource")
    resource_mapping = resource if isinstance(resource, Mapping) else {}
    hosted_free_bound = (
        resource_mapping.get("resource_class") == GITHUB_HOSTED_FREE
        and resource_mapping.get("maximum_cost_usd") == 0
        and resource_mapping.get("materially_paid") is False
    )
    if not hosted_free_bound:
        blockers.append("manifest_not_bound_to_github_hosted_free")

    if contract_errors:
        blockers.append("launch_manifest_contract_invalid")

    # Fail closed until the already-owned canonical upstream lineages expose their
    # authenticated verification/execution adapters.  Do not replace these with CLI
    # booleans or workflow inputs: those would be self-asserted authority.
    blockers.extend(_REQUIRED_UPSTREAM_INTEGRATIONS)
    blockers = sorted(set(blockers))

    context_valid = not any(
        blocker
        in {
            "event_must_be_workflow_dispatch",
            "repository_mismatch",
            "ref_must_be_main",
            "github_sha_invalid",
            "requested_source_sha_invalid",
            "requested_source_sha_not_current_checkout",
            "manifest_source_git_sha_not_current_checkout",
        }
        for blocker in blockers
    )
    manifest_valid = not contract_errors

    return TrainingExecutionAssessment(
        context_valid=context_valid,
        manifest_valid=manifest_valid,
        github_hosted_free_bound=hosted_free_bound,
        optimizer_start_permitted=False,
        contract_errors=contract_errors,
        blockers=tuple(blockers),
        scientific_truth_changed=False,
    )


def _read_repo_json(repo_root: Path, locator: str) -> dict[str, Any]:
    candidate = Path(locator)
    if candidate.is_absolute():
        raise ValueError("manifest_path_must_be_repo_relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest_path_escapes_repository") from exc
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest_root_must_be_object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed learned-20M GitHub-hosted execution carrier"
    )
    parser.add_argument("--manifest", required=True, help="repo-relative launch-manifest JSON")
    parser.add_argument("--requested-source-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = ExecutionContext(
        event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
        repository=os.environ.get("GITHUB_REPOSITORY", ""),
        ref=os.environ.get("GITHUB_REF", ""),
        github_sha=os.environ.get("GITHUB_SHA", ""),
        requested_source_sha=args.requested_source_sha,
    )
    try:
        manifest = _read_repo_json(Path(args.repo_root), args.manifest)
        assessment = assess_training_execution(manifest, context)
        payload = assessment.as_dict()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        payload = {
            "context_valid": False,
            "manifest_valid": False,
            "github_hosted_free_bound": False,
            "optimizer_start_permitted": False,
            "contract_errors": [str(exc)],
            "blockers": ["carrier_input_invalid"],
            "scientific_truth_changed": False,
        }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if payload["optimizer_start_permitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
