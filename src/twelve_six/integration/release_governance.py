"""Fail-closed repository governance checks for release promotion authority.

This module deliberately sits *after* the D10 release-attestation and live-authority
validators. Those validators bind component ancestry, candidate CI, artifacts, supply
chain evidence, and independent audits. This layer answers a different question:
whether the GitHub control plane that is allowed to promote a candidate is itself a
trusted, protected root rather than mutable evidence supplied by the candidate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .live_authority import (
    CANONICAL_REPOSITORY,
    CANDIDATE_WORKFLOW_NAME,
    GITHUB_API_ROOT,
    GITHUB_WEB_ROOT,
    JsonGetter,
    LiveAuthorityError,
    github_json_get,
)

SCHEMA_VERSION = "12-6.release-governance.v1"
CANONICAL_DEFAULT_BRANCH = "main"
AUTHORITATIVE_WORKFLOW_PATH = ".github/workflows/ci.yml"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_PROMOTION_STATES = frozenset({"CANDIDATE", "AUDITED_CANDIDATE", "STABLE"})


class ReleaseGovernanceError(LiveAuthorityError):
    """Raised when the repository promotion root is mutable, stale, or ambiguous."""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseGovernanceError(f"{field_name} must be an object")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseGovernanceError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseGovernanceError(f"{field_name} must be a positive integer")
    return value


def _first_attempt(value: Any, field_name: str) -> int:
    attempt = _positive_int(value, field_name)
    if attempt != 1:
        raise ReleaseGovernanceError("release governance v1 admits only first-attempt CI")
    return attempt


def _git_sha(value: Any, field_name: str) -> str:
    text = _string(value, field_name)
    if _GIT_SHA_RE.fullmatch(text) is None:
        raise ReleaseGovernanceError(f"{field_name} must be an exact lowercase 40-hex Git SHA")
    return text


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ReleaseGovernanceError(f"{field_name} must be a non-empty list")
    result: list[str] = []
    for index, raw in enumerate(value):
        text = _string(raw, f"{field_name}[{index}]")
        if text in result:
            raise ReleaseGovernanceError(f"{field_name} contains duplicate entries")
        result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class ReleaseGovernanceExpectation:
    """Exact live GitHub authority facts a gated candidate claims to depend on."""

    repository: str
    promotion_state: str
    candidate_sha: str
    candidate_pr_number: int | None
    candidate_ci_run_id: int
    candidate_ci_run_attempt: int
    candidate_ci_workflow_id: int
    trusted_main_sha: str
    trusted_workflow_blob_sha: str
    required_status_checks: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReleaseGovernanceExpectation":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ReleaseGovernanceError(f"schema_version must be {SCHEMA_VERSION!r}")
        repository = _string(raw.get("repository"), "repository")
        if repository != CANONICAL_REPOSITORY:
            raise ReleaseGovernanceError("repository is not the canonical physical repository")
        state = _string(raw.get("promotion_state"), "promotion_state")
        if state not in _ALLOWED_PROMOTION_STATES:
            raise ReleaseGovernanceError("promotion_state is not a gated release state")
        candidate_sha = _git_sha(raw.get("candidate_sha"), "candidate_sha")
        candidate_pr_raw = raw.get("candidate_pr_number")
        candidate_pr_number: int | None
        if state == "STABLE":
            if candidate_pr_raw is not None:
                raise ReleaseGovernanceError("STABLE authority must not depend on a PR head")
            candidate_pr_number = None
        else:
            candidate_pr_number = _positive_int(candidate_pr_raw, "candidate_pr_number")
        candidate_ci = _mapping(raw.get("candidate_ci"), "candidate_ci")
        trusted_main = _mapping(raw.get("trusted_main"), "trusted_main")
        if trusted_main.get("branch") != CANONICAL_DEFAULT_BRANCH:
            raise ReleaseGovernanceError("trusted_main.branch must be the canonical default branch")
        workflow_path = trusted_main.get("workflow_path")
        if workflow_path != AUTHORITATIVE_WORKFLOW_PATH:
            raise ReleaseGovernanceError(
                "trusted_main.workflow_path must be the authoritative CI workflow path"
            )
        return cls(
            repository=repository,
            promotion_state=state,
            candidate_sha=candidate_sha,
            candidate_pr_number=candidate_pr_number,
            candidate_ci_run_id=_positive_int(candidate_ci.get("run_id"), "candidate_ci.run_id"),
            candidate_ci_run_attempt=_first_attempt(
                candidate_ci.get("run_attempt"),
                "candidate_ci.run_attempt",
            ),
            candidate_ci_workflow_id=_positive_int(
                candidate_ci.get("workflow_id"),
                "candidate_ci.workflow_id",
            ),
            trusted_main_sha=_git_sha(trusted_main.get("sha"), "trusted_main.sha"),
            trusted_workflow_blob_sha=_git_sha(
                trusted_main.get("workflow_blob_sha"),
                "trusted_main.workflow_blob_sha",
            ),
            required_status_checks=_string_tuple(
                trusted_main.get("required_status_checks"),
                "trusted_main.required_status_checks",
            ),
        )


def _repo_full_name(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if isinstance(value, Mapping):
        full_name = value.get("full_name")
        if isinstance(full_name, str):
            return full_name
    return None


def _verify_repository_identity(
    expectation: ReleaseGovernanceExpectation,
    *,
    get_json: JsonGetter,
) -> Mapping[str, Any]:
    repo = get_json(GITHUB_API_ROOT)
    if repo.get("full_name") != expectation.repository:
        raise ReleaseGovernanceError("live repository identity differs from governance evidence")
    if repo.get("default_branch") != CANONICAL_DEFAULT_BRANCH:
        raise ReleaseGovernanceError("canonical repository default branch is not main")
    if repo.get("archived") is not False or repo.get("disabled") is not False:
        raise ReleaseGovernanceError("canonical repository is archived or disabled")
    return repo


def _status_check_contexts(protection: Mapping[str, Any]) -> set[str]:
    required = _mapping(protection.get("required_status_checks"), "required_status_checks")
    if required.get("strict") is not True:
        raise ReleaseGovernanceError("required status checks must require an up-to-date branch")
    contexts: set[str] = set()
    raw_contexts = required.get("contexts")
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if isinstance(item, str) and item:
                contexts.add(item)
    raw_checks = required.get("checks")
    if isinstance(raw_checks, list):
        for raw in raw_checks:
            if isinstance(raw, Mapping):
                context = raw.get("context")
                if isinstance(context, str) and context:
                    contexts.add(context)
    if not contexts:
        raise ReleaseGovernanceError("protected main has no required status checks")
    return contexts


def _verify_protected_main(
    expectation: ReleaseGovernanceExpectation,
    *,
    get_json: JsonGetter,
) -> Mapping[str, Any]:
    branch = get_json(f"{GITHUB_API_ROOT}/branches/{CANONICAL_DEFAULT_BRANCH}")
    if branch.get("name") != CANONICAL_DEFAULT_BRANCH:
        raise ReleaseGovernanceError("live default branch identity mismatch")
    commit = _mapping(branch.get("commit"), "main.commit")
    if commit.get("sha") != expectation.trusted_main_sha:
        raise ReleaseGovernanceError("trusted main SHA is stale")
    if branch.get("protected") is not True:
        raise ReleaseGovernanceError("main is not protected")

    protection = get_json(
        f"{GITHUB_API_ROOT}/branches/{CANONICAL_DEFAULT_BRANCH}/protection"
    )
    enforce_admins = _mapping(protection.get("enforce_admins"), "enforce_admins")
    if enforce_admins.get("enabled") is not True:
        raise ReleaseGovernanceError("main protection does not apply to administrators")
    reviews = _mapping(
        protection.get("required_pull_request_reviews"),
        "required_pull_request_reviews",
    )
    review_count = reviews.get("required_approving_review_count")
    if isinstance(review_count, bool) or not isinstance(review_count, int) or review_count < 1:
        raise ReleaseGovernanceError("main protection requires no approving review")
    if reviews.get("dismiss_stale_reviews") is not True:
        raise ReleaseGovernanceError("main protection does not dismiss stale approvals")
    if _mapping(protection.get("allow_force_pushes"), "allow_force_pushes").get("enabled"):
        raise ReleaseGovernanceError("main protection allows force pushes")
    if _mapping(protection.get("allow_deletions"), "allow_deletions").get("enabled"):
        raise ReleaseGovernanceError("main protection allows branch deletion")
    if (
        _mapping(
            protection.get("required_conversation_resolution"),
            "required_conversation_resolution",
        ).get("enabled")
        is not True
    ):
        raise ReleaseGovernanceError("main protection does not require conversation resolution")

    observed_contexts = _status_check_contexts(protection)
    missing = sorted(set(expectation.required_status_checks) - observed_contexts)
    if missing:
        raise ReleaseGovernanceError(
            "main protection is missing required promotion status checks: "
            + ", ".join(missing)
        )
    return branch


def _workflow_blob(
    sha: str,
    *,
    get_json: JsonGetter,
) -> Mapping[str, Any]:
    payload = get_json(
        f"{GITHUB_API_ROOT}/contents/{AUTHORITATIVE_WORKFLOW_PATH}?ref={sha}"
    )
    if payload.get("type") != "file" or payload.get("path") != AUTHORITATIVE_WORKFLOW_PATH:
        raise ReleaseGovernanceError("authoritative CI workflow is missing or not a file")
    blob_sha = payload.get("sha")
    if not isinstance(blob_sha, str) or _GIT_SHA_RE.fullmatch(blob_sha) is None:
        raise ReleaseGovernanceError("authoritative CI workflow blob SHA is malformed")
    return payload


def _verify_candidate_pr(
    expectation: ReleaseGovernanceExpectation,
    *,
    get_json: JsonGetter,
) -> Mapping[str, Any]:
    if expectation.candidate_pr_number is None:
        raise ReleaseGovernanceError("pre-merge candidate is missing candidate_pr_number")
    pr = get_json(f"{GITHUB_API_ROOT}/pulls/{expectation.candidate_pr_number}")
    if pr.get("number") != expectation.candidate_pr_number:
        raise ReleaseGovernanceError("candidate PR number differs from governance evidence")
    head = _mapping(pr.get("head"), "candidate PR head")
    base = _mapping(pr.get("base"), "candidate PR base")
    if head.get("sha") != expectation.candidate_sha:
        raise ReleaseGovernanceError("candidate PR head SHA is stale")
    if base.get("ref") != CANONICAL_DEFAULT_BRANCH:
        raise ReleaseGovernanceError("candidate PR does not target protected main")
    if _repo_full_name(head, "repo") != CANONICAL_REPOSITORY:
        raise ReleaseGovernanceError("candidate PR head comes from a foreign repository")
    if _repo_full_name(base, "repo") != CANONICAL_REPOSITORY:
        raise ReleaseGovernanceError("candidate PR base repository identity mismatch")
    return pr


def _verify_candidate_run(
    expectation: ReleaseGovernanceExpectation,
    *,
    get_json: JsonGetter,
) -> Mapping[str, Any]:
    run_id = expectation.candidate_ci_run_id
    run = get_json(f"{GITHUB_API_ROOT}/actions/runs/{run_id}")
    if run.get("id") != run_id:
        raise ReleaseGovernanceError("candidate CI run id differs from governance evidence")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ReleaseGovernanceError("candidate CI is not completed-success")
    if run.get("head_sha") != expectation.candidate_sha:
        raise ReleaseGovernanceError("candidate CI head SHA is stale")
    if run.get("run_attempt") != expectation.candidate_ci_run_attempt:
        raise ReleaseGovernanceError("candidate CI run attempt differs from governance evidence")
    if run.get("workflow_id") != expectation.candidate_ci_workflow_id:
        raise ReleaseGovernanceError("candidate CI workflow id differs from governance evidence")
    if run.get("name") != CANDIDATE_WORKFLOW_NAME:
        raise ReleaseGovernanceError("candidate CI is not the authoritative workflow")
    if run.get("path") != AUTHORITATIVE_WORKFLOW_PATH:
        raise ReleaseGovernanceError("candidate CI workflow path is not authoritative")
    expected_url = f"{GITHUB_WEB_ROOT}/actions/runs/{run_id}"
    if run.get("html_url") != expected_url:
        raise ReleaseGovernanceError("candidate CI canonical URL differs from expected authority")
    if _repo_full_name(run, "repository") != CANONICAL_REPOSITORY:
        raise ReleaseGovernanceError("candidate CI repository identity mismatch")

    if expectation.promotion_state == "STABLE":
        if expectation.candidate_sha != expectation.trusted_main_sha:
            raise ReleaseGovernanceError("STABLE candidate is not the protected main head")
        if run.get("event") != "push" or run.get("head_branch") != CANONICAL_DEFAULT_BRANCH:
            raise ReleaseGovernanceError("STABLE authority requires exact main push CI")
    else:
        if run.get("event") != "pull_request":
            raise ReleaseGovernanceError("pre-merge candidate authority requires pull_request CI")
        _verify_candidate_pr(expectation, get_json=get_json)
    return run


def verify_release_governance(
    expectation: ReleaseGovernanceExpectation,
    *,
    get_json: JsonGetter = github_json_get,
) -> dict[str, Any]:
    """Verify the live protected promotion root without granting promotion itself."""

    _verify_repository_identity(expectation, get_json=get_json)
    _verify_protected_main(expectation, get_json=get_json)

    trusted_workflow = _workflow_blob(expectation.trusted_main_sha, get_json=get_json)
    if trusted_workflow.get("sha") != expectation.trusted_workflow_blob_sha:
        raise ReleaseGovernanceError("trusted workflow blob SHA is stale")

    if (
        expectation.promotion_state == "STABLE"
        and expectation.candidate_sha != expectation.trusted_main_sha
    ):
        raise ReleaseGovernanceError("STABLE candidate is not the protected main head")

    candidate_workflow = _workflow_blob(expectation.candidate_sha, get_json=get_json)
    if candidate_workflow.get("sha") != expectation.trusted_workflow_blob_sha:
        raise ReleaseGovernanceError(
            "candidate modifies the authoritative workflow relative to protected main"
        )

    run = _verify_candidate_run(expectation, get_json=get_json)
    return {
        "repository": expectation.repository,
        "promotion_state": expectation.promotion_state,
        "candidate_sha": expectation.candidate_sha,
        "candidate_ci_run_id": expectation.candidate_ci_run_id,
        "candidate_ci_run_attempt": expectation.candidate_ci_run_attempt,
        "candidate_ci_workflow_id": expectation.candidate_ci_workflow_id,
        "trusted_main_sha": expectation.trusted_main_sha,
        "trusted_workflow_blob_sha": expectation.trusted_workflow_blob_sha,
        "required_status_checks": list(expectation.required_status_checks),
        "run_event": run.get("event"),
        "governance_gate": "PASS",
        "promotion_granted": False,
    }


def verify_release_governance_dict(
    raw: Mapping[str, Any],
    *,
    get_json: JsonGetter = github_json_get,
) -> dict[str, Any]:
    """Parse and verify one governance document."""

    return verify_release_governance(
        ReleaseGovernanceExpectation.from_dict(raw),
        get_json=get_json,
    )
