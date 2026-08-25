from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pytest

from twelve_six.integration import release_governance as governance

MAIN_SHA = "a" * 40
CANDIDATE_SHA = "b" * 40
WORKFLOW_BLOB_SHA = "c" * 40
WORKFLOW_ID = 340559512
RUN_ID = 4242
PR_NUMBER = 81
CHECKS = ["CI / locked-x86-64", "CI / locked-arm64"]


def _raw(*, state: str = "CANDIDATE") -> dict[str, Any]:
    candidate_sha = MAIN_SHA if state == "STABLE" else CANDIDATE_SHA
    return {
        "schema_version": governance.SCHEMA_VERSION,
        "repository": governance.CANONICAL_REPOSITORY,
        "promotion_state": state,
        "candidate_sha": candidate_sha,
        "candidate_pr_number": None if state == "STABLE" else PR_NUMBER,
        "candidate_ci": {
            "run_id": RUN_ID,
            "run_attempt": 1,
            "workflow_id": WORKFLOW_ID,
        },
        "trusted_main": {
            "branch": governance.CANONICAL_DEFAULT_BRANCH,
            "sha": MAIN_SHA,
            "workflow_path": governance.AUTHORITATIVE_WORKFLOW_PATH,
            "workflow_blob_sha": WORKFLOW_BLOB_SHA,
            "required_status_checks": list(CHECKS),
        },
    }


def _repo() -> dict[str, Any]:
    return {
        "full_name": governance.CANONICAL_REPOSITORY,
        "default_branch": governance.CANONICAL_DEFAULT_BRANCH,
        "archived": False,
        "disabled": False,
    }


def _branch() -> dict[str, Any]:
    return {
        "name": governance.CANONICAL_DEFAULT_BRANCH,
        "commit": {"sha": MAIN_SHA},
        "protected": True,
    }


def _protection() -> dict[str, Any]:
    return {
        "enforce_admins": {"enabled": True},
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
        },
        "required_status_checks": {
            "strict": True,
            "contexts": list(CHECKS),
            "checks": [],
        },
        "allow_force_pushes": {"enabled": False},
        "allow_deletions": {"enabled": False},
        "required_conversation_resolution": {"enabled": True},
    }


def _workflow() -> dict[str, Any]:
    return {
        "type": "file",
        "path": governance.AUTHORITATIVE_WORKFLOW_PATH,
        "sha": WORKFLOW_BLOB_SHA,
    }


def _pr() -> dict[str, Any]:
    repo = {"full_name": governance.CANONICAL_REPOSITORY}
    return {
        "number": PR_NUMBER,
        "head": {"sha": CANDIDATE_SHA, "repo": repo},
        "base": {
            "ref": governance.CANONICAL_DEFAULT_BRANCH,
            "repo": repo,
        },
    }


def _run(*, state: str = "CANDIDATE") -> dict[str, Any]:
    candidate_sha = MAIN_SHA if state == "STABLE" else CANDIDATE_SHA
    return {
        "id": RUN_ID,
        "status": "completed",
        "conclusion": "success",
        "head_sha": candidate_sha,
        "run_attempt": 1,
        "workflow_id": WORKFLOW_ID,
        "name": governance.CANDIDATE_WORKFLOW_NAME,
        "path": governance.AUTHORITATIVE_WORKFLOW_PATH,
        "html_url": f"{governance.GITHUB_WEB_ROOT}/actions/runs/{RUN_ID}",
        "repository": {"full_name": governance.CANONICAL_REPOSITORY},
        "event": "push" if state == "STABLE" else "pull_request",
        "head_branch": governance.CANONICAL_DEFAULT_BRANCH if state == "STABLE" else "feature",
    }


def _responses(*, state: str = "CANDIDATE") -> dict[str, Mapping[str, Any]]:
    candidate_sha = MAIN_SHA if state == "STABLE" else CANDIDATE_SHA
    responses: dict[str, Mapping[str, Any]] = {
        governance.GITHUB_API_ROOT: _repo(),
        f"{governance.GITHUB_API_ROOT}/branches/main": _branch(),
        f"{governance.GITHUB_API_ROOT}/branches/main/protection": _protection(),
        (
            f"{governance.GITHUB_API_ROOT}/contents/"
            f"{governance.AUTHORITATIVE_WORKFLOW_PATH}?ref={MAIN_SHA}"
        ): _workflow(),
        f"{governance.GITHUB_API_ROOT}/actions/runs/{RUN_ID}": _run(state=state),
    }
    responses[
        (
            f"{governance.GITHUB_API_ROOT}/contents/"
            f"{governance.AUTHORITATIVE_WORKFLOW_PATH}?ref={candidate_sha}"
        )
    ] = _workflow()
    if state != "STABLE":
        responses[f"{governance.GITHUB_API_ROOT}/pulls/{PR_NUMBER}"] = _pr()
    return responses


def _getter(responses: Mapping[str, Mapping[str, Any]]):
    def get_json(url: str) -> Mapping[str, Any]:
        try:
            return responses[url]
        except KeyError as exc:
            raise AssertionError(f"unexpected live URL: {url}") from exc

    return get_json


@pytest.mark.parametrize("state", ["CANDIDATE", "AUDITED_CANDIDATE", "STABLE"])
def test_protected_governance_root_accepts_exact_live_authority(state: str) -> None:
    expectation = governance.ReleaseGovernanceExpectation.from_dict(_raw(state=state))
    result = governance.verify_release_governance(
        expectation,
        get_json=_getter(_responses(state=state)),
    )
    assert result["governance_gate"] == "PASS"
    assert result["promotion_granted"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.__setitem__("repository", "foreign/repo"), "canonical physical"),
        (
            lambda raw: raw["candidate_ci"].__setitem__("run_attempt", 2),
            "first-attempt",
        ),
        (
            lambda raw: raw["trusted_main"].__setitem__("required_status_checks", []),
            "non-empty list",
        ),
        (
            lambda raw: raw["trusted_main"].__setitem__(
                "workflow_path",
                ".github/workflows/x.yml",
            ),
            "authoritative CI workflow path",
        ),
    ],
)
def test_schema_rejects_ambiguous_or_weakened_authority(
    mutation,
    message: str,
) -> None:
    raw = _raw()
    mutation(raw)
    with pytest.raises(governance.ReleaseGovernanceError, match=message):
        governance.ReleaseGovernanceExpectation.from_dict(raw)


def test_unprotected_main_fails_closed_before_candidate_evidence() -> None:
    responses = _responses()
    branch = dict(responses[f"{governance.GITHUB_API_ROOT}/branches/main"])
    branch["protected"] = False
    responses[f"{governance.GITHUB_API_ROOT}/branches/main"] = branch
    with pytest.raises(governance.ReleaseGovernanceError, match="main is not protected"):
        governance.verify_release_governance_dict(
            _raw(),
            get_json=_getter(responses),
        )


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        ("protection", "enforce_admins", {"enabled": False}, "administrators"),
        (
            "protection",
            "allow_force_pushes",
            {"enabled": True},
            "force pushes",
        ),
        ("protection", "allow_deletions", {"enabled": True}, "branch deletion"),
        (
            "protection",
            "required_conversation_resolution",
            {"enabled": False},
            "conversation resolution",
        ),
    ],
)
def test_branch_protection_cannot_be_softened(
    path: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    assert path == "protection"
    responses = _responses()
    url = f"{governance.GITHUB_API_ROOT}/branches/main/protection"
    protection = deepcopy(responses[url])
    protection[field] = value
    responses[url] = protection
    with pytest.raises(governance.ReleaseGovernanceError, match=message):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


def test_branch_protection_requires_review_and_stale_approval_dismissal() -> None:
    responses = _responses()
    url = f"{governance.GITHUB_API_ROOT}/branches/main/protection"
    protection = deepcopy(responses[url])
    protection["required_pull_request_reviews"]["required_approving_review_count"] = 0
    responses[url] = protection
    with pytest.raises(governance.ReleaseGovernanceError, match="no approving review"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))

    protection = _protection()
    protection["required_pull_request_reviews"]["dismiss_stale_reviews"] = False
    responses[url] = protection
    with pytest.raises(governance.ReleaseGovernanceError, match="dismiss stale approvals"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


def test_required_status_checks_cannot_be_removed_or_made_non_strict() -> None:
    responses = _responses()
    url = f"{governance.GITHUB_API_ROOT}/branches/main/protection"
    protection = deepcopy(responses[url])
    protection["required_status_checks"]["contexts"] = [CHECKS[0]]
    responses[url] = protection
    with pytest.raises(governance.ReleaseGovernanceError, match="missing required"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))

    protection = _protection()
    protection["required_status_checks"]["strict"] = False
    responses[url] = protection
    with pytest.raises(governance.ReleaseGovernanceError, match="up-to-date branch"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


def test_candidate_cannot_rewrite_authoritative_ci_workflow() -> None:
    responses = _responses()
    url = (
        f"{governance.GITHUB_API_ROOT}/contents/"
        f"{governance.AUTHORITATIVE_WORKFLOW_PATH}?ref={CANDIDATE_SHA}"
    )
    candidate_workflow = dict(responses[url])
    candidate_workflow["sha"] = "d" * 40
    responses[url] = candidate_workflow
    with pytest.raises(governance.ReleaseGovernanceError, match="modifies the authoritative"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "in_progress", "not completed-success"),
        ("conclusion", "failure", "not completed-success"),
        ("head_sha", "d" * 40, "head SHA is stale"),
        ("run_attempt", 2, "run attempt differs"),
        ("workflow_id", WORKFLOW_ID + 1, "workflow id differs"),
        ("name", "Fast CI", "not the authoritative workflow"),
        ("path", ".github/workflows/fast-ci.yml", "path is not authoritative"),
        ("event", "workflow_dispatch", "requires pull_request CI"),
    ],
)
def test_candidate_run_must_be_exact_authoritative_execution(
    field: str,
    value: Any,
    message: str,
) -> None:
    responses = _responses()
    url = f"{governance.GITHUB_API_ROOT}/actions/runs/{RUN_ID}"
    run = dict(responses[url])
    run[field] = value
    responses[url] = run
    with pytest.raises(governance.ReleaseGovernanceError, match=message):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda pr: pr["head"].__setitem__("sha", "d" * 40),
            "PR head SHA is stale",
        ),
        (
            lambda pr: pr["base"].__setitem__("ref", "release"),
            "does not target protected main",
        ),
        (
            lambda pr: pr["head"].__setitem__("repo", {"full_name": "fork/repo"}),
            "foreign repository",
        ),
    ],
)
def test_candidate_pr_cannot_bypass_protected_main(
    mutation,
    message: str,
) -> None:
    responses = _responses()
    url = f"{governance.GITHUB_API_ROOT}/pulls/{PR_NUMBER}"
    pr = deepcopy(responses[url])
    mutation(pr)
    responses[url] = pr
    with pytest.raises(governance.ReleaseGovernanceError, match=message):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))


def test_stable_requires_main_head_and_push_ci() -> None:
    raw = _raw(state="STABLE")
    raw["candidate_sha"] = CANDIDATE_SHA
    with pytest.raises(governance.ReleaseGovernanceError, match="STABLE candidate"):
        governance.verify_release_governance(
            governance.ReleaseGovernanceExpectation.from_dict(raw),
            get_json=_getter(_responses(state="STABLE")),
        )

    responses = _responses(state="STABLE")
    url = f"{governance.GITHUB_API_ROOT}/actions/runs/{RUN_ID}"
    run = dict(responses[url])
    run["event"] = "pull_request"
    responses[url] = run
    with pytest.raises(governance.ReleaseGovernanceError, match="exact main push CI"):
        governance.verify_release_governance_dict(
            _raw(state="STABLE"),
            get_json=_getter(responses),
        )


def test_stale_main_or_trusted_workflow_identity_is_rejected() -> None:
    responses = _responses()
    branch_url = f"{governance.GITHUB_API_ROOT}/branches/main"
    branch = deepcopy(responses[branch_url])
    branch["commit"]["sha"] = "d" * 40
    responses[branch_url] = branch
    with pytest.raises(governance.ReleaseGovernanceError, match="trusted main SHA is stale"):
        governance.verify_release_governance_dict(_raw(), get_json=_getter(responses))

    raw = _raw()
    raw["trusted_main"]["workflow_blob_sha"] = "d" * 40
    responses = _responses()
    with pytest.raises(governance.ReleaseGovernanceError, match="workflow blob SHA is stale"):
        governance.verify_release_governance_dict(raw, get_json=_getter(responses))


def test_live_lookup_failure_remains_fail_closed() -> None:
    def unavailable(_: str) -> Mapping[str, Any]:
        raise governance.ReleaseGovernanceError("live lookup unavailable")

    with pytest.raises(governance.ReleaseGovernanceError, match="unavailable"):
        governance.verify_release_governance_dict(_raw(), get_json=unavailable)
