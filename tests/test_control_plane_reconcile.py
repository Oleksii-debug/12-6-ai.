from twelve_six.integration.control_plane_reconcile import (
    BLOCKED,
    RECONCILED,
    assess_control_plane_reconciliation,
)


def _evidence() -> dict[str, object]:
    candidate = "67ad5277e7a8b1f73dc11e4e6b71d4d4e5aaf355"
    return {
        "live_main_sha": "5020afd671a3885c1b738c8b4eafe7525f630546",
        "candidate_sha": candidate,
        "merge_base_sha": "f2e94c7212888cdb960bb66154d56d210e9b27ab",
        "ahead_by": 798,
        "behind_by": 0,
        "live_main_is_ancestor": True,
        "exact_candidate_ci": {
            "source_sha": candidate,
            "run_id": 1,
            "status": "completed",
            "conclusion": "success",
        },
        "historical_parent_green_promoted": False,
        "training_authorized": False,
        "stage_promoted": False,
    }


def test_reconciled_requires_live_main_ancestry_and_exact_green() -> None:
    result = assess_control_plane_reconciliation(_evidence())
    assert result == {"status": RECONCILED, "ready": True, "blockers": []}


def test_live_divergence_fails_closed() -> None:
    evidence = _evidence()
    evidence["behind_by"] = 33
    evidence["live_main_is_ancestor"] = False
    result = assess_control_plane_reconciliation(evidence)
    assert result["status"] == BLOCKED
    assert "live_main.not_ancestor_of_candidate" in result["blockers"]
    assert "candidate.behind_live_main" in result["blockers"]


def test_historical_green_cannot_replace_exact_candidate_ci() -> None:
    evidence = _evidence()
    evidence["exact_candidate_ci"] = {
        "source_sha": "3883d6a6519b44a5a0a0fd9cb45a9db73c9071f5",
        "run_id": 34003469442,
        "status": "completed",
        "conclusion": "success",
    }
    evidence["historical_parent_green_promoted"] = True
    result = assess_control_plane_reconciliation(evidence)
    assert result["status"] == BLOCKED
    assert "exact_candidate_ci.source_sha_mismatch" in result["blockers"]
    assert "historical_green.must_not_promote" in result["blockers"]


def test_queued_or_failed_exact_ci_is_not_pass() -> None:
    for status, conclusion in (("queued", None), ("completed", "failure")):
        evidence = _evidence()
        evidence["exact_candidate_ci"] = {
            "source_sha": evidence["candidate_sha"],
            "run_id": 42,
            "status": status,
            "conclusion": conclusion,
        }
        result = assess_control_plane_reconciliation(evidence)
        assert result["status"] == BLOCKED


def test_authority_escalation_is_rejected() -> None:
    evidence = _evidence()
    evidence["training_authorized"] = True
    evidence["stage_promoted"] = True
    result = assess_control_plane_reconciliation(evidence)
    assert result["status"] == BLOCKED
    assert "training_authorized.must_remain_false" in result["blockers"]
    assert "stage_promoted.must_remain_false" in result["blockers"]
