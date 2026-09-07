import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "swarm" / "swarm300_protocol_v2.json"
AUTOPULSE = ROOT / "docs" / "AUTOPULSE_CONTROL.md"
CI_POLICY = ROOT / "docs" / "CI_SWARM_POLICY.md"
STABILITY_DOC = ROOT / "docs" / "SWARM_VALIDATION_HEAD_STABILITY.md"
UNIVERSAL_PROMPT = ROOT / "docs" / "UNIVERSAL_SWARM_PROMPT.md"


def _stability_contract() -> dict:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    return payload["ci_backpressure"]["validation_head_stability"]


def test_main_advance_alone_never_requires_bridge():
    contract = _stability_contract()
    assert contract["main_advance_alone_requires_bridge"] is False
    assert contract[
        "do_not_chase_live_main_while_exact_head_ci_queued_or_running"
    ] is True
    assert contract["pr_head_change_invalidates_prior_ci_authority"] is True


def test_required_bridge_conditions_remain_fail_closed():
    required = set(_stability_contract()["bridge_required_when_any"])
    assert "github_reports_non_mergeable_or_conflict" in required
    assert "live_main_changes_overlap_owned_surfaces" in required
    assert (
        "live_main_changes_overlap_declared_dependency_test_workflow_or_authority_surfaces"
        in required
    )
    assert "exact_scientific_or_release_authority_requires_new_base_sha" in required
    assert "dependency_semantics_changed_or_integration_is_uncertain" in required


def test_disjoint_intake_still_requires_exact_head_and_collision_checks():
    contract = _stability_contract()
    checks = set(contract["disjoint_main_advance_merge_checks"])
    assert "exact_pr_head_ci_success" in checks
    assert "successful_ci_sha_equals_current_pr_head" in checks
    assert "current_pr_mergeable" in checks
    assert "no_owned_surface_overlap" in checks
    assert "no_declared_dependency_test_workflow_or_authority_overlap" in checks
    assert "no_exact_base_authority_mismatch" in checks
    assert "expected_head_sha_or_equivalent_compare_and_swap_protection" in checks
    assert contract["uncertainty_policy"] == "bridge_once_late_then_rerun_exact_head_ci"


def test_amber_red_blocks_gratuitous_bridge_requeue_churn():
    contract = _stability_contract()
    assert contract["amber_red_gratuitous_bridge_requeue_prohibited"] is True
    assert contract[
        "do_not_chase_further_disjoint_main_commits_during_replacement_ci"
    ] is True


def test_human_policies_state_the_same_fail_closed_boundary():
    autopulse = AUTOPULSE.read_text(encoding="utf-8")
    ci_policy = CI_POLICY.read_text(encoding="utf-8")
    stability = STABILITY_DOC.read_text(encoding="utf-8")

    for text in (autopulse, ci_policy, stability):
        assert "main" in text.lower()
        assert "exact-head" in text.lower()
        assert "bridge" in text.lower()

    assert "Any PR-head change invalidates prior exact-head CI authority" in autopulse
    assert "Live-main movement alone is not a bridge/rebase trigger" in ci_policy
    assert "This policy never turns a stale or untested head green" in stability


def test_universal_prompt_loads_the_updated_policy_authorities_before_work():
    prompt = UNIVERSAL_PROMPT.read_text(encoding="utf-8")
    assert "docs/AUTOPULSE_CONTROL.md" in prompt
    assert "docs/CI_SWARM_POLICY.md" in prompt
    assert "configs/swarm/swarm300_protocol_v2.json" in prompt
