from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import twelve_six.github_hosted_session_plan as session_plan
from twelve_six.github_hosted_session_plan import (
    _read_json,
    assess_session_launch,
    build_session_plan,
    canonical_sha256,
    validate_hosted_carrier_evidence,
    validate_previous_handoff,
    validate_profile,
    validate_session_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/research/r01_github_hosted_local_free_cpu_session_v1.json"
SOURCE_SHA = "a" * 40


def _carrier_evidence(*, source_sha: str = SOURCE_SHA) -> dict:
    evidence = {
        "schema": "twelve-six-github-hosted-carrier-preflight-v1",
        "source_sha": source_sha,
        "expected_source_sha": source_sha,
        "observed_checkout_sha": source_sha,
        "result": "PASS",
        "carrier_mode": "preflight",
        "repository": "Oleksii-debug/12-6-ai.",
        "repository_visibility": "public",
        "runner": {
            "provider": "github-hosted",
            "environment": "github-hosted",
            "label": "ubuntu-24.04",
            "os": "Linux",
            "arch": "X64",
        },
        "worker_identity": {"protocol": "test-only-fixture"},
        "scientific_effects": {
            "worker_invoked": False,
            "real_target_execution_supported": False,
            "authorized_optimized_target_exposure": 0,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
        },
        "authority_boundary": {
            "worker_authority_environment_present": False,
            "paid_compute_authorized": False,
            "foreign_pretrained_weights_permitted": False,
            "external_llm_or_api_for_data_or_intelligence_permitted": False,
        },
        "future_real_target_requirements": ["CANONICAL_AUTHORITY_STILL_REQUIRED"],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _carrier_kwargs() -> dict:
    evidence = _carrier_evidence()
    return {
        "hosted_carrier_evidence": evidence,
        "expected_hosted_carrier_evidence_sha256": evidence["evidence_sha256"],
    }


def _profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _handoff(plan: dict) -> dict:
    return {
        "plan_sha256": plan["plan_sha256"],
        "session_index": 1,
        "run_id": "run-1",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "checkpoint_uri": "file:///tmp/checkpoint-1",
        "hosted_carrier_evidence_sha256": _carrier_evidence()["evidence_sha256"],
    }


def _packet(handoff: dict | None = None, *, limit: int = 360) -> dict:
    mode = "RESUME" if handoff else "FRESH_START"
    checkpoint = {
        "mode": mode,
        "session_time_limit_minutes": limit,
        "first_checkpoint_deadline_minutes": min(60, limit - 1),
    }
    if handoff:
        checkpoint["lineage"] = {
            "source_provider": "OTHER_FREE",
            "cross_provider_transfer": False,
            "parent_checkpoint_sha256": handoff["checkpoint_sha256"],
            "parent_manifest_sha256": handoff["checkpoint_manifest_sha256"],
            "previous_run_id": handoff["run_id"],
        }
    return {
        "identities": {"source_git_sha": SOURCE_SHA},
        "resource": {
            "resource_class": "LOCAL_FREE",
            "provider": "OTHER_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "checkpoint": checkpoint,
    }


def test_profile_and_deterministic_session_slicing_are_fail_closed() -> None:
    profile = _profile()
    assert validate_profile(profile) == []
    plan = build_session_plan(profile, 800)
    assert validate_session_plan(plan, profile) == []
    assert [row["planned_work_minutes"] for row in plan["sessions"]] == [330, 330, 140]
    assert [row["mode"] for row in plan["sessions"]] == [
        "FRESH_START",
        "RESUME",
        "RESUME",
    ]
    assert plan["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert plan["truth_boundary"]["training_executed"] is False
    sealed = copy.deepcopy(plan)
    digest = sealed.pop("plan_sha256")
    assert digest == canonical_sha256(sealed)


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("configured_job_budget_minutes", True, "profile_configured_job_budget_minutes_invalid"),
        (
            "checkpoint_safety_margin_minutes",
            True,
            "profile_checkpoint_safety_margin_minutes_invalid",
        ),
        ("max_session_work_minutes", True, "profile_max_session_work_minutes_invalid"),
        ("maximum_cost_usd", True, "profile_maximum_cost_usd_must_be_zero"),
        ("materially_paid", 0, "profile_materially_paid_must_be_false"),
    ],
)
def test_profile_rejects_bool_numeric_coercions_and_paid_drift(
    field: str, replacement: object, expected: str
) -> None:
    profile = _profile()
    profile[field] = replacement
    assert expected in validate_profile(profile)


def test_plan_rejects_unsafe_window_truth_widening_sequence_gap_and_reseal() -> None:
    profile = _profile()
    unsafe = copy.deepcopy(profile)
    unsafe["max_session_work_minutes"] = 331
    assert "profile_work_window_exceeds_safe_job_budget" in validate_profile(unsafe)

    plan = build_session_plan(profile, 700)
    widened = copy.deepcopy(plan)
    widened["truth_boundary"]["training_executed"] = True
    assert "plan_truth_training_executed_must_be_false" in validate_session_plan(
        widened, profile
    )
    gap = copy.deepcopy(plan)
    gap["sessions"][1]["session_index"] = 3
    assert "session_2_index_mismatch" in validate_session_plan(gap, profile)
    resealed_wrong = copy.deepcopy(plan)
    resealed_wrong["estimated_runtime_minutes"] = 701
    assert "plan_sha256_mismatch" in validate_session_plan(resealed_wrong, profile)


def test_handoff_rejects_credentials_and_wrong_predecessor() -> None:
    plan_hash = "a" * 64
    handoff = {
        "plan_sha256": plan_hash,
        "session_index": 1,
        "run_id": "run-1",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "checkpoint_uri": "https://user:pass@example.invalid/checkpoint?token=secret",
    }
    errors = validate_previous_handoff(
        handoff, plan_sha256=plan_hash, expected_session_index=3
    )
    assert "handoff_checkpoint_uri_invalid_or_credential_bearing" in errors
    assert "handoff_session_index_mismatch" in errors


def test_build_rejects_bool_and_nonpositive_runtime() -> None:
    profile = _profile()
    for value in (True, 0, -1):
        with pytest.raises(ValueError, match="estimated_runtime_minutes"):
            build_session_plan(profile, value)


def test_initial_session_consumes_incumbent_portable_readiness(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 45)
    packet = _packet()
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=True,
            ready_for_same_provider_fresh_process_resume=False,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=("not_resume",),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(plan, profile, 1, packet, **_carrier_kwargs())
    assert result.ready
    assert result.blockers == ()


def test_same_provider_signal_authorizes_only_when_canonical_portable_does(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    packet = _packet(handoff)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=True,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=(),
            resume_blockers=("cross_provider_source_and_target_must_differ",),
        ),
    )
    result = assess_session_launch(
        plan,
        profile,
        2,
        packet,
        previous_handoff=handoff,
        **_carrier_kwargs(),
    )
    assert result.ready
    assert result.blockers == ()


def test_cross_provider_readiness_cannot_authorize_same_provider_resume(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    packet = _packet(handoff)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=False,
            ready_for_cross_provider_resume=True,
            launch_blockers=(),
            same_provider_resume_blockers=("trusted_parent_recovery_binding_missing",),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(
        plan, profile, 2, packet, previous_handoff=handoff
    )
    assert not result.ready
    assert "canonical_same_provider_resume_authority_unavailable" in result.blockers
    assert "portable:trusted_parent_recovery_binding_missing" in result.blockers


def test_coherent_handoff_and_packet_reseal_stays_blocked_without_trusted_recovery(
    monkeypatch,
) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    handoff["run_id"] = "attacker-resealed-run"
    handoff["checkpoint_sha256"] = "d" * 64
    handoff["checkpoint_manifest_sha256"] = "e" * 64
    packet = _packet(handoff)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=False,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=("trusted_parent_recovery_binding_missing",),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(
        plan, profile, 2, packet, previous_handoff=handoff
    )
    assert not result.ready
    assert "portable:trusted_parent_recovery_binding_missing" in result.blockers


def test_bound_portable_time_limit_can_only_narrow_profile_window(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    packet = _packet(handoff, limit=300)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=True,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=(),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(
        plan, profile, 2, packet, previous_handoff=handoff
    )
    assert not result.ready
    assert "planned_session_exceeds_portable_time_limit" in result.blockers
    assert "plan_checkpoint_deadline_not_before_portable_limit" in result.blockers


def test_missing_same_provider_signal_fails_closed(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    packet = _packet(handoff)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_cross_provider_resume=True,
            launch_blockers=(),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(
        plan, profile, 2, packet, previous_handoff=handoff
    )
    assert not result.ready
    assert "portable:same_provider_resume_signal_unavailable" in result.blockers

def test_concrete_hosted_carrier_evidence_is_required_even_when_portable_is_ready(
    monkeypatch,
) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 45)
    packet = _packet()
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=True,
            ready_for_same_provider_fresh_process_resume=False,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=(),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(plan, profile, 1, packet)
    assert not result.ready
    assert "hosted_carrier_evidence_missing" in result.blockers
    assert "hosted_carrier_expected_evidence_sha256_invalid" in result.blockers


def test_carrier_substitution_cannot_self_reseal_under_fixed_external_root() -> None:
    original = _carrier_evidence()
    candidate = copy.deepcopy(original)
    candidate["runner"]["environment"] = "self-hosted"
    candidate["runner"]["provider"] = "other-free"
    candidate["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in candidate.items() if key != "evidence_sha256"}
    )

    errors = validate_hosted_carrier_evidence(
        candidate,
        expected_evidence_sha256=original["evidence_sha256"],
        expected_source_git_sha=SOURCE_SHA,
    )
    assert "hosted_carrier_external_authority_mismatch" in errors
    assert "hosted_carrier_runner_identity_mismatch" in errors


def test_carrier_source_sha_must_match_portable_source_identity() -> None:
    evidence = _carrier_evidence(source_sha="b" * 40)
    errors = validate_hosted_carrier_evidence(
        evidence,
        expected_evidence_sha256=evidence["evidence_sha256"],
        expected_source_git_sha=SOURCE_SHA,
    )
    assert "hosted_carrier_source_sha_mismatch" in errors
    assert "hosted_carrier_expected_source_sha_mismatch" in errors
    assert "hosted_carrier_observed_checkout_sha_mismatch" in errors


def test_resume_handoff_binds_same_concrete_hosted_carrier(monkeypatch) -> None:
    profile = _profile()
    plan = build_session_plan(profile, 700)
    handoff = _handoff(plan)
    handoff["hosted_carrier_evidence_sha256"] = "f" * 64
    packet = _packet(handoff)
    monkeypatch.setattr(
        session_plan,
        "assess_portable_run_packet",
        lambda _value: SimpleNamespace(
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=True,
            ready_for_cross_provider_resume=False,
            launch_blockers=(),
            same_provider_resume_blockers=(),
            resume_blockers=(),
        ),
    )
    result = assess_session_launch(
        plan,
        profile,
        2,
        packet,
        previous_handoff=handoff,
        **_carrier_kwargs(),
    )
    assert not result.ready
    assert "handoff_hosted_carrier_evidence_sha256_mismatch" in result.blockers


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"provider":"first","provider":"second"}', "duplicate_json_key:provider"),
        ('{"value":NaN}', "nonfinite_json_constant:NaN"),
        ('{"value":Infinity}', "nonfinite_json_constant:Infinity"),
        ('{"value":-Infinity}', "nonfinite_json_constant:-Infinity"),
        ('{"value":1e400}', "nonfinite_json_number:root.value"),
    ],
)
def test_raw_json_trust_boundary_rejects_duplicates_and_nonfinite(
    tmp_path: Path, raw: str, expected: str
) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        _read_json(path)

