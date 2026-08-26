from __future__ import annotations

import pytest

from twelve_six.training.source_family_generalization_v2 import (
    DATA230_WORKER_ID,
    FAMILY_PROJECTION_SCHEMA,
    OPTIMIZED_TOKEN_BUDGET,
    Eval237Error,
    FamilyDescriptor,
    assess_diversity,
    blocked_missing_data230_report,
    build_matched_arm_plan,
    family_comparison,
    load_family_projection,
    validate_blocked_report,
)


def _family(
    family_id: str,
    language: str,
    domain: str,
    publisher: str,
    *,
    train_tokens: int = 100_000,
    holdout_tokens: int = 8_192,
    holdout_records: int = 8,
) -> FamilyDescriptor:
    return FamilyDescriptor(
        family_id=family_id,
        language=language,
        domain=domain,
        publisher_id=publisher,
        origin="EXTERNAL_REAL",
        training_authorized=True,
        independent_family=True,
        train_loss_tokens=train_tokens,
        holdout_loss_tokens=holdout_tokens,
        holdout_record_count=holdout_records,
    )


def _identifiable_six_family_design() -> list[FamilyDescriptor]:
    return [
        _family("ua-news-a", "uk", "news", "ua-pub-a"),
        _family("ua-news-b", "uk", "news", "ua-pub-b"),
        _family("ua-legal", "uk", "legal", "ua-pub-c"),
        _family("en-news-a", "en", "news", "en-pub-a"),
        _family("en-news-b", "en", "news", "en-pub-b"),
        _family("en-legal", "en", "legal", "en-pub-c"),
    ]


def test_old_two_family_cross_confounded_regime_is_blocked() -> None:
    families = [
        _family("ua-rada", "uk", "legal", "rada"),
        _family("en-manual", "en", "manual", "standard-ebooks"),
    ]
    assessment = assess_diversity(families)
    assert assessment["ready"] is False
    assert "fewer_than_four_independent_meaningful_holdout_families" in assessment[
        "blockers"
    ]
    assert "one_source_per_language_regime_not_closed" in assessment["blockers"]
    assert "language_effect_not_identifiable" in assessment["blockers"]
    assert "domain_effect_not_identifiable" in assessment["blockers"]
    assert "publisher_source_family_effect_not_identifiable" in assessment[
        "blockers"
    ]


def test_crossed_replicated_design_identifies_requested_effects() -> None:
    assessment = assess_diversity(_identifiable_six_family_design())
    assert assessment["ready"] is True
    assert assessment["blockers"] == []
    assert assessment["family_counts_by_language"] == {"en": 3, "uk": 3}
    assert assessment["identifiability"]["language_effect"]["identifiable"] is True
    assert assessment["identifiability"]["domain_effect"]["identifiable"] is True
    assert (
        assessment["identifiability"]["publisher_source_family_effect"][
            "identifiable"
        ]
        is True
    )


def test_arm_plan_freezes_identity_and_actual_loss_token_budget() -> None:
    families = _identifiable_six_family_design()
    plan = build_matched_arm_plan(families)
    assert len(plan["arms"]) == 1 + len(families)
    for arm in plan["arms"]:
        assert arm["optimized_token_budget"] == OPTIMIZED_TOKEN_BUDGET == 64_512
        assert arm["model"]["parameter_count"] == 467_808
        assert arm["model"]["seed"] == 1337
        assert arm["exposure_accounting"] == "actual_source_loss_tokens_only"
        assert arm["allow_padded_tensor_tokens_in_budget"] is False
        assert arm["allow_example_or_loss_token_repetition"] is False
        omitted = arm["omitted_family_id"]
        if omitted is not None:
            assert omitted not in arm["train_family_ids"]


def test_leave_one_out_requires_unique_token_supply() -> None:
    families = [
        _family("ua-news-a", "uk", "news", "ua-a", train_tokens=10_000),
        _family("ua-news-b", "uk", "news", "ua-b", train_tokens=10_000),
        _family("ua-legal", "uk", "legal", "ua-c", train_tokens=10_000),
        _family("en-news-a", "en", "news", "en-a", train_tokens=10_000),
        _family("en-news-b", "en", "news", "en-b", train_tokens=10_000),
        _family("en-legal", "en", "legal", "en-c", train_tokens=10_000),
    ]
    assessment = assess_diversity(families)
    assert assessment["ready"] is False
    assert "mixed_arm_insufficient_unique_loss_tokens" in assessment["blockers"]
    assert any(
        blocker.startswith("loo_arm_insufficient_unique_loss_tokens::")
        for blocker in assessment["blockers"]
    )


def test_comparison_signs_match_preregistered_interpretation() -> None:
    comparison = family_comparison(
        random_init_bpb=8.0,
        mixed_direct_exposure_bpb=4.0,
        leave_one_family_out_bpb=5.5,
    )
    assert comparison["random_init_improvement_bpb"] == 2.5
    assert comparison["direct_exposure_advantage_bpb"] == 1.5


def test_projection_is_exactly_bound_to_data230() -> None:
    projection = {
        "schema_version": FAMILY_PROJECTION_SCHEMA,
        "producer_worker_id": DATA230_WORKER_ID,
        "data230_registry_identity": "registry-sha-identity",
        "families": [
            {
                "family_id": "ua-a",
                "language": "uk",
                "domain": "news",
                "publisher_id": "publisher-a",
                "origin": "EXTERNAL_REAL",
                "training_authorized": True,
                "independent_family": True,
                "train_loss_tokens": 100_000,
                "holdout_loss_tokens": 8_192,
                "holdout_record_count": 8,
            }
        ],
    }
    families = load_family_projection(projection)
    assert [family.family_id for family in families] == ["ua-a"]

    projection["producer_worker_id"] = "DATA-110"
    with pytest.raises(Eval237Error, match="not bound to DATA-230"):
        load_family_projection(projection)


def test_blocked_report_is_self_hashed_and_truthful() -> None:
    report = blocked_missing_data230_report()
    validate_blocked_report(report)
    assert report["status"] == "BLOCKED_MISSING_DATA230"
    assert report["numerical_training_executed"] is False
    assert report["numerical_result_claimed"] is False
    assert report["scientific_boundary"][
        "old_two_family_regime_is_not_reused_as_v2_evidence"
    ]
