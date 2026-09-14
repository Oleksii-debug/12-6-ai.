from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from twelve_six.learned20m_training_lease import (
    TERMINAL_AUTHORITY_SCHEMA,
    assess_terminal_launch_authority,
    build_authorized_training_run_lease,
    finalize_launch_manifest,
    launch_manifest_sha256,
    terminal_authority_sha256,
    validate_launch_manifest,
)

NOW = datetime(2026, 9, 14, 19, 30, tzinfo=UTC)


def _manifest() -> dict:
    hashes = {
        "modelspec_sha256": "1" * 64,
        "initspec_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "corpus_manifest_sha256": "4" * 64,
        "split_sha256": "5" * 64,
        "packing_sha256": "6" * 64,
        "unique_loss_ledger_sha256": "7" * 64,
        "training_config_sha256": "8" * 64,
        "portable_run_packet_sha256": "9" * 64,
        "portable_run_binding_sha256": "a" * 64,
    }
    return {
        "schema_version": 1,
        "manifest_id": "R01-LEARNED20M-LAUNCH-MANIFEST-V1",
        "stage": "LEARNED_20M",
        "identities": {"source_git_sha": "b" * 40, **hashes},
        "recipe": {
            "optimizer_scheduler_precision": "AdamW|constant|fp32",
            "seed": 20260826,
            "target_unique_loss_positions": 20_000_000,
            "maximum_total_exposures": 24_000_000,
        },
        "checkpoint": {
            "lineage": "learned20m-v1",
            "checkpoint_contract_sha256": "c" * 64,
        },
        "evaluation": {
            "firewall_sha256": "d" * 64,
            "final_test_payload_access": False,
        },
        "resource": {
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "authorities": {
            "training": {
                "reference": "github:issue/548#training-authority",
                "evidence_sha256": "e" * 64,
            },
            "compute": {
                "reference": "github:issue/548#compute-authority",
                "evidence_sha256": "f" * 64,
            },
        },
        "execution_backend": "PROJECT_NATIVE_PYTORCH",
    }


def _terminal_authority(manifest: dict, *, exposure: int = 0) -> dict:
    identities = manifest["identities"]
    value = {
        "schema": TERMINAL_AUTHORITY_SCHEMA,
        "authority_identity_sha256": "0" * 64,
        "source_git_sha": identities["source_git_sha"],
        "carrier_authority_sha256": "0" * 64,
        "modelspec_sha256": identities["modelspec_sha256"],
        "initspec_sha256": identities["initspec_sha256"],
        "random_init": True,
        "foreign_pretrained_weights_used": False,
        "launch_input_authority_sha256": "1" * 64,
        "corpus_manifest_sha256": identities["corpus_manifest_sha256"],
        "split_sha256": identities["split_sha256"],
        "tokenizer_decision_sha256": "2" * 64,
        "tokenizer_sha256": identities["tokenizer_sha256"],
        "packing_sha256": identities["packing_sha256"],
        "loss_bearing_manifest_sha256": "3" * 64,
        "unique_loss_ledger_sha256": identities["unique_loss_ledger_sha256"],
        "exposure_plan_sha256": "4" * 64,
        "portable_run_packet_sha256": identities["portable_run_packet_sha256"],
        "portable_run_binding_sha256": identities["portable_run_binding_sha256"],
        "recipe_authority_sha256": "5" * 64,
        "training_config_sha256": identities["training_config_sha256"],
        "seed_vector_sha256": "6" * 64,
        "target_unique_loss_positions": manifest["recipe"]["target_unique_loss_positions"],
        "maximum_total_exposures": manifest["recipe"]["maximum_total_exposures"],
        "replay_cap": 4_000_000,
        "recovery_run_id": "learned20m-run-001",
        "recovery_run_manifest_sha256": "7" * 64,
        "recovery_attempt_authority_sha256": "8" * 64,
        "checkpoint_contract_sha256": manifest["checkpoint"]["checkpoint_contract_sha256"],
        "checkpoint_cadence_sha256": "9" * 64,
        "resume_rules_sha256": "a" * 64,
        "safe_stop_current_run_sha256": "b" * 64,
        "evaluation_schedule_sha256": "c" * 64,
        "evaluation_firewall_sha256": manifest["evaluation"]["firewall_sha256"],
        "poison_stop_semantics_sha256": "d" * 64,
        "resource_evidence_sha256": "e" * 64,
        "execution_target_sha256": "f" * 64,
        "measured_resource_envelope_sha256": "0" * 64,
        "resource_class": manifest["resource"]["resource_class"],
        "maximum_cost_usd": 0,
        "materially_paid": False,
        "final_test_payload_access": False,
        "training_authority_ref": manifest["authorities"]["training"]["reference"],
        "training_authority_sha256": manifest["authorities"]["training"]["evidence_sha256"],
        "compute_authority_ref": manifest["authorities"]["compute"]["reference"],
        "compute_authority_sha256": manifest["authorities"]["compute"]["evidence_sha256"],
        "execution_backend": manifest["execution_backend"],
        "authorized_optimized_target_exposure": exposure,
    }
    value["authority_identity_sha256"] = terminal_authority_sha256(value)
    return value


def _finalized(*, exposure: int = 0) -> tuple[dict, dict]:
    manifest = _manifest()
    authority = _terminal_authority(manifest, exposure=exposure)
    return finalize_launch_manifest(manifest, authority), authority


def test_legacy_manifest_remains_structurally_valid_but_not_launch_authoritative():
    manifest = _manifest()
    assert validate_launch_manifest(manifest) == ()
    assessment = assess_terminal_launch_authority(
        manifest,
        expected_terminal_authority_sha256="0" * 64,
    )
    assert assessment.manifest_valid is True
    assert assessment.terminal_authority_valid is False
    assert assessment.external_authority_verified is False
    assert assessment.ready_for_training_run_lease is False
    assert assessment.blockers == ("terminal_authority_missing",)
    assert assessment.optimizer_start_permitted_by_this_module is False
    assert assessment.scientific_truth_changed is False


def test_zero_real_exposure_remains_blocked_even_with_exact_external_root():
    manifest, authority = _finalized(exposure=0)
    assessment = assess_terminal_launch_authority(
        manifest,
        expected_terminal_authority_sha256=authority["authority_identity_sha256"],
    )
    assert validate_launch_manifest(manifest) == ()
    assert assessment.external_authority_verified is True
    assert assessment.authorized_optimized_target_exposure == 0
    assert assessment.ready_for_training_run_lease is False
    assert assessment.blockers == ("authorized_optimized_target_exposure_not_positive",)
    with pytest.raises(ValueError, match="authorized_optimized_target_exposure_not_positive"):
        build_authorized_training_run_lease(
            manifest,
            expected_terminal_authority_sha256=authority["authority_identity_sha256"],
            run_id="run-a",
            holder_id="runner-a",
            ttl_seconds=60,
            now=NOW,
        )


def test_positive_independently_rooted_exposure_opens_only_the_lease_gate():
    manifest, authority = _finalized(exposure=1_000)
    expected = authority["authority_identity_sha256"]
    assessment = assess_terminal_launch_authority(
        manifest,
        expected_terminal_authority_sha256=expected,
    )
    assert assessment.external_authority_verified is True
    assert assessment.ready_for_training_run_lease is True
    assert assessment.optimizer_start_permitted_by_this_module is False
    assert assessment.scientific_truth_changed is False

    lease = build_authorized_training_run_lease(
        manifest,
        expected_terminal_authority_sha256=expected,
        run_id="run-a",
        holder_id="runner-a",
        ttl_seconds=60,
        now=NOW,
    )
    assert lease.manifest_sha256 == launch_manifest_sha256(manifest)


def test_candidate_cannot_self_rehash_a_substituted_terminal_root():
    manifest, authority = _finalized(exposure=1_000)
    expected = authority["authority_identity_sha256"]
    candidate = deepcopy(manifest)
    candidate["terminal_authority"]["loss_bearing_manifest_sha256"] = "f" * 64
    candidate["terminal_authority"]["authority_identity_sha256"] = terminal_authority_sha256(
        candidate["terminal_authority"]
    )
    assessment = assess_terminal_launch_authority(
        candidate,
        expected_terminal_authority_sha256=expected,
    )
    assert validate_launch_manifest(candidate) == ()
    assert assessment.terminal_authority_valid is True
    assert assessment.external_authority_verified is False
    assert assessment.ready_for_training_run_lease is False
    assert "terminal_authority_not_independently_expected" in assessment.blockers


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("authorized_optimized_target_exposure", True, "authorized_optimized_target_exposure_invalid"),
        ("foreign_pretrained_weights_used", True, "foreign_pretrained_weights_used_must_be_false"),
        ("final_test_payload_access", True, "terminal_final_test_payload_access_must_be_false"),
        ("materially_paid", True, "terminal_materially_paid_must_be_false"),
        ("maximum_cost_usd", 1, "terminal_maximum_cost_usd_must_be_zero"),
    ],
)
def test_truth_widening_and_bool_integer_aliases_fail_closed(field, value, expected):
    manifest, _ = _finalized(exposure=1_000)
    candidate = deepcopy(manifest)
    candidate["terminal_authority"][field] = value
    candidate["terminal_authority"]["authority_identity_sha256"] = terminal_authority_sha256(
        candidate["terminal_authority"]
    )
    assert expected in validate_launch_manifest(candidate)


def test_manifest_identity_substitution_fails_even_after_terminal_self_rehash():
    manifest, authority = _finalized(exposure=1_000)
    expected = authority["authority_identity_sha256"]
    candidate = deepcopy(manifest)
    candidate["terminal_authority"]["portable_run_binding_sha256"] = "f" * 64
    candidate["terminal_authority"]["authority_identity_sha256"] = terminal_authority_sha256(
        candidate["terminal_authority"]
    )
    errors = validate_launch_manifest(candidate)
    assert "terminal_authority_manifest_mismatch:portable_run_binding_sha256" in errors
    assessment = assess_terminal_launch_authority(
        candidate,
        expected_terminal_authority_sha256=expected,
    )
    assert assessment.ready_for_training_run_lease is False


def test_wrong_or_malformed_external_root_never_authenticates_candidate():
    manifest, _ = _finalized(exposure=1_000)
    wrong = assess_terminal_launch_authority(
        manifest,
        expected_terminal_authority_sha256="f" * 64,
    )
    assert wrong.external_authority_verified is False
    assert wrong.ready_for_training_run_lease is False
    assert "terminal_authority_not_independently_expected" in wrong.blockers

    malformed = assess_terminal_launch_authority(
        manifest,
        expected_terminal_authority_sha256="not-a-sha",
    )
    assert malformed.external_authority_verified is False
    assert malformed.ready_for_training_run_lease is False
    assert "expected_terminal_authority_sha256_invalid" in malformed.blockers
