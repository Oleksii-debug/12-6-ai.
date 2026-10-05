from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from twelve_six.learned20m_training_lease import (
    LOCAL_ATOMICITY_SCOPE,
    acquire_local_training_run_lease,
    assess_training_run_lease,
    build_training_run_lease,
    canonical_json_bytes,
    launch_manifest_sha256,
    local_training_run_lease_path,
    renew_training_run_lease,
    terminate_training_run_lease,
    training_side_effect_idempotency_key,
    validate_launch_manifest,
    validate_lease_transition,
    validate_training_run_lease,
)


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
            "optimizer_scheduler_precision": "AdamW|cosine|fp32",
            "seed": 1337,
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
                "reference": "github:issue/653#comment-training",
                "evidence_sha256": "e" * 64,
            },
            "compute": {
                "reference": "github:issue/653#comment-compute",
                "evidence_sha256": "f" * 64,
            },
        },
        "execution_backend": "PROJECT_NATIVE_PYTORCH",
    }


NOW = datetime(2026, 9, 14, 0, 40, tzinfo=UTC)


def test_manifest_digest_is_canonical_and_order_independent():
    manifest = _manifest()
    reordered = {key: manifest[key] for key in reversed(manifest)}
    assert validate_launch_manifest(manifest) == ()
    assert launch_manifest_sha256(manifest) == launch_manifest_sha256(reordered)
    assert canonical_json_bytes(manifest) == canonical_json_bytes(reordered)


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda m: m["resource"].update({"maximum_cost_usd": 1}), "maximum_cost_usd_must_be_zero"),
        (lambda m: m["resource"].update({"materially_paid": True}), "materially_paid_must_be_false"),
        (
            lambda m: m["evaluation"].update({"final_test_payload_access": True}),
            "final_test_payload_access_must_be_false",
        ),
        (
            lambda m: m["authorities"]["training"].update({"decision": "TRAINING_AUTHORIZED"}),
            "training_authority_fields_mismatch",
        ),
        (
            lambda m: m["recipe"].update({"maximum_total_exposures": 19_999_999}),
            "maximum_total_exposures_below_target",
        ),
    ],
)
def test_manifest_fails_closed_on_launch_boundary_drift(mutator, expected):
    manifest = _manifest()
    mutator(manifest)
    assert expected in validate_launch_manifest(manifest)


def test_manifest_rejects_floating_point_identity_ambiguity():
    manifest = _manifest()
    manifest["recipe"]["seed"] = 1.0
    errors = validate_launch_manifest(manifest)
    assert "floating_point_forbidden:root.recipe.seed" in errors
    with pytest.raises(ValueError, match="floating_point_forbidden"):
        launch_manifest_sha256(manifest)


def test_active_lease_binds_manifest_but_never_grants_training_authority():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="gh-12345",
        holder_id="runner-1",
        now=NOW,
        ttl_seconds=3600,
    )
    assert validate_training_run_lease(lease.as_dict(), manifest) == ()
    assessment = assess_training_run_lease(manifest, lease.as_dict(), now=NOW)
    assert assessment.contract_valid is True
    assert assessment.active_lease_matches_manifest is True
    assert assessment.local_duplicate_guard_open is True
    assert assessment.local_atomicity_scope == LOCAL_ATOMICITY_SCOPE
    assert assessment.global_exclusivity_proven is False
    assert assessment.external_authority_verified is False
    assert assessment.optimizer_start_permitted_by_this_module is False
    assert assessment.scientific_truth_changed is False


def test_expired_lease_is_valid_evidence_but_not_active():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="gh-12345",
        holder_id="runner-1",
        now=NOW,
        ttl_seconds=60,
    )
    assessment = assess_training_run_lease(
        manifest,
        lease.as_dict(),
        now=NOW + timedelta(seconds=60),
    )
    assert assessment.contract_valid is True
    assert assessment.lease_valid is True
    assert assessment.local_duplicate_guard_open is False
    assert assessment.blockers == ("training_run_lease_expired",)


def test_lease_rejects_manifest_or_authority_mismatch():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="gh-12345",
        holder_id="runner-1",
        now=NOW,
        ttl_seconds=3600,
    ).as_dict()
    lease["training_authority_ref"] = "github:other"
    assert "training_authority_ref_mismatch" in validate_training_run_lease(lease, manifest)

    other = deepcopy(manifest)
    other["identities"]["corpus_manifest_sha256"] = "0" * 64
    assert "lease_manifest_sha256_mismatch" in validate_training_run_lease(lease, other)


def test_local_acquisition_is_exclusive_and_preserves_first_record(tmp_path):
    manifest = _manifest()
    first = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=3600,
    )
    second = build_training_run_lease(
        manifest,
        run_id="run-b",
        holder_id="runner-b",
        now=NOW,
        ttl_seconds=3600,
    )
    acquired = acquire_local_training_run_lease(tmp_path, manifest, first.as_dict(), now=NOW)
    denied = acquire_local_training_run_lease(tmp_path, manifest, second.as_dict(), now=NOW)

    assert acquired.acquired is True
    assert denied.acquired is False
    assert denied.blockers == ("training_run_lease_already_exists",)
    path = local_training_run_lease_path(tmp_path, launch_manifest_sha256(manifest))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["run_id"] == "run-a"


def test_existing_record_is_never_silently_replaced_after_expiry(tmp_path):
    manifest = _manifest()
    running = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=60,
    )
    first = acquire_local_training_run_lease(tmp_path, manifest, running.as_dict(), now=NOW)
    assert first.acquired is True

    later = build_training_run_lease(
        manifest,
        run_id="run-b",
        holder_id="runner-b",
        now=NOW + timedelta(hours=1),
        ttl_seconds=60,
    )
    denied = acquire_local_training_run_lease(
        tmp_path,
        manifest,
        later.as_dict(),
        now=NOW + timedelta(hours=1),
    )
    assert denied.acquired is False
    assert denied.blockers == ("training_run_lease_already_exists",)


def test_renewal_and_terminal_transition_semantics():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=300,
    )
    renewed = renew_training_run_lease(
        lease,
        now=NOW + timedelta(seconds=120),
        ttl_seconds=300,
    )
    assert renewed.renewal_sequence == 1
    assert validate_lease_transition(lease.as_dict(), renewed.as_dict(), manifest) == ()

    terminal = terminate_training_run_lease(
        renewed,
        status="COMPLETED",
        now=NOW + timedelta(seconds=180),
    )
    assert validate_lease_transition(renewed.as_dict(), terminal.as_dict(), manifest) == ()
    errors = validate_lease_transition(terminal.as_dict(), terminal.as_dict(), manifest)
    assert "terminal_lease_cannot_transition" in errors


def test_invalid_transition_cannot_change_identity_or_skip_sequence():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=300,
    )
    candidate = renew_training_run_lease(
        lease,
        now=NOW + timedelta(seconds=120),
        ttl_seconds=300,
    ).as_dict()
    candidate["run_id"] = "run-b"
    candidate["renewal_sequence"] = 7
    errors = validate_lease_transition(lease.as_dict(), candidate, manifest)
    assert "lease_transition_changes_immutable_field:run_id" in errors
    assert "renewal_sequence_must_increment_by_one" in errors


def test_idempotency_key_is_stable_and_dimension_separated():
    manifest_hash = launch_manifest_sha256(_manifest())
    kwargs = {
        "manifest_sha256": manifest_hash,
        "run_id": "run-a",
        "effect_kind": "checkpoint.publish",
        "logical_step": 42,
    }
    key = training_side_effect_idempotency_key(**kwargs)
    assert key == training_side_effect_idempotency_key(**kwargs)
    assert key != training_side_effect_idempotency_key(**{**kwargs, "logical_step": 43})
    assert key != training_side_effect_idempotency_key(**{**kwargs, "run_id": "run-b"})
    assert key != training_side_effect_idempotency_key(
        **{**kwargs, "effect_kind": "log.publish"}
    )


def test_missing_lease_and_naive_time_fail_closed():
    manifest = _manifest()
    assessment = assess_training_run_lease(manifest, None, now=NOW)
    assert assessment.manifest_valid is True
    assert assessment.lease_valid is False
    assert assessment.local_duplicate_guard_open is False
    assert assessment.blockers == ("training_run_lease_missing",)
    with pytest.raises(ValueError, match="timezone-aware"):
        assess_training_run_lease(manifest, None, now=datetime(2026, 9, 14))  # noqa: DTZ001


def test_expired_lease_cannot_be_renewed():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=60,
    )
    with pytest.raises(ValueError, match="expired_lease_cannot_be_renewed"):
        renew_training_run_lease(
            lease,
            now=NOW + timedelta(seconds=60),
            ttl_seconds=60,
        )


def test_transition_validator_cannot_resurrect_expired_running_lease():
    manifest = _manifest()
    previous = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=60,
    ).as_dict()
    candidate = dict(previous)
    candidate["renewed_at_utc"] = "2026-09-14T00:41:00Z"
    candidate["expires_at_utc"] = "2026-09-14T00:42:00Z"
    candidate["renewal_sequence"] = 1
    errors = validate_lease_transition(previous, candidate, manifest)
    assert "expired_lease_cannot_be_renewed" in errors


def test_completed_transition_must_precede_previous_lease_expiry_but_failure_is_evidence():
    manifest = _manifest()
    previous = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=60,
    ).as_dict()
    completed = dict(previous)
    completed["status"] = "COMPLETED"
    completed["terminal_at_utc"] = "2026-09-14T00:41:00Z"
    assert "completed_after_lease_expiry" in validate_lease_transition(previous, completed, manifest)

    failed = dict(previous)
    failed["status"] = "FAILED"
    failed["terminal_at_utc"] = "2026-09-14T00:41:00Z"
    assert validate_lease_transition(previous, failed, manifest) == ()


def test_json_array_enum_values_fail_closed_without_typeerror():
    manifest = _manifest()
    bad_manifest = deepcopy(manifest)
    bad_manifest["resource"]["resource_class"] = []
    assert "resource_class_not_free_only" in validate_launch_manifest(bad_manifest)

    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=60,
    ).as_dict()
    bad_lease = dict(lease)
    bad_lease["status"] = []
    assert "lease_status_invalid" in validate_training_run_lease(bad_lease, manifest)

    bad_previous = dict(lease)
    bad_previous["status"] = []
    assert "previous_lease_not_running" in validate_lease_transition(bad_previous, lease, manifest)

    bad_candidate = dict(lease)
    bad_candidate["status"] = []
    assert "lease_transition_status_invalid" in validate_lease_transition(
        lease, bad_candidate, manifest
    )


def test_future_dated_lease_is_valid_evidence_but_never_active_or_persisted(tmp_path):
    manifest = _manifest()
    future = build_training_run_lease(
        manifest,
        run_id="run-future",
        holder_id="runner-future",
        now=NOW + timedelta(hours=1),
        ttl_seconds=3600,
    )
    assessment = assess_training_run_lease(manifest, future.as_dict(), now=NOW)
    assert assessment.contract_valid is True
    assert assessment.local_duplicate_guard_open is False
    assert assessment.blockers == ("training_run_lease_not_yet_active",)

    acquired = acquire_local_training_run_lease(tmp_path, manifest, future.as_dict(), now=NOW)
    assert acquired.acquired is False
    assert acquired.blockers == ("training_run_lease_not_yet_active",)
    assert list(tmp_path.iterdir()) == []


def test_renewed_in_future_is_not_active_at_trusted_assessment_time():
    manifest = _manifest()
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        now=NOW,
        ttl_seconds=300,
    )
    renewed = renew_training_run_lease(
        lease,
        now=NOW + timedelta(seconds=120),
        ttl_seconds=300,
    )
    assessment = assess_training_run_lease(
        manifest,
        renewed.as_dict(),
        now=NOW + timedelta(seconds=60),
    )
    assert assessment.contract_valid is True
    assert assessment.local_duplicate_guard_open is False
    assert assessment.blockers == ("training_run_lease_renewed_in_future",)
