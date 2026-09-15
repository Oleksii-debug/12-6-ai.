from __future__ import annotations

import copy

import pytest

from twelve_six.preoptimizer_authority import (
    PREOPTIMIZER_SCHEMA,
    bind_preoptimizer_to_packet,
    canonical_sha256,
    validate_preoptimizer_authorities,
)
from twelve_six.readiness_trust_root import (
    authenticated_trusted_launch_bundle,
    trusted_readiness_bundle_sha256,
)

TOKENIZER = "1" * 64
PACKING = "2" * 64
LEDGER = "3" * 64
LAUNCH = "4" * 64
CONTENT = "5" * 64
DECISION = "6" * 64
RESOURCE = "7" * 64
POSITIONS = 20_000_000


def _preoptimizer() -> dict:
    return {
        "schema": PREOPTIMIZER_SCHEMA,
        "launch_input": {
            "schema": "12-6.learned20m-launch-input-authority.v2",
            "authority_identity_sha256": LAUNCH,
            "tokenizer_identity_sha256": TOKENIZER,
            "packing_identity_sha256": PACKING,
            "unique_loss_ledger_identity_sha256": LEDGER,
            "one_pass_unique_nonignored_causal_loss_positions": POSITIONS,
        },
        "loss_bearing_content": {
            "schema": "12-6.d04-loss-bearing-content-manifest.v2",
            "manifest_identity_sha256": CONTENT,
            "tokenizer_identity_sha256": TOKENIZER,
            "packing_identity_sha256": PACKING,
            "unique_loss_ledger_identity_sha256": LEDGER,
            "one_pass_unique_nonignored_causal_loss_positions": POSITIONS,
        },
        "tokenizer_decision": {
            "schema": "12-6.d04-learned20m-tokenizer-decision.v1",
            "decision": "RETAIN_BYTE_BASELINE",
            "decision_identity_sha256": DECISION,
            "tokenizer_identity_sha256": TOKENIZER,
        },
        "resource_evidence": {
            "evidence_identity_sha256": RESOURCE,
            "resource_class": "LOCAL_FREE",
            "mechanics_scope": "forward+causal_ce+backward_only",
            "median_causal_targets_per_second": 609.5197637935258,
            "process_hwm_mib_approx": 478.56,
            "mechanics_only_lower_bound_seconds": 32812.71779527559,
            "cross_host_extrapolation_allowed": False,
            "paid_compute_used": False,
            "authorized_optimized_target_exposure": 0,
            "optimizer_updates_executed_on_real_targets": 0,
        },
    }


def _packet() -> dict:
    return {
        "identities": {
            "tokenizer_sha256": TOKENIZER,
            "packing_sha256": PACKING,
            "unique_loss_ledger_sha256": LEDGER,
        },
        "recipe": {"available_unique_loss_positions": POSITIONS},
        "resource": {"resource_class": "LOCAL_FREE", "maximum_cost_usd": 0},
        "binding": {},
    }


def _bundle(preoptimizer: dict | None = None) -> dict:
    return {
        "schema_version": 3,
        "scientific_authorities": {},
        "verified_authorization_refs": [],
        "portable_execution": {},
        "preoptimizer_authorities": preoptimizer or _preoptimizer(),
    }


def test_valid_projection_round_trips_under_external_v3_root() -> None:
    bundle = _bundle()
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None
    resolved = authenticated_trusted_launch_bundle(
        bundle,
        expected_identity_sha256=expected,
    )
    assert resolved is not None
    scientific, refs, execution, preoptimizer = resolved
    assert scientific == set()
    assert refs == set()
    assert execution == {}
    assert preoptimizer == _preoptimizer()


def test_content_root_substitution_cannot_reuse_old_external_root() -> None:
    bundle = _bundle()
    expected = trusted_readiness_bundle_sha256(bundle)
    assert expected is not None
    tampered = copy.deepcopy(bundle)
    tampered["preoptimizer_authorities"]["loss_bearing_content"][
        "manifest_identity_sha256"
    ] = "9" * 64
    assert validate_preoptimizer_authorities(tampered["preoptimizer_authorities"]) == []
    assert trusted_readiness_bundle_sha256(tampered) != expected
    assert (
        authenticated_trusted_launch_bundle(
            tampered,
            expected_identity_sha256=expected,
        )
        is None
    )


def test_tokenizer_decision_must_use_exact_integrated_semantics() -> None:
    value = _preoptimizer()
    value["tokenizer_decision"]["decision"] = "BYTE_BASELINE_RETAINED"
    assert "tokenizer_decision_must_retain_byte_baseline" in validate_preoptimizer_authorities(
        value
    )


def test_resource_bool_int_aliases_fail_closed() -> None:
    value = _preoptimizer()
    value["resource_evidence"]["authorized_optimized_target_exposure"] = False
    assert "resource_evidence_authorized_exposure_must_be_zero" in (
        validate_preoptimizer_authorities(value)
    )


def test_packet_crossbind_rejects_same_shape_packing_substitution() -> None:
    value = _preoptimizer()
    value["launch_input"]["packing_identity_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="launch_input_packing_identity_sha256_packet_mismatch"):
        bind_preoptimizer_to_packet(
            _packet(),
            value,
            trusted_readiness_bundle_sha256="a" * 64,
        )


def test_packet_carries_external_root_and_exact_preoptimizer_projection() -> None:
    value = _preoptimizer()
    packet = bind_preoptimizer_to_packet(
        _packet(),
        value,
        trusted_readiness_bundle_sha256="a" * 64,
    )
    assert packet["binding"]["trusted_readiness_bundle_sha256"] == "a" * 64
    assert packet["binding"]["preoptimizer_authorities_sha256"] == canonical_sha256(value)
    assert packet["binding"]["preoptimizer_authorities"] == value


def test_resource_measurements_must_be_finite_positive_and_not_cross_host() -> None:
    value = _preoptimizer()
    value["resource_evidence"]["median_causal_targets_per_second"] = float("nan")
    value["resource_evidence"]["cross_host_extrapolation_allowed"] = True
    errors = validate_preoptimizer_authorities(value)
    assert "resource_evidence_median_causal_targets_per_second_invalid" in errors
    assert "resource_evidence_cross_host_extrapolation_must_be_false" in errors
