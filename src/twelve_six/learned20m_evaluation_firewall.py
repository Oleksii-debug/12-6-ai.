"""Fail-closed evaluation firewall for the first learned ~20M Base campaign.

This module defines authority and phase-boundary mechanics only. It never resolves
held-out payload text, scores a model, updates model state, or grants compute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from copy import deepcopy
from typing import Any

from twelve_six.learned20m_readiness import scientific_authority_token

REPOSITORY = "Oleksii-debug/12-6-ai."
SCHEMA_VERSION = "12-6.learned20m-evaluation-firewall.v1"
POLICY_ID = "D06-LEARNED20M-EVALUATION-FIREWALL-V1"
SELECTION_LOCK_AUTHORITY_ROLE = "selection_lock"
SELECTION_LOCK_AUTHORITY_SCHEMA_VERSION = "12-6.learned20m-selection-lock-authority.v1"

EVAL303_SELECTION_AUTHORITY = {
    "repository": REPOSITORY,
    "git_sha": "ac55652c92289ca2b212c72ec879ed0b38105237",
    "evidence_sha256": "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd",
    "workflow_run_id": 34413932880,
    "workflow_conclusion": "success",
    "terminal": True,
}

EVAL233_FINAL_TEST_RESERVATION_AUTHORITY = {
    "repository": REPOSITORY,
    "git_sha": "b5512b4648cb09dd052b08884dc53f291e1ce935",
    "evidence_sha256": "86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116",
    "workflow_run_id": 32957254139,
    "workflow_conclusion": "success",
    "terminal": True,
}

EVAL303_MANIFEST_BLOB_SHA1 = "c2e5b27bac541e55be2c807ea65b0e2ed77b7019"
EVAL233_SOURCE_AUTHORITY_IDENTITY_SHA256 = (
    "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f"
)
EVAL233_PRESERVED_SEED_BLOB_SHA1 = "4bfbfbf29fa9538cabda6068efd3a1fd036a9479"
CHECKPOINT_BOUNDARIES_PERCENT = (0, 10, 25, 50, 75, 90, 100)
PRIMARY_SELECTION_METRIC = "BITS_PER_BYTE"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_AUTHORITY_KEYS = {
    "repository",
    "git_sha",
    "evidence_sha256",
    "workflow_run_id",
    "workflow_conclusion",
    "terminal",
}
_POLICY_KEYS = {
    "schema_version",
    "policy_id",
    "execution_profile",
    "selection_validation",
    "final_test_reservation",
    "evaluation_protocol",
    "truth_boundary",
    "policy_identity_sha256",
}
_SELECTION_KEYS = {
    "authority",
    "manifest_git_blob_sha1",
    "purpose",
    "may_select_checkpoint",
    "may_mutate_recipe",
    "may_fit_tokenizer",
    "may_train",
    "may_update_model",
    "may_report_final_test",
}
_FINAL_KEYS = {
    "authority",
    "source_authority_identity_sha256",
    "preserved_seed_git_blob_sha1",
    "payload_access_before_selection_lock",
    "outcomes_access_before_selection_lock",
    "may_trigger_reselection",
    "may_mutate_recipe",
    "may_mutate_training_data",
    "may_update_model",
}
_PROTOCOL_KEYS = {
    "checkpoint_boundaries_percent",
    "primary_selection_metric",
    "final_test_access_requires_selection_lock",
    "selection_lock_is_immutable",
    "model_mutation_during_evaluation",
    "optimizer_updates_during_evaluation",
    "optimized_target_delta_during_evaluation",
}
_TRUTH_KEYS = {
    "training_authorized",
    "compute_authorized",
    "authorized_optimized_targets",
    "optimizer_updates_executed_by_this_package",
    "final_test_payload_consumed_by_this_package",
    "final_test_outcomes_consumed_by_this_package",
    "evaluation_payload_embedded",
}
_SELECTION_LOCK_KEYS = {
    "selection_complete",
    "selection_lock_identity_sha256",
    "selected_checkpoint_sha256",
    "selection_validation_evidence_sha256",
    "recipe_identity_sha256",
    "train_trace_identity_sha256",
    "created_from_final_test",
    "reselection_allowed",
    "optimizer_updates_after_lock_allowed",
    "final_test_payload_consumed_before_lock",
}
_TRUSTED_SELECTION_LOCK_KEYS = {
    "schema_version",
    "authority",
    "selection_lock_identity_sha256",
    "selected_checkpoint_sha256",
    "selection_validation_evidence_sha256",
    "recipe_identity_sha256",
    "train_trace_identity_sha256",
}
_SELECTION_LOCK_IDENTITY_FIELDS = (
    "selection_lock_identity_sha256",
    "selected_checkpoint_sha256",
    "selection_validation_evidence_sha256",
    "recipe_identity_sha256",
    "train_trace_identity_sha256",
)


class EvaluationFirewallError(ValueError):
    """Raised when a learned-20M evaluation authority fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationFirewallError(message)


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    keys = set(value)
    _require(keys == expected, f"{label} keys drift: {sorted(keys ^ expected)}")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_sha1(value: Any) -> bool:
    return isinstance(value, str) and _SHA1_RE.fullmatch(value) is not None


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def self_identity(policy: dict[str, Any]) -> str:
    clone = deepcopy(policy)
    clone.pop("policy_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(clone)).hexdigest()


def _validate_dynamic_authority_ref(value: Any, label: str) -> dict[str, Any]:
    ref = _require_exact_keys(value, _AUTHORITY_KEYS, label)
    _require(ref.get("repository") == REPOSITORY, f"{label} repository drift")
    _require(_is_sha1(ref.get("git_sha")), f"{label} git SHA invalid")
    _require(_is_sha256(ref.get("evidence_sha256")), f"{label} evidence SHA-256 invalid")
    run_id = ref.get("workflow_run_id")
    _require(
        isinstance(run_id, int) and not isinstance(run_id, bool) and run_id > 0,
        f"{label} workflow run invalid",
    )
    _require(ref.get("workflow_conclusion") == "success", f"{label} workflow not success")
    _require(ref.get("terminal") is True, f"{label} is not terminal")
    return ref


def _validate_authority_ref(value: Any, expected: dict[str, Any], label: str) -> None:
    ref = _validate_dynamic_authority_ref(value, label)
    _require(ref == expected, f"{label} exact authority drift")


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable policy without reading any evaluation payload."""
    root = _require_exact_keys(policy, _POLICY_KEYS, "policy")
    _require(root.get("schema_version") == SCHEMA_VERSION, "schema version drift")
    _require(root.get("policy_id") == POLICY_ID, "policy id drift")
    _require(root.get("execution_profile") == "LOCAL_FREE", "execution profile drift")

    selection = _require_exact_keys(
        root.get("selection_validation"), _SELECTION_KEYS, "selection_validation"
    )
    _validate_authority_ref(
        selection.get("authority"), EVAL303_SELECTION_AUTHORITY, "selection authority"
    )
    _require(
        selection.get("manifest_git_blob_sha1") == EVAL303_MANIFEST_BLOB_SHA1,
        "EVAL-303 manifest blob drift",
    )
    _require(selection.get("purpose") == "selection-validation", "selection purpose drift")
    _require(selection.get("may_select_checkpoint") is True, "checkpoint selection disabled")
    for key in (
        "may_mutate_recipe",
        "may_fit_tokenizer",
        "may_train",
        "may_update_model",
        "may_report_final_test",
    ):
        _require(selection.get(key) is False, f"selection boundary weakened: {key}")

    final_test = _require_exact_keys(
        root.get("final_test_reservation"), _FINAL_KEYS, "final_test_reservation"
    )
    _validate_authority_ref(
        final_test.get("authority"),
        EVAL233_FINAL_TEST_RESERVATION_AUTHORITY,
        "final-test reservation authority",
    )
    _require(
        final_test.get("source_authority_identity_sha256")
        == EVAL233_SOURCE_AUTHORITY_IDENTITY_SHA256,
        "final-test source authority drift",
    )
    _require(
        final_test.get("preserved_seed_git_blob_sha1") == EVAL233_PRESERVED_SEED_BLOB_SHA1,
        "final-test seed blob drift",
    )
    for key in (
        "payload_access_before_selection_lock",
        "outcomes_access_before_selection_lock",
        "may_trigger_reselection",
        "may_mutate_recipe",
        "may_mutate_training_data",
        "may_update_model",
    ):
        _require(final_test.get(key) is False, f"final-test boundary weakened: {key}")

    protocol = _require_exact_keys(
        root.get("evaluation_protocol"), _PROTOCOL_KEYS, "evaluation_protocol"
    )
    _require(
        protocol.get("checkpoint_boundaries_percent") == list(CHECKPOINT_BOUNDARIES_PERCENT),
        "checkpoint/evaluation cadence drift",
    )
    _require(
        protocol.get("primary_selection_metric") == PRIMARY_SELECTION_METRIC,
        "primary selection metric drift",
    )
    _require(
        protocol.get("final_test_access_requires_selection_lock") is True,
        "final-test lock requirement weakened",
    )
    _require(
        protocol.get("selection_lock_is_immutable") is True,
        "selection lock immutability weakened",
    )
    _require(
        protocol.get("model_mutation_during_evaluation") is False,
        "evaluation may mutate model",
    )
    for key in (
        "optimizer_updates_during_evaluation",
        "optimized_target_delta_during_evaluation",
    ):
        value = protocol.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value == 0,
            f"evaluation mutation counter drift: {key}",
        )

    truth = _require_exact_keys(root.get("truth_boundary"), _TRUTH_KEYS, "truth_boundary")
    for key in ("training_authorized", "compute_authorized"):
        _require(truth.get(key) is False, f"truth boundary widened: {key}")
    for key in (
        "authorized_optimized_targets",
        "optimizer_updates_executed_by_this_package",
    ):
        value = truth.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value == 0,
            f"truth boundary counter drift: {key}",
        )
    for key in (
        "final_test_payload_consumed_by_this_package",
        "final_test_outcomes_consumed_by_this_package",
        "evaluation_payload_embedded",
    ):
        _require(truth.get(key) is False, f"truth boundary widened: {key}")

    identity = root.get("policy_identity_sha256")
    _require(_is_sha256(identity), "policy identity invalid")
    _require(identity == self_identity(root), "policy self-identity mismatch")
    return {
        "status": "PASS",
        "policy_identity_sha256": identity,
        "selection_identity_sha256": EVAL303_SELECTION_AUTHORITY["evidence_sha256"],
        "final_test_reservation_identity_sha256": (
            EVAL233_FINAL_TEST_RESERVATION_AUTHORITY["evidence_sha256"]
        ),
        "final_test_access_authorized": False,
        "training_authorized": False,
        "compute_authorized": False,
        "optimizer_updates_executed": 0,
    }


def preselection_binding(policy: dict[str, Any]) -> dict[str, Any]:
    """Produce a deterministic, non-authorizing pre-selection boundary receipt."""
    validated = validate_policy(policy)
    payload = {
        "schema_version": "12-6.learned20m-evaluation-firewall-binding.v1",
        "policy_identity_sha256": validated["policy_identity_sha256"],
        "selection_authority": deepcopy(EVAL303_SELECTION_AUTHORITY),
        "final_test_reservation_authority": deepcopy(EVAL233_FINAL_TEST_RESERVATION_AUTHORITY),
        "selection_lock_bound": False,
        "selection_validation_access_authorized": True,
        "final_test_access_authorized": False,
        "final_test_outcomes_reporting_authorized": False,
        "training_authorized": False,
        "compute_authorized": False,
        "optimizer_updates_executed": 0,
        "optimized_target_delta": 0,
    }
    payload["binding_identity_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _validate_selection_lock_shape(selection_lock: Any) -> dict[str, Any]:
    lock = _require_exact_keys(selection_lock, _SELECTION_LOCK_KEYS, "selection_lock")
    _require(lock.get("selection_complete") is True, "selection is not complete")
    for key in _SELECTION_LOCK_IDENTITY_FIELDS:
        _require(_is_sha256(lock.get(key)), f"selection lock identity invalid: {key}")
    for key in (
        "created_from_final_test",
        "reselection_allowed",
        "optimizer_updates_after_lock_allowed",
        "final_test_payload_consumed_before_lock",
    ):
        _require(lock.get(key) is False, f"selection lock boundary weakened: {key}")
    return lock


def selection_lock_identity(selection_lock: dict[str, Any]) -> str:
    """Return the canonical self-identity for an immutable selection lock."""
    lock = _validate_selection_lock_shape(selection_lock)
    payload = {
        key: lock[key]
        for key in sorted(_SELECTION_LOCK_KEYS)
        if key != "selection_lock_identity_sha256"
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def trusted_selection_lock_evidence_identity(
    trusted_selection_lock_authority: dict[str, Any],
) -> str:
    """Hash the trusted expected lock semantics that a live resolver authenticates."""
    trusted = _require_exact_keys(
        trusted_selection_lock_authority,
        _TRUSTED_SELECTION_LOCK_KEYS,
        "trusted_selection_lock_authority",
    )
    _require(
        trusted.get("schema_version") == SELECTION_LOCK_AUTHORITY_SCHEMA_VERSION,
        "trusted selection-lock schema drift",
    )
    for key in _SELECTION_LOCK_IDENTITY_FIELDS:
        _require(_is_sha256(trusted.get(key)), f"trusted selection-lock identity invalid: {key}")
    payload = {
        "schema_version": trusted["schema_version"],
        **{key: trusted[key] for key in _SELECTION_LOCK_IDENTITY_FIELDS},
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_trusted_selection_lock_authority(
    trusted_selection_lock_authority: Any,
    verified_scientific_authorities: Collection[str],
) -> tuple[dict[str, Any], str]:
    _require(
        trusted_selection_lock_authority is not None,
        "trusted selection-lock authority required",
    )
    trusted = _require_exact_keys(
        trusted_selection_lock_authority,
        _TRUSTED_SELECTION_LOCK_KEYS,
        "trusted_selection_lock_authority",
    )
    expected_evidence_identity = trusted_selection_lock_evidence_identity(trusted)
    authority = _validate_dynamic_authority_ref(
        trusted.get("authority"),
        "trusted selection-lock authority",
    )
    _require(
        authority.get("evidence_sha256") == expected_evidence_identity,
        "trusted selection-lock evidence identity mismatch",
    )
    token = scientific_authority_token(
        SELECTION_LOCK_AUTHORITY_ROLE,
        authority,
        require_workflow=True,
    )
    verified = {
        value.strip()
        for value in verified_scientific_authorities
        if isinstance(value, str) and value.strip()
    }
    _require(
        token is not None and token in verified,
        "trusted selection-lock authority unverified",
    )
    return trusted, token


def authorize_final_test_reporting(
    policy: dict[str, Any],
    selection_lock: dict[str, Any],
    trusted_selection_lock_authority: dict[str, Any] | None = None,
    *,
    verified_scientific_authorities: Collection[str] = (),
) -> dict[str, Any]:
    """Open final-test reporting only after an independently trusted immutable lock.

    The candidate lock is never trusted merely because it is well formed or self-consistent.
    Its canonical digest and all selection identities must match a separate authority packet
    whose role-bound token was verified out of packet by the caller's live resolver.
    """
    validated = validate_policy(policy)
    lock = _validate_selection_lock_shape(selection_lock)
    _require(
        lock["selection_lock_identity_sha256"] == selection_lock_identity(lock),
        "selection lock self-identity mismatch",
    )
    trusted, trusted_token = _validate_trusted_selection_lock_authority(
        trusted_selection_lock_authority,
        verified_scientific_authorities,
    )
    for key in _SELECTION_LOCK_IDENTITY_FIELDS:
        _require(
            lock[key] == trusted[key],
            f"trusted selection-lock mismatch: {key}",
        )

    receipt = {
        "schema_version": "12-6.learned20m-final-test-reporting-authorization.v1",
        "policy_identity_sha256": validated["policy_identity_sha256"],
        "selection_lock_identity_sha256": lock["selection_lock_identity_sha256"],
        "selected_checkpoint_sha256": lock["selected_checkpoint_sha256"],
        "selection_validation_evidence_sha256": lock[
            "selection_validation_evidence_sha256"
        ],
        "recipe_identity_sha256": lock["recipe_identity_sha256"],
        "train_trace_identity_sha256": lock["train_trace_identity_sha256"],
        "selection_lock_authority": deepcopy(trusted["authority"]),
        "selection_lock_authority_token": trusted_token,
        "final_test_reservation_authority": deepcopy(
            EVAL233_FINAL_TEST_RESERVATION_AUTHORITY
        ),
        "final_test_access_authorized": True,
        "final_test_outcomes_reporting_authorized": True,
        "reselection_authorized": False,
        "recipe_mutation_authorized": False,
        "training_data_mutation_authorized": False,
        "optimizer_updates_authorized": False,
    }
    receipt["authorization_identity_sha256"] = hashlib.sha256(
        _canonical_bytes(receipt)
    ).hexdigest()
    return receipt


def readiness_evaluation_evidence(
    policy: dict[str, Any], firewall_authority: dict[str, Any]
) -> dict[str, Any]:
    """Return the shape consumed by learned20m_readiness.

    ``firewall_authority`` is intentionally not trusted merely because it is supplied
    here. The existing readiness assessor still requires its role-bound token to be
    independently included in ``verified_scientific_authorities``.
    """
    validate_policy(policy)
    _require_exact_keys(firewall_authority, _AUTHORITY_KEYS, "firewall_authority")
    _require(firewall_authority.get("repository") == REPOSITORY, "firewall repo drift")
    _require(_is_sha1(firewall_authority.get("git_sha")), "firewall git SHA invalid")
    _require(
        _is_sha256(firewall_authority.get("evidence_sha256")),
        "firewall evidence SHA-256 invalid",
    )
    run_id = firewall_authority.get("workflow_run_id")
    _require(
        isinstance(run_id, int) and not isinstance(run_id, bool) and run_id > 0,
        "firewall workflow run invalid",
    )
    _require(
        firewall_authority.get("workflow_conclusion") == "success",
        "firewall workflow not success",
    )
    _require(firewall_authority.get("terminal") is True, "firewall authority not terminal")
    return {
        "status": "PASS",
        "firewall_authority": deepcopy(firewall_authority),
        "selection_validation_authority": deepcopy(EVAL303_SELECTION_AUTHORITY),
    }
