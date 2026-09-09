"""Fail-closed late binding for learned-20M terminal data and carrier authority.

This module deliberately does not implement split, tokenizer, packing, unique-loss,
readiness, recipe, or training semantics. It binds already-produced terminal,
self-hashed authority envelopes into one text-free D10 launch-input authority that
can be consumed by the existing readiness/recipe layers.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

AUTHORITY_SCHEMA = "12-6.learned20m-launch-input-component.v1"
OUTPUT_SCHEMA = "12-6.learned20m-launch-input-authority.v1"
_TERMINAL_SUCCESS = "TERMINAL_SUCCESS"
_ALLOWED_STATUSES = {_TERMINAL_SUCCESS, "BLOCKED", "PENDING", "FAILURE"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")

_COMPONENT_FIELDS = {
    "split": {
        "schema_version", "kind", "status", "split_identity_sha256",
        "selected_training_membership_sha256", "final_test_outcomes_accessed",
        "training_executed", "paid_compute_used", "identity_sha256",
    },
    "tokenizer": {
        "schema_version", "kind", "status", "tokenizer_identity_sha256",
        "split_identity_sha256", "final_test_outcomes_accessed", "training_executed",
        "paid_compute_used", "identity_sha256",
    },
    "packing": {
        "schema_version", "kind", "status", "packing_identity_sha256",
        "split_identity_sha256", "tokenizer_identity_sha256",
        "packed_training_payload_sha256", "final_test_outcomes_accessed",
        "training_executed", "paid_compute_used", "identity_sha256",
    },
    "two_clean_build": {
        "schema_version", "kind", "status", "packing_identity_sha256",
        "build_a_authority_sha256", "build_b_authority_sha256",
        "build_a_payload_sha256", "build_b_payload_sha256", "byte_identical",
        "final_test_outcomes_accessed", "training_executed", "paid_compute_used",
        "identity_sha256",
    },
    "unique_loss": {
        "schema_version", "kind", "status", "unique_loss_ledger_identity_sha256",
        "packing_identity_sha256", "packed_training_payload_sha256",
        "unique_causal_loss_positions", "replay_generated_unique_positions",
        "padding_generated_unique_positions", "replacement_generated_unique_positions",
        "final_test_outcomes_accessed", "training_executed", "paid_compute_used",
        "identity_sha256",
    },
    "carrier": {
        "schema_version", "kind", "status", "git_sha", "model_spec_sha256",
        "init_spec_sha256", "ci_head_sha", "ci_state", "ci_conclusion",
        "final_test_outcomes_accessed", "training_executed", "paid_compute_used",
        "identity_sha256",
    },
}

_EXPECTED_FIELDS = {
    "split_component_identity_sha256", "tokenizer_component_identity_sha256",
    "packing_component_identity_sha256", "two_clean_build_component_identity_sha256",
    "unique_loss_component_identity_sha256", "carrier_component_identity_sha256",
    "carrier_git_sha", "model_spec_sha256", "init_spec_sha256",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "text", "payload_text", "raw_payload", "final_test_outcome", "answer",
    "prediction", "score", "logit",
)


class LaunchInputAuthorityError(RuntimeError):
    """Raised when an authority envelope violates the fail-closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchInputAuthorityError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return value


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None,
        f"{label} must be lowercase 40-hex Git SHA",
    )
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _verify_text_free(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            _require(isinstance(raw_key, str), f"{path} contains a non-text key")
            key = raw_key.lower()
            for fragment in _FORBIDDEN_KEY_FRAGMENTS:
                _require(
                    fragment not in key or key == "final_test_outcomes_accessed",
                    f"forbidden durable field at {path}.{raw_key}",
                )
            _verify_text_free(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_text_free(child, f"{path}[{index}]")


def _verify_component(
    kind: str, raw: Mapping[str, Any], expected_identity: str
) -> dict[str, Any]:
    _require(kind in _COMPONENT_FIELDS, f"unknown component kind: {kind}")
    _require(isinstance(raw, Mapping), f"{kind} authority must be an object")
    value = dict(raw)
    _require(set(value) == _COMPONENT_FIELDS[kind], f"{kind} authority schema drift")
    _require(value["schema_version"] == AUTHORITY_SCHEMA, f"{kind} schema version drift")
    _require(value["kind"] == kind, f"{kind} authority kind mismatch")
    _require(value["status"] in _ALLOWED_STATUSES, f"{kind} status is invalid")
    for key in ("final_test_outcomes_accessed", "training_executed", "paid_compute_used"):
        _require(value[key] is False, f"{kind} widens forbidden truth boundary: {key}")
    _verify_text_free(value, kind)
    claimed = _require_sha256(value["identity_sha256"], f"{kind}.identity_sha256")
    expected = _require_sha256(expected_identity, f"expected {kind} identity")
    _require(claimed == expected, f"{kind} authority identity is not independently expected")
    body = deepcopy(value)
    body.pop("identity_sha256")
    _require(
        _sha256_bytes(_canonical_bytes(body)) == claimed,
        f"{kind} authority self-hash mismatch",
    )
    return value


def _verify_expected(expected: Mapping[str, Any]) -> dict[str, str]:
    _require(isinstance(expected, Mapping), "expected authorities must be an object")
    value = dict(expected)
    _require(set(value) == _EXPECTED_FIELDS, "expected authority schema drift")
    for key in _EXPECTED_FIELDS - {"carrier_git_sha"}:
        _require_sha256(value[key], f"expected.{key}")
    _require_git_sha(value["carrier_git_sha"], "expected.carrier_git_sha")
    return value


def _blocked_reasons(components: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons = []
    for kind in (
        "split", "tokenizer", "packing", "two_clean_build", "unique_loss", "carrier"
    ):
        if components[kind]["status"] != _TERMINAL_SUCCESS:
            reasons.append(f"{kind}_not_terminal_success")
    carrier = components["carrier"]
    if carrier["ci_state"] != "COMPLETED":
        reasons.append("carrier_ci_not_completed")
    if carrier["ci_conclusion"] != "SUCCESS":
        reasons.append("carrier_ci_not_success")
    two_build = components["two_clean_build"]
    if two_build["byte_identical"] is not True:
        reasons.append("two_clean_build_not_byte_identical")
    if components["unique_loss"]["unique_causal_loss_positions"] <= 0:
        reasons.append("unique_causal_loss_positions_not_positive")
    return sorted(set(reasons))


def bind_learned20m_launch_input_authority(
    *,
    split_authority: Mapping[str, Any],
    tokenizer_authority: Mapping[str, Any],
    packing_authority: Mapping[str, Any],
    two_clean_build_authority: Mapping[str, Any],
    unique_loss_authority: Mapping[str, Any],
    carrier_authority: Mapping[str, Any],
    expected: Mapping[str, Any],
    requested_unique_optimized_target_exposure: int,
) -> dict[str, Any]:
    """Bind upstream identities into a non-authorizing D10 launch input.

    ``expected`` is intentionally separate from every authority envelope. Callers
    must source it from independent live/control authority; deriving it from the
    supplied envelopes defeats the API contract and is not accepted as evidence.
    """
    exp = _verify_expected(expected)
    requested = _require_nonnegative_int(
        requested_unique_optimized_target_exposure,
        "requested_unique_optimized_target_exposure",
    )
    components = {
        "split": _verify_component(
            "split", split_authority, exp["split_component_identity_sha256"]
        ),
        "tokenizer": _verify_component(
            "tokenizer", tokenizer_authority, exp["tokenizer_component_identity_sha256"]
        ),
        "packing": _verify_component(
            "packing", packing_authority, exp["packing_component_identity_sha256"]
        ),
        "two_clean_build": _verify_component(
            "two_clean_build",
            two_clean_build_authority,
            exp["two_clean_build_component_identity_sha256"],
        ),
        "unique_loss": _verify_component(
            "unique_loss", unique_loss_authority, exp["unique_loss_component_identity_sha256"]
        ),
        "carrier": _verify_component(
            "carrier", carrier_authority, exp["carrier_component_identity_sha256"]
        ),
    }

    split = components["split"]
    tokenizer = components["tokenizer"]
    packing = components["packing"]
    two_build = components["two_clean_build"]
    unique_loss = components["unique_loss"]
    carrier = components["carrier"]

    split_id = _require_sha256(split["split_identity_sha256"], "split identity")
    tokenizer_id = _require_sha256(tokenizer["tokenizer_identity_sha256"], "tokenizer identity")
    packing_id = _require_sha256(packing["packing_identity_sha256"], "packing identity")
    packed_payload = _require_sha256(
        packing["packed_training_payload_sha256"], "packed training payload"
    )
    ledger_id = _require_sha256(
        unique_loss["unique_loss_ledger_identity_sha256"], "unique-loss ledger identity"
    )

    _require(tokenizer["split_identity_sha256"] == split_id, "tokenizer/split binding mismatch")
    _require(packing["split_identity_sha256"] == split_id, "packing/split binding mismatch")
    _require(
        packing["tokenizer_identity_sha256"] == tokenizer_id,
        "packing/tokenizer binding mismatch",
    )
    _require(
        two_build["packing_identity_sha256"] == packing_id,
        "two-clean-build/packing binding mismatch",
    )
    _require(
        unique_loss["packing_identity_sha256"] == packing_id,
        "unique-loss/packing binding mismatch",
    )
    _require(
        unique_loss["packed_training_payload_sha256"] == packed_payload,
        "unique-loss/packed-payload binding mismatch",
    )
    for label in ("build_a_payload_sha256", "build_b_payload_sha256"):
        _require_sha256(two_build[label], f"two_clean_build.{label}")
    _require(
        two_build["build_a_payload_sha256"] == packed_payload,
        "clean build A payload differs from packing authority",
    )
    _require(
        two_build["build_b_payload_sha256"] == packed_payload,
        "clean build B payload differs from packing authority",
    )
    _require_sha256(two_build["build_a_authority_sha256"], "build A authority")
    _require_sha256(two_build["build_b_authority_sha256"], "build B authority")

    unique_positions = _require_nonnegative_int(
        unique_loss["unique_causal_loss_positions"], "unique_causal_loss_positions"
    )
    for key in (
        "replay_generated_unique_positions",
        "padding_generated_unique_positions",
        "replacement_generated_unique_positions",
    ):
        manufactured = _require_nonnegative_int(unique_loss[key], key)
        _require(manufactured == 0, f"{key} must be zero")
    _require(
        requested <= unique_positions,
        "requested unique exposure exceeds terminal unique causal-loss supply",
    )

    carrier_git_sha = _require_git_sha(carrier["git_sha"], "carrier.git_sha")
    _require(carrier_git_sha == exp["carrier_git_sha"], "carrier Git SHA substitution")
    _require(
        carrier["ci_head_sha"] == carrier_git_sha,
        "carrier CI is not bound to the exact carrier Git SHA",
    )
    _require(
        carrier["model_spec_sha256"] == exp["model_spec_sha256"],
        "ModelSpec substitution",
    )
    _require(
        carrier["init_spec_sha256"] == exp["init_spec_sha256"],
        "InitSpec substitution",
    )
    _require_sha256(carrier["model_spec_sha256"], "carrier.model_spec_sha256")
    _require_sha256(carrier["init_spec_sha256"], "carrier.init_spec_sha256")

    reasons = _blocked_reasons(components)
    if requested == 0:
        reasons.append("requested_unique_exposure_is_zero")
    reasons = sorted(set(reasons))
    state = "BOUND_READY_FOR_READINESS_GATE" if not reasons else "BLOCKED"

    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "state": state,
        "blocked_reasons": reasons,
        "component_authority_identities": {
            "split": split["identity_sha256"],
            "tokenizer": tokenizer["identity_sha256"],
            "packing": packing["identity_sha256"],
            "two_clean_build": two_build["identity_sha256"],
            "unique_loss": unique_loss["identity_sha256"],
            "carrier": carrier["identity_sha256"],
        },
        "split_identity_sha256": split_id,
        "selected_training_membership_sha256": _require_sha256(
            split["selected_training_membership_sha256"], "selected training membership"
        ),
        "tokenizer_identity_sha256": tokenizer_id,
        "packing_identity_sha256": packing_id,
        "packed_training_payload_sha256": packed_payload,
        "two_clean_build_byte_identical": two_build["byte_identical"] is True,
        "unique_loss_ledger_identity_sha256": ledger_id,
        "terminal_unique_causal_loss_positions": unique_positions,
        "requested_unique_optimized_target_exposure": requested,
        "carrier_git_sha": carrier_git_sha,
        "model_spec_sha256": carrier["model_spec_sha256"],
        "init_spec_sha256": carrier["init_spec_sha256"],
        "carrier_ci_state": carrier["ci_state"],
        "carrier_ci_conclusion": carrier["ci_conclusion"],
        "authorized_optimized_target_exposure": 0,
        "training_authorized": False,
        "readiness_gate_bypassed": False,
        "recipe_gate_bypassed": False,
        "final_test_outcomes_accessed": False,
        "tokenizer_fit_executed_by_binder": False,
        "optimizer_updates_executed_by_binder": 0,
        "model_training_executed_by_binder": False,
        "paid_compute_used_by_binder": False,
    }
    _verify_text_free(output, "output")
    output["authority_identity_sha256"] = _sha256_bytes(_canonical_bytes(output))
    return output


def verify_learned20m_launch_input_authority(authority: Mapping[str, Any]) -> None:
    """Verify deterministic self-identity and non-authorizing truth boundaries."""
    _require(isinstance(authority, Mapping), "launch-input authority must be an object")
    value = dict(authority)
    claimed = _require_sha256(
        value.get("authority_identity_sha256"), "authority_identity_sha256"
    )
    body = deepcopy(value)
    body.pop("authority_identity_sha256", None)
    _require(
        _sha256_bytes(_canonical_bytes(body)) == claimed,
        "launch-input self-hash mismatch",
    )
    _verify_text_free(value, "output")
    _require(value.get("authorized_optimized_target_exposure") == 0, "binder authorized exposure")
    _require(value.get("training_authorized") is False, "binder authorized training")
    _require(value.get("readiness_gate_bypassed") is False, "readiness gate bypassed")
    _require(value.get("recipe_gate_bypassed") is False, "recipe gate bypassed")
    _require(value.get("final_test_outcomes_accessed") is False, "final-test outcomes accessed")
    _require(value.get("tokenizer_fit_executed_by_binder") is False, "binder fit tokenizer")
    _require(value.get("optimizer_updates_executed_by_binder") == 0, "binder ran optimizer")
    _require(value.get("model_training_executed_by_binder") is False, "binder trained model")
    _require(value.get("paid_compute_used_by_binder") is False, "binder used paid compute")
