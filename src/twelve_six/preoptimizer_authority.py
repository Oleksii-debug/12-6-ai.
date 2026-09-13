"""Bind independently pinned pre-optimizer authorities into a portable run packet.

This module does not authorize training.  It validates a small identity-bound
projection that must itself live inside the externally pinned readiness trust
bundle.  The projection exists only to make already-qualified D10/D04/tokenizer/
resource evidence portable without letting a candidate packet choose its own roots.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

from twelve_six.data.loss_bearing_content_binding_v1 import CONTENT_MANIFEST_SCHEMA
from twelve_six.learned20m_launch_input import LAUNCH_INPUT_SCHEMA
from twelve_six.tokenization.decision_authority import (
    DECISION as TOKENIZER_DECISION,
    SCHEMA as TOKENIZER_DECISION_SCHEMA,
)

PREOPTIMIZER_SCHEMA = "12-6.r01-preoptimizer-authority-carrier.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ROOT_KEYS = {
    "schema",
    "launch_input",
    "loss_bearing_content",
    "tokenizer_decision",
    "resource_evidence",
}
_LAUNCH_KEYS = {
    "schema",
    "authority_identity_sha256",
    "tokenizer_identity_sha256",
    "packing_identity_sha256",
    "unique_loss_ledger_identity_sha256",
    "one_pass_unique_nonignored_causal_loss_positions",
}
_CONTENT_KEYS = {
    "schema",
    "manifest_identity_sha256",
    "tokenizer_identity_sha256",
    "packing_identity_sha256",
    "unique_loss_ledger_identity_sha256",
    "one_pass_unique_nonignored_causal_loss_positions",
}
_TOKENIZER_KEYS = {
    "schema",
    "decision",
    "decision_identity_sha256",
    "tokenizer_identity_sha256",
}
_RESOURCE_KEYS = {
    "evidence_identity_sha256",
    "resource_class",
    "mechanics_scope",
    "median_causal_targets_per_second",
    "process_hwm_mib_approx",
    "mechanics_only_lower_bound_seconds",
    "cross_host_extrapolation_allowed",
    "paid_compute_used",
    "authorized_optimized_target_exposure",
    "optimizer_updates_executed_on_real_targets",
}


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON identity used by the packet binding."""
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(errors: list[str], value: Any, keys: set[str], prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{prefix}_must_be_object")
        return {}
    missing = keys - set(value)
    unexpected = set(value) - keys
    errors.extend(f"{prefix}_{key}_missing" for key in sorted(missing))
    errors.extend(f"{prefix}_{key}_unexpected" for key in sorted(unexpected))
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def validate_preoptimizer_authorities(value: Any) -> list[str]:
    """Validate the closed-world zero-credit authority projection."""
    errors: list[str] = []
    root = _mapping(errors, value, _ROOT_KEYS, "preoptimizer")
    if root.get("schema") != PREOPTIMIZER_SCHEMA:
        errors.append("preoptimizer_schema_mismatch")

    launch = _mapping(errors, root.get("launch_input"), _LAUNCH_KEYS, "launch_input")
    if launch.get("schema") != LAUNCH_INPUT_SCHEMA:
        errors.append("launch_input_schema_mismatch")
    for field in (
        "authority_identity_sha256",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
        "unique_loss_ledger_identity_sha256",
    ):
        if not _is_sha256(launch.get(field)):
            errors.append(f"launch_input_{field}_invalid")
    if not _positive_int(launch.get("one_pass_unique_nonignored_causal_loss_positions")):
        errors.append("launch_input_unique_loss_positions_invalid")

    content = _mapping(
        errors,
        root.get("loss_bearing_content"),
        _CONTENT_KEYS,
        "loss_bearing_content",
    )
    if content.get("schema") != CONTENT_MANIFEST_SCHEMA:
        errors.append("loss_bearing_content_schema_mismatch")
    for field in (
        "manifest_identity_sha256",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
        "unique_loss_ledger_identity_sha256",
    ):
        if not _is_sha256(content.get(field)):
            errors.append(f"loss_bearing_content_{field}_invalid")
    if not _positive_int(content.get("one_pass_unique_nonignored_causal_loss_positions")):
        errors.append("loss_bearing_content_unique_loss_positions_invalid")

    tokenizer = _mapping(
        errors,
        root.get("tokenizer_decision"),
        _TOKENIZER_KEYS,
        "tokenizer_decision",
    )
    if tokenizer.get("schema") != TOKENIZER_DECISION_SCHEMA:
        errors.append("tokenizer_decision_schema_mismatch")
    if tokenizer.get("decision") != TOKENIZER_DECISION:
        errors.append("tokenizer_decision_must_retain_byte_baseline")
    for field in ("decision_identity_sha256", "tokenizer_identity_sha256"):
        if not _is_sha256(tokenizer.get(field)):
            errors.append(f"tokenizer_decision_{field}_invalid")

    resource = _mapping(
        errors,
        root.get("resource_evidence"),
        _RESOURCE_KEYS,
        "resource_evidence",
    )
    if not _is_sha256(resource.get("evidence_identity_sha256")):
        errors.append("resource_evidence_identity_sha256_invalid")
    if resource.get("resource_class") != "LOCAL_FREE":
        errors.append("resource_evidence_resource_class_must_be_local_free")
    if resource.get("mechanics_scope") != "forward+causal_ce+backward_only":
        errors.append("resource_evidence_mechanics_scope_mismatch")
    for field in (
        "median_causal_targets_per_second",
        "process_hwm_mib_approx",
        "mechanics_only_lower_bound_seconds",
    ):
        if not _positive_finite_number(resource.get(field)):
            errors.append(f"resource_evidence_{field}_invalid")
    if resource.get("cross_host_extrapolation_allowed") is not False:
        errors.append("resource_evidence_cross_host_extrapolation_must_be_false")
    if resource.get("paid_compute_used") is not False:
        errors.append("resource_evidence_paid_compute_used_must_be_false")
    exposure = resource.get("authorized_optimized_target_exposure")
    if not isinstance(exposure, int) or isinstance(exposure, bool) or exposure != 0:
        errors.append("resource_evidence_authorized_exposure_must_be_zero")
    updates = resource.get("optimizer_updates_executed_on_real_targets")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates != 0:
        errors.append("resource_evidence_optimizer_updates_must_be_zero")

    return sorted(set(errors))


def bind_preoptimizer_to_packet(
    packet: Any,
    preoptimizer: Any,
    *,
    trusted_readiness_bundle_sha256: Any,
) -> dict[str, Any]:
    """Cross-bind trusted pre-optimizer roots to the packet's exact identities.

    ``trusted_readiness_bundle_sha256`` is independently supplied by the caller.
    This function never derives that expected trust root from the candidate packet.
    """
    errors = validate_preoptimizer_authorities(preoptimizer)
    if errors:
        raise ValueError("invalid preoptimizer authorities: " + ",".join(errors))
    if not _is_sha256(trusted_readiness_bundle_sha256):
        raise ValueError("trusted readiness bundle identity must be 64 lowercase hex")
    if not isinstance(packet, dict):
        raise ValueError("portable packet must be an object")

    identities = packet.get("identities")
    recipe = packet.get("recipe")
    resource = packet.get("resource")
    binding = packet.get("binding")
    if not isinstance(identities, dict) or not isinstance(recipe, dict):
        raise ValueError("portable packet identities/recipe missing")
    if not isinstance(resource, dict) or not isinstance(binding, dict):
        raise ValueError("portable packet resource/binding missing")

    launch = preoptimizer["launch_input"]
    content = preoptimizer["loss_bearing_content"]
    tokenizer = preoptimizer["tokenizer_decision"]
    measured = preoptimizer["resource_evidence"]

    expected_pairs = (
        ("tokenizer_sha256", "tokenizer_identity_sha256"),
        ("packing_sha256", "packing_identity_sha256"),
        ("unique_loss_ledger_sha256", "unique_loss_ledger_identity_sha256"),
    )
    for packet_field, authority_field in expected_pairs:
        packet_value = identities.get(packet_field)
        if launch.get(authority_field) != packet_value:
            raise ValueError(f"launch_input_{authority_field}_packet_mismatch")
        if content.get(authority_field) != packet_value:
            raise ValueError(f"loss_bearing_content_{authority_field}_packet_mismatch")

    if tokenizer.get("tokenizer_identity_sha256") != identities.get("tokenizer_sha256"):
        raise ValueError("tokenizer_decision_identity_packet_mismatch")

    available = recipe.get("available_unique_loss_positions")
    if launch.get("one_pass_unique_nonignored_causal_loss_positions") != available:
        raise ValueError("launch_input_unique_loss_positions_packet_mismatch")
    if content.get("one_pass_unique_nonignored_causal_loss_positions") != available:
        raise ValueError("loss_bearing_content_unique_loss_positions_packet_mismatch")

    if resource.get("resource_class") != "LOCAL_FREE":
        raise ValueError("portable packet is not LOCAL_FREE")
    maximum_cost = resource.get("maximum_cost_usd")
    if isinstance(maximum_cost, bool) or maximum_cost not in (0, 0.0):
        raise ValueError("portable packet maximum cost is not zero")
    if measured.get("resource_class") != resource.get("resource_class"):
        raise ValueError("resource evidence class does not match portable packet")

    result = copy.deepcopy(packet)
    result_binding = result["binding"]
    result_binding["trusted_readiness_bundle_sha256"] = trusted_readiness_bundle_sha256
    result_binding["preoptimizer_authorities_sha256"] = canonical_sha256(preoptimizer)
    result_binding["preoptimizer_authorities"] = copy.deepcopy(preoptimizer)
    return result
