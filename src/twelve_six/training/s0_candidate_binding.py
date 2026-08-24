"""Bind D02 real-training evidence to exact D08 environment authority."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from .s0_evidence_contract import (
    SCHEMA_VERSION,
    S0EvidenceContractError,
    validate_locked_environment_evidence,
    validate_s0_training_evidence,
)

_SOURCE_SCHEMA = "12-6.s0-real-training-evidence.v1"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def bind_candidate_training_evidence(
    evidence: Mapping[str, Any],
    locked_environment_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Upgrade v1 D02 evidence into exact-candidate, locked-environment v2 evidence."""
    if evidence.get("schema_version") != _SOURCE_SCHEMA:
        raise S0EvidenceContractError("candidate binding requires D02 v1 source evidence")
    identity = evidence.get("identity")
    if not isinstance(identity, Mapping):
        raise S0EvidenceContractError("source evidence identity block missing")
    source_sha = identity.get("source_sha")
    if not isinstance(source_sha, str):
        raise S0EvidenceContractError("source evidence source SHA missing")

    environment_binding = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    bound = copy.deepcopy(dict(evidence))
    bound["schema_version"] = SCHEMA_VERSION
    bound_identity = copy.deepcopy(dict(identity))
    bound_identity["environment"] = environment_binding
    bound["identity"] = bound_identity
    bound["identity_sha256"] = _canonical_hash(bound_identity)
    bound.pop("evidence_sha256", None)
    bound["evidence_sha256"] = _canonical_hash(bound)
    validate_s0_training_evidence(bound, require_locked_environment=True)
    return bound
