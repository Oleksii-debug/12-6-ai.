"""Plan 9 optional scale decisions; consumes Plan 7 contracts, never launches training.

This is not an optimizer, admission simulator, owner authorization, or champion promoter.
Only honest NOT_ACTIVATED decisions are supported until independently authorized
production data, model, evaluation, backend and budget evidence exists.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# Exact accepted-main Git blob SHA-1 for the Plan 7 source providers.
_PLAN7 = {
    17: ("src/twelve_six/large_scale_paths.py",
         "699150eea29633d94aa6853acba216c52c41e113", ("3B", "7B", "13B")),
}
_FIELDS = frozenset({"schema_version", "plan", "section", "outcome", "tiers",
                     "producer_path", "producer_git_blob_sha1", "reason_codes",
                     "activated", "trained", "production_candidate", "champion_promoted",
                     "training_authorized", "compute_authorized", "release_blocking",
                     "activation_prerequisites", "receipt_sha256"})
_PREREQUISITES = (
    "OWNER_EXPLICIT_ACTIVATION_AND_BUDGET",
    "ACCEPTED_PRODUCTION_CORPUS_TOKENIZER_EXPOSURES",
    "TERMINAL_BASE_AND_PARENT_MODEL_LINEAGE",
    "ACTUAL_BACKEND_CHECKPOINT_AND_SAME_RUN_RECOVERY",
    "INDEPENDENT_FROZEN_EVALUATION_AND_PROMOTION",
)


class EvolutionDecisionDenied(ValueError):
    """Malformed or forged campaign decision is not a scale authorization."""


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def not_activated(section: int, *, owner_activation: bool = False) -> dict[str, Any]:
    """Produce a reproducible nonactivation decision, not a resource estimate."""
    if type(section) is not int or section not in _PLAN7:
        raise EvolutionDecisionDenied("unsupported Plan9 section")
    if type(owner_activation) is not bool or owner_activation:
        raise EvolutionDecisionDenied("activation needs a separate owner-approved campaign")
    provider, blob, tiers = _PLAN7[section]
    packet: dict[str, Any] = {
        "schema_version": 1, "plan": 9, "section": section,
        "outcome": "NOT_ACTIVATED", "tiers": list(tiers),
        "producer_path": provider, "producer_git_blob_sha1": blob,
        "reason_codes": ["NO_EXPLICIT_OWNER_SCALE_ACTIVATION", "UPSTREAM_PRODUCTION_BASE_MISSING"],
        "activated": False, "trained": False,
        "production_candidate": False, "champion_promoted": False,
        "training_authorized": False, "compute_authorized": False,
        "release_blocking": False,
        "activation_prerequisites": list(_PREREQUISITES),
    }
    packet["receipt_sha256"] = _digest(packet)
    return packet


def verify_not_activated(packet: Mapping[str, Any]) -> None:
    """Fail closed across restart, JSON roundtrip, tamper and tier substitution."""
    if type(packet) is not dict or frozenset(packet) != _FIELDS:
        raise EvolutionDecisionDenied("unexpected decision envelope")
    section = packet.get("section")
    if type(section) is not int or section not in _PLAN7:
        raise EvolutionDecisionDenied("unrecognized decision section")
    expected = not_activated(section)
    if packet != expected or any(type(packet[k]) is not bool for k in (
        "activated", "trained", "production_candidate", "champion_promoted",
        "training_authorized", "compute_authorized", "release_blocking"
    )):
        raise EvolutionDecisionDenied("decision forged or out of date")
