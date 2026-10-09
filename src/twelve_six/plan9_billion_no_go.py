"""Plan 9 Section 16: evidence-bound 1B NO_GO; no training or GO authority.

Consumes accepted Plan-7 systems gate as a provenance pointer and Plan-9 Section-20
campaign economics as a verified *context*, not as measured 1B economics.
The absence of production upstream is an admission STOP, never a mock 1B run.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from twelve_six.plan9_campaign_economics import verify_campaign_economics_receipt
from twelve_six.plan9_optional_evolution import not_activated

PLAN7_GATE_PATH = "src/twelve_six/billion_systems_gate.py"
PLAN7_GATE_GIT_BLOB_SHA1 = "58f31c8db03cdc2d32b2db9e96f9ce6c91e67fa3"
READINESS_GIT_BLOB_SHA1 = "753c906ef053b997f4518ad825688dc03037ea73"
S20_ECONOMICS_RECEIPT_SHA256 = (
    "73d79c59e73f9b71c81b556237a1d756bdfa18c8aacb3104c502bbe32ece2487"
)
_FIELDS = frozenset({
    "schema_version", "plan", "section", "tier", "decision", "reason_codes",
    "plan7_systems_source_path", "plan7_systems_source_git_blob_sha1",
    "readiness_git_blob_sha1", "readiness_sha256", "economics_receipt_sha256",
    "systems_admission", "one_b_cost_observation", "one_b_capacity_observation",
    "learned_parent_200m", "real_backend", "candidate", "champion_promoted",
    "training_authorized", "compute_authorized", "launch_authorized", "receipt_sha256",
})
_REASONS = [
    "NO_ACCEPTED_LEARNED_20M_BASE_OR_200M_PARENT",
    "NO_PRODUCTION_CORPUS_TOKENIZER_LOSS_LEDGER",
    "NO_1B_PHYSICAL_SYSTEMS_ADMISSION",
    "NO_MEASURED_1B_RESOURCE_OR_ECONOMIC_EVIDENCE",
    "NO_EXPLICIT_TRAINING_OR_COMPUTE_AUTHORITY",
]


class BillionDecisionDenied(ValueError):
    """Invalid or outdated inputs must never be treated as a current NO_GO receipt."""


def _sha(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BillionDecisionDenied("noncanonical evidence") from exc
    return hashlib.sha256(encoded).hexdigest()


def _blocked_readiness(readiness: Mapping[str, Any]) -> bool:
    if type(readiness) is not dict or readiness.get("schema_version") != 1:
        return False
    if readiness.get("campaign_id") != "R01-LEARNED-20M-LAUNCH-V1":
        return False
    evidence = readiness.get("evidence")
    if type(evidence) is not dict:
        return False
    try:
        corpus = evidence["corpus"]
        tokenizer = evidence["tokenizer"]
        ledger = evidence["loss_ledger"]
        pilot = evidence["bounded_pilot"]
        return (
            corpus["manifest_sha256"] is None
            and corpus["packing_sha256"] is None
            and corpus["authority"] is None
            and tokenizer["identity_sha256"] is None
            and tokenizer["authority"] is None
            and type(ledger["unique_causal_loss_positions"]) is int
            and ledger["unique_causal_loss_positions"] == 0
            and ledger["data_budget_status"] == "BLOCKED"
            and pilot["status"] == "NOT_RUN"
            and evidence["training_recipe"]["status"] == "BLOCKED"
            and evidence["compute_authorization"]["status"] == "NOT_AUTHORIZED"
            and evidence["training_authorization"]["status"] == "NOT_AUTHORIZED"
            and evidence["cost_envelope"]["status"] == "NOT_ESTIMATED"
        )
    except (KeyError, TypeError):
        return False


def decide_1b_no_go(readiness: Mapping[str, Any], economics: Mapping[str, Any]
                    ) -> dict[str, Any]:
    """Decision for *current absent* evidence; no production 1B promotion/launch.

    If source evidence changes, this function fails rather than indefinitely
    restamping NO_GO from an obsolete snapshot. A future GO requires a separate
    real Plan-7 systems admission, terminal predecessor, budget and audit.
    """
    if not _blocked_readiness(readiness):
        raise BillionDecisionDenied("R01 production prerequisite truth changed or invalid")
    try:
        verify_campaign_economics_receipt(
            economics, not_activated(19),
            expected_receipt_sha256=S20_ECONOMICS_RECEIPT_SHA256,
        )
    except ValueError as exc:
        raise BillionDecisionDenied("invalid pinned campaign economics context") from exc
    if economics["decision"] != "NO_GO":
        raise BillionDecisionDenied("current economics context not NO_GO")
    packet: dict[str, Any] = {
        "schema_version": 1, "plan": 9, "section": 16, "tier": "1B",
        "decision": "NO_GO", "reason_codes": list(_REASONS),
        "plan7_systems_source_path": PLAN7_GATE_PATH,
        "plan7_systems_source_git_blob_sha1": PLAN7_GATE_GIT_BLOB_SHA1,
        "readiness_git_blob_sha1": READINESS_GIT_BLOB_SHA1,
        "readiness_sha256": _sha(readiness),
        "economics_receipt_sha256": S20_ECONOMICS_RECEIPT_SHA256,
        "systems_admission": "NOT_EXECUTED_PRODUCTION", "one_b_cost_observation": None,
        "one_b_capacity_observation": None, "learned_parent_200m": None,
        "real_backend": False, "candidate": False, "champion_promoted": False,
        "training_authorized": False, "compute_authorized": False,
        "launch_authorized": False,
    }
    packet["receipt_sha256"] = _sha(packet)
    return packet


def verify_1b_no_go(packet: Mapping[str, Any], readiness: Mapping[str, Any],
                    economics: Mapping[str, Any], *, expected_receipt_sha256: str
                    ) -> None:
    """Independent expected hash plus full source recomputation on cold readback."""
    if type(packet) is not dict or frozenset(packet) != _FIELDS:
        raise BillionDecisionDenied("invalid decision envelope")
    if (type(expected_receipt_sha256) is not str or len(expected_receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_receipt_sha256)):
        raise BillionDecisionDenied("independent lowercase SHA256 identity required")
    if packet.get("receipt_sha256") != expected_receipt_sha256:
        raise BillionDecisionDenied("pinned decision hash mismatch")
    expected = decide_1b_no_go(readiness, economics)
    if packet != expected:
        raise BillionDecisionDenied("forged or obsolete 1B NO_GO decision")
