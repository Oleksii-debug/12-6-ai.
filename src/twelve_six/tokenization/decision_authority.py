"""Fail-closed learned-20M decision authority for the canonical byte tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)

SCHEMA = "12-6.d04-learned20m-tokenizer-decision.v1"
DECISION = "RETAIN_BYTE_BASELINE"
STATUS = "TERMINAL_TOKENIZER_DECISION"

_REPORT_KEYS = {
    "schema",
    "status",
    "decision",
    "corpus_authority_sha256",
    "split_authority_sha256",
    "tokenizer_version",
    "tokenizer_config_sha256",
    "tokenizer_vocab_sha256",
    "vocab_size",
    "normalization",
    "encoding",
    "tokenizer_fit_executed",
    "training_authorized_by_this_report",
    "compute_authorized_by_this_report",
    "authorized_optimized_target_exposure",
    "decision_identity_sha256",
}


class TokenizerDecisionError(ValueError):
    """Raised when terminal tokenizer-decision evidence fails closed."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TokenizerDecisionError("authority must be canonical-JSON serializable") from exc


def authority_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical JSON identity used to bind an upstream authority object."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _require_terminal_authority(
    authority: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_status: str,
    role: str,
) -> str:
    if not isinstance(authority, Mapping):
        raise TokenizerDecisionError(f"{role} authority must be a mapping")
    expected = _require_sha256(expected_sha256, field=f"expected_{role}_sha256")
    if not isinstance(expected_status, str) or not expected_status:
        raise TokenizerDecisionError(f"{role} expected terminal status must be non-empty")
    observed = authority_sha256(authority)
    if observed != expected:
        raise TokenizerDecisionError(f"{role} authority identity mismatch")
    if authority.get("status") != expected_status:
        raise TokenizerDecisionError(f"{role} authority is not terminal")
    return observed


def bind_byte_baseline_decision(
    corpus_authority: Mapping[str, Any],
    split_authority: Mapping[str, Any],
    *,
    expected_corpus_sha256: str,
    expected_split_sha256: str,
    corpus_terminal_status: str,
    split_terminal_status: str,
) -> dict[str, Any]:
    """Bind terminal corpus/split evidence to the already-canonical byte tokenizer.

    Expected upstream SHA-256 identities are supplied out-of-packet. This function
    never trains or fits a tokenizer and grants no data, training, compute, or exposure
    authority.
    """

    corpus_sha = _require_terminal_authority(
        corpus_authority,
        expected_sha256=expected_corpus_sha256,
        expected_status=corpus_terminal_status,
        role="corpus",
    )
    split_sha = _require_terminal_authority(
        split_authority,
        expected_sha256=expected_split_sha256,
        expected_status=split_terminal_status,
        role="split",
    )

    tokenizer = ByteTokenizer().identity
    if tokenizer.version != BYTE_TOKENIZER_VERSION:
        raise TokenizerDecisionError("canonical byte tokenizer version drift")
    if tokenizer.config_sha256 != BYTE_TOKENIZER_HASH:
        raise TokenizerDecisionError("canonical byte tokenizer config identity drift")
    if tokenizer.vocab_sha256 != BYTE_VOCAB_HASH:
        raise TokenizerDecisionError("canonical byte tokenizer vocab identity drift")

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "corpus_authority_sha256": corpus_sha,
        "split_authority_sha256": split_sha,
        "tokenizer_version": tokenizer.version,
        "tokenizer_config_sha256": tokenizer.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.vocab_sha256,
        "vocab_size": tokenizer.vocab_size,
        "normalization": tokenizer.normalization,
        "encoding": tokenizer.encoding,
        "tokenizer_fit_executed": False,
        "training_authorized_by_this_report": False,
        "compute_authorized_by_this_report": False,
        "authorized_optimized_target_exposure": 0,
    }
    return {**core, "decision_identity_sha256": authority_sha256(core)}


def verify_byte_baseline_decision(
    report: Mapping[str, Any],
    corpus_authority: Mapping[str, Any],
    split_authority: Mapping[str, Any],
    *,
    expected_corpus_sha256: str,
    expected_split_sha256: str,
    corpus_terminal_status: str,
    split_terminal_status: str,
) -> None:
    """Verify the report and rebind the two externally expected upstream objects."""

    if not isinstance(report, Mapping):
        raise TokenizerDecisionError("report must be a mapping")
    if set(report) != _REPORT_KEYS:
        raise TokenizerDecisionError("report fields are not closed-world")
    if report["schema"] != SCHEMA or report["status"] != STATUS:
        raise TokenizerDecisionError("report schema/status mismatch")
    if report["decision"] != DECISION:
        raise TokenizerDecisionError("unexpected tokenizer decision")

    corpus_sha = _require_terminal_authority(
        corpus_authority,
        expected_sha256=expected_corpus_sha256,
        expected_status=corpus_terminal_status,
        role="corpus",
    )
    split_sha = _require_terminal_authority(
        split_authority,
        expected_sha256=expected_split_sha256,
        expected_status=split_terminal_status,
        role="split",
    )
    if report["corpus_authority_sha256"] != corpus_sha:
        raise TokenizerDecisionError("report corpus authority mismatch")
    if report["split_authority_sha256"] != split_sha:
        raise TokenizerDecisionError("report split authority mismatch")

    tokenizer = ByteTokenizer().identity
    expected_tokenizer = {
        "tokenizer_version": tokenizer.version,
        "tokenizer_config_sha256": tokenizer.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.vocab_sha256,
        "vocab_size": tokenizer.vocab_size,
        "normalization": tokenizer.normalization,
        "encoding": tokenizer.encoding,
    }
    for key, expected in expected_tokenizer.items():
        if report[key] != expected:
            raise TokenizerDecisionError(f"report {key} drift")

    if report["tokenizer_fit_executed"] is not False:
        raise TokenizerDecisionError("byte-baseline decision cannot claim tokenizer fitting")
    if report["training_authorized_by_this_report"] is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize training")
    if report["compute_authorized_by_this_report"] is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize compute")
    if isinstance(report["authorized_optimized_target_exposure"], bool):
        raise TokenizerDecisionError("optimized-target exposure cannot be bool")
    if report["authorized_optimized_target_exposure"] != 0:
        raise TokenizerDecisionError("tokenizer decision cannot authorize exposure")

    supplied_identity = _require_sha256(
        report["decision_identity_sha256"], field="decision_identity_sha256"
    )
    core = {key: report[key] for key in report if key != "decision_identity_sha256"}
    if authority_sha256(core) != supplied_identity:
        raise TokenizerDecisionError("decision report identity mismatch")
