"""Fail-closed learned-20M decision authority for the canonical byte tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from twelve_six.data.balanced_split_application_v1 import (
    APPLICATION_SCHEMA,
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    SELECTION_SCHEMA,
    BalancedSplitApplicationError,
    verify_balanced_selection,
)

from .byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)

SCHEMA = "12-6.d04-learned20m-tokenizer-decision.v1"
DECISION = "RETAIN_BYTE_BASELINE"
STATUS = "TERMINAL_TOKENIZER_DECISION_ZERO_CREDIT"

_UPSTREAM_IDENTITY_FIELDS = (
    "retained_inventory_identity_sha256",
    "decontamination_authority_sha256",
    "dedup_authority_sha256",
    "balance_policy_identity_sha256",
    "balance_result_identity_sha256",
)
_APPLICATION_KEYS = {
    "schema",
    "status",
    "balanced_selection_identity_sha256",
    *_UPSTREAM_IDENTITY_FIELDS,
    "canonical_split_git_blob_sha1",
    "selected_record_count",
    "selected_source_bytes",
    "selected_family_source_bytes",
    "selected_stratum_source_bytes",
    "split_family",
    "claim_boundary",
    "application_identity_sha256",
}
_REPORT_KEYS = {
    "schema",
    "status",
    "decision",
    "balanced_selection_identity_sha256",
    "split_application_identity_sha256",
    *_UPSTREAM_IDENTITY_FIELDS,
    "canonical_split_git_blob_sha1",
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
_ZERO_CREDIT_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}


class TokenizerDecisionError(ValueError):
    """Raised when terminal tokenizer-decision evidence fails closed."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TokenizerDecisionError("authority must be canonical-JSON serializable") from exc


def authority_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a canonical JSON mapping."""

    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _self_hash(value: Mapping[str, Any], identity_field: str) -> str:
    core = dict(value)
    core.pop(identity_field, None)
    return authority_sha256(core)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _verify_selection(
    selection: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(selection, Mapping) or selection.get("schema") != SELECTION_SCHEMA:
        raise TokenizerDecisionError("unsupported balanced-selection authority")
    try:
        _, totals = verify_balanced_selection(
            selection,
            expected_selection_identity_sha256=expected_selection_identity_sha256,
            expected_retained_inventory_identity_sha256=(
                expected_retained_inventory_identity_sha256
            ),
            expected_decontamination_authority_sha256=(
                expected_decontamination_authority_sha256
            ),
            expected_dedup_authority_sha256=expected_dedup_authority_sha256,
            expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
            expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
        )
    except BalancedSplitApplicationError as exc:
        raise TokenizerDecisionError(str(exc)) from exc
    identity = _require_sha256(
        selection.get("balanced_selection_identity_sha256"),
        field="balanced_selection_identity_sha256",
    )
    return identity, totals


def _verify_split_application(
    application: Mapping[str, Any],
    selection: Mapping[str, Any],
    totals: Mapping[str, Any],
    *,
    expected_application_identity_sha256: str,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> str:
    if not isinstance(application, Mapping) or set(application) != _APPLICATION_KEYS:
        raise TokenizerDecisionError("split application fields are not closed-world")
    if application.get("schema") != APPLICATION_SCHEMA:
        raise TokenizerDecisionError("unsupported split-application authority")
    if application.get("status") != "PASS_ZERO_CREDIT":
        raise TokenizerDecisionError("split application is not canonical PASS_ZERO_CREDIT")

    expected_application = _require_sha256(
        expected_application_identity_sha256,
        field="expected_application_identity_sha256",
    )
    claimed_application = _require_sha256(
        application.get("application_identity_sha256"),
        field="application_identity_sha256",
    )
    if _self_hash(application, "application_identity_sha256") != claimed_application:
        raise TokenizerDecisionError("split application self-hash mismatch")
    if claimed_application != expected_application:
        raise TokenizerDecisionError("split application identity mismatch")

    expected_selection = _require_sha256(
        expected_selection_identity_sha256,
        field="expected_selection_identity_sha256",
    )
    if application.get("balanced_selection_identity_sha256") != expected_selection:
        raise TokenizerDecisionError("split application is from a different balanced selection")
    if selection.get("balanced_selection_identity_sha256") != expected_selection:
        raise TokenizerDecisionError("selection identity drift")

    expected_upstreams = {
        "retained_inventory_identity_sha256": expected_retained_inventory_identity_sha256,
        "decontamination_authority_sha256": expected_decontamination_authority_sha256,
        "dedup_authority_sha256": expected_dedup_authority_sha256,
        "balance_policy_identity_sha256": expected_balance_policy_identity_sha256,
        "balance_result_identity_sha256": expected_balance_result_identity_sha256,
    }
    for field, expected in expected_upstreams.items():
        expected_sha = _require_sha256(expected, field=f"expected_{field}")
        if selection.get(field) != expected_sha or application.get(field) != expected_sha:
            raise TokenizerDecisionError(f"{field} lineage mismatch")

    if application.get("canonical_split_git_blob_sha1") != CANONICAL_SPLIT_GIT_BLOB_SHA1:
        raise TokenizerDecisionError("split mechanics identity drift")
    if application.get("claim_boundary") != _ZERO_CREDIT_BOUNDARY:
        raise TokenizerDecisionError("split application truth boundary widened")
    accounting = {
        "selected_record_count": "record_count",
        "selected_source_bytes": "source_bytes",
        "selected_family_source_bytes": "family_source_bytes",
        "selected_stratum_source_bytes": "stratum_source_bytes",
    }
    for application_field, totals_field in accounting.items():
        if application.get(application_field) != totals.get(totals_field):
            raise TokenizerDecisionError(f"split application {application_field} drift")
    return claimed_application


def _bind_upstreams(
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> tuple[str, str]:
    selection_identity, totals = _verify_selection(
        selection,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    application_identity = _verify_split_application(
        application,
        selection,
        totals,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    return selection_identity, application_identity


def bind_byte_baseline_decision(
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> dict[str, Any]:
    """Bind canonical balanced-selection/split lineage to the frozen byte tokenizer."""

    selection_identity, application_identity = _bind_upstreams(
        selection,
        application,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
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
        "balanced_selection_identity_sha256": selection_identity,
        "split_application_identity_sha256": application_identity,
        **{field: selection[field] for field in _UPSTREAM_IDENTITY_FIELDS},
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
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
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> None:
    """Verify decision identity and rebind the canonical upstream lineage."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TokenizerDecisionError("report fields are not closed-world")
    if report.get("schema") != SCHEMA or report.get("status") != STATUS:
        raise TokenizerDecisionError("report schema/status mismatch")
    if report.get("decision") != DECISION:
        raise TokenizerDecisionError("unexpected tokenizer decision")

    selection_identity, application_identity = _bind_upstreams(
        selection,
        application,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    if report.get("balanced_selection_identity_sha256") != selection_identity:
        raise TokenizerDecisionError("report balanced-selection identity mismatch")
    if report.get("split_application_identity_sha256") != application_identity:
        raise TokenizerDecisionError("report split-application identity mismatch")
    for field in _UPSTREAM_IDENTITY_FIELDS:
        if report.get(field) != selection.get(field):
            raise TokenizerDecisionError(f"report {field} drift")
    if report.get("canonical_split_git_blob_sha1") != CANONICAL_SPLIT_GIT_BLOB_SHA1:
        raise TokenizerDecisionError("report split mechanics identity drift")

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
        if report.get(key) != expected:
            raise TokenizerDecisionError(f"report {key} drift")
    if report.get("tokenizer_fit_executed") is not False:
        raise TokenizerDecisionError("byte-baseline decision cannot claim tokenizer fitting")
    if report.get("training_authorized_by_this_report") is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize training")
    if report.get("compute_authorized_by_this_report") is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize compute")
    exposure = report.get("authorized_optimized_target_exposure")
    if isinstance(exposure, bool) or exposure != 0:
        raise TokenizerDecisionError("tokenizer decision cannot authorize exposure")

    supplied_identity = _require_sha256(
        report.get("decision_identity_sha256"), field="decision_identity_sha256"
    )
    if _self_hash(report, "decision_identity_sha256") != supplied_identity:
        raise TokenizerDecisionError("decision report identity mismatch")
