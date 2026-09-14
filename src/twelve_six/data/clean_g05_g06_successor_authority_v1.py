"""Clean-successor authority binder for the canonical D03 G05/G06 path.

This module does not execute quality/privacy science and does not grant training
credit. It binds an independently expected clean-input authority to the incumbent
G05/G06 coverage authority and requires two independently rooted LOCAL_FREE replay
receipts to agree before emitting a text-free zero-credit successor authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.final_g05_g06_coverage_v1 import (
    QUALITY_GRANULARITY_IDENTITY_SHA256,
    QUALITY_POLICY_IDENTITY_SHA256,
    verify_final_g05_g06_coverage,
)

SCHEMA = "12-6.d03-clean-g05-g06-successor-authority.v1"
CLEAN_INPUT_BINDING_SCHEMA = "12-6.d03-clean-g05-g06-input-binding.v1"
REPLAY_RECEIPT_SCHEMA = "12-6.d03-clean-g05-g06-replay-receipt.v1"
CLEAN_INPUT_STATUS = "CLEAN_SUCCESSOR_BOUND"
REPLAY_STATUS = "PASS_LOCAL_FREE_REPLAY"
AUTHORITY_STATUS = "PASS_REPLAY_BOUND_ZERO_CREDIT"
REPLAY_EXECUTION_POLICY = "LOCAL_FREE"
PROVENANCE_SCOPE = "WHOLE_CORPUS_EXTERNAL_LLM_CLEANLINESS_NOT_CLAIMED"

NOMIS_RECORD_ID = "ua.verba.nomis1864.bounded24"
NOMIS_FAMILY = "ua.verba.public-domain.nomis1864"
NOMIS_PAYLOAD_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
PR462_AUTHORITY_SHA256 = (
    "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")

_CLEAN_INPUT_KEYS = frozenset(
    {
        "schema",
        "status",
        "source_rebuild_authority_sha256",
        "retained_inventory_identity_sha256",
        "records_jsonl_sha256",
        "decontamination_authority_sha256",
        "nomis_record_id",
        "nomis_family",
        "nomis_payload_sha256",
        "pr462_authority_sha256",
        "nomis1864_admitted",
        "pr462_authority_admitted",
        "nomis1864_quarantine_enforced",
        "whole_corpus_external_llm_cleanliness_claimed",
        "current_retained_corpus_launch_authoritative",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "optimizer_updates_executed_on_real_targets",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "clean_input_binding_identity_sha256",
    }
)

_REPLAY_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_id",
        "execution_policy",
        "clean_input_binding_identity_sha256",
        "g05_g06_coverage_identity_sha256",
        "retained_inventory_identity_sha256",
        "records_jsonl_sha256",
        "decontamination_authority_sha256",
        "quality_policy_identity_sha256",
        "quality_granularity_identity_sha256",
        "privacy_policy_identity_sha256",
        "privacy_implementation_git_blob_sha",
        "qualification_authority_set_identity_sha256",
        "covered_record_count",
        "covered_payload_bytes",
        "whole_corpus_external_llm_cleanliness_claimed",
        "current_retained_corpus_launch_authoritative",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "optimizer_updates_executed_on_real_targets",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "replay_receipt_identity_sha256",
    }
)

_AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "status",
        "provenance_scope",
        "clean_input_binding_identity_sha256",
        "source_rebuild_authority_sha256",
        "g05_g06_coverage_identity_sha256",
        "replay_receipt_identities_sha256",
        "deterministic_replay_projection_sha256",
        "retained_inventory_identity_sha256",
        "records_jsonl_sha256",
        "decontamination_authority_sha256",
        "quality_policy_identity_sha256",
        "quality_granularity_identity_sha256",
        "privacy_policy_identity_sha256",
        "privacy_implementation_git_blob_sha",
        "qualification_authority_set_identity_sha256",
        "covered_record_count",
        "covered_payload_bytes",
        "whole_corpus_external_llm_cleanliness_claimed",
        "current_retained_corpus_launch_authoritative",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "optimizer_updates_executed_on_real_targets",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "successor_authority_identity_sha256",
    }
)

_ZERO_FALSE_FIELDS = (
    "whole_corpus_external_llm_cleanliness_claimed",
    "current_retained_corpus_launch_authoritative",
    "tokenizer_fit_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
)
_ZERO_INT_FIELDS = (
    "authorized_optimized_target_exposure",
    "optimizer_updates_executed_on_real_targets",
)


class CleanG05G06SuccessorError(ValueError):
    """Raised when clean-successor authority cannot be proved fail-closed."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanG05G06SuccessorError(message)


def _cjson(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return _sha(_cjson(body))


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CleanG05G06SuccessorError(f"{field} must be lowercase SHA-256")
    return value


def _git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise CleanG05G06SuccessorError(
            f"{field} must be a lowercase 40-hex Git object id"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanG05G06SuccessorError(f"{field} must be a positive integer")
    return value


def _verify_zero_boundary(document: Mapping[str, Any], *, prefix: str) -> None:
    for field in _ZERO_FALSE_FIELDS:
        _require(document.get(field) is False, f"{prefix}.{field} must be false")
    for field in _ZERO_INT_FIELDS:
        value = document.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value == 0,
            f"{prefix}.{field} must be zero",
        )


def _verify_clean_input(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_source_rebuild_authority_sha256: str,
) -> dict[str, str]:
    _require(set(document) == _CLEAN_INPUT_KEYS, "clean input root schema drift")
    _require(
        document.get("schema") == CLEAN_INPUT_BINDING_SCHEMA,
        "clean input schema drift",
    )
    _require(document.get("status") == CLEAN_INPUT_STATUS, "clean input is not bound")

    expected_identity = _sha256(
        expected_identity_sha256,
        "expected clean input binding identity",
    )
    claimed_identity = _sha256(
        document.get("clean_input_binding_identity_sha256"),
        "clean input binding identity",
    )
    _require(
        claimed_identity == expected_identity,
        "clean input binding is not independently expected",
    )
    _require(
        claimed_identity == _self_hash(document, "clean_input_binding_identity_sha256"),
        "clean input binding self-hash mismatch",
    )

    source_root = _sha256(
        expected_source_rebuild_authority_sha256,
        "expected source rebuild authority",
    )
    _require(
        document.get("source_rebuild_authority_sha256") == source_root,
        "source rebuild authority drift",
    )

    expected_quarantine = {
        "nomis_record_id": NOMIS_RECORD_ID,
        "nomis_family": NOMIS_FAMILY,
        "nomis_payload_sha256": NOMIS_PAYLOAD_SHA256,
        "pr462_authority_sha256": PR462_AUTHORITY_SHA256,
        "nomis1864_admitted": False,
        "pr462_authority_admitted": False,
        "nomis1864_quarantine_enforced": True,
        "whole_corpus_external_llm_cleanliness_claimed": False,
    }
    for key, value in expected_quarantine.items():
        _require(document.get(key) == value, f"clean input quarantine drift: {key}")

    _verify_zero_boundary(document, prefix="clean_input")
    return {
        "identity": claimed_identity,
        "source_root": source_root,
        "inventory": _sha256(
            document.get("retained_inventory_identity_sha256"),
            "retained inventory identity",
        ),
        "records": _sha256(document.get("records_jsonl_sha256"), "records JSONL"),
        "decontam": _sha256(
            document.get("decontamination_authority_sha256"),
            "decontamination authority",
        ),
    }


def _coverage_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clean_input_binding_identity_sha256": None,
        "g05_g06_coverage_identity_sha256": _sha256(
            document.get("g05_g06_coverage_identity_sha256"),
            "G05/G06 coverage identity",
        ),
        "retained_inventory_identity_sha256": _sha256(
            document.get("retained_inventory_identity_sha256"),
            "coverage retained inventory identity",
        ),
        "records_jsonl_sha256": _sha256(
            document.get("records_jsonl_sha256"),
            "coverage records JSONL",
        ),
        "decontamination_authority_sha256": _sha256(
            document.get("decontamination_authority_sha256"),
            "coverage decontamination authority",
        ),
        "quality_policy_identity_sha256": _sha256(
            document.get("quality_policy_identity_sha256"),
            "quality policy identity",
        ),
        "quality_granularity_identity_sha256": _sha256(
            document.get("quality_granularity_identity_sha256"),
            "quality granularity identity",
        ),
        "privacy_policy_identity_sha256": _sha256(
            document.get("privacy_policy_identity_sha256"),
            "privacy policy identity",
        ),
        "privacy_implementation_git_blob_sha": _git_sha(
            document.get("privacy_implementation_git_blob_sha"),
            "privacy implementation",
        ),
        "qualification_authority_set_identity_sha256": _sha256(
            document.get("qualification_authority_set_identity_sha256"),
            "qualification authority set identity",
        ),
        "covered_record_count": _positive_int(
            document.get("covered_record_count"),
            "covered record count",
        ),
        "covered_payload_bytes": _positive_int(
            document.get("covered_payload_bytes"),
            "covered payload bytes",
        ),
    }


def _verify_replay(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    clean_input_identity: str,
    coverage_projection: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    _require(set(document) == _REPLAY_KEYS, "replay receipt root schema drift")
    _require(document.get("schema") == REPLAY_RECEIPT_SCHEMA, "replay schema drift")
    _require(document.get("status") == REPLAY_STATUS, "replay is not terminal PASS")
    _require(
        document.get("execution_policy") == REPLAY_EXECUTION_POLICY,
        "replay is not LOCAL_FREE",
    )

    run_id = document.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id), "replay run_id missing")
    expected_identity = _sha256(
        expected_identity_sha256,
        "expected replay receipt identity",
    )
    claimed_identity = _sha256(
        document.get("replay_receipt_identity_sha256"),
        "replay receipt identity",
    )
    _require(
        claimed_identity == expected_identity,
        "replay receipt is not independently expected",
    )
    _require(
        claimed_identity == _self_hash(document, "replay_receipt_identity_sha256"),
        "replay receipt self-hash mismatch",
    )
    _require(
        document.get("clean_input_binding_identity_sha256") == clean_input_identity,
        "replay clean-input binding drift",
    )
    _verify_zero_boundary(document, prefix="replay")

    expected_fields = dict(coverage_projection)
    expected_fields["clean_input_binding_identity_sha256"] = clean_input_identity
    for key, value in expected_fields.items():
        _require(document.get(key) == value, f"replay projection drift: {key}")

    semantic_projection = {
        key: document[key]
        for key in sorted(_REPLAY_KEYS - {"run_id", "replay_receipt_identity_sha256"})
    }
    return run_id, claimed_identity, semantic_projection


def build_clean_g05_g06_successor_authority(
    *,
    clean_input_binding: Mapping[str, Any],
    expected_clean_input_binding_identity_sha256: str,
    expected_source_rebuild_authority_sha256: str,
    g05_g06_coverage: Mapping[str, Any],
    expected_g05_g06_coverage_identity_sha256: str,
    expected_privacy_policy_identity_sha256: str,
    expected_privacy_implementation_git_blob_sha: str,
    replay_receipts: Sequence[Mapping[str, Any]],
    expected_replay_receipt_identities_sha256: Sequence[str],
) -> dict[str, Any]:
    """Bind clean input + incumbent coverage + two exact LOCAL_FREE replays."""
    clean = _verify_clean_input(
        clean_input_binding,
        expected_identity_sha256=expected_clean_input_binding_identity_sha256,
        expected_source_rebuild_authority_sha256=(
            expected_source_rebuild_authority_sha256
        ),
    )
    coverage_identity = _sha256(
        expected_g05_g06_coverage_identity_sha256,
        "expected G05/G06 coverage identity",
    )
    verify_final_g05_g06_coverage(
        g05_g06_coverage,
        expected_identity_sha256=coverage_identity,
        expected_retained_inventory_identity_sha256=clean["inventory"],
        expected_decontamination_authority_sha256=clean["decontam"],
        expected_records_jsonl_sha256=clean["records"],
        expected_privacy_policy_identity_sha256=(
            expected_privacy_policy_identity_sha256
        ),
        expected_privacy_implementation_git_blob_sha=(
            expected_privacy_implementation_git_blob_sha
        ),
    )
    projection = _coverage_projection(g05_g06_coverage)
    _require(
        projection["g05_g06_coverage_identity_sha256"] == coverage_identity,
        "coverage identity projection drift",
    )
    _require(
        projection["quality_policy_identity_sha256"]
        == QUALITY_POLICY_IDENTITY_SHA256,
        "noncanonical quality policy in successor",
    )
    _require(
        projection["quality_granularity_identity_sha256"]
        == QUALITY_GRANULARITY_IDENTITY_SHA256,
        "noncanonical quality granularity in successor",
    )

    _require(
        isinstance(replay_receipts, Sequence)
        and not isinstance(replay_receipts, (str, bytes))
        and len(replay_receipts) == 2,
        "exactly two replay receipts are required",
    )
    expected_replays = list(expected_replay_receipt_identities_sha256)
    _require(len(expected_replays) == 2, "exactly two expected replay roots are required")
    _require(
        len(set(expected_replays)) == 2,
        "expected replay roots must be distinct",
    )

    run_ids: list[str] = []
    replay_ids: list[str] = []
    semantic_projections: list[dict[str, Any]] = []
    for receipt, expected_root in zip(replay_receipts, expected_replays, strict=True):
        run_id, receipt_identity, semantic_projection = _verify_replay(
            receipt,
            expected_identity_sha256=expected_root,
            clean_input_identity=clean["identity"],
            coverage_projection=projection,
        )
        run_ids.append(run_id)
        replay_ids.append(receipt_identity)
        semantic_projections.append(semantic_projection)

    _require(len(set(run_ids)) == 2, "replay run_ids must be distinct")
    _require(
        semantic_projections[0] == semantic_projections[1],
        "LOCAL_FREE replay semantic projections differ",
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": AUTHORITY_STATUS,
        "provenance_scope": PROVENANCE_SCOPE,
        "clean_input_binding_identity_sha256": clean["identity"],
        "source_rebuild_authority_sha256": clean["source_root"],
        "g05_g06_coverage_identity_sha256": coverage_identity,
        "replay_receipt_identities_sha256": sorted(replay_ids),
        "deterministic_replay_projection_sha256": _sha(
            _cjson(semantic_projections[0])
        ),
        "retained_inventory_identity_sha256": projection[
            "retained_inventory_identity_sha256"
        ],
        "records_jsonl_sha256": projection["records_jsonl_sha256"],
        "decontamination_authority_sha256": projection[
            "decontamination_authority_sha256"
        ],
        "quality_policy_identity_sha256": projection["quality_policy_identity_sha256"],
        "quality_granularity_identity_sha256": projection[
            "quality_granularity_identity_sha256"
        ],
        "privacy_policy_identity_sha256": projection["privacy_policy_identity_sha256"],
        "privacy_implementation_git_blob_sha": projection[
            "privacy_implementation_git_blob_sha"
        ],
        "qualification_authority_set_identity_sha256": projection[
            "qualification_authority_set_identity_sha256"
        ],
        "covered_record_count": projection["covered_record_count"],
        "covered_payload_bytes": projection["covered_payload_bytes"],
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "current_retained_corpus_launch_authoritative": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    result["successor_authority_identity_sha256"] = _self_hash(
        result,
        "successor_authority_identity_sha256",
    )
    return result


def verify_clean_g05_g06_successor_authority(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_clean_input_binding_identity_sha256: str,
    expected_source_rebuild_authority_sha256: str,
    expected_g05_g06_coverage_identity_sha256: str,
) -> None:
    """Verify the durable zero-credit successor authority envelope."""
    _require(set(document) == _AUTHORITY_KEYS, "successor authority root schema drift")
    _require(document.get("schema") == SCHEMA, "successor authority schema drift")
    _require(document.get("status") == AUTHORITY_STATUS, "successor authority status drift")
    _require(
        document.get("provenance_scope") == PROVENANCE_SCOPE,
        "successor provenance scope drift",
    )

    expected_identity = _sha256(
        expected_identity_sha256,
        "expected successor authority identity",
    )
    claimed_identity = _sha256(
        document.get("successor_authority_identity_sha256"),
        "successor authority identity",
    )
    _require(
        claimed_identity == expected_identity,
        "successor authority is not independently expected",
    )
    _require(
        claimed_identity
        == _self_hash(document, "successor_authority_identity_sha256"),
        "successor authority self-hash mismatch",
    )
    expected_pairs = {
        "clean_input_binding_identity_sha256": _sha256(
            expected_clean_input_binding_identity_sha256,
            "expected clean input binding identity",
        ),
        "source_rebuild_authority_sha256": _sha256(
            expected_source_rebuild_authority_sha256,
            "expected source rebuild authority",
        ),
        "g05_g06_coverage_identity_sha256": _sha256(
            expected_g05_g06_coverage_identity_sha256,
            "expected G05/G06 coverage identity",
        ),
        "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
        "quality_granularity_identity_sha256": (
            QUALITY_GRANULARITY_IDENTITY_SHA256
        ),
    }
    for key, value in expected_pairs.items():
        _require(document.get(key) == value, f"successor authority drift: {key}")

    replay_ids = document.get("replay_receipt_identities_sha256")
    _require(
        isinstance(replay_ids, list) and len(replay_ids) == 2,
        "successor must bind exactly two replay roots",
    )
    normalized_replay_ids = [
        _sha256(value, f"replay receipt root[{index}]")
        for index, value in enumerate(replay_ids)
    ]
    _require(
        normalized_replay_ids == sorted(normalized_replay_ids),
        "successor replay roots are not sorted",
    )
    _require(
        len(set(normalized_replay_ids)) == 2,
        "successor replay roots are not distinct",
    )
    _sha256(
        document.get("deterministic_replay_projection_sha256"),
        "deterministic replay projection",
    )
    for field in (
        "retained_inventory_identity_sha256",
        "records_jsonl_sha256",
        "decontamination_authority_sha256",
        "privacy_policy_identity_sha256",
        "qualification_authority_set_identity_sha256",
    ):
        _sha256(document.get(field), field)
    _git_sha(
        document.get("privacy_implementation_git_blob_sha"),
        "privacy implementation",
    )
    _positive_int(document.get("covered_record_count"), "covered record count")
    _positive_int(document.get("covered_payload_bytes"), "covered payload bytes")
    _verify_zero_boundary(document, prefix="successor")
