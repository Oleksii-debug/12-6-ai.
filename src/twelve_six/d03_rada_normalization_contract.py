"""Exact production contract for the Rada bulk normalization successor.

This module intentionally contains no corpus-admission or training authority.  It
only prevents a mutable normalization config from silently changing the exact
#618 parent or the v1 visible-text policy while retaining the same schema name.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = "12-6.d03-rada-bulk-normalization.v1"
WORKER_ID = "D03-RADA-BULK-NORMALIZATION-20260826"
PROBE_REPORT_SCHEMA = "12-6.d03-rada-bulk-source-probe-report.v1"

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "worker_id",
    "local_free_only",
    "parent_probe",
    "normalization",
    "output_contract",
    "downstream_required",
    "claim_boundary",
}

EXPECTED_PARENT_PROBE: dict[str, Any] = {
    "pr": 618,
    "head_sha": "bed8c3237379194b90e54d558697a0ddc7ea4f95",
    "probe_report_schema": PROBE_REPORT_SCHEMA,
    "probe_worker_id": "D03-RADA-BULK-SOURCE-PROBE-20260826",
    "probe_config_identity_sha256": (
        "c2f198120cae00ba247c4eaad36d2a357770a47c7fa9a7608cc5ec182971b82b"
    ),
    "source_family": "ua.rada.open-data.laws-texts",
    "source_family_identity_sha256": (
        "b8f1d2f99a3db71d894a3233e9417d6283d11768c41b1634bc8b096ab77aba4e"
    ),
}

EXPECTED_NORMALIZATION: dict[str, Any] = {
    "name": "RADA_VISIBLE_TEXT_HTML_UTF8_CP1251_NFKC_V1",
    "decode": "STRICT_UTF8_THEN_WINDOWS_1251_FALLBACK",
    "legacy_fallback_encoding": "windows-1251",
    "unicode_normalization": "NFKC",
    "collapse_inline_whitespace": True,
    "collapse_blank_lines": True,
    "hidden_tags": [
        "head",
        "script",
        "style",
        "noscript",
        "template",
        "svg",
    ],
    "block_tags": [
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ],
    "record_id_prefix": "ua.rada.open-data.laws-texts.",
}

EXPECTED_OUTPUT_CONTRACT: dict[str, Any] = {
    "jsonl_record_fields": [
        "record_id",
        "source_path",
        "source_encoding",
        "raw_bytes",
        "raw_sha256",
        "normalized_bytes",
        "normalized_sha256",
        "text",
    ],
    "manifest_is_text_free": True,
    "record_order": "LEXICOGRAPHIC_CANONICAL_BASENAME",
    "exact_probe_inventory_required": True,
    "duplicate_record_ids_rejected": True,
    "deterministic_json_serialization": True,
}

EXPECTED_DOWNSTREAM_REQUIRED = [
    "QUALITY_FILTER",
    "PRIVACY_PII_FILTER",
    "GLOBAL_CROSS_SOURCE_EXACT_NEAR_DEDUP",
    "EVALUATION_DECONTAMINATION",
    "BALANCE_DIVERSITY_AND_FAMILY_CAP_RETEST",
    "DETERMINISTIC_SPLIT_SHARD_PACK",
    "UNIQUE_CAUSAL_LOSS_LEDGER",
    "TOKENIZER_FIT_AUTHORIZATION",
    "LEARNED_20M_COMPUTE_AUTHORIZATION",
]

EXPECTED_CLAIM_BOUNDARY: dict[str, Any] = {
    "bulk_source_admitted": False,
    "normalized_capacity_credited": False,
    "training_authorized_bytes": 0,
    "tokenizer_fit_authorized": False,
    "model_training_executed": False,
    "paid_compute_used": False,
    "research_corpus_v1_released": False,
    "learned_20m_claimed": False,
    "safe_result": "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED",
}


class NormalizationContractError(RuntimeError):
    """Raised when the production v1 normalization authority drifts."""


def _require_exact_mapping(
    config: Mapping[str, Any], field: str, expected: Mapping[str, Any]
) -> None:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise NormalizationContractError(f"{field} must be an object")
    if dict(value) != dict(expected):
        raise NormalizationContractError(f"{field} drifted from the pinned v1 contract")


def validate_production_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless *all* production v1 authority surfaces are exact."""
    if set(config) != EXPECTED_TOP_LEVEL_KEYS:
        raise NormalizationContractError("top-level normalization contract keys drifted")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise NormalizationContractError("normalization schema drifted")
    if config.get("worker_id") != WORKER_ID:
        raise NormalizationContractError("normalization worker identity drifted")
    if config.get("local_free_only") is not True:
        raise NormalizationContractError("normalization must remain LOCAL_FREE")

    _require_exact_mapping(config, "parent_probe", EXPECTED_PARENT_PROBE)
    _require_exact_mapping(config, "normalization", EXPECTED_NORMALIZATION)
    _require_exact_mapping(config, "output_contract", EXPECTED_OUTPUT_CONTRACT)
    if config.get("downstream_required") != EXPECTED_DOWNSTREAM_REQUIRED:
        raise NormalizationContractError("downstream_required drifted from pinned v1 contract")
    _require_exact_mapping(config, "claim_boundary", EXPECTED_CLAIM_BOUNDARY)


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the canonical config identity after exact validation."""
    validate_production_config(config)
    payload = json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_manifest_to_contract(
    manifest: Mapping[str, Any], *, normalization_contract_sha256: str
) -> dict[str, Any]:
    """Bind a normalization manifest to the exact validated production contract.

    The incoming materializer manifest self-identifies before this field exists,
    so the identity is recomputed over the strengthened manifest.
    """
    if len(normalization_contract_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in normalization_contract_sha256
    ):
        raise NormalizationContractError("normalization contract identity must be SHA-256")

    strengthened = dict(manifest)
    strengthened.pop("manifest_identity_sha256", None)
    strengthened["normalization_contract_sha256"] = normalization_contract_sha256
    canonical = json.dumps(
        strengthened,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    strengthened["manifest_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return strengthened
