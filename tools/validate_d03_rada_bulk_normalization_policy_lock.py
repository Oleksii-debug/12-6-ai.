#!/usr/bin/env python3
"""Fail closed if canonical D03 Rada bulk normalization V1 semantics drift."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CONFIG_SCHEMA = "12-6.d03-rada-bulk-normalization.v1"
DEFAULT_CONFIG = Path("configs/data/d03_rada_bulk_normalization_v1.json")
SAFE_RESULT = "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED"

EXPECTED_HIDDEN_TAGS = (
    "head",
    "script",
    "style",
    "noscript",
    "template",
    "svg",
)
EXPECTED_BLOCK_TAGS = (
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
)
EXPECTED_OUTPUT_FIELDS = (
    "record_id",
    "source_path",
    "raw_bytes",
    "raw_sha256",
    "normalized_bytes",
    "normalized_sha256",
    "text",
)
EXPECTED_DOWNSTREAM = (
    "QUALITY_FILTER",
    "PRIVACY_PII_FILTER",
    "GLOBAL_CROSS_SOURCE_EXACT_NEAR_DEDUP",
    "EVALUATION_DECONTAMINATION",
    "BALANCE_DIVERSITY_AND_FAMILY_CAP_RETEST",
    "DETERMINISTIC_SPLIT_SHARD_PACK",
    "UNIQUE_CAUSAL_LOSS_LEDGER",
    "TOKENIZER_FIT_AUTHORIZATION",
    "LEARNED_20M_COMPUTE_AUTHORIZATION",
)


class PolicyLockError(RuntimeError):
    """Raised when the canonical V1 normalization policy is weakened or changed."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_exact(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise PolicyLockError(f"{label} drift: expected {expected!r}, got {observed!r}")


def semantic_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact canonical semantics that downstream evidence must bind."""
    normalization = config.get("normalization")
    output = config.get("output_contract")
    boundary = config.get("claim_boundary")
    if not isinstance(normalization, Mapping):
        raise PolicyLockError("normalization policy missing")
    if not isinstance(output, Mapping):
        raise PolicyLockError("output contract missing")
    if not isinstance(boundary, Mapping):
        raise PolicyLockError("claim boundary missing")

    return {
        "schema_version": config.get("schema_version"),
        "normalization": {
            "name": normalization.get("name"),
            "decode": normalization.get("decode"),
            "unicode_normalization": normalization.get("unicode_normalization"),
            "collapse_inline_whitespace": normalization.get("collapse_inline_whitespace"),
            "collapse_blank_lines": normalization.get("collapse_blank_lines"),
            "hidden_tags": normalization.get("hidden_tags"),
            "block_tags": normalization.get("block_tags"),
            "record_id_prefix": normalization.get("record_id_prefix"),
        },
        "output_contract": {
            "jsonl_record_fields": output.get("jsonl_record_fields"),
            "manifest_is_text_free": output.get("manifest_is_text_free"),
            "record_order": output.get("record_order"),
            "exact_probe_inventory_required": output.get("exact_probe_inventory_required"),
            "duplicate_record_ids_rejected": output.get("duplicate_record_ids_rejected"),
            "deterministic_json_serialization": output.get("deterministic_json_serialization"),
        },
        "downstream_required": config.get("downstream_required"),
        "claim_boundary": {
            "bulk_source_admitted": boundary.get("bulk_source_admitted"),
            "normalized_capacity_credited": boundary.get("normalized_capacity_credited"),
            "training_authorized_bytes": boundary.get("training_authorized_bytes"),
            "tokenizer_fit_authorized": boundary.get("tokenizer_fit_authorized"),
            "model_training_executed": boundary.get("model_training_executed"),
            "paid_compute_used": boundary.get("paid_compute_used"),
            "research_corpus_v1_released": boundary.get("research_corpus_v1_released"),
            "learned_20m_claimed": boundary.get("learned_20m_claimed"),
            "safe_result": boundary.get("safe_result"),
        },
    }


def validate_policy_lock(config: Mapping[str, Any]) -> str:
    """Validate exact V1 semantics and return a stable semantic SHA-256 identity."""
    projection = semantic_projection(config)
    _require_exact("schema", projection["schema_version"], CONFIG_SCHEMA)

    normalization = projection["normalization"]
    _require_exact("normalization name", normalization["name"], "RADA_VISIBLE_TEXT_HTML_NFKC_V1")
    _require_exact("decode", normalization["decode"], "STRICT_UTF8_OPTIONAL_BOM")
    _require_exact("unicode normalization", normalization["unicode_normalization"], "NFKC")
    _require_exact("inline whitespace", normalization["collapse_inline_whitespace"], True)
    _require_exact("blank-line collapse", normalization["collapse_blank_lines"], True)
    _require_exact("hidden tags", tuple(normalization["hidden_tags"] or ()), EXPECTED_HIDDEN_TAGS)
    _require_exact("block tags", tuple(normalization["block_tags"] or ()), EXPECTED_BLOCK_TAGS)
    _require_exact(
        "record id prefix",
        normalization["record_id_prefix"],
        "ua.rada.open-data.laws-texts.",
    )

    output = projection["output_contract"]
    _require_exact("record fields", tuple(output["jsonl_record_fields"] or ()), EXPECTED_OUTPUT_FIELDS)
    _require_exact("text-free manifest", output["manifest_is_text_free"], True)
    _require_exact("record order", output["record_order"], "LEXICOGRAPHIC_CANONICAL_BASENAME")
    _require_exact("probe inventory requirement", output["exact_probe_inventory_required"], True)
    _require_exact("duplicate id rejection", output["duplicate_record_ids_rejected"], True)
    _require_exact("deterministic JSON", output["deterministic_json_serialization"], True)

    _require_exact(
        "downstream gate order",
        tuple(projection["downstream_required"] or ()),
        EXPECTED_DOWNSTREAM,
    )

    boundary = projection["claim_boundary"]
    for key in (
        "bulk_source_admitted",
        "normalized_capacity_credited",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
        "research_corpus_v1_released",
        "learned_20m_claimed",
    ):
        _require_exact(f"claim boundary {key}", boundary[key], False)
    _require_exact("training authorization", boundary["training_authorized_bytes"], 0)
    _require_exact("safe result", boundary["safe_result"], SAFE_RESULT)

    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyLockError(f"cannot load normalization config: {path}") from exc
    if not isinstance(value, dict):
        raise PolicyLockError("normalization config root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    identity = validate_policy_lock(_load(args.config))
    print(
        json.dumps(
            {
                "status": "PASS_CANONICAL_NORMALIZATION_POLICY_LOCK",
                "semantic_identity_sha256": identity,
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
