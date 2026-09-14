"""Authenticated ArXiv + LangUK candidate rematerialization helpers.

This module owns no source-selection, privacy, rights, matcher, or dedup science.  It
pins the exact historical materializer programs that already produced the terminal
source-admitted candidates, verifies reconstructed candidate bytes, and builds a
text-free two-pass replay receipt.  The actual source acquisition/materialization and
global-dedup execution remain delegated to those historical programs and to the exact
audited PR #1800 runner respectively.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final

PARENT_PR1800_HEAD: Final = "6187887c01bdbc1857ca9f63145374d2a2ccb20b"
PARENT_INTAKE_BLOB_SHA1: Final = "322f1441326ca17447581be8ebf2d98c2385bd38"
PARENT_RUNNER_BLOB_SHA1: Final = "6f7e68e4d7f36d0ae0bf792e1d07db767374a3cf"
RECEIPT_SCHEMA: Final = "12-6.d03-arxiv-languk-rematerialized-v9-replay.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class RematerializationError(RuntimeError):
    """Raised when historical-program or reconstructed-candidate identity drifts."""


@dataclass(frozen=True)
class HistoricalMaterializerSpec:
    key: str
    execution_commit: str
    tool_path: str
    tool_blob_sha1: str
    config_path: str
    config_blob_sha1: str
    source_arg: str
    source_filename: str
    candidate_sha256: str
    retained_records: int
    retained_normalized_bytes: int
    row_keys: frozenset[str]
    source_key: str | None = None
    source_label: str | None = None


ARXIV: Final = HistoricalMaterializerSpec(
    key="arxiv",
    execution_commit="37289b5af224c58f0b169f55f4e0f8b466b90f6d",
    tool_path="tools/materialize_d03_common_pile_arxiv_abstracts_v1.py",
    tool_blob_sha1="38c244a5a2da40ed4fb67f801f02e625bc040564",
    config_path="configs/data/d03_common_pile_arxiv_abstracts_bounded_v1.json",
    config_blob_sha1="4a819b3980634c8a2ba7de55cf854d44bcfd5757",
    source_arg="--shard",
    source_filename="00003_arxiv-abstracts.jsonl.gz",
    candidate_sha256="21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327",
    retained_records=1024,
    retained_normalized_bytes=1_139_552,
    row_keys=frozenset(
        {
            "record_id",
            "source_key",
            "source_label",
            "normalized_sha256",
            "normalized_bytes",
            "text",
        }
    ),
    source_key="arxiv_abstracts",
    source_label="arxiv-abstracts",
)

LANGUK: Final = HistoricalMaterializerSpec(
    key="languk",
    execution_commit="5d38a32b37490b11f6fa77b3bcd21855f3e28605",
    tool_path="tools/retest_d03_languk_supreme_court.py",
    tool_blob_sha1="5401881e170f2d1ae70e777a0cf3e2cfa152e8f6",
    config_path="configs/data/d03_languk_supreme_court_retest_v1.json",
    config_blob_sha1="b4ee7b0f698660145cb3c1db0f8ccb690c255a5e",
    source_arg="--parquet",
    source_filename="2024-5K-supreme-court-decisions-deduplicated.parquet",
    candidate_sha256="03bf5089bb6ff4c304e3a299e2480b263a161aad8abe831db0b196af7c585db3",
    retained_records=256,
    retained_normalized_bytes=2_809_632,
    row_keys=frozenset({"record_id", "normalized_sha256", "normalized_bytes", "text"}),
)

SPECS: Final = (ARXIV, LANGUK)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RematerializationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def verify_git_blob(raw: bytes, expected_sha1: str, *, label: str) -> None:
    _require(
        isinstance(expected_sha1, str) and _SHA1_RE.fullmatch(expected_sha1) is not None,
        f"{label} expected Git blob SHA-1 malformed",
    )
    _require(git_blob_sha1(raw) == expected_sha1, f"{label} Git blob drift")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RematerializationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_candidate_rows(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    _require(type(raw) is bytes and bool(raw), f"{label} candidate must be non-empty bytes")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        _require(bool(line), f"{label} candidate contains blank row {line_no}")
        try:
            value = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RematerializationError(
                f"{label} candidate row {line_no} is invalid UTF-8 JSON"
            ) from exc
        _require(type(value) is dict, f"{label} candidate row {line_no} must be object")
        rows.append(value)
    return rows


def verify_candidate(spec: HistoricalMaterializerSpec, raw: bytes) -> dict[str, Any]:
    """Verify exact reconstructed candidate bytes and every retained text identity."""
    _require(
        sha256_bytes(raw) == spec.candidate_sha256,
        f"{spec.key} candidate SHA-256 drift",
    )
    rows = _load_candidate_rows(raw, label=spec.key)
    _require(
        len(rows) == spec.retained_records,
        f"{spec.key} retained record count drift",
    )

    seen_ids: set[str] = set()
    normalized_bytes = 0
    for row_no, row in enumerate(rows, 1):
        _require(
            set(row) == set(spec.row_keys),
            f"{spec.key} row {row_no} keyset drift",
        )
        record_id = row.get("record_id")
        _require(
            isinstance(record_id, str) and bool(record_id),
            f"{spec.key} row {row_no} record id invalid",
        )
        _require(record_id not in seen_ids, f"{spec.key} duplicate record id")
        seen_ids.add(record_id)

        if spec.key == "arxiv":
            _require(
                row.get("source_key") == spec.source_key
                and type(row.get("source_key")) is str,
                "arxiv source key drift",
            )
            _require(
                row.get("source_label") == spec.source_label
                and type(row.get("source_label")) is str,
                "arxiv source label drift",
            )
        elif spec.key == "languk":
            _require(record_id.isdigit(), "languk record id must remain decimal")
        else:
            raise RematerializationError(f"unknown materializer spec: {spec.key}")

        text = row.get("text")
        _require(
            isinstance(text, str) and bool(text),
            f"{spec.key} row {row_no} text invalid",
        )
        text_raw = text.encode("utf-8")
        row_sha = row.get("normalized_sha256")
        _require(
            isinstance(row_sha, str) and _SHA256_RE.fullmatch(row_sha) is not None,
            f"{spec.key} row {row_no} normalized SHA malformed",
        )
        _require(
            sha256_bytes(text_raw) == row_sha,
            f"{spec.key} row {row_no} normalized SHA drift",
        )
        size = row.get("normalized_bytes")
        _require(
            type(size) is int and size == len(text_raw),
            f"{spec.key} row {row_no} normalized byte count drift",
        )
        normalized_bytes += size

    _require(
        normalized_bytes == spec.retained_normalized_bytes,
        f"{spec.key} retained normalized byte total drift",
    )
    return {
        "candidate_sha256": spec.candidate_sha256,
        "retained_records": spec.retained_records,
        "retained_normalized_bytes": spec.retained_normalized_bytes,
    }


def build_receipt(
    *,
    pass_results: list[dict[str, Any]],
    incumbent_runner_blob_sha1: str,
    incumbent_intake_blob_sha1: str,
) -> dict[str, Any]:
    """Build a text-free receipt only after two byte-identical physical replay passes."""
    _require(len(pass_results) == 2, "exactly two clean replay passes are required")
    _require(
        incumbent_runner_blob_sha1 == PARENT_RUNNER_BLOB_SHA1,
        "incumbent runner blob drift",
    )
    _require(
        incumbent_intake_blob_sha1 == PARENT_INTAKE_BLOB_SHA1,
        "incumbent intake blob drift",
    )

    required_pass_keys = {
        "arxiv_candidate_sha256",
        "languk_candidate_sha256",
        "report_file_sha256",
        "survivor_file_sha256",
        "report_identity_sha256",
        "survivor_authority_sha256",
    }
    for index, result in enumerate(pass_results, 1):
        _require(type(result) is dict, f"pass {index} result must be object")
        _require(
            set(result) == required_pass_keys,
            f"pass {index} result keyset drift",
        )
        for key, value in result.items():
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"pass {index} {key} malformed",
            )
        _require(
            result["arxiv_candidate_sha256"] == ARXIV.candidate_sha256,
            f"pass {index} ArXiv candidate drift",
        )
        _require(
            result["languk_candidate_sha256"] == LANGUK.candidate_sha256,
            f"pass {index} LangUK candidate drift",
        )

    left, right = pass_results
    for key in (
        "arxiv_candidate_sha256",
        "languk_candidate_sha256",
        "report_file_sha256",
        "survivor_file_sha256",
        "report_identity_sha256",
        "survivor_authority_sha256",
    ):
        _require(left[key] == right[key], f"two-pass replay mismatch: {key}")

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PHYSICAL_REMATERIALIZATION_AND_V9_REPLAY_EXECUTED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "parent_authority": {
            "product_pr": 1800,
            "exact_head_sha": PARENT_PR1800_HEAD,
            "intake_blob_sha1": PARENT_INTAKE_BLOB_SHA1,
            "runner_blob_sha1": PARENT_RUNNER_BLOB_SHA1,
        },
        "historical_materializers": {
            spec.key: {
                "execution_commit": spec.execution_commit,
                "tool_path": spec.tool_path,
                "tool_blob_sha1": spec.tool_blob_sha1,
                "config_path": spec.config_path,
                "config_blob_sha1": spec.config_blob_sha1,
                "candidate_sha256": spec.candidate_sha256,
                "retained_records": spec.retained_records,
                "retained_normalized_bytes": spec.retained_normalized_bytes,
            }
            for spec in SPECS
        },
        "two_clean_passes": pass_results,
        "reproducibility": {
            "candidate_identities_equal": True,
            "report_files_byte_identical": True,
            "survivor_files_byte_identical": True,
            "report_identities_equal": True,
            "survivor_authority_identities_equal": True,
            "raw_payloads_retained_after_success": False,
        },
        "truth_boundary": {
            "global_dedup_replay_executed": True,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
