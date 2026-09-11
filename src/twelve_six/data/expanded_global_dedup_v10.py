"""Bind real PEP+LoC candidate materializations into incumbent D03 global dedup.

This is an additive post-V9 composition layer.  It does not implement matching.
The exact real LOCAL_FREE materialization bytes are authenticated, parsed with
duplicate-key rejection, cross-bound to their zero-credit reports, converted to
the incumbent #824 matcher inventory shape, and compared together with the exact
DATA-526+Rada V9 graph.

All outputs remain zero-credit.  Decontamination, canonical quality/privacy,
balance, split, packing, tokenizer fit and training authorization are downstream.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from twelve_six.data import expanded_global_dedup_v9 as v9

REPORT_SCHEMA: Final = "12-6.d03-expanded-global-dedup-v10-pep-loc-report.v1"
SURVIVOR_SCHEMA: Final = "12-6.d03-expanded-global-dedup-v10-pep-loc-survivors.v1"

PEP_FAMILY: Final = "en.python.peps.public-domain"
PEP_REVISION: Final = "24419b92ae550bf2878716f57c257cba00d3c1a1"
PEP_TREE_SHA1: Final = "fa6b95d941e6fbefc473fbcc50ca93458cdf0857"
PEP_CONTRACT_SHA256: Final = (
    "c0cd68d0b5cfd90e75993c2c9232608d22a616c7213597bb48abca5608d5578b"
)
PEP_PRODUCT_HEAD: Final = "930471ef19c1afff70dee6ed46c4813ded035c9e"
PEP_EVIDENCE_HEAD: Final = "facef815c8c3deac0dfa1843928241e162c7864b"
PEP_EXECUTION_RUN: Final = 34570806209
PEP_EXECUTION_JOB: Final = 103172292323
PEP_CANDIDATE_BYTES: Final = 5_152_269
PEP_CANDIDATE_SHA256: Final = (
    "65d1cf23354efc5a66d7132b67602a7af1715e18a0fb77e729595866c25d33c3"
)
PEP_REPORT_BYTES: Final = 1_270
PEP_REPORT_SHA256: Final = (
    "bc217b3b795bfe5001b8ef266270eee10089c88bc148e8a8f0fa540c2451edcb"
)
PEP_ACCEPTED_RECORDS: Final = 86
PEP_ACCEPTED_UTF8_BYTES: Final = 4_987_775
PEP_INVENTORY_SHA256: Final = (
    "70c29175bf0ca278a19c4b6cef89a532b21a375a908b7103183cebe2841421cf"
)

LOC_FAMILY: Final = "en.loc.selected-digitized-books"
LOC_REVISION: Final = "d31bdba02cdad5104ccec2c02ae799c0bcb5a9a7"
LOC_SHARD_PATH: Final = "data/00000_loc_books.jsonl.gz"
LOC_SHARD_SHA256: Final = (
    "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
)
LOC_CONTRACT_SHA256: Final = (
    "9f6fc98e3a3b720f58963b52b3f3cf3b651bdb9e35f08fa380c99f7b7112f828"
)
LOC_EVIDENCE_HEAD: Final = "cdde55b75a10639e8233de7ccccaaea335b4055a"
LOC_EXECUTION_HEAD: Final = "8ce516160b91ba48b4dd4096e0ffeea409d221de"
LOC_EXECUTION_RUN: Final = 34606219695
LOC_EXECUTION_JOB: Final = 103285166822
LOC_CANDIDATE_BYTES: Final = 4_680_968
LOC_CANDIDATE_SHA256: Final = (
    "f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20"
)
LOC_REPORT_BYTES: Final = 1_538
LOC_REPORT_SHA256: Final = (
    "ae8de852ffeea6d2f697641b2f941eeb044e3377565d0bc1b3b7ba8b6cf3133a"
)
LOC_ACCEPTED_RECORDS: Final = 28
LOC_ACCEPTED_UTF8_BYTES: Final = 4_499_908
LOC_INVENTORY_SHA256: Final = (
    "d3906e11d3779b8d151753a5a126b7578d99d298b16cc6aa3aef129b55463511"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

_PEP_ROW_KEYS = frozenset(
    {
        "record_id",
        "source_family",
        "source_revision",
        "source_path",
        "source_git_blob_sha1",
        "text",
        "text_sha256",
        "utf8_bytes",
        "training_eligible",
    }
)
_LOC_ROW_KEYS = frozenset(
    {
        "record_id",
        "source_family",
        "source_revision",
        "source_shard_path",
        "source_shard_lfs_sha256",
        "source_record_id",
        "item_url",
        "text_file_url",
        "text",
        "text_sha256",
        "utf8_bytes",
        "training_eligible",
        "evaluation_eligible",
    }
)


class ExpandedDedupV10Error(RuntimeError):
    """Raised when PEP/LoC authority or post-V9 composition fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedDedupV10Error(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExpandedDedupV10Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExpandedDedupV10Error(f"{label} is invalid UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be object")
    return value


def _load_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        _require(bool(line.strip()), f"{label} blank row: {line_no}")
        try:
            text = line.decode("utf-8", errors="strict")
            value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExpandedDedupV10Error(
                f"{label} invalid row {line_no}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"{label} row {line_no} must be object")
        rows.append(value)
    _require(bool(rows), f"{label} has no rows")
    return rows


def _require_exact_raw(
    raw: bytes,
    *,
    label: str,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    _require(type(raw) is bytes, f"{label} must be exact bytes")
    _require(len(raw) == expected_bytes, f"{label} byte length drift")
    _require(_sha256(raw) == expected_sha256, f"{label} SHA-256 drift")


def _inventory_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = _canonical(list(rows)) + b"\n"
    return _sha256(raw)


def _validate_text_row(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_keys: frozenset[str],
) -> tuple[str, bytes]:
    _require(type(row) is dict, f"{label} row must be exact dict")
    _require(set(row) == expected_keys, f"{label} row keyset drift")
    record_id = row.get("record_id")
    _require(isinstance(record_id, str) and record_id, f"{label} record_id invalid")
    text = row.get("text")
    _require(isinstance(text, str) and text, f"{label} text missing")
    raw = text.encode("utf-8")
    text_sha = row.get("text_sha256")
    _require(
        isinstance(text_sha, str) and _SHA256_RE.fullmatch(text_sha) is not None,
        f"{label} text SHA malformed: {record_id}",
    )
    _require(_sha256(raw) == text_sha, f"{label} text SHA drift: {record_id}")
    utf8_bytes = row.get("utf8_bytes")
    _require(
        type(utf8_bytes) is int and utf8_bytes == len(raw),
        f"{label} UTF-8 bytes drift: {record_id}",
    )
    _require(
        type(row.get("training_eligible")) is bool
        and row.get("training_eligible") is False,
        f"{label} preclaims training: {record_id}",
    )
    return record_id, raw


def _validate_pep(
    candidate_raw: bytes,
    report_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_exact_raw(
        candidate_raw,
        label="PEP candidate",
        expected_bytes=PEP_CANDIDATE_BYTES,
        expected_sha256=PEP_CANDIDATE_SHA256,
    )
    _require_exact_raw(
        report_raw,
        label="PEP report",
        expected_bytes=PEP_REPORT_BYTES,
        expected_sha256=PEP_REPORT_SHA256,
    )
    rows = _load_jsonl(candidate_raw, label="PEP candidate")
    report = _load_json_object(report_raw, label="PEP report")
    exact = {
        "schema_version": "12-6.d03-pep-public-domain-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": PEP_FAMILY,
        "upstream_revision": PEP_REVISION,
        "upstream_tree_sha1": PEP_TREE_SHA1,
        "contract_identity_sha256": PEP_CONTRACT_SHA256,
        "accepted_documents": PEP_ACCEPTED_RECORDS,
        "accepted_normalized_utf8_bytes": PEP_ACCEPTED_UTF8_BYTES,
        "candidate_inventory_identity_sha256": PEP_INVENTORY_SHA256,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "corpus_admitted": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "split_gate": "NOT_RUN",
        "packing_gate": "NOT_RUN",
        "model_training_executed": False,
        "paid_compute_used": False,
    }
    for key, expected in exact.items():
        value = report.get(key)
        _require(
            value == expected and type(value) is type(expected),
            f"PEP report drift: {key}",
        )
    _require(len(rows) == PEP_ACCEPTED_RECORDS, "PEP candidate row count drift")
    seen: set[str] = set()
    total = 0
    inventory: list[dict[str, Any]] = []
    for row in rows:
        record_id, raw = _validate_text_row(
            row,
            label="PEP",
            expected_keys=_PEP_ROW_KEYS,
        )
        _require(record_id.startswith("pep:peps/pep-"), f"PEP record id drift: {record_id}")
        _require(record_id not in seen, f"duplicate PEP record id: {record_id}")
        seen.add(record_id)
        _require(row["source_family"] == PEP_FAMILY, f"PEP family drift: {record_id}")
        _require(row["source_revision"] == PEP_REVISION, f"PEP revision drift: {record_id}")
        source_path = row["source_path"]
        _require(
            isinstance(source_path, str) and record_id == f"pep:{source_path}",
            f"PEP source path drift: {record_id}",
        )
        blob = row["source_git_blob_sha1"]
        _require(
            isinstance(blob, str) and _SHA1_RE.fullmatch(blob) is not None,
            f"PEP Git blob malformed: {record_id}",
        )
        total += len(raw)
        inventory.append(
            {
                "record_id": record_id,
                "source_git_blob_sha1": blob,
                "text_sha256": row["text_sha256"],
                "utf8_bytes": row["utf8_bytes"],
            }
        )
    _require(total == PEP_ACCEPTED_UTF8_BYTES, "PEP candidate byte total drift")
    _require(_inventory_identity(inventory) == PEP_INVENTORY_SHA256, "PEP inventory drift")
    return rows, report


def _validate_loc(
    candidate_raw: bytes,
    report_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_exact_raw(
        candidate_raw,
        label="LoC candidate",
        expected_bytes=LOC_CANDIDATE_BYTES,
        expected_sha256=LOC_CANDIDATE_SHA256,
    )
    _require_exact_raw(
        report_raw,
        label="LoC report",
        expected_bytes=LOC_REPORT_BYTES,
        expected_sha256=LOC_REPORT_SHA256,
    )
    rows = _load_jsonl(candidate_raw, label="LoC candidate")
    report = _load_json_object(report_raw, label="LoC report")
    exact = {
        "schema_version": "12-6.d03-common-pile-loc-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": LOC_FAMILY,
        "contract_identity_sha256": LOC_CONTRACT_SHA256,
        "upstream_revision": LOC_REVISION,
        "source_shard_path": LOC_SHARD_PATH,
        "source_shard_lfs_sha256": LOC_SHARD_SHA256,
        "full_shard_hash_verified": True,
        "accepted_documents": LOC_ACCEPTED_RECORDS,
        "accepted_normalized_utf8_bytes": LOC_ACCEPTED_UTF8_BYTES,
        "candidate_inventory_identity_sha256": LOC_INVENTORY_SHA256,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "evaluation_eligible": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "balance_family_caps_gate": "NOT_RUN",
        "cluster_safe_split_gate": "NOT_RUN",
        "packing_two_clean_builds_gate": "NOT_RUN",
        "positive_unique_loss_ledger_gate": "NOT_RUN",
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
    for key, expected in exact.items():
        value = report.get(key)
        _require(
            value == expected and type(value) is type(expected),
            f"LoC report drift: {key}",
        )
    _require(len(rows) == LOC_ACCEPTED_RECORDS, "LoC candidate row count drift")
    seen: set[str] = set()
    total = 0
    inventory: list[dict[str, Any]] = []
    for row in rows:
        record_id, raw = _validate_text_row(
            row,
            label="LoC",
            expected_keys=_LOC_ROW_KEYS,
        )
        _require(record_id.startswith("loc:"), f"LoC record id drift: {record_id}")
        _require(record_id not in seen, f"duplicate LoC record id: {record_id}")
        seen.add(record_id)
        _require(row["source_family"] == LOC_FAMILY, f"LoC family drift: {record_id}")
        _require(row["source_revision"] == LOC_REVISION, f"LoC revision drift: {record_id}")
        _require(
            row["source_shard_path"] == LOC_SHARD_PATH,
            f"LoC shard path drift: {record_id}",
        )
        _require(
            row["source_shard_lfs_sha256"] == LOC_SHARD_SHA256,
            f"LoC shard SHA drift: {record_id}",
        )
        source_record_id = row["source_record_id"]
        _require(
            isinstance(source_record_id, str)
            and source_record_id
            and record_id == f"loc:{source_record_id}",
            f"LoC source record id drift: {record_id}",
        )
        for field in ("item_url", "text_file_url"):
            _require(
                isinstance(row[field], str) and row[field].startswith("https://"),
                f"LoC {field} drift: {record_id}",
            )
        _require(
            type(row["evaluation_eligible"]) is bool
            and row["evaluation_eligible"] is False,
            f"LoC preclaims evaluation: {record_id}",
        )
        total += len(raw)
        inventory.append(
            {
                "record_id": record_id,
                "source_record_id": source_record_id,
                "item_url": row["item_url"],
                "text_file_url": row["text_file_url"],
                "text_sha256": row["text_sha256"],
                "utf8_bytes": row["utf8_bytes"],
            }
        )
    _require(total == LOC_ACCEPTED_UTF8_BYTES, "LoC candidate byte total drift")
    _require(_inventory_identity(inventory) == LOC_INVENTORY_SHA256, "LoC inventory drift")
    return rows, report


def _pep_matcher_inputs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    inventory: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for row in rows:
        record_id = str(row["record_id"])
        source_path = str(row["source_path"])
        payload = str(row["text"]).encode("utf-8")
        source_id = f"pep-materialized:{record_id}"
        _require(source_id not in payloads, f"duplicate PEP matcher id: {source_id}")
        inventory.append(
            {
                "source_id": source_id,
                "source_family": PEP_FAMILY,
                "stable_origin_id": f"python/peps@{PEP_REVISION}:{source_path}",
                "stable_object_id": f"sha256:{row['text_sha256']}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"PR1231:{PEP_EVIDENCE_HEAD}",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": row["text_sha256"],
                "acquisition_url": (
                    "https://raw.githubusercontent.com/python/peps/"
                    f"{PEP_REVISION}/{source_path}"
                ),
                "origin_key": f"pep-public-domain:{record_id}",
            }
        )
        payloads[source_id] = payload
    return inventory, payloads


def _loc_matcher_inputs(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    inventory: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    shard_url = (
        "https://huggingface.co/datasets/common-pile/library_of_congress/resolve/"
        f"{LOC_REVISION}/{LOC_SHARD_PATH}"
    )
    for row in rows:
        record_id = str(row["record_id"])
        source_record_id = str(row["source_record_id"])
        payload = str(row["text"]).encode("utf-8")
        source_id = f"loc-materialized:{record_id}"
        _require(source_id not in payloads, f"duplicate LoC matcher id: {source_id}")
        inventory.append(
            {
                "source_id": source_id,
                "source_family": LOC_FAMILY,
                "stable_origin_id": (
                    "common-pile/library_of_congress@"
                    f"{LOC_REVISION}:{source_record_id}"
                ),
                "stable_object_id": f"sha256:{row['text_sha256']}",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"PR1276:{LOC_EVIDENCE_HEAD}",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": row["text_sha256"],
                "acquisition_url": shard_url,
                "origin_key": f"loc-selected-digitized-books:{source_record_id}",
            }
        )
        payloads[source_id] = payload
    return inventory, payloads


def _outer_survivor_authority(
    dedup: Mapping[str, Any],
    selection_projection: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "selection_projection_schema": selection_projection.get("schema_version"),
        "selection_projection_sha256": selection_projection.get(
            "survivor_authority_sha256"
        ),
        "matcher_report_sha256": dedup.get("report_sha256"),
        "pre_dedup_source_object_count": selection_projection.get(
            "pre_dedup_source_object_count"
        ),
        "post_dedup_survivor_source_object_count": selection_projection.get(
            "post_dedup_survivor_source_object_count"
        ),
        "pre_dedup_declared_capacity_bytes": selection_projection.get(
            "pre_dedup_declared_capacity_bytes"
        ),
        "post_dedup_declared_capacity_bytes": selection_projection.get(
            "post_dedup_declared_capacity_bytes"
        ),
        "duplicate_discount_bytes": selection_projection.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": selection_projection.get("duplicate_cluster_count"),
        "duplicate_clusters": copy.deepcopy(
            selection_projection.get("duplicate_clusters")
        ),
        "survivor_source_ids": copy.deepcopy(
            selection_projection.get("survivor_source_ids")
        ),
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def run_expanded_dedup_v10(
    *,
    matcher_audit: Callable[[Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]],
    matcher_verify: Callable[[Mapping[str, Any]], None],
    reconstructed_v8_inventory: Mapping[str, Any],
    reconstructed_v8_payloads: Mapping[str, bytes],
    v8_survivor_authority: Mapping[str, Any],
    data526_evidence: Mapping[str, Any],
    data526_record_inventory: Mapping[str, Any],
    rada_language_report: Mapping[str, Any],
    rada_quality_privacy_report: Mapping[str, Any],
    expected_rada_report_sha256: str,
    rada_rows: Sequence[Mapping[str, Any]],
    rada_raw_jsonl: bytes,
    pep_candidate_raw: bytes,
    pep_report_raw: bytes,
    loc_candidate_raw: bytes,
    loc_report_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exact V9 authority first, then add authenticated PEP+LoC to the same matcher."""

    v9_report, _ = v9.run_expanded_dedup(
        matcher_audit=matcher_audit,
        matcher_verify=matcher_verify,
        reconstructed_v8_inventory=reconstructed_v8_inventory,
        reconstructed_v8_payloads=reconstructed_v8_payloads,
        v8_survivor_authority=v8_survivor_authority,
        data526_evidence=data526_evidence,
        data526_record_inventory=data526_record_inventory,
        rada_language_report=rada_language_report,
        rada_quality_privacy_report=rada_quality_privacy_report,
        expected_rada_report_sha256=expected_rada_report_sha256,
        rada_rows=rada_rows,
        rada_raw_jsonl=rada_raw_jsonl,
    )

    # Re-bind runtime semantics immediately before the expanded call.  V9 already
    # proves the complete DATA526+Rada authority; this second closure check prevents
    # a caller from swapping executable matcher state between V9 and V10.
    v9._verify_matcher_semantic_closure(matcher_audit, matcher_verify)
    pep_rows, _ = _validate_pep(pep_candidate_raw, pep_report_raw)
    loc_rows, _ = _validate_loc(loc_candidate_raw, loc_report_raw)
    verified_rada_rows = v9.validate_rada_rows(
        rada_rows,
        rada_raw_jsonl,
        rada_quality_privacy_report,
    )
    rada_report_sha = v9.validate_rada_quality_privacy_report(
        rada_quality_privacy_report,
        expected_report_sha256=expected_rada_report_sha256,
    )

    prepared_inventory = v9._restrict_lineage_to_survivors(
        reconstructed_v8_inventory,
        v8_survivor_authority,
    )
    base_rows, base_payloads = v9.filter_v8_survivor_inputs(
        prepared_inventory,
        reconstructed_v8_payloads,
        v8_survivor_authority,
    )
    rada_inventory, rada_payloads = v9.build_rada_matcher_inputs(
        verified_rada_rows,
        authority_report_sha256=rada_report_sha,
    )
    pep_inventory, pep_payloads = _pep_matcher_inputs(pep_rows)
    loc_inventory, loc_payloads = _loc_matcher_inputs(loc_rows)

    input_groups = (base_payloads, rada_payloads, pep_payloads, loc_payloads)
    seen_ids: set[str] = set()
    for group in input_groups:
        _require(not (seen_ids & set(group)), "V10 matcher source-id collision")
        seen_ids.update(group)

    v9_vector = v9_report.get("source_vector")
    _require(isinstance(v9_vector, Mapping), "V9 source vector missing")
    base_rada_count = len(base_payloads) + len(rada_payloads)
    base_rada_bytes = sum(len(raw) for raw in base_payloads.values()) + sum(
        len(raw) for raw in rada_payloads.values()
    )
    _require(
        v9_vector.get("pre_dedup_source_object_count") == base_rada_count,
        "V9 reconstructed source count drift before V10",
    )
    _require(
        v9_vector.get("pre_dedup_declared_capacity_bytes") == base_rada_bytes,
        "V9 reconstructed byte total drift before V10",
    )

    combined_inventory = copy.deepcopy(dict(prepared_inventory))
    combined_inventory["sources"] = [
        *base_rows,
        *rada_inventory,
        *pep_inventory,
        *loc_inventory,
    ]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_rule"] = (
        "V10 composes exact hardened V9 DATA526+Rada inputs with independently "
        "qualified real PEP and LoC materialization bytes; matching delegates to "
        "the exact incumbent PR #824 V3 semantic closure."
    )
    combined_payloads: dict[str, bytes] = {}
    for group in input_groups:
        combined_payloads.update(group)

    dedup = matcher_audit(combined_inventory, combined_payloads)
    _require(isinstance(dedup, Mapping), "incumbent matcher returned non-object report")
    matcher_verify(dedup)
    matcher_sha = dedup.get("report_sha256")
    _require(
        isinstance(matcher_sha, str) and _SHA256_RE.fullmatch(matcher_sha) is not None,
        "V10 matcher report identity missing",
    )
    terminal = dedup.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "V10 matcher terminal result missing")
    expected_before = base_rada_bytes + PEP_ACCEPTED_UTF8_BYTES + LOC_ACCEPTED_UTF8_BYTES
    _require(
        terminal.get("declared_capacity_bytes_before") == expected_before,
        "V10 pre-dedup byte total drift",
    )
    _require(dedup.get("source_count") == len(combined_payloads), "V10 source count drift")

    core = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "matcher_lineage": "MERGED_PR_824_V3_REUSED_WITHOUT_SEMANTIC_CHANGES",
        "v9_dependency": {
            "product_head_sha": "5dbf143c7ca15999eadb28fecb952118b59040de",
            "validated_v9_report_sha256": v9_report["report_sha256"],
            "source_object_count": base_rada_count,
            "pre_dedup_payload_bytes": base_rada_bytes,
        },
        "pep_public_domain": {
            "product_head_sha": PEP_PRODUCT_HEAD,
            "evidence_head_sha": PEP_EVIDENCE_HEAD,
            "real_execution_run": PEP_EXECUTION_RUN,
            "real_execution_job": PEP_EXECUTION_JOB,
            "candidate_sha256": PEP_CANDIDATE_SHA256,
            "report_sha256": PEP_REPORT_SHA256,
            "candidate_inventory_sha256": PEP_INVENTORY_SHA256,
            "source_object_count": len(pep_rows),
            "payload_bytes": PEP_ACCEPTED_UTF8_BYTES,
            "corpus_credit_added": 0,
        },
        "loc_selected_digitized_books": {
            "evidence_head_sha": LOC_EVIDENCE_HEAD,
            "execution_head_sha": LOC_EXECUTION_HEAD,
            "real_execution_run": LOC_EXECUTION_RUN,
            "real_execution_job": LOC_EXECUTION_JOB,
            "candidate_sha256": LOC_CANDIDATE_SHA256,
            "report_sha256": LOC_REPORT_SHA256,
            "candidate_inventory_sha256": LOC_INVENTORY_SHA256,
            "source_object_count": len(loc_rows),
            "payload_bytes": LOC_ACCEPTED_UTF8_BYTES,
            "corpus_credit_added": 0,
        },
        "source_vector": {
            "pre_dedup_source_object_count": dedup["source_count"],
            "pre_dedup_declared_capacity_bytes": terminal[
                "declared_capacity_bytes_before"
            ],
            "post_dedup_conservative_unique_bytes": terminal[
                "conservative_unique_capacity_bytes_after"
            ],
            "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
            "duplicate_cluster_count": terminal["duplicate_cluster_count"],
        },
        "dedup_v3": copy.deepcopy(dict(dedup)),
        "raw_text_emitted": False,
        "claim_boundary": {
            "pep_real_materialization_authenticated": True,
            "loc_real_materialization_authenticated": True,
            "expanded_global_dedup_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "two_clean_builds_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    report = {**core, "report_sha256": _sha256(_canonical(core))}
    projection = v9._derive_survivors(dedup)
    survivors = _outer_survivor_authority(dedup, projection)
    return report, survivors
