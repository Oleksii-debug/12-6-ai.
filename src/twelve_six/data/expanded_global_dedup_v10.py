"""Bind clean retained DATA-232 plus real PEP+LoC bytes into canonical global dedup.

This module owns composition and authority validation only; duplicate matching is
delegated to the already-integrated canonical indexed incumbent V3 executor.
The clean retained carrier is bound to the independently qualified PR #2183
outputs, while PEP and LoC remain bound to their exact LOCAL_FREE materializations.

All outputs remain zero-credit. Decontamination, canonical quality/privacy,
balance, split, packing, tokenizer fit and training authorization are downstream.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from twelve_six.data import incumbent_dedup_indexed_execution as indexed

_FROZEN_INDEXED_AUDIT = indexed.audit_payloads_indexed
_FROZEN_INDEXED_ATTEST = indexed.attest_incumbent_runtime
_FROZEN_INDEXED_AUDIT_CODE = _FROZEN_INDEXED_AUDIT.__code__
_FROZEN_INDEXED_ATTEST_CODE = _FROZEN_INDEXED_ATTEST.__code__
_FROZEN_INDEXED_AUDIT_DEFAULTS = _FROZEN_INDEXED_AUDIT.__defaults__
_FROZEN_INDEXED_ATTEST_DEFAULTS = _FROZEN_INDEXED_ATTEST.__defaults__

REPORT_SCHEMA: Final = "12-6.d03-expanded-global-dedup-v10-pep-loc-report.v1"
SURVIVOR_SCHEMA: Final = "12-6.d03-expanded-global-dedup-v10-pep-loc-survivors.v1"

SELECTION_PROJECTION_SCHEMA: Final = "12-6.d03-expanded-global-dedup-v10-selection.v1"
SELECTION_RULE: Final = "largest_declared_capacity_then_lexicographically_smallest_source_id"
V3_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"

CLEAN_HANDOFF_SCHEMA: Final = "12-6.postdedup-decontam-handoff.v1"
CLEAN_PRODUCT_HEAD: Final = "22cdc8696a11cb3490538d6060fe41f9b4d67a95"
CLEAN_TRAINING_RECORDS_SHA256: Final = (
    "3458afe0380ea45d328ad3f004b21845a188ca69f2f7a83c24999e9e53268e53"
)
CLEAN_TRAINING_RECORDS_BYTES: Final = 5_851_879
CLEAN_TRAINING_HANDOFF_SHA256: Final = (
    "80bcf2dd28f0d13795ceea01b358c7149b636f55f17b29575d14313b5cee99ee"
)
CLEAN_MATERIALIZATION_IDENTITY_SHA256: Final = (
    "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
)
CLEAN_COMPOSITION_PREFLIGHT_SHA256: Final = (
    "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95"
)
CLEAN_RETAINED_SOURCE_COUNT: Final = 257
CLEAN_DISTINCT_PHYSICAL_SOURCE_COUNT: Final = 244
CLEAN_RETAINED_PAYLOAD_BYTES: Final = 5_601_716

INDEXED_EXECUTOR_GIT_BLOB_SHA1: Final = "f75336008839198b6d46bea4954f120e2d81613c"
INDEXED_CORE_GIT_BLOB_SHA1: Final = "af7be7909501ea9d76604ebed084cec32fbd9456"
DEFAULT_MAX_CANDIDATE_PAIRS: Final = 5_000_000
DEFAULT_MAX_INDEX_POSTINGS: Final = indexed.DEFAULT_MAX_INDEX_POSTINGS
DEFAULT_MAX_PAIR_EXPANSIONS: Final = indexed.DEFAULT_MAX_PAIR_EXPANSIONS

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
    """Raised when clean-retained/PEP/LoC composition fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpandedDedupV10Error(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


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



_CLEAN_RECORD_KEYS = frozenset(
    {"record_id", "source_id", "source_family", "modality", "text"}
)
_CLEAN_PROJECTION_KEYS = frozenset(
    {
        "record_id",
        "source_id",
        "source_family",
        "modality",
        "text_sha256",
        "text_utf8_bytes",
    }
)
_CLEAN_HANDOFF_KEYS = frozenset(
    {
        "schema_version",
        "postdedup_inventory_identity_sha256",
        "input_survivor_authority_sha256",
        "retained_source_count",
        "matcher_input_projection",
        "matcher_input_projection_sha256",
        "raw_text_persisted_in_evidence",
        "final_test_payload_accessed",
        "final_test_outcomes_accessed",
        "authorized_training_exposure",
        "handoff_identity_sha256",
    }
)


def _git_blob_sha1(raw: bytes) -> str:
    prefix = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + raw).hexdigest()


def _module_blob_sha1(module: Any, *, label: str) -> str:
    raw_path = getattr(module, "__file__", None)
    _require(type(raw_path) is str and bool(raw_path), f"{label} source path missing")
    path = Path(raw_path)
    _require(path.suffix == ".py", f"{label} must resolve to Python source")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExpandedDedupV10Error(f"cannot read {label} source") from exc
    return _git_blob_sha1(raw)


def _verify_indexed_runtime(v3_module: Any) -> None:
    _require(
        _module_blob_sha1(indexed, label="indexed executor")
        == INDEXED_EXECUTOR_GIT_BLOB_SHA1,
        "indexed executor Git blob drift",
    )
    core = getattr(indexed, "_core", None)
    _require(core is not None, "indexed executor core missing")
    _require(
        _module_blob_sha1(core, label="indexed executor core")
        == INDEXED_CORE_GIT_BLOB_SHA1,
        "indexed executor core Git blob drift",
    )
    _require(
        indexed.audit_payloads_indexed is _FROZEN_INDEXED_AUDIT
        and _FROZEN_INDEXED_AUDIT.__code__ is _FROZEN_INDEXED_AUDIT_CODE
        and _FROZEN_INDEXED_AUDIT.__defaults__ is _FROZEN_INDEXED_AUDIT_DEFAULTS,
        "indexed audit callable drift",
    )
    _require(
        indexed.attest_incumbent_runtime is _FROZEN_INDEXED_ATTEST
        and _FROZEN_INDEXED_ATTEST.__code__ is _FROZEN_INDEXED_ATTEST_CODE
        and _FROZEN_INDEXED_ATTEST.__defaults__ is _FROZEN_INDEXED_ATTEST_DEFAULTS,
        "indexed attestation callable drift",
    )
    _require(
        _FROZEN_INDEXED_AUDIT.__globals__.get("attest_incumbent_runtime")
        is _FROZEN_INDEXED_ATTEST,
        "indexed core attestation binding drift",
    )
    _FROZEN_INDEXED_ATTEST(v3_module)


def _validate_clean_retained(
    records_raw: bytes,
    handoff_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    _require_exact_raw(
        records_raw,
        label="clean DATA-232 training records",
        expected_bytes=CLEAN_TRAINING_RECORDS_BYTES,
        expected_sha256=CLEAN_TRAINING_RECORDS_SHA256,
    )
    _require(
        type(handoff_raw) is bytes and bool(handoff_raw),
        "clean DATA-232 handoff is empty",
    )
    _require(
        _sha256(handoff_raw) == CLEAN_TRAINING_HANDOFF_SHA256,
        "clean DATA-232 handoff SHA-256 drift",
    )

    records = _load_jsonl(records_raw, label="clean DATA-232 training records")
    handoff = _load_json_object(handoff_raw, label="clean DATA-232 handoff")
    _require(set(handoff) == _CLEAN_HANDOFF_KEYS, "clean handoff keyset drift")
    _require(handoff.get("schema_version") == CLEAN_HANDOFF_SCHEMA, "clean handoff schema drift")
    _require(
        handoff.get("postdedup_inventory_identity_sha256")
        == CLEAN_MATERIALIZATION_IDENTITY_SHA256,
        "clean materialization identity drift",
    )
    _require(
        handoff.get("input_survivor_authority_sha256")
        == CLEAN_COMPOSITION_PREFLIGHT_SHA256,
        "clean composition-preflight identity drift",
    )
    _require(
        type(handoff.get("retained_source_count")) is int
        and handoff["retained_source_count"] == CLEAN_RETAINED_SOURCE_COUNT,
        "clean retained source count drift",
    )
    for key in (
        "raw_text_persisted_in_evidence",
        "final_test_payload_accessed",
        "final_test_outcomes_accessed",
    ):
        _require(handoff.get(key) is False, f"clean handoff truth drift: {key}")
    _require(
        type(handoff.get("authorized_training_exposure")) is int
        and handoff["authorized_training_exposure"] == 0,
        "clean handoff widens training exposure",
    )

    handoff_identity = handoff.get("handoff_identity_sha256")
    _require(
        isinstance(handoff_identity, str)
        and _SHA256_RE.fullmatch(handoff_identity) is not None,
        "clean handoff identity missing",
    )
    handoff_core = dict(handoff)
    handoff_core.pop("handoff_identity_sha256")
    _require(
        _sha256(_canonical(handoff_core)) == handoff_identity,
        "clean handoff self-hash mismatch",
    )

    projection = handoff.get("matcher_input_projection")
    _require(type(projection) is list, "clean matcher projection must be a list")
    _require(
        len(projection) == CLEAN_RETAINED_SOURCE_COUNT,
        "clean matcher projection count drift",
    )
    _require(
        handoff.get("matcher_input_projection_sha256")
        == _sha256(_canonical(projection)),
        "clean matcher projection identity drift",
    )
    by_record: dict[str, dict[str, Any]] = {}
    for raw_projection in projection:
        _require(type(raw_projection) is dict, "clean projection row must be exact dict")
        _require(
            set(raw_projection) == _CLEAN_PROJECTION_KEYS,
            "clean projection row keyset drift",
        )
        record_id = raw_projection.get("record_id")
        _require(
            isinstance(record_id, str) and record_id and record_id not in by_record,
            "clean projection record_id invalid/duplicate",
        )
        for key in ("source_id", "source_family", "modality", "text_sha256"):
            _require(
                isinstance(raw_projection.get(key), str) and bool(raw_projection[key]),
                f"clean projection {key} invalid: {record_id}",
            )
        _require(
            _SHA256_RE.fullmatch(str(raw_projection["text_sha256"])) is not None,
            f"clean projection text SHA malformed: {record_id}",
        )
        _require(
            type(raw_projection.get("text_utf8_bytes")) is int
            and raw_projection["text_utf8_bytes"] > 0,
            f"clean projection text bytes invalid: {record_id}",
        )
        by_record[record_id] = raw_projection

    _require(
        len(records) == CLEAN_RETAINED_SOURCE_COUNT,
        "clean training record count drift",
    )
    matcher_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    seen_records: set[str] = set()
    physical_sources: set[str] = set()
    total_payload_bytes = 0
    for row in records:
        _require(type(row) is dict, "clean training row must be exact dict")
        _require(set(row) == _CLEAN_RECORD_KEYS, "clean training row keyset drift")
        record_id = row.get("record_id")
        _require(
            isinstance(record_id, str) and record_id and record_id not in seen_records,
            "clean training record_id invalid/duplicate",
        )
        seen_records.add(record_id)
        expected = by_record.get(record_id)
        _require(expected is not None, f"clean projection missing record: {record_id}")
        for key in ("source_id", "source_family", "modality"):
            _require(
                isinstance(row.get(key), str)
                and bool(row[key])
                and row[key] == expected[key],
                f"clean row/projection {key} drift: {record_id}",
            )
        text = row.get("text")
        _require(isinstance(text, str) and bool(text), f"clean text missing: {record_id}")
        payload = text.encode("utf-8")
        _require(
            len(payload) == expected["text_utf8_bytes"],
            f"clean text bytes drift: {record_id}",
        )
        _require(
            _sha256(payload) == expected["text_sha256"],
            f"clean text SHA drift: {record_id}",
        )
        source_id = str(row["source_id"])
        physical_sources.add(source_id)
        matcher_source_id = f"clean-retained:{record_id}"
        _require(
            matcher_source_id not in payloads,
            f"clean matcher source-id collision: {matcher_source_id}",
        )
        matcher_rows.append(
            {
                "source_id": matcher_source_id,
                "source_family": row["source_family"],
                "stable_origin_id": f"clean-data232-source:{source_id}",
                "stable_object_id": f"sha256:{expected['text_sha256']}",
                "modality": row["modality"],
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"PR2183:{CLEAN_PRODUCT_HEAD}",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": expected["text_sha256"],
                "acquisition_url": "https://github.com/Oleksii-debug/12-6-ai./pull/2183",
                "origin_key": f"clean-data232-record:{record_id}",
            }
        )
        payloads[matcher_source_id] = payload
        total_payload_bytes += len(payload)

    _require(
        set(by_record) == seen_records,
        "clean handoff/training record coverage drift",
    )
    _require(
        len(physical_sources) == CLEAN_DISTINCT_PHYSICAL_SOURCE_COUNT,
        "clean distinct physical source count drift",
    )
    _require(
        total_payload_bytes == CLEAN_RETAINED_PAYLOAD_BYTES,
        "clean retained payload byte total drift",
    )
    return matcher_rows, payloads, handoff


def _derive_selection_projection(dedup_report: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = dedup_report.get("sources")
    _require(isinstance(source_rows, list) and bool(source_rows), "matcher report source rows missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "matcher source row must be object")
        source_id = row.get("source_id")
        capacity = row.get("declared_capacity_bytes")
        _require(
            isinstance(source_id, str) and source_id and source_id not in by_id,
            "matcher source id invalid/duplicate",
        )
        _require(
            type(capacity) is int and capacity > 0,
            f"matcher capacity invalid: {source_id}",
        )
        by_id[source_id] = row

    terminal = dedup_report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "matcher terminal candidates missing")
    clusters = terminal.get("duplicate_clusters")
    _require(isinstance(clusters, list), "matcher duplicate clusters missing")
    dropped: set[str] = set()
    cluster_members: set[str] = set()
    normalized_clusters: list[dict[str, Any]] = []
    for raw_cluster in clusters:
        _require(
            isinstance(raw_cluster, Sequence) and not isinstance(raw_cluster, (str, bytes)),
            "invalid matcher duplicate cluster",
        )
        cluster = sorted(str(source_id) for source_id in raw_cluster)
        _require(
            len(cluster) >= 2 and len(cluster) == len(set(cluster)),
            "matcher cluster cardinality invalid",
        )
        _require(
            all(source_id in by_id for source_id in cluster),
            "matcher cluster references unknown source",
        )
        _require(
            not (cluster_members & set(cluster)),
            "matcher duplicate clusters overlap",
        )
        cluster_members.update(cluster)
        maximum = max(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in cluster)
        selected = min(
            source_id
            for source_id in cluster
            if int(by_id[source_id]["declared_capacity_bytes"]) == maximum
        )
        dropped.update(source_id for source_id in cluster if source_id != selected)
        normalized_clusters.append(
            {
                "member_source_ids": cluster,
                "selected_source_id": selected,
                "selected_declared_capacity_bytes": maximum,
            }
        )

    survivors = sorted(source_id for source_id in by_id if source_id not in dropped)
    capacity = sum(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in survivors)
    _require(
        capacity == terminal.get("conservative_unique_capacity_bytes_after"),
        "derived survivor bytes do not reproduce matcher report",
    )
    core = {
        "schema_version": SELECTION_PROJECTION_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "matcher_report_sha256": dedup_report.get("report_sha256"),
        "pre_dedup_source_object_count": len(by_id),
        "post_dedup_survivor_source_object_count": len(survivors),
        "pre_dedup_declared_capacity_bytes": terminal.get("declared_capacity_bytes_before"),
        "post_dedup_declared_capacity_bytes": capacity,
        "duplicate_discount_bytes": terminal.get("duplicate_discount_bytes"),
        "duplicate_cluster_count": len(normalized_clusters),
        "duplicate_clusters": sorted(
            normalized_clusters,
            key=lambda item: tuple(item["member_source_ids"]),
        ),
        "survivor_source_ids": survivors,
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "retained_inventory_freeze_complete": False,
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
    v3_module: Any,
    clean_training_records_raw: bytes,
    clean_training_handoff_raw: bytes,
    pep_candidate_raw: bytes,
    pep_report_raw: bytes,
    loc_candidate_raw: bytes,
    loc_report_raw: bytes,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
    max_index_postings: int = DEFAULT_MAX_INDEX_POSTINGS,
    max_pair_expansions: int = DEFAULT_MAX_PAIR_EXPANSIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compose exact clean retained + PEP + LoC inputs and run canonical indexed V3."""

    _verify_indexed_runtime(v3_module)
    clean_inventory, clean_payloads, clean_handoff = _validate_clean_retained(
        clean_training_records_raw,
        clean_training_handoff_raw,
    )
    pep_rows, _ = _validate_pep(pep_candidate_raw, pep_report_raw)
    loc_rows, _ = _validate_loc(loc_candidate_raw, loc_report_raw)
    pep_inventory, pep_payloads = _pep_matcher_inputs(pep_rows)
    loc_inventory, loc_payloads = _loc_matcher_inputs(loc_rows)

    input_groups = (clean_payloads, pep_payloads, loc_payloads)
    seen_ids: set[str] = set()
    for group in input_groups:
        _require(not (seen_ids & set(group)), "V10 matcher source-id collision")
        seen_ids.update(group)

    combined_inventory = {
        "schema_version": V3_INVENTORY_SCHEMA,
        "local_free_only": True,
        "model_training_executed": False,
        "sources": [
            *clean_inventory,
            *pep_inventory,
            *loc_inventory,
        ],
        "lineage_edges": [],
    }
    combined_payloads: dict[str, bytes] = {}
    for group in input_groups:
        combined_payloads.update(group)

    dedup = _FROZEN_INDEXED_AUDIT(
        v3_module,
        combined_inventory,
        combined_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    _require(isinstance(dedup, Mapping), "indexed matcher returned non-object report")

    # Re-attest immediately before invoking the incumbent report verifier so a
    # post-execution in-memory swap cannot turn verification into caller policy.
    _verify_indexed_runtime(v3_module)
    matcher_verify = getattr(v3_module, "verify_report", None)
    _require(callable(matcher_verify), "incumbent V3 report verifier missing")
    matcher_verify(dedup)

    matcher_sha = dedup.get("report_sha256")
    _require(
        isinstance(matcher_sha, str) and _SHA256_RE.fullmatch(matcher_sha) is not None,
        "V10 matcher report identity missing",
    )
    terminal = dedup.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "V10 matcher terminal result missing")
    expected_before = (
        CLEAN_RETAINED_PAYLOAD_BYTES
        + PEP_ACCEPTED_UTF8_BYTES
        + LOC_ACCEPTED_UTF8_BYTES
    )
    expected_source_count = (
        CLEAN_RETAINED_SOURCE_COUNT + PEP_ACCEPTED_RECORDS + LOC_ACCEPTED_RECORDS
    )
    expected_stable_origins = (
        CLEAN_DISTINCT_PHYSICAL_SOURCE_COUNT + PEP_ACCEPTED_RECORDS + LOC_ACCEPTED_RECORDS
    )
    _require(
        terminal.get("declared_capacity_bytes_before") == expected_before,
        "V10 pre-dedup byte total drift",
    )
    _require(
        dedup.get("source_count") == expected_source_count == len(combined_payloads),
        "V10 source count drift",
    )
    _require(
        terminal.get("stable_origin_count") == expected_stable_origins,
        "V10 stable-origin count drift",
    )

    core = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "matcher_lineage": "MERGED_PR_1459_INDEXED_INCUMBENT_V3",
        "clean_retained_dependency": {
            "product_head_sha": CLEAN_PRODUCT_HEAD,
            "training_records_sha256": CLEAN_TRAINING_RECORDS_SHA256,
            "training_handoff_sha256": CLEAN_TRAINING_HANDOFF_SHA256,
            "handoff_identity_sha256": clean_handoff["handoff_identity_sha256"],
            "materialization_identity_sha256": CLEAN_MATERIALIZATION_IDENTITY_SHA256,
            "composition_preflight_identity_sha256": CLEAN_COMPOSITION_PREFLIGHT_SHA256,
            "source_object_count": CLEAN_RETAINED_SOURCE_COUNT,
            "distinct_physical_source_count": CLEAN_DISTINCT_PHYSICAL_SOURCE_COUNT,
            "payload_bytes": CLEAN_RETAINED_PAYLOAD_BYTES,
            "corpus_credit_added": 0,
        },
        "indexed_executor": {
            "module_git_blob_sha1": INDEXED_EXECUTOR_GIT_BLOB_SHA1,
            "core_git_blob_sha1": INDEXED_CORE_GIT_BLOB_SHA1,
            "max_candidate_pairs": max_candidate_pairs,
            "max_index_postings": max_index_postings,
            "max_pair_expansions": max_pair_expansions,
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
            "clean_retained_authority_authenticated": True,
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
    projection = _derive_selection_projection(dedup)
    survivors = _outer_survivor_authority(dedup, projection)
    return report, survivors

