"""Fail-closed ArXiv + LangUK successor intake over the incumbent D03 V9 graph.

This is the repaired successor of the abandoned PR #1404 intake.  It authenticates
the two already-real source-admitted candidates, executes the incumbent V9 authority
first, reconstructs the exact V8-survivor + Rada matcher inputs from V9-authorized
inputs, cross-checks that reconstruction against the V9 source vector, and only then
adds ArXiv + LangUK to the same incumbent matcher callbacks.

No caller-supplied "incumbent graph + trusted head label" shortcut exists here.
All outputs remain zero corpus/training credit until downstream gates consume a real
global-dedup execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from twelve_six.data import expanded_global_dedup_v9 as v9

REPORT_SCHEMA: Final = "12-6.d03-arxiv-languk-postadmission-global-dedup.v2"
SURVIVOR_SCHEMA: Final = "12-6.d03-arxiv-languk-postadmission-survivors.v2"
MATCHER_LINEAGE: Final = "MERGED_PR_824_V3_REUSED_WITHOUT_SEMANTIC_CHANGES"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PostAdmissionDedupIntakeError(RuntimeError):
    """Raised when source authority, V9 authority, or matcher composition drifts."""


@dataclass(frozen=True)
class _SourceSpec:
    key: str
    authority_schema: str
    authority_status: str
    authority_blob_sha1: str
    product_pr: int
    product_head: str
    ci_run: int
    audit_issue: int
    audit_verdict: str
    candidate_sha256: str
    retained_records: int
    retained_normalized_bytes: int
    family: str
    modality: str
    source_dataset: str
    source_revision: str
    source_file: str
    source_sha256: str
    source_bytes: int
    source_id_prefix: str


ARXIV: Final = _SourceSpec(
    key="arxiv",
    authority_schema="12-6.d03-arxiv-postrights-source-admission.v1",
    authority_status="PAYLOAD_SOURCE_ADMISSION_EXECUTED_ZERO_CREDIT",
    authority_blob_sha1="2a812fceb29dc273d75eec46f1957e445738f526",
    product_pr=1373,
    product_head="0494681a9429eedcfd3e9a70e76b1b99a4cd8709",
    ci_run=34657962292,
    audit_issue=1375,
    audit_verdict="PASS_FOR_INTEGRATION_ARXIV_POSTRIGHTS_SOURCE_ADMISSION",
    candidate_sha256="21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327",
    retained_records=1024,
    retained_normalized_bytes=1_139_552,
    family="common-pile/arxiv_abstracts",
    modality="en",
    source_dataset="common-pile/arxiv_abstracts",
    source_revision="46de78c48636c0b46f60049dfd1c5a3710d233f9",
    source_file="00003_arxiv-abstracts.jsonl.gz",
    source_sha256="3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7",
    source_bytes=232_616_926,
    source_id_prefix="arxiv-admitted",
)

LANGUK: Final = _SourceSpec(
    key="languk",
    authority_schema="12-6.d03-languk-postexecution-rights-admission.v1",
    authority_status="SOURCE_RIGHTS_POSTEXEC_ADMISSION_EXECUTED_ZERO_CREDIT",
    authority_blob_sha1="693bf6d83dbf4cbf402693f806573ffdeba2ea7d",
    product_pr=1383,
    product_head="0f98088b61f8708c7a89c706471b4030508b742e",
    ci_run=34659796309,
    audit_issue=1385,
    audit_verdict="PASS_FOR_INTEGRATION_LANGUK_POSTEXEC_SOURCE_ADMISSION",
    candidate_sha256="03bf5089bb6ff4c304e3a299e2480b263a161aad8abe831db0b196af7c585db3",
    retained_records=256,
    retained_normalized_bytes=2_809_632,
    family="ua.languk.supreme-court-decisions",
    modality="uk",
    source_dataset="lang-uk/court-decisions-uk",
    source_revision="2dcac4c941b87bf9c242bdc919cef4b40f4a4813",
    source_file="2024-5K-supreme-court-decisions-deduplicated.parquet",
    source_sha256="9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597",
    source_bytes=20_220_778,
    source_id_prefix="languk-admitted",
)

_ARXIV_ROW_KEYS = frozenset(
    {
        "record_id",
        "source_key",
        "source_label",
        "normalized_sha256",
        "normalized_bytes",
        "text",
    }
)
_LANGUK_ROW_KEYS = frozenset(
    {"record_id", "normalized_sha256", "normalized_bytes", "text"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostAdmissionDedupIntakeError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()


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
            raise PostAdmissionDedupIntakeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} must be non-empty exact bytes")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostAdmissionDedupIntakeError(f"{label} is invalid UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be object")
    return value


def _load_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    _require(type(raw) is bytes and bool(raw), f"{label} must be non-empty exact bytes")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        _require(bool(line.strip()), f"{label} contains blank row {line_no}")
        try:
            value = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PostAdmissionDedupIntakeError(
                f"{label} contains invalid row {line_no}"
            ) from exc
        _require(type(value) is dict, f"{label} row {line_no} must be object")
        rows.append(value)
    _require(bool(rows), f"{label} has no rows")
    return rows


def _strict_equal(value: Any, expected: Any, message: str) -> None:
    _require(value == expected and type(value) is type(expected), message)


def _zero_truth_boundary(boundary: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "family_credit_added",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
        "optimizer_updates",
        "evaluation_authorized_bytes",
    ):
        _strict_equal(boundary.get(key), 0, f"{label} widened {key}")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "learned_weights_created",
        "final_test_payload_accessed",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
        "external_llm_or_api_used_for_data_or_intelligence",
    ):
        _strict_equal(boundary.get(key), False, f"{label} widened {key}")


def _validate_authority(spec: _SourceSpec, raw: bytes) -> dict[str, Any]:
    _require(
        _git_blob_sha1(raw) == spec.authority_blob_sha1,
        f"{spec.key} authority blob drift",
    )
    authority = _load_json_object(raw, label=f"{spec.key} authority")
    _strict_equal(
        authority.get("schema_version"),
        spec.authority_schema,
        f"{spec.key} authority schema drift",
    )
    _strict_equal(
        authority.get("status"),
        spec.authority_status,
        f"{spec.key} authority status drift",
    )
    _strict_equal(
        authority.get("execution_profile"),
        "LOCAL_FREE",
        f"{spec.key} execution profile drift",
    )

    if spec.key == "arxiv":
        historical = authority.get("historical_execution")
        candidate = authority.get("admitted_candidate")
        source = historical if isinstance(historical, Mapping) else {}
        _require(isinstance(candidate, Mapping), "ArXiv admitted candidate missing")
        _strict_equal(source.get("run_id"), 34563405053, "ArXiv execution run drift")
        _strict_equal(
            candidate.get("candidate_sha256"),
            spec.candidate_sha256,
            "ArXiv candidate drift",
        )
        _strict_equal(
            candidate.get("retained_records"),
            spec.retained_records,
            "ArXiv count drift",
        )
        _strict_equal(
            candidate.get("retained_normalized_bytes"),
            spec.retained_normalized_bytes,
            "ArXiv byte total drift",
        )
        _strict_equal(
            candidate.get("source_sha256"),
            spec.source_sha256,
            "ArXiv source SHA drift",
        )
        _strict_equal(
            candidate.get("source_bytes"),
            spec.source_bytes,
            "ArXiv source bytes drift",
        )
        _strict_equal(
            candidate.get("payload_rematerialized_in_this_package"),
            False,
            "ArXiv replay lie",
        )
    elif spec.key == "languk":
        source = authority.get("source_scope")
        candidate = authority.get("admitted_rights_candidate")
        _require(isinstance(source, Mapping), "LangUK source scope missing")
        _require(isinstance(candidate, Mapping), "LangUK admitted candidate missing")
        _strict_equal(source.get("dataset"), spec.source_dataset, "LangUK dataset drift")
        _strict_equal(source.get("revision"), spec.source_revision, "LangUK revision drift")
        _strict_equal(source.get("file"), spec.source_file, "LangUK file drift")
        _strict_equal(source.get("sha256"), spec.source_sha256, "LangUK source SHA drift")
        _strict_equal(source.get("bytes"), spec.source_bytes, "LangUK source bytes drift")
        _strict_equal(
            candidate.get("retained_jsonl_sha256"),
            spec.candidate_sha256,
            "LangUK candidate drift",
        )
        _strict_equal(
            candidate.get("retained_records"),
            spec.retained_records,
            "LangUK count drift",
        )
        _strict_equal(
            candidate.get("retained_normalized_bytes"),
            spec.retained_normalized_bytes,
            "LangUK byte total drift",
        )
        _strict_equal(
            candidate.get("privacy_quality_retest_passed_for_retained_rows"),
            True,
            "LangUK Q/P not passed",
        )
        _strict_equal(
            candidate.get("payload_rematerialized_in_this_package"),
            False,
            "LangUK replay lie",
        )
    else:
        raise PostAdmissionDedupIntakeError(f"unknown source spec: {spec.key}")

    boundary = authority.get("truth_boundary")
    _require(isinstance(boundary, Mapping), f"{spec.key} truth boundary missing")
    _zero_truth_boundary(boundary, label=spec.key)
    return authority


def _validate_row(
    spec: _SourceSpec,
    row: Mapping[str, Any],
) -> tuple[str, bytes, str]:
    expected_keys = _ARXIV_ROW_KEYS if spec.key == "arxiv" else _LANGUK_ROW_KEYS
    _require(
        type(row) is dict and set(row) == expected_keys,
        f"{spec.key} row keyset drift",
    )
    record_id = row.get("record_id")
    _require(
        isinstance(record_id, str) and bool(record_id),
        f"{spec.key} record id invalid",
    )
    if spec.key == "arxiv":
        _strict_equal(
            row.get("source_key"),
            "arxiv_abstracts",
            "ArXiv source key drift",
        )
        _strict_equal(
            row.get("source_label"),
            "arxiv-abstracts",
            "ArXiv source label drift",
        )
    else:
        _require(record_id.isdigit(), "LangUK record id must remain decimal")

    text = row.get("text")
    _require(isinstance(text, str) and bool(text), f"{spec.key} text missing")
    payload = text.encode("utf-8")
    digest = row.get("normalized_sha256")
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        f"{spec.key} normalized SHA malformed",
    )
    _require(_sha256(payload) == digest, f"{spec.key} normalized SHA drift")
    normalized_bytes = row.get("normalized_bytes")
    _require(
        type(normalized_bytes) is int and normalized_bytes == len(payload),
        f"{spec.key} normalized byte count drift",
    )
    return record_id, payload, digest


def _matcher_row(
    spec: _SourceSpec,
    record_id: str,
    digest: str,
    size: int,
) -> dict[str, Any]:
    source_id = f"{spec.source_id_prefix}:{record_id}"
    return {
        "source_id": source_id,
        "source_family": spec.family,
        "stable_origin_id": (
            f"{spec.source_dataset}@{spec.source_revision}:{spec.source_file}#{record_id}"
        ),
        "stable_object_id": f"sha256:{digest}",
        "modality": spec.modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"PR{spec.product_pr}:{spec.product_head}",
        "declared_capacity_bytes": size,
        "expected_raw_bytes": size,
        "expected_raw_sha256": digest,
        "acquisition_url": (
            f"https://huggingface.co/datasets/{spec.source_dataset}/resolve/"
            f"{spec.source_revision}/{spec.source_file}"
        ),
        "origin_key": f"{spec.key}:{record_id}",
    }


def _prepare_source(
    spec: _SourceSpec,
    *,
    authority_raw: bytes,
    candidate_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    _validate_authority(spec, authority_raw)
    _require(
        _sha256(candidate_raw) == spec.candidate_sha256,
        f"{spec.key} candidate SHA drift",
    )
    rows = _load_jsonl(candidate_raw, label=f"{spec.key} candidate")
    _require(
        len(rows) == spec.retained_records,
        f"{spec.key} retained count drift",
    )

    inventory: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    seen_records: set[str] = set()
    total = 0
    for row in rows:
        record_id, payload, digest = _validate_row(spec, row)
        _require(
            record_id not in seen_records,
            f"duplicate {spec.key} record id",
        )
        seen_records.add(record_id)
        matcher_row = _matcher_row(spec, record_id, digest, len(payload))
        source_id = matcher_row["source_id"]
        _require(
            source_id not in payloads,
            f"duplicate {spec.key} matcher source id",
        )
        inventory.append(matcher_row)
        payloads[source_id] = payload
        total += len(payload)
    _require(
        total == spec.retained_normalized_bytes,
        f"{spec.key} retained bytes drift",
    )

    receipt = {
        "source": spec.key,
        "source_family": spec.family,
        "source_admission_product_pr": spec.product_pr,
        "source_admission_head_sha": spec.product_head,
        "source_admission_ci_run": spec.ci_run,
        "source_admission_audit_issue": spec.audit_issue,
        "source_admission_audit_verdict": spec.audit_verdict,
        "authority_blob_sha1": spec.authority_blob_sha1,
        "candidate_sha256": spec.candidate_sha256,
        "source_object_count": len(inventory),
        "pre_dedup_payload_bytes": total,
        "canonical_capacity_credit_added": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
    }
    return inventory, payloads, receipt


def build_post_admission_intake(
    *,
    arxiv_authority_raw: bytes,
    arxiv_candidate_raw: bytes,
    languk_authority_raw: bytes,
    languk_candidate_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    """Authenticate both admitted payloads without granting corpus authority."""

    arxiv_rows, arxiv_payloads, arxiv_receipt = _prepare_source(
        ARXIV,
        authority_raw=arxiv_authority_raw,
        candidate_raw=arxiv_candidate_raw,
    )
    languk_rows, languk_payloads, languk_receipt = _prepare_source(
        LANGUK,
        authority_raw=languk_authority_raw,
        candidate_raw=languk_candidate_raw,
    )

    overlap = set(arxiv_payloads) & set(languk_payloads)
    _require(not overlap, "cross-source matcher source-id collision")
    payloads = {**arxiv_payloads, **languk_payloads}
    sources = [*arxiv_rows, *languk_rows]
    expected_count = ARXIV.retained_records + LANGUK.retained_records
    expected_bytes = ARXIV.retained_normalized_bytes + LANGUK.retained_normalized_bytes
    _require(len(sources) == expected_count, "combined source count drift")
    _require(
        sum(len(raw) for raw in payloads.values()) == expected_bytes,
        "combined bytes drift",
    )

    core = {
        "schema_version": "12-6.d03-arxiv-languk-source-intake-receipt.v2",
        "execution_profile": "LOCAL_FREE",
        "sources": [arxiv_receipt, languk_receipt],
        "combined_pre_dedup_source_object_count": expected_count,
        "combined_pre_dedup_payload_bytes": expected_bytes,
        "raw_text_emitted_in_receipt": False,
        "truth_boundary": {
            "source_admission_consumed_for_intake": True,
            "global_dedup_executed": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    receipt = {**core, "intake_identity_sha256": _sha256(_canonical(core))}
    return sources, payloads, receipt


def _validate_v9_reconstruction(
    v9_report: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    rada_payloads: Mapping[str, bytes],
) -> tuple[int, int]:
    vector = v9_report.get("source_vector")
    _require(isinstance(vector, Mapping), "V9 source vector missing")
    base_rada_count = len(base_payloads) + len(rada_payloads)
    base_rada_bytes = sum(len(raw) for raw in base_payloads.values()) + sum(
        len(raw) for raw in rada_payloads.values()
    )
    _strict_equal(
        vector.get("pre_dedup_source_object_count"),
        base_rada_count,
        "V9 reconstructed source count drift before ArXiv+LangUK",
    )
    _strict_equal(
        vector.get("pre_dedup_declared_capacity_bytes"),
        base_rada_bytes,
        "V9 reconstructed byte total drift before ArXiv+LangUK",
    )
    v9_sha = v9_report.get("report_sha256")
    _require(
        isinstance(v9_sha, str) and _SHA256_RE.fullmatch(v9_sha) is not None,
        "V9 report identity missing",
    )
    return base_rada_count, base_rada_bytes


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
        "duplicate_discount_bytes": selection_projection.get(
            "duplicate_discount_bytes"
        ),
        "duplicate_cluster_count": selection_projection.get(
            "duplicate_cluster_count"
        ),
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
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    return {**core, "survivor_authority_sha256": _sha256(_canonical(core))}


def run_post_admission_global_dedup(
    *,
    matcher_audit: Callable[
        [Mapping[str, Any], Mapping[str, bytes]],
        Mapping[str, Any],
    ],
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
    arxiv_authority_raw: bytes,
    arxiv_candidate_raw: bytes,
    languk_authority_raw: bytes,
    languk_candidate_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute V9 authority first, then extend that exact graph with admitted rows."""

    # Snapshot every authority-bearing mutable input once. The same exact snapshots
    # feed both the V9 proof and the post-proof reconstruction, closing TOCTOU/stateful
    # Mapping substitutions between proof and use.
    v8_inventory_snapshot = copy.deepcopy(dict(reconstructed_v8_inventory))
    v8_payloads_snapshot = dict(reconstructed_v8_payloads)
    _require(
        all(
            type(source_id) is str and type(raw) is bytes
            for source_id, raw in v8_payloads_snapshot.items()
        ),
        "reconstructed V8 payload map must be exact str->bytes",
    )
    v8_survivor_snapshot = copy.deepcopy(dict(v8_survivor_authority))
    data526_evidence_snapshot = copy.deepcopy(dict(data526_evidence))
    data526_inventory_snapshot = copy.deepcopy(dict(data526_record_inventory))
    rada_language_snapshot = copy.deepcopy(dict(rada_language_report))
    rada_qp_snapshot = copy.deepcopy(dict(rada_quality_privacy_report))
    _require(type(rada_raw_jsonl) is bytes, "Rada raw JSONL must be exact bytes")
    rada_raw_snapshot = rada_raw_jsonl
    _require(
        isinstance(rada_rows, Sequence)
        and not isinstance(rada_rows, (str, bytes, bytearray)),
        "Rada rows must be a sequence",
    )
    rada_rows_snapshot: list[dict[str, Any]] = []
    for row in rada_rows:
        _require(isinstance(row, Mapping), "Rada row must be an object")
        rada_rows_snapshot.append(copy.deepcopy(dict(row)))

    v9_report, _ = v9.run_expanded_dedup(
        matcher_audit=matcher_audit,
        matcher_verify=matcher_verify,
        reconstructed_v8_inventory=v8_inventory_snapshot,
        reconstructed_v8_payloads=v8_payloads_snapshot,
        v8_survivor_authority=v8_survivor_snapshot,
        data526_evidence=data526_evidence_snapshot,
        data526_record_inventory=data526_inventory_snapshot,
        rada_language_report=rada_language_snapshot,
        rada_quality_privacy_report=rada_qp_snapshot,
        expected_rada_report_sha256=expected_rada_report_sha256,
        rada_rows=rada_rows_snapshot,
        rada_raw_jsonl=rada_raw_snapshot,
    )

    v9._verify_matcher_semantic_closure(matcher_audit, matcher_verify)

    extension_rows, extension_payloads, intake_receipt = build_post_admission_intake(
        arxiv_authority_raw=arxiv_authority_raw,
        arxiv_candidate_raw=arxiv_candidate_raw,
        languk_authority_raw=languk_authority_raw,
        languk_candidate_raw=languk_candidate_raw,
    )
    verified_rada_rows = v9.validate_rada_rows(
        rada_rows_snapshot,
        rada_raw_snapshot,
        rada_qp_snapshot,
    )
    rada_report_sha = v9.validate_rada_quality_privacy_report(
        rada_qp_snapshot,
        expected_report_sha256=expected_rada_report_sha256,
    )
    prepared_inventory = v9._restrict_lineage_to_survivors(
        v8_inventory_snapshot,
        v8_survivor_snapshot,
    )
    base_rows, base_payloads = v9.filter_v8_survivor_inputs(
        prepared_inventory,
        v8_payloads_snapshot,
        v8_survivor_snapshot,
    )
    rada_inventory, rada_payloads = v9.build_rada_matcher_inputs(
        verified_rada_rows,
        authority_report_sha256=rada_report_sha,
    )
    base_rada_count, base_rada_bytes = _validate_v9_reconstruction(
        v9_report,
        base_payloads,
        rada_payloads,
    )

    seen_ids: set[str] = set()
    for group in (base_payloads, rada_payloads, extension_payloads):
        overlap = seen_ids & set(group)
        _require(not overlap, "ArXiv+LangUK matcher source-id collision")
        seen_ids.update(group)

    combined_inventory = copy.deepcopy(dict(prepared_inventory))
    combined_inventory["sources"] = [
        *base_rows,
        *rada_inventory,
        *copy.deepcopy(extension_rows),
    ]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_rule"] = (
        "V9 authority is executed first; exact authenticated ArXiv+LangUK rows are "
        "then composed with the reconstructed V8-survivor+Rada graph and delegated "
        "to the same terminal PR #824 V3 matcher closure."
    )

    combined_payloads: dict[str, bytes] = {}
    for group in (base_payloads, rada_payloads, extension_payloads):
        combined_payloads.update(group)

    dedup = matcher_audit(combined_inventory, combined_payloads)
    _require(isinstance(dedup, Mapping), "incumbent matcher returned non-object report")
    matcher_verify(dedup)
    matcher_sha = dedup.get("report_sha256")
    _require(
        isinstance(matcher_sha, str) and _SHA256_RE.fullmatch(matcher_sha) is not None,
        "ArXiv+LangUK matcher report identity missing",
    )
    terminal = dedup.get("terminal_candidates")
    _require(
        isinstance(terminal, Mapping),
        "ArXiv+LangUK matcher terminal result missing",
    )
    extension_bytes = ARXIV.retained_normalized_bytes + LANGUK.retained_normalized_bytes
    expected_before = base_rada_bytes + extension_bytes
    _strict_equal(
        terminal.get("declared_capacity_bytes_before"),
        expected_before,
        "ArXiv+LangUK pre-dedup byte total drift",
    )
    _strict_equal(
        dedup.get("source_count"),
        len(combined_payloads),
        "ArXiv+LangUK source count drift",
    )

    core = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "matcher_lineage": MATCHER_LINEAGE,
        "v9_dependency": {
            "validated_v9_report_sha256": v9_report["report_sha256"],
            "source_object_count": base_rada_count,
            "pre_dedup_payload_bytes": base_rada_bytes,
        },
        "post_admission_sources": copy.deepcopy(intake_receipt["sources"]),
        "intake_identity_sha256": intake_receipt["intake_identity_sha256"],
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
            "arxiv_source_admission_authenticated": True,
            "languk_source_admission_authenticated": True,
            "incumbent_v9_authority_executed": True,
            "expanded_global_dedup_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "two_clean_builds_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    report = {**core, "report_sha256": _sha256(_canonical(core))}
    projection = v9._derive_survivors(dedup)
    survivors = _outer_survivor_authority(dedup, projection)
    return report, survivors
