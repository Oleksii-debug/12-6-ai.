#!/usr/bin/env python3
"""Execute exact Lesia Wikisource global-dedup survivors through DATA-232/G05/G06.

Execution-only carrier. It consumes the immutable #2807 two-clean global-dedup
authority, freshly rematerializes the exact Lesia 1892 Wikisource candidate, retains
only the physically proven parent survivor IDs in ephemeral memory, and delegates
reserved-evaluation decontamination, G05 quality, and G06 privacy to the incumbent
current-clean implementation. Durable output is text-free and grants zero canonical,
tokenizer, training, or scale authority.

The parent matcher reports 169,399 selected extension payload bytes but only 167,920
marginal globally-unique bytes. Because post-QP filtering can remove arbitrary
records, this carrier deliberately does not add surviving payload bytes directly to
the canonical UA capacity ledger. A later cross-lineage re-dedup/composition gate
must recompute exact marginal capacity over the post-QP survivors.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_code_delta_decontam_qp_v1 as current_qp
from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved

SCHEMA = "12-6.d03-wikisource-lesia1892-decontam-g05-g06-execution.v1"
BASE_EXECUTION_HEAD = "ce30dd256e5b583ed6920d78930ee4b1a20429c5"
CURRENT_QP_RUNNER_BLOB = "3e1d9a68622a3e709488a128967f7d83ff1074ea"
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
CURRENT_RESERVED_BLOB = "e5c555e3cd27844e98d4ae91af0b746e427f36c9"

PARENT_EXECUTION_HEAD = "0a06d20e77d3a5f50f838c6fe4eb39ce40cccf0d"
PARENT_WORKFLOW_RUN_ID = 37375534207
PARENT_ARTIFACT_ID = 11373941621
PARENT_PROOF_ARTIFACT_ID = 11373372018
PARENT_ARTIFACT_ZIP_SHA256 = (
    "4d26bc6d451bf7765b341317ed73cf40ff24631ae02fbf2565b6b3b8a55bae80"
)
PARENT_PROOF_ARTIFACT_ZIP_SHA256 = (
    "e94814590131fddb0727b7afca763a006e69187d855dc7cf5d64074fc767be9d"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "293cd4257ab2f1ccdf93dc9140db0db94c1dc5a5469b646e0489bc843ba266cd"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "acc0da17667115a49a1cd4312583c04f689bc5864f65984f5b62600d78e5ed70"
)
PARENT_PROOF_FILE_SHA256 = (
    "d109e9584b8b0d1f42e64706623a69d1fa89426a55fc48e6c1091eaa1bf83511"
)
PARENT_MATCHER_REPORT_SHA256 = (
    "751a7a685051df3c2a36a1c729f3c2fa941c20a93db5c343e33654ee3e011cc4"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "1a396534e0c78c7062204c69f99781bf09f8dde5066bc38cd0e160043b37f7b5"
)
PARENT_EVIDENCE_IDENTITY_SHA256 = (
    "316bdaba189d344acfcb77634324dc3669679f17f40bf0bc8679598ef76f7d57"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "82adeb19349df0cc28265f5d2d4c846985f7458afc2192474ae2e8416d28808d"
)

SOURCE_FAMILY = "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv"
EXPECTED_CANDIDATE_SHA256 = (
    "13025e767ff3f92c0809de807893a5002b691c296b345c12978249c11bc44d12"
)
EXPECTED_MATERIALIZATION_REPORT_SHA256 = (
    "29ebf68ff368fedb1e5bdae7ecfae2f81e4b96b8236f92f630114f1567375e9a"
)
EXPECTED_MATERIALIZATION_REPORT_IDENTITY = (
    "8f3ec6c8ecba70f74c6b024873afe9c1d4574be5415d7dcd54bfd1cdd0aa81ce"
)
EXPECTED_LESIA_OBJECTS = 107
EXPECTED_LESIA_PAYLOAD_BYTES = 169_399
EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES = 167_920

CURRENT_POST_QP_UA_BYTES_REFERENCE = 205_315
UA_TARGET_BYTES = 9_000_000
CURRENT_UA_GAP_BYTES = UA_TARGET_BYTES - CURRENT_POST_QP_UA_BYTES_REFERENCE

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_KEYS = {
    "source_id",
    "source_family_id",
    "language",
    "modality",
    "page_number",
    "page_title",
    "page_revision_id",
    "normalized_sha256",
    "normalized_utf8_bytes",
    "training_eligible",
    "evaluation_eligible",
    "text",
}
_PARENT_SURVIVOR_KEYS = {
    "declared_capacity_bytes",
    "expected_raw_sha256",
    "origin_key",
    "source_family",
    "source_id",
    "stable_object_id",
    "stable_origin_id",
}


class WikisourceCleanExecutionError(RuntimeError):
    """Fail-closed Lesia post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WikisourceCleanExecutionError(message)


def canonical(value: Any) -> bytes:
    return current_qp.canonical(value)


def sha256(raw: bytes) -> str:
    return current_qp.sha256(raw)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
        raise WikisourceCleanExecutionError(
            "git failed: " + " ".join(args) + ": " + detail
        )
    return result


def _verify_self_hash(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    *,
    label: str,
) -> None:
    require(type(value) is dict, f"{label}: root must be exact object")
    copied = copy.deepcopy(value)
    identity = copied.pop(field, None)
    require(
        type(identity) is str and _SHA64.fullmatch(identity) is not None,
        f"{label}: identity malformed",
    )
    require(identity == expected, f"{label}: identity drift")
    require(sha256(canonical(copied)) == identity, f"{label}: self-hash mismatch")


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        type(expected_execution_head) is str
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(
        git("rev-parse", "HEAD").stdout.strip() == expected_execution_head,
        "HEAD drift",
    )
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        BASE_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "#2782 execution base is not ancestor")

    expected_blobs = {
        "tools/run_d03_code_delta_decontam_qp_v1.py": CURRENT_QP_RUNNER_BLOB,
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
        "src/twelve_six/data/current_reserved_decontamination_v1.py": (
            CURRENT_RESERVED_BLOB
        ),
    }
    for path, expected_blob in expected_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}").stdout.strip() == expected_blob,
            f"Git blob drift: {path}",
        )
        require(
            git("hash-object", str(ROOT / path)).stdout.strip() == expected_blob,
            f"worktree blob drift: {path}",
        )

    carrier = "tools/run_d03_wikisource_lesia1892_decontam_qp_v1.py"
    carrier_blob = git("rev-parse", f"HEAD:{carrier}").stdout.strip()
    require(
        git("hash-object", str(ROOT / carrier)).stdout.strip() == carrier_blob,
        "executing carrier worktree drift",
    )
    dependency_blobs = clean.verify_dependency_blobs()
    return {
        **expected_blobs,
        **{f"current_clean:{key}": value for key, value in dependency_blobs.items()},
        carrier: carrier_blob,
    }


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
    proof_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    survivors = current_qp.load_json(
        survivors_path,
        "Lesia parent survivors",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = current_qp.load_json(
        evidence_path,
        "Lesia parent evidence",
        expected_sha256=PARENT_EVIDENCE_FILE_SHA256,
    )
    proof = current_qp.load_json(
        proof_path,
        "Lesia parent two-clean proof",
        expected_sha256=PARENT_PROOF_FILE_SHA256,
    )
    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="Lesia parent survivors",
    )
    _verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        PARENT_EVIDENCE_IDENTITY_SHA256,
        label="Lesia parent evidence",
    )
    _verify_self_hash(
        proof,
        "proof_identity_sha256",
        PARENT_PROOF_IDENTITY_SHA256,
        label="Lesia parent two-clean proof",
    )

    require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Lesia parent execution head drift",
    )
    require(
        evidence.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Lesia evidence/survivor root drift",
    )
    require(
        proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Lesia proof execution head drift",
    )
    require(
        proof.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Lesia proof survivor root drift",
    )
    require(
        proof.get("combined_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Lesia proof matcher report drift",
    )
    require(
        proof.get("two_fresh_executions_converged") is True,
        "Lesia parent lacks two-clean convergence",
    )
    require(
        proof.get("candidate_records") == EXPECTED_LESIA_OBJECTS,
        "Lesia proof candidate count drift",
    )
    require(
        proof.get("candidate_normalized_utf8_bytes")
        == EXPECTED_LESIA_PAYLOAD_BYTES,
        "Lesia proof candidate bytes drift",
    )
    require(
        proof.get("marginal_global_unique_capacity_bytes")
        == EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
        "Lesia proof marginal capacity drift",
    )

    source = evidence.get("wikisource_lesia1892")
    require(isinstance(source, Mapping), "Lesia parent source evidence missing")
    require(source.get("candidate_sha256") == EXPECTED_CANDIDATE_SHA256, "candidate SHA drift")
    require(
        source.get("materialization_report_identity_sha256")
        == EXPECTED_MATERIALIZATION_REPORT_IDENTITY,
        "materialization report identity drift",
    )
    require(
        source.get("candidate_source_object_count") == EXPECTED_LESIA_OBJECTS,
        "candidate object count drift",
    )
    require(
        source.get("candidate_normalized_utf8_bytes") == EXPECTED_LESIA_PAYLOAD_BYTES,
        "candidate byte count drift",
    )
    require(
        source.get("post_dedup_selected_extension_source_object_count")
        == EXPECTED_LESIA_OBJECTS,
        "selected extension object count drift",
    )
    require(
        source.get("post_dedup_selected_extension_payload_bytes")
        == EXPECTED_LESIA_PAYLOAD_BYTES,
        "selected extension payload bytes drift",
    )
    require(
        source.get("marginal_global_unique_capacity_bytes")
        == EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
        "marginal global unique capacity drift",
    )
    require(
        source.get("global_dedup_loss_vs_raw_candidate_bytes")
        == EXPECTED_LESIA_PAYLOAD_BYTES - EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
        "parent global-dedup loss arithmetic drift",
    )

    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Lesia survivor/matcher lineage drift",
    )
    require(survivors.get("source_family") == SOURCE_FAMILY, "Lesia family drift")
    require(
        survivors.get("candidate_source_object_count") == EXPECTED_LESIA_OBJECTS,
        "Lesia survivor candidate count drift",
    )
    require(
        survivors.get("candidate_payload_utf8_bytes") == EXPECTED_LESIA_PAYLOAD_BYTES,
        "Lesia survivor candidate bytes drift",
    )
    require(
        survivors.get("post_dedup_selected_extension_source_object_count")
        == EXPECTED_LESIA_OBJECTS,
        "Lesia survivor selected count drift",
    )
    require(
        survivors.get("post_dedup_selected_extension_payload_bytes")
        == EXPECTED_LESIA_PAYLOAD_BYTES,
        "Lesia survivor selected bytes drift",
    )
    require(
        survivors.get("marginal_global_unique_capacity_bytes")
        == EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
        "Lesia survivor marginal capacity drift",
    )

    rows = survivors.get("survivors")
    require(
        isinstance(rows, list) and len(rows) == EXPECTED_LESIA_OBJECTS,
        "Lesia survivor rows missing or cardinality drift",
    )
    ids: list[str] = []
    total = 0
    for index, row in enumerate(rows):
        require(type(row) is dict, f"Lesia survivor row {index} is not exact object")
        require(
            set(row) == _PARENT_SURVIVOR_KEYS,
            f"Lesia survivor row {index} schema drift",
        )
        source_id = row.get("source_id")
        require(type(source_id) is str and bool(source_id), "Lesia survivor ID invalid")
        require(row.get("source_family") == SOURCE_FAMILY, "Lesia survivor family drift")
        byte_count = row.get("declared_capacity_bytes")
        require(type(byte_count) is int and byte_count > 0, "Lesia survivor bytes invalid")
        digest = row.get("expected_raw_sha256")
        require(
            type(digest) is str and _SHA64.fullmatch(digest) is not None,
            "Lesia survivor payload digest invalid",
        )
        require(
            row.get("stable_object_id") == f"sha256:{digest}",
            "Lesia survivor stable object drift",
        )
        ids.append(source_id)
        total += byte_count
    require(len(ids) == len(set(ids)), "duplicate Lesia survivor source ID")
    require(total == EXPECTED_LESIA_PAYLOAD_BYTES, "Lesia survivor byte total drift")

    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "Lesia parent truth boundary missing")
    require(
        truth.get("global_dedup_execution_complete") is True,
        "Lesia global dedup not terminal",
    )
    require(
        truth.get("reserved_evaluation_decontamination_complete") is False,
        "Lesia parent unexpectedly claims DATA-232 complete",
    )
    require(truth.get("canonical_capacity_credited") == 0, "parent fabricated credit")
    require(truth.get("training_executed") is False, "parent executed training")
    require(truth.get("tokenizer_fit_authorized") is False, "parent authorized tokenizer")
    return survivors, evidence, rows


def acquire_lesia_survivors(
    *,
    candidate_jsonl: Path,
    materialization_report_json: Path,
    parent_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    candidate_raw = candidate_jsonl.read_bytes()
    require(sha256(candidate_raw) == EXPECTED_CANDIDATE_SHA256, "fresh candidate SHA drift")
    report = current_qp.load_json(
        materialization_report_json,
        "fresh Lesia materialization report",
        expected_sha256=EXPECTED_MATERIALIZATION_REPORT_SHA256,
    )
    require(
        report.get("report_sha256") == EXPECTED_MATERIALIZATION_REPORT_IDENTITY,
        "fresh materialization report identity drift",
    )
    candidate_summary = report.get("candidate")
    require(isinstance(candidate_summary, Mapping), "fresh candidate summary missing")
    require(
        candidate_summary.get("candidate_jsonl_sha256") == EXPECTED_CANDIDATE_SHA256,
        "fresh report candidate SHA drift",
    )
    require(
        candidate_summary.get("page_count") == EXPECTED_LESIA_OBJECTS,
        "fresh report page count drift",
    )
    require(
        candidate_summary.get("normalized_utf8_bytes") == EXPECTED_LESIA_PAYLOAD_BYTES,
        "fresh report byte count drift",
    )

    parent_by_id = {str(row["source_id"]): row for row in parent_rows}
    require(
        len(parent_by_id) == EXPECTED_LESIA_OBJECTS,
        "parent survivor ID map cardinality drift",
    )
    candidate_rows = current_qp.load_jsonl(candidate_jsonl, "fresh Lesia candidate")
    require(
        len(candidate_rows) == EXPECTED_LESIA_OBJECTS,
        "fresh candidate row count drift",
    )

    source_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    seen_pages: set[int] = set()
    seen_revisions: set[int] = set()
    total = 0
    for index, row in enumerate(candidate_rows, start=1):
        require(set(row) == _CANDIDATE_KEYS, f"candidate row {index} schema drift")
        source_id = row["source_id"]
        family = row["source_family_id"]
        language = row["language"]
        modality = row["modality"]
        page_number = row["page_number"]
        revision_id = row["page_revision_id"]
        digest = row["normalized_sha256"]
        byte_count = row["normalized_utf8_bytes"]
        text = row["text"]
        require(type(source_id) is str and bool(source_id), "candidate source ID invalid")
        require(family == SOURCE_FAMILY, "candidate family drift")
        require(language == "uk" and modality == "text", "candidate language/modality drift")
        require(type(page_number) is int and page_number > 0, "candidate page invalid")
        require(type(revision_id) is int and revision_id > 0, "candidate revision invalid")
        require(page_number not in seen_pages, "duplicate candidate page")
        require(revision_id not in seen_revisions, "duplicate candidate revision")
        require(
            type(digest) is str and _SHA64.fullmatch(digest) is not None,
            "candidate digest invalid",
        )
        require(type(byte_count) is int and byte_count > 0, "candidate byte count invalid")
        require(type(text) is str and bool(text), "candidate text invalid")
        require(row["training_eligible"] is False, "candidate training flag drift")
        require(row["evaluation_eligible"] is False, "candidate evaluation flag drift")
        raw = text.encode("utf-8")
        require(len(raw) == byte_count, "candidate physical byte count drift")
        require(sha256(raw) == digest, "candidate physical SHA drift")

        matcher_source_id = f"wikisource-lesia1892:{source_id}:rev:{revision_id}"
        parent_row = parent_by_id.get(matcher_source_id)
        require(type(parent_row) is dict, "fresh candidate row absent from parent survivors")
        require(
            parent_row["source_family"] == SOURCE_FAMILY
            and parent_row["declared_capacity_bytes"] == byte_count
            and parent_row["expected_raw_sha256"] == digest
            and parent_row["stable_object_id"] == f"sha256:{digest}",
            "fresh candidate/parent survivor binding drift",
        )
        require(
            parent_row["stable_origin_id"]
            == f"uk.wikisource.org:{source_id}:oldid:{revision_id}",
            "fresh candidate stable origin drift",
        )
        require(
            parent_row["origin_key"]
            == f"wikisource-lesia1892:page:{page_number}:revision:{revision_id}",
            "fresh candidate origin key drift",
        )
        source_rows.append(parent_row)
        payloads[matcher_source_id] = raw
        total += byte_count
        seen_pages.add(page_number)
        seen_revisions.add(revision_id)

    require(
        set(payloads) == set(parent_by_id),
        "fresh candidate does not exactly cover parent survivor IDs",
    )
    require(total == EXPECTED_LESIA_PAYLOAD_BYTES, "fresh candidate byte total drift")
    source_rows.sort(key=lambda row: row["source_id"])
    authority = {
        "source_family": SOURCE_FAMILY,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "materialization_report_identity_sha256": (
            EXPECTED_MATERIALIZATION_REPORT_IDENTITY
        ),
        "post_global_dedup_survivor_authority_sha256": (
            PARENT_SURVIVOR_AUTHORITY_SHA256
        ),
        "post_global_dedup_survivor_objects": len(source_rows),
        "post_global_dedup_selected_payload_bytes": total,
        "parent_marginal_global_unique_capacity_bytes": (
            EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES
        ),
        "survivor_ids_sha256": sha256(canonical(sorted(payloads))),
    }
    return source_rows, payloads, authority


def build_training_authorities(
    source_rows: list[dict[str, Any]],
    payloads: Mapping[str, bytes],
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    str,
]:
    records: list[dict[str, str]] = []
    inventory_rows: list[dict[str, Any]] = []
    for row in source_rows:
        source_id = str(row["source_id"])
        raw = payloads[source_id]
        text = raw.decode("utf-8", errors="strict")
        require(text.encode("utf-8") == raw, f"{source_id}: UTF-8 round-trip drift")
        records.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": SOURCE_FAMILY,
                "modality": "uk",
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": SOURCE_FAMILY,
                "modality": "uk",
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )

    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(len(records) == EXPECTED_LESIA_OBJECTS, "Lesia training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows)
        == EXPECTED_LESIA_PAYLOAD_BYTES,
        "Lesia training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-wikisource-lesia1892-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_LESIA_OBJECTS,
        "total_payload_bytes": EXPECTED_LESIA_PAYLOAD_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-wikisource-lesia1892-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_LESIA_OBJECTS,
        "selected_extension_payload_bytes": EXPECTED_LESIA_PAYLOAD_BYTES,
        "parent_marginal_global_unique_capacity_bytes": (
            EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES
        ),
        "source_ids_sha256": sha256(canonical(sorted(payloads))),
    }
    subset = {
        **subset_core,
        "subset_authority_sha256": sha256(canonical(subset_core)),
    }

    projection = reserved._record_projection(records)
    handoff_core = {
        "schema_version": reserved.TRAINING_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": inventory[
            "inventory_identity_sha256"
        ],
        "input_survivor_authority_sha256": subset["subset_authority_sha256"],
        "retained_source_count": EXPECTED_LESIA_OBJECTS,
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": sha256(canonical(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff = {
        **handoff_core,
        "handoff_identity_sha256": sha256(canonical(handoff_core)),
    }
    training_records_sha = clean._sha256(
        b"".join(clean._cjson(row) for row in records)
    )
    training_handoff_raw_sha = clean._sha256(clean._cjson(handoff))
    return (
        records,
        inventory,
        subset,
        handoff,
        training_records_sha,
        training_handoff_raw_sha,
    )


def execute(
    *,
    expected_execution_head: str,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
    parent_proof_json: Path,
    candidate_jsonl: Path,
    materialization_report_json: Path,
    evaluation_records_jsonl: Path,
    base_reserved_binding_json: Path,
    eval647_manifest_json: Path,
    eval647_materialization_evidence_json: Path,
    expected_base_reserved_binding_identity_sha256: str,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    output_evidence: Path,
) -> dict[str, Any]:
    authority_blobs = verify_local_authority(expected_execution_head)
    _survivors, _parent_evidence, parent_rows = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_proof_json,
    )
    source_rows, payloads, source_authority = acquire_lesia_survivors(
        candidate_jsonl=candidate_jsonl,
        materialization_report_json=materialization_report_json,
        parent_rows=parent_rows,
    )
    (
        training_records,
        inventory,
        subset,
        handoff,
        training_records_sha,
        handoff_raw_sha,
    ) = build_training_authorities(source_rows, payloads)

    evaluation_records = current_qp.load_jsonl(
        evaluation_records_jsonl,
        "reserved evaluation records",
    )
    base_reserved_binding = current_qp.load_json(
        base_reserved_binding_json,
        "base reserved binding",
    )
    eval647_manifest = current_qp.load_json(eval647_manifest_json, "EVAL-647 manifest")
    eval647_evidence = current_qp.load_json(
        eval647_materialization_evidence_json,
        "EVAL-647 materialization evidence",
    )

    original = current_qp._patch_clean_release(
        count=EXPECTED_LESIA_OBJECTS,
        training_records_sha=training_records_sha,
        handoff_raw_sha=handoff_raw_sha,
        inventory_identity=inventory["inventory_identity_sha256"],
        subset_authority=subset["subset_authority_sha256"],
    )
    try:
        (
            receipt,
            _data232,
            _decontam,
            _eval647_receipt,
            _quality,
            _privacy,
            final_survivors,
            survivor_inventory,
        ) = clean.execute_current_clean_composition(
            training_records,
            evaluation_records,
            training_handoff_evidence=handoff,
            base_reserved_binding=base_reserved_binding,
            eval647_manifest=eval647_manifest,
            eval647_materialization_evidence=eval647_evidence,
            expected_base_reserved_binding_identity_sha256=(
                expected_base_reserved_binding_identity_sha256
            ),
            expected_composed_reserved_binding_identity_sha256=(
                expected_composed_reserved_binding_identity_sha256
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                expected_eval647_materialization_evidence_identity_sha256
            ),
            expected_eval647_object_set_identity_sha256=(
                expected_eval647_object_set_identity_sha256
            ),
            expected_inventory_identity_sha256=inventory[
                "inventory_identity_sha256"
            ],
            expected_survivor_authority_sha256=subset[
                "subset_authority_sha256"
            ],
            expected_training_handoff_identity_sha256=handoff[
                "handoff_identity_sha256"
            ],
            expected_selection_validation_identity_sha256=(
                expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=expected_final_test_identity_sha256,
        )
        clean.verify_current_clean_composition_receipt(
            receipt,
            expected_receipt_identity_sha256=receipt["receipt_identity_sha256"],
            expected_data232_report_sha256=receipt["data232_report_sha256"],
            expected_decontamination_execution_identity_sha256=receipt[
                "decontamination_execution_identity_sha256"
            ],
            expected_eval647_execution_receipt_identity_sha256=receipt[
                "eval647_execution_receipt_identity_sha256"
            ],
            expected_quality_execution_identity_sha256=receipt[
                "quality_execution_identity_sha256"
            ],
            expected_privacy_execution_identity_sha256=receipt[
                "privacy_execution_identity_sha256"
            ],
            expected_post_decontamination_input_rows_sha256=receipt[
                "post_decontamination_input_rows_sha256"
            ],
            expected_post_quality_input_rows_sha256=receipt[
                "post_quality_input_rows_sha256"
            ],
            expected_survivor_jsonl_sha256=receipt["survivor_jsonl_sha256"],
            expected_survivor_record_inventory_digest_sha256=receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            expected_survivor_payload_inventory_digest_sha256=receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
        )
    finally:
        current_qp._restore_clean_release(original)

    final_bytes = int(receipt["survivor_payload_bytes"])
    gate_loss_bytes = EXPECTED_LESIA_PAYLOAD_BYTES - final_bytes
    require(gate_loss_bytes >= 0, "later gates widened Lesia payload bytes")
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "final survivor count drift",
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": expected_execution_head,
        "parent": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "workflow_run_id": PARENT_WORKFLOW_RUN_ID,
            "artifact_id": PARENT_ARTIFACT_ID,
            "proof_artifact_id": PARENT_PROOF_ARTIFACT_ID,
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "proof_artifact_zip_sha256": PARENT_PROOF_ARTIFACT_ZIP_SHA256,
            "matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "selected_extension_objects": EXPECTED_LESIA_OBJECTS,
            "selected_extension_payload_bytes": EXPECTED_LESIA_PAYLOAD_BYTES,
            "marginal_global_unique_capacity_bytes": (
                EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES
            ),
        },
        "lesia_authority": {
            "source_family": SOURCE_FAMILY,
            "source_object_count": EXPECTED_LESIA_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_LESIA_PAYLOAD_BYTES,
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "subset_authority_sha256": subset["subset_authority_sha256"],
            "training_records_sha256": training_records_sha,
            "training_handoff_raw_sha256": handoff_raw_sha,
            "training_handoff_identity_sha256": handoff[
                "handoff_identity_sha256"
            ],
            "source_authority": source_authority,
        },
        "gate_execution": {
            "data232_report_sha256": receipt["data232_report_sha256"],
            "decontamination_execution_identity_sha256": receipt[
                "decontamination_execution_identity_sha256"
            ],
            "eval647_execution_receipt_identity_sha256": receipt[
                "eval647_execution_receipt_identity_sha256"
            ],
            "quality_execution_identity_sha256": receipt[
                "quality_execution_identity_sha256"
            ],
            "privacy_execution_identity_sha256": receipt[
                "privacy_execution_identity_sha256"
            ],
            "composition_receipt_identity_sha256": receipt[
                "receipt_identity_sha256"
            ],
            "input_records": receipt["input_training_records"],
            "post_decontamination_records": receipt[
                "post_decontamination_records"
            ],
            "post_quality_records": receipt["post_quality_records"],
            "survivor_records": receipt["survivor_records"],
            "survivor_source_objects": receipt["survivor_source_objects"],
            "survivor_payload_bytes": final_bytes,
            "survivor_jsonl_sha256": receipt["survivor_jsonl_sha256"],
            "survivor_record_inventory_digest_sha256": receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            "survivor_payload_inventory_digest_sha256": receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
            "rejection_counts": receipt["rejection_counts"],
            "privacy_detector_counts": receipt["privacy_detector_counts"],
            "later_gate_loss_bytes": gate_loss_bytes,
        },
        "survivor_inventory": survivor_inventory,
        "capacity_observation": {
            "current_post_qp_ua_bytes_reference": CURRENT_POST_QP_UA_BYTES_REFERENCE,
            "ua_target_bytes": UA_TARGET_BYTES,
            "current_exact_ua_gap_bytes": CURRENT_UA_GAP_BYTES,
            "parent_selected_extension_payload_bytes": EXPECTED_LESIA_PAYLOAD_BYTES,
            "parent_marginal_global_unique_capacity_bytes": (
                EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES
            ),
            "qualified_post_qp_survivor_payload_bytes": final_bytes,
            "safe_incremental_unique_upper_bound_bytes": min(
                final_bytes,
                EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
            ),
            "exact_incremental_unique_capacity_credited_bytes": 0,
            "exact_cross_lineage_rededup_required": True,
            "balance_retest_authorized": False,
        },
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_complete_for_lesia_parent": True,
            "reserved_evaluation_decontamination_complete_for_lesia": True,
            "canonical_quality_privacy_complete_for_lesia": True,
            "cross_lineage_rededup_after_post_qp_complete": False,
            "balance_diversity_retest_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "deterministic_pack_two_clean_complete": False,
            "positive_exact_unique_loss_ledger": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "scale_promotion_authorized": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    output_evidence.parent.mkdir(parents=True, exist_ok=True)
    output_evidence.write_bytes(canonical(evidence) + b"\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--parent-survivors-json", type=Path, required=True)
    parser.add_argument("--parent-evidence-json", type=Path, required=True)
    parser.add_argument("--parent-proof-json", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-report-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument("--eval647-manifest-json", type=Path, required=True)
    parser.add_argument("--eval647-materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-base-reserved-binding-identity-sha256", required=True)
    parser.add_argument(
        "--expected-composed-reserved-binding-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-eval647-materialization-evidence-identity-sha256",
        required=True,
    )
    parser.add_argument("--expected-eval647-object-set-identity-sha256", required=True)
    parser.add_argument("--expected-selection-validation-identity-sha256", required=True)
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    try:
        evidence = execute(
            expected_execution_head=args.expected_execution_head,
            parent_survivors_json=args.parent_survivors_json,
            parent_evidence_json=args.parent_evidence_json,
            parent_proof_json=args.parent_proof_json,
            candidate_jsonl=args.candidate_jsonl,
            materialization_report_json=args.materialization_report_json,
            evaluation_records_jsonl=args.evaluation_records_jsonl,
            base_reserved_binding_json=args.base_reserved_binding_json,
            eval647_manifest_json=args.eval647_manifest_json,
            eval647_materialization_evidence_json=(
                args.eval647_materialization_evidence_json
            ),
            expected_base_reserved_binding_identity_sha256=(
                args.expected_base_reserved_binding_identity_sha256
            ),
            expected_composed_reserved_binding_identity_sha256=(
                args.expected_composed_reserved_binding_identity_sha256
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                args.expected_eval647_materialization_evidence_identity_sha256
            ),
            expected_eval647_object_set_identity_sha256=(
                args.expected_eval647_object_set_identity_sha256
            ),
            expected_selection_validation_identity_sha256=(
                args.expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
            output_evidence=args.output_evidence,
        )
    except (
        WikisourceCleanExecutionError,
        current_qp.CodeDeltaCleanExecutionError,
        clean.CurrentCleanExecutionError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_WIKISOURCE_LESIA_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_WIKISOURCE_LESIA_POST_QP=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(
        "POST_QP_LESIA_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_LESIA_OBJECTS="
        f"{evidence['gate_execution']['survivor_records']}"
    )
    print(
        "SAFE_INCREMENTAL_UNIQUE_UPPER_BOUND_BYTES="
        f"{evidence['capacity_observation']['safe_incremental_unique_upper_bound_bytes']}"
    )
    print("EXACT_INCREMENTAL_UNIQUE_CAPACITY_CREDITED_BYTES=0")
    print("CROSS_LINEAGE_REDEDUP_REQUIRED=true")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
