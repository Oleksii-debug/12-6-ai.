#!/usr/bin/env python3
"""Execute exact NBU current-40 global-dedup survivors through DATA-232/G05/G06.

Execution-only carrier. It consumes the immutable #2809 two-clean global-dedup
authority, freshly rematerializes the exact #2804 NBU current-40 candidate, validates
that candidate with the exact #2809 intake adapter, retains only the physically proven
parent survivor IDs in ephemeral memory, and delegates reserved-evaluation
decontamination, G05 quality, and G06 privacy to the incumbent #2782 current-clean
implementation. Durable output is text-free and grants zero canonical, tokenizer,
training, final-test, paid-compute, or scale authority.

All 40 NBU source objects survived parent global dedup with zero NBU-specific loss.
This carrier still credits zero incremental canonical capacity: the later composition
ledger must rebind the post-QP survivors to the current canonical cohort before any
training-authorized byte or unique-loss authority can be granted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
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

SCHEMA = "12-6.d03-nbu-current40-decontam-g05-g06-execution.v1"
BASE_EXECUTION_HEAD = "ce30dd256e5b583ed6920d78930ee4b1a20429c5"
CURRENT_QP_RUNNER_BLOB = "3e1d9a68622a3e709488a128967f7d83ff1074ea"
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
CURRENT_RESERVED_BLOB = "e5c555e3cd27844e98d4ae91af0b746e427f36c9"

PARENT_EXECUTION_HEAD = "b5235cfd83852854e345dea6c2e89cfbcd1cef79"
PARENT_WORKFLOW_RUN_ID = 37382526670
PARENT_ARTIFACT_ID = 11375922011
PARENT_ARTIFACT_ZIP_SHA256 = (
    "8a5b5624d75caa57734d3efa86fb0a89f537a7e978bdd6edc4118be1d27ce744"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "e95f2af8c8ee2c1d8264b2e9cffa3f7147101640b74a8842e4968b49854934ab"
)
PARENT_DEDUP_REPORT_FILE_SHA256 = (
    "b049793ee0cb6e1381da6ff41d0e010e71c20a9491ea8c88326d3de72d798d7f"
)
PARENT_PROOF_FILE_SHA256 = (
    "8cef03ad210da947fe1cbab8a66b4c2e58b3b99cb75857d3fef5aa89af862549"
)
PARENT_MATCHER_REPORT_SHA256 = (
    "b38afd2597a627379b9f6e0eee3cfc2a7feea17572e8c6e85c8365e656578819"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "7f65a0f8b1a813d6f21145020eb4acd621290cb14a3d23fffa0913c1fb91b685"
)
PARENT_TWO_CLEAN_AUTHORITY_SHA256 = (
    "7bd8591659d7be1fd94559621baeed2ba6558cc82603c157c99a40feacfef827"
)

NBU_PHYSICAL_HEAD = "1695c16ab038960d74da5a9911643e9426bb9ad4"
NBU_INTAKE_BLOB = "43e54663d5666e4fd26b1f56a792f03aa1ef08ec"
NBU_INTAKE_RECEIPT_IDENTITY = (
    "9d7a6ed644840024bc1855e6ffd5cb36d2f63def2117698f1c5fc4087fa4bd03"
)
SOURCE_FAMILY = "ua.nbu.official-resolutions"
EXPECTED_CANDIDATE_SHA256 = (
    "c1dbff49a6fb62c0ab07305aa9f5c9870e0bccdfd754d3644e4d7d6a6b033eaf"
)
EXPECTED_MATERIALIZATION_REPORT_IDENTITY = (
    "3bd13beb2d737162cff9f93270570acc7c69034df35fd9550bb4b8300fb3f71f"
)
EXPECTED_NBU_OBJECTS = 40
EXPECTED_NBU_BYTES = 891_494
EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES = 891_494

CURRENT_POST_QP_UA_BYTES_REFERENCE = 205_315
UA_TARGET_BYTES = 9_000_000
CURRENT_UA_GAP_BYTES = UA_TARGET_BYTES - CURRENT_POST_QP_UA_BYTES_REFERENCE

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")

class NbuPostQpExecutionError(RuntimeError):
    """Fail-closed NBU post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NbuPostQpExecutionError(message)


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
        raise NbuPostQpExecutionError(
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

    carrier = "tools/run_d03_nbu_current40_decontam_qp_v1.py"
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
    dedup_report_path: Path,
    proof_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    survivors = current_qp.load_json(
        survivors_path,
        "NBU parent survivor authority",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    report = current_qp.load_json(
        dedup_report_path,
        "NBU parent dedup report",
        expected_sha256=PARENT_DEDUP_REPORT_FILE_SHA256,
    )
    proof = current_qp.load_json(
        proof_path,
        "NBU parent two-clean authority",
        expected_sha256=PARENT_PROOF_FILE_SHA256,
    )

    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="NBU parent survivor authority",
    )
    require(
        report.get("report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "NBU parent matcher report identity drift",
    )
    _verify_self_hash(
        proof,
        "two_clean_authority_sha256",
        PARENT_TWO_CLEAN_AUTHORITY_SHA256,
        label="NBU parent two-clean authority",
    )
    require(
        proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "NBU parent execution head drift",
    )
    require(
        proof.get("status")
        == "PASS_TWO_CLEAN_DEDUP_OVER_EXACT_AUDITED_NBU_COPIES_ZERO_CREDIT",
        "NBU parent two-clean status drift",
    )
    dedup = proof.get("dedup")
    require(isinstance(dedup, Mapping), "NBU parent dedup block missing")
    require(dedup.get("fresh_process_count") == 2, "NBU parent replay count drift")
    require(
        dedup.get("report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "NBU parent proof matcher identity drift",
    )
    require(
        dedup.get("survivor_authority_sha256") == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "NBU parent proof survivor authority drift",
    )
    require(
        dedup.get("nbu_survivor_source_object_count") == EXPECTED_NBU_OBJECTS,
        "NBU parent survivor count drift",
    )
    require(
        dedup.get("nbu_survivor_declared_capacity_bytes") == EXPECTED_NBU_BYTES,
        "NBU parent survivor byte drift",
    )

    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "NBU survivor matcher identity drift",
    )
    require(
        survivors.get("nbu_survivor_source_object_count") == EXPECTED_NBU_OBJECTS,
        "NBU survivor object count drift",
    )
    require(
        survivors.get("nbu_survivor_declared_capacity_bytes") == EXPECTED_NBU_BYTES,
        "NBU survivor byte count drift",
    )
    ids = survivors.get("nbu_survivor_source_ids")
    require(
        type(ids) is list
        and len(ids) == EXPECTED_NBU_OBJECTS
        and all(type(item) is str and item.startswith("nbu-admitted:") for item in ids)
        and len(set(ids)) == EXPECTED_NBU_OBJECTS,
        "NBU survivor ID set drift",
    )

    rows = report.get("sources")
    require(type(rows) is list, "NBU parent dedup source rows missing")
    nbu_rows = [
        row
        for row in rows
        if type(row) is dict and row.get("source_family") == SOURCE_FAMILY
    ]
    require(len(nbu_rows) == EXPECTED_NBU_OBJECTS, "NBU dedup row count drift")
    require(
        {row.get("source_id") for row in nbu_rows} == set(ids),
        "NBU dedup rows/survivor IDs drift",
    )
    require(
        sum(int(row["declared_capacity_bytes"]) for row in nbu_rows)
        == EXPECTED_NBU_BYTES,
        "NBU dedup row byte total drift",
    )
    require(
        survivors.get("canonical_capacity_credited") == 0
        and survivors.get("training_authorized_bytes") == 0,
        "NBU parent survivor authority widened",
    )
    truth = proof.get("truth_boundary")
    require(isinstance(truth, Mapping), "NBU parent truth boundary missing")
    for key in (
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        require(truth.get(key) == 0, f"NBU parent fabricated numeric credit: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
    ):
        require(truth.get(key) is False, f"NBU parent widened truth boundary: {key}")
    nbu_rows.sort(key=lambda row: row["source_id"])
    return survivors, report, nbu_rows


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def acquire_nbu_survivors(
    *,
    candidate_jsonl: Path,
    materialization_report_json: Path,
    nbu_intake_path: Path,
    parent_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    module_raw = nbu_intake_path.read_bytes()
    require(_git_blob_sha1(module_raw) == NBU_INTAKE_BLOB, "NBU intake blob drift")
    spec = importlib.util.spec_from_file_location(
        "g6132_nbu_current40_dedup_intake",
        nbu_intake_path,
    )
    require(spec is not None and spec.loader is not None, "cannot load exact NBU intake")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    projection = module.validate_and_project_nbu(
        candidate_jsonl,
        materialization_report_json,
        retain_payloads=True,
    )
    receipt = projection.receipt
    require(
        receipt.get("receipt_identity_sha256") == NBU_INTAKE_RECEIPT_IDENTITY,
        "NBU intake receipt identity drift",
    )
    candidate = receipt.get("candidate")
    require(isinstance(candidate, Mapping), "NBU intake candidate receipt missing")
    require(candidate.get("record_count") == EXPECTED_NBU_OBJECTS, "NBU candidate count drift")
    require(candidate.get("text_utf8_bytes") == EXPECTED_NBU_BYTES, "NBU candidate byte drift")
    require(
        candidate.get("jsonl_sha256") == EXPECTED_CANDIDATE_SHA256,
        "NBU candidate SHA drift",
    )
    sources = projection.sources
    payloads = projection.payloads
    require(
        type(sources) is tuple
        and len(sources) == EXPECTED_NBU_OBJECTS
        and type(payloads) is dict
        and len(payloads) == EXPECTED_NBU_OBJECTS,
        "NBU intake projection cardinality drift",
    )

    parent_by_id = {str(row["source_id"]): row for row in parent_rows}
    require(len(parent_by_id) == EXPECTED_NBU_OBJECTS, "NBU parent row map drift")
    source_rows: list[dict[str, Any]] = []
    total = 0
    for source in sources:
        require(type(source) is dict, "NBU projected source row must be exact dict")
        source_id = source.get("source_id")
        require(type(source_id) is str and source_id in parent_by_id, "NBU projected source not a parent survivor")
        raw = payloads.get(source_id)
        require(type(raw) is bytes and bool(raw), "NBU projected payload missing")
        parent = parent_by_id[source_id]
        digest = sha256(raw)
        require(
            source.get("source_family") == SOURCE_FAMILY
            and source.get("declared_capacity_bytes") == len(raw)
            and source.get("expected_raw_bytes") == len(raw)
            and source.get("expected_raw_sha256") == digest
            and source.get("stable_object_id") == f"sha256:{digest}",
            "NBU projected matcher row binding drift",
        )
        require(
            parent.get("declared_capacity_bytes") == len(raw)
            and parent.get("verified_raw_bytes") == len(raw)
            and parent.get("verified_raw_sha256") == digest,
            "NBU parent report payload binding drift",
        )
        require(
            parent.get("stable_object_id_sha256")
            == hashlib.sha256(str(source["stable_object_id"]).encode("utf-8")).hexdigest(),
            "NBU parent stable-object hash drift",
        )
        require(
            parent.get("stable_origin_id_sha256")
            == hashlib.sha256(str(source["stable_origin_id"]).encode("utf-8")).hexdigest(),
            "NBU parent stable-origin hash drift",
        )
        source_rows.append(copy.deepcopy(source))
        total += len(raw)

    require(set(payloads) == set(parent_by_id), "fresh NBU candidate does not exactly cover parent survivors")
    require(total == EXPECTED_NBU_BYTES, "fresh NBU candidate byte total drift")
    source_rows.sort(key=lambda row: row["source_id"])
    authority = {
        "source_family": SOURCE_FAMILY,
        "physical_head_sha": NBU_PHYSICAL_HEAD,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "materialization_report_identity_sha256": EXPECTED_MATERIALIZATION_REPORT_IDENTITY,
        "intake_adapter_blob_sha1": NBU_INTAKE_BLOB,
        "intake_receipt_identity_sha256": NBU_INTAKE_RECEIPT_IDENTITY,
        "post_global_dedup_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "post_global_dedup_survivor_objects": len(source_rows),
        "post_global_dedup_selected_payload_bytes": total,
        "parent_marginal_global_unique_capacity_bytes": EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES,
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
    require(len(records) == EXPECTED_NBU_OBJECTS, "NBU training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows)
        == EXPECTED_NBU_BYTES,
        "NBU training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-nbu-current40-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_NBU_OBJECTS,
        "total_payload_bytes": EXPECTED_NBU_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-nbu-current40-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_NBU_OBJECTS,
        "selected_extension_payload_bytes": EXPECTED_NBU_BYTES,
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
        "retained_source_count": EXPECTED_NBU_OBJECTS,
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
    nbu_intake_path: Path,
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
    source_rows, payloads, source_authority = acquire_nbu_survivors(
        candidate_jsonl=candidate_jsonl,
        materialization_report_json=materialization_report_json,
        nbu_intake_path=nbu_intake_path,
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
        count=EXPECTED_NBU_OBJECTS,
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
    gate_loss_bytes = EXPECTED_NBU_BYTES - final_bytes
    require(gate_loss_bytes >= 0, "later gates widened NBU payload bytes")
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
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_authority_sha256": PARENT_TWO_CLEAN_AUTHORITY_SHA256,
            "selected_extension_objects": EXPECTED_NBU_OBJECTS,
            "selected_extension_payload_bytes": EXPECTED_NBU_BYTES,
            "marginal_global_unique_capacity_bytes": (
                EXPECTED_PARENT_MARGINAL_UNIQUE_BYTES
            ),
        },
        "nbu_authority": {
            "source_family": SOURCE_FAMILY,
            "source_object_count": EXPECTED_NBU_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_NBU_BYTES,
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
            "parent_selected_extension_payload_bytes": EXPECTED_NBU_BYTES,
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
            "global_cross_source_dedup_complete_for_nbu_parent": True,
            "reserved_evaluation_decontamination_complete_for_nbu": True,
            "canonical_quality_privacy_complete_for_nbu": True,
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
    parser.add_argument("--nbu-intake-path", type=Path, required=True)
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
            nbu_intake_path=args.nbu_intake_path,
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
        NbuPostQpExecutionError,
        current_qp.CodeDeltaCleanExecutionError,
        clean.CurrentCleanExecutionError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_WIKISOURCE_NBU_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_WIKISOURCE_NBU_POST_QP=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(
        "POST_QP_NBU_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_NBU_OBJECTS="
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
