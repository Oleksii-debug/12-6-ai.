#!/usr/bin/env python3
"""Execute exact Franko1901 global-dedup survivors through current DATA-232/G05/G06.

Execution-only carrier. It consumes the exact terminal Franko1901 two-clean
survivor authority, freshly rematerializes the pinned Franko source, retains only
the physically proven global-dedup survivor IDs in ephemeral memory, and delegates
reserved-evaluation decontamination, G05 quality, and G06 privacy to the already
qualified current-clean implementation. Durable output is text-free and grants
zero canonical/training/tokenizer authority.
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
from twelve_six.data import franko1901_dedup_intake as franko

SCHEMA = "12-6.d03-franko1901-decontam-g05-g06-execution.v1"
BASE_EXECUTION_HEAD = "d724db3873d33808361731d1f399f0d9a43079cf"
CURRENT_QP_RUNNER_BLOB = "3e1d9a68622a3e709488a128967f7d83ff1074ea"
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
FRANKO_INTAKE_BLOB = "617b22a3272665337cbe2c542fff17f61e88d6ec"

PARENT_EXECUTION_HEAD = "5e9a9c27257824faf43915a8a75ef57f9ab6d341"
PARENT_WORKFLOW_RUN_ID = 37300292765
PARENT_ARTIFACT_ID = 11341482304
PARENT_ARTIFACT_ZIP_SHA256 = (
    "60cd01afb53bbec2f0b2cb66dbe2b299ffec60592a8e9c5fe75a178476f803ac"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "de2ff6f13ab138caa4ad4471326ab16b97abdb5167a95a48272963445e976839"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "6112dafe7839018fd96293d970843cb5d3c3547a90608face4506ae2e636fd3d"
)
PARENT_MATCHER_REPORT_SHA256 = (
    "f02573a77ab649d03a7fe14353da4d9d67cde28699fc94059d7553d35c51955f"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "2da8566250e168659c280e0261f8e3b42a07618c260835946438f7abeee8156a"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "258a447dec49b2bb54d22af7f85458a51f786efe4f37c688ded9ca309614006c"
)
PARENT_PROOF_FILE_SHA256 = (
    "db859857d483c0d77d1e1900ac48ee3064d014214e088556824549a7ebda18bc"
)
PARENT_PASS1_EVIDENCE_IDENTITY_SHA256 = (
    "0be4cb9526411cb49bf85e1803a80ee6db49e9d91c3c9c077e6e563a1b8c15ac"
)
EXPECTED_FRANKO_SURVIVOR_OBJECTS = 30_636
EXPECTED_FRANKO_SURVIVOR_BYTES = 1_760_562
CURRENT_POST_QP_UA_BYTES = 10_632
UA_TARGET_BYTES = 9_000_000
UA_GAP_BYTES = UA_TARGET_BYTES - CURRENT_POST_QP_UA_BYTES

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class FrankoCleanExecutionError(RuntimeError):
    """Fail-closed Franko post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FrankoCleanExecutionError(message)


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
        raise FrankoCleanExecutionError(
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
    require(
        sha256(canonical(copied)) == identity,
        f"{label}: self-hash mismatch",
    )


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
    require(ancestor.returncode == 0, "#2763 execution base is not ancestor")

    expected_blobs = {
        "tools/run_d03_code_delta_decontam_qp_v1.py": CURRENT_QP_RUNNER_BLOB,
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
        "src/twelve_six/data/franko1901_dedup_intake.py": FRANKO_INTAKE_BLOB,
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

    carrier = "tools/run_d03_franko_decontam_qp_v1.py"
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
) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
    survivors = current_qp.load_json(
        survivors_path,
        "Franko parent survivors",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = current_qp.load_json(
        evidence_path,
        "Franko parent evidence",
        expected_sha256=PARENT_EVIDENCE_FILE_SHA256,
    )
    proof = current_qp.load_json(
        proof_path,
        "Franko parent two-clean proof",
        expected_sha256=PARENT_PROOF_FILE_SHA256,
    )
    _verify_self_hash(
        proof,
        "proof_identity_sha256",
        PARENT_PROOF_IDENTITY_SHA256,
        label="Franko parent two-clean proof",
    )
    require(
        proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Franko proof execution head drift",
    )
    require(
        proof.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Franko proof matcher root drift",
    )
    require(
        proof.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Franko proof survivor root drift",
    )
    require(
        proof.get("two_fresh_executions_converged") is True,
        "Franko parent lacks two-clean convergence",
    )
    require(
        proof.get("canonical_capacity_credited") == 0
        and proof.get("training_executed") is False
        and proof.get("tokenizer_fit_authorized") is False,
        "Franko proof truth boundary drift",
    )
    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="Franko parent survivors",
    )
    _verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        PARENT_PASS1_EVIDENCE_IDENTITY_SHA256,
        label="Franko parent evidence",
    )
    require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Franko parent execution head drift",
    )
    combined = evidence.get("combined")
    require(isinstance(combined, Mapping), "Franko parent combined evidence missing")
    require(
        combined.get("indexed_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Franko parent matcher report drift",
    )
    require(
        evidence.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Franko parent evidence/survivor authority drift",
    )
    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Franko parent survivor/matcher lineage drift",
    )

    franko_summary = survivors.get("franko1901")
    require(isinstance(franko_summary, Mapping), "Franko survivor summary missing")
    require(
        franko_summary.get("source_family") == franko.SOURCE_FAMILY,
        "Franko survivor family drift",
    )
    require(
        franko_summary.get("post_dedup_survivor_source_object_count")
        == EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "Franko survivor object count drift",
    )
    require(
        franko_summary.get("post_dedup_survivor_declared_capacity_bytes")
        == EXPECTED_FRANKO_SURVIVOR_BYTES,
        "Franko survivor byte count drift",
    )

    ids = survivors.get("survivor_source_ids")
    require(
        isinstance(ids, list)
        and len(ids) == len(set(ids))
        and all(type(value) is str and value for value in ids),
        "Franko parent survivor IDs invalid",
    )
    franko_ids = frozenset(
        value for value in ids if value.startswith("franko1901-admitted:")
    )
    require(
        len(franko_ids) == EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "Franko survivor ID cardinality drift",
    )

    truth = survivors.get("truth_boundary")
    require(isinstance(truth, Mapping), "Franko parent truth boundary missing")
    require(
        truth.get("global_dedup_execution_complete") is True,
        "Franko global dedup not terminal",
    )
    require(
        truth.get("reserved_evaluation_decontamination_complete") is False,
        "Franko parent unexpectedly claims DATA-232 complete",
    )
    require(
        truth.get("canonical_capacity_credited") == 0,
        "Franko parent fabricated canonical credit",
    )
    require(
        truth.get("model_training_executed") is False,
        "Franko parent executed training",
    )
    return survivors, evidence, franko_ids


def acquire_franko_survivors(
    *,
    candidate_jsonl: Path,
    historical_terminal_evidence_json: Path,
    fresh_execution_authority_json: Path,
    expected_survivor_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projection = franko.validate_and_project_franko(
        candidate_jsonl,
        historical_terminal_evidence_json,
        fresh_execution_authority_json,
        retain_payloads=True,
    )
    require(projection.sources is not None, "Franko source projection payloads absent")
    require(projection.payloads is not None, "Franko payload projection absent")

    by_id = {str(row["source_id"]): row for row in projection.sources}
    require(
        len(by_id) == franko.CANDIDATE_RECORDS,
        "Franko projected source cardinality drift",
    )
    require(
        set(projection.payloads) == set(by_id),
        "Franko projected source/payload coverage drift",
    )
    missing = sorted(expected_survivor_ids - set(by_id))
    require(not missing, "Franko survivor IDs absent from fresh projection")
    require(
        all(source_id.startswith("franko1901-admitted:") for source_id in expected_survivor_ids),
        "non-Franko ID reached Franko subset",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for source_id in sorted(expected_survivor_ids):
        row = by_id[source_id]
        raw = projection.payloads[source_id]
        require(
            row.get("source_family") == franko.SOURCE_FAMILY,
            f"{source_id}: source family drift",
        )
        require(row.get("modality") == "uk", f"{source_id}: modality drift")
        require(
            int(row.get("declared_capacity_bytes", -1)) == len(raw),
            f"{source_id}: declared/physical byte drift",
        )
        require(
            row.get("expected_raw_sha256") == sha256(raw),
            f"{source_id}: physical payload SHA drift",
        )
        rows.append(row)
        payloads[source_id] = raw
        total += len(raw)

    require(
        len(rows) == EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "fresh Franko survivor object count drift",
    )
    require(total == EXPECTED_FRANKO_SURVIVOR_BYTES, "fresh Franko survivor bytes drift")

    source_authority = {
        "candidate_sha256": franko.CANDIDATE_SHA256,
        "candidate_records": franko.CANDIDATE_RECORDS,
        "candidate_text_utf8_bytes": franko.CANDIDATE_TEXT_UTF8_BYTES,
        "intake_receipt_identity_sha256": projection.receipt[
            "receipt_identity_sha256"
        ],
        "post_global_dedup_survivor_ids_sha256": sha256(
            canonical(sorted(expected_survivor_ids))
        ),
        "post_global_dedup_survivor_objects": len(rows),
        "post_global_dedup_survivor_bytes": total,
    }
    return rows, payloads, source_authority


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
        family = str(row["source_family"])
        records.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": "uk",
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": "uk",
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )

    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(
        len(records) == EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "Franko training record count drift",
    )
    require(
        sum(row["payload_bytes"] for row in inventory_rows)
        == EXPECTED_FRANKO_SURVIVOR_BYTES,
        "Franko training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-franko1901-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "total_payload_bytes": EXPECTED_FRANKO_SURVIVOR_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-franko1901-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_FRANKO_SURVIVOR_OBJECTS,
        "declared_capacity_bytes": EXPECTED_FRANKO_SURVIVOR_BYTES,
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
        "retained_source_count": EXPECTED_FRANKO_SURVIVOR_OBJECTS,
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
    historical_terminal_evidence_json: Path,
    fresh_execution_authority_json: Path,
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
    _survivors, parent_evidence, survivor_ids = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_proof_json,
    )
    source_rows, payloads, source_authority = acquire_franko_survivors(
        candidate_jsonl=candidate_jsonl,
        historical_terminal_evidence_json=historical_terminal_evidence_json,
        fresh_execution_authority_json=fresh_execution_authority_json,
        expected_survivor_ids=survivor_ids,
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
        count=EXPECTED_FRANKO_SURVIVOR_OBJECTS,
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
    gate_loss_bytes = EXPECTED_FRANKO_SURVIVOR_BYTES - final_bytes
    require(gate_loss_bytes >= 0, "later gates widened Franko payload bytes")
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "final survivor count drift",
    )

    projected_ua_bytes = CURRENT_POST_QP_UA_BYTES + final_bytes
    projected_gap = UA_TARGET_BYTES - projected_ua_bytes
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
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "post_global_dedup_franko_objects": EXPECTED_FRANKO_SURVIVOR_OBJECTS,
            "post_global_dedup_franko_bytes": EXPECTED_FRANKO_SURVIVOR_BYTES,
        },
        "franko_authority": {
            "source_family": franko.SOURCE_FAMILY,
            "source_object_count": EXPECTED_FRANKO_SURVIVOR_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_FRANKO_SURVIVOR_BYTES,
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
        "capacity_projection": {
            "current_post_qp_ua_bytes_reference": CURRENT_POST_QP_UA_BYTES,
            "ua_target_bytes": UA_TARGET_BYTES,
            "pre_gate_ua_gap_bytes": UA_GAP_BYTES,
            "qualified_franko_survivor_bytes": final_bytes,
            "projected_post_qp_ua_bytes": projected_ua_bytes,
            "projected_ua_gap_after_decontam_g05_g06": max(projected_gap, 0),
            "ua_raw_target_reached_before_balance_split_pack": projected_gap <= 0,
        },
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_complete_for_franko": True,
            "reserved_evaluation_decontamination_complete_for_franko": True,
            "canonical_quality_privacy_complete_for_franko": True,
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
    parser.add_argument("--historical-terminal-evidence-json", type=Path, required=True)
    parser.add_argument("--fresh-execution-authority-json", type=Path, required=True)
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
            historical_terminal_evidence_json=args.historical_terminal_evidence_json,
            fresh_execution_authority_json=args.fresh_execution_authority_json,
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
        FrankoCleanExecutionError,
        current_qp.CodeDeltaCleanExecutionError,
        clean.CurrentCleanExecutionError,
        franko.FrankoDedupIntakeError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_FRANKO_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_FRANKO_POST_QP=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(
        "POST_QP_FRANKO_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_FRANKO_OBJECTS="
        f"{evidence['gate_execution']['survivor_records']}"
    )
    print(
        "PROJECTED_POST_QP_UA_BYTES="
        f"{evidence['capacity_projection']['projected_post_qp_ua_bytes']}"
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
