#!/usr/bin/env python3
"""Execute exact current-40 NBU global-dedup survivors through DATA-232/G05/G06.

Execution-only carrier. It authenticates the terminal #2809 incumbent-global-dedup
evidence, freshly rematerializes the exact audited NBU current-40 candidate, retains
only exact parent survivors in ephemeral memory, and delegates DATA-232, G05 quality,
and G06 privacy to the incumbent current-clean implementation. Durable output is
text-free and grants zero corpus/tokenizer/training/scale authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved
from twelve_six.data import nbu_current40_dedup_intake as nbu

SCHEMA = "12-6.d03-nbu-current40-decontam-g05-g06-execution.v1"
PARENT_EXECUTION_HEAD = "b5235cfd83852854e345dea6c2e89cfbcd1cef79"
PARENT_WORKFLOW_RUN_ID = 37382526670
PARENT_ARTIFACT_ID = 11375922011
PARENT_ARTIFACT_ZIP_SHA256 = (
    "8a5b5624d75caa57734d3efa86fb0a89f537a7e978bdd6edc4118be1d27ce744"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "e95f2af8c8ee2c1d8264b2e9cffa3f7147101640b74a8842e4968b49854934ab"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "8e74e1f1229d06c09a6838e0a21fcad87a562079e6ef147067fdf6860c604ba7"
)
PARENT_TWO_CLEAN_FILE_SHA256 = (
    "8cef03ad210da947fe1cbab8a66b4c2e58b3b99cb75857d3fef5aa89af862549"
)
PARENT_SUMMARY_FILE_SHA256 = (
    "bf9e9df0651355e1c507b215f6f3f8cba824bce15e998d993be4aae7e4b008de"
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
PARENT_SUMMARY_IDENTITY_SHA256 = (
    "c8ac2c067a1190335fd6f92b9e6360ead4d344bc37982e6cd583e3d5644b0c79"
)
EXPECTED_NBU_OBJECTS = 40
EXPECTED_NBU_BYTES = 891_494
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
CURRENT_RESERVED_BLOB = "e5c555e3cd27844e98d4ae91af0b746e427f36c9"
NBU_INTAKE_BLOB = "43e54663d5666e4fd26b1f56a792f03aa1ef08ec"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class NbuCleanExecutionError(RuntimeError):
    """Fail-closed current-NBU post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NbuCleanExecutionError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        raise NbuCleanExecutionError("git failed: " + " ".join(args) + ": " + detail)
    return result


def strict_json(path: Path, label: str, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256(raw) == expected_sha256, f"{label}: raw SHA-256 drift")
    seen_duplicate = False

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal seen_duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                seen_duplicate = True
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                NbuCleanExecutionError(f"{label}: non-finite constant {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NbuCleanExecutionError(f"{label}: invalid strict JSON") from exc
    require(not seen_duplicate, f"{label}: duplicate JSON member")
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def self_hash(value: Mapping[str, Any], field: str, expected: str, label: str) -> None:
    require(type(value) is dict, f"{label}: root must be exact object")
    body = copy.deepcopy(dict(value))
    claimed = body.pop(field, None)
    require(
        type(claimed) is str and _SHA64.fullmatch(claimed) is not None,
        f"{label}: identity malformed",
    )
    require(claimed == expected, f"{label}: identity drift")
    require(sha256(canonical(body)) == claimed, f"{label}: self-hash mismatch")


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        type(expected_execution_head) is str
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be exact lowercase 40-hex SHA",
    )
    require(git("rev-parse", "HEAD").stdout.strip() == expected_execution_head, "HEAD drift")
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        PARENT_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "#2809 global-dedup parent is not ancestor")

    expected = {
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
        "src/twelve_six/data/current_reserved_decontamination_v1.py": CURRENT_RESERVED_BLOB,
        "src/twelve_six/data/nbu_current40_dedup_intake.py": NBU_INTAKE_BLOB,
    }
    for path, blob in expected.items():
        require(
            git("rev-parse", f"HEAD:{path}").stdout.strip() == blob,
            f"Git blob drift: {path}",
        )
        require(
            git("hash-object", str(ROOT / path)).stdout.strip() == blob,
            f"worktree blob drift: {path}",
        )
    carrier = "tools/run_d03_nbu_current40_decontam_qp_v1.py"
    carrier_blob = git("rev-parse", f"HEAD:{carrier}").stdout.strip()
    require(
        git("hash-object", str(ROOT / carrier)).stdout.strip() == carrier_blob,
        "executing carrier worktree drift",
    )
    return {
        **expected,
        **{
            f"current_clean:{key}": value
            for key, value in clean.verify_dependency_blobs().items()
        },
        carrier: carrier_blob,
    }


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
    two_clean_path: Path,
    summary_path: Path,
) -> set[str]:
    survivors = strict_json(
        survivors_path,
        "NBU parent survivor authority",
        PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = strict_json(
        evidence_path,
        "NBU parent execution evidence",
        PARENT_EVIDENCE_FILE_SHA256,
    )
    two_clean = strict_json(
        two_clean_path,
        "NBU parent two-clean authority",
        PARENT_TWO_CLEAN_FILE_SHA256,
    )
    summary = strict_json(
        summary_path,
        "NBU parent terminal summary",
        PARENT_SUMMARY_FILE_SHA256,
    )
    self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        "NBU parent survivor authority",
    )
    self_hash(
        two_clean,
        "two_clean_authority_sha256",
        PARENT_TWO_CLEAN_AUTHORITY_SHA256,
        "NBU parent two-clean authority",
    )
    self_hash(
        summary,
        "summary_identity_sha256",
        PARENT_SUMMARY_IDENTITY_SHA256,
        "NBU parent summary",
    )

    require(
        survivors.get("schema_version")
        == "12-6.d03-nbu-current40-global-dedup-survivors.v1",
        "parent survivor schema drift",
    )
    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent matcher report drift",
    )
    require(
        survivors.get("nbu_survivor_source_object_count") == EXPECTED_NBU_OBJECTS,
        "parent NBU survivor count drift",
    )
    require(
        survivors.get("nbu_survivor_declared_capacity_bytes") == EXPECTED_NBU_BYTES,
        "parent NBU survivor bytes drift",
    )
    ids = survivors.get("nbu_survivor_source_ids")
    require(
        type(ids) is list
        and len(ids) == EXPECTED_NBU_OBJECTS
        and len(set(ids)) == EXPECTED_NBU_OBJECTS
        and all(type(value) is str and value.startswith("nbu-admitted:") for value in ids),
        "parent NBU survivor ID set drift",
    )

    require(
        evidence.get("schema_version")
        == "12-6.d03-nbu-current40-global-dedup-execution.v1",
        "parent evidence schema drift",
    )
    require(evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "parent head drift")
    require(
        evidence.get("survivor_authority_sha256") == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent evidence/survivor root drift",
    )
    combined = evidence.get("combined")
    require(isinstance(combined, Mapping), "parent combined evidence missing")
    require(
        combined.get("indexed_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent indexed matcher root drift",
    )
    nbu_evidence = evidence.get("nbu")
    require(isinstance(nbu_evidence, Mapping), "parent NBU evidence missing")
    require(
        nbu_evidence.get("source_object_count") == EXPECTED_NBU_OBJECTS
        and nbu_evidence.get("declared_capacity_bytes") == EXPECTED_NBU_BYTES,
        "parent NBU physical arithmetic drift",
    )

    dedup = two_clean.get("dedup")
    require(isinstance(dedup, Mapping), "parent two-clean dedup block missing")
    require(two_clean.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "two-clean head drift")
    require(dedup.get("fresh_process_count") == 2, "parent lacks two fresh dedup executions")
    require(
        dedup.get("report_sha256") == PARENT_MATCHER_REPORT_SHA256
        and dedup.get("survivor_authority_sha256") == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent two-clean roots drift",
    )
    require(
        dedup.get("nbu_survivor_source_object_count") == EXPECTED_NBU_OBJECTS
        and dedup.get("nbu_survivor_declared_capacity_bytes") == EXPECTED_NBU_BYTES,
        "parent two-clean NBU arithmetic drift",
    )

    require(summary.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "summary head drift")
    summary_dedup = summary.get("dedup")
    require(isinstance(summary_dedup, Mapping), "summary dedup block missing")
    require(
        summary_dedup.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "summary survivor authority drift",
    )
    require(
        summary_dedup.get("two_clean_authority_sha256")
        == PARENT_TWO_CLEAN_AUTHORITY_SHA256,
        "summary two-clean authority drift",
    )
    for truth in (
        evidence.get("truth_boundary"),
        two_clean.get("truth_boundary"),
        summary.get("truth_boundary"),
    ):
        require(isinstance(truth, Mapping), "parent truth boundary missing")
        require(
            truth.get("canonical_capacity_credited") == 0
            and truth.get("training_authorized_bytes") == 0
            and truth.get("authorized_unique_loss_positions") == 0
            and truth.get("authorized_optimized_target_exposure") == 0
            and truth.get("tokenizer_fit_authorized") is False
            and truth.get("training_executed") is False
            and truth.get("learned_weights_created") is False
            and truth.get("final_test_outcomes_read") is False
            and truth.get("paid_compute_used") is False,
            "parent truth boundary widened",
        )
    return set(ids)


def acquire_nbu_survivors(
    *,
    candidate_jsonl: Path,
    materialization_evidence_json: Path,
    parent_survivor_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projection = nbu.validate_and_project_nbu(
        candidate_jsonl,
        materialization_evidence_json,
        retain_payloads=True,
    )
    require(
        projection.sources is not None and projection.payloads is not None,
        "NBU projection payloads missing",
    )
    rows_by_id = {str(row["source_id"]): row for row in projection.sources}
    require(
        set(rows_by_id) == parent_survivor_ids,
        "fresh NBU projection/parent survivor set drift",
    )
    require(
        set(projection.payloads) == parent_survivor_ids,
        "fresh NBU payload/parent survivor set drift",
    )
    rows = [rows_by_id[source_id] for source_id in sorted(parent_survivor_ids)]
    payloads = {
        source_id: projection.payloads[source_id]
        for source_id in sorted(parent_survivor_ids)
    }
    require(len(rows) == EXPECTED_NBU_OBJECTS, "fresh NBU survivor count drift")
    require(
        sum(len(raw) for raw in payloads.values()) == EXPECTED_NBU_BYTES,
        "fresh NBU survivor byte drift",
    )
    source_authority = {
        "candidate_sha256": nbu.CANDIDATE_SHA256,
        "materialization_evidence_identity_sha256": nbu.EVIDENCE_IDENTITY_SHA256,
        "intake_receipt_identity_sha256": projection.receipt[
            "receipt_identity_sha256"
        ],
        "post_global_dedup_survivor_ids_sha256": sha256(
            canonical(sorted(parent_survivor_ids))
        ),
        "post_global_dedup_survivor_objects": len(rows),
        "post_global_dedup_survivor_bytes": sum(
            len(raw) for raw in payloads.values()
        ),
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
                "modality": nbu.MATCHER_MODALITY,
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": nbu.MATCHER_MODALITY,
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )
    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(len(records) == EXPECTED_NBU_OBJECTS, "NBU training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows) == EXPECTED_NBU_BYTES,
        "NBU training byte drift",
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
        "parent_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent_two_clean_authority_sha256": PARENT_TWO_CLEAN_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_NBU_OBJECTS,
        "declared_capacity_bytes": EXPECTED_NBU_BYTES,
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
    handoff_raw_sha = clean._sha256(clean._cjson(handoff))
    return (
        records,
        inventory,
        subset,
        handoff,
        training_records_sha,
        handoff_raw_sha,
    )


def patch_clean_release(
    *,
    training_records_sha: str,
    handoff_raw_sha: str,
    inventory_identity: str,
    subset_authority: str,
) -> dict[str, Any]:
    names = (
        "PRODUCTION_INPUT_RECORD_COUNT",
        "PRODUCTION_TRAINING_RECORDS_SHA256",
        "PRODUCTION_TRAINING_HANDOFF_SHA256",
        "PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256",
        "PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256",
    )
    original = {name: getattr(clean, name) for name in names}
    clean.PRODUCTION_INPUT_RECORD_COUNT = EXPECTED_NBU_OBJECTS
    clean.PRODUCTION_TRAINING_RECORDS_SHA256 = training_records_sha
    clean.PRODUCTION_TRAINING_HANDOFF_SHA256 = handoff_raw_sha
    clean.PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256 = (
        inventory_identity
    )
    clean.PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = subset_authority
    return original


def restore_clean_release(original: Mapping[str, Any]) -> None:
    for key, value in original.items():
        setattr(clean, key, value)


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        require(bool(raw.strip()), f"{label}: blank line {number}")
        try:
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NbuCleanExecutionError(
                f"{label}: invalid row {number}"
            ) from exc
        require(type(value) is dict, f"{label}: row {number} must be exact object")
        rows.append(value)
    require(bool(rows), f"{label}: no rows")
    return rows


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NbuCleanExecutionError(f"{label}: invalid JSON") from exc
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def execute(
    *,
    expected_execution_head: str,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
    parent_two_clean_json: Path,
    parent_summary_json: Path,
    candidate_jsonl: Path,
    materialization_evidence_json: Path,
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
    parent_survivor_ids = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_two_clean_json,
        parent_summary_json,
    )
    source_rows, payloads, source_authority = acquire_nbu_survivors(
        candidate_jsonl=candidate_jsonl,
        materialization_evidence_json=materialization_evidence_json,
        parent_survivor_ids=parent_survivor_ids,
    )
    (
        records,
        inventory,
        subset,
        handoff,
        records_sha,
        handoff_raw_sha,
    ) = build_training_authorities(source_rows, payloads)
    evaluation_records = load_jsonl(
        evaluation_records_jsonl,
        "reserved evaluation records",
    )
    base_reserved_binding = load_json(
        base_reserved_binding_json,
        "base reserved binding",
    )
    eval647_manifest = load_json(eval647_manifest_json, "EVAL-647 manifest")
    eval647_evidence = load_json(
        eval647_materialization_evidence_json,
        "EVAL-647 materialization evidence",
    )

    original = patch_clean_release(
        training_records_sha=records_sha,
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
            records,
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
            expected_final_test_identity_sha256=(
                expected_final_test_identity_sha256
            ),
        )
        clean.verify_current_clean_composition_receipt(
            receipt,
            expected_receipt_identity_sha256=receipt[
                "receipt_identity_sha256"
            ],
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
        restore_clean_release(original)

    final_bytes = int(receipt["survivor_payload_bytes"])
    require(0 <= final_bytes <= EXPECTED_NBU_BYTES, "NBU post-QP byte widening")
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "NBU survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "NBU survivor count drift",
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
            "source_objects": EXPECTED_NBU_OBJECTS,
            "payload_bytes": EXPECTED_NBU_BYTES,
        },
        "nbu_authority": {
            "source_family": nbu.SOURCE_FAMILY,
            "source_object_count": EXPECTED_NBU_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_NBU_BYTES,
            "inventory_identity_sha256": inventory[
                "inventory_identity_sha256"
            ],
            "subset_authority_sha256": subset["subset_authority_sha256"],
            "training_records_sha256": records_sha,
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
            "later_gate_loss_bytes": EXPECTED_NBU_BYTES - final_bytes,
        },
        "survivor_inventory": survivor_inventory,
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
    parser.add_argument("--parent-two-clean-json", type=Path, required=True)
    parser.add_argument("--parent-summary-json", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument(
        "--eval647-manifest-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--eval647-materialization-evidence-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-base-reserved-binding-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-composed-reserved-binding-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-eval647-materialization-evidence-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-eval647-object-set-identity-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-selection-validation-identity-sha256",
        required=True,
    )
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = execute(
            expected_execution_head=args.expected_execution_head,
            parent_survivors_json=args.parent_survivors_json,
            parent_evidence_json=args.parent_evidence_json,
            parent_two_clean_json=args.parent_two_clean_json,
            parent_summary_json=args.parent_summary_json,
            candidate_jsonl=args.candidate_jsonl,
            materialization_evidence_json=args.materialization_evidence_json,
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
            expected_final_test_identity_sha256=(
                args.expected_final_test_identity_sha256
            ),
            output_evidence=args.output_evidence,
        )
    except (
        NbuCleanExecutionError,
        nbu.NbuDedupIntakeError,
        clean.CurrentCleanExecutionError,
        reserved.CurrentDecontaminationExecutionError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_NBU_CURRENT40_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_NBU_CURRENT40_POST_QP=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(
        "POST_QP_NBU_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_NBU_OBJECTS="
        f"{evidence['gate_execution']['survivor_records']}"
    )
    print("EXACT_INCREMENTAL_UNIQUE_CAPACITY_CREDITED_BYTES=0")
    print("CROSS_LINEAGE_REDEDUP_REQUIRED=true")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
