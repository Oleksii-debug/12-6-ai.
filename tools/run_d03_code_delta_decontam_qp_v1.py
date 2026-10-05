#!/usr/bin/env python3
"""Execute the exact post-dedup code delta through incumbent DATA-232, G05 and G06.

This is an execution-only carrier. It consumes the exact #2748 two-clean survivor
authority, reacquires only the 18 globally-unique delta source objects, and runs the
already-merged current reserved-evaluation decontamination, quality and privacy
semantics. Raw source/evaluation/survivor text remains ephemeral. Durable outputs
are text-free and grant zero canonical/training authority.
"""
from __future__ import annotations

import argparse
import hashlib
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

import run_d03_code4_flask_global_dedup_v1 as flask
import run_d03_code4_rich_fastapi_flask_reserve_global_dedup_v1 as parent
import run_d03_code4_rich_fastapi_reserve_global_dedup_v1 as reserve
import run_d03_four_code_sources_global_dedup_v1 as code4
from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved

SCHEMA = "12-6.d03-code-delta-decontam-g05-g06-execution.v1"
PARENT_EXECUTION_HEAD = "cc8dbf35f592f9c9a856e2b711278f4ee0205a5c"
PARENT_RUNNER_BLOB = "0a3e0df65078052ea29f20c25f002e2d7c3047a3"
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
PARENT_ARTIFACT_ID = 11334617491
PARENT_ARTIFACT_ZIP_SHA256 = (
    "81f9d7375f7ad18b884ede5b0b9a7768702a62474893d24fe294b9dbcba3eac1"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "aeddd3e5a1435043553fdb4384ad6c5e81a88381c5b0038493ed1cf8a944867d"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "a512ebde58912d694dcbf729f87aca8240319fffb7c4334c1189ded4ff06e633"
)
PARENT_COMBINED_REPORT_SHA256 = (
    "b354114771d22fc1be59a74007baea2956a9e5b8311596c05eb3f6c92f6ea418"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "1312186bb19a1e65603afd95fa29bb8f44488b28b86df7136b0a99b3ef8b0254"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "5f5fe9551692da2b8c037c20b1261bdbb3d707dbc7bfb13776429ab4e792b1bb"
)
FLASK_CTX_SOURCE_ID = "code.flask.3_1_3.03.src_flask_ctx_py"
EXPECTED_FLASK_CTX_BYTES = 15_521
EXPECTED_DELTA_OBJECTS = 18
EXPECTED_DELTA_BYTES = (
    code4.EXPECTED_EXTENSION_BYTES
    + reserve.RICH_EXPECTED_BYTES
    + reserve.FASTAPI_EXPECTED_BYTES
    + EXPECTED_FLASK_CTX_BYTES
)
CURRENT_POST_QP_CODE_BYTES = 3_664_247
CODE_TARGET_BYTES = 4_000_000
CODE_GAP_BYTES = CODE_TARGET_BYTES - CURRENT_POST_QP_CODE_BYTES

EXPECTED_DELTA_IDS = frozenset(
    {
        "code.fastapi.fastapi.datastructures",
        "code.fastapi.fastapi.exceptions",
        "code.fastapi.fastapi.sse",
        "code.fastapi.typer.utils",
        FLASK_CTX_SOURCE_ID,
        "code.pandas-dev.pandas.core.accessor",
        "code.pydantic.pydantic.fields",
        "code.pydantic.pydantic.functional_validators",
        "code.pydantic.pydantic.main",
        "code.pydantic.pydantic.type_adapter",
        "code.scipy.project.0",
        "code.scipy.project.1",
        "code.textualize.rich.align",
        "code.textualize.rich.ansi",
        "code.textualize.rich.columns",
        "code.textualize.rich.measure",
        "code.textualize.rich.padding",
        "code.textualize.rich.panel",
    }
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class CodeDeltaCleanExecutionError(RuntimeError):
    """Fail-closed exact-delta or clean-gate execution mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodeDeltaCleanExecutionError(message)


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


def _reject_constant(value: str) -> None:
    raise CodeDeltaCleanExecutionError(f"non-finite JSON constant rejected: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CodeDeltaCleanExecutionError(f"duplicate JSON member: {key}")
        out[key] = value
    return out


def strict_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeDeltaCleanExecutionError(f"{label}: strict JSON parse failed") from exc
    return value


def load_json(path: Path, label: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 is not None:
        require(sha256(raw) == expected_sha256, f"{label}: raw SHA-256 drift")
    value = strict_loads(raw, label)
    require(type(value) is dict, f"{label}: root must be exact object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            require(bool(raw.strip()), f"{label}: blank line {line_number}")
            value = strict_loads(raw, f"{label}:line {line_number}")
            require(type(value) is dict, f"{label}:line {line_number} must be object")
            rows.append(value)
    require(bool(rows), f"{label}: no rows")
    return rows


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and result.returncode != 0:
        raise CodeDeltaCleanExecutionError(
            "git failed: " + " ".join(args) + ": " + result.stderr.strip()
        )
    return result


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
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
    require(ancestor.returncode == 0, "terminal #2748 head is not execution ancestor")
    for path, expected_blob in (
        (
            "tools/run_d03_code4_rich_fastapi_flask_reserve_global_dedup_v1.py",
            PARENT_RUNNER_BLOB,
        ),
        ("src/twelve_six/data/current_clean_execution_v1.py", CURRENT_CLEAN_BLOB),
    ):
        require(
            git("rev-parse", f"HEAD:{path}").stdout.strip() == expected_blob,
            f"Git blob drift: {path}",
        )
        require(
            git("hash-object", str(ROOT / path)).stdout.strip() == expected_blob,
            f"worktree blob drift: {path}",
        )
    upstream = parent.verify_local_authority(expected_execution_head)
    dependency_blobs = clean.verify_dependency_blobs()
    return {
        **upstream,
        **{f"current_clean:{key}": value for key, value in dependency_blobs.items()},
        "tools/run_d03_code4_rich_fastapi_flask_reserve_global_dedup_v1.py": (
            PARENT_RUNNER_BLOB
        ),
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
    }


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    survivors = load_json(
        survivors_path,
        "parent survivors",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = load_json(
        evidence_path,
        "parent evidence",
        expected_sha256=PARENT_EVIDENCE_FILE_SHA256,
    )
    require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent execution head drift",
    )
    combined = evidence.get("combined")
    require(isinstance(combined, Mapping), "parent combined evidence missing")
    require(
        combined.get("matcher_report_sha256") == PARENT_COMBINED_REPORT_SHA256,
        "parent matcher report drift",
    )
    require(
        combined.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent evidence survivor authority drift",
    )
    require(
        survivors.get("matcher_report_sha256") == PARENT_COMBINED_REPORT_SHA256,
        "parent survivor/matcher lineage drift",
    )
    require(
        survivors.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent survivor authority drift",
    )
    survivor_ids = survivors.get("survivor_source_ids")
    require(
        isinstance(survivor_ids, list)
        and len(survivor_ids) == len(set(survivor_ids)),
        "parent survivor ids invalid",
    )
    missing = sorted(EXPECTED_DELTA_IDS - set(survivor_ids))
    require(not missing, "expected delta IDs are not parent survivors: " + ", ".join(missing))
    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "parent truth boundary missing")
    require(truth.get("canonical_capacity_credited") == 0, "parent fabricated credit")
    require(truth.get("training_executed") is False, "parent executed training")
    return survivors, evidence


def acquire_delta_sources() -> tuple[
    list[dict[str, Any]],
    dict[str, bytes],
    dict[str, Any],
]:
    physical_summary, captured = code4.capture_physical_sources()
    code4_rows, code4_payloads, _ = code4.build_extension_graph(captured)

    rich, fast = reserve.validate_reserve_contracts()
    reserve_rows, reserve_payloads, _ = reserve.build_reserve_sources(rich, fast)

    flask_rows, flask_payloads, _flask_edges, flask_authority = (
        flask.capture_flask_sources()
    )
    flask_by_id = {row["source_id"]: row for row in flask_rows}
    require(FLASK_CTX_SOURCE_ID in flask_by_id, "Flask ctx source missing")
    require(FLASK_CTX_SOURCE_ID in flask_payloads, "Flask ctx payload missing")
    require(
        int(flask_by_id[FLASK_CTX_SOURCE_ID]["declared_capacity_bytes"])
        == EXPECTED_FLASK_CTX_BYTES,
        "Flask ctx declared byte count drift",
    )
    require(
        len(flask_payloads[FLASK_CTX_SOURCE_ID]) == EXPECTED_FLASK_CTX_BYTES,
        "Flask ctx physical byte count drift",
    )

    rows = [*code4_rows, *reserve_rows, flask_by_id[FLASK_CTX_SOURCE_ID]]
    payloads = {
        **code4_payloads,
        **reserve_payloads,
        FLASK_CTX_SOURCE_ID: flask_payloads[FLASK_CTX_SOURCE_ID],
    }
    by_id = {row["source_id"]: row for row in rows}
    require(set(by_id) == EXPECTED_DELTA_IDS, "delta source-id set drift")
    require(set(payloads) == EXPECTED_DELTA_IDS, "delta payload set drift")
    require(len(rows) == EXPECTED_DELTA_OBJECTS, "delta object count drift")
    require(
        sum(int(row["declared_capacity_bytes"]) for row in rows)
        == EXPECTED_DELTA_BYTES,
        "delta declared byte count drift",
    )
    require(
        sum(len(payloads[source_id]) for source_id in sorted(payloads))
        == EXPECTED_DELTA_BYTES,
        "delta physical byte count drift",
    )
    source_authority = {
        "code4_observed_summary_sha256": sha256(
            code4.physical.canonical(physical_summary)
        ),
        "flask_physical_report_sha256": flask_authority["report_sha256"],
        "flask_snapshot_manifest_sha256": flask_authority[
            "snapshot_manifest_sha256"
        ],
    }
    return [by_id[source_id] for source_id in sorted(by_id)], payloads, source_authority


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
                "modality": "code",
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": "code",
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )

    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(len(records) == EXPECTED_DELTA_OBJECTS, "training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows) == EXPECTED_DELTA_BYTES,
        "training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-code-delta-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_DELTA_OBJECTS,
        "total_payload_bytes": EXPECTED_DELTA_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-code-delta-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_COMBINED_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_DELTA_OBJECTS,
        "declared_capacity_bytes": EXPECTED_DELTA_BYTES,
        "source_ids": sorted(EXPECTED_DELTA_IDS),
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
        "retained_source_count": EXPECTED_DELTA_OBJECTS,
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


def _patch_clean_release(
    *,
    count: int,
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
    clean.PRODUCTION_INPUT_RECORD_COUNT = count
    clean.PRODUCTION_TRAINING_RECORDS_SHA256 = training_records_sha
    clean.PRODUCTION_TRAINING_HANDOFF_SHA256 = handoff_raw_sha
    clean.PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256 = (
        inventory_identity
    )
    clean.PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = subset_authority
    return original


def _restore_clean_release(original: Mapping[str, Any]) -> None:
    for key, value in original.items():
        setattr(clean, key, value)


def execute(
    *,
    expected_execution_head: str,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
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
    _parent_survivors, parent_evidence = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
    )

    source_rows, payloads, source_authority = acquire_delta_sources()
    (
        training_records,
        inventory,
        subset,
        handoff,
        training_records_sha,
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

    original = _patch_clean_release(
        count=EXPECTED_DELTA_OBJECTS,
        training_records_sha=training_records_sha,
        handoff_raw_sha=handoff_raw_sha,
        inventory_identity=inventory["inventory_identity_sha256"],
        subset_authority=subset["subset_authority_sha256"],
    )
    try:
        (
            receipt,
            data232,
            decontam,
            eval647_receipt,
            quality,
            privacy,
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
        _restore_clean_release(original)

    require(
        receipt.get("input_training_records") == EXPECTED_DELTA_OBJECTS,
        "later-gate input record count drift",
    )
    final_bytes = int(receipt["survivor_payload_bytes"])
    projected_code_bytes = CURRENT_POST_QP_CODE_BYTES + final_bytes
    projected_headroom = projected_code_bytes - CODE_TARGET_BYTES
    input_bytes = EXPECTED_DELTA_BYTES
    gate_loss_bytes = input_bytes - final_bytes
    require(gate_loss_bytes >= 0, "later gate byte accounting widened source payload")
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
            "workflow_run_id": 37286710344,
            "artifact_id": PARENT_ARTIFACT_ID,
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "combined_matcher_report_sha256": PARENT_COMBINED_REPORT_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "marginal_global_unique_reserve_bytes": parent_evidence[
                "reserve_delta"
            ]["marginal_global_unique_capacity_bytes"],
            "projected_late_gate_headroom_before_execution": parent_evidence[
                "reserve_delta"
            ]["projected_late_gate_headroom_after_global_dedup"],
        },
        "delta_authority": {
            "source_object_count": EXPECTED_DELTA_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_DELTA_BYTES,
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "subset_authority_sha256": subset["subset_authority_sha256"],
            "training_records_sha256": training_records_sha,
            "training_handoff_raw_sha256": handoff_raw_sha,
            "training_handoff_identity_sha256": handoff[
                "handoff_identity_sha256"
            ],
            "source_ids_sha256": sha256(canonical(sorted(EXPECTED_DELTA_IDS))),
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
            "current_post_qp_code_bytes_reference": CURRENT_POST_QP_CODE_BYTES,
            "code_target_bytes": CODE_TARGET_BYTES,
            "pre_gate_gap_bytes": CODE_GAP_BYTES,
            "qualified_delta_survivor_bytes": final_bytes,
            "projected_post_qp_code_bytes": projected_code_bytes,
            "projected_headroom_after_decontam_g05_g06": projected_headroom,
            "still_possible_to_reach_code_target_after_balance_split_pack": (
                projected_headroom >= 0
            ),
        },
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_complete_for_delta": True,
            "reserved_evaluation_decontamination_complete_for_delta": True,
            "canonical_quality_privacy_complete_for_delta": True,
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
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument("--eval647-manifest-json", type=Path, required=True)
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
        CodeDeltaCleanExecutionError,
        clean.CurrentCleanExecutionError,
        reserved.CurrentDecontaminationExecutionError,
        parent.CombinedReserveGlobalDedupError,
        reserve.ReserveGlobalDedupError,
        flask.Code4FlaskGlobalDedupError,
        code4.FourCodeGlobalDedupError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(
            json.dumps(
                {
                    "status": "BLOCKED_CODE_DELTA_DECONTAM_G05_G06",
                    "error_type": type(exc).__name__,
                    "error_detail": detail,
                    "canonical_capacity_credited": 0,
                    "authorized_optimized_target_exposure": 0,
                    "training_executed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    cap = evidence["capacity_projection"]
    gate = evidence["gate_execution"]
    print("D03_CODE_DELTA_DECONTAM_G05_G06=PASS_ZERO_CREDIT")
    print("DELTA_SURVIVOR_PAYLOAD_BYTES=" + str(gate["survivor_payload_bytes"]))
    print(
        "PROJECTED_POST_QP_CODE_BYTES="
        + str(cap["projected_post_qp_code_bytes"])
    )
    print(
        "PROJECTED_HEADROOM_AFTER_DECONTAM_G05_G06="
        + str(cap["projected_headroom_after_decontam_g05_g06"])
    )
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
