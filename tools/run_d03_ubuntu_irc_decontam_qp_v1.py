#!/usr/bin/env python3
"""Execute exact Ubuntu IRC global-dedup survivors through current DATA-232/G05/G06.

Execution-only carrier. It consumes the exact terminal Ubuntu IRC two-clean
survivor authority, freshly rematerializes the pinned Ubuntu IRC candidate,
retains only physically proven global-dedup survivors in ephemeral memory, and
delegates reserved-evaluation decontamination, G05 quality, and G06 privacy to
the already-qualified current-clean implementation. Durable output is text-free
and grants zero canonical/training/tokenizer authority.
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
for location in (str(ROOT / "tools"), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved
from twelve_six.data import ubuntu_irc_v9_intake as ubuntu

SCHEMA = "12-6.d03-ubuntu-irc-decontam-g05-g06-execution.v1"
PARENT_EXECUTION_HEAD = "45c1d93e8b3b6d1ea12d448583ce815f9705832e"
PARENT_WORKFLOW_RUN_ID = 37312839166
PARENT_ARTIFACT_ID = 11347350536
PARENT_ARTIFACT_ZIP_SHA256 = (
    "24750d7f3d7f179f6632da142eeb76283ea6e96a20ba77f5d293c39ece1cb33c"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "6cd93687c6b1063f62c19fc7be90001bc06a87cc320b1cabaa1b31a3a59bbdb2"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "581ee1d16abed2542978300475f93b41f6ff386d9fcb6ab6fa97d12dd2a15705"
)
PARENT_PROOF_FILE_SHA256 = (
    "27c7addb9897c1317eff260767ee67c6bd7761532e9ba2a9de5237c67efb7a18"
)
PARENT_MATCHER_REPORT_SHA256 = (
    "1ab1c3f6b6ff3efa0e468809c791a46512043d655dc3056da1313433db0cfa65"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "a4f531cbf2876676f852d8b4eae3816cba59e8b55234457e2ab8a04466fad7a7"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "09d58abe8a67f64e412401da7ebda2ec809e49213ace25b675e5924e8dfab54a"
)
PARENT_PASS1_EVIDENCE_IDENTITY_SHA256 = (
    "b30d6e970d7d7966b0a6ccf6ec08836e2380dac451b803489283162b60abfeba"
)
EXPECTED_UBUNTU_SURVIVOR_OBJECTS = 959
EXPECTED_UBUNTU_SURVIVOR_BYTES = 4_775_057
CURRENT_CLEAN_BLOB = "b82ed11626a267dfffa16acd79e45c0cf3e6750b"
UBUNTU_INTAKE_BLOB = "4b1bc9e2fb5c826a452670266e9230bc0b653093"
PARENT_DEDUP_RUNNER_BLOB = "1772feab6ed1c4c6de34f94feebd944cab9d63db"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class UbuntuCleanExecutionError(RuntimeError):
    """Fail-closed Ubuntu post-dedup/current-clean mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuCleanExecutionError(message)


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
    raise UbuntuCleanExecutionError(f"non-finite JSON constant rejected: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UbuntuCleanExecutionError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def strict_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UbuntuCleanExecutionError(f"{label}: strict JSON parse failed") from exc


def load_json(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
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
        timeout=30,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
        raise UbuntuCleanExecutionError(
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
        PARENT_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "terminal Ubuntu dedup head is not ancestor")

    expected_blobs = {
        "tools/run_d03_ubuntu_irc_global_dedup_execution_v1.py": (
            PARENT_DEDUP_RUNNER_BLOB
        ),
        "src/twelve_six/data/current_clean_execution_v1.py": CURRENT_CLEAN_BLOB,
        "src/twelve_six/data/ubuntu_irc_v9_intake.py": UBUNTU_INTAKE_BLOB,
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

    carrier = "tools/run_d03_ubuntu_irc_decontam_qp_v1.py"
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
    survivors = load_json(
        survivors_path,
        "Ubuntu parent survivors",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = load_json(
        evidence_path,
        "Ubuntu parent evidence",
        expected_sha256=PARENT_EVIDENCE_FILE_SHA256,
    )
    proof = load_json(
        proof_path,
        "Ubuntu parent two-clean proof",
        expected_sha256=PARENT_PROOF_FILE_SHA256,
    )

    _verify_self_hash(
        proof,
        "proof_identity_sha256",
        PARENT_PROOF_IDENTITY_SHA256,
        label="Ubuntu parent two-clean proof",
    )
    require(
        proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Ubuntu proof execution head drift",
    )
    require(
        proof.get("combined_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Ubuntu proof matcher root drift",
    )
    require(
        proof.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Ubuntu proof survivor root drift",
    )
    require(
        proof.get("two_fresh_executions_converged") is True,
        "Ubuntu parent lacks two-clean convergence",
    )
    require(
        proof.get("canonical_capacity_credited") == 0
        and proof.get("training_executed") is False
        and proof.get("tokenizer_fit_authorized") is False,
        "Ubuntu proof truth boundary drift",
    )

    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="Ubuntu parent survivors",
    )
    require(
        survivors.get("schema_version")
        == "12-6.d03-ubuntu-irc-global-dedup-survivors.v1",
        "Ubuntu survivor schema drift",
    )
    require(
        survivors.get("matcher_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Ubuntu survivor/matcher lineage drift",
    )
    require(
        survivors.get("ubuntu_source_family") == ubuntu.SOURCE_FAMILY,
        "Ubuntu survivor family drift",
    )
    require(
        survivors.get("pre_dedup_source_object_count") == ubuntu.CANDIDATE_RECORDS,
        "Ubuntu pre-dedup object count drift",
    )
    require(
        survivors.get("pre_dedup_declared_capacity_bytes")
        == ubuntu.CANDIDATE_NORMALIZED_BYTES,
        "Ubuntu pre-dedup byte count drift",
    )
    require(
        survivors.get("post_dedup_survivor_source_object_count")
        == EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "Ubuntu survivor object count drift",
    )
    require(
        survivors.get("post_dedup_survivor_declared_capacity_bytes")
        == EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "Ubuntu survivor byte count drift",
    )

    survivor_rows = survivors.get("survivors")
    require(
        type(survivor_rows) is list
        and len(survivor_rows) == EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "Ubuntu survivor rows missing",
    )
    required_row_keys = {
        "source_id",
        "source_family",
        "stable_origin_id",
        "stable_object_id",
        "declared_capacity_bytes",
        "expected_raw_sha256",
        "origin_key",
    }
    ids: list[str] = []
    total = 0
    for row in survivor_rows:
        require(
            type(row) is dict and set(row) == required_row_keys,
            "Ubuntu survivor row schema drift",
        )
        source_id = row.get("source_id")
        require(
            type(source_id) is str and bool(source_id),
            "Ubuntu survivor source ID invalid",
        )
        require(
            row.get("source_family") == ubuntu.SOURCE_FAMILY,
            "Ubuntu survivor family mismatch",
        )
        payload_bytes = row.get("declared_capacity_bytes")
        require(
            type(payload_bytes) is int and payload_bytes > 0,
            "Ubuntu survivor bytes invalid",
        )
        payload_sha = row.get("expected_raw_sha256")
        require(
            type(payload_sha) is str
            and _SHA64.fullmatch(payload_sha) is not None,
            "Ubuntu survivor SHA invalid",
        )
        ids.append(source_id)
        total += payload_bytes
    require(len(ids) == len(set(ids)), "Ubuntu survivor IDs are not unique")
    require(
        total == EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "Ubuntu survivor byte arithmetic drift",
    )

    survivor_truth = survivors.get("truth_boundary")
    require(
        isinstance(survivor_truth, Mapping),
        "Ubuntu survivor truth boundary missing",
    )
    require(
        survivor_truth.get("global_dedup_execution_complete") is True
        and survivor_truth.get("reserved_evaluation_decontamination_complete")
        is False
        and survivor_truth.get("canonical_quality_privacy_complete") is False,
        "Ubuntu parent gate truth drift",
    )
    require(
        survivor_truth.get("canonical_capacity_credited") == 0
        and survivor_truth.get("training_executed") is False,
        "Ubuntu parent survivor fabricated credit",
    )

    _verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        PARENT_PASS1_EVIDENCE_IDENTITY_SHA256,
        label="Ubuntu parent evidence",
    )
    require(
        evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD,
        "Ubuntu parent execution head drift",
    )
    combined = evidence.get("combined")
    require(
        isinstance(combined, Mapping),
        "Ubuntu parent combined evidence missing",
    )
    require(
        combined.get("report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "Ubuntu parent matcher report drift",
    )
    ubuntu_summary = evidence.get("ubuntu_irc")
    require(isinstance(ubuntu_summary, Mapping), "Ubuntu parent summary missing")
    require(
        ubuntu_summary.get("post_dedup_survivor_source_object_count")
        == EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "Ubuntu parent evidence survivor count drift",
    )
    require(
        ubuntu_summary.get("post_dedup_survivor_declared_capacity_bytes")
        == EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "Ubuntu parent evidence survivor bytes drift",
    )
    require(
        evidence.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "Ubuntu parent evidence/survivor authority drift",
    )
    truth = evidence.get("truth_boundary")
    require(isinstance(truth, Mapping), "Ubuntu parent evidence truth missing")
    require(
        truth.get("canonical_capacity_credited") == 0
        and truth.get("training_executed") is False
        and truth.get("tokenizer_fit_authorized") is False,
        "Ubuntu parent evidence truth drift",
    )
    return survivors, evidence, survivor_rows


def acquire_ubuntu_survivors(
    *,
    candidate_jsonl: Path,
    expected_survivors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    require(
        not candidate_jsonl.is_symlink() and candidate_jsonl.is_file(),
        "Ubuntu candidate must be a regular file",
    )
    candidate_bytes = candidate_jsonl.read_bytes()
    require(
        sha256(candidate_bytes) == ubuntu.CANDIDATE_SHA256,
        "Ubuntu candidate payload SHA-256 drift",
    )

    facade_bytes = (
        SRC / "twelve_six/data/expanded_global_dedup_v9.py"
    ).read_bytes()
    crossbind_bytes = (
        ROOT
        / "configs/data/d03_ubuntu_irc_repaired_execution_rights_crossbind_v1.json"
    ).read_bytes()
    rights_bytes = (
        ROOT / "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
    ).read_bytes()
    parent_registry_bytes = (
        ROOT / "configs/data/common_pile_source_rights_v1.json"
    ).read_bytes()
    execution_evidence_bytes = (
        ROOT / "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
    ).read_bytes()

    projected_rows, projected_payloads, intake_receipt = (
        ubuntu.prepare_ubuntu_v9_intake(
            incumbent_v9_product_head=ubuntu.INCUMBENT_V9_PRODUCT_HEAD,
            incumbent_v9_facade_bytes=facade_bytes,
            crossbind_bytes=crossbind_bytes,
            rights_authority_bytes=rights_bytes,
            parent_registry_bytes=parent_registry_bytes,
            execution_evidence_bytes=execution_evidence_bytes,
            candidate_bytes=candidate_bytes,
        )
    )
    by_id = {str(row["source_id"]): row for row in projected_rows}
    require(
        len(by_id) == ubuntu.CANDIDATE_RECORDS,
        "Ubuntu fresh projection source cardinality drift",
    )
    require(
        set(projected_payloads) == set(by_id),
        "Ubuntu fresh projection payload coverage drift",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for expected in sorted(
        expected_survivors,
        key=lambda row: str(row["source_id"]),
    ):
        source_id = str(expected["source_id"])
        fresh = by_id.get(source_id)
        require(
            type(fresh) is dict,
            f"Ubuntu survivor absent from fresh projection: {source_id}",
        )
        raw = projected_payloads.get(source_id)
        require(
            type(raw) is bytes,
            f"Ubuntu survivor payload absent: {source_id}",
        )
        for key in (
            "source_family",
            "stable_origin_id",
            "stable_object_id",
            "declared_capacity_bytes",
            "expected_raw_sha256",
            "origin_key",
        ):
            require(
                fresh.get(key) == expected.get(key),
                f"Ubuntu survivor authority drift for {source_id}: {key}",
            )
        require(
            fresh.get("source_family") == ubuntu.SOURCE_FAMILY,
            "Ubuntu source family drift",
        )
        require(fresh.get("modality") == ubuntu.MODALITY, "Ubuntu modality drift")
        require(
            type(fresh.get("declared_capacity_bytes")) is int
            and fresh["declared_capacity_bytes"] == len(raw),
            f"Ubuntu survivor byte drift: {source_id}",
        )
        require(
            fresh.get("expected_raw_sha256") == sha256(raw),
            f"Ubuntu survivor payload SHA drift: {source_id}",
        )
        rows.append(fresh)
        payloads[source_id] = raw
        total += len(raw)

    require(
        len(rows) == EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "fresh Ubuntu survivor object count drift",
    )
    require(
        total == EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "fresh Ubuntu survivor byte count drift",
    )
    source_authority = {
        "candidate_sha256": ubuntu.CANDIDATE_SHA256,
        "candidate_records": ubuntu.CANDIDATE_RECORDS,
        "candidate_normalized_utf8_bytes": ubuntu.CANDIDATE_NORMALIZED_BYTES,
        "intake_receipt_sha256": sha256(canonical(intake_receipt)),
        "post_global_dedup_survivor_ids_sha256": sha256(
            canonical(sorted(payloads))
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
                "modality": ubuntu.MODALITY,
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": ubuntu.MODALITY,
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )

    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(
        len(records) == EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "Ubuntu training record count drift",
    )
    require(
        sum(row["payload_bytes"] for row in inventory_rows)
        == EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "Ubuntu training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-ubuntu-irc-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "total_payload_bytes": EXPECTED_UBUNTU_SURVIVOR_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-ubuntu-irc-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
        "declared_capacity_bytes": EXPECTED_UBUNTU_SURVIVOR_BYTES,
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
        "retained_source_count": EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
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
    parent_proof_json: Path,
    candidate_jsonl: Path,
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
    _survivors, _evidence, survivor_rows = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_proof_json,
    )
    source_rows, payloads, source_authority = acquire_ubuntu_survivors(
        candidate_jsonl=candidate_jsonl,
        expected_survivors=survivor_rows,
    )
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
        count=EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
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
            expected_survivor_jsonl_sha256=receipt[
                "survivor_jsonl_sha256"
            ],
            expected_survivor_record_inventory_digest_sha256=receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            expected_survivor_payload_inventory_digest_sha256=receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
        )
    finally:
        _restore_clean_release(original)

    final_bytes = int(receipt["survivor_payload_bytes"])
    gate_loss_bytes = EXPECTED_UBUNTU_SURVIVOR_BYTES - final_bytes
    require(gate_loss_bytes >= 0, "later gates widened Ubuntu payload bytes")
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "Ubuntu survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "Ubuntu final survivor count drift",
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
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "post_global_dedup_ubuntu_objects": (
                EXPECTED_UBUNTU_SURVIVOR_OBJECTS
            ),
            "post_global_dedup_ubuntu_bytes": EXPECTED_UBUNTU_SURVIVOR_BYTES,
        },
        "ubuntu_authority": {
            "source_family": ubuntu.SOURCE_FAMILY,
            "source_object_count": EXPECTED_UBUNTU_SURVIVOR_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_UBUNTU_SURVIVOR_BYTES,
            "inventory_identity_sha256": inventory[
                "inventory_identity_sha256"
            ],
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
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_complete_for_ubuntu": True,
            "reserved_evaluation_decontamination_complete_for_ubuntu": True,
            "canonical_quality_privacy_complete_for_ubuntu": True,
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
            parent_proof_json=args.parent_proof_json,
            candidate_jsonl=args.candidate_jsonl,
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
        UbuntuCleanExecutionError,
        clean.CurrentCleanExecutionError,
        reserved.CurrentDecontaminationExecutionError,
        ubuntu.UbuntuIrcV9IntakeError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:320]
        print(f"D03_UBUNTU_IRC_POST_QP=BLOCKED: {detail}")
        return 2

    print("D03_UBUNTU_IRC_POST_QP=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(
        "POST_QP_UBUNTU_BYTES="
        f"{evidence['gate_execution']['survivor_payload_bytes']}"
    )
    print(
        "POST_QP_UBUNTU_OBJECTS="
        f"{evidence['gate_execution']['survivor_records']}"
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
