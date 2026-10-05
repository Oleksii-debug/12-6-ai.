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


UBUNTU_CLEAN_PARTIAL_SCHEMA = (
    "12-6.d03-ubuntu-irc-current-clean-partial-materialization.v1"
)


def _materialize_quality_survivors_with_partial(
    inputs: list[dict[str, str]],
    metadata: Mapping[str, Mapping[str, str]],
    quality: Mapping[str, Any],
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
    dict[str, Any],
]:
    """Materialize authoritative G05 accepted units, including partial windows.

    This is the exact seam proven by the terminal Rada quality-window lineage:
    a RETAIN_PARTIAL natural-language document emits each accepted authoritative
    quality window independently; rejected siblings are not allowed to evict
    accepted siblings. The current G05 authority remains the sole source of
    spans, decisions, hashes, and byte counts.
    """
    raw_rows = quality.get("records")
    require(isinstance(raw_rows, list), "G05 records missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        require(isinstance(row, Mapping), "G05 record is not an object")
        record_id = row.get("record_id")
        require(
            isinstance(record_id, str) and bool(record_id),
            "G05 record_id missing",
        )
        require(record_id not in by_id, "duplicate G05 record_id")
        by_id[record_id] = row
    require(
        set(by_id) == {row["id"] for row in inputs},
        "G05/input record-set drift",
    )

    output: list[dict[str, str]] = []
    seen_output_ids: set[str] = set()
    partial_projection: list[dict[str, Any]] = []
    stats = {
        "g05_reject_documents": 0,
        "g05_partial_documents": 0,
        "g05_rejected_units": 0,
        "g05_rejected_utf8_bytes": 0,
    }
    detail = {
        "input_documents": len(inputs),
        "retain_all_documents": 0,
        "partial_documents": 0,
        "reject_documents": 0,
        "partial_emitted_units": 0,
        "partial_rejected_units": 0,
        "input_utf8_bytes": 0,
        "retained_utf8_bytes": 0,
        "rejected_utf8_bytes": 0,
    }

    for source in inputs:
        record_id = source["id"]
        text = source["text"]
        mode = source["mode"]
        row = by_id[record_id]
        payload = text.encode("utf-8")
        require(
            row.get("payload_sha256") == sha256(payload)
            and row.get("utf8_bytes") == len(payload)
            and row.get("mode") == mode,
            f"G05 payload binding drift: {record_id}",
        )
        status = row.get("status")
        units = row.get("units")
        authoritative_unit = row.get("authoritative_unit")
        require(
            status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"},
            f"G05 status drift: {record_id}",
        )
        require(
            authoritative_unit
            in {"DOCUMENT", "BOUNDED_NATURAL_LANGUAGE_WINDOW"},
            f"G05 authoritative unit drift: {record_id}",
        )
        require(
            isinstance(units, list) and bool(units),
            f"G05 units missing: {record_id}",
        )

        detail["input_utf8_bytes"] += len(payload)
        accepted_bytes = 0
        rejected_bytes = 0
        expected_start = 0
        accepted_units: list[tuple[Mapping[str, Any], str]] = []
        seen_unit_ids: set[str] = set()
        for index, unit in enumerate(units):
            require(isinstance(unit, Mapping), "G05 unit must be an object")
            start = unit.get("start_char")
            end = unit.get("end_char")
            accepted = unit.get("accepted")
            unit_id = unit.get("unit_id")
            require(
                type(start) is int
                and type(end) is int
                and start == expected_start
                and start < end <= len(text),
                f"G05 unit partition drift: {record_id}",
            )
            require(
                type(accepted) is bool,
                f"G05 unit accepted type drift: {record_id}",
            )
            require(
                isinstance(unit_id, str)
                and bool(unit_id)
                and unit_id not in seen_unit_ids,
                f"G05 unit_id drift: {record_id}",
            )
            seen_unit_ids.add(unit_id)
            if authoritative_unit == "DOCUMENT":
                require(
                    len(units) == 1
                    and index == 0
                    and unit_id == record_id
                    and start == 0
                    and end == len(text),
                    f"G05 document unit drift: {record_id}",
                )
            else:
                require(
                    unit_id
                    == f"{record_id}#quality-window-{index:04d}",
                    f"G05 quality-window identity drift: {record_id}",
                )

            piece = text[start:end]
            piece_raw = piece.encode("utf-8")
            piece_sha = sha256(piece_raw)
            require(
                unit.get("payload_sha256") == piece_sha
                and unit.get("utf8_bytes") == len(piece_raw),
                f"G05 unit payload drift: {unit_id}",
            )
            decision_sha = unit.get("decision_sha256")
            require(
                isinstance(decision_sha, str)
                and _SHA64.fullmatch(decision_sha) is not None,
                f"G05 decision identity drift: {unit_id}",
            )
            if accepted:
                accepted_bytes += len(piece_raw)
                accepted_units.append((unit, piece))
            else:
                rejected_bytes += len(piece_raw)
                stats["g05_rejected_units"] += 1
                stats["g05_rejected_utf8_bytes"] += len(piece_raw)
            expected_start = end

        require(
            expected_start == len(text),
            f"G05 units do not reconstruct document: {record_id}",
        )
        require(
            accepted_bytes == row.get("retained_utf8_bytes")
            and rejected_bytes == row.get("rejected_utf8_bytes"),
            f"G05 retained/rejected byte accounting drift: {record_id}",
        )
        detail["retained_utf8_bytes"] += accepted_bytes
        detail["rejected_utf8_bytes"] += rejected_bytes

        meta = metadata[record_id]
        if status == "RETAIN_ALL":
            require(
                accepted_bytes == len(payload)
                and rejected_bytes == 0
                and len(accepted_units) == len(units),
                f"G05 RETAIN_ALL vector drift: {record_id}",
            )
            require(
                record_id not in seen_output_ids,
                f"G05 materialized record-id collision: {record_id}",
            )
            seen_output_ids.add(record_id)
            output.append(
                {
                    "record_id": record_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "normalized_payload": text,
                }
            )
            detail["retain_all_documents"] += 1
            continue

        if status == "REJECT_DOCUMENT":
            require(
                accepted_bytes == 0
                and rejected_bytes == len(payload)
                and not accepted_units,
                f"G05 rejected record retained bytes: {record_id}",
            )
            stats["g05_reject_documents"] += 1
            detail["reject_documents"] += 1
            continue

        require(
            authoritative_unit == "BOUNDED_NATURAL_LANGUAGE_WINDOW",
            f"G05 partial row is not authoritative windows: {record_id}",
        )
        require(
            accepted_bytes > 0
            and rejected_bytes > 0
            and accepted_units
            and len(accepted_units) < len(units),
            f"G05 partial vector drift: {record_id}",
        )
        stats["g05_partial_documents"] += 1
        detail["partial_documents"] += 1
        rejected_unit_count = len(units) - len(accepted_units)
        detail["partial_rejected_units"] += rejected_unit_count

        for unit, piece in accepted_units:
            piece_raw = piece.encode("utf-8")
            piece_sha = sha256(piece_raw)
            unit_id = str(unit["unit_id"])
            derived_id = f"{unit_id}:{piece_sha}"
            require(
                derived_id not in seen_output_ids,
                f"G05 derived record-id collision: {derived_id}",
            )
            seen_output_ids.add(derived_id)
            output.append(
                {
                    "record_id": derived_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "normalized_payload": piece,
                }
            )
            partial_projection.append(
                {
                    "parent_record_id": record_id,
                    "record_id": derived_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "unit_id": unit_id,
                    "start_char": unit["start_char"],
                    "end_char": unit["end_char"],
                    "payload_sha256": piece_sha,
                    "utf8_bytes": len(piece_raw),
                    "decision_sha256": unit["decision_sha256"],
                }
            )
            detail["partial_emitted_units"] += 1

    require(bool(output), "G05 removed every post-decontamination record")
    require(
        detail["input_utf8_bytes"]
        == detail["retained_utf8_bytes"] + detail["rejected_utf8_bytes"],
        "G05 byte conservation drift",
    )
    require(
        detail["partial_documents"] == stats["g05_partial_documents"],
        "G05 partial-document accounting drift",
    )
    require(
        detail["reject_documents"] == stats["g05_reject_documents"],
        "G05 rejected-document accounting drift",
    )
    require(
        detail["partial_rejected_units"]
        <= stats["g05_rejected_units"],
        "G05 partial rejected-unit accounting drift",
    )
    require(
        len(output)
        == detail["retain_all_documents"] + detail["partial_emitted_units"],
        "G05 materialized output-count drift",
    )
    output.sort(key=lambda row: row["record_id"])
    detail["partial_unit_projection_sha256"] = sha256(
        canonical(partial_projection)
    )
    detail["partial_unit_projection_count"] = len(partial_projection)
    return output, stats, detail


def _execute_ubuntu_clean_with_partial(
    training_records: list[dict[str, str]],
    evaluation_records: list[dict[str, Any]],
    *,
    training_handoff_evidence: Mapping[str, Any],
    base_reserved_binding: Mapping[str, Any],
    eval647_manifest: Mapping[str, Any],
    eval647_materialization_evidence: Mapping[str, Any],
    expected_base_reserved_binding_identity_sha256: str,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_training_handoff_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
]:
    """Run current clean authorities with proven partial-window materialization."""
    dependency_blobs = clean.verify_dependency_blobs()
    clean._verify_clean_release_binding(
        training_records,
        training_handoff_evidence,
    )

    report, decontam_evidence, eval647_receipt = (
        clean.execute_eval647_reserved_decontamination(
            training_records,
            evaluation_records,
            training_handoff_evidence=training_handoff_evidence,
            base_reserved_binding=base_reserved_binding,
            manifest=eval647_manifest,
            materialization_evidence=eval647_materialization_evidence,
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
            expected_inventory_identity_sha256=(
                expected_inventory_identity_sha256
            ),
            expected_survivor_authority_sha256=(
                expected_survivor_authority_sha256
            ),
            expected_training_handoff_identity_sha256=(
                expected_training_handoff_identity_sha256
            ),
            expected_selection_validation_identity_sha256=(
                expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=(
                expected_final_test_identity_sha256
            ),
        )
    )
    clean.verify_eval647_reserved_decontamination_receipt(
        eval647_receipt,
        expected_composed_reserved_binding_identity_sha256=(
            expected_composed_reserved_binding_identity_sha256
        ),
        expected_eval647_materialization_evidence_identity_sha256=(
            expected_eval647_materialization_evidence_identity_sha256
        ),
        expected_eval647_object_set_identity_sha256=(
            expected_eval647_object_set_identity_sha256
        ),
    )

    quality_inputs, metadata, decontam_excluded = (
        clean._post_decontamination_records(training_records, report)
    )
    decontam_projection = clean._input_projection(quality_inputs)
    post_decontam_rows_sha256 = clean._sha256(
        clean._cjson(decontam_projection)
    )
    decontam_identity = clean._require_sha256(
        decontam_evidence.get("execution_identity_sha256"),
        "decontamination execution identity",
    )
    quality = clean.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha256,
    )
    quality_identity = clean._require_sha256(
        quality.get("execution_identity_sha256"),
        "quality execution identity",
    )
    clean.verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha256,
        expected_execution_identity_sha256=quality_identity,
    )

    quality_survivors, quality_stats, partial_detail = (
        _materialize_quality_survivors_with_partial(
            quality_inputs,
            metadata,
            quality,
        )
    )
    privacy_inputs = clean._quality_records_for_privacy(quality_survivors)
    privacy_projection = clean._input_projection(privacy_inputs)
    post_quality_rows_sha256 = clean._sha256(
        clean._cjson(privacy_projection)
    )
    privacy = clean.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha256,
    )
    privacy_identity = clean._require_sha256(
        privacy.get("execution_identity_sha256"),
        "privacy execution identity",
    )
    clean.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha256,
        expected_execution_identity_sha256=privacy_identity,
    )

    final_survivors, privacy_stats = clean._materialize_privacy_survivors(
        quality_survivors,
        privacy,
    )
    survivor_inventory = clean.materialize_record_inventory(final_survivors)
    survivor_jsonl_sha256 = clean._sha256(
        clean.canonical_record_bytes(final_survivors)
    )
    require(
        survivor_inventory.get("record_count") == len(final_survivors),
        "survivor inventory record count drift",
    )
    require(
        survivor_inventory.get("total_payload_bytes")
        == sum(
            len(row["normalized_payload"].encode("utf-8"))
            for row in final_survivors
        ),
        "survivor inventory byte count drift",
    )

    rejection_counts = {
        "data232_excluded_records": decontam_excluded,
        **quality_stats,
        **privacy_stats,
    }
    require(
        set(rejection_counts) == clean._REJECTION_KEYS,
        "rejection-count schema drift",
    )
    detector_counts = privacy.get("detector_counts")
    require(isinstance(detector_counts, Mapping), "G06 detector counts missing")
    privacy_detector_counts: dict[str, int] = {}
    for key, value in detector_counts.items():
        require(
            isinstance(key, str)
            and bool(key)
            and type(value) is int
            and value >= 0,
            "G06 detector count malformed",
        )
        privacy_detector_counts[key] = value

    receipt_core: dict[str, Any] = {
        "schema_version": UBUNTU_CLEAN_PARTIAL_SCHEMA,
        "status": (
            "UBUNTU_CLEAN_PARTIAL_SURVIVOR_MATERIALIZED_"
            "PENDING_INDEPENDENT_QUALIFICATION"
        ),
        "clean_training_records_sha256": (
            clean.PRODUCTION_TRAINING_RECORDS_SHA256
        ),
        "clean_training_handoff_sha256": (
            clean.PRODUCTION_TRAINING_HANDOFF_SHA256
        ),
        "data232_report_sha256": clean._require_sha256(
            report.get("report_sha256"),
            "DATA-232 report identity",
        ),
        "decontamination_execution_identity_sha256": decontam_identity,
        "eval647_execution_receipt_identity_sha256": clean._require_sha256(
            eval647_receipt.get("receipt_identity_sha256"),
            "EVAL-647 receipt identity",
        ),
        "quality_execution_identity_sha256": quality_identity,
        "privacy_execution_identity_sha256": privacy_identity,
        "post_decontamination_input_rows_sha256": (
            post_decontam_rows_sha256
        ),
        "post_quality_input_rows_sha256": post_quality_rows_sha256,
        "survivor_jsonl_sha256": survivor_jsonl_sha256,
        "survivor_record_inventory_digest_sha256": clean._require_sha256(
            survivor_inventory.get("record_inventory_digest_sha256"),
            "survivor record inventory",
        ),
        "survivor_payload_inventory_digest_sha256": clean._require_sha256(
            survivor_inventory.get("payload_inventory_digest_sha256"),
            "survivor payload inventory",
        ),
        "input_training_records": len(training_records),
        "post_decontamination_records": len(quality_inputs),
        "post_quality_records": len(quality_survivors),
        "survivor_records": len(final_survivors),
        "survivor_payload_bytes": survivor_inventory["total_payload_bytes"],
        "survivor_source_objects": len(
            {row["source_id"] for row in final_survivors}
        ),
        "rejection_counts": rejection_counts,
        "g05_partial_materialization": partial_detail,
        "privacy_detector_counts": dict(
            sorted(privacy_detector_counts.items())
        ),
        "dependency_git_blobs": dependency_blobs,
        "durable_evidence_hash_only": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
        "current_corpus_launch_authority_promoted": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": clean._sha256(
            clean._cjson(receipt_core)
        ),
    }
    _verify_ubuntu_clean_partial_receipt(receipt)
    return (
        receipt,
        report,
        decontam_evidence,
        eval647_receipt,
        quality,
        privacy,
        final_survivors,
        survivor_inventory,
    )


def _verify_ubuntu_clean_partial_receipt(
    receipt: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "clean_training_records_sha256",
        "clean_training_handoff_sha256",
        "data232_report_sha256",
        "decontamination_execution_identity_sha256",
        "eval647_execution_receipt_identity_sha256",
        "quality_execution_identity_sha256",
        "privacy_execution_identity_sha256",
        "post_decontamination_input_rows_sha256",
        "post_quality_input_rows_sha256",
        "survivor_jsonl_sha256",
        "survivor_record_inventory_digest_sha256",
        "survivor_payload_inventory_digest_sha256",
        "input_training_records",
        "post_decontamination_records",
        "post_quality_records",
        "survivor_records",
        "survivor_payload_bytes",
        "survivor_source_objects",
        "rejection_counts",
        "g05_partial_materialization",
        "privacy_detector_counts",
        "dependency_git_blobs",
        "durable_evidence_hash_only",
        "terminal_post_g05_g06_authority",
        "independent_qualification_required",
        "current_corpus_launch_authority_promoted",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "optimizer_updates_executed_on_real_targets",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "receipt_identity_sha256",
    }
    require(
        type(receipt) is dict and set(receipt) == expected_keys,
        "Ubuntu clean partial receipt schema is not closed",
    )
    require(
        receipt.get("schema_version") == UBUNTU_CLEAN_PARTIAL_SCHEMA,
        "Ubuntu clean partial receipt schema drift",
    )
    body = dict(receipt)
    claimed = body.pop("receipt_identity_sha256", None)
    require(
        isinstance(claimed, str)
        and _SHA64.fullmatch(claimed) is not None
        and clean._sha256(clean._cjson(body)) == claimed,
        "Ubuntu clean partial receipt self-hash drift",
    )
    require(
        receipt.get("clean_training_records_sha256")
        == clean.PRODUCTION_TRAINING_RECORDS_SHA256
        and receipt.get("clean_training_handoff_sha256")
        == clean.PRODUCTION_TRAINING_HANDOFF_SHA256,
        "Ubuntu clean release root drift",
    )
    require(
        receipt.get("dependency_git_blobs")
        == clean.EXPECTED_DEPENDENCY_BLOBS,
        "Ubuntu clean dependency binding drift",
    )
    for key in (
        "data232_report_sha256",
        "decontamination_execution_identity_sha256",
        "eval647_execution_receipt_identity_sha256",
        "quality_execution_identity_sha256",
        "privacy_execution_identity_sha256",
        "post_decontamination_input_rows_sha256",
        "post_quality_input_rows_sha256",
        "survivor_jsonl_sha256",
        "survivor_record_inventory_digest_sha256",
        "survivor_payload_inventory_digest_sha256",
    ):
        value = receipt.get(key)
        require(
            isinstance(value, str) and _SHA64.fullmatch(value) is not None,
            f"Ubuntu clean nested root malformed: {key}",
        )

    for key in (
        "input_training_records",
        "post_decontamination_records",
        "post_quality_records",
        "survivor_records",
        "survivor_payload_bytes",
        "survivor_source_objects",
    ):
        require(
            type(receipt.get(key)) is int and receipt[key] > 0,
            f"Ubuntu clean invalid positive count: {key}",
        )
    require(
        receipt["input_training_records"]
        == clean.PRODUCTION_INPUT_RECORD_COUNT,
        "Ubuntu clean production input count drift",
    )

    rejection = receipt.get("rejection_counts")
    require(
        isinstance(rejection, Mapping)
        and set(rejection) == clean._REJECTION_KEYS,
        "Ubuntu clean rejection schema drift",
    )
    require(
        all(type(value) is int and value >= 0 for value in rejection.values()),
        "Ubuntu clean rejection value drift",
    )
    require(
        receipt["input_training_records"]
        - receipt["post_decontamination_records"]
        == rejection["data232_excluded_records"],
        "Ubuntu DATA-232 rejection/count drift",
    )

    detail = receipt.get("g05_partial_materialization")
    require(isinstance(detail, Mapping), "Ubuntu G05 partial detail missing")
    detail_keys = {
        "input_documents",
        "retain_all_documents",
        "partial_documents",
        "reject_documents",
        "partial_emitted_units",
        "partial_rejected_units",
        "input_utf8_bytes",
        "retained_utf8_bytes",
        "rejected_utf8_bytes",
        "partial_unit_projection_sha256",
        "partial_unit_projection_count",
    }
    require(
        set(detail) == detail_keys,
        "Ubuntu G05 partial detail schema drift",
    )
    require(
        all(
            type(detail[key]) is int and detail[key] >= 0
            for key in detail_keys
            if key != "partial_unit_projection_sha256"
        ),
        "Ubuntu G05 partial detail integer drift",
    )
    require(
        isinstance(detail["partial_unit_projection_sha256"], str)
        and _SHA64.fullmatch(
            detail["partial_unit_projection_sha256"]
        )
        is not None,
        "Ubuntu G05 partial projection identity drift",
    )
    require(
        detail["input_documents"]
        == receipt["post_decontamination_records"],
        "Ubuntu G05 input-document count drift",
    )
    require(
        detail["input_documents"]
        == (
            detail["retain_all_documents"]
            + detail["partial_documents"]
            + detail["reject_documents"]
        ),
        "Ubuntu G05 document classification drift",
    )
    require(
        detail["partial_documents"] == rejection["g05_partial_documents"]
        and detail["reject_documents"] == rejection["g05_reject_documents"],
        "Ubuntu G05 status accounting drift",
    )
    require(
        detail["partial_unit_projection_count"]
        == detail["partial_emitted_units"],
        "Ubuntu G05 partial projection count drift",
    )
    require(
        receipt["post_quality_records"]
        == detail["retain_all_documents"] + detail["partial_emitted_units"],
        "Ubuntu G05 physical output-count drift",
    )
    require(
        detail["input_utf8_bytes"]
        == detail["retained_utf8_bytes"] + detail["rejected_utf8_bytes"],
        "Ubuntu G05 physical byte conservation drift",
    )
    require(
        detail["rejected_utf8_bytes"]
        == rejection["g05_rejected_utf8_bytes"],
        "Ubuntu G05 rejected-byte accounting drift",
    )
    require(
        receipt["post_quality_records"] - receipt["survivor_records"]
        == (
            rejection["g06_quarantine_records"]
            + rejection["g06_exclude_records"]
        ),
        "Ubuntu G06 drop/count drift",
    )
    require(
        0 < receipt["survivor_source_objects"]
        <= receipt["survivor_records"],
        "Ubuntu survivor source-object count drift",
    )
    require(receipt.get("durable_evidence_hash_only") is True, "raw evidence widened")
    require(
        receipt.get("terminal_post_g05_g06_authority") is False
        and receipt.get("independent_qualification_required") is True
        and receipt.get("current_corpus_launch_authority_promoted") is False,
        "Ubuntu clean authority boundary widened",
    )
    require(
        receipt.get("authorized_optimized_target_exposure") == 0
        and receipt.get("optimizer_updates_executed_on_real_targets") == 0,
        "Ubuntu clean optimized-target authority widened",
    )
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        require(receipt.get(key) is False, f"Ubuntu clean truth drift: {key}")


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
        ) = _execute_ubuntu_clean_with_partial(
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
        _verify_ubuntu_clean_partial_receipt(receipt)
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
            "g05_partial_materialization": receipt[
                "g05_partial_materialization"
            ],
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
