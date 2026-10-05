#!/usr/bin/env python3
"""Execute exact PEP+LoC V10 global-dedup survivors through DATA-232/G05/G06.

Execution-only carrier. It authenticates the independently-qualified #2786
two-clean survivor authority, freshly reconstructs the exact clean-retained,
PEP and Library-of-Congress payload universe from pinned authorities, admits
only the exact 370 globally-unique survivors, and delegates DATA-232, G05 and
G06 to the unchanged current-clean authority.

Durable output is text-free and grants zero canonical/tokenizer/training credit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for location in (str(ROOT / "tools"), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved
from twelve_six.data import expanded_global_dedup_v10 as v10

SCHEMA = "12-6.d03-pep-loc-v10-decontam-g05-g06-execution.v1"
PARENT_EXECUTION_HEAD = "935d508130b4ba1cb5a3a81c2272fd4d19552579"
PARENT_WORKFLOW_RUN_ID = 37338684311
PARENT_PASS_ARTIFACT_ID = 11357173588
PARENT_PASS_ARTIFACT_ZIP_SHA256 = (
    "379a7984b86afde8ea3bf0d55ee9f3c19a045799b0ee106e8bb19bf4680054d5"
)
PARENT_PROOF_ARTIFACT_ID = 11357043314
PARENT_PROOF_ARTIFACT_ZIP_SHA256 = (
    "befa277e2c624b7719295907c863db383a8bc2f206271b073447f2e4d95a254c"
)
PARENT_SURVIVORS_FILE_SHA256 = (
    "6f93ca5d730aaa1e07d8429f77606a43705697667e5a786bfda17506ac394d72"
)
PARENT_EVIDENCE_FILE_SHA256 = (
    "a7c57f08708b718833272b9d41a67e39cc679476661486a7689f96cb405615fd"
)
PARENT_PROOF_FILE_SHA256 = (
    "2f3ce844a8d62203fa1ef54407842ad474059640ef06444fa62bf83ea8b79400"
)
PARENT_REPORT_IDENTITY_SHA256 = (
    "cf7e4b4f1758510a6f01068ac30aebb0696623664d38e13ab2290b63a6da9de0"
)
PARENT_SURVIVOR_AUTHORITY_SHA256 = (
    "01727ad4796cddd0a6c4ec6c2add9fda0c1e523155f4f9fa024d7be0956613c9"
)
PARENT_EVIDENCE_IDENTITY_SHA256 = (
    "40f3c25c679739040940d06b6650aedaa140612e6a28fe5408935eb22b94c19f"
)
PARENT_PROOF_IDENTITY_SHA256 = (
    "bb0a65d958791b05df2ba47b5b108c93f39ccf59f8bf8115c00fb1f9b90d4b6c"
)
EXPECTED_PARENT_OBJECTS = 370
EXPECTED_PARENT_BYTES = 15_014_196
EXPECTED_PRE_DEDUP_OBJECTS = 371
EXPECTED_PRE_DEDUP_BYTES = 15_089_399

EXPECTED_BLOBS = {
    "src/twelve_six/data/current_clean_execution_v1.py":
        "b82ed11626a267dfffa16acd79e45c0cf3e6750b",
    "src/twelve_six/data/expanded_global_dedup_v10.py":
        "3d4f14f75cada51805afaa84080a0092fcf3bb92",
    "tools/materialize_d03_peps.py":
        "310f5431dcad851c522d565d079513072ec9a1dc",
    "tools/materialize_d03_common_pile_loc.py":
        "7d121300cd241672fd3f451451c573a98db680ed",
    "tools/prepare_data232_ephemeral_handoff_v1.py":
        "88e75ea2cefaedf61f850a36376a6ebde7ee4ce8",
}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


class PepLocPostQpError(RuntimeError):
    """Fail-closed PEP+LoC post-global-dedup execution mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PepLocPostQpError(message)


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
    raise PepLocPostQpError(f"non-finite JSON constant rejected: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON member: {key}")
        result[key] = value
    return result


def strict_loads(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PepLocPostQpError(f"{label}: strict JSON parse failed") from exc


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
        raise PepLocPostQpError("git failed: " + " ".join(args) + ": " + detail)
    return result


def _verify_self_hash(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    *,
    label: str,
) -> None:
    require(type(value) is dict, f"{label}: root must be exact object")
    body = copy.deepcopy(value)
    identity = body.pop(field, None)
    require(
        isinstance(identity, str) and _SHA64.fullmatch(identity) is not None,
        f"{label}: identity malformed",
    )
    require(identity == expected, f"{label}: identity drift")
    require(sha256(canonical(body)) == identity, f"{label}: self-hash mismatch")


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(
        isinstance(expected_execution_head, str)
        and _SHA40.fullmatch(expected_execution_head) is not None,
        "expected execution head must be lowercase 40-hex",
    )
    require(git("rev-parse", "HEAD").stdout.strip() == expected_execution_head, "HEAD drift")
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        PARENT_EXECUTION_HEAD,
        expected_execution_head,
        check=False,
    )
    require(ancestor.returncode == 0, "qualified PEP+LoC parent is not ancestor")
    observed: dict[str, str] = {}
    for path, expected_blob in EXPECTED_BLOBS.items():
        head_blob = git("rev-parse", f"HEAD:{path}").stdout.strip()
        worktree_blob = git("hash-object", str(ROOT / path)).stdout.strip()
        require(head_blob == expected_blob, f"Git blob drift: {path}")
        require(worktree_blob == expected_blob, f"worktree blob drift: {path}")
        observed[path] = expected_blob
    for key, value in clean.verify_dependency_blobs().items():
        observed[f"current_clean:{key}"] = value
    return dict(sorted(observed.items()))


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
    proof_path: Path,
) -> list[str]:
    survivors = load_json(
        survivors_path,
        "PEP+LoC parent survivors",
        expected_sha256=PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = load_json(
        evidence_path,
        "PEP+LoC parent evidence",
        expected_sha256=PARENT_EVIDENCE_FILE_SHA256,
    )
    proof = load_json(
        proof_path,
        "PEP+LoC parent two-clean proof",
        expected_sha256=PARENT_PROOF_FILE_SHA256,
    )

    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="PEP+LoC parent survivors",
    )
    _verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        PARENT_EVIDENCE_IDENTITY_SHA256,
        label="PEP+LoC parent evidence",
    )
    _verify_self_hash(
        proof,
        "proof_identity_sha256",
        PARENT_PROOF_IDENTITY_SHA256,
        label="PEP+LoC parent two-clean proof",
    )

    require(
        survivors.get("schema_version")
        == "12-6.d03-expanded-global-dedup-v10-pep-loc-survivors.v1",
        "parent survivor schema drift",
    )
    require(
        survivors.get("pre_dedup_source_object_count") == EXPECTED_PRE_DEDUP_OBJECTS
        and survivors.get("pre_dedup_declared_capacity_bytes") == EXPECTED_PRE_DEDUP_BYTES,
        "parent pre-dedup facts drift",
    )
    require(
        survivors.get("post_dedup_survivor_source_object_count") == EXPECTED_PARENT_OBJECTS
        and survivors.get("post_dedup_declared_capacity_bytes") == EXPECTED_PARENT_BYTES,
        "parent survivor facts drift",
    )
    ids = survivors.get("survivor_source_ids")
    require(
        type(ids) is list
        and len(ids) == EXPECTED_PARENT_OBJECTS
        and all(isinstance(item, str) and item for item in ids)
        and ids == sorted(ids)
        and len(set(ids)) == len(ids),
        "parent survivor IDs drift",
    )
    truth = survivors.get("truth_boundary")
    require(isinstance(truth, Mapping), "parent survivor truth missing")
    require(
        truth.get("global_dedup_execution_complete") is True
        and truth.get("reserved_evaluation_decontamination_complete") is False
        and truth.get("canonical_quality_privacy_complete") is False,
        "parent gate truth drift",
    )
    require(
        truth.get("training_authorized_bytes") == 0
        and truth.get("tokenizer_fit_authorized") is False
        and truth.get("model_training_executed") is False,
        "parent fabricated science authority",
    )

    expected_evidence = {
        "execution_head_sha": PARENT_EXECUTION_HEAD,
        "pre_dedup_source_objects": EXPECTED_PRE_DEDUP_OBJECTS,
        "pre_dedup_payload_bytes": EXPECTED_PRE_DEDUP_BYTES,
        "post_dedup_source_objects": EXPECTED_PARENT_OBJECTS,
        "post_dedup_unique_bytes": EXPECTED_PARENT_BYTES,
        "matcher_report_sha256": PARENT_REPORT_IDENTITY_SHA256,
        "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "raw_candidate_payloads_retained": False,
    }
    for key, expected in expected_evidence.items():
        require(
            evidence.get(key) == expected and type(evidence.get(key)) is type(expected),
            f"parent evidence drift: {key}",
        )

    expected_proof = {
        "execution_head_sha": PARENT_EXECUTION_HEAD,
        "pre_dedup_source_objects": EXPECTED_PRE_DEDUP_OBJECTS,
        "pre_dedup_payload_bytes": EXPECTED_PRE_DEDUP_BYTES,
        "post_dedup_source_objects": EXPECTED_PARENT_OBJECTS,
        "post_dedup_unique_bytes": EXPECTED_PARENT_BYTES,
        "matcher_report_sha256": PARENT_REPORT_IDENTITY_SHA256,
        "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "two_fresh_physical_executions_converged": True,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "raw_candidate_payloads_retained": False,
    }
    for key, expected in expected_proof.items():
        require(
            proof.get(key) == expected and type(proof.get(key)) is type(expected),
            f"parent proof drift: {key}",
        )
    return list(ids)


def reconstruct_exact_survivors(
    *,
    survivor_ids: Sequence[str],
    clean_training_records: Path,
    clean_training_handoff: Path,
    pep_candidate: Path,
    pep_report: Path,
    loc_candidate: Path,
    loc_report: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    clean_rows, clean_payloads, clean_handoff = v10._validate_clean_retained(
        clean_training_records.read_bytes(),
        clean_training_handoff.read_bytes(),
    )
    pep_rows, _ = v10._validate_pep(pep_candidate.read_bytes(), pep_report.read_bytes())
    pep_matcher_rows, pep_payloads = v10._pep_matcher_inputs(pep_rows)
    loc_rows, _ = v10._validate_loc(loc_candidate.read_bytes(), loc_report.read_bytes())
    loc_matcher_rows, loc_payloads = v10._loc_matcher_inputs(loc_rows)

    rows = [*clean_rows, *pep_matcher_rows, *loc_matcher_rows]
    payloads: dict[str, bytes] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row["source_id"])
        require(source_id not in by_id, f"source row collision: {source_id}")
        by_id[source_id] = dict(row)
    for source in (clean_payloads, pep_payloads, loc_payloads):
        for source_id, raw in source.items():
            require(source_id not in payloads, f"payload collision: {source_id}")
            payloads[source_id] = raw
    require(set(payloads) == set(by_id), "source/payload universe coverage drift")
    require(len(by_id) == EXPECTED_PRE_DEDUP_OBJECTS, "pre-dedup object count drift")
    require(
        sum(len(raw) for raw in payloads.values()) == EXPECTED_PRE_DEDUP_BYTES,
        "pre-dedup byte count drift",
    )

    selected_rows: list[dict[str, Any]] = []
    selected_payloads: dict[str, bytes] = {}
    for source_id in survivor_ids:
        require(source_id in by_id and source_id in payloads, f"missing survivor: {source_id}")
        row = by_id[source_id]
        raw = payloads[source_id]
        require(
            row.get("declared_capacity_bytes") == len(raw),
            f"survivor declared byte drift: {source_id}",
        )
        require(
            row.get("expected_raw_sha256") == sha256(raw),
            f"survivor payload SHA drift: {source_id}",
        )
        selected_rows.append(row)
        selected_payloads[source_id] = raw

    require(len(selected_rows) == EXPECTED_PARENT_OBJECTS, "selected survivor count drift")
    require(
        sum(len(raw) for raw in selected_payloads.values()) == EXPECTED_PARENT_BYTES,
        "selected survivor byte count drift",
    )
    source_authority = {
        "clean_training_records_sha256": sha256(clean_training_records.read_bytes()),
        "clean_training_handoff_sha256": sha256(clean_training_handoff.read_bytes()),
        "pep_candidate_sha256": sha256(pep_candidate.read_bytes()),
        "pep_report_sha256": sha256(pep_report.read_bytes()),
        "loc_candidate_sha256": sha256(loc_candidate.read_bytes()),
        "loc_report_sha256": sha256(loc_report.read_bytes()),
        "pre_dedup_source_objects": len(by_id),
        "pre_dedup_payload_bytes": sum(len(raw) for raw in payloads.values()),
        "post_global_dedup_survivor_ids_sha256": sha256(canonical(sorted(selected_payloads))),
        "post_global_dedup_survivor_objects": len(selected_rows),
        "post_global_dedup_survivor_bytes": sum(
            len(raw) for raw in selected_payloads.values()
        ),
        "clean_handoff_identity_sha256": clean_handoff.get("handoff_identity_sha256"),
    }
    return selected_rows, selected_payloads, source_authority


def build_training_authorities(
    source_rows: Sequence[Mapping[str, Any]],
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
        modality = str(row["modality"])
        records.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )
    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(len(records) == EXPECTED_PARENT_OBJECTS, "training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows) == EXPECTED_PARENT_BYTES,
        "training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-pep-loc-v10-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_PARENT_OBJECTS,
        "total_payload_bytes": EXPECTED_PARENT_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-pep-loc-v10-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_matcher_report_sha256": PARENT_REPORT_IDENTITY_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_PARENT_OBJECTS,
        "declared_capacity_bytes": EXPECTED_PARENT_BYTES,
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
        "retained_source_count": EXPECTED_PARENT_OBJECTS,
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
    return records, inventory, subset, handoff, training_records_sha, handoff_raw_sha


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
    clean.PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256 = inventory_identity
    clean.PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = subset_authority
    return original


def _restore_clean_release(original: Mapping[str, Any]) -> None:
    for key, value in original.items():
        setattr(clean, key, value)


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



def execute(
    *,
    expected_execution_head: str,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
    parent_proof_json: Path,
    clean_training_records: Path,
    clean_training_handoff: Path,
    pep_candidate: Path,
    pep_report: Path,
    loc_candidate: Path,
    loc_report: Path,
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
    survivor_ids = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_proof_json,
    )
    source_rows, payloads, source_authority = reconstruct_exact_survivors(
        survivor_ids=survivor_ids,
        clean_training_records=clean_training_records,
        clean_training_handoff=clean_training_handoff,
        pep_candidate=pep_candidate,
        pep_report=pep_report,
        loc_candidate=loc_candidate,
        loc_report=loc_report,
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

    original_release = _patch_clean_release(
        count=EXPECTED_PARENT_OBJECTS,
        training_records_sha=training_records_sha,
        handoff_raw_sha=handoff_raw_sha,
        inventory_identity=inventory["inventory_identity_sha256"],
        subset_authority=subset["subset_authority_sha256"],
    )
    original_materializer = clean._materialize_quality_survivors
    partial_detail: dict[str, Any] = {}

    def patched_materializer(
        inputs: Sequence[Mapping[str, str]],
        metadata: Mapping[str, Mapping[str, str]],
        quality: Mapping[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        output, stats, detail = _materialize_quality_survivors_with_partial(
            list(inputs),
            metadata,
            quality,
        )
        partial_detail.clear()
        partial_detail.update(detail)
        return output, stats

    clean._materialize_quality_survivors = patched_materializer
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
    finally:
        clean._materialize_quality_survivors = original_materializer
        _restore_clean_release(original_release)

    require(bool(partial_detail), "G05 physical materialization detail missing")
    final_bytes = int(receipt["survivor_payload_bytes"])
    gate_loss_bytes = EXPECTED_PARENT_BYTES - final_bytes
    require(gate_loss_bytes >= 0, "later gates widened parent payload bytes")
    require(
        survivor_inventory.get("total_payload_bytes") == final_bytes,
        "survivor inventory/receipt byte drift",
    )
    require(
        len(final_survivors) == receipt["survivor_records"],
        "final survivor count drift",
    )

    input_by_modality: dict[str, int] = {}
    input_family_counts: dict[str, int] = {}
    for row in source_rows:
        source_id = str(row["source_id"])
        modality = str(row["modality"])
        family = str(row["source_family"])
        input_by_modality[modality] = input_by_modality.get(modality, 0) + len(
            payloads[source_id]
        )
        input_family_counts[family] = input_family_counts.get(family, 0) + 1

    final_rows = survivor_inventory.get("records")
    require(type(final_rows) is list, "survivor inventory records missing")
    final_by_modality: dict[str, int] = {}
    final_family_counts: dict[str, int] = {}
    for row in final_rows:
        require(type(row) is dict, "survivor inventory row invalid")
        modality = str(row["modality"])
        family = str(row["family"])
        payload_bytes = row["payload_bytes"]
        require(type(payload_bytes) is int and payload_bytes > 0, "survivor bytes invalid")
        final_by_modality[modality] = final_by_modality.get(modality, 0) + payload_bytes
        final_family_counts[family] = final_family_counts.get(family, 0) + 1

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": expected_execution_head,
        "parent": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "workflow_run_id": PARENT_WORKFLOW_RUN_ID,
            "pass_artifact_id": PARENT_PASS_ARTIFACT_ID,
            "pass_artifact_zip_sha256": PARENT_PASS_ARTIFACT_ZIP_SHA256,
            "proof_artifact_id": PARENT_PROOF_ARTIFACT_ID,
            "proof_artifact_zip_sha256": PARENT_PROOF_ARTIFACT_ZIP_SHA256,
            "matcher_report_sha256": PARENT_REPORT_IDENTITY_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "post_global_dedup_objects": EXPECTED_PARENT_OBJECTS,
            "post_global_dedup_payload_bytes": EXPECTED_PARENT_BYTES,
        },
        "source_authority": source_authority,
        "postdedup_authority": {
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "subset_authority_sha256": subset["subset_authority_sha256"],
            "training_records_sha256": training_records_sha,
            "training_handoff_raw_sha256": handoff_raw_sha,
            "training_handoff_identity_sha256": handoff[
                "handoff_identity_sha256"
            ],
            "input_payload_bytes_by_modality": dict(sorted(input_by_modality.items())),
            "input_source_family_counts": dict(sorted(input_family_counts.items())),
        },
        "gate_execution": {
            "composition_receipt": receipt,
            "g05_partial_materialization": partial_detail,
            "input_records": receipt["input_training_records"],
            "post_decontamination_records": receipt["post_decontamination_records"],
            "post_quality_records": receipt["post_quality_records"],
            "survivor_records": receipt["survivor_records"],
            "survivor_source_objects": receipt["survivor_source_objects"],
            "survivor_payload_bytes": final_bytes,
            "later_gate_loss_bytes": gate_loss_bytes,
            "survivor_payload_bytes_by_modality": dict(
                sorted(final_by_modality.items())
            ),
            "survivor_source_family_counts": dict(
                sorted(final_family_counts.items())
            ),
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
            "global_cross_source_dedup_complete_for_parent": True,
            "reserved_evaluation_decontamination_complete": True,
            "canonical_quality_privacy_complete": True,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--parent-survivors-json", type=Path, required=True)
    parser.add_argument("--parent-evidence-json", type=Path, required=True)
    parser.add_argument("--parent-proof-json", type=Path, required=True)
    parser.add_argument("--clean-training-records", type=Path, required=True)
    parser.add_argument("--clean-training-handoff", type=Path, required=True)
    parser.add_argument("--pep-candidate", type=Path, required=True)
    parser.add_argument("--pep-report", type=Path, required=True)
    parser.add_argument("--loc-candidate", type=Path, required=True)
    parser.add_argument("--loc-report", type=Path, required=True)
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
            clean_training_records=args.clean_training_records,
            clean_training_handoff=args.clean_training_handoff,
            pep_candidate=args.pep_candidate,
            pep_report=args.pep_report,
            loc_candidate=args.loc_candidate,
            loc_report=args.loc_report,
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
        PepLocPostQpError,
        clean.CurrentCleanExecutionError,
        reserved.CurrentDecontaminationExecutionError,
        v10.ExpandedDedupV10Error,
        OSError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    gate = evidence["gate_execution"]
    print("D03_PEP_LOC_POST_QP=PASS_ZERO_CREDIT")
    print("SURVIVOR_PAYLOAD_BYTES=" + str(gate["survivor_payload_bytes"]))
    print("LATER_GATE_LOSS_BYTES=" + str(gate["later_gate_loss_bytes"]))
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
