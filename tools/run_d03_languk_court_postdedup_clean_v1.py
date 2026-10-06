"""Execute exact Lang-UK global-dedup survivors through current reserved-eval/G05/G06 gates.

Execution glue only. This carrier consumes the byte-qualified two-clean global-dedup
authority from PR #2981, re-materializes the exact pinned Lang-UK source, selects only
the authenticated global-dedup survivor IDs, and delegates reserved-evaluation
decontamination plus current G05/G06 mechanics to the incumbent current-clean
orchestrator. Durable outputs are text-free and grant zero corpus/tokenizer/training
authority until downstream balance/caps/split/packing/unique-loss gates qualify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (str(ROOT / "tools"), str(ROOT / "src")):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_languk_court_global_dedup_v1 as languk_parent

from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data import current_reserved_decontamination_v1 as reserved

SCHEMA = "12-6.d03-languk-court-postdedup-clean-execution.v1"
PARENT_EXECUTION_HEAD = "75b5f376679f2b1e82967914b5128863f3ad3e27"
PARENT_WORKFLOW_RUN_ID = 37515051295
PARENT_ARTIFACT_ID = 11436803293
PARENT_ARTIFACT_ZIP_SHA256 = "60e28a851ddadcee7eda44c827d70988004fff60d241da16e4f8acd89457c961"
PARENT_SURVIVORS_FILE_SHA256 = "b65c71e64345fbce0f7b6593acc5cc924696a1aba81861b3e49fc2138fd83a58"
PARENT_EVIDENCE_FILE_SHA256 = "2b758d3bd37913daad19c80e70e820218fd47578f910018c0d3f2c38cd5149aa"
PARENT_PROOF_ARTIFACT_ID = 11437337310
PARENT_PROOF_ZIP_SHA256 = "8135bb7e2d62a0a39e460dc72a735f7d8bdc8eb3ac8691e8074d47b7141b3a0e"
PARENT_PROOF_FILE_SHA256 = "ebc3f6db77718727b45fa43c76a023a87d86980c32cf75a657663fc1391a5781"
PARENT_MATCHER_REPORT_SHA256 = "a8d51c1c1c78b8e03865608fbbbe19d584d04114efa65f233cc2bd8cf08e4fd9"
PARENT_SURVIVOR_AUTHORITY_SHA256 = "3c7e0466a70d5d4fdb4257c62a08a9af70df060d523b6b56f1aa94376e3e888a"
PARENT_PROOF_IDENTITY_SHA256 = "b8acf15aa93d7ceea93b655bbed26d850b499894d6ab98d565d570b5c8debc93"
EXPECTED_SOURCE_FAMILY = "ua.languk.supreme-court-decisions"
EXPECTED_SURVIVOR_OBJECTS = 256
EXPECTED_SURVIVOR_BYTES = 2_880_510
EXPECTED_SURVIVOR_IDS_SHA256 = "e6fd65b8d6820919ad425a19d99dbec13fc9d0a2529c9da8cb7c819a01721be3"
CARRIER_PATH = "tools/run_d03_languk_court_postdedup_clean_v1.py"

AUTHORITY_PATHS = (
    "tools/run_d03_languk_court_global_dedup_v1.py",
    "tools/run_d03_languk_court_privacy_retest_v1.py",
    "tools/retest_d03_languk_supreme_court.py",
    "src/twelve_six/data/current_clean_execution_v1.py",
    "src/twelve_six/data/current_reserved_decontamination_v1.py",
    "src/twelve_six/data/eval647_reserved_decontamination_v1.py",
    "src/twelve_six/data/eval647_future_training_exclusion_v1.py",
    "src/twelve_six/data/quality_execution_authority.py",
    "src/twelve_six/data/privacy_execution_authority.py",
    "src/twelve_six/data/post_g05_g06_materialization_v1.py",
    "src/twelve_six/data/eval303_selection_payload_resolver_v1.py",
    "src/twelve_six/data/eval233_final_test_resolver_v1.py",
    "configs/evaluation/eval_code_reserve_v1.json",
    "evidence/eval647/code_selection_source_materialization_v1.json",
)


class LangUkPostDedupCleanError(RuntimeError):
    """Fail-closed Lang-UK post-global-dedup execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LangUkPostDedupCleanError(message)


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


def _read_json(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 is not None:
        require(sha256(raw) == expected_sha256, f"{label} file SHA-256 drift")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LangUkPostDedupCleanError(f"{label} is not strict UTF-8 JSON") from exc
    require(type(value) is dict, f"{label} root must be object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(path.read_bytes().splitlines()):
        require(bool(raw), f"{label}[{index}] empty row")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LangUkPostDedupCleanError(f"{label}[{index}] invalid JSON") from exc
        require(type(value) is dict, f"{label}[{index}] must be object")
        rows.append(value)
    require(bool(rows), f"{label} is empty")
    return rows


def _verify_self_hash(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    *,
    label: str,
) -> None:
    claimed = value.get(field)
    require(claimed == expected, f"{label} expected identity drift")
    core = deepcopy(dict(value))
    core.pop(field, None)
    require(sha256(canonical(core)) == claimed, f"{label} self-hash mismatch")


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_local_authority(expected_execution_head: str) -> dict[str, str]:
    require(_git("rev-parse", "HEAD") == expected_execution_head, "execution HEAD drift")
    require(
        _git("merge-base", PARENT_EXECUTION_HEAD, "HEAD") == PARENT_EXECUTION_HEAD,
        "parent global-dedup head is not an ancestor",
    )
    require(not _git("status", "--porcelain=v1", "--untracked-files=all"), "dirty worktree")
    observed: dict[str, str] = {}
    for authority_path in AUTHORITY_PATHS:
        parent_blob = _git("rev-parse", f"{PARENT_EXECUTION_HEAD}:{authority_path}")
        head_blob = _git("rev-parse", f"HEAD:{authority_path}")
        worktree_blob = _git("hash-object", str(ROOT / authority_path))
        require(
            bool(parent_blob) and head_blob == parent_blob and worktree_blob == head_blob,
            f"authority path drift: {authority_path}",
        )
        observed[authority_path] = head_blob
    return dict(sorted(observed.items()))


def verify_carrier() -> str:
    return _git("hash-object", str(ROOT / CARRIER_PATH))


def verify_parent_artifact(
    survivors_path: Path,
    evidence_path: Path,
    proof_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
    survivors = _read_json(
        survivors_path,
        "Lang-UK parent survivors",
        PARENT_SURVIVORS_FILE_SHA256,
    )
    evidence = _read_json(
        evidence_path,
        "Lang-UK parent evidence",
        PARENT_EVIDENCE_FILE_SHA256,
    )
    proof = _read_json(
        proof_path,
        "Lang-UK parent two-clean proof",
        PARENT_PROOF_FILE_SHA256,
    )
    _verify_self_hash(
        survivors,
        "survivor_authority_sha256",
        PARENT_SURVIVOR_AUTHORITY_SHA256,
        label="Lang-UK parent survivors",
    )
    _verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        "4f90bebbaa8a8e23d27c953a5a1bd8569247c1c2e623e2f360e5e05de9224cca",
        label="Lang-UK parent evidence",
    )
    _verify_self_hash(
        proof,
        "proof_identity_sha256",
        PARENT_PROOF_IDENTITY_SHA256,
        label="Lang-UK parent two-clean proof",
    )
    require(proof.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "parent proof head drift")
    require(
        proof.get("combined_report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent matcher report drift",
    )
    require(
        proof.get("survivor_authority_sha256") == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent proof survivor root drift",
    )
    require(proof.get("two_fresh_executions_converged") is True, "parent lacks two-clean proof")
    require(proof.get("candidate_records") == EXPECTED_SURVIVOR_OBJECTS, "parent candidate count drift")
    require(
        proof.get("marginal_global_unique_capacity_bytes") == EXPECTED_SURVIVOR_BYTES,
        "parent marginal capacity drift",
    )
    require(
        proof.get("canonical_capacity_credited") == 0
        and proof.get("authorized_optimized_target_exposure") == 0
        and proof.get("tokenizer_fit_authorized") is False
        and proof.get("training_executed") is False,
        "parent truth boundary widened",
    )

    require(evidence.get("execution_head_sha") == PARENT_EXECUTION_HEAD, "parent evidence head drift")
    combined = evidence.get("combined")
    require(isinstance(combined, Mapping), "parent combined evidence missing")
    require(
        combined.get("report_sha256") == PARENT_MATCHER_REPORT_SHA256,
        "parent evidence matcher root drift",
    )
    require(
        evidence.get("survivor_authority_sha256") == PARENT_SURVIVOR_AUTHORITY_SHA256,
        "parent evidence survivor root drift",
    )

    require(survivors.get("source_family") == EXPECTED_SOURCE_FAMILY, "parent family drift")
    require(
        survivors.get("candidate_source_object_count") == EXPECTED_SURVIVOR_OBJECTS,
        "parent candidate object drift",
    )
    require(
        survivors.get("candidate_payload_utf8_bytes") == EXPECTED_SURVIVOR_BYTES,
        "parent candidate byte drift",
    )
    require(
        survivors.get("post_dedup_selected_extension_source_object_count")
        == EXPECTED_SURVIVOR_OBJECTS,
        "parent selected object drift",
    )
    require(
        survivors.get("post_dedup_selected_extension_payload_bytes")
        == EXPECTED_SURVIVOR_BYTES,
        "parent selected byte drift",
    )
    require(
        survivors.get("marginal_global_unique_capacity_bytes") == EXPECTED_SURVIVOR_BYTES,
        "parent marginal byte drift",
    )
    rows = survivors.get("survivors")
    require(isinstance(rows, list) and len(rows) == EXPECTED_SURVIVOR_OBJECTS, "parent survivor rows drift")
    ids = [row.get("source_id") for row in rows if isinstance(row, Mapping)]
    require(
        len(ids) == EXPECTED_SURVIVOR_OBJECTS
        and all(type(value) is str and bool(value) for value in ids)
        and len(set(ids)) == EXPECTED_SURVIVOR_OBJECTS,
        "parent survivor IDs invalid",
    )
    require(
        sha256(canonical(sorted(ids))) == EXPECTED_SURVIVOR_IDS_SHA256,
        "parent survivor ID-set drift",
    )
    truth = survivors.get("truth_boundary")
    require(isinstance(truth, Mapping), "parent survivor truth boundary missing")
    require(truth.get("global_dedup_execution_complete") is True, "parent global dedup incomplete")
    require(
        truth.get("reserved_evaluation_decontamination_complete") is False
        and truth.get("post_dedup_quality_privacy_complete") is False
        and truth.get("canonical_capacity_credited") == 0
        and truth.get("training_authorized_bytes") == 0
        and truth.get("tokenizer_fit_authorized") is False
        and truth.get("training_executed") is False,
        "parent unexpectedly claims downstream authority",
    )
    return survivors, evidence, frozenset(ids)


def acquire_languk_survivors(
    *,
    source_parquet: Path,
    parent_survivors: Mapping[str, Any],
    expected_survivor_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    projected_sources, projected_payloads, receipt = languk_parent.reproduce_source_candidate(
        source_parquet
    )
    by_id = {str(row["source_id"]): row for row in projected_sources}
    require(len(by_id) == languk_parent.EXPECTED_CANDIDATE_RECORDS, "fresh projection count drift")
    require(set(projected_payloads) == set(by_id), "fresh projection payload coverage drift")
    require(expected_survivor_ids <= set(by_id), "parent survivor absent from fresh projection")

    parent_rows_raw = parent_survivors.get("survivors")
    require(isinstance(parent_rows_raw, list), "parent survivor vector missing")
    parent_rows = {
        str(row["source_id"]): row
        for row in parent_rows_raw
        if isinstance(row, Mapping) and type(row.get("source_id")) is str
    }
    require(set(parent_rows) == set(expected_survivor_ids), "parent survivor vector/ID-set drift")

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for source_id in sorted(expected_survivor_ids):
        row = by_id[source_id]
        parent_row = parent_rows[source_id]
        raw = projected_payloads[source_id]
        require(row.get("source_family") == EXPECTED_SOURCE_FAMILY, f"{source_id}: family drift")
        require(row.get("modality") == "uk", f"{source_id}: modality drift")
        require(type(raw) is bytes and len(raw) > 0, f"{source_id}: payload missing")
        require(row.get("declared_capacity_bytes") == len(raw), f"{source_id}: byte count drift")
        require(row.get("expected_raw_sha256") == sha256(raw), f"{source_id}: payload SHA drift")
        for key in (
            "source_family",
            "stable_origin_id",
            "stable_object_id",
            "declared_capacity_bytes",
            "expected_raw_sha256",
            "origin_key",
        ):
            require(row.get(key) == parent_row.get(key), f"{source_id}: parent {key} drift")
        rows.append(dict(row))
        payloads[source_id] = raw
        total += len(raw)

    require(len(rows) == EXPECTED_SURVIVOR_OBJECTS, "fresh survivor object count drift")
    require(total == EXPECTED_SURVIVOR_BYTES, "fresh survivor payload bytes drift")
    source_authority = {
        "source_family": EXPECTED_SOURCE_FAMILY,
        "source_sha256": languk_parent.source_gate.SOURCE_SHA256,
        "source_report_identity_sha256": languk_parent.SOURCE_REPORT_IDENTITY,
        "selected_manifest_identity_sha256": languk_parent.EXPECTED_SELECTED_MANIFEST,
        "source_local_privacy_execution_identity_sha256": languk_parent.EXPECTED_PRIVACY_IDENTITY,
        "source_local_quality_execution_identity_sha256": languk_parent.EXPECTED_QUALITY_IDENTITY,
        "post_global_dedup_survivor_ids_sha256": EXPECTED_SURVIVOR_IDS_SHA256,
        "post_global_dedup_survivor_objects": len(rows),
        "post_global_dedup_survivor_bytes": total,
        "fresh_intake_receipt_identity_sha256": receipt["receipt_identity_sha256"],
    }
    return rows, payloads, source_authority


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
        records.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": EXPECTED_SOURCE_FAMILY,
                "modality": "uk",
                "text": text,
            }
        )
        inventory_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": EXPECTED_SOURCE_FAMILY,
                "modality": "uk",
                "payload_sha256": sha256(raw),
                "payload_bytes": len(raw),
            }
        )
    records.sort(key=lambda row: row["record_id"])
    inventory_rows.sort(key=lambda row: row["record_id"])
    require(len(records) == EXPECTED_SURVIVOR_OBJECTS, "training record count drift")
    require(
        sum(row["payload_bytes"] for row in inventory_rows) == EXPECTED_SURVIVOR_BYTES,
        "training inventory byte drift",
    )

    inventory_core = {
        "schema_version": "12-6.d03-languk-court-postdedup-inventory.v1",
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "record_count": EXPECTED_SURVIVOR_OBJECTS,
        "total_payload_bytes": EXPECTED_SURVIVOR_BYTES,
        "records": inventory_rows,
    }
    inventory = {
        **inventory_core,
        "inventory_identity_sha256": sha256(canonical(inventory_core)),
    }
    subset_core = {
        "schema_version": "12-6.d03-languk-court-postdedup-subset.v1",
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_combined_matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "source_object_count": EXPECTED_SURVIVOR_OBJECTS,
        "declared_capacity_bytes": EXPECTED_SURVIVOR_BYTES,
        "source_ids_sha256": EXPECTED_SURVIVOR_IDS_SHA256,
    }
    subset = {
        **subset_core,
        "subset_authority_sha256": sha256(canonical(subset_core)),
    }

    projection = reserved._record_projection(records)
    handoff_core = {
        "schema_version": reserved.TRAINING_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "input_survivor_authority_sha256": subset["subset_authority_sha256"],
        "retained_source_count": EXPECTED_SURVIVOR_OBJECTS,
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
    training_records_sha = clean._sha256(b"".join(clean._cjson(row) for row in records))
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


def execute(
    *,
    expected_execution_head: str,
    source_parquet: Path,
    parent_survivors_json: Path,
    parent_evidence_json: Path,
    parent_proof_json: Path,
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
    carrier_blob = verify_carrier()
    parent_survivors, _parent_evidence, survivor_ids = verify_parent_artifact(
        parent_survivors_json,
        parent_evidence_json,
        parent_proof_json,
    )
    source_rows, payloads, source_authority = acquire_languk_survivors(
        source_parquet=source_parquet,
        parent_survivors=parent_survivors,
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

    evaluation_records = _read_jsonl(evaluation_records_jsonl, "reserved evaluation records")
    base_reserved_binding = _read_json(base_reserved_binding_json, "base reserved binding")
    eval647_manifest = _read_json(eval647_manifest_json, "EVAL-647 manifest")
    eval647_evidence = _read_json(
        eval647_materialization_evidence_json,
        "EVAL-647 materialization evidence",
    )

    original = _patch_clean_release(
        count=EXPECTED_SURVIVOR_OBJECTS,
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
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
            expected_survivor_authority_sha256=subset["subset_authority_sha256"],
            expected_training_handoff_identity_sha256=handoff["handoff_identity_sha256"],
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

    final_bytes = int(receipt["survivor_payload_bytes"])
    later_gate_loss = EXPECTED_SURVIVOR_BYTES - final_bytes
    require(later_gate_loss >= 0, "later gates widened Lang-UK payload bytes")
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
        "execution_carrier_git_blob_sha1": carrier_blob,
        "parent": {
            "execution_head_sha": PARENT_EXECUTION_HEAD,
            "workflow_run_id": PARENT_WORKFLOW_RUN_ID,
            "artifact_id": PARENT_ARTIFACT_ID,
            "artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
            "proof_artifact_id": PARENT_PROOF_ARTIFACT_ID,
            "proof_artifact_zip_sha256": PARENT_PROOF_ZIP_SHA256,
            "matcher_report_sha256": PARENT_MATCHER_REPORT_SHA256,
            "survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY_SHA256,
            "two_clean_proof_identity_sha256": PARENT_PROOF_IDENTITY_SHA256,
            "post_global_dedup_languk_objects": EXPECTED_SURVIVOR_OBJECTS,
            "post_global_dedup_languk_bytes": EXPECTED_SURVIVOR_BYTES,
        },
        "languk_authority": {
            "source_family": EXPECTED_SOURCE_FAMILY,
            "source_object_count": EXPECTED_SURVIVOR_OBJECTS,
            "pre_gate_payload_bytes": EXPECTED_SURVIVOR_BYTES,
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "subset_authority_sha256": subset["subset_authority_sha256"],
            "training_records_sha256": training_records_sha,
            "training_handoff_raw_sha256": handoff_raw_sha,
            "training_handoff_identity_sha256": handoff["handoff_identity_sha256"],
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
            "composition_receipt_identity_sha256": receipt["receipt_identity_sha256"],
            "input_records": receipt["input_training_records"],
            "post_decontamination_records": receipt["post_decontamination_records"],
            "post_quality_records": receipt["post_quality_records"],
            "survivor_records": receipt["survivor_records"],
            "survivor_source_objects": receipt["survivor_source_objects"],
            "survivor_payload_bytes": final_bytes,
            "later_gate_loss_bytes": later_gate_loss,
            "rejection_counts": deepcopy(receipt["rejection_counts"]),
            "privacy_detector_counts": deepcopy(receipt["privacy_detector_counts"]),
            "survivor_jsonl_sha256": receipt["survivor_jsonl_sha256"],
            "survivor_record_inventory_digest_sha256": receipt[
                "survivor_record_inventory_digest_sha256"
            ],
            "survivor_payload_inventory_digest_sha256": receipt[
                "survivor_payload_inventory_digest_sha256"
            ],
        },
        "survivor_inventory": deepcopy(survivor_inventory),
        "authority_blobs": authority_blobs,
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "source_parquet_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "global_cross_source_dedup_complete_for_languk": True,
            "reserved_evaluation_decontamination_complete_for_languk": True,
            "canonical_quality_privacy_complete_for_languk": True,
            "late_registry_refresh_complete": False,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--parent-survivors-json", type=Path, required=True)
    parser.add_argument("--parent-evidence-json", type=Path, required=True)
    parser.add_argument("--parent-proof-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument("--eval647-manifest-json", type=Path, required=True)
    parser.add_argument("--eval647-materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--expected-base-reserved-binding-identity-sha256", required=True)
    parser.add_argument("--expected-composed-reserved-binding-identity-sha256", required=True)
    parser.add_argument("--expected-eval647-materialization-evidence-identity-sha256", required=True)
    parser.add_argument("--expected-eval647-object-set-identity-sha256", required=True)
    parser.add_argument("--expected-selection-validation-identity-sha256", required=True)
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    evidence = execute(
        expected_execution_head=args.expected_execution_head,
        source_parquet=args.source_parquet,
        parent_survivors_json=args.parent_survivors_json,
        parent_evidence_json=args.parent_evidence_json,
        parent_proof_json=args.parent_proof_json,
        evaluation_records_jsonl=args.evaluation_records_jsonl,
        base_reserved_binding_json=args.base_reserved_binding_json,
        eval647_manifest_json=args.eval647_manifest_json,
        eval647_materialization_evidence_json=args.eval647_materialization_evidence_json,
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
    gate = evidence["gate_execution"]
    print("D03_LANGUK_POSTDEDUP_CLEAN=PASS_ZERO_CREDIT")
    print(f"EVIDENCE_IDENTITY_SHA256={evidence['evidence_identity_sha256']}")
    print(f"SURVIVOR_RECORDS={gate['survivor_records']}")
    print(f"SURVIVOR_PAYLOAD_BYTES={gate['survivor_payload_bytes']}")
    print(f"LATER_GATE_LOSS_BYTES={gate['later_gate_loss_bytes']}")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
