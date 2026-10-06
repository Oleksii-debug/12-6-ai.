#!/usr/bin/env python3
"""Execute the physically qualified Lang-UK Supreme Court slice through incumbent global dedup.

Execution glue only. This runner re-materializes the exact pinned parquet, reproduces
source framing + current G06/G05 selection exactly, then delegates global duplicate
semantics to the incumbent independently qualified indexed executor. Durable outputs
are text-free and grant zero corpus, tokenizer, training, evaluation, or compute
authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import ssl
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
for location in (str(ROOT / "tools"), str(ROOT / "src")):
    if location not in sys.path:
        sys.path.insert(0, location)

import run_d03_franko1901_global_dedup_execution_v1 as incumbent
import run_d03_languk_court_privacy_retest_v1 as source_gate
import run_next100_065f_global_dedup_v8 as v8
from twelve_six.data import expanded_global_dedup_v9 as v9_semantics
from twelve_six.data import incumbent_dedup_indexed_execution as indexed

SCHEMA = "12-6.d03-languk-court-global-dedup-execution.v1"
SURVIVOR_SCHEMA = "12-6.d03-languk-court-global-dedup-survivors.v1"
PARENT_SOURCE_HEAD = "c0c39803063b44b87b5256ca8934345b3af0dc9e"
SOURCE_EXECUTION_RUN_ID = 37498170854
SOURCE_EXECUTION_JOB_ID = 112387976094
SOURCE_ARTIFACT_ID = 11428951409
SOURCE_ARTIFACT_ZIP_SHA256 = (
    "b4d5a23490d10ea6c00cdbc97ee5ab22b5ac972ef07296460df8a7feeee129ba"
)
SOURCE_REPORT_IDENTITY = (
    "0860d85ef0c04699d91d0c06ac2cae4aad810050bfb98446014fa980239ec338"
)
SOURCE_RUNNER_BLOB = "c0527b5b9e51c84549eaf757e274a02c677ab5b2"
EXPECTED_SOURCE_ROWS = 5_125
EXPECTED_SOURCE_CONTRACT_PASS = 4_995
EXPECTED_SOURCE_CONTRACT_DISPOSITIONS = {
    "accepted": 4_995,
    "annotation_contract_inconsistent": 130,
}
EXPECTED_PRIVACY_INPUT_ROOT = (
    "6eb48759fd628a5eb714408aa589c3d7e328d15f1748cf09afc2b137752ce5d7"
)
EXPECTED_PRIVACY_IDENTITY = (
    "f1c64058aaa9a2cc5be895f8d7a4eb644901b7d6bbcf2ae30868c94292af8109"
)
EXPECTED_PRIVACY_COUNTS = {
    "allow": 4_857,
    "exclude": 0,
    "quarantine": 0,
    "records": 4_995,
    "redact": 138,
}
EXPECTED_PRIVACY_DETECTORS = {"email": 11, "phone": 155, "sensitive_id": 23}
EXPECTED_SELECTED_MANIFEST = (
    "ea7452115377a85124b935377016e88269793dafb99a8aeec154ae741cc47586"
)
EXPECTED_QUALITY_INPUT_ROOT = (
    "0bd76945d40f8743f4f88fd3e9e69a62bd33ec776f78068047b9f0bd0d782946"
)
EXPECTED_QUALITY_IDENTITY = (
    "c1c2bbe6dc9ef0447a7ad4e9490c9bab867d823d98609e7e6fb722fb78741efc"
)
EXPECTED_QUALITY_COUNTS = {
    "accepted_quality_units": 256,
    "quality_units": 256,
    "records": 256,
    "reject_document": 0,
    "rejected_quality_units": 0,
    "retain_all": 256,
    "retain_partial": 0,
}
EXPECTED_QUALITY_BYTES = {
    "input_utf8_bytes": 2_880_510,
    "rejected_utf8_bytes": 0,
    "retained_utf8_bytes": 2_880_510,
}
EXPECTED_CANDIDATE_RECORDS = 256
EXPECTED_CANDIDATE_BYTES = 2_880_510
EXPECTED_BASE_OBJECTS = incumbent.EXPECTED_BASE_OBJECTS
EXPECTED_BASE_BYTES = incumbent.EXPECTED_BASE_BYTES
EXPECTED_COMBINED_OBJECTS = EXPECTED_BASE_OBJECTS + EXPECTED_CANDIDATE_RECORDS
EXPECTED_COMBINED_BYTES = EXPECTED_BASE_BYTES + EXPECTED_CANDIDATE_BYTES
CARRIER_PATH = "tools/run_d03_languk_court_global_dedup_v1.py"

AUTHORITY_PATHS = (
    "tools/run_d03_languk_court_privacy_retest_v1.py",
    "tools/retest_d03_languk_supreme_court.py",
    "configs/data/d03_languk_supreme_court_retest_v1.json",
    "src/twelve_six/data/document_quality.py",
    "src/twelve_six/data/quality_granularity.py",
    "src/twelve_six/data/quality_execution_authority.py",
    "src/twelve_six/data/privacy_execution_authority.py",
    "src/twelve_six/data/privacy_filter_v3.py",
    "src/twelve_six/data/expanded_global_dedup_v9.py",
    "src/twelve_six/data/_expanded_global_dedup_v9_impl.py",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "tools/run_d03_franko1901_global_dedup_execution_v1.py",
    "tools/run_next100_065f_global_dedup_v8.py",
    "configs/data/next100_065f_global_dedup_v8.json",
)


class LangUkGlobalDedupError(RuntimeError):
    """Fail-closed physical execution or authority mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LangUkGlobalDedupError(message)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_authority_paths() -> dict[str, str]:
    ancestor = incumbent._git(
        "merge-base", "--is-ancestor", PARENT_SOURCE_HEAD, "HEAD", check=False
    )
    require(ancestor.returncode == 0, "Lang-UK physical parent is not an ancestor")
    blobs: dict[str, str] = {}
    for authority_path in AUTHORITY_PATHS:
        parent_blob = incumbent._git(
            "rev-parse", f"{PARENT_SOURCE_HEAD}:{authority_path}"
        ).stdout.strip()
        head_blob = incumbent._git(
            "rev-parse", f"HEAD:{authority_path}"
        ).stdout.strip()
        worktree_blob = incumbent._git(
            "hash-object", str(ROOT / authority_path)
        ).stdout.strip()
        require(
            bool(parent_blob)
            and parent_blob == head_blob
            and head_blob == worktree_blob,
            f"authority path drift: {authority_path}",
        )
        blobs[authority_path] = head_blob
    require(
        blobs["tools/run_d03_languk_court_privacy_retest_v1.py"]
        == SOURCE_RUNNER_BLOB,
        "source runner blob drift",
    )
    incumbent.verify_repository_authority()
    return blobs


def verify_carrier() -> str:
    expected = incumbent._git("rev-parse", f"HEAD:{CARRIER_PATH}").stdout.strip()
    observed = incumbent._git("hash-object", str(ROOT / CARRIER_PATH)).stdout.strip()
    require(len(expected) == 40 and expected == observed, "execution carrier drift")
    return observed


def reconstruct_v8_with_bounded_tls_retry(
    *,
    v7_root: Path,
    bulk_workspace: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Replay exact V7 with the already-qualified bounded TLS-EOF transport retry."""

    original_capture = v8._capture_terminal_v7

    def capture_with_retry(
        historical_root: Path,
        historical_config: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, bytes]]:
        historical_v7 = v8._load_v7(historical_root)
        fetch_module = historical_v7.v6.v5.v1
        original_fetch = fetch_module.fetch_exact_source

        def retry_fetch(url: str) -> bytes:
            for attempt in range(1, 4):
                try:
                    return original_fetch(url)
                except OSError as exc:
                    retryable = (
                        isinstance(exc, URLError)
                        and isinstance(exc.reason, ssl.SSLEOFError)
                    )
                    if retryable and attempt < 3:
                        time.sleep(0.25 * attempt)
                        continue
                    try:
                        host = urlsplit(url).hostname or "unknown"
                    except ValueError:
                        host = "invalid-url"
                    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    exc.add_note(
                        "historical V7 source fetch failed: "
                        f"host={host}; acquisition_url_sha256={url_digest}; "
                        f"attempts={attempt}"
                    )
                    raise
            raise AssertionError("unreachable historical fetch attempt state")

        fetch_module.fetch_exact_source = retry_fetch
        try:
            return original_capture(historical_root, historical_config)
        finally:
            fetch_module.fetch_exact_source = original_fetch

    v8._capture_terminal_v7 = capture_with_retry
    try:
        return incumbent._reconstruct_v8_with_historical_namespace(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    finally:
        v8._capture_terminal_v7 = original_capture


def reproduce_source_candidate(
    source_parquet: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    source_rows, dispositions, source_total = source_gate.load_source(source_parquet)
    require(source_total == EXPECTED_SOURCE_ROWS, "source row count drift")
    require(
        len(source_rows) == EXPECTED_SOURCE_CONTRACT_PASS,
        "source-contract pass count drift",
    )
    require(
        dispositions == EXPECTED_SOURCE_CONTRACT_DISPOSITIONS,
        "source-contract disposition drift",
    )

    privacy_inputs = [
        {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
        for row in source_rows
    ]
    privacy_input_root = source_gate.input_rows_sha256(privacy_inputs)
    require(
        privacy_input_root == EXPECTED_PRIVACY_INPUT_ROOT,
        "privacy input identity drift",
    )
    privacy = source_gate.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
    )
    source_gate.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=privacy_input_root,
        expected_execution_identity_sha256=EXPECTED_PRIVACY_IDENTITY,
    )
    require(
        privacy["execution_identity_sha256"] == EXPECTED_PRIVACY_IDENTITY,
        "privacy execution identity drift",
    )
    require(privacy["counts"] == EXPECTED_PRIVACY_COUNTS, "privacy count drift")
    require(
        privacy["detector_counts"] == EXPECTED_PRIVACY_DETECTORS,
        "privacy detector-count drift",
    )
    privacy_by_id = {row["record_id"]: row for row in privacy["records"]}
    allow_rows = [
        row
        for row in source_rows
        if privacy_by_id[row["record_id"]]["action"] == "ALLOW"
    ]
    selected = allow_rows[: source_gate.MAX_RECORDS]
    require(
        len(selected) == EXPECTED_CANDIDATE_RECORDS,
        "selected Lang-UK record count drift",
    )

    selected_manifest_rows = [
        {
            "record_id": row["record_id"],
            "source_id_sha256": source_gate.sha256(row["source_id"].encode("utf-8")),
            "raw_payload_sha256": source_gate.sha256(row["raw_text"].encode("utf-8")),
            "normalized_payload_sha256": source_gate.sha256(
                row["normalized_text"].encode("utf-8")
            ),
            "normalized_payload_bytes": len(row["normalized_text"].encode("utf-8")),
        }
        for row in selected
    ]
    selected_manifest = {
        "dataset": source_gate.DATASET,
        "revision": source_gate.REVISION,
        "source_file": source_gate.SOURCE_FILE,
        "source_family": source_gate.FAMILY,
        "family_admitted": False,
        "max_records": source_gate.MAX_RECORDS,
        "selection_rule": "privacy_action_ALLOW_then_ascending_stable_source_id",
        "records": selected_manifest_rows,
    }
    selected_manifest_id = source_gate.sha256(source_gate.canonical(selected_manifest))
    require(
        selected_manifest_id == EXPECTED_SELECTED_MANIFEST,
        "selected-manifest identity drift",
    )

    quality_inputs = [
        {"id": row["record_id"], "text": row["normalized_text"], "mode": "uk"}
        for row in selected
    ]
    quality_input_root = source_gate._quality_input_rows_sha(quality_inputs)
    require(
        quality_input_root == EXPECTED_QUALITY_INPUT_ROOT,
        "quality input identity drift",
    )
    quality = source_gate.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=selected_manifest_id,
        expected_input_rows_sha256=quality_input_root,
    )
    source_gate.verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=selected_manifest_id,
        expected_input_rows_sha256=quality_input_root,
        expected_execution_identity_sha256=EXPECTED_QUALITY_IDENTITY,
    )
    require(
        quality["execution_identity_sha256"] == EXPECTED_QUALITY_IDENTITY,
        "quality execution identity drift",
    )
    require(quality["counts"] == EXPECTED_QUALITY_COUNTS, "quality count drift")
    require(quality["bytes"] == EXPECTED_QUALITY_BYTES, "quality byte drift")
    quality_by_id = {row["record_id"]: row for row in quality["records"]}
    require(
        set(quality_by_id) == {row["record_id"] for row in selected},
        "quality record coverage drift",
    )
    require(
        all(quality_by_id[row["record_id"]]["status"] == "RETAIN_ALL" for row in selected),
        "source-local G05 no longer retains every selected document",
    )

    extension_sources: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    projection_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for row in selected:
        source_id = row["source_id"]
        record_id = row["record_id"]
        payload = row["normalized_text"].encode("utf-8")
        digest = sha256(payload)
        byte_count = len(payload)
        require(
            quality_by_id[record_id]["payload_sha256"] == digest,
            "quality/source payload identity mismatch",
        )
        require(
            quality_by_id[record_id]["utf8_bytes"] == byte_count,
            "quality/source byte-count mismatch",
        )
        matcher_source_id = f"languk-supreme-2024:{record_id}"
        require(matcher_source_id not in payloads, "matcher source-id collision")
        stable_origin = (
            f"huggingface:{source_gate.DATASET}@{source_gate.REVISION}:"
            f"{source_gate.SOURCE_FILE}#id={source_id}"
        )
        matcher_row = {
            "source_id": matcher_source_id,
            "source_family": source_gate.FAMILY,
            "stable_origin_id": stable_origin,
            "stable_object_id": f"sha256:{digest}",
            "modality": "uk",
            "evidence_status": "DEDICATED_TERMINAL",
            "authority_ref": (
                f"PR2848:{PARENT_SOURCE_HEAD}:run:{SOURCE_EXECUTION_RUN_ID}:"
                f"job:{SOURCE_EXECUTION_JOB_ID}:artifact:{SOURCE_ARTIFACT_ID}"
            ),
            "declared_capacity_bytes": byte_count,
            "expected_raw_bytes": byte_count,
            "expected_raw_sha256": digest,
            "acquisition_url": (
                "https://huggingface.co/datasets/"
                + source_gate.DATASET
                + "/blob/"
                + source_gate.REVISION
                + "/"
                + quote(source_gate.SOURCE_FILE)
                + "#row-"
                + quote(source_id)
            ),
            "origin_key": (
                f"languk-court:{source_gate.REVISION}:"
                f"{source_gate.SOURCE_FILE}:source-id:{source_id}"
            ),
        }
        extension_sources.append(matcher_row)
        payloads[matcher_source_id] = payload
        projection_rows.append(copy.deepcopy(matcher_row))
        total_bytes += byte_count

    require(total_bytes == EXPECTED_CANDIDATE_BYTES, "candidate payload byte total drift")
    require(
        incumbent._declared_capacity_bytes(
            {"sources": extension_sources},
            payloads,
            label="Lang-UK projection",
        )
        == EXPECTED_CANDIDATE_BYTES,
        "Lang-UK declared-capacity drift",
    )
    receipt_core = {
        "schema_version": "12-6.d03-languk-court-dedup-intake-receipt.v1",
        "source_family": source_gate.FAMILY,
        "source_sha256": source_gate.SOURCE_SHA256,
        "source_report_identity_sha256": SOURCE_REPORT_IDENTITY,
        "selected_manifest_identity_sha256": selected_manifest_id,
        "privacy_execution_identity_sha256": EXPECTED_PRIVACY_IDENTITY,
        "quality_execution_identity_sha256": EXPECTED_QUALITY_IDENTITY,
        "record_count": EXPECTED_CANDIDATE_RECORDS,
        "payload_utf8_bytes": EXPECTED_CANDIDATE_BYTES,
        "matcher_projection_identity_sha256": sha256(canonical(projection_rows)),
        "source_execution": {
            "head_sha": PARENT_SOURCE_HEAD,
            "workflow_run_id": SOURCE_EXECUTION_RUN_ID,
            "job_id": SOURCE_EXECUTION_JOB_ID,
            "artifact_id": SOURCE_ARTIFACT_ID,
            "artifact_zip_sha256": SOURCE_ARTIFACT_ZIP_SHA256,
            "two_fresh_source_executions_converged": True,
        },
        "truth_boundary": {
            "family_admitted": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": sha256(canonical(receipt_core)),
    }
    return extension_sources, payloads, receipt


def compose_graph(
    base_inventory: Mapping[str, Any],
    base_payloads: Mapping[str, bytes],
    extension_sources: list[dict[str, Any]],
    extension_payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows = base_inventory.get("sources")
    require(type(rows) is list and bool(rows), "base source rows missing")
    base_ids = {row.get("source_id") for row in rows if type(row) is dict}
    extension_ids = {row.get("source_id") for row in extension_sources}
    require(
        len(base_ids) == len(rows) and None not in base_ids,
        "base source identities invalid",
    )
    require(base_ids == set(base_payloads), "base inventory/payload coverage mismatch")
    require(
        len(extension_ids) == len(extension_sources) and None not in extension_ids,
        "Lang-UK source identities invalid",
    )
    require(
        extension_ids == set(extension_payloads),
        "Lang-UK inventory/payload coverage mismatch",
    )
    require(not (base_ids & extension_ids), "Lang-UK source-id collision with incumbent")
    inventory = copy.deepcopy(dict(base_inventory))
    inventory["sources"] = [*copy.deepcopy(rows), *copy.deepcopy(extension_sources)]
    inventory["final_refresh_required"] = False
    inventory["terminal_refresh_rule"] = (
        "Exact reconstructed Nomis-free V8 graph plus the source-local G06/G05-qualified "
        "Lang-UK Supreme Court records; pair decisions delegate unchanged to terminal "
        "V3 semantics through the independently qualified indexed executor."
    )
    payloads = dict(base_payloads)
    payloads.update(extension_payloads)
    return inventory, payloads


def terminal(
    report: Mapping[str, Any],
    expected_objects: int,
    expected_bytes: int,
) -> Mapping[str, Any]:
    require(report.get("source_count") == expected_objects, "dedup source count drift")
    value = report.get("terminal_candidates")
    require(isinstance(value, Mapping), "terminal dedup summary missing")
    before = value.get("declared_capacity_bytes_before")
    after = value.get("conservative_unique_capacity_bytes_after")
    discount = value.get("duplicate_discount_bytes")
    require(type(before) is int and before == expected_bytes, "pre-dedup byte drift")
    require(type(after) is int and 0 < after <= before, "post-dedup byte invalid")
    require(
        type(discount) is int and discount >= 0 and before - after == discount,
        "dedup byte arithmetic drift",
    )
    return value


def validate_projection(
    report: Mapping[str, Any],
    projection: Mapping[str, Any],
    expected_objects: int,
    expected_bytes: int,
) -> None:
    value = terminal(report, expected_objects, expected_bytes)
    require(
        projection.get("schema_version") == v9_semantics.SURVIVOR_SCHEMA,
        "survivor projection schema drift",
    )
    core = dict(projection)
    claimed = core.pop("survivor_authority_sha256", None)
    require(
        type(claimed) is str and claimed == sha256(canonical(core)),
        "projection self-hash drift",
    )
    require(
        projection.get("matcher_report_sha256") == report.get("report_sha256"),
        "projection/report identity drift",
    )
    ids = projection.get("survivor_source_ids")
    require(
        type(ids) is list
        and all(type(item) is str and item for item in ids)
        and len(ids) == len(set(ids)),
        "survivor ids invalid",
    )
    require(
        projection.get("pre_dedup_source_object_count") == expected_objects,
        "projection pre-dedup object count drift",
    )
    require(
        projection.get("pre_dedup_declared_capacity_bytes") == expected_bytes,
        "projection pre-dedup byte count drift",
    )
    require(
        projection.get("post_dedup_declared_capacity_bytes")
        == value.get("conservative_unique_capacity_bytes_after"),
        "projection post-dedup byte count drift",
    )
    require(
        projection.get("post_dedup_survivor_source_object_count") == len(ids),
        "projection survivor count drift",
    )


def build_survivor_authority(
    extension_sources: list[dict[str, Any]],
    combined_report: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    marginal_unique_bytes: int,
) -> dict[str, Any]:
    extension_by_id = {row["source_id"]: row for row in extension_sources}
    require(
        len(extension_by_id) == EXPECTED_CANDIDATE_RECORDS,
        "Lang-UK extension identity cardinality drift",
    )
    survivor_ids = projection.get("survivor_source_ids")
    require(type(survivor_ids) is list, "combined survivor ids missing")
    selected_rows: list[dict[str, Any]] = []
    for source_id in survivor_ids:
        if source_id not in extension_by_id:
            continue
        row = extension_by_id[source_id]
        selected_rows.append(
            {
                "source_id": source_id,
                "source_family": row["source_family"],
                "stable_origin_id": row["stable_origin_id"],
                "stable_object_id": row["stable_object_id"],
                "declared_capacity_bytes": row["declared_capacity_bytes"],
                "expected_raw_sha256": row["expected_raw_sha256"],
                "origin_key": row["origin_key"],
            }
        )
    selected_payload_bytes = sum(
        row["declared_capacity_bytes"] for row in selected_rows
    )
    require(
        0 <= marginal_unique_bytes <= EXPECTED_CANDIDATE_BYTES,
        "Lang-UK marginal capacity invalid",
    )
    core = {
        "schema_version": SURVIVOR_SCHEMA,
        "matcher_report_sha256": combined_report.get("report_sha256"),
        "selection_projection_sha256": projection.get("survivor_authority_sha256"),
        "source_family": source_gate.FAMILY,
        "candidate_source_object_count": EXPECTED_CANDIDATE_RECORDS,
        "candidate_payload_utf8_bytes": EXPECTED_CANDIDATE_BYTES,
        "post_dedup_selected_extension_source_object_count": len(selected_rows),
        "post_dedup_selected_extension_payload_bytes": selected_payload_bytes,
        "marginal_global_unique_capacity_bytes": marginal_unique_bytes,
        "survivors": selected_rows,
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "post_dedup_quality_privacy_complete": False,
            "late_registry_refresh_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    return {**core, "survivor_authority_sha256": sha256(canonical(core))}


def execute(
    *,
    source_parquet: Path,
    v7_root: Path,
    bulk_workspace: Path,
    output_base_report: Path,
    output_combined_report: Path,
    output_survivors: Path,
    output_evidence: Path,
    expected_execution_head: str,
    max_candidate_pairs: int,
    max_index_postings: int,
    max_pair_expansions: int,
) -> dict[str, Any]:
    execution_head = incumbent._bind_execution_head(expected_execution_head)
    authority_blobs = verify_authority_paths()
    carrier_blob = verify_carrier()
    bulk_workspace = incumbent._prepare_empty_bulk_workspace(bulk_workspace)
    verified_v7_head = incumbent._verify_v7_worktree(v7_root)
    config = v8.load_config(ROOT / "configs/data/next100_065f_global_dedup_v8.json")
    matcher, base_inventory, base_payloads, removal = (
        reconstruct_v8_with_bounded_tls_retry(
            v7_root=v7_root,
            bulk_workspace=bulk_workspace,
            config=config,
        )
    )
    require(
        incumbent._verify_v7_worktree(v7_root) == verified_v7_head,
        "V7 worktree drifted during reconstruction",
    )
    require(len(base_payloads) == EXPECTED_BASE_OBJECTS, "base object-count drift")
    require(
        incumbent._declared_capacity_bytes(
            base_inventory, base_payloads, label="Lang-UK execution base"
        )
        == EXPECTED_BASE_BYTES,
        "base declared-capacity drift",
    )

    extension_sources, extension_payloads, intake_receipt = reproduce_source_candidate(
        source_parquet
    )
    inventory, payloads = compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined object-count drift")
    require(
        incumbent._declared_capacity_bytes(inventory, payloads, label="combined graph")
        == EXPECTED_COMBINED_BYTES,
        "combined declared-capacity drift",
    )

    indexed.attest_incumbent_runtime(matcher)
    base_report = indexed.audit_payloads_indexed(
        matcher,
        base_inventory,
        base_payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(base_report)
    base_terminal = terminal(base_report, EXPECTED_BASE_OBJECTS, EXPECTED_BASE_BYTES)
    base_report_sha256 = base_report.get("report_sha256")
    require(
        type(base_report_sha256) is str and len(base_report_sha256) == 64,
        "base report identity invalid",
    )

    indexed.attest_incumbent_runtime(matcher)
    combined_report = indexed.audit_payloads_indexed(
        matcher,
        inventory,
        payloads,
        max_candidate_pairs=max_candidate_pairs,
        max_index_postings=max_index_postings,
        max_pair_expansions=max_pair_expansions,
    )
    matcher.verify_report(combined_report)
    combined_terminal = terminal(
        combined_report,
        EXPECTED_COMBINED_OBJECTS,
        EXPECTED_COMBINED_BYTES,
    )
    projection = v9_semantics._derive_survivors(combined_report)
    validate_projection(
        combined_report,
        projection,
        EXPECTED_COMBINED_OBJECTS,
        EXPECTED_COMBINED_BYTES,
    )
    marginal = (
        combined_terminal["conservative_unique_capacity_bytes_after"]
        - base_terminal["conservative_unique_capacity_bytes_after"]
    )
    require(
        0 <= marginal <= EXPECTED_CANDIDATE_BYTES,
        "Lang-UK marginal unique capacity invalid",
    )
    survivors = build_survivor_authority(
        extension_sources,
        combined_report,
        projection,
        marginal_unique_bytes=marginal,
    )

    evidence_core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "parent_source_head_sha": PARENT_SOURCE_HEAD,
        "execution_carrier_git_blob_sha1": carrier_blob,
        "authority_path_blobs": authority_blobs,
        "v7_head_sha": verified_v7_head,
        "base": {
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "declared_capacity_bytes": EXPECTED_BASE_BYTES,
            "report_sha256": base_report_sha256,
            "post_dedup_unique_bytes": base_terminal[
                "conservative_unique_capacity_bytes_after"
            ],
            "nomis1864_deauthorization": removal,
        },
        "languk_supreme_court": {
            "source_family": source_gate.FAMILY,
            "source_sha256": source_gate.SOURCE_SHA256,
            "source_report_identity_sha256": SOURCE_REPORT_IDENTITY,
            "selected_manifest_identity_sha256": EXPECTED_SELECTED_MANIFEST,
            "privacy_execution_identity_sha256": EXPECTED_PRIVACY_IDENTITY,
            "quality_execution_identity_sha256": EXPECTED_QUALITY_IDENTITY,
            "candidate_source_object_count": EXPECTED_CANDIDATE_RECORDS,
            "candidate_payload_utf8_bytes": EXPECTED_CANDIDATE_BYTES,
            "matcher_projection_identity_sha256": intake_receipt[
                "matcher_projection_identity_sha256"
            ],
            "intake_receipt_identity_sha256": intake_receipt[
                "receipt_identity_sha256"
            ],
            "post_dedup_selected_extension_source_object_count": survivors[
                "post_dedup_selected_extension_source_object_count"
            ],
            "post_dedup_selected_extension_payload_bytes": survivors[
                "post_dedup_selected_extension_payload_bytes"
            ],
            "marginal_global_unique_capacity_bytes": marginal,
            "global_dedup_loss_vs_source_local_candidate_bytes": (
                EXPECTED_CANDIDATE_BYTES - marginal
            ),
            "source_execution": copy.deepcopy(intake_receipt["source_execution"]),
        },
        "combined": {
            "source_object_count": EXPECTED_COMBINED_OBJECTS,
            "declared_capacity_bytes": EXPECTED_COMBINED_BYTES,
            "report_sha256": combined_report.get("report_sha256"),
            "post_dedup_unique_bytes": combined_terminal[
                "conservative_unique_capacity_bytes_after"
            ],
        },
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "content_boundary": {
            "source_parquet_uploaded": False,
            "source_text_emitted_in_evidence": False,
            "dedup_reports_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "global_dedup_execution_complete": True,
            "reserved_evaluation_decontamination_complete": False,
            "post_dedup_quality_privacy_complete": False,
            "late_registry_refresh_complete": False,
            "family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "scale_promotion_authorized": False,
        },
    }
    evidence = {
        **evidence_core,
        "evidence_identity_sha256": sha256(canonical(evidence_core)),
    }
    incumbent._publish_json_outputs(
        (
            (output_base_report, dict(base_report)),
            (output_combined_report, dict(combined_report)),
            (output_survivors, survivors),
            (output_evidence, evidence),
        )
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--output-base-report", type=Path, required=True)
    parser.add_argument("--output-combined-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument(
        "--max-index-postings",
        type=int,
        default=indexed.DEFAULT_MAX_INDEX_POSTINGS,
    )
    parser.add_argument(
        "--max-pair-expansions",
        type=int,
        default=indexed.DEFAULT_MAX_PAIR_EXPANSIONS,
    )
    args = parser.parse_args()
    evidence = execute(
        source_parquet=args.source_parquet,
        v7_root=args.v7_root,
        bulk_workspace=args.bulk_workspace,
        output_base_report=args.output_base_report,
        output_combined_report=args.output_combined_report,
        output_survivors=args.output_survivors,
        output_evidence=args.output_evidence,
        expected_execution_head=args.expected_execution_head,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    source = evidence["languk_supreme_court"]
    print("D03_LANGUK_COURT_GLOBAL_DEDUP_EXECUTION=PASS_ZERO_CREDIT")
    print("EVIDENCE_IDENTITY_SHA256=" + evidence["evidence_identity_sha256"])
    print(
        "MARGINAL_GLOBAL_UNIQUE_CAPACITY_BYTES="
        + str(source["marginal_global_unique_capacity_bytes"])
    )
    print("SURVIVOR_AUTHORITY_SHA256=" + evidence["survivor_authority_sha256"])
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
