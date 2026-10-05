#!/usr/bin/env python3
"""Replay terminal #2746, bind an exact training handoff, and execute DATA-232 -> G05 -> G06.

This is an execution-only carrier. Raw training/evaluation payloads stay ephemeral. Durable
outputs are hash/count receipts only and grant no corpus, tokenizer, optimizer, training,
final-test-outcome, or paid-compute authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tools.run_d03_code4_rich_fastapi_reserve_global_dedup_v1 as reserve
from twelve_six.data import current_clean_execution_v1 as clean
from twelve_six.data.current_reserved_decontamination_v1 import (
    _record_projection,
    build_reserved_payload_binding,
)
from twelve_six.data.eval233_final_test_resolver_v1 import (
    DATA232_FINAL_TEST_ID,
    resolve_eval233_final_test,
)
from twelve_six.data.eval303_selection_payload_resolver_v1 import (
    EVAL303_SELECTION_ID,
    resolve_eval303_selection_payloads_from_reconstructed,
)
from twelve_six.data.eval647_future_training_exclusion_v1 import (
    build_eval647_auxiliary_reserved_set,
    compose_eval647_future_training_exclusion,
)
from twelve_six.data.eval647_reserved_decontamination_v1 import (
    execute_eval647_reserved_decontamination,
    verify_eval647_reserved_decontamination_receipt,
)

PARENT_2746_HEAD = "b3d6fbef0c9cc1f0fbd4340e5eb7376fce01261a"
PARENT_2746_RUN = 37283627767
PARENT_COMBINED_REPORT = "697cafac8044229d3805df1a352af2a6219037b1e7b817c170165e9083758456"
PARENT_SURVIVOR_AUTHORITY = "e3c49cbe00c9b076914b9199be0293ba789d248cafde2d5a3e8269520068b050"
PARENT_TWO_CLEAN_PROOF = "2f1e2de990ac481947c8f416e87da442d0d6a1ec4db9eebf97cd090b62e11c93"
PARENT_RUNNER_BLOB = "5349eacdbeafe357000c3411d5997354bd58b140"
EXPECTED_BASE_BINDING = "1bffbf426fa88a26996ace524f83a2f10e301b9e0ba3172acfb92c07fba4b6e9"
EXPECTED_COMPOSED_BINDING = "c188aae751560f2d70504314fae0a46c990328c2920c1fa32564f34b6405fc0e"
EXPECTED_EVAL647_EVIDENCE = "3401db10bad35fd1c6fac2839413fc6afffac58fa5f2135d2202b944bc2fda82"
EXPECTED_EVAL647_OBJECT_SET = "0557410622403b411ac9d6d8fb01a57ef461617c19d06f112aa61dd57f670048"
CODE_TARGET_BYTES = 4_000_000
CURRENT_POST_QP_CODE_BYTES_REFERENCE = 3_664_247

PREFLIGHT_SCHEMA = "12-6.d03-code4-rich-fastapi-postdedup-preflight.v1"
EXECUTION_SCHEMA = "12-6.d03-code4-rich-fastapi-postqp-execution.v1"
INVENTORY_SCHEMA = "12-6.d03-code4-rich-fastapi-postdedup-inventory.v1"
HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_BLOBS = {
    "tools/run_d03_code4_rich_fastapi_reserve_global_dedup_v1.py": PARENT_RUNNER_BLOB,
    "src/twelve_six/data/current_reserved_decontamination_v1.py": "e5c555e3cd27844e98d4ae91af0b746e427f36c9",
    "src/twelve_six/data/eval647_reserved_decontamination_v1.py": "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    "src/twelve_six/data/eval647_future_training_exclusion_v1.py": "5516577a0720150a7ec12c1bf8898972968e6970",
    "src/twelve_six/data/eval303_selection_payload_resolver_v1.py": "659cb17fdba903d9c50c1dfc5becf046f523e5ee",
    "src/twelve_six/data/eval233_final_test_resolver_v1.py": "71dd98204c588bd0bb67b7a7b3ddc5ae8aa87c00",
    "src/twelve_six/data/current_clean_execution_v1.py": "b82ed11626a267dfffa16acd79e45c0cf3e6750b",
    "src/twelve_six/data/quality_execution_authority.py": "4659a9d4aba49908f372250904a54361c8d8cf46",
    "src/twelve_six/data/privacy_execution_authority.py": "9215287e81c0a82f05ec8405dc4f34c60313c193",
    "src/twelve_six/data/post_g05_g06_materialization_v1.py": "830087f91d1fa24385c5cbc8d2f687e4cc46b419",
    "src/twelve_six/data/privacy_filter_v3.py": "bcc5938395724f6728ab212f98b39f2334b0f37d",
    "src/twelve_six/data/decontamination_authority_v2.py": "c6e72011cd5add28bff7ed879ff6d32a7eb5ef33",
    "src/twelve_six/data/_data232_decontamination_matching.py": "5f57807478f8098f224a57f1e05ef5719cb30d35",
    "tools/materialize_eval_code_reserve_v1.py": "13000d9b911bf0e0c7567613e68f2b9b113bd955",
    "configs/evaluation/eval_code_reserve_v1.json": "89af932a1b0e2dae30b08f766505c8dba9d3e39f",
    "evidence/eval647/code_selection_source_materialization_v1.json": "efa5b77275919fa9417e35efb30c8c9a527bd960",
    "data/evaluation/eval303/selection-validation/composite-membership.jsonl": "04cf8b3fc1857f0d7b51e581aab8e4dc68f7f47d",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    require(type(value) is dict, f"{path} must contain an object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(dict(value)) + b"\n")


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_checkout(expected_execution_head: str) -> dict[str, str]:
    require(
        _HEX40.fullmatch(expected_execution_head) is not None,
        "expected execution head malformed",
    )
    require(
        git("rev-parse", "HEAD") == expected_execution_head,
        "checkout is not exact execution head",
    )
    require(
        not git("status", "--porcelain=v1", "--untracked-files=all"),
        "execution checkout is dirty",
    )
    merge_base = git("merge-base", PARENT_2746_HEAD, expected_execution_head)
    require(
        merge_base == PARENT_2746_HEAD,
        "terminal #2746 is not an ancestor of execution head",
    )
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_BLOBS.items():
        blob = git("rev-parse", f"{expected_execution_head}:{path}")
        require(blob == expected, f"Git blob drift: {path}")
        observed[path] = blob
    clean.verify_dependency_blobs()
    return dict(sorted(observed.items()))


def replay_parent(
    v7_root: Path,
    bulk_workspace: Path,
    execution_head: str,
) -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
]:
    captured: dict[str, Any] = {}
    original = reserve.compose_reserve_graph

    def capture_graph(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        inventory, payloads = original(*args, **kwargs)
        captured["inventory"] = copy.deepcopy(inventory)
        captured["payloads"] = dict(payloads)
        return inventory, payloads

    reserve.compose_reserve_graph = capture_graph
    try:
        with tempfile.TemporaryDirectory(prefix="d03-2746-replay-") as td:
            root = Path(td)
            evidence = reserve.execute(
                v7_root=v7_root,
                bulk_workspace=bulk_workspace,
                expected_execution_head=execution_head,
                output_code4_report=root / "code4.json",
                output_combined_report=root / "combined.json",
                output_survivors=root / "survivors.json",
                output_evidence=root / "evidence.json",
                max_candidate_pairs=5_000_000,
                max_index_postings=(
                    reserve.code4.incumbent.indexed.DEFAULT_MAX_INDEX_POSTINGS
                ),
                max_pair_expansions=(
                    reserve.code4.incumbent.indexed.DEFAULT_MAX_PAIR_EXPANSIONS
                ),
            )
            combined = load_json(root / "combined.json")
            survivors = load_json(root / "survivors.json")
            evidence_file = load_json(root / "evidence.json")
            require(
                evidence_file == evidence,
                "parent replay evidence publication drift",
            )
    finally:
        reserve.compose_reserve_graph = original

    require(
        combined.get("report_sha256") == PARENT_COMBINED_REPORT,
        "#2746 combined matcher report drift",
    )
    require(
        survivors.get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY,
        "#2746 survivor authority drift",
    )
    require(
        evidence_file.get("combined", {}).get("survivor_authority_sha256")
        == PARENT_SURVIVOR_AUTHORITY,
        "#2746 evidence survivor drift",
    )
    require(
        evidence_file.get("truth_boundary", {}).get(
            "reserved_evaluation_decontamination_complete"
        )
        is False,
        "#2746 truth boundary unexpectedly changed",
    )
    require(
        "inventory" in captured and "payloads" in captured,
        "combined graph capture missing",
    )
    return (
        captured["inventory"],
        captured["payloads"],
        survivors,
        evidence_file,
    )


def build_training_handoff(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    survivors: Mapping[str, Any],
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
]:
    rows = inventory.get("sources")
    require(
        isinstance(rows, list),
        "combined inventory sources missing",
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "combined source row malformed")
        source_id = row.get("source_id")
        require(
            isinstance(source_id, str)
            and source_id
            and source_id not in by_id,
            "combined source_id invalid/duplicate",
        )
        by_id[source_id] = row

    survivor_ids = survivors.get("survivor_source_ids")
    require(
        isinstance(survivor_ids, list) and survivor_ids,
        "survivor source IDs missing",
    )
    records: list[dict[str, str]] = []
    family_bytes: Counter[str] = Counter()
    modality_bytes: Counter[str] = Counter()
    physical_bytes = 0

    for source_id in sorted(survivor_ids):
        source = by_id.get(source_id)
        payload = payloads.get(source_id)
        require(
            source is not None and isinstance(payload, bytes),
            f"survivor payload missing: {source_id}",
        )
        expected_bytes = source.get("declared_capacity_bytes")
        require(
            type(expected_bytes) is int
            and len(payload) == expected_bytes,
            f"survivor byte accounting drift: {source_id}",
        )
        text = payload.decode("utf-8", errors="strict")
        family = source.get("source_family")
        modality = source.get("modality")
        require(
            isinstance(family, str) and family,
            f"survivor family missing: {source_id}",
        )
        require(
            modality in {"uk", "ua", "en", "code", "text"},
            f"survivor modality unsupported: {source_id}",
        )
        record_id = "d03-" + sha256(source_id.encode("utf-8"))
        records.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "source_family": family,
                "modality": str(modality),
                "text": text,
            }
        )
        family_bytes[family] += len(payload)
        modality_bytes[str(modality)] += len(payload)
        physical_bytes += len(payload)

    require(
        len(records) == len(set(row["record_id"] for row in records)),
        "derived record_id collision",
    )
    expected_unique = survivors.get("post_dedup_declared_capacity_bytes")
    require(
        type(expected_unique) is int
        and physical_bytes == expected_unique,
        "physical survivor bytes differ from #2746 survivor authority",
    )

    projection = _record_projection(records)
    projection_sha = sha256(canonical(projection))
    inventory_core = {
        "schema_version": INVENTORY_SCHEMA,
        "parent_execution_head_sha": PARENT_2746_HEAD,
        "parent_run_id": PARENT_2746_RUN,
        "parent_combined_report_sha256": PARENT_COMBINED_REPORT,
        "parent_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY,
        "parent_two_clean_proof_identity_sha256": PARENT_TWO_CLEAN_PROOF,
        "retained_source_count": len(records),
        "retained_payload_bytes": physical_bytes,
        "matcher_input_projection_sha256": projection_sha,
        "family_payload_bytes": dict(sorted(family_bytes.items())),
        "modality_payload_bytes": dict(sorted(modality_bytes.items())),
        "raw_text_persisted": False,
    }
    inventory_identity = sha256(canonical(inventory_core))
    handoff: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": inventory_identity,
        "input_survivor_authority_sha256": PARENT_SURVIVOR_AUTHORITY,
        "retained_source_count": len(records),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": projection_sha,
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff["handoff_identity_sha256"] = sha256(canonical(handoff))
    return (
        records,
        handoff,
        {
            **inventory_core,
            "inventory_identity_sha256": inventory_identity,
        },
    )


def build_preflight(
    execution_head: str,
    dependency_blobs: Mapping[str, str],
    handoff: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "POSTDEDUP_HANDOFF_BOUND_PENDING_RESERVED_EVALUATION",
        "execution_head_sha": execution_head,
        "execution_profile": "LOCAL_FREE",
        "terminal_parent_head_sha": PARENT_2746_HEAD,
        "terminal_parent_run_id": PARENT_2746_RUN,
        "terminal_parent_combined_report_sha256": PARENT_COMBINED_REPORT,
        "terminal_parent_survivor_authority_sha256": (
            PARENT_SURVIVOR_AUTHORITY
        ),
        "terminal_parent_two_clean_proof_identity_sha256": (
            PARENT_TWO_CLEAN_PROOF
        ),
        "training_handoff_identity_sha256": (
            handoff["handoff_identity_sha256"]
        ),
        "postdedup_inventory_identity_sha256": (
            inventory["inventory_identity_sha256"]
        ),
        "retained_source_count": inventory["retained_source_count"],
        "retained_payload_bytes": inventory["retained_payload_bytes"],
        "matcher_input_projection_sha256": (
            inventory["matcher_input_projection_sha256"]
        ),
        "family_payload_bytes": inventory["family_payload_bytes"],
        "modality_payload_bytes": inventory["modality_payload_bytes"],
        "dependency_git_blobs": dict(dependency_blobs),
        "reserved_evaluation_payload_accessed": False,
        "durable_evidence_hash_only": True,
        "canonical_capacity_credit_bytes": 0,
        "tokenizer_fit_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt["receipt_identity_sha256"] = sha256(canonical(receipt))
    return receipt


def load_eval647_materializer() -> Any:
    path = ROOT / "tools/materialize_eval_code_reserve_v1.py"
    spec = importlib.util.spec_from_file_location(
        "d03_eval647_source_materializer",
        path,
    )
    require(
        spec is not None and spec.loader is not None,
        "cannot load EVAL-647 source materializer",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_reserved_universe(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
]:
    selection_rows, selection_set, _ = (
        resolve_eval303_selection_payloads_from_reconstructed(
            eval290_data_jsonl=args.eval290_data_jsonl,
            eval290_manifest_json=args.eval290_manifest_json,
            eval291_data_jsonl=args.eval291_data_jsonl,
            eval291_authority_json=args.eval291_authority_json,
            eval303_membership_jsonl=(
                ROOT
                / "data/evaluation/eval303/selection-validation/"
                "composite-membership.jsonl"
            ),
        )
    )
    final_rows, final_set, _ = resolve_eval233_final_test(ROOT)
    require(
        EVAL303_SELECTION_ID
        == args.expected_selection_validation_identity_sha256,
        "selection-validation identity drift",
    )
    require(
        DATA232_FINAL_TEST_ID == args.expected_final_test_identity_sha256,
        "final-test identity drift",
    )

    manifest = load_json(
        ROOT / "configs/evaluation/eval_code_reserve_v1.json"
    )
    evidence = load_json(
        ROOT
        / "evidence/eval647/"
        "code_selection_source_materialization_v1.json"
    )
    require(
        evidence.get("evidence_identity_sha256")
        == EXPECTED_EVAL647_EVIDENCE,
        "EVAL-647 evidence identity drift",
    )
    require(
        evidence.get("object_set_identity_sha256")
        == EXPECTED_EVAL647_OBJECT_SET,
        "EVAL-647 object-set identity drift",
    )

    materializer = load_eval647_materializer()
    observed = materializer.materialize(manifest, timeout=60)
    require(
        observed.get("object_set_identity_sha256")
        == EXPECTED_EVAL647_OBJECT_SET,
        "fresh EVAL-647 object set drift",
    )
    auxiliary = build_eval647_auxiliary_reserved_set(manifest, evidence)
    members = {
        member["source_id"]: member
        for member in auxiliary["members"]
    }
    eval647_rows: list[dict[str, str]] = []
    for obj in manifest["objects"]:
        payload = materializer._fetch(
            materializer._raw_url(obj),
            60,
        )
        require(
            len(payload) == obj["expected_raw_bytes"],
            "EVAL-647 raw byte count drift",
        )
        require(
            sha256(payload) == obj["raw_sha256"],
            "EVAL-647 raw SHA drift",
        )
        source_id = (
            f"{obj['repository']}@{obj['revision']}:{obj['path']}"
        )
        member = members.get(source_id)
        require(
            isinstance(member, Mapping)
            and member.get("content_sha256") == obj["raw_sha256"],
            "EVAL-647 member binding drift",
        )
        eval647_rows.append(
            {
                "record_id": member["record_id"],
                "source_id": source_id,
                "source_family": member["source_family"],
                "modality": "code",
                "text": payload.decode("utf-8", errors="strict"),
            }
        )

    base_binding = build_reserved_payload_binding(
        [selection_set, final_set]
    )
    require(
        base_binding["binding_identity_sha256"]
        == EXPECTED_BASE_BINDING,
        "base reserved binding drift",
    )
    composed_binding, _ = compose_eval647_future_training_exclusion(
        base_binding,
        manifest,
        evidence,
    )
    require(
        composed_binding["binding_identity_sha256"]
        == EXPECTED_COMPOSED_BINDING,
        "composed reserved binding drift",
    )
    evaluation_rows = [
        *selection_rows,
        *final_rows,
        *eval647_rows,
    ]
    expected_members = sum(
        len(item["members"])
        for item in composed_binding["reserved_sets"]
    )
    require(
        len(evaluation_rows) == expected_members == 28,
        "reserved evaluation coverage drift",
    )
    return (
        evaluation_rows,
        base_binding,
        {
            "manifest": manifest,
            "evidence": evidence,
            "composed": composed_binding,
        },
    )


def execute_postqp(
    args: argparse.Namespace,
    execution_head: str,
    dependency_blobs: Mapping[str, str],
    training_records: Sequence[Mapping[str, Any]],
    handoff: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        _HEX64.fullmatch(
            args.expected_training_handoff_identity_sha256 or ""
        )
        is not None,
        "expected handoff identity must be independently pinned",
    )
    require(
        handoff["handoff_identity_sha256"]
        == args.expected_training_handoff_identity_sha256,
        "preflight-pinned training handoff identity drift",
    )

    evaluation_rows, base_binding, eval647 = (
        resolve_reserved_universe(args)
    )
    report, decontam, eval647_receipt = (
        execute_eval647_reserved_decontamination(
            training_records,
            evaluation_rows,
            training_handoff_evidence=handoff,
            base_reserved_binding=base_binding,
            manifest=eval647["manifest"],
            materialization_evidence=eval647["evidence"],
            expected_base_reserved_binding_identity_sha256=(
                EXPECTED_BASE_BINDING
            ),
            expected_composed_reserved_binding_identity_sha256=(
                EXPECTED_COMPOSED_BINDING
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                EXPECTED_EVAL647_EVIDENCE
            ),
            expected_eval647_object_set_identity_sha256=(
                EXPECTED_EVAL647_OBJECT_SET
            ),
            expected_inventory_identity_sha256=(
                inventory["inventory_identity_sha256"]
            ),
            expected_survivor_authority_sha256=(
                PARENT_SURVIVOR_AUTHORITY
            ),
            expected_training_handoff_identity_sha256=(
                handoff["handoff_identity_sha256"]
            ),
            expected_selection_validation_identity_sha256=(
                args.expected_selection_validation_identity_sha256
            ),
            expected_final_test_identity_sha256=(
                args.expected_final_test_identity_sha256
            ),
        )
    )
    verify_eval647_reserved_decontamination_receipt(
        eval647_receipt,
        expected_composed_reserved_binding_identity_sha256=(
            EXPECTED_COMPOSED_BINDING
        ),
        expected_eval647_materialization_evidence_identity_sha256=(
            EXPECTED_EVAL647_EVIDENCE
        ),
        expected_eval647_object_set_identity_sha256=(
            EXPECTED_EVAL647_OBJECT_SET
        ),
    )

    quality_inputs, metadata, decontam_excluded = (
        clean._post_decontamination_records(
            training_records,
            report,
        )
    )
    post_decontam_rows_sha = sha256(
        clean._cjson(clean._input_projection(quality_inputs))
    )
    decontam_identity = decontam["execution_identity_sha256"]
    quality = clean.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha,
    )
    quality_identity = quality["execution_identity_sha256"]
    clean.verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha,
        expected_execution_identity_sha256=quality_identity,
    )
    quality_survivors, quality_stats = (
        clean._materialize_quality_survivors(
            quality_inputs,
            metadata,
            quality,
        )
    )

    privacy_inputs = clean._quality_records_for_privacy(
        quality_survivors
    )
    post_quality_rows_sha = sha256(
        clean._cjson(clean._input_projection(privacy_inputs))
    )
    privacy = clean.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha,
    )
    privacy_identity = privacy["execution_identity_sha256"]
    clean.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha,
        expected_execution_identity_sha256=privacy_identity,
    )
    final_survivors, privacy_stats = (
        clean._materialize_privacy_survivors(
            quality_survivors,
            privacy,
        )
    )
    final_inventory = clean.materialize_record_inventory(
        final_survivors
    )

    family_bytes: Counter[str] = Counter()
    modality_bytes: Counter[str] = Counter()
    for row in final_survivors:
        size = len(row["normalized_payload"].encode("utf-8"))
        family_bytes[row["family"]] += size
        modality_bytes[row["modality"]] += size

    post_qp_code_bytes = modality_bytes.get("code", 0)
    headroom = post_qp_code_bytes - CODE_TARGET_BYTES
    next_gate = (
        "EXTEND_TRUSTED_FAMILY_AUTHORITY_THEN_BALANCE"
        if headroom >= 0
        else "SOURCE_CAPACITY_STILL_REQUIRED_AFTER_QP"
    )
    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": (
            "POST_QP_EXECUTED_ZERO_CREDIT_"
            "PENDING_FAMILY_AUTHORITY_AND_BALANCE"
        ),
        "execution_head_sha": execution_head,
        "execution_profile": "LOCAL_FREE",
        "terminal_parent_head_sha": PARENT_2746_HEAD,
        "terminal_parent_run_id": PARENT_2746_RUN,
        "terminal_parent_combined_report_sha256": (
            PARENT_COMBINED_REPORT
        ),
        "terminal_parent_survivor_authority_sha256": (
            PARENT_SURVIVOR_AUTHORITY
        ),
        "training_handoff_identity_sha256": (
            handoff["handoff_identity_sha256"]
        ),
        "postdedup_inventory_identity_sha256": (
            inventory["inventory_identity_sha256"]
        ),
        "base_reserved_binding_identity_sha256": (
            EXPECTED_BASE_BINDING
        ),
        "composed_reserved_binding_identity_sha256": (
            EXPECTED_COMPOSED_BINDING
        ),
        "eval647_materialization_evidence_identity_sha256": (
            EXPECTED_EVAL647_EVIDENCE
        ),
        "eval647_object_set_identity_sha256": (
            EXPECTED_EVAL647_OBJECT_SET
        ),
        "data232_report_sha256": report["report_sha256"],
        "decontamination_execution_identity_sha256": (
            decontam_identity
        ),
        "eval647_execution_receipt_identity_sha256": (
            eval647_receipt["receipt_identity_sha256"]
        ),
        "quality_execution_identity_sha256": quality_identity,
        "privacy_execution_identity_sha256": privacy_identity,
        "post_decontamination_input_rows_sha256": (
            post_decontam_rows_sha
        ),
        "post_quality_input_rows_sha256": post_quality_rows_sha,
        "input_records": len(training_records),
        "post_decontamination_records": len(quality_inputs),
        "post_quality_records": len(quality_survivors),
        "post_qp_records": len(final_survivors),
        "post_qp_payload_bytes": (
            final_inventory["total_payload_bytes"]
        ),
        "post_qp_record_inventory_digest_sha256": (
            final_inventory["record_inventory_digest_sha256"]
        ),
        "post_qp_payload_inventory_digest_sha256": (
            final_inventory["payload_inventory_digest_sha256"]
        ),
        "post_qp_family_payload_bytes": (
            dict(sorted(family_bytes.items()))
        ),
        "post_qp_modality_payload_bytes": (
            dict(sorted(modality_bytes.items()))
        ),
        "post_qp_code_bytes": post_qp_code_bytes,
        "code_target_bytes": CODE_TARGET_BYTES,
        "current_post_qp_code_bytes_reference": (
            CURRENT_POST_QP_CODE_BYTES_REFERENCE
        ),
        "post_qp_code_target_headroom_bytes": headroom,
        "rejection_counts": {
            "data232_excluded_records": decontam_excluded,
            **quality_stats,
            **privacy_stats,
        },
        "next_causal_gate": next_gate,
        "dependency_git_blobs": dict(dependency_blobs),
        "reserved_evaluation_decontamination_complete_for_this_execution": (
            True
        ),
        "canonical_quality_privacy_complete_for_this_execution": True,
        "trusted_family_authority_extended_for_new_families": False,
        "balance_diversity_retest_complete": False,
        "family_caps_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_pack_two_clean_complete": False,
        "durable_evidence_hash_only": True,
        "canonical_capacity_credit_bytes": 0,
        "tokenizer_fit_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    receipt["receipt_identity_sha256"] = sha256(canonical(receipt))
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "execute"))
    p.add_argument("--v7-root", type=Path, required=True)
    p.add_argument("--bulk-workspace", type=Path, required=True)
    p.add_argument("--expected-execution-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-training-handoff-identity-sha256")
    p.add_argument("--eval290-data-jsonl", type=Path)
    p.add_argument("--eval290-manifest-json", type=Path)
    p.add_argument("--eval291-data-jsonl", type=Path)
    p.add_argument("--eval291-authority-json", type=Path)
    p.add_argument(
        "--expected-selection-validation-identity-sha256",
        default=EVAL303_SELECTION_ID,
    )
    p.add_argument(
        "--expected-final-test-identity-sha256",
        default=DATA232_FINAL_TEST_ID,
    )
    return p


def main() -> int:
    args = parser().parse_args()
    dependencies = verify_checkout(
        args.expected_execution_head
    )
    inventory, payloads, survivors, _ = replay_parent(
        args.v7_root,
        args.bulk_workspace,
        args.expected_execution_head,
    )
    training_records, handoff, retained_inventory = (
        build_training_handoff(
            inventory,
            payloads,
            survivors,
        )
    )
    if args.command == "preflight":
        receipt = build_preflight(
            args.expected_execution_head,
            dependencies,
            handoff,
            retained_inventory,
        )
    else:
        for name in (
            "eval290_data_jsonl",
            "eval290_manifest_json",
            "eval291_data_jsonl",
            "eval291_authority_json",
        ):
            require(
                getattr(args, name) is not None,
                f"--{name.replace('_', '-')} is required for execute",
            )
        receipt = execute_postqp(
            args,
            args.expected_execution_head,
            dependencies,
            training_records,
            handoff,
            retained_inventory,
        )
    write_json(args.output, receipt)
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
