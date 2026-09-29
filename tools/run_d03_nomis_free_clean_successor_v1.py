#!/usr/bin/env python3
"""Build the physical Nomis1864-free D03 source successor on incumbent V8 mechanics.

This is a successor binding/execution layer, not a second matcher. It authenticates
the immutable terminal V7 graph, independently pinned Nomis quarantine authority,
and current V8/bulk mechanics; removes exactly the one quarantined source before any
new global-dedup invocation; then re-runs the incumbent V3 matcher on (a) the clean
historical graph and (b) the clean historical graph plus DATA-BULK-CODE-1.

No passing result from this tool is a corpus release, tokenizer authority, optimized
training exposure, or learned-weight claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-nomis-free-clean-successor.v1"
REPORT_SCHEMA = "12-6.d03-nomis-free-clean-successor-report.v1"
WORKER_ID = "SWARM-2065-NOMIS-FREE-CLEAN-SUCCESSOR-V1"

INCUMBENT_V8_PATH = Path("tools/run_next100_065f_global_dedup_v8.py")
INCUMBENT_SURVIVOR_PATH = Path("tools/derive_next100_065f_v8_survivors.py")
QUARANTINE_MODULE_PATH = Path("src/twelve_six/data/external_llm_provenance_quarantine_v1.py")
QUARANTINE_CONFIG_PATH = Path("configs/data/d03_external_llm_provenance_quarantine_v1.json")
INCUMBENT_V8_CONFIG_PATH = Path("configs/data/next100_065f_global_dedup_v8.json")

EXPECTED_MAIN = "0a594f91ceb61586fdf5d7062e4eae2a53ed90f7"
EXPECTED_V8_BLOB = "3a4df4b3cf6381893a58641dde479d9fd7fbe46d"
EXPECTED_SURVIVOR_BLOB = "ae91d60e5d62466c69394abb1c4b27d2e49f40e3"
EXPECTED_QUARANTINE_MODULE_BLOB = "5615c2e4732d14cbd5dc418bb6277e52afd579e6"
EXPECTED_QUARANTINE_CONFIG_BLOB = "c746f130d3100f90b233c80ffed6f877fa6eabcf"
EXPECTED_QUARANTINE_IDENTITY = "e9f29dd9f710fac057550e5cd671b7f412720e1ceb36568565a79909f11cf5b6"
BLOCKED_SOURCE_ID = "ua.verba.nomis1864.bounded24"
BLOCKED_FAMILY = "ua.verba.public-domain.nomis1864"
BLOCKED_SHA256 = "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
BLOCKED_BYTES = 1659

EXPECTED_CLEAN_HISTORICAL = {
    "source_object_count": 34,
    "source_capacity_bytes_before_global_dedup": 2_213_956,
    "source_family_counts": {"uk": 3, "en": 5, "code": 6},
}
EXPECTED_CLEAN_COMPOSED = {
    "source_object_count": 263,
    "source_capacity_bytes_before_global_dedup": 6_093_965,
    "source_family_counts": {"uk": 3, "en": 5, "code": 12},
    "conservative_unique_capacity_bytes_after_global_dedup": 6_093_662,
    "duplicate_discount_bytes": 303,
    "duplicate_cluster_count": 1,
    "post_dedup_survivor_source_object_count": 261,
}
TRUTH_BOUNDARY = {
    "source_object_authority_only": True,
    "clean_historical_record_materialization_required": True,
    "data526_clean_record_graph_materialized": False,
    "decontamination_executed_for_successor": False,
    "post_composition_quality_privacy_passed": False,
    "balance_release_claimed": False,
    "split_pack_complete": False,
    "tokenizer_fit_authorized": False,
    "authorized_training_exposure": 0,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "learned_weights_created": False,
    "final_test_payload_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}


class CleanSuccessorError(RuntimeError):
    """Fail-closed clean-successor error."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanSuccessorError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return _sha256(_canonical(body))


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanSuccessorError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_blob(root: Path, rel: Path, expected: str) -> None:
    try:
        raw = (root / rel).read_bytes()
    except OSError as exc:
        raise CleanSuccessorError(f"cannot read bound file {rel}: {exc}") from exc
    _require(_git_blob_sha1(raw) == expected, f"bound Git blob drift: {rel}")


def validate_runtime_bindings(root: Path) -> None:
    _verify_blob(root, INCUMBENT_V8_PATH, EXPECTED_V8_BLOB)
    _verify_blob(root, INCUMBENT_SURVIVOR_PATH, EXPECTED_SURVIVOR_BLOB)
    _verify_blob(root, QUARANTINE_MODULE_PATH, EXPECTED_QUARANTINE_MODULE_BLOB)
    _verify_blob(root, QUARANTINE_CONFIG_PATH, EXPECTED_QUARANTINE_CONFIG_BLOB)


def _family_counts(dedup: Mapping[str, Any]) -> dict[str, int]:
    sources = dedup.get("sources", [])
    _require(isinstance(sources, list), "dedup sources missing")
    return {
        modality: len(
            {
                row["source_family"]
                for row in sources
                if isinstance(row, Mapping) and row.get("modality") == modality
            }
        )
        for modality in ("uk", "en", "code")
    }


def _source_capacity(inventory: Mapping[str, Any]) -> int:
    rows = inventory.get("sources")
    _require(isinstance(rows, list), "inventory sources missing")
    total = 0
    for row in rows:
        _require(isinstance(row, Mapping), "inventory source must be an object")
        value = row.get("declared_capacity_bytes")
        _require(type(value) is int and value > 0, "invalid declared source capacity")
        total += value
    return total


def deauthorize_exact_nomis(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    quarantine_authority: Mapping[str, Any],
    quarantine_module: Any,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Authenticate and remove exactly the quarantined root before dedup.

    Silent filtering is forbidden. Removal is permitted only after the historical
    graph physically reproduces the exact blocked identity/hash/bytes and the
    independently pinned quarantine authority verifies.
    """
    identity = quarantine_module.validate_authority(quarantine_authority)
    _require(identity == EXPECTED_QUARANTINE_IDENTITY, "quarantine identity drift")

    rows = inventory.get("sources")
    _require(isinstance(rows, list), "inventory sources missing")
    source_rows = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("source_id") == BLOCKED_SOURCE_ID
    ]
    _require(len(source_rows) == 1, "exactly one authenticated Nomis source is required")
    row = source_rows[0]
    _require(row.get("source_family") == BLOCKED_FAMILY, "Nomis family drift")
    _require(row.get("declared_capacity_bytes") == BLOCKED_BYTES, "Nomis declared bytes drift")
    payload = payloads.get(BLOCKED_SOURCE_ID)
    _require(isinstance(payload, bytes), "Nomis physical payload missing")
    _require(len(payload) == BLOCKED_BYTES, "Nomis physical byte count drift")
    _require(_sha256(payload) == BLOCKED_SHA256, "Nomis physical SHA-256 drift")

    before_ids = [str(item["source_id"]) for item in rows if isinstance(item, Mapping)]
    _require(len(before_ids) == len(set(before_ids)), "historical source IDs are not unique")
    _require(set(payloads) == set(before_ids), "historical inventory/payload key mismatch")

    clean_inventory = copy.deepcopy(dict(inventory))
    clean_inventory["sources"] = [
        copy.deepcopy(dict(item))
        for item in rows
        if item.get("source_id") != BLOCKED_SOURCE_ID
    ]
    clean_payloads = dict(payloads)
    removed_payload = clean_payloads.pop(BLOCKED_SOURCE_ID)
    _require(removed_payload == payload, "removed Nomis payload identity drift")

    after_ids = [str(item["source_id"]) for item in clean_inventory["sources"]]
    _require(
        [source_id for source_id in before_ids if source_id != BLOCKED_SOURCE_ID] == after_ids,
        "non-Nomis source ordering/identity drift",
    )
    _require(set(clean_payloads) == set(after_ids), "clean inventory/payload key mismatch")

    text_free_rows = []
    for item in clean_inventory["sources"]:
        source_id = str(item["source_id"])
        raw = clean_payloads[source_id]
        text_free_rows.append(
            {
                "source_id": source_id,
                "family": item.get("source_family"),
                "payload_sha256": _sha256(raw),
                "payload_bytes": len(raw),
            }
        )
    quarantine_module.reject_quarantined_inventory_rows(
        text_free_rows,
        quarantine_authority,
    )

    proof = {
        "quarantine_identity_sha256": identity,
        "blocked_source_id": BLOCKED_SOURCE_ID,
        "blocked_family": BLOCKED_FAMILY,
        "blocked_payload_sha256": BLOCKED_SHA256,
        "blocked_payload_bytes": BLOCKED_BYTES,
        "pre_source_object_count": len(before_ids),
        "post_source_object_count": len(after_ids),
        "pre_source_capacity_bytes": _source_capacity(inventory),
        "post_source_capacity_bytes": _source_capacity(clean_inventory),
        "removed_before_new_global_dedup": True,
        "non_blocked_source_ids_byte_for_byte_preserved": True,
    }
    _require(proof["pre_source_object_count"] == 35, "historical object-count drift")
    _require(proof["post_source_object_count"] == 34, "clean historical object-count drift")
    _require(proof["pre_source_capacity_bytes"] == 2_215_615, "historical byte-capacity drift")
    _require(proof["post_source_capacity_bytes"] == 2_213_956, "clean historical byte-capacity drift")
    return clean_inventory, clean_payloads, proof


def _assert_clean_dedup(
    dedup: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(dedup.get("source_count") == expected["source_object_count"], f"{label} source-count drift")
    _require(_family_counts(dedup) == expected["source_family_counts"], f"{label} family-vector drift")
    terminal = dedup.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), f"{label} terminal candidates missing")
    _require(
        terminal.get("declared_capacity_bytes_before")
        == expected["source_capacity_bytes_before_global_dedup"],
        f"{label} pre-dedup capacity drift",
    )
    if "conservative_unique_capacity_bytes_after_global_dedup" in expected:
        _require(
            terminal.get("conservative_unique_capacity_bytes_after")
            == expected["conservative_unique_capacity_bytes_after_global_dedup"],
            f"{label} post-dedup capacity drift",
        )
        _require(
            terminal.get("duplicate_discount_bytes") == expected["duplicate_discount_bytes"],
            f"{label} duplicate-discount drift",
        )
        _require(
            terminal.get("duplicate_cluster_count") == expected["duplicate_cluster_count"],
            f"{label} duplicate-cluster drift",
        )


def _source_vector(dedup: Mapping[str, Any]) -> dict[str, Any]:
    terminal = dedup["terminal_candidates"]
    return {
        "source_object_count": dedup["source_count"],
        "source_family_counts": _family_counts(dedup),
        "source_capacity_bytes_before_global_dedup": terminal["declared_capacity_bytes_before"],
        "conservative_unique_capacity_bytes_after_global_dedup": terminal[
            "conservative_unique_capacity_bytes_after"
        ],
        "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
        "duplicate_cluster_count": terminal["duplicate_cluster_count"],
        "effective_independent_origin_count": terminal["effective_independent_origin_count"],
        "by_modality": {
            modality: {
                "source_count": terminal["by_modality"][modality]["source_count"],
                "source_family_count": terminal["by_modality"][modality][
                    "declared_source_family_count"
                ],
                "capacity_bytes_before_global_dedup": terminal["by_modality"][modality][
                    "declared_capacity_bytes_before"
                ],
                "conservative_unique_capacity_bytes_after_global_dedup": terminal[
                    "by_modality"
                ][modality]["conservative_unique_capacity_bytes_after"],
                "duplicate_discount_bytes": terminal["by_modality"][modality][
                    "duplicate_discount_bytes"
                ],
            }
            for modality in ("uk", "en", "code")
        },
    }


def run(
    current_root: Path,
    v7_root: Path,
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_runtime_bindings(current_root)
    v8 = _load_module("_swarm2065_incumbent_v8", current_root / INCUMBENT_V8_PATH)
    survivor_tool = _load_module(
        "_swarm2065_incumbent_survivor",
        current_root / INCUMBENT_SURVIVOR_PATH,
    )
    quarantine_module = _load_module(
        "_swarm2065_quarantine",
        current_root / QUARANTINE_MODULE_PATH,
    )
    quarantine_authority = _read_json(current_root / QUARANTINE_CONFIG_PATH)
    old_v8_config = v8.load_config(current_root / INCUMBENT_V8_CONFIG_PATH)

    v7, baseline_report, inventory, payloads = v8._capture_terminal_v7(v7_root, old_v8_config)
    clean_inventory, clean_payloads, removal = deauthorize_exact_nomis(
        inventory,
        payloads,
        quarantine_authority,
        quarantine_module,
    )

    clean_historical_dedup = v7.v6.v3.audit_payloads(clean_inventory, clean_payloads)
    v7.v6.v3.verify_report(clean_historical_dedup)
    _assert_clean_dedup(
        clean_historical_dedup,
        EXPECTED_CLEAN_HISTORICAL,
        label="clean historical",
    )

    bulk_report, bulk_rows, bulk_payloads = v8._materialize_bulk(
        current_root,
        workspace,
        old_v8_config,
    )
    combined_inventory = copy.deepcopy(clean_inventory)
    existing_ids = {row["source_id"] for row in combined_inventory["sources"]}
    _require(not (existing_ids & set(bulk_payloads)), "bulk source id collides with clean historical graph")
    combined_inventory["sources"] = [*combined_inventory["sources"], *bulk_rows]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_cutoff_utc"] = "2026-09-14T19:14:21Z"
    combined_inventory["terminal_refresh_rule"] = (
        "SWARM-2065 authenticates terminal V7, deauthorizes exact quarantined Nomis1864 "
        "before matcher invocation, then composes the unchanged DATA-BULK-CODE-1 payloads. "
        "All capacity remains zero-credit pre-decontamination source authority."
    )
    combined_payloads = dict(clean_payloads)
    combined_payloads.update(bulk_payloads)

    dedup = v7.v6.v3.audit_payloads(combined_inventory, combined_payloads)
    v7.v6.v3.verify_report(dedup)
    _assert_clean_dedup(dedup, EXPECTED_CLEAN_COMPOSED, label="clean composed")

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": WORKER_ID,
        "execution_profile": "LOCAL_FREE",
        "base_main_sha": EXPECTED_MAIN,
        "incumbent_bindings": {
            "v8_tool_git_blob_sha1": EXPECTED_V8_BLOB,
            "survivor_tool_git_blob_sha1": EXPECTED_SURVIVOR_BLOB,
            "quarantine_module_git_blob_sha1": EXPECTED_QUARANTINE_MODULE_BLOB,
            "quarantine_config_git_blob_sha1": EXPECTED_QUARANTINE_CONFIG_BLOB,
            "historical_v7_head_sha": old_v8_config["baseline_v7"]["head_sha"],
            "historical_v7_report_sha256": baseline_report["report_sha256"],
            "historical_v7_nested_v3_sha256": baseline_report["dedup_v3"]["report_sha256"],
            "bulk_terminal_report_identity_sha256": bulk_report["report_identity_sha256"],
        },
        "deauthorization": removal,
        "clean_historical": {
            "source_vector": _source_vector(clean_historical_dedup),
            "dedup_v3": clean_historical_dedup,
        },
        "source_vector": _source_vector(dedup),
        "dedup_v3": dedup,
        "raw_text_emitted": False,
        "truth_boundary": copy.deepcopy(TRUTH_BOUNDARY),
        "remaining_blockers": [
            "fresh_clean_historical_data526_record_materialization",
            "fresh_clean_data526_record_graph_composition",
            "reserved_evaluation_decontamination",
            "post_composition_quality_privacy",
            "balance_and_family_caps",
            "cluster_safe_split",
            "deterministic_tokenizer_and_packing",
            "post_pack_unique_loss_ledger",
        ],
    }
    core["report_sha256"] = _self_hash(core)

    survivor = survivor_tool.derive_survivor_authority(core)
    _require(
        survivor.get("post_dedup_survivor_source_object_count")
        == EXPECTED_CLEAN_COMPOSED["post_dedup_survivor_source_object_count"],
        "clean survivor object-count drift",
    )
    _require(
        survivor.get("post_dedup_declared_capacity_bytes")
        == EXPECTED_CLEAN_COMPOSED["conservative_unique_capacity_bytes_after_global_dedup"],
        "clean survivor capacity drift",
    )
    _require(survivor.get("duplicate_cluster_count") == 1, "clean survivor cluster-count drift")
    return core, survivor


def verify_report(report: Mapping[str, Any], survivor: Mapping[str, Any], current_root: Path) -> None:
    validate_runtime_bindings(current_root)
    _require(report.get("schema_version") == REPORT_SCHEMA, "clean report schema drift")
    _require(report.get("worker_id") == WORKER_ID, "clean report worker drift")
    _require(report.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary weakened")
    _require(report.get("base_main_sha") == EXPECTED_MAIN, "clean report main binding drift")
    _require(report.get("report_sha256") == _self_hash(report), "clean report self-hash mismatch")
    _require(report.get("raw_text_emitted") is False, "durable report leaked raw text")
    _require(report.get("truth_boundary") == TRUTH_BOUNDARY, "clean report truth boundary drift")
    removal = report.get("deauthorization")
    _require(isinstance(removal, Mapping), "deauthorization proof missing")
    _require(removal.get("removed_before_new_global_dedup") is True, "Nomis was not removed pre-dedup")
    _require(removal.get("blocked_payload_sha256") == BLOCKED_SHA256, "blocked payload binding drift")
    vector = report.get("source_vector")
    _require(isinstance(vector, Mapping), "clean source vector missing")
    for key, expected in EXPECTED_CLEAN_COMPOSED.items():
        if key == "post_dedup_survivor_source_object_count":
            continue
        _require(vector.get(key) == expected, f"clean report vector drift: {key}")

    survivor_tool = _load_module(
        "_swarm2065_verify_survivor",
        current_root / INCUMBENT_SURVIVOR_PATH,
    )
    survivor_tool.verify_survivor_authority(report, survivor)
    _require(
        survivor.get("post_dedup_survivor_source_object_count")
        == EXPECTED_CLEAN_COMPOSED["post_dedup_survivor_source_object_count"],
        "survivor count drift",
    )
    _require(
        survivor.get("post_dedup_declared_capacity_bytes")
        == EXPECTED_CLEAN_COMPOSED["conservative_unique_capacity_bytes_after_global_dedup"],
        "survivor capacity drift",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check-bindings", "run", "verify"))
    parser.add_argument("--current-root", type=Path, default=Path("."))
    parser.add_argument("--v7-root", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--survivor", type=Path)
    args = parser.parse_args()

    if args.command == "check-bindings":
        validate_runtime_bindings(args.current_root)
        print("PASS_SWARM2065_RUNTIME_BINDINGS")
        return 0

    _require(args.report is not None, "--report is required")
    _require(args.survivor is not None, "--survivor is required")
    if args.command == "run":
        _require(args.v7_root is not None, "--v7-root is required")
        _require(args.workspace is not None, "--workspace is required")
        args.workspace.mkdir(parents=True, exist_ok=False)
        report, survivor = run(args.current_root, args.v7_root, args.workspace)
        verify_report(report, survivor, args.current_root)
        _write_json(args.report, report)
        _write_json(args.survivor, survivor)
        print(f"report_sha256={report['report_sha256']}")
        print(f"survivor_authority_sha256={survivor['survivor_authority_sha256']}")
        print(
            "post_global_dedup_unique_bytes="
            f"{report['source_vector']['conservative_unique_capacity_bytes_after_global_dedup']}"
        )
        return 0

    report = _read_json(args.report)
    survivor = _read_json(args.survivor)
    verify_report(report, survivor, args.current_root)
    print("PASS_SWARM2065_CLEAN_SUCCESSOR_REPORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
