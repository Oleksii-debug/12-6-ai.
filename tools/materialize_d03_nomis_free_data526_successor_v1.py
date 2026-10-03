#!/usr/bin/env python3
"""Materialize the clean DATA526 successor from SWARM-2065 source authority.

This tool deliberately does not consume the historical contaminated 48-row PR623
record JSONL. It reproduces the clean 34-source historical graph, reuses the exact
PR623 DATA213/KMU/CPython transform implementation from its immutable checkout,
materializes a fresh 47-row historical record set, then composes it with freshly
materialized DATA-BULK-CODE-1 payloads selected by the clean post-dedup survivor
authority.

Raw payload JSONL is a local execution product only. Durable evidence is text-free.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_RUNNER = Path("tools/run_d03_nomis_free_clean_successor_v1.py")
INCUMBENT_V8 = Path("tools/run_next100_065f_global_dedup_v8.py")
INVENTORY_TOOL = Path("tools/materialize_data526_record_inventory_v1.py")
QUARANTINE_MODULE = Path("src/twelve_six/data/external_llm_provenance_quarantine_v1.py")
QUARANTINE_CONFIG = Path("configs/data/d03_external_llm_provenance_quarantine_v1.json")

HISTORICAL_MATERIALIZER_HEAD = "70d6ccc87396d129d00771bbf0b6b29bc673bfc4"
HISTORICAL_MATERIALIZER_PATH = Path("tools/materialize_data526_records_from_v7.py")
HISTORICAL_MATERIALIZER_CONFIG = Path("configs/data/data526_record_materialization_v5.json")
HISTORICAL_MATERIALIZER_BLOB = "190d8d5d7727230aefb04cc1464ba42e023b32eb"
HISTORICAL_MATERIALIZER_CONFIG_BLOB = "ccc6ed614955739439e4ea09423a0ecbe2f9d3b9"
INVENTORY_TOOL_BLOB = "9b0c5b9df66df468e7b6a4a317c2cd82412ee659"

SCHEMA = "12-6.d03-nomis-free-data526-successor-evidence.v1"
EXPECTED_HISTORICAL_SOURCES = 34
EXPECTED_HISTORICAL_RECORDS = 47
EXPECTED_HISTORICAL_BYTES = 2_213_956
EXPECTED_HISTORICAL_DIRECT_SOURCES = 24
EXPECTED_HISTORICAL_DIRECT_BYTES = 2_015_905
EXPECTED_HISTORICAL_MODALITY_BYTES = {"uk": 99_197, "en": 1_838_293, "code": 276_466}
EXPECTED_HISTORICAL_FAMILIES = {"uk": 3, "en": 5, "code": 6}
EXPECTED_BULK_SOURCES = 229
EXPECTED_COMPOSED_SOURCES = 263
EXPECTED_SURVIVOR_SOURCES = 261
EXPECTED_FINAL_BYTES = 6_093_662
EXPECTED_FINAL_RECORDS = 274

TRUTH_BOUNDARY = {
    "clean_data526_record_graph_materialized": True,
    "corpus_released": False,
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
    "raw_payloads_committed_to_repository": False,
    "raw_payloads_uploaded_as_public_evidence": False,
}


class CleanData526Error(RuntimeError):
    """Fail-closed clean DATA526 materialization error."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanData526Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanData526Error(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_blob(root: Path, rel: Path, expected: str) -> None:
    raw = (root / rel).read_bytes()
    _require(_git_blob_sha1(raw) == expected, f"bound Git blob drift: {rel}")


def _validated_git_sha(value: str) -> str:
    _require(
        len(value) == 40 and all(ch in "0123456789abcdef" for ch in value),
        "execution head SHA must be lowercase 40-hex",
    )
    return value


def _load_historical_materializer(root: Path) -> tuple[Any, dict[str, Any]]:
    _require(_git_head(root) == HISTORICAL_MATERIALIZER_HEAD, "historical materializer checkout drift")
    _verify_blob(root, HISTORICAL_MATERIALIZER_PATH, HISTORICAL_MATERIALIZER_BLOB)
    _verify_blob(
        root,
        HISTORICAL_MATERIALIZER_CONFIG,
        HISTORICAL_MATERIALIZER_CONFIG_BLOB,
    )
    module = _load_module(
        "_swarm2065_historical_data526_materializer",
        root / HISTORICAL_MATERIALIZER_PATH,
    )
    config = _read_json(root / HISTORICAL_MATERIALIZER_CONFIG)
    module.verify_config(config)
    return module, config


def _historical_source_facts(dedup: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = dedup.get("sources")
    _require(isinstance(rows, list), "clean historical dedup sources missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "clean historical source row invalid")
        source_id = row.get("source_id")
        _require(
            isinstance(source_id, str) and source_id and source_id not in result,
            "clean historical source id invalid/duplicate",
        )
        result[source_id] = row
    _require(len(result) == EXPECTED_HISTORICAL_SOURCES, "clean historical source-count drift")
    return result


def _capture_clean_historical(
    current_root: Path,
    v7_root: Path,
    source_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    source_runner = _load_module("_swarm2065_source_runner", current_root / SOURCE_RUNNER)
    v8 = _load_module("_swarm2065_v8_for_records", current_root / INCUMBENT_V8)
    quarantine = _load_module("_swarm2065_quarantine_for_records", current_root / QUARANTINE_MODULE)
    quarantine_authority = _read_json(current_root / QUARANTINE_CONFIG)
    v8_config = v8.load_config(current_root / "configs/data/next100_065f_global_dedup_v8.json")
    v7, _baseline, inventory, payloads = v8._capture_terminal_v7(v7_root, v8_config)
    clean_inventory, clean_payloads, _proof = source_runner.deauthorize_exact_nomis(
        inventory,
        payloads,
        quarantine_authority,
        quarantine,
    )
    dedup = v7.v6.v3.audit_payloads(clean_inventory, clean_payloads)
    v7.v6.v3.verify_report(dedup)
    expected = source_report.get("clean_historical", {}).get("dedup_v3")
    _require(isinstance(expected, Mapping), "source report clean historical authority missing")
    _require(dict(dedup) == dict(expected), "fresh clean historical dedup differs from source authority")
    return clean_inventory, clean_payloads, dedup


def _record(source: Mapping[str, Any], payload: bytes, record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": str(source["source_id"]),
        "family": str(source["source_family"]),
        "modality": str(source["modality"]),
        "normalized_payload": payload.decode("utf-8", errors="strict"),
    }


def materialize_clean_historical_records(
    *,
    historical_module: Any,
    historical_config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    clean_dedup: Mapping[str, Any],
    data213_zip: Path,
    quarantine_module: Any,
    quarantine_authority: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Reuse immutable PR623 transforms against fresh clean V3 source facts."""
    rows = inventory.get("sources")
    _require(isinstance(rows, list), "clean historical inventory sources missing")
    source_by_id = {str(row["source_id"]): row for row in rows if isinstance(row, Mapping)}
    _require(len(source_by_id) == EXPECTED_HISTORICAL_SOURCES, "clean historical inventory count drift")
    _require(set(source_by_id) == set(payloads), "clean historical inventory/payload key mismatch")
    _require("ua.verba.nomis1864.bounded24" not in source_by_id, "Nomis source survived clean cut")

    report_sources = _historical_source_facts(clean_dedup)
    _require(set(report_sources) == set(source_by_id), "clean dedup source set differs from physical graph")

    data213_payloads = historical_module._load_data213_payloads(data213_zip, historical_config)
    data213_specs = historical_config["data213_normalized_artifact"]["sources"]
    kmu_specs = historical_config["kmu_authority"]["sources"]
    cpython = historical_config["cpython_authority"]
    cpython_id = cpython["source_id"]

    records: list[dict[str, str]] = []
    direct_count = 0
    direct_bytes = 0
    modality_bytes = {"uk": 0, "en": 0, "code": 0}

    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        payload = payloads[source_id]
        if source_id in data213_specs:
            payload = data213_payloads[source_id]
            records.append(_record(source, payload, source_id))
        elif source_id in kmu_specs:
            payload = historical_module._normalize_kmu(payload)
            historical_module._verify_payload(payload, kmu_specs[source_id], source_id=source_id)
            records.append(_record(source, payload, source_id))
        elif source_id == cpython_id:
            for index, chunk in enumerate(
                historical_module._split_cpython(payload, cpython),
                start=1,
            ):
                chunk_hash = historical_module.sha256(chunk)
                record_id = f"{source_id}#accepted-{index:02d}-{chunk_hash[:12]}"
                records.append(_record(source, chunk, record_id))
                modality_bytes[str(source["modality"])] += len(chunk)
            continue
        else:
            fact = report_sources[source_id]
            declared = int(fact["declared_capacity_bytes"])
            comparison = int(fact["comparison_payload_bytes"])
            _require(
                comparison == declared == len(payload),
                f"unsupported clean direct source requires explicit adapter: {source_id}",
            )
            _require(
                _sha256(payload) == fact["comparison_payload_sha256"],
                f"clean direct payload SHA drift: {source_id}",
            )
            records.append(_record(source, payload, source_id))
            direct_count += 1
            direct_bytes += len(payload)
        modality_bytes[str(source["modality"])] += len(payload)

    records.sort(key=lambda row: row["record_id"])
    total_bytes = sum(len(row["normalized_payload"].encode("utf-8")) for row in records)
    _require(len(records) == EXPECTED_HISTORICAL_RECORDS, "clean historical record-count drift")
    _require(total_bytes == EXPECTED_HISTORICAL_BYTES, "clean historical payload-byte drift")
    _require(direct_count == EXPECTED_HISTORICAL_DIRECT_SOURCES, "clean historical direct-count drift")
    _require(direct_bytes == EXPECTED_HISTORICAL_DIRECT_BYTES, "clean historical direct-byte drift")
    _require(modality_bytes == EXPECTED_HISTORICAL_MODALITY_BYTES, "clean historical modality-byte drift")

    family_counts = {
        modality: len(
            {
                row["family"]
                for row in records
                if row["modality"] == modality
            }
        )
        for modality in ("uk", "en", "code")
    }
    _require(family_counts == EXPECTED_HISTORICAL_FAMILIES, "clean historical family-count drift")
    quarantine_module.reject_quarantined_records(records, quarantine_authority)

    return records, {
        "source_object_count": len(source_by_id),
        "record_count": len(records),
        "total_payload_bytes": total_bytes,
        "direct_source_count": direct_count,
        "direct_payload_bytes": direct_bytes,
    }


def _survivor_by_id(authority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = authority.get("survivors")
    _require(isinstance(rows, list), "clean survivor rows missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "clean survivor row invalid")
        source_id = row.get("source_id")
        _require(
            isinstance(source_id, str) and source_id and source_id not in result,
            "clean survivor source id invalid/duplicate",
        )
        result[source_id] = row
    _require(len(result) == EXPECTED_SURVIVOR_SOURCES, "clean survivor count drift")
    return result


def _historical_source_aggregates(
    records: list[dict[str, Any]],
) -> dict[str, tuple[str, str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_id"]), []).append(record)
    result: dict[str, tuple[str, str, int]] = {}
    for source_id, group in grouped.items():
        families = {str(row["family"]) for row in group}
        modalities = {str(row["modality"]) for row in group}
        _require(len(families) == 1, f"mixed historical family: {source_id}")
        _require(len(modalities) == 1, f"mixed historical modality: {source_id}")
        payload_bytes = sum(len(str(row["normalized_payload"]).encode("utf-8")) for row in group)
        result[source_id] = (next(iter(families)), next(iter(modalities)), payload_bytes)
    return result


def _validate_survivor_binding(
    row: Mapping[str, Any],
    *,
    source_id: str,
    family: str,
    modality: str,
    payload_bytes: int,
    raw_sha256: str | None = None,
) -> None:
    _require(row.get("source_id") == source_id, f"survivor source-id drift: {source_id}")
    _require(row.get("source_family") == family, f"survivor family drift: {source_id}")
    _require(row.get("modality") == modality, f"survivor modality drift: {source_id}")
    _require(row.get("declared_capacity_bytes") == payload_bytes, f"survivor byte drift: {source_id}")
    if raw_sha256 is not None:
        _require(row.get("verified_raw_sha256") == raw_sha256, f"survivor raw hash drift: {source_id}")


def compose_clean_records(
    *,
    historical_records: list[dict[str, Any]],
    bulk_rows: list[dict[str, Any]],
    bulk_payloads: Mapping[str, bytes],
    survivor_authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    survivor_by_id = _survivor_by_id(survivor_authority)
    survivor_ids = set(survivor_by_id)

    historical_aggregates = _historical_source_aggregates(historical_records)
    _require(
        len(historical_aggregates) == EXPECTED_HISTORICAL_SOURCES,
        "clean historical source-object count drift",
    )
    for source_id, (family, modality, payload_bytes) in historical_aggregates.items():
        _require(source_id in survivor_by_id, f"clean historical source unexpectedly dropped: {source_id}")
        _validate_survivor_binding(
            survivor_by_id[source_id],
            source_id=source_id,
            family=family,
            modality=modality,
            payload_bytes=payload_bytes,
        )

    bulk_by_id: dict[str, Mapping[str, Any]] = {}
    for row in bulk_rows:
        source_id = str(row["source_id"])
        _require(source_id not in bulk_by_id, f"duplicate bulk source id: {source_id}")
        bulk_by_id[source_id] = row
    _require(len(bulk_by_id) == EXPECTED_BULK_SOURCES, "fresh bulk source-count drift")
    _require(set(bulk_by_id) == set(bulk_payloads), "fresh bulk row/payload key mismatch")
    _require(
        not (set(historical_aggregates) & set(bulk_by_id)),
        "historical/bulk source namespace collision",
    )
    _require(
        len(historical_aggregates) + len(bulk_by_id) == EXPECTED_COMPOSED_SOURCES,
        "clean composed source-count drift",
    )
    _require(
        survivor_ids <= (set(historical_aggregates) | set(bulk_by_id)),
        "clean survivor references unavailable source",
    )

    selected = [
        copy.deepcopy(row)
        for row in historical_records
        if str(row["source_id"]) in survivor_ids
    ]
    for source_id in sorted(set(bulk_by_id) & survivor_ids):
        row = bulk_by_id[source_id]
        raw = bulk_payloads[source_id]
        raw_sha = _sha256(raw)
        _require(raw_sha == row["expected_raw_sha256"], f"fresh bulk physical hash drift: {source_id}")
        _require(len(raw) == row["declared_capacity_bytes"], f"fresh bulk physical byte drift: {source_id}")
        _validate_survivor_binding(
            survivor_by_id[source_id],
            source_id=source_id,
            family=str(row["source_family"]),
            modality="code",
            payload_bytes=len(raw),
            raw_sha256=raw_sha,
        )
        selected.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "family": str(row["source_family"]),
                "modality": "code",
                "normalized_payload": raw.decode("utf-8", errors="strict"),
            }
        )

    selected.sort(key=lambda row: str(row["record_id"]))
    covered = {str(row["source_id"]) for row in selected}
    _require(covered == survivor_ids, "clean record graph does not exactly cover survivor source set")
    total_bytes = sum(len(str(row["normalized_payload"]).encode("utf-8")) for row in selected)
    _require(total_bytes == EXPECTED_FINAL_BYTES, "clean DATA526 payload-byte drift")
    _require(len(selected) == EXPECTED_FINAL_RECORDS, "clean DATA526 record-count drift")
    return selected


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    raw = b"".join(_canonical(row) + b"\n" for row in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256(raw)


def run(
    *,
    current_root: Path,
    execution_head_sha: str,
    v7_root: Path,
    historical_materializer_root: Path,
    data213_zip: Path,
    bulk_workspace: Path,
    source_report: Mapping[str, Any],
    survivor: Mapping[str, Any],
    historical_records_path: Path,
    records_path: Path,
    inventory_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    execution_head_sha = _validated_git_sha(execution_head_sha)
    _require(_git_head(current_root) == execution_head_sha, "current checkout/head authority drift")
    _verify_blob(current_root, INVENTORY_TOOL, INVENTORY_TOOL_BLOB)

    source_runner = _load_module("_swarm2065_source_verify", current_root / SOURCE_RUNNER)
    source_runner.verify_report(source_report, survivor, current_root)

    historical_module, historical_config = _load_historical_materializer(
        historical_materializer_root
    )
    quarantine_module = _load_module(
        "_swarm2065_quarantine_data526",
        current_root / QUARANTINE_MODULE,
    )
    quarantine_authority = _read_json(current_root / QUARANTINE_CONFIG)
    inventory_tool = _load_module(
        "_swarm2065_inventory",
        current_root / INVENTORY_TOOL,
    )

    clean_inventory, clean_payloads, clean_dedup = _capture_clean_historical(
        current_root,
        v7_root,
        source_report,
    )
    historical_records, historical_stats = materialize_clean_historical_records(
        historical_module=historical_module,
        historical_config=historical_config,
        inventory=clean_inventory,
        payloads=clean_payloads,
        clean_dedup=clean_dedup,
        data213_zip=data213_zip,
        quarantine_module=quarantine_module,
        quarantine_authority=quarantine_authority,
    )
    historical_jsonl_sha = _write_jsonl(historical_records_path, historical_records)
    historical_inventory = inventory_tool.materialize(historical_records)
    _require(
        historical_inventory["record_count"] == EXPECTED_HISTORICAL_RECORDS,
        "clean historical inventory count drift",
    )
    _require(
        historical_inventory["total_payload_bytes"] == EXPECTED_HISTORICAL_BYTES,
        "clean historical inventory bytes drift",
    )

    v8 = _load_module("_swarm2065_bulk_for_data526", current_root / INCUMBENT_V8)
    old_v8_config = v8.load_config(current_root / "configs/data/next100_065f_global_dedup_v8.json")
    bulk_report, bulk_rows, bulk_payloads = v8._materialize_bulk(
        current_root,
        bulk_workspace,
        old_v8_config,
    )
    _require(
        bulk_report["eligible_file_count"] == EXPECTED_BULK_SOURCES,
        "fresh bulk report count drift",
    )

    records = compose_clean_records(
        historical_records=historical_records,
        bulk_rows=bulk_rows,
        bulk_payloads=bulk_payloads,
        survivor_authority=survivor,
    )
    records_sha = _write_jsonl(records_path, records)
    final_inventory = inventory_tool.materialize(records)
    _require(final_inventory["record_count"] == EXPECTED_FINAL_RECORDS, "final inventory count drift")
    _require(final_inventory["total_payload_bytes"] == EXPECTED_FINAL_BYTES, "final inventory byte drift")
    quarantine_module.reject_quarantined_inventory_rows(
        final_inventory["records"],
        quarantine_authority,
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_bytes(_canonical(final_inventory) + b"\n")

    core = {
        "schema_version": SCHEMA,
        "worker_id": "SWARM-2065-NOMIS-FREE-DATA526-SUCCESSOR-V1",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head_sha,
        "source_report_sha256": source_report["report_sha256"],
        "survivor_authority_sha256": survivor["survivor_authority_sha256"],
        "historical_materializer": {
            "head_sha": HISTORICAL_MATERIALIZER_HEAD,
            "tool_git_blob_sha1": HISTORICAL_MATERIALIZER_BLOB,
            "config_git_blob_sha1": HISTORICAL_MATERIALIZER_CONFIG_BLOB,
            "transform_contract_reused_without_modification": True,
        },
        "clean_historical": {
            **historical_stats,
            "record_payload_jsonl_sha256": historical_jsonl_sha,
            "record_inventory_digest_sha256": historical_inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": historical_inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "bulk_report_identity_sha256": bulk_report["report_identity_sha256"],
        "clean_data526": {
            "record_count": final_inventory["record_count"],
            "source_object_count": survivor["post_dedup_survivor_source_object_count"],
            "total_payload_bytes": final_inventory["total_payload_bytes"],
            "record_payload_jsonl_sha256": records_sha,
            "record_inventory_digest_sha256": final_inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": final_inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "raw_text_emitted_to_durable_evidence": False,
        "truth_boundary": copy.deepcopy(TRUTH_BOUNDARY),
        "remaining_blockers": [
            "reserved_evaluation_decontamination",
            "post_composition_quality_privacy",
            "balance_and_family_caps",
            "cluster_safe_split",
            "deterministic_tokenizer_and_packing",
            "post_pack_unique_loss_ledger",
        ],
    }
    evidence = {**core, "evidence_identity_sha256": _sha256(_canonical(core))}
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(_canonical(evidence) + b"\n")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, default=Path("."))
    parser.add_argument("--execution-head-sha", required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--historical-materializer-root", type=Path, required=True)
    parser.add_argument("--data213-zip", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--survivor", type=Path, required=True)
    parser.add_argument("--historical-records-jsonl", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.bulk_workspace.exists(), "bulk workspace must not already exist")
    args.bulk_workspace.mkdir(parents=True)

    evidence = run(
        current_root=args.current_root,
        execution_head_sha=args.execution_head_sha,
        v7_root=args.v7_root,
        historical_materializer_root=args.historical_materializer_root,
        data213_zip=args.data213_zip,
        bulk_workspace=args.bulk_workspace,
        source_report=_read_json(args.source_report),
        survivor=_read_json(args.survivor),
        historical_records_path=args.historical_records_jsonl,
        records_path=args.records_jsonl,
        inventory_path=args.inventory_json,
        evidence_path=args.evidence_json,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
