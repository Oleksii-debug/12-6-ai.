#!/usr/bin/env python3
"""Compose the current-main DATA-526 pre-decontamination record graph from V8.

The tool deliberately separates three authorities:
1. the exact historical 48-record V7 materialization from DATA-526/#623;
2. a fresh DATA-BULK-CODE-1 materialization workspace/report for the 229 new files;
3. terminal V8 global-dedup survivor authority selecting source objects.

It cannot run while the config still says WAIT_EXACT_TERMINAL_V8_EVIDENCE. Once V8
is terminal and the exact authority is sealed, this tool produces a deterministic raw
record JSONL for local downstream processing plus text-free inventory/evidence. Raw
payloads must not be committed or uploaded as public evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from materialize_data526_record_inventory_v1 import canonical_json, load_jsonl, materialize

DEFAULT_CONFIG = Path("configs/data/data526_v8_record_composition_v1.json")
CONFIG_SCHEMA = "12-6.data526-v8-record-composition.v1"
V8_REPORT_SCHEMA = "12-6.next100-065f-global-dedup-report.v8"
V8_SURVIVOR_SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
TERMINAL_V8_STATUS = "TERMINAL_EXACT_V8_EVIDENCE"
EXPECTED_HISTORICAL_RECORDS = 48
EXPECTED_HISTORICAL_SOURCES = 35
EXPECTED_HISTORICAL_BYTES = 2_215_615
EXPECTED_BULK_FILES = 229
EXPECTED_BULK_BYTES = 3_880_009
EXPECTED_COMPOSED_SOURCES = EXPECTED_HISTORICAL_SOURCES + EXPECTED_BULK_FILES
EXPECTED_V8_REPORT_SHA256 = "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a"
EXPECTED_V8_NESTED_V3_SHA256 = "e7244cb7f6df6062dc838112b1e3e6fe87e41644a9e8b0071e93c50f1166b67d"
EXPECTED_V8_SURVIVOR_SHA256 = "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf"
EXPECTED_V8_RUN_ID = 34159790818
EXPECTED_V8_ARTIFACT_ID = 10032510626
EXPECTED_V8_ARTIFACT_DIGEST = "sha256:960c678fdbe6245658fa69c39edf355eb7b38ef39b7f8f833155c894dbd90385"
SURVIVOR_SOURCE_BINDING_FIELDS = (
    "source_family",
    "modality",
    "declared_capacity_bytes",
    "verified_raw_sha256",
    "normalized_sha256",
    "stable_origin_id_sha256",
    "stable_object_id_sha256",
)


class Data526V8Error(RuntimeError):
    """Raised when an authority or record-composition invariant fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Data526V8Error(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validated_git_sha(value: str) -> str:
    _require(
        len(value) == 40 and all(ch in "0123456789abcdef" for ch in value),
        "execution head SHA must be lowercase 40-hex",
    )
    return value


def _canonical_ascii(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_utf8(value: Any) -> bytes:
    """Canonicalization used by the V8 report producer."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Data526V8Error(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _expected_v8_terminal_input() -> dict[str, Any]:
    return {
        "pr": 824,
        "required_report_schema": V8_REPORT_SCHEMA,
        "required_survivor_schema": V8_SURVIVOR_SCHEMA,
        "report_sha256": EXPECTED_V8_REPORT_SHA256,
        "nested_v3_report_sha256": EXPECTED_V8_NESTED_V3_SHA256,
        "survivor_authority_sha256": EXPECTED_V8_SURVIVOR_SHA256,
        "workflow_run_id": EXPECTED_V8_RUN_ID,
        "workflow_conclusion": "success",
        "artifact_id": EXPECTED_V8_ARTIFACT_ID,
        "artifact_digest": EXPECTED_V8_ARTIFACT_DIGEST,
        "status": TERMINAL_V8_STATUS,
    }


def verify_config(config: Mapping[str, Any], *, require_terminal_v8: bool = True) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "DATA-526 V8 config schema drift")
    _require(config.get("worker_id") == "DATA-526-CURRENT-MAIN-V8-RECORD-COMPOSER", "DATA-526 worker drift")
    _require(config.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary weakened")

    historical = config.get("historical_v7_record_authority")
    _require(isinstance(historical, Mapping), "historical authority missing")
    exact_historical = {
        "source_pr": 623,
        "head_sha": "70d6ccc87396d129d00771bbf0b6b29bc673bfc4",
        "workflow_run_id": 34069812891,
        "workflow_conclusion": "success",
        "artifact_id": 10000091203,
        "artifact_digest": "sha256:9fb809ab0256d485d21a4e686e4e7d72d4c9b2d05adbc10a8b3f426ef19674ff",
        "v7_head_sha": "d3333ec1b4a508df232a5aefccd6686adda745fb",
        "v7_report_sha256": "80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267",
        "record_payload_jsonl_sha256": "2f21f7655c9287bbc7424b410788f431f20f0936f445fefd26fcf6114b772dbd",
        "record_inventory_digest_sha256": "55f01d2027057f30cfc43d4ebe668779fc48413781d8146402dd53f81e2dbf31",
        "payload_inventory_digest_sha256": "c3ef21cc7a73df450798e422fb1e6d034c0d17ff38ba733cd9e25b93a31a770f",
        "record_count": EXPECTED_HISTORICAL_RECORDS,
        "source_object_count": EXPECTED_HISTORICAL_SOURCES,
        "total_payload_bytes": EXPECTED_HISTORICAL_BYTES,
    }
    _require(dict(historical) == exact_historical, "historical DATA-526 authority drift")

    bulk = config.get("bulk_code_authority")
    _require(isinstance(bulk, Mapping), "bulk authority missing")
    _require(bulk.get("source_pr") == 818, "bulk PR drift")
    _require(bulk.get("execution_head_sha") == "a045c602bfcead862f7852924fb78a5d78c992d6", "bulk head drift")
    _require(bulk.get("workflow_run_id") == 34155446113, "bulk run drift")
    _require(bulk.get("workflow_conclusion") == "success", "bulk workflow non-success")
    _require(bulk.get("artifact_id") == 10030825509, "bulk artifact drift")
    _require(
        bulk.get("artifact_digest") == "sha256:9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac",
        "bulk artifact digest drift",
    )
    _require(
        bulk.get("contract_identity_sha256") == "7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6",
        "bulk contract drift",
    )
    _require(
        bulk.get("report_identity_sha256") == "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985",
        "bulk report drift",
    )
    _require(bulk.get("source_family_count") == 6, "bulk family count drift")
    _require(bulk.get("eligible_file_count") == EXPECTED_BULK_FILES, "bulk file count drift")
    _require(bulk.get("eligible_utf8_bytes") == EXPECTED_BULK_BYTES, "bulk byte drift")

    rule = config.get("composition_rule")
    _require(isinstance(rule, Mapping), "composition rule missing")
    true_rules = (
        "historical_records_are_selected_by_source_id",
        "all_records_for_a_surviving_historical_source_are_retained",
        "bulk_file_record_id_equals_v8_source_id",
        "bulk_payload_is_strict_utf8_identity_preserved",
        "all_v8_survivor_source_ids_must_be_covered",
        "no_non_survivor_source_may_enter_record_graph",
        "output_payload_bytes_must_equal_v8_survivor_capacity",
    )
    for key in true_rules:
        _require(rule.get(key) is True, f"composition rule weakened: {key}")
    _require(rule.get("record_sort_key") == "record_id", "record sort rule drift")

    boundary = config.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim boundary missing")
    _require(boundary.get("authorized_unique_optimized_targets") == 0, "optimized targets fabricated")
    _require(boundary.get("optimizer_updates") == 0, "optimizer updates fabricated")
    for key in (
        "terminal_v8_consumed",
        "record_graph_materialized",
        "decontamination_executed",
        "tokenizer_fit_executed",
        "training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
        "raw_payloads_committed_to_repository",
        "raw_payloads_uploaded_as_public_evidence",
    ):
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")

    v8 = config.get("v8_terminal_input")
    _require(isinstance(v8, Mapping), "V8 input authority missing")
    expected_v8 = _expected_v8_terminal_input()
    if require_terminal_v8:
        _require(dict(v8) == expected_v8, "terminal V8 authority drift/substitution")
    else:
        _require(v8.get("pr") == 824, "V8 owner drift")
        _require(v8.get("required_report_schema") == V8_REPORT_SCHEMA, "V8 report schema binding drift")
        _require(v8.get("required_survivor_schema") == V8_SURVIVOR_SCHEMA, "V8 survivor schema binding drift")


def validate_historical_records(
    records: list[dict[str, Any]], authority: Mapping[str, Any]
) -> dict[str, Any]:
    _require(len(records) == int(authority["record_count"]), "historical record count drift")
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    _require(_sha256(raw) == authority["record_payload_jsonl_sha256"], "historical record JSONL identity drift")
    inventory = materialize(records)
    _require(inventory["record_count"] == int(authority["record_count"]), "historical inventory count drift")
    _require(inventory["total_payload_bytes"] == int(authority["total_payload_bytes"]), "historical payload bytes drift")
    _require(
        inventory["record_inventory_digest_sha256"] == authority["record_inventory_digest_sha256"],
        "historical record inventory digest drift",
    )
    _require(
        inventory["payload_inventory_digest_sha256"] == authority["payload_inventory_digest_sha256"],
        "historical payload inventory digest drift",
    )
    source_ids = {str(record["source_id"]) for record in records}
    _require(len(source_ids) == int(authority["source_object_count"]), "historical source-object count drift")
    return inventory


def validate_bulk_report(report: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    _require(report.get("report_identity_sha256") == authority["report_identity_sha256"], "fresh bulk report identity drift")
    _require(report.get("source_family_count") == authority["source_family_count"], "fresh bulk family drift")
    _require(report.get("eligible_file_count") == authority["eligible_file_count"], "fresh bulk file drift")
    _require(report.get("eligible_utf8_bytes") == authority["eligible_utf8_bytes"], "fresh bulk byte drift")
    _require(report.get("security", {}).get("credential_scan") == "PASS_NO_HIGH_CONFIDENCE_HITS", "fresh bulk credential scan not clean")
    core = dict(report)
    claimed = core.pop("report_identity_sha256", None)
    _require(claimed == _sha256(_canonical_ascii(core)), "fresh bulk report self-identity mismatch")


def _bulk_source_id(repository: str, path: str) -> str:
    return f"data-bulk-code1:{repository}:{path}"


def _survivor_rows(authority: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = authority.get("survivors")
    _require(isinstance(rows, list) and rows, "V8 survivor list missing")
    result: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "V8 survivor row must be an object")
        source_id = row.get("source_id")
        _require(isinstance(source_id, str) and source_id and source_id not in ids, "invalid/duplicate V8 survivor source id")
        ids.add(source_id)
        result.append(row)
    return result


def _survivor_ids(authority: Mapping[str, Any]) -> set[str]:
    return {str(row["source_id"]) for row in _survivor_rows(authority)}


def _survivor_by_id(authority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["source_id"]): row for row in _survivor_rows(authority)}


def _validate_v8_report_self_hash(report: Mapping[str, Any]) -> None:
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    _require(claimed == _sha256(_canonical_utf8(body)), "V8 report self-hash mismatch")


def _validate_survivor_rows_against_report(
    report: Mapping[str, Any], survivor: Mapping[str, Any]
) -> None:
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 nested V3 missing")
    source_rows = nested.get("sources")
    _require(isinstance(source_rows, list) and source_rows, "V8 nested source rows missing")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in source_rows:
        _require(isinstance(raw, Mapping), "V8 nested source row must be an object")
        source_id = raw.get("source_id")
        _require(
            isinstance(source_id, str) and source_id and source_id not in source_by_id,
            "invalid/duplicate V8 nested source id",
        )
        declared = raw.get("declared_capacity_bytes")
        _require(
            isinstance(declared, int) and not isinstance(declared, bool) and declared > 0,
            f"invalid V8 nested source capacity: {source_id}",
        )
        source_by_id[source_id] = raw

    rows = _survivor_rows(survivor)
    for row in rows:
        source_id = str(row["source_id"])
        declared = row.get("declared_capacity_bytes")
        _require(
            isinstance(declared, int) and not isinstance(declared, bool) and declared > 0,
            f"invalid V8 survivor source capacity: {source_id}",
        )
        source = source_by_id.get(source_id)
        _require(source is not None, f"V8 survivor references unknown nested source: {source_id}")
        for key in SURVIVOR_SOURCE_BINDING_FIELDS:
            _require(row.get(key) == source.get(key), f"V8 survivor {key} drift: {source_id}")

    pre_count = survivor.get("pre_dedup_source_object_count")
    post_count = survivor.get("post_dedup_survivor_source_object_count")
    _require(pre_count == len(source_rows) == EXPECTED_COMPOSED_SOURCES, "V8 source-row count summary drift")
    _require(post_count == len(rows), "V8 survivor-row count summary drift")

    source_bytes = sum(row["declared_capacity_bytes"] for row in source_rows)
    survivor_bytes = sum(row["declared_capacity_bytes"] for row in rows)
    _require(source_bytes == survivor.get("pre_dedup_declared_capacity_bytes"), "V8 pre-dedup row-byte summary drift")
    _require(survivor_bytes == survivor.get("post_dedup_declared_capacity_bytes"), "V8 survivor row-byte summary drift")
    _require(
        source_bytes - survivor_bytes == survivor.get("duplicate_discount_bytes"),
        "V8 duplicate-discount arithmetic drift",
    )

    clusters = survivor.get("duplicate_clusters")
    _require(isinstance(clusters, list), "V8 duplicate-cluster list missing")
    _require(len(clusters) == survivor.get("duplicate_cluster_count"), "V8 duplicate-cluster count drift")

    by_modality = survivor.get("by_modality")
    _require(isinstance(by_modality, Mapping), "V8 survivor modality summary missing")
    for modality in ("uk", "en", "code"):
        expected = {
            "source_object_count": sum(1 for row in rows if row.get("modality") == modality),
            "declared_capacity_bytes": sum(
                int(row["declared_capacity_bytes"]) for row in rows if row.get("modality") == modality
            ),
        }
        _require(by_modality.get(modality) == expected, f"V8 survivor {modality} summary drift")


def _validate_materialized_source_binding(
    survivor_row: Mapping[str, Any],
    *,
    source_id: str,
    family: str,
    modality: str,
    declared_capacity_bytes: int,
    verified_raw_sha256: str | None = None,
) -> None:
    _require(survivor_row.get("source_id") == source_id, f"survivor source-id binding drift: {source_id}")
    _require(survivor_row.get("source_family") == family, f"survivor family/materialized family drift: {source_id}")
    _require(survivor_row.get("modality") == modality, f"survivor modality/materialized modality drift: {source_id}")
    _require(
        survivor_row.get("declared_capacity_bytes") == declared_capacity_bytes,
        f"survivor bytes/materialized bytes drift: {source_id}",
    )
    if verified_raw_sha256 is not None:
        _require(
            survivor_row.get("verified_raw_sha256") == verified_raw_sha256,
            f"survivor raw hash/materialized raw hash drift: {source_id}",
        )


def validate_v8_inputs(
    report: Mapping[str, Any], survivor: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    sealed = config["v8_terminal_input"]
    _require(dict(sealed) == _expected_v8_terminal_input(), "terminal V8 authority drift/substitution")
    _require(report.get("schema_version") == V8_REPORT_SCHEMA, "V8 report schema drift")
    _require(report.get("report_sha256") == EXPECTED_V8_REPORT_SHA256, "V8 report identity drift")
    _validate_v8_report_self_hash(report)
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 nested V3 missing")
    _require(nested.get("report_sha256") == EXPECTED_V8_NESTED_V3_SHA256, "V8 nested V3 identity drift")
    _require(survivor.get("schema_version") == V8_SURVIVOR_SCHEMA, "V8 survivor schema drift")
    _require(survivor.get("v8_report_sha256") == EXPECTED_V8_REPORT_SHA256, "V8 survivor/report binding drift")
    _require(survivor.get("nested_v3_report_sha256") == EXPECTED_V8_NESTED_V3_SHA256, "V8 survivor/nested binding drift")
    _require(survivor.get("survivor_authority_sha256") == EXPECTED_V8_SURVIVOR_SHA256, "V8 survivor identity drift")
    body = dict(survivor)
    claimed = body.pop("survivor_authority_sha256", None)
    _require(claimed == _sha256(_canonical_ascii(body)), "V8 survivor self-hash mismatch")
    _require(survivor.get("pre_dedup_source_object_count") == EXPECTED_COMPOSED_SOURCES, "V8 pre-dedup source count drift")
    _require(survivor.get("pre_dedup_declared_capacity_bytes") == EXPECTED_HISTORICAL_BYTES + EXPECTED_BULK_BYTES, "V8 pre-dedup capacity drift")
    _validate_survivor_rows_against_report(report, survivor)
    truth = survivor.get("truth_boundary")
    _require(isinstance(truth, Mapping), "V8 survivor truth boundary missing")
    _require(truth.get("source_object_authority_only") is True, "V8 survivor purpose drift")
    _require(truth.get("training_record_inventory_materialized") is False, "V8 survivor prematurely claims record graph")
    _require(truth.get("authorized_training_exposure") == 0, "V8 survivor fabricated training exposure")


def _historical_source_aggregates(
    records: list[dict[str, Any]],
) -> dict[str, tuple[str, str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_id"]), []).append(record)
    aggregates: dict[str, tuple[str, str, int]] = {}
    for source_id, source_records in grouped.items():
        families = {str(record["family"]) for record in source_records}
        modalities = {str(record["modality"]) for record in source_records}
        _require(len(families) == 1, f"historical source has mixed families: {source_id}")
        _require(len(modalities) == 1, f"historical source has mixed modalities: {source_id}")
        payload_bytes = sum(len(str(record["normalized_payload"]).encode("utf-8")) for record in source_records)
        aggregates[source_id] = (next(iter(families)), next(iter(modalities)), payload_bytes)
    return aggregates


def compose_records(
    *,
    historical_records: list[dict[str, Any]],
    bulk_report: Mapping[str, Any],
    bulk_workspace: Path,
    survivor_authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    survivor_by_id = _survivor_by_id(survivor_authority)
    survivor_ids = set(survivor_by_id)
    historical_source_ids = {str(record["source_id"]) for record in historical_records}
    historical_aggregates = _historical_source_aggregates(historical_records)
    for source_id, (family, modality, payload_bytes) in historical_aggregates.items():
        if source_id in survivor_by_id:
            _validate_materialized_source_binding(
                survivor_by_id[source_id],
                source_id=source_id,
                family=family,
                modality=modality,
                declared_capacity_bytes=payload_bytes,
            )

    bulk_records: list[dict[str, Any]] = []
    bulk_source_ids: set[str] = set()
    for source in bulk_report.get("sources", []):
        _require(isinstance(source, Mapping), "fresh bulk source row invalid")
        repository = str(source["repository"])
        family = str(source["family_id"])
        repo_dir = bulk_workspace / repository.replace("/", "__")
        for item in source.get("files", []):
            _require(isinstance(item, Mapping), "fresh bulk file row invalid")
            path = str(item["path"])
            sid = _bulk_source_id(repository, path)
            _require(sid not in bulk_source_ids, f"duplicate fresh bulk source id: {sid}")
            bulk_source_ids.add(sid)
            raw = (repo_dir / path).read_bytes()
            _require(len(raw) == int(item["utf8_bytes"]), f"fresh bulk byte drift: {sid}")
            _require(_sha256(raw) == item["sha256"], f"fresh bulk hash drift: {sid}")
            try:
                payload = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise Data526V8Error(f"fresh bulk strict UTF-8 drift: {sid}") from exc
            if sid in survivor_by_id:
                _validate_materialized_source_binding(
                    survivor_by_id[sid],
                    source_id=sid,
                    family=family,
                    modality="code",
                    declared_capacity_bytes=len(raw),
                    verified_raw_sha256=str(item["sha256"]),
                )
                bulk_records.append(
                    {
                        "record_id": sid,
                        "source_id": sid,
                        "family": family,
                        "modality": "code",
                        "normalized_payload": payload,
                    }
                )

    _require(len(bulk_source_ids) == EXPECTED_BULK_FILES, "fresh bulk source-object count drift")
    all_sources = historical_source_ids | bulk_source_ids
    _require(len(historical_source_ids) == EXPECTED_HISTORICAL_SOURCES, "historical source-object count drift")
    _require(len(all_sources) == EXPECTED_COMPOSED_SOURCES, "historical/bulk source namespaces collide")
    _require(survivor_ids <= all_sources, "V8 survivor authority references an unavailable source")

    selected = [copy.deepcopy(record) for record in historical_records if str(record["source_id"]) in survivor_ids]
    selected.extend(bulk_records)
    selected.sort(key=lambda record: str(record["record_id"]))
    covered = {str(record["source_id"]) for record in selected}
    _require(covered == survivor_ids, "record graph does not cover exactly the V8 survivor source set")

    payload_bytes = sum(len(str(record["normalized_payload"]).encode("utf-8")) for record in selected)
    expected_bytes = survivor_authority.get("post_dedup_declared_capacity_bytes")
    _require(payload_bytes == expected_bytes, "record payload bytes do not reproduce V8 survivor capacity")
    return selected


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> str:
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose DATA-526 records from terminal V8 survivor authority")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--historical-records-jsonl", type=Path, required=True)
    parser.add_argument("--bulk-report", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--v8-survivors", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--execution-head-sha", required=True)
    args = parser.parse_args()
    execution_head_sha = _validated_git_sha(args.execution_head_sha)

    config = _read_json(args.config)
    verify_config(config, require_terminal_v8=True)
    historical_records = load_jsonl(args.historical_records_jsonl)
    validate_historical_records(historical_records, config["historical_v7_record_authority"])
    bulk_report = _read_json(args.bulk_report)
    validate_bulk_report(bulk_report, config["bulk_code_authority"])
    v8_report = _read_json(args.v8_report)
    survivor = _read_json(args.v8_survivors)
    validate_v8_inputs(v8_report, survivor, config)

    records = compose_records(
        historical_records=historical_records,
        bulk_report=bulk_report,
        bulk_workspace=args.bulk_workspace,
        survivor_authority=survivor,
    )
    records_sha = _write_jsonl(records, args.records_jsonl)
    inventory = materialize(records)
    _require(inventory["total_payload_bytes"] == survivor["post_dedup_declared_capacity_bytes"], "inventory/V8 capacity mismatch")
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.write_bytes(canonical_json(inventory) + b"\n")

    core = {
        "schema_version": "12-6.data526-v8-record-composition-evidence.v1",
        "historical_record_authority": copy.deepcopy(config["historical_v7_record_authority"]),
        "bulk_report_identity_sha256": bulk_report["report_identity_sha256"],
        "v8_report_sha256": v8_report["report_sha256"],
        "v8_nested_v3_report_sha256": v8_report["dedup_v3"]["report_sha256"],
        "v8_survivor_authority_sha256": survivor["survivor_authority_sha256"],
        "record_payload_jsonl_sha256": records_sha,
        "record_inventory_digest_sha256": inventory["record_inventory_digest_sha256"],
        "payload_inventory_digest_sha256": inventory["payload_inventory_digest_sha256"],
        "record_count": inventory["record_count"],
        "source_object_count": survivor["post_dedup_survivor_source_object_count"],
        "total_payload_bytes": inventory["total_payload_bytes"],
        "raw_payloads_emitted_to_public_evidence": False,
        "decontamination_executed": False,
        "authorized_unique_optimized_targets": 0,
        "tokenizer_fit_executed": False,
        "training_executed": False,
        "optimizer_updates": 0,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
        "execution_head_sha": execution_head_sha,
    }
    evidence = {**core, "evidence_identity_sha256": _sha256(canonical_json(core))}
    args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_json.write_bytes(canonical_json(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
