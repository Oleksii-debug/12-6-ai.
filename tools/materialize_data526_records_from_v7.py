from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/data/data526_record_materialization_v5.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "12-6.data526-record-materialization.v5":
        raise ValueError("unsupported DATA-526 V5 materialization config")
    projected = copy.deepcopy(config)
    expected = projected.pop("evidence_identity_sha256", None)
    if expected != sha256(canonical_json(projected)):
        raise ValueError("materialization config identity mismatch")
    if config.get("execution_profile") != "LOCAL_FREE":
        raise ValueError("LOCAL_FREE boundary weakened")
    boundary = config["claim_boundary"]
    if boundary["authorized_unique_optimized_targets"] != 0:
        raise ValueError("optimized targets fabricated before decontamination/packing")
    for key in (
        "decontamination_executed",
        "tokenizer_fit_executed",
        "training_executed",
        "paid_compute_used",
        "final_test_payload_accessed",
        "raw_payloads_committed_to_repository",
        "raw_payloads_uploaded_as_public_evidence",
    ):
        if boundary[key] is not False:
            raise ValueError(f"claim boundary weakened: {key}")


def _normalize_kmu(payload: bytes) -> bytes:
    text = payload.decode("utf-8", errors="strict")
    text = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(
        line
        for line in (" ".join(raw_line.split()) for raw_line in text.splitlines())
        if line
    ).strip() + "\n"
    return normalized.encode("utf-8")


def _verify_payload(payload: bytes, spec: dict[str, Any], *, source_id: str) -> None:
    if len(payload) != int(spec["payload_bytes"]):
        raise ValueError(f"payload byte-count drift: {source_id}")
    if sha256(payload) != spec["payload_sha256"]:
        raise ValueError(f"payload SHA-256 drift: {source_id}")


def _split_cpython(payload: bytes, spec: dict[str, Any]) -> list[bytes]:
    if spec.get("comparison_separator") != "DOUBLE_LF":
        raise ValueError("unsupported CPython comparison separator")
    parts = payload.split(b"\n\n")
    hashes = [sha256(part) for part in parts]
    if hashes != spec["accepted_normalized_sha256"]:
        raise ValueError("CPython accepted chunk identity/order drift")
    if len(parts) != int(spec["accepted_chunk_count"]):
        raise ValueError("CPython accepted chunk-count drift")
    if sum(len(part) for part in parts) != int(spec["accepted_capacity_bytes"]):
        raise ValueError("CPython accepted capacity drift")
    return parts


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_v7(root: Path):
    src = str((root / "src").resolve())
    sys.path.insert(0, src)
    try:
        from twelve_six.data import cross_source_capacity_audit_v7 as v7
    except Exception:
        sys.path.remove(src)
        raise
    return v7, src


def _capture_v7_materialization(v7: Any, root: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    paths = (
        root / "configs/data/next100_065_cross_source_dedup_v3.json",
        root / "configs/data/next100_065b_cross_source_dedup_v4.json",
        root / "configs/data/next100_065c_cross_source_dedup_v5.json",
        root / "configs/data/next100_065d_cross_source_dedup_v6.json",
        root / "configs/data/next100_065e_cross_source_dedup_v7.json",
    )
    base, v4_ext, v5_cfg, v6_cfg, v7_cfg = v7.load_inputs(*paths)
    original = v7.v6.v3.audit_payloads
    captured: list[tuple[dict[str, Any], dict[str, bytes]]] = []

    def capture(inventory: dict[str, Any], payloads: dict[str, bytes]) -> dict[str, Any]:
        captured.append((copy.deepcopy(inventory), dict(payloads)))
        return original(inventory, payloads)

    v7.v6.v3.audit_payloads = capture
    try:
        report = v7.audit_live(base, v4_ext, v5_cfg, v6_cfg, v7_cfg)
        v7.verify_report(report)
    finally:
        v7.v6.v3.audit_payloads = original
    if len(captured) != 1:
        raise ValueError(f"expected exactly one V7 payload audit call, got {len(captured)}")
    inventory, payloads = captured[0]
    return inventory, payloads, report


def _load_data213_payloads(archive_path: Path, config: dict[str, Any]) -> dict[str, bytes]:
    raw_archive = archive_path.read_bytes()
    authority = config["data213_normalized_artifact"]
    if sha256(raw_archive) != authority["artifact_zip_sha256"]:
        raise ValueError("DATA-213 artifact ZIP identity drift")
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path) as package:
        for source_id, spec in authority["sources"].items():
            payload = package.read(spec["path"])
            _verify_payload(payload, spec, source_id=source_id)
            result[source_id] = payload
    return result


def _dedup_sources_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = report["dedup_v3"]["sources"]
    result = {str(item["source_id"]): item for item in sources}
    if len(result) != len(sources):
        raise ValueError("V7 source IDs are not unique")
    return result


def _record(source: dict[str, Any], payload: bytes, record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": str(source["source_id"]),
        "family": str(source["source_family"]),
        "modality": str(source["modality"]),
        "normalized_payload": payload.decode("utf-8", errors="strict"),
    }


def materialize_records(
    *,
    config: dict[str, Any],
    inventory: dict[str, Any],
    payloads: dict[str, bytes],
    v7_report: dict[str, Any],
    data213_payloads: dict[str, bytes],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    expected = config["expected_materialization"]
    if len(inventory["sources"]) != int(expected["source_object_count"]):
        raise ValueError("source-object count drift")
    source_by_id = {str(item["source_id"]): item for item in inventory["sources"]}
    if set(source_by_id) != set(payloads):
        raise ValueError("captured V7 inventory/payload key mismatch")
    report_sources = _dedup_sources_by_id(v7_report)
    if set(report_sources) != set(source_by_id):
        raise ValueError("captured V7 source set differs from terminal report")

    data213_specs = config["data213_normalized_artifact"]["sources"]
    kmu_specs = config["kmu_authority"]["sources"]
    cpython = config["cpython_authority"]
    cpython_id = cpython["source_id"]
    records: list[dict[str, str]] = []
    direct_count = direct_bytes = 0
    modality_bytes = {"uk": 0, "en": 0, "code": 0}

    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        payload = payloads[source_id]
        if source_id in data213_specs:
            payload = data213_payloads[source_id]
            records.append(_record(source, payload, source_id))
        elif source_id in kmu_specs:
            payload = _normalize_kmu(payload)
            _verify_payload(payload, kmu_specs[source_id], source_id=source_id)
            records.append(_record(source, payload, source_id))
        elif source_id == cpython_id:
            for index, chunk in enumerate(_split_cpython(payload, cpython), start=1):
                chunk_hash = sha256(chunk)
                record_id = f"{source_id}#accepted-{index:02d}-{chunk_hash[:12]}"
                records.append(_record(source, chunk, record_id))
                modality_bytes[str(source["modality"])] += len(chunk)
            continue
        else:
            report_source = report_sources[source_id]
            declared = int(report_source["declared_capacity_bytes"])
            comparison = int(report_source["comparison_payload_bytes"])
            if comparison != declared or len(payload) != declared:
                raise ValueError(f"unsupported post-filter source requires explicit authority adapter: {source_id}")
            if sha256(payload) != report_source["comparison_payload_sha256"]:
                raise ValueError(f"V7 direct payload SHA drift: {source_id}")
            records.append(_record(source, payload, source_id))
            direct_count += 1
            direct_bytes += len(payload)
        modality_bytes[str(source["modality"])] += len(payload)

    records.sort(key=lambda item: item["record_id"])
    if len(records) != int(expected["record_count"]):
        raise ValueError(f"record-count drift: {len(records)}")
    total_payload_bytes = sum(len(item["normalized_payload"].encode("utf-8")) for item in records)
    if total_payload_bytes != int(expected["total_payload_bytes"]):
        raise ValueError(f"payload capacity drift: {total_payload_bytes}")
    if direct_count != int(expected["direct_v7_source_count"]):
        raise ValueError(f"direct V7 source-count drift: {direct_count}")
    if direct_bytes != int(expected["direct_v7_payload_bytes"]):
        raise ValueError(f"direct V7 payload-byte drift: {direct_bytes}")
    if modality_bytes != expected["by_modality_payload_bytes"]:
        raise ValueError(f"modality payload-byte drift: {modality_bytes}")
    return records, {
        "direct_v7_source_count": direct_count,
        "direct_v7_payload_bytes": direct_bytes,
        "record_count": len(records),
        "total_payload_bytes": total_payload_bytes,
    }


def _write_jsonl(records: list[dict[str, str]], path: Path) -> str:
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256(raw)


def _build_text_free_inventory(records: list[dict[str, str]]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from tools.materialize_data526_record_inventory_v1 import materialize

    result = materialize(records)
    if "normalized_payload" in json.dumps(result, ensure_ascii=False):
        raise ValueError("text-free inventory leaked payload")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize exact DATA-526 normalized records from terminal V7 authorities")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--data213-zip", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    verify_config(config)
    expected_v7_head = config["v7_authority"]["head_sha"]
    observed_v7_head = _git_head(args.v7_root)
    if observed_v7_head != expected_v7_head:
        raise ValueError(f"V7 checkout drift: {observed_v7_head}")

    v7, inserted_path = _load_v7(args.v7_root)
    try:
        inventory, payloads, v7_report = _capture_v7_materialization(v7, args.v7_root)
    finally:
        if inserted_path in sys.path:
            sys.path.remove(inserted_path)
    if v7_report["report_sha256"] != config["v7_authority"]["report_sha256"]:
        raise ValueError("live V7 report identity drift")

    data213_payloads = _load_data213_payloads(args.data213_zip, config)
    records, stats = materialize_records(
        config=config,
        inventory=inventory,
        payloads=payloads,
        v7_report=v7_report,
        data213_payloads=data213_payloads,
    )
    records_sha = _write_jsonl(records, args.records_jsonl)
    text_free = _build_text_free_inventory(records)
    if text_free["record_count"] != config["expected_materialization"]["record_count"]:
        raise ValueError("text-free inventory record-count drift")
    if text_free["total_payload_bytes"] != config["expected_materialization"]["total_payload_bytes"]:
        raise ValueError("text-free inventory payload-byte drift")
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.write_bytes(canonical_json(text_free) + b"\n")

    core = {
        "schema_version": "12-6.data526-record-materialization-evidence.v5",
        "config_identity_sha256": config["evidence_identity_sha256"],
        "v7_head_sha": expected_v7_head,
        "v7_report_sha256": v7_report["report_sha256"],
        "data213_artifact_zip_sha256": config["data213_normalized_artifact"]["artifact_zip_sha256"],
        "record_payload_jsonl_sha256": records_sha,
        "record_inventory_digest_sha256": text_free["record_inventory_digest_sha256"],
        "payload_inventory_digest_sha256": text_free["payload_inventory_digest_sha256"],
        "record_count": text_free["record_count"],
        "total_payload_bytes": text_free["total_payload_bytes"],
        "stats": stats,
        "raw_payloads_emitted_to_public_evidence": False,
        "authorized_unique_optimized_targets": 0,
        "decontamination_executed": False,
        "tokenizer_fit_executed": False,
        "training_executed": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "final_test_payload_accessed": False,
    }
    evidence = {**core, "evidence_identity_sha256": sha256(canonical_json(core))}
    args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_json.write_bytes(canonical_json(evidence) + b"\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
