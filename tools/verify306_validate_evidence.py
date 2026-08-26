#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "evidence/verify306/verify306_corpus_quality_v03.json"
DATA300_PATH = ROOT / "configs/data/data300_corpus_v03_frozen_build_contract_v2.json"
DATA301_PATH = ROOT / "configs/data/data301_corpus_v03_terminal_build_v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha_without_identity(report: dict) -> str:
    core = dict(report)
    core.pop("evidence_identity_sha256", None)
    core.pop("evidence_identity_scope", None)
    payload = (
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    report = load(REPORT_PATH)
    data300 = load(DATA300_PATH)
    data301 = load(DATA301_PATH)

    assert report["schema_version"] == "12-6.verify306-corpus-quality-v03.v1"
    assert report["worker_id"] == "VERIFY-306-CORPUS-QUALITY-V03"
    assert report["repository"] == "Oleksii-debug/12-6-ai."
    assert report["execution_profile"] == "LOCAL_FREE"
    assert report["final_test_bytes_read"] is False
    assert report["final_test_outcomes_read"] is False

    target = report["audit_target"]
    assert target["data300_contract_identity_sha256"] == data300["contract_identity_sha256"]
    assert target["data301_evidence_identity_sha256"] == data301["evidence_identity_sha256"]
    assert target["data301_head_sha"] == "8820ba1b255f6bb95c7db0531fd846078a1aae01"
    assert data301["base_data300"]["contract_identity_sha256"] == data300["contract_identity_sha256"]

    inventory = data300["exact_training_candidate_inventory"]
    assert inventory["source_count"] == target["source_count"] == 5
    assert inventory["independent_family_count"] == target["independent_family_count"] == 4
    assert inventory["admitted_source_bytes"] == target["normalized_unique_bytes"] == 183061

    source_lengths = {item["source_id"]: item["normalized_bytes"] for item in inventory["sources"]}
    assert source_lengths == report["document_lengths"]["sources"]
    values = sorted(source_lengths.values())
    length_stats = report["document_lengths"]["source_normalized_utf8_bytes"]
    assert length_stats["min"] == min(values)
    assert length_stats["median"] == statistics.median(values)
    assert length_stats["max"] == max(values)
    assert abs(length_stats["mean"] - statistics.mean(values)) < 1e-9

    expected_family_bytes: dict[str, int] = {}
    for item in inventory["sources"]:
        expected_family_bytes[item["family"]] = expected_family_bytes.get(item["family"], 0) + item["normalized_bytes"]
    observed_families = report["retained_rejected"]["by_family"]
    assert set(expected_family_bytes) == set(observed_families)
    for family, expected_bytes in expected_family_bytes.items():
        row = observed_families[family]
        assert row["bytes"] == expected_bytes
        assert row["retained_bytes"] + row["rejected_bytes"] == row["bytes"]
        assert 0 <= row["rejected_records"] <= row["records"]

    modes = report["retained_rejected"]["by_mode"]
    assert modes["en"]["bytes"] == inventory["by_stratum_bytes"]["en"]
    assert modes["uk"]["bytes"] == inventory["by_stratum_bytes"]["uk"]
    assert modes["code"]["bytes"] == inventory["by_stratum_bytes"]["code"]
    for row in modes.values():
        assert row["retained_bytes"] + row["rejected_bytes"] == row["bytes"]

    overall = report["retained_rejected"]["overall"]
    assert overall["bytes"] == sum(row["bytes"] for row in modes.values()) == 183061
    assert overall["retained_bytes"] == sum(row["retained_bytes"] for row in modes.values()) == 178064
    assert overall["rejected_bytes"] == sum(row["rejected_bytes"] for row in modes.values()) == 4997
    assert overall["records"] == sum(row["records"] for row in modes.values()) == 72
    assert overall["rejected_records"] == sum(row["rejected_records"] for row in modes.values()) == 2

    enc = report["encoding_language_code"]
    assert enc["text_decode_failures"] == 0
    assert enc["text_sources_wrong_language"] == 0
    assert enc["code_sources_strict_utf8_failures"] == 0
    assert enc["code_full_source_parse_failures"] == 0
    assert enc["code_full_source_parse_valid_sources"] == 2

    malformed = report["malformation_and_modality"]
    assert malformed["pathologically_malformed_families"] == []
    assert malformed["structural_malformed_rejection_records"] == 0
    assert malformed["modality_destroyed"] is False
    assert not any(malformed["systematic_deletion_detected"].values())

    authority = report["quality_authority_rerun"]
    assert authority["workflow_conclusion"] == "success"
    assert authority["head_sha"] == "e44ac40c071834d677e00255d4f8e862f4a7ac4b"
    assert authority["report_sha256"] == "827c4ca7b1f6c1e86fab563ffe5f9b4259d8a3ab5ecce50b9f8ded8e060094b1"
    assert authority["incumbent_policy_sha256"] == "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
    assert authority["model_results_read"] is False
    assert authority["final_test_outcomes_read"] is False

    verdict = report["verdict"]
    assert verdict["prebuild_quality"] == "PASS_NO_PATHOLOGICAL_FAMILY_NO_MODALITY_DESTRUCTION"
    assert verdict["g05_exact_final_record_coverage"] == "NOT_CLOSED_NO_FINAL_RECORD_MATERIALIZATION"
    assert verdict["quality_pathology_requires_exclusion"] is False
    assert verdict["corpus_freeze_authorized"] is False
    assert verdict["status"] == "BLOCK_FREEZE"
    assert data301["terminal_verdict"]["corpus_frozen"] is False
    assert data301["terminal_verdict"]["corpus_identity"] is None
    assert data301["terminal_verdict"]["shard_identity"] is None

    digest = canonical_sha_without_identity(report)
    assert digest == report["evidence_identity_sha256"]

    print("VERIFY306_EVIDENCE_SHA256=" + digest)
    print("VERIFY306_PREBUILD_QUALITY=" + verdict["prebuild_quality"])
    print("VERIFY306_G05=" + verdict["g05_exact_final_record_coverage"])
    print("VERIFY306_VERDICT=" + verdict["status"])


if __name__ == "__main__":
    main()
