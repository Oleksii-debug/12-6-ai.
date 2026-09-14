#!/usr/bin/env python3
"""Fail-closed validator for the Rada_Trees acquisition probe."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_rada_trees_acquisition_probe_v1.json"

EXPECTED_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_DATASET = "uacorpus/Rada_Trees"
EXPECTED_FAMILY = "ua.rada.plenary-transcripts.1990-2024"
EXISTING_LAWS_FAMILY = "ua.rada.open-data.laws-texts"


def validate(value: dict[str, Any]) -> None:
    assert value["schema_version"] == "12-6.d03-rada-trees-acquisition-probe.v1"
    assert value["worker_id"] == "D03-RADA-TREES-ACQUISITION-PROBE-20260826"
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["purpose"] == "DISCOVERY_AND_IMMUTABLE_ACQUISITION_PLANNING_ONLY"

    source = value["source"]
    assert source["dataset"] == EXPECTED_DATASET
    assert re.fullmatch(r"[0-9a-f]{40}", source["observed_head_sha"])
    assert source["observed_head_sha"] == EXPECTED_HEAD
    assert source["observed_head_short"] == EXPECTED_HEAD[:7]
    assert source["language"] == "uk"
    assert source["modality"] == "text"
    assert source["reported_tokens_approx"] == 88_000_000
    assert set(source["reported_formats"]) == {
        "plain_text_original_transcripts",
        "universal_dependencies_annotation",
        "nlp_uk_annotation",
    }

    files = value["observed_repository_files"]
    assert [item["path"] for item in files] == ["Rada_Trees.7z", "rada_xtag_texts.7z"]
    assert len({item["path"] for item in files}) == len(files)
    for item in files:
        assert item["exact_blob_identity"] is None
        assert item["training_capacity_credit_bytes"] == 0
        assert isinstance(item["displayed_size"], str) and item["displayed_size"].endswith(" MB")
    assert files[0]["role"] == "PRIMARY_ARCHIVE_CANDIDATE_CONTENTS_NOT_YET_INVENTORIED"
    assert files[1]["role"] == "ANNOTATION_OR_DERIVED_ARCHIVE_HOLD_UNTIL_CONTENTS_AND_LINEAGE_AUDIT"

    rights = value["rights"]
    assert rights["dataset_card_license"] == "CC-BY-4.0"
    assert rights["redistribution_obligation"] == "ATTRIBUTION_REQUIRED"
    assert rights["model_training_decision"] == "RETEST_EXACT_ARCHIVE_AND_MEMBER_SCOPE_BEFORE_ADMISSION"
    assert rights["evaluation_decision"] == "NOT_SEPARATELY_ADMITTED"
    assert rights["final_test_decision"] == "PROHIBITED"

    lineage = value["lineage"]
    assert lineage["provisional_family_id"] == EXPECTED_FAMILY
    assert lineage["existing_family_not_alias"] == EXISTING_LAWS_FAMILY
    assert lineage["provisional_family_id"] != lineage["existing_family_not_alias"]
    assert lineage["one_credit_per_underlying_transcript_rule"] is True
    assert lineage["annotation_layers_create_new_family_credit"] is False
    assert lineage["cross_source_lineage_dedup_required"] is True
    surfaces = lineage["known_or_reported_same_lineage_surfaces"]
    assert len(surfaces) == len(set(surfaces)) >= 5
    assert any("ParlaMint-UA" in item for item in surfaces)
    assert any("GRAC" in item for item in surfaces)

    policy = value["acquisition_policy"]
    assert policy["preferred_training_payload"] == "PLAIN_TEXT_TRANSCRIPT_LAYER_ONLY"
    assert policy["annotated_payload_default"] == "HOLD_ZERO_CREDIT"
    for key in (
        "required_exact_archive_sha256",
        "required_archive_member_inventory",
        "required_member_sha256",
        "required_member_path_normalization",
        "required_duplicate_member_rejection",
        "required_plain_text_vs_annotation_classification",
        "required_attribution_manifest",
        "required_member_level_language_quality_privacy_scan",
    ):
        assert policy[key] is True
    assert policy["max_single_member_bytes"] > 0
    assert policy["max_total_uncompressed_bytes"] > policy["max_single_member_bytes"]

    steps = value["downstream_required"]
    assert len(steps) == len(set(steps)) == 10
    assert steps[0] == "PIN_EXACT_DATASET_HEAD_AND_XET_OR_LFS_ARCHIVE_OBJECT_IDENTITY"
    assert steps[-1] == "ONLY_THEN_PROPOSE_NONZERO_SOURCE_CAPACITY_CREDIT"

    boundary = value["claim_boundary"]
    assert boundary["archive_downloaded"] is False
    assert boundary["archive_sha256_pinned"] is False
    assert boundary["archive_members_inventoried"] is False
    assert boundary["bulk_source_admitted"] is False
    assert boundary["family_independence_terminal"] is False
    assert boundary["normalized_capacity_claimed"] is False
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["training_exposure_authorized"] is False
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["paid_compute_used"] is False
    assert boundary["safe_result"] == "HIGH_LEVERAGE_SOURCE_DISCOVERED_EXACT_ACQUISITION_REQUIRED"


def load_and_validate(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate(value)
    return value


def main() -> int:
    value = load_and_validate()
    print("D03_RADA_TREES_PROBE=PASS_FAIL_CLOSED")
    print("OBSERVED_HEAD=" + value["source"]["observed_head_sha"])
    print("REPORTED_TOKENS_APPROX=88000000")
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=PIN_ARCHIVE_AND_MEMBER_IDENTITIES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
