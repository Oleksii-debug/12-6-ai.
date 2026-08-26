#!/usr/bin/env python3
"""Fail-closed validator for the PluG/PluG2 historical Ukrainian source probe."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_plug2_historical_ua_probe_v1.json"
EXPECTED_REPO = "Dandelliony/pluperfect_grac"
EXPECTED_COMMIT = "27e503ecc2b553d52bfc121b8320144bb25294d8"
EXPECTED_TREE = "99dec65503cdb9a0b0865901145b7a4311a8bdd2"
EXPECTED_FAMILY = "ua.plug.historical-literature-1816-1954"
EXPECTED_ROOT_OBJECTS = {
    "PluG_metadata.psv": ("blob", "d3d4303030b5a0ddb0f5f42b519e8f1660daee67", 18513955),
    "PluG_texts": ("tree", "815488d4bdd56f0e7b6d608934e4d5d446bc120f", None),
    "PluG2_metadata.psv": ("blob", "ba51dcc47c0ca5cda9cc6926ae5f32a6d62cb1d2", 17473448),
    "PluG2_texts": ("tree", "10985cdca9dd6c25f1b1e2d087b772104aa9a656", None),
}


def validate(value: dict[str, Any]) -> None:
    assert value["schema_version"] == "12-6.d03-plug2-historical-ua-probe.v1"
    assert value["worker_id"] == "D03-PLUG2-HISTORICAL-UA-PROBE-20260826"
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["purpose"] == "DISCOVERY_AND_IMMUTABLE_ACQUISITION_PLANNING_ONLY"

    source = value["source"]
    assert source["repository"] == EXPECTED_REPO
    assert source["pinned_commit_sha"] == EXPECTED_COMMIT
    assert source["pinned_root_tree_sha"] == EXPECTED_TREE
    assert re.fullmatch(r"[0-9a-f]{40}", source["pinned_commit_sha"])
    assert re.fullmatch(r"[0-9a-f]{40}", source["pinned_root_tree_sha"])
    assert source["language"] == "uk"
    assert source["modality"] == "text"
    assert source["reported_plug_period"] == "1816-1954"
    assert source["reported_plug_tokens"] == 58_676_313
    assert source["reported_plug_file_count"] == 42_000
    assert source["reported_plug2_tokens"] == 73_900_596
    assert source["reported_unique_authors"] == 7_590
    assert source["reported_unique_translators"] == 44
    assert set(source["reported_historical_orthographies"]) == {
        "Kulishivka",
        "Zhelekhivka",
        "Skrypnykivka",
    }

    objects = value["pinned_root_objects"]
    assert len(objects) == len(EXPECTED_ROOT_OBJECTS)
    assert len({item["path"] for item in objects}) == len(objects)
    by_path = {item["path"]: item for item in objects}
    assert set(by_path) == set(EXPECTED_ROOT_OBJECTS)
    for path, (expected_type, expected_oid, expected_bytes) in EXPECTED_ROOT_OBJECTS.items():
        item = by_path[path]
        assert item["type"] == expected_type
        assert item["oid"] == expected_oid
        assert re.fullmatch(r"[0-9a-f]{40}", item["oid"])
        if expected_bytes is None:
            assert "bytes" not in item
        else:
            assert item["bytes"] == expected_bytes
        assert item["training_capacity_credit_bytes"] == 0

    rights = value["rights"]
    assert rights["dataset_license_family"] == "CC-BY"
    assert rights["exact_cc_by_version_pinned"] is False
    assert rights["redistribution_obligation"] == "ATTRIBUTION_REQUIRED"
    assert rights["model_training_decision"] == (
        "HOLD_UNTIL_EXACT_LICENSE_VERSION_AND_MEMBER_PROVENANCE"
    )
    assert rights["evaluation_decision"] == "NOT_SEPARATELY_ADMITTED"
    assert rights["final_test_decision"] == "PROHIBITED"

    lineage = value["lineage"]
    assert lineage["provisional_family_id"] == EXPECTED_FAMILY
    assert lineage["plug_and_plug2_share_one_family"] is True
    assert lineage["annotation_or_repository_version_does_not_create_family_credit"] is True
    assert lineage["known_parent_or_overlap_surface"] == "GRAC"
    assert lineage["cross_source_lineage_dedup_required"] is True
    assert lineage["independent_from_other_literary_corpora"] == "NOT_ESTABLISHED"

    quality = value["quality_strata"]
    assert quality["modern_standard_ukrainian"] == "DO_NOT_ASSUME"
    assert quality["historical_orthography_present"] is True
    assert quality["historical_language_variant_present"] is True
    assert quality["ocr_origin_present"] is True
    assert quality["required_period_provenance"] is True
    assert quality["required_orthography_label"] is True
    assert quality["required_ocr_quality_scan"] is True
    assert quality["family_cap_must_apply_after_quality_and_lineage_dedup"] is True
    assert quality["default_training_mix_status"] == "HOLD_UNTIL_STRATIFIED"

    policy = value["acquisition_policy"]
    assert policy["preferred_first_payload"] == "PluG_texts"
    assert policy["plug2_default"] == (
        "HOLD_UNTIL_HISTORICAL_ORTHOGRAPHY_ABLATION_OR_EXPLICIT_CAP"
    )
    for key in (
        "required_exact_commit_and_tree",
        "required_recursive_tree_complete",
        "required_text_blob_inventory",
        "required_blob_oid_and_size",
        "required_metadata_row_linkage",
        "required_member_sha256_after_download",
        "required_attribution_manifest",
        "required_language_quality_privacy_scan",
        "required_exact_and_near_lineage_dedup",
        "required_evaluation_decontamination",
    ):
        assert policy[key] is True
    assert policy["max_single_text_blob_bytes"] == 50_000_000

    steps = value["downstream_required"]
    assert len(steps) == len(set(steps)) == 11
    assert steps[0] == "LIVE_VERIFY_PINNED_COMMIT_ROOT_TREE_AND_ROOT_OBJECTS"
    assert steps[-1] == "ONLY_THEN_PROPOSE_NONZERO_SOURCE_CAPACITY_CREDIT"

    boundary = value["claim_boundary"]
    for key in (
        "full_recursive_inventory_materialized",
        "text_members_downloaded",
        "member_sha256_pinned",
        "exact_license_version_pinned",
        "bulk_source_admitted",
        "normalized_capacity_claimed",
        "training_exposure_authorized",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        assert boundary[key] is False
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["optimizer_updates"] == 0
    assert boundary["safe_result"] == (
        "HIGH_LEVERAGE_HISTORICAL_UA_SOURCE_DISCOVERED_EXACT_AUDIT_REQUIRED"
    )


def load_and_validate(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    validate(value)
    return value


def main() -> int:
    value = load_and_validate()
    print("D03_PLUG2_HISTORICAL_UA_PROBE=PASS_FAIL_CLOSED")
    print("PINNED_COMMIT=" + value["source"]["pinned_commit_sha"])
    print("PINNED_ROOT_TREE=" + value["source"]["pinned_root_tree_sha"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=LIVE_TREE_INVENTORY_AND_RIGHTS_BINDING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
