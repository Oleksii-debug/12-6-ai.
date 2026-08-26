#!/usr/bin/env python3
"""Validate the fail-closed NEXT100-110 Ukrainian heritage source audit.

The default path is offline and deterministic. ``--live-card`` performs one
bounded metadata-only read of README.md at the exact Hugging Face revision; it
never downloads parquet payloads and never grants training capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "12-6.next100-110-ua-cultural-heritage-source-audit.v1"
WORKER = "AUTODEV-NEXT100-110-UA-CULTURAL-HERITAGE"
REVISION = "54fc0f867ea4029b4f4155baa934375095d3d992"
DATASET = "PleIAs/Ukrainian-CulturalHeritage-Books"
MAX_LIVE_CARD_BYTES = 64 * 1024


class AuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _identity(config: dict[str, Any]) -> str:
    core = dict(config)
    core.pop("authority_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


def _validate_live_card(config: dict[str, Any]) -> None:
    source = config["source"]
    url = (
        "https://huggingface.co/datasets/"
        f"{source['dataset']}/resolve/{source['revision']}/README.md"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-NEXT100-110/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - exact HTTPS host/revision
        payload = response.read(MAX_LIVE_CARD_BYTES + 1)
    _require(len(payload) <= MAX_LIVE_CARD_BYTES, "dataset card exceeds bounded metadata read")
    text = payload.decode("utf-8", errors="strict")
    required_markers = (
        "19,574 digitized files",
        "462M words",
        "author is dead for more than 70 years",
        "published prior to 1884",
        "training of Large Language Models",
        "public domain in all regions",
        "Optical Character Recognition (OCR)",
    )
    missing = [marker for marker in required_markers if marker not in text]
    _require(not missing, f"exact-revision dataset card markers drifted: {missing}")


def validate(config: dict[str, Any], *, live_card: bool = False) -> None:
    _require(config.get("schema_version") == SCHEMA, "schema drift")
    _require(config.get("worker_id") == WORKER, "worker drift")
    _require(config.get("execution_profile") == "LOCAL_FREE_METADATA_ONLY", "execution profile drift")
    _require(config.get("authority_identity_sha256") == _identity(config), "authority identity mismatch")

    source = config["source"]
    _require(source.get("dataset") == DATASET, "dataset identity drift")
    _require(source.get("revision") == REVISION, "dataset revision drift")
    _require(source.get("format") == "parquet", "source format drift")
    _require(source.get("language_label") == "Ukrainian", "source language-label drift")
    _require(source.get("row_count_published") == 19_574, "published row count drift")
    _require(source.get("upstream_digitized_file_count_published") == 19_574, "published file-count drift")
    _require(source.get("total_file_size_display_published") == "2.91 GB", "published size-display drift")
    _require(source.get("word_count_display_published") == "462M words", "published word-count display drift")

    rights = config["published_rights_claims"]
    _require(rights.get("collection_statement") == "most_of_collection_public_domain", "collection rights wording drift")
    _require(rights.get("curation_statement") == "author_dead_more_than_70_years", "curation rights wording drift")
    _require(
        rights.get("initial_cutoff_statement")
        == "as_of_march_2024_retained_exclusively_titles_published_before_1884",
        "initial cutoff wording drift",
    )
    _require(rights.get("license_statement") == "entire_collection_public_domain_in_all_regions", "license wording drift")

    conflict = config["observed_live_card_conflict"]
    _require(conflict.get("present") is True, "known rights/card conflict suppressed")
    _require(conflict.get("blanket_training_admission_allowed") is False, "blanket admission illegally enabled")
    dates = conflict.get("example_dates_seen")
    _require(dates == [1919, 1935, 1947], "observed post-cutoff examples drift")
    _require(all(year > 1884 for year in dates), "post-cutoff conflict examples are not post-cutoff")

    quality = config["quality_risks"]
    for key in (
        "ocr_generated",
        "dataset_card_acknowledges_ocr_errors",
        "dataset_card_acknowledges_unwanted_headers_page_counts",
        "dataset_card_acknowledges_multicolumn_table_formatting_risk",
    ):
        _require(quality.get(key) is True, f"quality risk suppressed: {key}")

    context = config["project_capacity_context"]
    _require(context.get("research_corpus_v1_planning_floor_bytes") == 20_000_000, "20M planning floor drift")
    _require(context.get("target_uk_share") == 0.45, "UK mixture drift")
    _require(context.get("target_uk_source_floor_bytes") == 9_000_000, "UK source-floor drift")
    _require(context.get("current_terminal_uk_source_bytes_before_successor_global_dedup") == 100_856, "current UK authority drift")
    _require(
        context.get("current_uk_gap_to_planning_floor_bytes")
        == context["target_uk_source_floor_bytes"]
        - context["current_terminal_uk_source_bytes_before_successor_global_dedup"],
        "UK capacity-gap arithmetic drift",
    )
    global_cap = int(context["research_corpus_v1_planning_floor_bytes"] * context["max_family_share_total"])
    stratum_cap = int(context["target_uk_source_floor_bytes"] * context["max_family_share_within_stratum"])
    expected_cap = min(global_cap, stratum_cap)
    _require(expected_cap == 5_000_000, "family-cap arithmetic no longer equals 5M")
    _require(context.get("effective_max_candidate_credit_bytes_if_one_family") == expected_cap, "candidate family cap drift")

    decision = config["qualification_decision"]
    _require(decision.get("status") == "RETEST", "candidate must remain RETEST")
    _require(decision.get("training_capacity_credit_bytes") == 0, "nonterminal candidate received byte credit")
    _require(decision.get("independent_family_credit") == 0, "nonterminal candidate received family credit")
    for key in (
        "corpus_admission",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "final_test_payload_accessed",
    ):
        _require(decision.get(key) is False, f"qualification boundary weakened: {key}")

    contract = config["bounded_successor_contract"]
    _require(contract.get("max_normalized_bytes") == expected_cap, "successor exceeds current one-family cap")
    _require(contract.get("single_family_until_lineage_proven") is True, "family lineage shortcut enabled")
    required = set(contract.get("required_before_nonzero_credit", ()))
    for gate in (
        "pin_exact_row_inventory_and_source_parquet_object_identity",
        "bind_each_selected_row_to_original_internet_archive_identifier",
        "establish_per_record_public_domain_or_compatible_training_rights",
        "materialize_exact_text_bytes_and_normalized_sha256",
        "ukrainian_language_and_script_validation",
        "quality_privacy_pii_screen",
        "exact_near_fragment_and_lineage_dedup_against_current_registry",
        "reserved_evaluation_decontamination_before_final_corpus_admission",
    ):
        _require(gate in required, f"required successor gate removed: {gate}")
    prohibited = set(contract.get("prohibited_shortcuts", ()))
    _require("credit_published_2_91_gb_as_training_capacity" in prohibited, "2.91GB shortcut no longer prohibited")
    _require("treat_dataset_card_license_prose_as_per_record_rights_proof" in prohibited, "blanket rights shortcut no longer prohibited")

    truth = config["truth_boundary"]
    _require(truth.get("payload_bytes_downloaded_by_this_authority") == 0, "metadata-only authority downloaded payload")
    _require(truth.get("optimizer_updates") == 0, "optimizer update claimed")
    for key in (
        "model_training_executed",
        "learned_20m_claimed",
        "research_corpus_v1_frozen",
        "source_bytes_equal_loss_positions_claimed",
    ):
        _require(truth.get(key) is False, f"truth boundary weakened: {key}")

    if live_card:
        _validate_live_card(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data/next100_110_ua_cultural_heritage_source_audit_v1.json",
    )
    parser.add_argument("--live-card", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise AuditError("config root must be an object")
    validate(config, live_card=args.live_card)
    print("NEXT100-110 source audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
