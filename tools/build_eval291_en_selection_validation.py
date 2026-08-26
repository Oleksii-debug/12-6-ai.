#!/usr/bin/env python3
"""Build/verify the immutable EVAL-291 English selection-validation authority.

This tool intentionally never reads final-test content. It consumes only the
committed EVAL-291 metadata/configuration and emits a deterministic fail-closed
authority until an English external-real object has both explicit evaluation
authorization and pre-training reservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation/eval291_en_selection_validation_v1.json"
MANIFEST = ROOT / "data/evaluation/selection-validation/en/v1/manifest.json"

EXPECTED_WORKER = "EVAL-291-EN-SELECTION-VALIDATION-V1"
EXPECTED_SOURCE_HEAD = "90bc0b7f8b696ec35202532b13edf6ab29a662fe"
EXPECTED_SOURCE_REGISTRY = "1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486"
EXPECTED_FINAL_TEST_ID = "86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116"
PROHIBITED_FINAL_TEST_PATH_PARTS = (
    "final-test",
    "recover174_real_holdout_seed.jsonl.gz",
)
EXPECTED_SOURCE_IDS = {
    "en.standardebooks.manual.8-typography",
    "en.standardebooks.manual.9-metadata",
}


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def validate_config(cfg: dict[str, Any]) -> None:
    assert cfg["worker_id"] == EXPECTED_WORKER
    assert cfg["language"] == "en"
    assert cfg["classification"] == "external_real_selection_validation"
    assert cfg["local_free_only"] is True

    source = cfg["source_authority"]
    assert source["head_sha"] == EXPECTED_SOURCE_HEAD
    assert source["terminal_status"] == "success"
    assert source["registry_identity_sha256"] == EXPECTED_SOURCE_REGISTRY

    seen = {s["source_id"] for s in cfg["terminal_english_sources"]}
    assert seen == EXPECTED_SOURCE_IDS
    for item in cfg["terminal_english_sources"]:
        assert item["registry_source_id"].startswith("external-real:")
        assert item["source_family_id"] == "en.standardebooks.manual"
        assert len(item["raw_sha256"]) == 64
        assert len(item["normalized_sha256"]) == 64
        assert len(item["rights_evidence_set_identity_sha256"]) == 64
        assert item["training_rights_status"] == "ALLOWED"
        # EVAL-291 must never infer evaluation permission from training permission.
        if item["evaluation_rights_status"] != "ALLOWED":
            assert item["selection_validation_admitted"] is False
        if not item["reserved_from_training"]:
            assert item["selection_validation_admitted"] is False

    guard = cfg["final_test_guard"]
    assert guard["data232_final_test_identity_sha256"] == EXPECTED_FINAL_TEST_ID
    assert guard["outcomes_inspected"] is False
    assert guard["content_bytes_inspected_by_eval291"] is False
    assert guard["metadata_only_binding"] is True
    assert guard["final_test_records_may_be_selected"] is False
    assert guard["final_test_bytes_may_be_copied"] is False

    # The config may bind final-test identities, never a readable final-test corpus path.
    encoded = canonical_bytes(cfg).decode("utf-8")
    for prohibited in PROHIBITED_FINAL_TEST_PATH_PARTS:
        if prohibited in encoded:
            # One string is permitted only as an explicit prohibited root.
            allowed_occurrences = sum(
                prohibited in p
                for p in cfg["purpose_separation"]["final_test_roots_prohibited"]
            )
            assert encoded.count(prohibited) == allowed_occurrences


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    admitted = [
        item
        for item in cfg["terminal_english_sources"]
        if item["selection_validation_admitted"]
    ]
    observed_families = sorted(
        {item["source_family_id"] for item in cfg["terminal_english_sources"]}
    )
    eligible_families = sorted({item["source_family_id"] for item in admitted})

    blockers: list[str] = []
    for item in cfg["terminal_english_sources"]:
        if item["evaluation_rights_status"] != "ALLOWED":
            blockers.append(
                f"{item['source_id']}:EVALUATION_RIGHTS_{item['evaluation_rights_status']}"
            )
        if not item["reserved_from_training"]:
            blockers.append(f"{item['source_id']}:NOT_RESERVED_FROM_TRAINING")

    usable = bool(admitted) and not blockers
    authority: dict[str, Any] = {
        "schema_version": "12-6.eval291-en-selection-validation-authority.v1",
        "worker_id": cfg["worker_id"],
        "status": (
            "READY"
            if usable
            else "BLOCKED_NO_TERMINAL_EN_EVALUATION_RESERVATION"
        ),
        "language": "en",
        "classification": "external_real_selection_validation",
        "local_free_only": True,
        "source_authority": cfg["source_authority"],
        "observed_terminal_source_objects": len(cfg["terminal_english_sources"]),
        "observed_terminal_source_families": observed_families,
        "eligible_source_objects": len(admitted),
        "eligible_source_families": eligible_families,
        "documents": 0,
        "records": [],
        "record_content_sha256": [],
        "blockers": sorted(blockers),
        "final_test_guard": cfg["final_test_guard"],
        "purpose_separation": cfg["purpose_separation"],
        "deterministic_rebuild": {
            "canonical_json": (
                "UTF-8; sort_keys=true; separators=(',',':'); trailing_newline=true"
            ),
            "network_required": False,
            "source_bytes_materialized": False,
        },
        "truth_boundary": {
            "selection_validation_is_usable": usable,
            "final_test_outcomes_inspected": False,
            "final_test_bytes_copied": False,
            "training_bytes_reclassified": False,
            "rights_inferred_from_training_permission": False,
        },
    }
    authority["authority_identity_sha256"] = hashlib.sha256(
        canonical_bytes(authority)
    ).hexdigest()
    return authority


def pretty_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def verify() -> None:
    cfg = load_config()
    first = pretty_bytes(build(cfg))
    second = pretty_bytes(build(cfg))
    assert first == second, "nondeterministic rebuild"
    committed = MANIFEST.read_bytes()
    assert committed == first, "committed manifest is not deterministic rebuild"
    rebuilt = json.loads(first)
    assert rebuilt["documents"] == 0
    assert rebuilt["records"] == []
    assert rebuilt["eligible_source_objects"] == 0
    assert rebuilt["status"] == "BLOCKED_NO_TERMINAL_EN_EVALUATION_RESERVATION"
    assert rebuilt["truth_boundary"]["final_test_outcomes_inspected"] is False
    assert rebuilt["truth_boundary"]["final_test_bytes_copied"] is False
    assert rebuilt["truth_boundary"]["training_bytes_reclassified"] is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config()
    authority = build(cfg)
    payload = pretty_bytes(authority)

    if args.verify:
        verify()
        print(
            "EVAL291 PASS: deterministic fail-closed authority "
            f"{authority['authority_identity_sha256']}"
        )
        return 0

    output = args.output or MANIFEST
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(authority["authority_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
