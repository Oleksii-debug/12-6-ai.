from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVIDENCE = Path("evidence/d03-rada-trees/secondary-archive-terminal-probe-v1.json")
EXPECTED_SOURCE_SHA256 = "737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e"
EXPECTED_XET = "46c56a953551f3d33800c8b50fbd355104197abec45c366f245103e830b17d30"
EXPECTED_LISTING = "9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a"
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:6e2818943b0968fc690e8e6faa3b9752afa66b1db5ce1a76239af7f5a8856bb8"
)


def _load() -> dict:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _core_identity(value: dict) -> str:
    core = dict(value)
    core.pop("evidence_identity_sha256")
    payload = (
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_secondary_terminal_evidence_is_self_hashed_and_exactly_bound() -> None:
    value = _load()

    assert value["schema_version"] == "12-6.d03-rada-trees-secondary-terminal-probe.v1"
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["evidence_identity_sha256"] == _core_identity(value)

    source = value["source"]
    assert source["dataset"] == "uacorpus/Rada_Trees"
    assert source["dataset_revision"] == "1b994a5804dcda122721e8d33a03fd172cf8d867"
    assert source["archive_path"] == "rada_xtag_texts.7z"
    assert source["xet_hash"] == EXPECTED_XET
    assert source["content_sha256"] == EXPECTED_SOURCE_SHA256
    assert source["compressed_bytes"] == 697_768_591

    execution = value["execution"]
    assert execution["head_sha"] == "1b9140bd78374cfa010801c71d0784e1fd9a4f9d"
    assert execution["workflow_run"] == 34_119_998_742
    assert execution["workflow_job"] == 101_735_625_798
    assert execution["conclusion"] == "success"
    assert execution["artifact_id"] == 10_017_906_020
    assert execution["artifact_digest"] == EXPECTED_ARTIFACT_DIGEST


def test_secondary_listing_accounting_is_exact_without_full_scan_overclaim() -> None:
    value = _load()
    inventory = value["inventory"]

    assert inventory["member_count"] == 8_782
    assert inventory["uncompressed_bytes"] == 19_711_802_635
    assert inventory["full_extraction_performed"] is False
    assert inventory["listing_identity_sha256"] == EXPECTED_LISTING
    assert inventory["full_inventory_identity_sha256"] is None
    assert inventory["member_suffixes"] == {".txt": 4_391, ".xtag": 4_391}
    assert inventory["member_suffix_bytes"] == {
        ".txt": 879_031_855,
        ".xtag": 18_832_770_780,
    }
    assert sum(inventory["member_suffixes"].values()) == inventory["member_count"]
    assert sum(inventory["member_suffix_bytes"].values()) == inventory["uncompressed_bytes"]

    classification = value["bounded_classification"]
    assert classification["full_member_classification_complete"] is False
    assert classification["sampled_member_count"] == 32
    assert classification["sampled_class_counts"] == {
        "PLAIN_TEXT_CANDIDATE": 16,
        "UNKNOWN_FORMAT_HOLD": 16,
    }
    assert classification["sampled_txt_members"] == 16
    assert classification["sampled_txt_plain_text_candidates"] == 16
    assert classification["sampled_xtag_members"] == 16
    assert classification["sampled_xtag_unknown_format_holds"] == 16
    assert classification["plain_text_candidate_members_full_archive"] is None
    assert classification["plain_text_candidate_bytes_full_archive"] is None
    assert classification["sample_is_not_capacity_authority"] is True
    assert classification["raw_member_text_emitted"] is False


def test_secondary_evidence_retains_zero_training_authority() -> None:
    value = _load()
    decision = value["decision"]
    boundary = value["claim_boundary"]

    assert decision["full_txt_layer_training_admission_claimed"] is False
    assert decision["rerun_with_pinned_sha_required_before_any_capacity_credit"] is True
    assert decision["next_gate"].startswith("FULL_STREAM_SCAN_4391_TXT_MEMBERS")

    assert boundary == {
        "training_authorized_bytes": 0,
        "unique_causal_loss_positions_authorized": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }
