from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "scan_d03_rada_trees_full_txt.py"
SPEC = importlib.util.spec_from_file_location("rada_full_txt_scan", TOOL_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def _synthetic_config(listing: list[dict[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {
        "source": {
            "dataset": "fixture",
            "dataset_revision": "a" * 40,
            "archive_path": "fixture.7z",
            "expected_size_bytes": 123,
            "content_sha256": "b" * 64,
            "xet_hash": "c" * 64,
        },
        "terminal_listing": {
            "listing_identity_sha256": "",
            "member_count": len(listing),
            "uncompressed_bytes": sum(int(item["size_bytes"]) for item in listing),
            "plain_text_suffix": ".txt",
            "plain_text_suffix_member_count": sum(
                str(item["path"]).lower().endswith(".txt") for item in listing
            ),
        },
        "scan_policy": {
            "max_single_member_bytes": 1000,
            "max_selected_total_uncompressed_bytes": 10000,
        },
    }
    terminal = value["terminal_listing"]
    assert isinstance(terminal, dict)
    terminal["listing_identity_sha256"] = tool._listing_identity(value, listing)
    return value


def _classified(
    path: str,
    size: int,
    sha256: str,
    classification: str = "PLAIN_TEXT_CANDIDATE",
) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": size,
        "sha256": sha256,
        "classification": classification,
        "decoded_encoding": "utf-8",
        "text_metrics": {"letters": 10, "characters": 12},
        "path_year_hints": [2024],
        "text_emitted": False,
    }


def test_committed_scan_config_pins_parent_and_zero_credit() -> None:
    value = tool.load_config()
    assert value["parent_scientific_authority"]["workflow_run"] == 34158830442
    assert value["source"]["content_sha256"] == tool.SOURCE_SHA256
    assert value["terminal_listing"]["listing_identity_sha256"] == tool.LISTING_SHA256
    assert value["terminal_listing"]["plain_text_suffix_member_count"] == 4391
    boundary = value["claim_boundary"]
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["training_admission_claimed"] is False


def test_validate_listing_binds_exact_listing_and_selects_only_txt() -> None:
    listing = [
        {"path": "a/001.txt", "size_bytes": 100},
        {"path": "a/001.xtag", "size_bytes": 200},
        {"path": "b/002.TXT", "size_bytes": 300},
    ]
    value = _synthetic_config(listing)
    selected = tool.validate_listing(value, listing)
    assert selected == [listing[0], listing[2]]


def test_validate_listing_rejects_same_count_with_changed_member_identity() -> None:
    listing = [
        {"path": "a/001.txt", "size_bytes": 100},
        {"path": "a/001.xtag", "size_bytes": 200},
    ]
    value = _synthetic_config(listing)
    tampered = [dict(item) for item in listing]
    tampered[1]["path"] = "a/evil.xtag"
    with pytest.raises(tool.FullTxtScanError, match="listing identity drift"):
        tool.validate_listing(value, tampered)


def test_validate_listing_rejects_selected_byte_envelope_overflow() -> None:
    listing = [
        {"path": "a/001.txt", "size_bytes": 600},
        {"path": "b/002.txt", "size_bytes": 600},
    ]
    value = _synthetic_config(listing)
    policy = value["scan_policy"]
    assert isinstance(policy, dict)
    policy["max_selected_total_uncompressed_bytes"] = 1000
    with pytest.raises(ValueError, match="max_total_uncompressed_bytes"):
        tool.validate_listing(value, listing)


def test_exact_duplicate_collapse_is_deterministic_and_conservative() -> None:
    classified = [
        _classified("z/duplicate.txt", 100, "1" * 64),
        _classified("a/survivor.txt", 100, "1" * 64),
        _classified("m/unique.txt", 250, "2" * 64),
        _classified("m/hold.txt", 400, "3" * 64, "UNKNOWN_FORMAT_HOLD"),
    ]
    summary = tool.summarize_classification(classified)
    assert summary["plain_text_candidate_members_before_exact_duplicate_collapse"] == 3
    assert summary["plain_text_candidate_bytes_before_exact_duplicate_collapse"] == 450
    assert summary["plain_text_candidate_members_after_exact_duplicate_collapse"] == 2
    assert summary["plain_text_candidate_bytes_after_exact_duplicate_collapse"] == 350
    assert summary["exact_duplicate_discount_bytes"] == 100
    assert summary["exact_duplicate_group_count"] == 1
    group = summary["exact_duplicate_groups"][0]
    assert group["survivor_path"] == "a/survivor.txt"
    assert group["duplicate_paths"] == ["z/duplicate.txt"]


def test_summary_never_persists_raw_text() -> None:
    summary = tool.summarize_classification(
        [_classified("a/plain.txt", 100, "4" * 64)]
    )
    assert summary["raw_member_text_emitted"] is False
    serialized = repr(summary)
    assert "text_emitted" not in serialized
    assert "raw_text" not in serialized


def test_member_list_rejects_non_txt_and_newline_paths(tmp_path: Path) -> None:
    output = tmp_path / "members.list"
    with pytest.raises(tool.FullTxtScanError, match="non-.txt"):
        tool._write_member_list(output, [{"path": "a/data.xml", "size_bytes": 1}])
    with pytest.raises(ValueError, match="unsafe member path"):
        tool._write_member_list(output, [{"path": "a/bad\nname.txt", "size_bytes": 1}])
