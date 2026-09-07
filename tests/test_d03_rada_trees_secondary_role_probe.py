from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "probe_d03_rada_trees_secondary_role.py"
SPEC = importlib.util.spec_from_file_location("rada_secondary_probe", TOOL_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def config() -> dict[str, object]:
    value = json.loads(
        (ROOT / "configs/data/d03_rada_trees_secondary_role_probe_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_committed_secondary_config_is_zero_credit() -> None:
    value = tool.load_config()
    boundary = value["claim_boundary"]
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["unique_causal_loss_positions_authorized"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False


def test_listing_bound_allows_large_archive_without_full_extract() -> None:
    policy = config()["inventory_policy"]
    listing = [
        {"path": "texts/a.txt", "size_bytes": 6_000_000_000},
        {"path": "texts/b.txt", "size_bytes": 6_000_000_000},
    ]
    # Per-member bound remains strict, so use many bounded members instead.
    listing = [
        {"path": f"texts/{index:04d}.txt", "size_bytes": 25_000_000}
        for index in range(480)
    ]
    total = tool._listing_bounds(listing, policy)
    assert total == 12_000_000_000
    assert total > policy["max_full_extract_total_uncompressed_bytes"]


def test_listing_bound_rejects_single_oversize_member() -> None:
    policy = config()["inventory_policy"]
    listing = [{"path": "texts/huge.txt", "size_bytes": 50_000_001}]
    with pytest.raises(tool.SecondaryProbeError, match="safe stream bound"):
        tool._listing_bounds(listing, policy)


def test_stream_sample_is_deterministic_and_prefers_text() -> None:
    listing = [
        {"path": f"conllu/{index:04d}.conllu", "size_bytes": 10}
        for index in range(100)
    ] + [
        {"path": f"texts/{index:04d}.txt", "size_bytes": 10}
        for index in range(100)
    ]
    first = tool._sample_listing(listing, 8)
    second = tool._sample_listing(listing, 8)
    assert first == second
    assert len(first) == 8
    assert all(item["path"].endswith(".txt") for item in first)
    assert first[0]["path"] == "texts/0000.txt"
    assert first[-1]["path"] == "texts/0099.txt"


def test_known_suffix_holds_do_not_invent_plain_text() -> None:
    listing = [
        {"path": "ud/a.conllu", "size_bytes": 100},
        {"path": "annotations/b.xml", "size_bytes": 200},
        {"path": "annotations/c.jsonl", "size_bytes": 300},
        {"path": "texts/d.txt", "size_bytes": 400},
    ]
    counts, sizes = tool._known_suffix_holds(listing)
    assert counts == {
        "DERIVED_UD_HOLD": 1,
        "DERIVED_ANNOTATION_HOLD": 2,
    }
    assert sizes == {
        "DERIVED_UD_HOLD": 100,
        "DERIVED_ANNOTATION_HOLD": 500,
    }


def test_no_sample_needed_for_annotation_only_listing() -> None:
    listing = [
        {"path": "ud/a.conllu", "size_bytes": 100},
        {"path": "ud/b.conllu", "size_bytes": 120},
    ]
    assert tool._sample_listing(listing, 32) == []
