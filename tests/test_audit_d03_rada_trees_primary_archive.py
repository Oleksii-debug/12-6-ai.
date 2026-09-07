from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_d03_rada_trees_primary_archive as audit


def slt(*blocks: str) -> str:
    return "\n\n".join(blocks) + "\n"


def test_parse_listing_is_deterministic_and_skips_archive_header() -> None:
    listing = slt(
        "Path = Rada_Trees.7z\nType = 7z",
        "Path = z/2024.txt\nSize = 5\nFolder = -",
        "Path = a/1990.txt\nSize = 3\nFolder = -",
        "Path = empty-dir\nSize = 0\nFolder = +",
    )
    assert audit.parse_7z_slt(listing) == [
        {"path": "a/1990.txt", "size_bytes": 3},
        {"path": "z/2024.txt", "size_bytes": 5},
    ]


@pytest.mark.parametrize(
    "raw",
    ["../escape.txt", "/absolute.txt", "C:\\drive.txt", "a/../../escape.txt", ""],
)
def test_unsafe_member_paths_fail_closed(raw: str) -> None:
    with pytest.raises(audit.AuditError):
        audit.normalize_member_path(raw)


def test_backslashes_are_normalized_for_duplicate_detection() -> None:
    listing = slt(
        "Path = a\\b.txt\nSize = 1\nFolder = -",
        "Path = a/b.txt\nSize = 1\nFolder = -",
    )
    with pytest.raises(audit.AuditError, match="duplicate normalized member path"):
        audit.parse_7z_slt(listing)


def test_member_size_limit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "MAX_MEMBER_BYTES", 10)
    listing = slt("Path = too-big.txt\nSize = 11\nFolder = -")
    with pytest.raises(audit.AuditError, match="member size outside policy"):
        audit.parse_7z_slt(listing)


def test_total_uncompressed_limit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "MAX_MEMBER_BYTES", 10)
    monkeypatch.setattr(audit, "MAX_TOTAL_UNCOMPRESSED_BYTES", 10)
    listing = slt(
        "Path = one.txt\nSize = 6\nFolder = -",
        "Path = two.txt\nSize = 5\nFolder = -",
    )
    with pytest.raises(audit.AuditError, match="total uncompressed bytes exceed policy"):
        audit.parse_7z_slt(listing)


def test_empty_regular_inventory_fails_closed() -> None:
    listing = slt("Path = dir\nSize = 0\nFolder = +")
    with pytest.raises(audit.AuditError, match="no regular members"):
        audit.parse_7z_slt(listing)
