from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("rada_provenance", TOOLS / "audit_d03_rada_trees_secondary_provenance.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(path: str, digest: str, size: int, year: int) -> dict:
    return {
        "classification": "PLAIN_TEXT_CANDIDATE",
        "path": path,
        "path_year_hints": [year],
        "sha256": digest,
        "size_bytes": size,
    }


def test_classify_path_accepts_valid_dated_session_shape() -> None:
    disposition, year, reason = mod.classify_path(row("texts/2023-12-15__ZASIDANNJa_1.txt", "a" * 64, 123, 2023))
    assert disposition == "RADA_PLENARY_SESSION_PATH_CANDIDATE"
    assert year == 2023
    assert reason == "dated_session_shape_and_year_hint_agree"


def test_classify_path_quarantines_non_session_path() -> None:
    disposition, year, reason = mod.classify_path(
        row("texts/stenogramy-zasidannya-rnbo-vid-28-lyutogo-2014-roku.txt", "b" * 64, 84_781, 2014)
    )
    assert disposition == "QUARANTINE_NON_PARLIAMENT_PATH"
    assert year is None
    assert reason == "path_not_dated_plenary_session_shape"


def test_classify_path_rejects_invalid_date_and_year_hint_drift() -> None:
    invalid = row("texts/2023-02-31__ZASIDANNJa.txt", "c" * 64, 1, 2023)
    try:
        mod.classify_path(invalid)
    except mod.ProvenanceAuditError:
        pass
    else:
        raise AssertionError("invalid calendar date must fail closed")

    mismatch = row("texts/2023-12-15__ZASIDANNJa.txt", "d" * 64, 1, 2022)
    try:
        mod.classify_path(mismatch)
    except mod.ProvenanceAuditError:
        pass
    else:
        raise AssertionError("path year hint mismatch must fail closed")


def test_exact_hash_survivor_is_lexicographic_path() -> None:
    original_count = mod.EXPECTED_UNIQUE_COUNT
    original_bytes = mod.EXPECTED_UNIQUE_BYTES
    mod.EXPECTED_UNIQUE_COUNT = 2
    mod.EXPECTED_UNIQUE_BYTES = 30
    try:
        items = [
            row("texts/2020-01-02__B.txt", "e" * 64, 10, 2020),
            row("texts/2020-01-01__A.txt", "e" * 64, 10, 2020),
            row("texts/2021-01-01__C.txt", "f" * 64, 20, 2021),
        ]
        survivors = mod.exact_hash_survivors(copy.deepcopy(items))
        assert [item["path"] for item in survivors] == [
            "texts/2020-01-01__A.txt",
            "texts/2021-01-01__C.txt",
        ]
    finally:
        mod.EXPECTED_UNIQUE_COUNT = original_count
        mod.EXPECTED_UNIQUE_BYTES = original_bytes
