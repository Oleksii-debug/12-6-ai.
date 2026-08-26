from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.cross_source_capacity_audit_v4 import (
    CrossSourceV4Error,
    NIST_POLICY,
    _nist_payload,
    _normalize_nist_extracted,
    compose_v3_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"
EXT_PATH = ROOT / "configs/data/next100_065b_cross_source_dedup_v4.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_compose_exact_21_object_converged_vector() -> None:
    base = _load(BASE_PATH)
    extension = _load(EXT_PATH)
    merged = compose_v3_inventory(base, extension)

    rows = merged["sources"]
    assert len(rows) == 21
    assert len({row["source_id"] for row in rows}) == 21
    by_modality = {"uk": 0, "en": 0, "code": 0}
    for row in rows:
        by_modality[row["modality"]] += row["declared_capacity_bytes"]
    assert by_modality == {"uk": 100856, "en": 144151, "code": 69133}
    assert sum(by_modality.values()) == 314140

    nist = [row for row in rows if row["source_id"].startswith("en.nist.")]
    assert len(nist) == 3
    for row in nist:
        assert "comparison_normalization" not in row
        assert row["expected_raw_bytes"] == row["declared_capacity_bytes"]
        assert row["acquisition_url"].startswith("materialized-v4://")


def test_capacity_binding_mutation_fails_closed() -> None:
    base = _load(BASE_PATH)
    extension = _load(EXT_PATH)
    mutated = copy.deepcopy(extension)
    mutated["expected_pre_dedup"]["capacity_bytes"]["en"] += 1

    with pytest.raises(CrossSourceV4Error, match="capacity drift"):
        compose_v3_inventory(base, mutated)


def test_nonterminal_late_source_cannot_receive_credit() -> None:
    base = _load(BASE_PATH)
    extension = _load(EXT_PATH)
    mutated = copy.deepcopy(extension)
    mutated["additions"][0]["evidence_status"] = "RETEST"

    with pytest.raises(CrossSourceV4Error, match="terminal dedicated"):
        compose_v3_inventory(base, mutated)


def test_nist_capacity_must_equal_frozen_normalized_bytes() -> None:
    base = _load(BASE_PATH)
    extension = _load(EXT_PATH)
    mutated = copy.deepcopy(extension)
    row = next(
        item
        for item in mutated["additions"]
        if item.get("comparison_normalization") == NIST_POLICY
    )
    row["declared_capacity_bytes"] += 1
    mutated["expected_pre_dedup"]["capacity_bytes"]["en"] += 1
    mutated["expected_pre_dedup"]["capacity_bytes"]["total"] += 1

    with pytest.raises(CrossSourceV4Error, match="capacity must equal bounded normalized bytes"):
        compose_v3_inventory(base, mutated)


def test_nist_normalization_matches_next100_034_rules() -> None:
    raw = "ＡＢＣ  \r\nContact: PERSON@EXAMPLE.COM\r\n\r\n\r\nTail  \f\n"
    normalized = _normalize_nist_extracted(raw)
    assert normalized == b"ABC\nContact: <EMAIL>\n\nTail\n"


def test_nist_raw_drift_fails_before_pdftotext() -> None:
    extension = _load(EXT_PATH)
    row = next(
        item
        for item in extension["additions"]
        if item.get("comparison_normalization") == NIST_POLICY
    )
    with pytest.raises(CrossSourceV4Error, match="PDF byte identity drift"):
        _nist_payload(row, b"%PDF-deliberately-corrupt")
