from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.successor_cross_source_dedup import (
    NIST_POLICY,
    SuccessorDedupError,
    _normalize_nist_extracted,
    _post_dedup_handoff,
    _verify_materialized_nist,
    build_successor_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"
CONFIG = ROOT / "configs/data/next100_071_successor_cross_source_dedup_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_successor_inventory_matches_convergence_pre_dedup_vector() -> None:
    inventory = build_successor_inventory(_load(BASE), _load(CONFIG))
    rows = inventory["sources"]
    assert len(rows) == 21
    assert sum(row["declared_capacity_bytes"] for row in rows) == 314140
    by_modality = {
        modality: sum(row["declared_capacity_bytes"] for row in rows if row["modality"] == modality)
        for modality in ("uk", "en", "code")
    }
    assert by_modality == {"uk": 100856, "en": 144151, "code": 69133}
    families = {
        modality: {row["source_family"] for row in rows if row["modality"] == modality}
        for modality in ("uk", "en", "code")
    }
    assert {key: len(value) for key, value in families.items()} == {"uk": 4, "en": 2, "code": 4}
    assert "python.cpython.documentation" not in {row["source_family"] for row in rows}


def test_successor_inventory_fails_closed_on_duplicate_late_source() -> None:
    base = _load(BASE)
    config = _load(CONFIG)
    config["late_numeric_sources"][0]["source_id"] = base["sources"][0]["source_id"]
    with pytest.raises(SuccessorDedupError, match="source_id collision"):
        build_successor_inventory(base, config)


def test_cpython_zero_credit_exclusion_is_hard_bound() -> None:
    config = _load(CONFIG)
    config["zero_credit_exclusions"][0]["numeric_capacity_bytes"] = 17901
    with pytest.raises(SuccessorDedupError, match="premature numeric capacity"):
        build_successor_inventory(_load(BASE), config)


def test_nist_normalization_matches_terminal_policy_primitives() -> None:
    text = "Title\r\n\r\n\r\nContact Test.User@example.org  \r\nLigature: ﬁ\fTail   \r\n"
    normalized = _normalize_nist_extracted(text)
    assert normalized == b"Title\n\nContact <EMAIL>\nLigature: fi\nTail\n"


def test_nist_normalization_truncates_at_safe_boundary() -> None:
    text = ("a" * 13000) + "\n\n" + ("b" * 9000) + "\n"
    normalized = _normalize_nist_extracted(text)
    assert 12000 <= len(normalized) <= 20000
    assert normalized.endswith(b"\n")
    assert b"b" not in normalized


def test_nist_materialization_identity_is_fail_closed() -> None:
    raw = b"%PDF-example"
    normalized = b"bounded text\n"
    row = {
        "source_id": "nist-test",
        "materialization_policy": NIST_POLICY,
        "upstream_raw_bytes": len(raw),
        "upstream_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_raw_bytes": len(normalized),
        "expected_raw_sha256": hashlib.sha256(normalized).hexdigest(),
    }
    _verify_materialized_nist(row, raw, normalized)
    with pytest.raises(SuccessorDedupError, match="normalized SHA-256 drift"):
        _verify_materialized_nist(row, raw, normalized + b"drift")


def test_post_dedup_handoff_only_advances_to_balance_retest() -> None:
    report = {
        "terminal_candidates": {
            "conservative_unique_capacity_bytes_after": 300000,
            "by_modality": {
                "uk": {
                    "conservative_unique_capacity_bytes_after": 95000,
                    "effective_independent_origin_count": 4,
                },
                "en": {
                    "conservative_unique_capacity_bytes_after": 136000,
                    "effective_independent_origin_count": 2,
                },
                "code": {
                    "conservative_unique_capacity_bytes_after": 69000,
                    "effective_independent_origin_count": 4,
                },
            },
        }
    }
    handoff = _post_dedup_handoff(report, 2)
    assert handoff["family_minimum_status"] == "PASS_POST_DEDUP_MINIMUM"
    assert handoff["next_gate"] == "BALANCE_DIVERSITY_RETEST"
    assert handoff["corpus_materialization_authorized"] is False
    assert handoff["tokenizer_fit_authorized"] is False
    assert handoff["learned_20m_campaign_authorized"] is False
