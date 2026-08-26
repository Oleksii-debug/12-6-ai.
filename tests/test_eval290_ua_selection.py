from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "src/twelve_six/eval290_ua_selection.py"
spec = importlib.util.spec_from_file_location("eval290_ua_selection", MODULE_PATH)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _selector() -> dict[str, object]:
    return {
        "seed": "seed",
        "min_alpha_ratio": 0.45,
        "min_cyrillic_alpha_ratio": 0.70,
    }


def test_candidate_identity_and_rank_are_deterministic() -> None:
    source = {"source_id": "ua.test"}
    text = "Український текст для перевірки детермінованого відбору. " * 12
    first = m._candidate(source=source, locator="row:1", text=text, selector=_selector())
    second = m._candidate(source=source, locator="row:1", text=text, selector=_selector())
    assert first == second
    assert first is not None
    assert first["content_sha256"] == m.sha256_bytes(text.strip().encode("utf-8"))


def test_quality_rejects_email() -> None:
    text = ("Український текст " * 40) + " test@example.org"
    assert m._quality_ok(text, _selector()) is False


def test_diversity_gate_accepts_two_balanced_independent_families() -> None:
    rows = [
        {"source_family": "family-a", "utf8_bytes": 1000},
        {"source_family": "family-a", "utf8_bytes": 1000},
        {"source_family": "family-a", "utf8_bytes": 1000},
        {"source_family": "family-b", "utf8_bytes": 950},
        {"source_family": "family-b", "utf8_bytes": 950},
        {"source_family": "family-b", "utf8_bytes": 950},
    ]
    gate = {
        "minimum_independent_source_families": 2,
        "minimum_records_per_family": 3,
        "maximum_family_byte_share": 0.70,
        "minimum_effective_family_count": 1.80,
    }
    result = m._validate_diversity(rows, gate)
    assert result["independent_source_families"] == 2
    assert result["top_family_byte_share"] < 0.70
    assert result["effective_family_count"] >= 1.80


def test_diversity_gate_rejects_single_family() -> None:
    rows = [{"source_family": "family-a", "utf8_bytes": 1000} for _ in range(4)]
    gate = {
        "minimum_independent_source_families": 2,
        "minimum_records_per_family": 3,
        "maximum_family_byte_share": 0.70,
        "minimum_effective_family_count": 1.80,
    }
    with pytest.raises(m.Eval290Error, match="source-family diversity"):
        m._validate_diversity(rows, gate)


def test_split_is_bounded_and_deterministic() -> None:
    text = "\n".join(("Українське речення з достатньою кількістю літер. " * 30) for _ in range(4))
    a = m._split_text(text, target_chars=900, min_chars=400, max_chars=1200)
    b = m._split_text(text, target_chars=900, min_chars=400, max_chars=1200)
    assert a == b
    assert a
    assert all(400 <= len(chunk) <= 1200 for chunk in a)


def test_canonical_hash_is_mapping_order_independent() -> None:
    assert m.hash_json({"a": 1, "b": 2}) == m.hash_json({"b": 2, "a": 1})
