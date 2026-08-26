from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data184_diversity import (
    diversity,
    near_dedup,
    universal_bootstrap,
)
from twelve_six.scaling_500k_evidence import _target_spec


def _row(record_id: str, family: str, text: str, *, language: str = "en", modality: str = "natural"):
    return {
        "id": record_id,
        "family": family,
        "text": text,
        "language": language,
        "modality": modality,
        "utf8_bytes": len(text.encode("utf-8")),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_500k_control_geometry_is_exact_incumbent():
    assert _target_spec().parameter_count() == 467_808


def test_diversity_reports_entropy_effective_count_and_mass():
    rows = [
        _row("a", "f1", "a" * 100),
        _row("b", "f2", "b" * 100, language="uk"),
    ]
    report = diversity(rows)
    assert report["families"] == 2
    assert report["token_mass_byte_tokens"] == 200
    assert report["top_family_share"] == pytest.approx(0.5)
    assert report["effective_source_count"] == pytest.approx(2.0)
    assert report["language_mass_bytes"] == {"en": 100, "uk": 100}


def test_cross_family_exact_and_near_duplicates_are_removed():
    base = [_row("a", "family-a", "one two three four five six seven")]
    exact = _row("b", "family-b", "one two three four five six seven")
    near = _row("c", "family-c", "one two three four five six seven eight")
    kept, removed = near_dedup(base, [exact, near], 0.70)
    assert kept == []
    assert {item["reason"] for item in removed} == {"exact_duplicate", "cross_family_near_duplicate"}


def test_same_family_variants_are_not_miscounted_as_cross_family_dedup():
    a = _row("a", "same-family", "one two three four five six seven")
    b = _row("b", "same-family", "one two three four five six seven eight")
    kept, removed = near_dedup([a], [b], 0.70)
    assert [row["id"] for row in kept] == ["b"]
    assert removed == []


def test_universal_bootstrap_requires_common_identity_and_reports_direction():
    control = {
        "per_chunk": {
            "x": {"bpb": 5.0},
            "y": {"bpb": 4.0},
            "z": {"bpb": 3.0},
        }
    }
    expanded = {
        "per_chunk": {
            "x": {"bpb": 4.5},
            "y": {"bpb": 3.5},
            "z": {"bpb": 2.5},
        }
    }
    result = universal_bootstrap(control, expanded, 100, 184)
    assert result["observed_mean_delta_bpb"] == pytest.approx(-0.5)
    assert result["probability_expanded_better"] == 1.0
    assert result["negative_delta_is_better"] is True


def test_data184_config_has_unique_real_source_families_and_exact_rights_evidence():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/data/data184_real_source_diversity_v1.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == "12-6.data184-real-source-diversity.v1"
    families = [source["family"] for source in cfg["sources"]]
    source_ids = [source["source_id"] for source in cfg["sources"]]
    assert len(families) == len(set(families))
    assert len(source_ids) == len(set(source_ids))
    assert {source["language"] for source in cfg["sources"]} >= {"uk", "en", "code"}
    assert {source["role"] for source in cfg["sources"]} == {"incumbent", "new"}
    for source in cfg["sources"]:
        evidence = root / source["rights_evidence_path"]
        assert evidence.is_file()
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == source["rights_evidence_sha256"]
        assert source["source_version"]
        assert source["license_id"]


def test_parent_project_code_is_bound_to_exact_incumbent_head():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/data/data184_parent_project_code_v1.json").read_text(encoding="utf-8"))
    assert cfg["schema_version"] == "12-6.data184-parent-project-code.v1"
    assert cfg["parent_git_sha"] == "117e0f156fff6b0226f00748ff25938f1b4a2612"
    evidence = root / cfg["rights_evidence_path"]
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == cfg["rights_evidence_sha256"]
