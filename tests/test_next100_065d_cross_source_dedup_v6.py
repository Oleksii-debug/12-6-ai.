from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v6 as v6


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_065d_cross_source_dedup_v6.json"


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v6_config_is_fail_closed_and_exact() -> None:
    config = _config()
    v6._validate_config(config)
    assert config["expected_vector"] == {
        "source_object_count": 31,
        "source_family_counts": {"uk": 4, "en": 5, "code": 5},
        "source_capacity_bytes_before_global_dedup": 2045180,
        "source_capacity_by_modality_before_global_dedup": {
            "uk": 100856,
            "en": 1838293,
            "code": 106031,
        },
        "independent_family_count": 14,
        "research_corpus_v1_acquisition_planning_target_bytes": 20000000,
        "planning_gap_before_global_dedup": 17954820,
    }


def test_v6_rejects_unit_or_capacity_rewrite() -> None:
    config = copy.deepcopy(_config())
    config["expected_vector"]["source_capacity_bytes_before_global_dedup"] = 2045181
    with pytest.raises(v6.CrossSourceV6Error, match="total capacity"):
        v6._validate_config(config)


def test_v6_rejects_training_or_evaluation_boundary_broadening() -> None:
    config = copy.deepcopy(_config())
    config["claim_boundary"]["training_authorized"] = True
    with pytest.raises(v6.CrossSourceV6Error, match="claim boundary"):
        v6._validate_config(config)

    config = copy.deepcopy(_config())
    config["gutenberg"]["evaluation"] = "ALLOWED"
    with pytest.raises(v6.CrossSourceV6Error, match="evaluation boundary"):
        v6._validate_config(config)


def test_gutenberg_normalizer_reproduces_preregistered_body_rule() -> None:
    raw = (
        b"header\r\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        b"\r\nAlpha\r\nBeta\r\n\r\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        b"footer\r\n"
    )
    assert v6._normalize_pg_body(raw, "ascii") == b"Alpha\nBeta\n"


def test_gutenberg_normalizer_fails_closed_on_marker_ambiguity() -> None:
    raw = (
        b"*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        b"one\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        b"two\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )
    with pytest.raises(v6.CrossSourceV6Error, match="START marker"):
        v6._normalize_pg_body(raw, "ascii")


def test_v6_expected_vector_matches_terminal_authority_arithmetic() -> None:
    config = _config()
    base = config["base_v5"]
    numpy_cfg = config["numpy"]
    gutenberg = config["gutenberg"]
    assert (
        base["expected_source_capacity_bytes"]
        + numpy_cfg["numeric_training_capacity_bytes"]
        + gutenberg["numeric_training_capacity_bytes"]
        == config["expected_vector"]["source_capacity_bytes_before_global_dedup"]
    )
    assert (
        config["expected_vector"]["source_capacity_bytes_before_global_dedup"]
        + config["expected_vector"]["planning_gap_before_global_dedup"]
        == config["expected_vector"]["research_corpus_v1_acquisition_planning_target_bytes"]
    )
