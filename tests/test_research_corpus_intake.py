from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.research_corpus_intake import (
    ResearchCorpusIntakeError,
    build_research_corpus_intake_report,
    validate_research_corpus_intake,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = (
    ROOT / "configs" / "data" / "research_corpus_v1_intake_convergence_v1.json"
)


def _authority() -> dict:
    return json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))


def test_terminal_intake_projects_family_diversity_without_terminal_pass() -> None:
    report = build_research_corpus_intake_report(_authority())

    assert report["known_terminal_intake_lower_bound_bytes"] == {
        "ua": 100856,
        "en": 162052,
        "code": 14977,
    }
    assert report["known_terminal_intake_total_bytes"] == 277885
    assert report["capacity_proxy_gap_bytes"] == {
        "ua": 8899144,
        "en": 6837948,
        "code": 3985023,
    }
    assert report["capacity_proxy_total_gap_bytes"] == 19722115

    projection = report[
        "projected_independent_families_if_global_dedup_does_not_collapse_lineage"
    ]
    assert projection["counts"] == {"ua": 4, "en": 3, "code": 3}
    assert projection["gate"] == {
        "ua": "PROJECTED_PASS_NOT_TERMINAL",
        "en": "PROJECTED_PASS_NOT_TERMINAL",
        "code": "PROJECTED_PASS_NOT_TERMINAL",
    }

    assert report["training_authorized"] is False
    assert report["authorized_unique_optimized_targets"] == 0
    assert report["hard_gates"]["final_corpus"] == "BLOCKED"
    assert report["hard_gates"]["real_20m_training"] == "BLOCKED"


def test_truth_boundary_rejects_early_training_authorization() -> None:
    authority = copy.deepcopy(_authority())
    authority["hard_truth_boundary"]["training_authorized"] = True

    with pytest.raises(
        ResearchCorpusIntakeError,
        match="training_authorized must remain false",
    ):
        validate_research_corpus_intake(authority)


def test_truth_boundary_rejects_nonzero_optimized_target_authorization() -> None:
    authority = copy.deepcopy(_authority())
    authority["hard_truth_boundary"]["authorized_unique_optimized_targets"] = 1

    with pytest.raises(ResearchCorpusIntakeError, match="must remain zero"):
        validate_research_corpus_intake(authority)


def test_duplicate_source_identity_is_rejected() -> None:
    authority = copy.deepcopy(_authority())
    authority["terminal_candidate_authorities"][0]["source_id"] = authority[
        "baseline_registry"
    ]["sources"][0]["source_id"]

    with pytest.raises(ResearchCorpusIntakeError, match="duplicate source_id"):
        validate_research_corpus_intake(authority)


def test_evaluation_admission_cannot_leak_into_training_intake() -> None:
    authority = copy.deepcopy(_authority())
    authority["terminal_candidate_authorities"][0]["evaluation_rights"] = "ALLOWED"

    with pytest.raises(ResearchCorpusIntakeError, match="must remain fail-closed"):
        validate_research_corpus_intake(authority)


def test_family_requirement_cannot_be_weakened_below_two() -> None:
    authority = copy.deepcopy(_authority())
    authority["capacity_proxy"]["minimum_independent_families_per_stratum"] = 1

    with pytest.raises(
        ResearchCorpusIntakeError,
        match="cannot be weakened below 2",
    ):
        validate_research_corpus_intake(authority)
