from __future__ import annotations

import copy

import pytest

from twelve_six.data.cross_source_capacity_guard_v6 import (
    V6PlanningVectorError,
    verify_exact_terminal_vector,
)


def _report() -> dict[str, object]:
    return {
        "source_vector": {
            "cpython_accepted_capacity_bytes": 15540,
            "source_capacity_bytes_before_global_dedup": 2045180,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": 100856,
                "en": 1838293,
                "code": 106031,
            },
            "research_corpus_v1_acquisition_target_bytes": 20000000,
            "pre_dedup_planning_gap_bytes": 17954820,
        }
    }


def test_exact_terminal_vector_accepts_canonical_v6_planning_numbers() -> None:
    verify_exact_terminal_vector(_report())


def test_exact_terminal_vector_rejects_recomputed_cpython_capacity_drift() -> None:
    report = copy.deepcopy(_report())
    report["source_vector"]["cpython_accepted_capacity_bytes"] = 15539
    report["source_vector"]["source_capacity_bytes_before_global_dedup"] = 2045179
    report["source_vector"]["pre_dedup_planning_gap_bytes"] = 17954821
    with pytest.raises(V6PlanningVectorError, match="accepted-only CPython capacity drift"):
        verify_exact_terminal_vector(report)


def test_exact_terminal_vector_rejects_full_cpython_source_credit() -> None:
    report = copy.deepcopy(_report())
    report["source_vector"]["cpython_accepted_capacity_bytes"] = 17901
    report["source_vector"]["source_capacity_bytes_before_global_dedup"] = 2047541
    report["source_vector"]["pre_dedup_planning_gap_bytes"] = 17952459
    with pytest.raises(V6PlanningVectorError, match="accepted-only CPython capacity drift"):
        verify_exact_terminal_vector(report)


def test_exact_terminal_vector_rejects_modality_arithmetic_drift() -> None:
    report = copy.deepcopy(_report())
    report["source_vector"]["source_capacity_by_modality_before_global_dedup"]["en"] -= 1
    with pytest.raises(V6PlanningVectorError, match="modality vector drift"):
        verify_exact_terminal_vector(report)
