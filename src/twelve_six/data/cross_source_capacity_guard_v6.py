"""Fail-closed exact planning-vector guard for NEXT100-065D/V6.

The V6 composition recomputes accepted-only CPython bytes from the exact source,
but terminal adapter authority fixes that accepted capacity at 15,540 bytes.
The current composed vector additionally binds terminal attrs authority at
170,435 bytes. Any drift must fail closed rather than become new training
capacity implicitly.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CPYTHON_ACCEPTED_ADAPTER_PR = 567
CPYTHON_ACCEPTED_ADAPTER_HEAD = "8f0cbc16f9a920ca9ab3e3061b53fbfec8838d77"
CPYTHON_ACCEPTED_ADAPTER_RUN = 33005689174
CPYTHON_ACCEPTED_CAPACITY_BYTES = 15540
CPYTHON_FULL_NORMALIZED_SOURCE_BYTES = 17901

ATTRS_PR = 474
ATTRS_HEAD = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
ATTRS_RUN = 33006080831
ATTRS_CAPACITY_BYTES = 170435

RESEARCH_CORPUS_V1_TARGET_BYTES = 20_000_000
EXPECTED_PRE_DEDUP_TOTAL_BYTES = 2_215_615
EXPECTED_PRE_DEDUP_GAP_BYTES = 17_784_385
EXPECTED_PRE_DEDUP_BY_MODALITY = {
    "uk": 100_856,
    "en": 1_838_293,
    "code": 276_466,
}


class V6PlanningVectorError(RuntimeError):
    """Raised when V6 drifts from exact terminal source-capacity authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V6PlanningVectorError(message)


def verify_exact_terminal_vector(report: Mapping[str, Any]) -> None:
    """Require the exact post-source, pre-global-dedup planning vector."""
    vector = report.get("source_vector")
    _require(isinstance(vector, Mapping), "V6 source_vector missing")

    cp_capacity = vector.get("cpython_accepted_capacity_bytes")
    _require(
        cp_capacity == CPYTHON_ACCEPTED_CAPACITY_BYTES,
        "accepted-only CPython capacity drift: "
        f"expected {CPYTHON_ACCEPTED_CAPACITY_BYTES}, got {cp_capacity}",
    )
    _require(
        cp_capacity != CPYTHON_FULL_NORMALIZED_SOURCE_BYTES,
        "full CPython normalized source bytes were credited",
    )

    total = vector.get("source_capacity_bytes_before_global_dedup")
    _require(
        total == EXPECTED_PRE_DEDUP_TOTAL_BYTES,
        f"V6 pre-dedup total drift: expected {EXPECTED_PRE_DEDUP_TOTAL_BYTES}, got {total}",
    )

    by_modality = vector.get("source_capacity_by_modality_before_global_dedup")
    _require(
        by_modality == EXPECTED_PRE_DEDUP_BY_MODALITY,
        "V6 pre-dedup modality vector drift: "
        f"expected {EXPECTED_PRE_DEDUP_BY_MODALITY}, got {by_modality}",
    )

    target = vector.get("research_corpus_v1_acquisition_target_bytes")
    _require(
        target == RESEARCH_CORPUS_V1_TARGET_BYTES,
        f"Research Corpus V1 target drift: expected {RESEARCH_CORPUS_V1_TARGET_BYTES}, got {target}",
    )
    gap = vector.get("pre_dedup_planning_gap_bytes")
    _require(
        gap == EXPECTED_PRE_DEDUP_GAP_BYTES,
        f"V6 pre-dedup planning gap drift: expected {EXPECTED_PRE_DEDUP_GAP_BYTES}, got {gap}",
    )
    _require(total + gap == target, "V6 target/capacity/gap arithmetic is inconsistent")
