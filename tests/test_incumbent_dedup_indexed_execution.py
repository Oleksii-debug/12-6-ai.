import pytest

from twelve_six.data.incumbent_dedup_indexed_execution import (
    IndexedExecutionError,
    candidate_pair_indices,
    execution_stats,
)


class FakeV1:
    @staticmethod
    def normalize_for_contamination(text: str, modality: str) -> str:
        del modality
        return " ".join(text.casefold().split())


def _fp(
    source_id: str,
    *,
    modality: str = "uk",
    origin: str | None = None,
    raw: str | None = None,
    normalized: str | None = None,
    shingles: frozenset[str] = frozenset(),
    skeleton: frozenset[str] = frozenset(),
    text: str = "unrelated body",
):
    return {
        "row": {
            "source_id": source_id,
            "modality": modality,
            "origin_key": origin or source_id,
        },
        "raw_sha256": raw or f"raw-{source_id}",
        "normalized_sha256": normalized or f"norm-{source_id}",
        "shingles": shingles,
        "skeleton_shingles": skeleton,
        "text": text,
    }


def test_candidate_index_contains_each_incumbent_necessary_condition():
    shared_edge = (
        "This shared publisher footer has comfortably more than thirty two characters."
    )
    rows = [
        _fp("a", origin="same-origin"),
        _fp("b", origin="same-origin"),
        _fp("c", raw="same-raw"),
        _fp("d", raw="same-raw"),
        _fp("e", normalized="same-normalized"),
        _fp("f", normalized="same-normalized"),
        _fp("g", shingles=frozenset({"natural-shingle"})),
        _fp("h", shingles=frozenset({"natural-shingle"})),
        _fp("i", modality="code", skeleton=frozenset({"skeleton-shingle"})),
        _fp("j", modality="code", skeleton=frozenset({"skeleton-shingle"})),
        _fp("k", text=f"header\n{shared_edge}\nbody"),
        _fp("l", text=f"other\n{shared_edge}\ntail"),
        _fp("m"),
    ]
    pairs = set(candidate_pair_indices(FakeV1, rows))
    assert (0, 1) in pairs
    assert (2, 3) in pairs
    assert (4, 5) in pairs
    assert (6, 7) in pairs
    assert (8, 9) in pairs
    assert (10, 11) in pairs
    assert all(12 not in pair for pair in pairs)


def test_content_shingles_do_not_cross_code_natural_boundary():
    rows = [
        _fp("natural", shingles=frozenset({"same"})),
        _fp("code", modality="code", shingles=frozenset({"same"})),
    ]
    assert candidate_pair_indices(FakeV1, rows) == []


def test_candidate_budget_fails_closed():
    rows = [_fp(str(index), origin="same") for index in range(5)]
    with pytest.raises(IndexedExecutionError, match="candidate pair budget exceeded"):
        candidate_pair_indices(FakeV1, rows, max_candidate_pairs=2)


def test_execution_stats_exact_rada_scale():
    rada = execution_stats(101_559, 0)
    assert rada["incumbent_all_pair_dispatches"] == 5_157_064_461
    combined = execution_stats(101_821, 0)
    assert combined["incumbent_all_pair_dispatches"] == 5_183_707_110


def test_execution_stats_rejects_impossible_candidate_count():
    with pytest.raises(IndexedExecutionError, match="exceeds all-pairs"):
        execution_stats(2, 2)
