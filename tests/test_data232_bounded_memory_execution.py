from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data._data232_decontamination_matching import (
    DEFAULT_THRESHOLDS,
    _fingerprint,
    _iter_blocked_pairs,
    _iter_train_pairs,
)


def _row(
    record_id: str,
    text: str,
    *,
    source: str = "source-a",
    family: str = "family-a",
    modality: str = "en",
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": source,
        "source_family": family,
        "modality": modality,
        "text": text,
    }


def _legacy_keys(fp: Mapping[str, Any]) -> set[str]:
    keys = {"r:" + str(fp["raw"]), "n:" + str(fp["normalized"])}
    keys |= {"s:" + value for value in fp["shingles"]}
    keys |= {"c:" + value for value in fp["skeleton_shingles"]}
    return keys


def _legacy_blocked_pairs(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> set[tuple[int, int]]:
    index: dict[str, set[int]] = {}
    for j, fp in enumerate(right):
        for key in _legacy_keys(fp):
            index.setdefault(key, set()).add(j)
    pairs: set[tuple[int, int]] = set()
    for i, fp in enumerate(left):
        for key in _legacy_keys(fp):
            pairs.update((i, j) for j in index.get(key, ()))
    return pairs


def _legacy_train_pairs(
    train: Sequence[Mapping[str, Any]],
) -> set[tuple[int, int]]:
    index: dict[str, set[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for i, fp in enumerate(train):
        keys = _legacy_keys(fp)
        candidates: set[int] = set()
        for key in keys:
            candidates.update(index.get(key, ()))
        pairs.update((j, i) for j in candidates)
        for key in keys:
            index.setdefault(key, set()).add(i)
    return pairs


def test_fingerprint_retains_counts_not_token_arrays() -> None:
    natural = _fingerprint(
        _row("n1", " ".join(f"word{i}" for i in range(200))),
        DEFAULT_THRESHOLDS,
    )
    assert natural["token_count"] == 200
    assert natural["skeleton_token_count"] == 0
    assert "tokens" not in natural
    assert "skeleton" not in natural

    code = _fingerprint(
        _row(
            "c1",
            "def compute(alpha, beta):\n"
            "    # comment\n"
            "    value = alpha + beta\n"
            "    return value * 3\n",
            source="repo",
            family="repo",
            modality="code",
        ),
        DEFAULT_THRESHOLDS,
    )
    assert code["token_count"] > 0
    assert code["skeleton_token_count"] > 0
    assert "tokens" not in code
    assert "skeleton" not in code


def test_streamed_candidate_relations_equal_incumbent_materialized_relations() -> None:
    base = " ".join(f"token{i}" for i in range(70))
    train_rows = [
        _row("t0", base, source="a", family="a"),
        _row("t1", base.replace("token20", "edit20"), source="b", family="b"),
        _row("t2", "unrelated independent text payload", source="c", family="c"),
        _row(
            "t3",
            "def compute(alpha, beta):\n"
            "    total = alpha + beta\n"
            "    if total > 10:\n"
            "        return total * 3\n"
            "    return total - 2\n",
            source="repo-a",
            family="repo-a",
            modality="code",
        ),
    ]
    eval_rows = [
        _row(
            "e0",
            base.replace("token40", "reserved40"),
            source="eval-a",
            family="eval-a",
        ),
        _row(
            "e1",
            "def calculate(left, right):\n"
            "    result = left + right\n"
            "    if result > 999:\n"
            "        return result * 44\n"
            "    return result - 88\n",
            source="eval-code",
            family="eval-code",
            modality="code",
        ),
    ]

    train = [_fingerprint(row, DEFAULT_THRESHOLDS) for row in train_rows]
    evaluation = [_fingerprint(row, DEFAULT_THRESHOLDS) for row in eval_rows]

    assert set(_iter_blocked_pairs(train, evaluation)) == _legacy_blocked_pairs(
        train,
        evaluation,
    )
    assert set(_iter_train_pairs(train)) == _legacy_train_pairs(train)


def test_streamed_candidate_relations_do_not_duplicate_pairs() -> None:
    repeated = "one two three four five six seven eight nine ten"
    train = [
        _fingerprint(
            _row(f"t{i}", repeated, source=f"s{i}", family=f"f{i}"),
            DEFAULT_THRESHOLDS,
        )
        for i in range(8)
    ]
    evaluation = [
        _fingerprint(
            _row("e0", repeated, source="eval", family="eval"),
            DEFAULT_THRESHOLDS,
        )
    ]

    blocked = list(_iter_blocked_pairs(train, evaluation))
    peer = list(_iter_train_pairs(train))

    assert len(blocked) == len(set(blocked))
    assert len(peer) == len(set(peer))
    assert set(blocked) == _legacy_blocked_pairs(train, evaluation)
    assert set(peer) == _legacy_train_pairs(train)
