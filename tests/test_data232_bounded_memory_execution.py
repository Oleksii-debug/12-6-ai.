from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data import decontamination_authority_v2 as authority
from twelve_six.data._data232_decontamination_matching import (
    DEFAULT_THRESHOLDS,
    _fingerprint,
    _iter_blocked_pairs,
    _iter_train_pairs,
    _iter_viable_train_pairs,
    _pair,
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


def _authorities() -> dict[str, Any]:
    return {
        "schema": "12-6.data232-reserved-authorities.v1",
        "authorities": [
            {
                "authority_id": "selection",
                "identity_sha256": "1" * 64,
                "role": "selection_validation",
                "source_sha": "a" * 40,
            },
            {
                "authority_id": "final",
                "identity_sha256": "2" * 64,
                "role": "final_test",
                "source_sha": "b" * 40,
            },
        ],
    }


def _build_report(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return authority.build_report(
        train,
        evaluation,
        training_corpus_identity="3" * 64,
        selection_validation_identity="4" * 64,
        final_test_identity="5" * 64,
        authorities=_authorities(),
        quarantine_cross_source_families=True,
    )


def test_streaming_report_bytes_equal_materialized_incumbent_candidate_order(
    monkeypatch: Any,
) -> None:
    base = " ".join(f"chain{i}" for i in range(80))
    train = [
        _row("t0", base, source="publisher-a", family="family-a"),
        _row(
            "t1",
            base.replace("chain10", "bridge10"),
            source="mirror-b",
            family="family-b",
        ),
        _row("t2", "independent clean sibling", source="publisher-a", family="family-a"),
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
    evaluation = [
        _row(
            "e0",
            base.replace("chain10", "bridge10").replace("chain35", "eval35"),
            source="eval",
            family="eval",
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

    streamed = _build_report(train, evaluation)

    def materialized_blocked(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]):
        pairs = _legacy_blocked_pairs(left, right)
        return iter(sorted(pairs))

    def materialized_train(
        rows: Sequence[dict[str, Any]],
        thresholds: Mapping[str, Any],
    ):
        del thresholds
        pairs = _legacy_train_pairs(rows)
        return iter(sorted(pairs))

    monkeypatch.setattr(authority, "_iter_blocked_pairs", materialized_blocked)
    monkeypatch.setattr(authority, "_iter_viable_train_pairs", materialized_train)
    incumbent_order = _build_report(train, evaluation)

    assert json.dumps(streamed, sort_keys=True, separators=(",", ":")) == json.dumps(
        incumbent_order,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert streamed["report_sha256"] == incumbent_order["report_sha256"]

def _matching_peer_pairs(
    fingerprints: Sequence[dict[str, Any]],
) -> tuple[set[tuple[int, int]], set[str]]:
    pairs: set[tuple[int, int]] = set()
    match_types: set[str] = set()
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            evidence = _pair(
                fingerprints[left],
                fingerprints[right],
                "peer",
                DEFAULT_THRESHOLDS,
            )
            if evidence:
                pairs.add((left, right))
                match_types.update(item["match_type"] for item in evidence)
    return pairs, match_types


def test_viable_peer_filter_covers_every_incumbent_match_type() -> None:
    natural = " ".join(f"natural{i}" for i in range(80))
    mixed = " ".join(f"mixed{i}" for i in range(70))
    rows = [
        _row("raw-a", "same exact short text", source="a", family="a"),
        _row("raw-b", "same exact short text", source="b", family="b"),
        _row("norm-a", "Hello   WORLD", source="c", family="c"),
        _row("norm-b", "hello world", source="d", family="d"),
        _row("near-a", natural, source="e", family="e"),
        _row(
            "near-b",
            natural.replace("natural41", "replacement41"),
            source="f",
            family="f",
        ),
        _row("mixed-a", mixed, source="g", family="g", modality="code"),
        _row(
            "mixed-b",
            mixed.replace("mixed35", "replacement35"),
            source="h",
            family="h",
            modality="en",
        ),
        _row(
            "code-a",
            "def compute(alpha, beta):\n"
            "    total = alpha + beta\n"
            "    if total > 10:\n"
            "        return total * 3\n"
            "    return total - 2\n",
            source="repo-a",
            family="repo-a",
            modality="code",
        ),
        _row(
            "code-b",
            "def calculate(left, right):\n"
            "    result = left + right\n"
            "    if result > 999:\n"
            "        return result * 44\n"
            "    return result - 88\n",
            source="repo-b",
            family="repo-b",
            modality="code",
        ),
    ]
    fingerprints = [_fingerprint(row, DEFAULT_THRESHOLDS) for row in rows]
    expected, match_types = _matching_peer_pairs(fingerprints)
    viable = set(_iter_viable_train_pairs(fingerprints, DEFAULT_THRESHOLDS))

    assert {
        "raw_exact",
        "normalized_exact",
        "near_match",
        "document_fragment",
        "code_fork_copy",
    } <= match_types
    assert expected <= viable


def test_viable_peer_filter_is_complete_on_deterministic_adversarial_sets() -> None:
    rng = random.Random(232)
    universe = [f"shingle-{index}" for index in range(48)]
    skeleton_universe = [f"skeleton-{index}" for index in range(36)]

    for round_index in range(6):
        fingerprints: list[dict[str, Any]] = []
        for index in range(64):
            is_code = rng.randrange(3) == 0
            content_count = rng.randrange(0, 31)
            skeleton_count = rng.randrange(0, 25) if is_code else 0
            fingerprints.append(
                {
                    "record": {
                        "record_id": f"r{round_index}-{index}",
                        "source_id": f"s{index}",
                        "source_family": f"f{index % 9}",
                        "modality": "code" if is_code else "en",
                        "text": "",
                    },
                    "raw": f"raw-{rng.randrange(0, 41)}",
                    "normalized": f"normalized-{rng.randrange(0, 43)}",
                    "token_count": rng.randrange(0, 60),
                    "shingles": frozenset(
                        rng.sample(universe, content_count)
                    ),
                    "skeleton_token_count": rng.randrange(0, 50),
                    "skeleton_shingles": frozenset(
                        rng.sample(skeleton_universe, skeleton_count)
                    ),
                }
            )

        expected, _ = _matching_peer_pairs(fingerprints)
        generated = list(
            _iter_viable_train_pairs(fingerprints, DEFAULT_THRESHOLDS)
        )
        assert len(generated) == len(set(generated))
        assert expected <= set(generated)


def test_viable_peer_filter_prunes_common_shingle_false_candidates() -> None:
    rows = [
        _row(
            f"record-{index}",
            "common bridge anchor "
            + " ".join(f"unique{index}_{part}" for part in range(45)),
            source=f"source-{index}",
            family=f"family-{index}",
        )
        for index in range(60)
    ]
    fingerprints = [_fingerprint(row, DEFAULT_THRESHOLDS) for row in rows]
    incumbent = set(_iter_train_pairs(fingerprints))
    viable = set(_iter_viable_train_pairs(fingerprints, DEFAULT_THRESHOLDS))
    actual_matches, _ = _matching_peer_pairs(fingerprints)

    assert len(incumbent) == 1770
    assert actual_matches == set()
    assert viable == set()

