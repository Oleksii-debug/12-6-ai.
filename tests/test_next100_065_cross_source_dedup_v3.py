from __future__ import annotations

import hashlib

import pytest

from twelve_six.data.cross_source_capacity_audit_v3 import (
    CrossSourceV3Error,
    audit_payloads,
    verify_report,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(
    source_id: str,
    family: str,
    origin: str,
    object_id: str,
    payload: bytes,
    *,
    modality: str = "en",
    capacity: int | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": family,
        "stable_origin_id": origin,
        "stable_object_id": object_id,
        "modality": modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": capacity or len(payload),
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": _sha(payload),
        "acquisition_url": f"https://different.example.invalid/{source_id}",
        "origin_key": f"url-origin:{source_id}",
    }


def _inventory(
    sources: list[dict[str, object]],
    edges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "12-6.next100-065-cross-source-dedup.v3",
        "local_free_only": True,
        "model_training_executed": False,
        "sources": sources,
        "lineage_edges": edges or [],
    }


def _edge(left: str, right: str, relation: str, *, capacity: bool = True) -> dict[str, object]:
    return {
        "left_source_id": left,
        "right_source_id": right,
        "relation": relation,
        "capacity_collapsing": capacity,
        "independence_collapsing": True,
        "evidence": "synthetic lineage authority for adversarial test",
    }


def test_different_urls_same_stable_object_cannot_multiply_capacity() -> None:
    a = b"canonical object payload alpha beta gamma delta epsilon"
    b = b"canonical object payload alpha beta gamma delta epsilon\n"
    sources = [
        _source("a", "family-a", "origin-a", "stable-object-1", a, capacity=100),
        _source("b", "family-b", "origin-b", "stable-object-1", b, capacity=100),
    ]
    report = audit_payloads(_inventory(sources), {"a": a, "b": b})
    verify_report(report)
    kinds = {match["match_type"] for match in report["matches"]}
    assert "lineage_same_origin_alias" in kinds
    scope = report["terminal_candidates"]
    assert scope["declared_capacity_bytes_before"] == 200
    assert scope["conservative_unique_capacity_bytes_after"] == 100
    assert scope["effective_independent_origin_count"] == 1


@pytest.mark.parametrize("relation", ["mirror", "fork", "vendor", "generated_derivative"])
def test_declared_derivative_relationship_collapses_independence_and_capacity(relation: str) -> None:
    a = b"upstream content that is deliberately byte-different from downstream wrapper"
    b = b"downstream rewritten wrapper with deliberately distinct lexical surface"
    sources = [
        _source("upstream", "fa", "origin-up", "object-up", a, capacity=90),
        _source("derived", "fb", "origin-down", "object-down", b, capacity=70),
    ]
    report = audit_payloads(_inventory(sources, [_edge("upstream", "derived", relation)]), {"upstream": a, "derived": b})
    scope = report["terminal_candidates"]
    assert scope["duplicate_cluster_count"] == 1
    assert scope["conservative_unique_capacity_bytes_after"] == 90
    assert scope["effective_independent_origin_count"] == 1


def test_repository_transfer_alias_collapses_even_without_url_or_family_identity() -> None:
    a = b"old repository transfer snapshot alpha beta gamma"
    b = b"new repository transfer snapshot alpha beta gamma plus wrapper"
    sources = [
        _source("old", "github:old/name", "repo-old", "old-object", a, capacity=60),
        _source("new", "github:new/name", "repo-new", "new-object", b, capacity=65),
    ]
    report = audit_payloads(
        _inventory(sources, [_edge("old", "new", "repository_transfer_alias")]),
        {"old": a, "new": b},
    )
    assert report["terminal_candidates"]["conservative_unique_capacity_bytes_after"] == 65
    assert report["terminal_candidates"]["effective_independent_origin_count"] == 1


def test_same_origin_siblings_are_one_origin_but_keep_distinct_capacity() -> None:
    a = b"chapter one astronomy stars planets galaxies quasars"
    b = b"chapter two botany roots leaves stems flowers pollen"
    sources = [
        _source("one", "manual", "github:publisher/manual", "object-1", a, capacity=50),
        _source("two", "manual", "github:publisher/manual", "object-2", b, capacity=70),
    ]
    edge = _edge("one", "two", "sibling_same_origin", capacity=False)
    report = audit_payloads(_inventory(sources, [edge]), {"one": a, "two": b})
    scope = report["terminal_candidates"]
    assert scope["conservative_unique_capacity_bytes_after"] == 120
    assert scope["duplicate_cluster_count"] == 0
    assert scope["stable_origin_count"] == 1
    assert scope["effective_independent_origin_count"] == 1


def test_connected_duplicate_graph_collapses_transitively() -> None:
    payloads = {
        "a": b"source a alpha beta gamma delta",
        "b": b"source b epsilon zeta eta theta",
        "c": b"source c iota kappa lambda mu",
    }
    sources = [
        _source("a", "fa", "oa", "xa", payloads["a"], capacity=100),
        _source("b", "fb", "ob", "xb", payloads["b"], capacity=80),
        _source("c", "fc", "oc", "xc", payloads["c"], capacity=60),
    ]
    edges = [_edge("a", "b", "fork"), _edge("b", "c", "vendor")]
    report = audit_payloads(_inventory(sources, edges), payloads)
    scope = report["terminal_candidates"]
    assert scope["duplicate_clusters"] == [["a", "b", "c"]]
    assert scope["declared_capacity_bytes_before"] == 240
    assert scope["conservative_unique_capacity_bytes_after"] == 100


def test_authority_specific_rust_prose_normalization_is_hash_bound() -> None:
    raw = b"# Heading\n\nPlain `code` prose.\n\n```rust\nfn main() {}\n```\n"
    expected = b"Heading\n\nPlain prose.\n"
    source = _source("rust", "rust-book", "rust-origin", "rust-object", raw, capacity=len(expected))
    source.update(
        {
            "comparison_normalization": "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1",
            "expected_comparison_bytes": len(expected),
            "expected_comparison_sha256": _sha(expected),
        }
    )
    report = audit_payloads(_inventory([source]), {"rust": raw})
    row = report["sources"][0]
    assert row["comparison_policy"] == "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1"
    assert row["comparison_payload_bytes"] == len(expected)
    assert row["comparison_payload_sha256"] == _sha(expected)
    assert report["terminal_candidates"]["declared_capacity_bytes_before"] == len(expected)


def test_authority_specific_normalization_drift_fails_closed() -> None:
    raw = b"# Heading\n\nPlain `code` prose.\n"
    expected = b"Heading\n\nPlain prose.\n"
    source = _source("rust", "rust-book", "rust-origin", "rust-object", raw, capacity=len(expected))
    source.update(
        {
            "comparison_normalization": "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1",
            "expected_comparison_bytes": len(expected),
            "expected_comparison_sha256": "0" * 64,
        }
    )
    with pytest.raises(CrossSourceV3Error, match="comparison SHA-256 changed"):
        audit_payloads(_inventory([source]), {"rust": raw})


def test_nonterminal_source_fails_closed() -> None:
    payload = b"candidate"
    source = _source("x", "fx", "ox", "xx", payload)
    source["evidence_status"] = "PROBE_NONTERMINAL"
    with pytest.raises(CrossSourceV3Error, match="nonterminal source"):
        audit_payloads(_inventory([source]), {"x": payload})
