from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.cross_source_capacity_audit import (
    CapacityAuditError,
    audit_payloads,
    verify_report,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source(
    source_id: str,
    family: str,
    modality: str,
    payload: bytes,
    *,
    capacity: int | None = None,
    status: str = "REGISTRY_TERMINAL",
    origin_key: str | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": family,
        "modality": modality,
        "evidence_status": status,
        "declared_capacity_bytes": capacity or len(payload),
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": _sha(payload),
        "acquisition_url": f"https://example.invalid/{source_id}",
        "origin_key": origin_key or f"origin:{source_id}",
    }


def _inventory(*sources: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "12-6.data298-cross-source-inventory.v1",
        "local_free_only": True,
        "sources": list(sources),
    }


def _types(report: dict[str, object]) -> set[str]:
    return {item["match_type"] for item in report["matches"]}


def test_raw_exact_alias_collapses_capacity_and_family_count() -> None:
    payload = ("same external source payload with enough words for stable comparison " * 3).encode()
    inv = _inventory(
        _source("a", "publisher:a", "en", payload, capacity=100),
        _source("b", "mirror:b", "en", payload, capacity=100),
    )
    report = audit_payloads(inv, {"a": payload, "b": payload})
    verify_report(report)
    assert "raw_exact" in _types(report)
    scope = report["scopes"]["canonical_registry"]
    assert scope["declared_capacity_bytes_before"] == 200
    assert scope["conservative_unique_capacity_bytes_after"] == 100
    assert scope["duplicate_discount_bytes"] == 100
    assert scope["declared_family_count"] == 2
    assert scope["effective_independent_family_count"] == 1


def test_origin_alias_collapses_even_when_wrapper_bytes_differ() -> None:
    a = b"Canonical object text one two three four five six seven eight nine ten."
    b = b"Canonical object text one two three four five six seven eight nine ten.\n"
    key = "github:owner/repo:deadbeef:path/file.txt"
    inv = _inventory(
        _source("source-a", "family-a", "en", a, origin_key=key),
        _source("source-b", "family-b", "en", b, origin_key=key),
    )
    report = audit_payloads(inv, {"source-a": a, "source-b": b})
    assert "origin_alias" in _types(report)
    assert report["scopes"]["canonical_registry"]["duplicate_cluster_count"] == 1


def test_nfkc_invisible_and_whitespace_normalized_exact_collapses() -> None:
    a = "Café has normalized words and spacing for duplicate detection.\nSecond line here.".encode("utf-8")
    b = "Cafe\u0301   has normalized words\u200b and spacing for duplicate detection.  Second line here.".encode("utf-8")
    inv = _inventory(
        _source("a", "family-a", "en", a),
        _source("b", "family-b", "en", b),
    )
    report = audit_payloads(inv, {"a": a, "b": b})
    assert "normalized_exact" in _types(report)
    assert report["scopes"]["canonical_registry"]["duplicate_discount_bytes"] > 0


def test_document_fragment_detects_copied_document_with_publisher_wrapper() -> None:
    core = " ".join(f"distinctword{i}" for i in range(80))
    header = "Publisher Network Standard Documentation Header Shared Across All Copies"
    footer = "Publisher Network Standard Documentation Footer Shared Across All Copies"
    a = f"{header}\n{core}\n{footer}".encode()
    b = f"{header}\n{core} appended appendix words alpha beta gamma delta epsilon zeta eta theta\n{footer}".encode()
    inv = _inventory(
        _source("upstream", "publisher", "en", a),
        _source("web-mirror", "mirror", "en", b),
    )
    report = audit_payloads(inv, {"upstream": a, "web-mirror": b})
    kinds = _types(report)
    assert "publisher_boilerplate" in kinds
    assert kinds & {"near_match", "document_fragment"}
    assert report["scopes"]["canonical_registry"]["duplicate_cluster_count"] == 1


def test_publisher_boilerplate_alone_does_not_collapse_unrelated_documents() -> None:
    header = "Publisher Network Standard Documentation Header Shared Across All Copies"
    footer = "Publisher Network Standard Documentation Footer Shared Across All Copies"
    a_body = " ".join(f"astronomy{i}" for i in range(70))
    b_body = " ".join(f"botany{i}" for i in range(70))
    a = f"{header}\n{a_body}\n{footer}".encode()
    b = f"{header}\n{b_body}\n{footer}".encode()
    inv = _inventory(
        _source("a", "publisher", "en", a),
        _source("b", "publisher", "en", b),
    )
    report = audit_payloads(inv, {"a": a, "b": b})
    assert "publisher_boilerplate" in _types(report)
    collapsing = [m for m in report["matches"] if m["capacity_collapsing"]]
    assert collapsing == []
    scope = report["scopes"]["canonical_registry"]
    assert scope["duplicate_cluster_count"] == 0
    assert scope["duplicate_discount_bytes"] == 0


def test_code_fork_copy_survives_identifier_literal_and_comment_changes() -> None:
    left = b"""def calculate_total(items):\n    # original comment\n    subtotal = sum(items)\n    tax = subtotal * 0.20\n    message = \"invoice\"\n    return subtotal + tax\n\ndef render(value):\n    return calculate_total(value)\n"""
    right = b"""def compute_amount(values):\n    # fork comment changed\n    base = sum(values)\n    fee = base * 0.15\n    label = \"receipt\"\n    return base + fee\n\ndef show(data):\n    return compute_amount(data)\n"""
    inv = _inventory(
        _source("repo-a", "github:upstream", "code", left),
        _source("repo-b", "github:fork", "code", right),
    )
    report = audit_payloads(inv, {"repo-a": left, "repo-b": right})
    assert "code_fork_copy" in _types(report)
    assert report["scopes"]["canonical_registry"]["effective_independent_family_count"] == 1


def test_status_scopes_do_not_promote_probe_nonterminal() -> None:
    a = b"terminal registry object unique words alpha beta gamma delta epsilon zeta"
    b = b"dedicated terminal code object one two three four five six seven eight"
    c = b"probe nonterminal object separate corpus words red orange yellow green blue"
    inv = _inventory(
        _source("a", "fa", "en", a, capacity=10, status="REGISTRY_TERMINAL"),
        _source("b", "fb", "code", b, capacity=20, status="DEDICATED_TERMINAL"),
        _source("c", "fc", "en", c, capacity=30, status="PROBE_NONTERMINAL"),
    )
    report = audit_payloads(inv, {"a": a, "b": b, "c": c})
    assert report["scopes"]["canonical_registry"]["declared_capacity_bytes_before"] == 10
    assert report["scopes"]["terminal_evidence"]["declared_capacity_bytes_before"] == 30
    assert report["scopes"]["all_observed"]["declared_capacity_bytes_before"] == 60


def test_repository_inventory_declared_capacity_boundaries() -> None:
    path = Path("configs/data/data298_cross_source_inventory_v1.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["sources"]
    canonical = [x for x in rows if x["evidence_status"] == "REGISTRY_TERMINAL"]
    terminal = [x for x in rows if x["evidence_status"] in {"REGISTRY_TERMINAL", "DEDICATED_TERMINAL"}]
    assert len(rows) == 7
    assert len({x["source_family"] for x in rows}) == 6
    assert sum(x["declared_capacity_bytes"] for x in canonical) == 173358
    assert sum(x["declared_capacity_bytes"] for x in terminal) == 183061
    assert sum(x["declared_capacity_bytes"] for x in rows) == 207771
    assert sum(x["expected_raw_bytes"] for x in rows) == 480273


def test_fail_closed_on_payload_identity_mismatch() -> None:
    payload = b"immutable payload"
    row = _source("a", "f", "en", payload)
    inv = _inventory(row)
    with pytest.raises(CapacityAuditError, match="raw size changed|raw SHA-256 mismatch"):
        audit_payloads(inv, {"a": payload + b"!"})
