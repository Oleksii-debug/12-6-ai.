from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "run_d03_caselaw_source_admission_replay_v1.py"
spec = importlib.util.spec_from_file_location("caselaw_source_admission_replay", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _raw_record(
    record_id: str,
    *,
    source: str = "Caselaw Access Project",
    url: str = "https://static.case.law/",
) -> dict:
    return {
        "id": record_id,
        "source": source,
        "added": "2024-08-24T03:29:51.129235",
        "created": "2024-08-24T03:29:51.129683",
        "metadata": {"author": "PER CURIAM", "license": "Public Domain", "url": url},
        "text": "Fixture text is not emitted by the execution report.",
    }


def _write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wb") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True).encode("utf-8") + b"\n")


def _admission_module() -> SimpleNamespace:
    def assess_record(record: dict, _policy: dict) -> dict:
        source = record["source"]
        if source in {"Court Listener", "CourtListener"}:
            return {
                "decision": "NOT_SOURCE_ADMITTED",
                "reason": "unsupported_courtlistener_source",
            }
        url = record["metadata"]["url"]
        if not url.startswith(("https://case.law/", "https://static.case.law/")):
            return {
                "decision": "NOT_SOURCE_ADMITTED",
                "reason": "cap_url_authority_mismatch",
            }
        return {
            "decision": "CONDITIONAL_SOURCE_ADMISSION",
            "reason": "exact_cap_source_license_and_authority",
        }

    return SimpleNamespace(assess_record=assess_record)


def test_scan_source_admission_applies_raw_record_authority(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.gz"
    _write_gzip(
        source,
        [
            _raw_record("cap/1"),
            _raw_record("court/1", source="Court Listener"),
            _raw_record("bad/1", url="https://example.com/"),
        ],
    )
    decisions, summary = mod.scan_source_admission(
        [source],
        _admission_module(),
        {},
        max_line_bytes=10_000,
        max_scanned_bytes=100_000,
    )
    assert decisions["cap/1"]["decision"] == "CONDITIONAL_SOURCE_ADMISSION"
    assert decisions["court/1"]["reason"] == "unsupported_courtlistener_source"
    assert decisions["bad/1"]["reason"] == "cap_url_authority_mismatch"
    assert summary["rows_scanned"] == 3
    assert summary["scan_budget_reached"] is False


def test_scan_source_admission_rejects_duplicate_raw_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.gz"
    _write_gzip(source, [_raw_record("dup"), _raw_record("dup")])
    with pytest.raises(mod.ReplayError, match="duplicate raw record id"):
        mod.scan_source_admission(
            [source],
            _admission_module(),
            {},
            max_line_bytes=10_000,
            max_scanned_bytes=100_000,
        )


def _candidate_row(record_id: str, source_kind: str, text: str) -> dict:
    payload = text.encode("utf-8")
    return {
        "record_id": record_id,
        "normalized_sha256": mod.sha256(payload),
        "normalized_utf8_bytes": len(payload),
        "text": text,
        "source_family": "en.common-pile.caselaw",
        "source_kind": source_kind,
        "rights_basis": "UPSTREAM_PUBLIC_DOMAIN_METADATA_REVIEW_REQUIRED",
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def test_filter_candidate_keeps_only_source_admitted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _candidate_row("cap/1", "Caselaw Access Project", "alpha"),
        _candidate_row("court/1", "Court Listener", "bravo"),
    ]
    raw = b"".join(mod.canonical(row) for row in rows)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    admitted = tmp_path / "admitted.jsonl"

    monkeypatch.setattr(mod, "HISTORICAL_CANDIDATE_SHA256", mod.sha256(raw))
    monkeypatch.setattr(mod, "HISTORICAL_RETAINED_RECORDS", 2)
    monkeypatch.setattr(
        mod,
        "HISTORICAL_RETAINED_NORMALIZED_BYTES",
        sum(row["normalized_utf8_bytes"] for row in rows),
    )
    decisions = {
        "cap/1": {
            "decision": "CONDITIONAL_SOURCE_ADMISSION",
            "reason": "exact_cap_source_license_and_authority",
        },
        "court/1": {
            "decision": "NOT_SOURCE_ADMITTED",
            "reason": "unsupported_courtlistener_source",
        },
    }
    summary = mod.filter_candidate(candidate, admitted, decisions)
    assert summary["source_admitted_records"] == 1
    assert summary["source_admitted_normalized_utf8_bytes"] == len("alpha".encode())
    assert summary["candidate_denied_reason_counts"] == {
        "unsupported_courtlistener_source": 1
    }
    assert admitted.read_bytes() == mod.canonical(rows[0])
    assert "text" not in summary


def test_filter_candidate_requires_raw_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row("cap/1", "Caselaw Access Project", "alpha")
    raw = mod.canonical(row)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    monkeypatch.setattr(mod, "HISTORICAL_CANDIDATE_SHA256", mod.sha256(raw))
    monkeypatch.setattr(mod, "HISTORICAL_RETAINED_RECORDS", 1)
    monkeypatch.setattr(mod, "HISTORICAL_RETAINED_NORMALIZED_BYTES", len("alpha"))
    with pytest.raises(mod.ReplayError, match="lacks raw source-admission decision"):
        mod.filter_candidate(candidate, tmp_path / "admitted.jsonl", {})


def test_filter_candidate_rejects_admitted_non_cap_source_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row("court/1", "Court Listener", "alpha")
    raw = mod.canonical(row)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    monkeypatch.setattr(mod, "HISTORICAL_CANDIDATE_SHA256", mod.sha256(raw))
    monkeypatch.setattr(mod, "HISTORICAL_RETAINED_RECORDS", 1)
    monkeypatch.setattr(mod, "HISTORICAL_RETAINED_NORMALIZED_BYTES", len("alpha"))
    decisions = {
        "court/1": {
            "decision": "CONDITIONAL_SOURCE_ADMISSION",
            "reason": "forged",
        }
    }
    with pytest.raises(mod.ReplayError, match="source-kind drift"):
        mod.filter_candidate(candidate, tmp_path / "admitted.jsonl", decisions)


def test_git_blob_identity_is_content_bound() -> None:
    raw = b"example\n"
    assert mod.git_blob_sha1(raw) != mod.git_blob_sha1(raw + b"x")
