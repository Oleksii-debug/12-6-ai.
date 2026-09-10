from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_d03_rada_trees_quality_windows_authorized.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_quality_windows_authorized", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _candidate_row(text: str = "Парламентський текст") -> dict[str, Any]:
    raw = text.encode("utf-8")
    source_sha = "b" * 64
    source_path = "texts/2020/01/01/session.txt"
    return {
        "record_id": f"rada-trees:{source_sha}:{source_path}",
        "source_family": MODULE.EXPECTED_FAMILY,
        "source_dataset": MODULE.EXPECTED_DATASET,
        "source_revision": MODULE.EXPECTED_REVISION,
        "source_archive": MODULE.EXPECTED_ARCHIVE,
        "source_path": source_path,
        "session_date": "2020-01-01",
        "source_payload_sha256": source_sha,
        "source_payload_bytes": 123,
        "decoded_encoding": "utf-8",
        "decoded_text_utf8_sha256": _sha(raw),
        "decoded_text_utf8_bytes": len(raw),
        "rights_scope_status": MODULE.EXPECTED_RIGHTS_STATUS,
        "attribution_required": True,
        "language_quality_privacy_complete": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def _write_candidate(path: Path, row: dict[str, Any]) -> None:
    path.write_bytes(MODULE.canonical_bytes(row))


def _upstream_report(candidate: Path, row: dict[str, Any]) -> dict[str, Any]:
    candidate_raw = candidate.read_bytes()
    core = {
        "schema_version": MODULE.UPSTREAM_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset": MODULE.EXPECTED_DATASET,
            "revision": MODULE.EXPECTED_REVISION,
            "archive": MODULE.EXPECTED_ARCHIVE,
            "archive_bytes": MODULE.EXPECTED_ARCHIVE_BYTES,
            "archive_sha256": MODULE.EXPECTED_ARCHIVE_SHA256,
            "family": MODULE.EXPECTED_FAMILY,
        },
        "parent_authority": {
            "full_scan_evidence_identity_sha256": MODULE.EXPECTED_FULL_SCAN_EVIDENCE_ID,
            "rights_config_sha256": MODULE.EXPECTED_RIGHTS_CONFIG_SHA256,
            "rights_report_sha256": MODULE.EXPECTED_RIGHTS_REPORT_SHA256,
            "accepted_path_inventory_sha256": MODULE.EXPECTED_ACCEPTED_INVENTORY,
            "held_path_inventory_sha256": MODULE.EXPECTED_HELD_INVENTORY,
        },
        "materialization": {
            "candidate_jsonl_sha256": _sha(candidate_raw),
            "candidate_jsonl_bytes": len(candidate_raw),
            "candidate_records": 1,
            "accepted_source_payload_bytes": row["source_payload_bytes"],
            "decoded_text_utf8_bytes": row["decoded_text_utf8_bytes"],
            "source_native_document_boundaries_preserved": True,
            "record_text_persisted_in_report": False,
        },
        "claim_boundary": {
            "candidate_jsonl_is_canonical_corpus": False,
            "language_quality_privacy_complete": False,
            "global_dedup_complete": False,
            "reserved_evaluation_decontamination_complete": False,
            "family_caps_complete": False,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
        "safe_result": MODULE.EXPECTED_SAFE_RESULT,
    }
    return {**core, "report_sha256": _sha(MODULE.canonical_bytes(core))}


def _write_upstream(
    path: Path,
    candidate: Path,
    row: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    report = _upstream_report(candidate, row)
    path.write_bytes(MODULE.canonical_bytes(report))
    monkeypatch.setattr(MODULE, "EXPECTED_UPSTREAM_REPORT_SHA256", report["report_sha256"])
    monkeypatch.setattr(MODULE, "EXPECTED_CANDIDATE_RECORDS", 1)
    monkeypatch.setattr(
        MODULE,
        "EXPECTED_ACCEPTED_SOURCE_PAYLOAD_BYTES",
        row["source_payload_bytes"],
    )
    return report


def _fake_delegate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    def fake(
        candidate: Path,
        output: Path,
        report_path: Path,
        *,
        expected_candidate_sha256: str,
    ) -> dict[str, Any]:
        seen.append(expected_candidate_sha256)
        output.write_bytes(b"")
        report_path.write_bytes(b"{}\n")
        return {
            "input_candidate_jsonl_sha256": expected_candidate_sha256,
            "report_sha256": "c" * 64,
            "output_jsonl_sha256": hashlib.sha256(b"").hexdigest(),
            "output_records": 0,
            "quality": {"policy_sha256": "e" * 64},
            "privacy": {"policy_sha256": "d" * 64},
            "claim_boundary": {
                "training_authorized_bytes": 0,
                "unique_causal_loss_positions_authorized": 0,
                "optimizer_updates": 0,
                "model_training_executed": False,
                "paid_compute_used": False,
            },
        }

    monkeypatch.setattr(MODULE._delegate, "materialize", fake)
    return seen


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: dict[str, Any] | None = None):
    row = row or _candidate_row()
    candidate = tmp_path / "candidate.jsonl"
    upstream = tmp_path / "handoff.json"
    output = tmp_path / "out.jsonl"
    materialization_report = tmp_path / "materialization.json"
    authority_report = tmp_path / "authority.json"
    _write_candidate(candidate, row)
    report = _write_upstream(upstream, candidate, row, monkeypatch)
    seen = _fake_delegate(monkeypatch)
    result = MODULE.materialize(
        candidate,
        upstream,
        output,
        materialization_report,
        authority_report,
    )
    return result, report, seen, authority_report


def test_valid_candidate_is_bound_to_upstream_report_not_caller_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, upstream, seen, authority_report = _run(tmp_path, monkeypatch)
    trusted = upstream["materialization"]["candidate_jsonl_sha256"]
    assert seen == [trusted]
    assert result["upstream_handoff_report_sha256"] == upstream["report_sha256"]
    assert result["input_candidate_jsonl_sha256"] == trusted
    assert result["claim_boundary"]["training_authorized_bytes"] == 0
    assert json.loads(authority_report.read_text()) == result
    core = dict(result)
    claimed = core.pop("report_sha256")
    assert claimed == MODULE.sha256_bytes(MODULE.canonical_bytes(core))


def test_substituted_candidate_fails_against_unchanged_terminal_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row("original")
    candidate = tmp_path / "candidate.jsonl"
    upstream = tmp_path / "handoff.json"
    _write_candidate(candidate, row)
    _write_upstream(upstream, candidate, row, monkeypatch)
    _write_candidate(candidate, _candidate_row("substituted"))
    _fake_delegate(monkeypatch)
    with pytest.raises(MODULE.AuthorizationError, match="candidate SHA does not match #916"):
        MODULE.materialize(
            candidate,
            upstream,
            tmp_path / "out.jsonl",
            tmp_path / "materialization.json",
            tmp_path / "authority.json",
        )


def test_unknown_candidate_field_is_rejected_before_privacy_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row()
    row["unscanned_secret_backup"] = "person@example.test"
    candidate = tmp_path / "candidate.jsonl"
    upstream = tmp_path / "handoff.json"
    _write_candidate(candidate, row)
    _write_upstream(upstream, candidate, row, monkeypatch)
    seen = _fake_delegate(monkeypatch)
    with pytest.raises(MODULE.AuthorizationError, match="unknown keys"):
        MODULE.materialize(
            candidate,
            upstream,
            tmp_path / "out.jsonl",
            tmp_path / "materialization.json",
            tmp_path / "authority.json",
        )
    assert seen == []


def test_self_consistent_but_wrong_parent_authority_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row()
    candidate = tmp_path / "candidate.jsonl"
    upstream = tmp_path / "handoff.json"
    _write_candidate(candidate, row)
    report = _upstream_report(candidate, row)
    report["parent_authority"]["rights_report_sha256"] = "0" * 64
    core = dict(report)
    core.pop("report_sha256")
    report["report_sha256"] = _sha(MODULE.canonical_bytes(core))
    upstream.write_bytes(MODULE.canonical_bytes(report))
    monkeypatch.setattr(MODULE, "EXPECTED_UPSTREAM_REPORT_SHA256", report["report_sha256"])
    monkeypatch.setattr(MODULE, "EXPECTED_CANDIDATE_RECORDS", 1)
    monkeypatch.setattr(MODULE, "EXPECTED_ACCEPTED_SOURCE_PAYLOAD_BYTES", row["source_payload_bytes"])
    with pytest.raises(MODULE.AuthorizationError, match="parent authority drift: rights_report_sha256"):
        MODULE._validate_upstream_handoff(upstream)


def test_wrong_record_to_source_binding_fails_even_when_report_hash_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _candidate_row()
    row["record_id"] = "rada-trees:wrong"
    candidate = tmp_path / "candidate.jsonl"
    upstream = tmp_path / "handoff.json"
    _write_candidate(candidate, row)
    _write_upstream(upstream, candidate, row, monkeypatch)
    with pytest.raises(MODULE.AuthorizationError, match="record/source identity drift"):
        MODULE._validate_candidate(candidate, MODULE._validate_upstream_handoff(upstream))
