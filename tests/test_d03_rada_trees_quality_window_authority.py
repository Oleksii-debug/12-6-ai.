from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_rada_trees_quality_windows_authoritative.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_authority", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(text: str = "абвгд") -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "record_id": "r1",
        "source_family": MODULE.base.EXPECTED_FAMILY,
        "source_dataset": MODULE.base.EXPECTED_DATASET,
        "source_revision": MODULE.base.EXPECTED_REVISION,
        "source_archive": MODULE.base.EXPECTED_ARCHIVE,
        "source_path": "texts/2020-01-01.txt",
        "session_date": "2020-01-01",
        "source_payload_sha256": "b" * 64,
        "source_payload_bytes": len(raw),
        "decoded_encoding": "utf-8",
        "decoded_text_utf8_sha256": sha(text),
        "decoded_text_utf8_bytes": len(raw),
        "rights_scope_status": MODULE.base.EXPECTED_RIGHTS_STATUS,
        "attribution_required": True,
        "language_quality_privacy_complete": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def write_candidate(path: Path, row: dict[str, Any]) -> None:
    path.write_bytes(MODULE.base.canonical_bytes(row))


def upstream_report(candidate: Path, source_bytes: int) -> dict[str, Any]:
    core = {
        "schema_version": MODULE.UPSTREAM_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset": MODULE.base.EXPECTED_DATASET,
            "revision": MODULE.base.EXPECTED_REVISION,
            "archive": MODULE.base.EXPECTED_ARCHIVE,
            "archive_bytes": MODULE.EXPECTED_ARCHIVE_BYTES,
            "archive_sha256": MODULE.EXPECTED_ARCHIVE_SHA256,
            "family": MODULE.base.EXPECTED_FAMILY,
        },
        "parent_authority": {
            "full_scan_evidence_identity_sha256": MODULE.EXPECTED_FULL_SCAN_EVIDENCE_ID,
            "rights_config_sha256": MODULE.EXPECTED_RIGHTS_CONFIG_SHA256,
            "rights_report_sha256": MODULE.EXPECTED_RIGHTS_REPORT_SHA256,
            "accepted_path_inventory_sha256": MODULE.EXPECTED_ACCEPTED_INVENTORY_SHA256,
            "held_path_inventory_sha256": MODULE.EXPECTED_HELD_INVENTORY_SHA256,
        },
        "materialization": {
            "candidate_jsonl_sha256": file_sha(candidate),
            "candidate_jsonl_bytes": candidate.stat().st_size,
            "candidate_records": 1,
            "accepted_source_payload_bytes": source_bytes,
            "decoded_text_utf8_bytes": source_bytes,
            "decoded_encoding_counts": {"utf-8": 1},
            "year_record_counts": {"2020": 1},
            "held_records_not_emitted": 1,
            "held_source_payload_bytes_not_emitted": 84_781,
            "source_native_document_boundaries_preserved": True,
            "record_text_persisted_in_report": False,
        },
        "rights": {
            "attribution_required": True,
            "project_rights_scope_status": "SUPPORTED_WITH_ATTRIBUTION_AND_SCOPE",
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
        "required_downstream_gates": ["LANGUAGE_QUALITY_PRIVACY"],
        "safe_result": MODULE.UPSTREAM_SAFE_RESULT,
    }
    return {**core, "report_sha256": MODULE.base.sha256_bytes(MODULE.base.canonical_bytes(core))}


class FakePrivacyResult:
    def __init__(self, text: str) -> None:
        raw = text.encode("utf-8")
        self._evidence = {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_bytes": len(raw),
            "action": "ALLOW",
            "detector_counts": {},
        }

    def evidence(self) -> dict[str, Any]:
        return dict(self._evidence)


def fake_quality(record_id: str, text: str, mode: str) -> dict[str, Any]:
    assert mode == "uk" and len(text) == 5
    spans = [(0, 0, 2, True), (1, 2, 3, False), (2, 3, 5, True)]
    windows = []
    retained = rejected = 0
    for index, start, end, accepted in spans:
        payload = text[start:end]
        nbytes = len(payload.encode("utf-8"))
        retained += nbytes if accepted else 0
        rejected += 0 if accepted else nbytes
        windows.append(
            {
                "index": index,
                "start_char": start,
                "end_char": end,
                "utf8_bytes": nbytes,
                "authoritative": True,
                "decision": {"accepted": accepted},
            }
        )
    return {
        "record_id": record_id,
        "mode": mode,
        "authoritative_unit": "BOUNDED_NATURAL_LANGUAGE_WINDOW",
        "family_eviction_authority": False,
        "status": "RETAIN_PARTIAL",
        "retained_utf8_bytes": retained,
        "rejected_utf8_bytes": rejected,
        "windows": windows,
    }


def fake_mechanics() -> tuple[Any, ...]:
    return (
        fake_quality,
        lambda: {"policy_sha256": "e" * 64},
        lambda text: FakePrivacyResult(text),
        lambda: {"policy_sha256": "d" * 64},
    )


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    candidate = tmp_path / "candidate.jsonl"
    row = record()
    write_candidate(candidate, row)
    source_bytes = int(row["source_payload_bytes"])
    monkeypatch.setattr(MODULE, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(MODULE, "EXPECTED_SOURCE_BYTES", source_bytes)
    monkeypatch.setattr(MODULE.base, "_load_mechanics", fake_mechanics)
    value = upstream_report(candidate, source_bytes)
    report = tmp_path / "upstream.json"
    report.write_bytes(MODULE.base.canonical_bytes(value))
    return candidate, report, str(value["report_sha256"])


def test_authoritative_materialization_binds_upstream_and_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, upstream, expected_upstream_sha = prepare(tmp_path, monkeypatch)
    output = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"
    report = MODULE.materialize_authoritative(
        candidate,
        upstream,
        output,
        report_path,
        expected_upstream_report_sha256=expected_upstream_sha,
    )
    assert report["schema_version"] == MODULE.SCHEMA
    assert report["authority_binding"]["upstream_handoff_report_sha256"] == expected_upstream_sha
    assert report["authority_binding"]["candidate_jsonl_sha256"] == file_sha(candidate)
    assert report["authority_binding"]["candidate_exact_keyset_enforced"] is True
    assert (
        report["authority_binding"]["privacy_filter_v3_git_blob_sha"]
        == MODULE.EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA
    )
    assert report["claim_boundary"]["upstream_handoff_authority_bound"] is True
    assert report["claim_boundary"]["candidate_schema_exact"] is True
    assert report["claim_boundary"]["canonical_privacy_repair_bound"] is True
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["safe_result"] == "RADA_TREES_QUALITY_WINDOWS_AUTHORITY_BOUND_ZERO_CREDIT"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["quality_privacy_complete"] is True for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["evaluation_eligible"] is False for row in rows)
    assert "raw_secret" not in output.read_text(encoding="utf-8")


def test_candidate_rehash_cannot_self_authorize_against_bound_upstream_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, upstream, expected_upstream_sha = prepare(tmp_path, monkeypatch)
    changed = record("абвге")
    write_candidate(candidate, changed)
    assert file_sha(candidate) != json.loads(upstream.read_text())["materialization"]["candidate_jsonl_sha256"]
    with pytest.raises(MODULE.AuthorityBindingError, match="does not match bound upstream"):
        MODULE.materialize_authoritative(
            candidate,
            upstream,
            tmp_path / "out.jsonl",
            tmp_path / "report.json",
            expected_upstream_report_sha256=expected_upstream_sha,
        )


def test_upstream_report_identity_is_externally_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, upstream, _ = prepare(tmp_path, monkeypatch)
    with pytest.raises(MODULE.AuthorityBindingError, match="report identity mismatch"):
        MODULE.materialize_authoritative(
            candidate,
            upstream,
            tmp_path / "out.jsonl",
            tmp_path / "report.json",
            expected_upstream_report_sha256="0" * 64,
        )


def test_unknown_candidate_payload_field_fails_even_with_matching_upstream_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    row = record()
    row["raw_secret"] = "should-never-survive-unscanned"
    write_candidate(candidate, row)
    source_bytes = int(row["source_payload_bytes"])
    monkeypatch.setattr(MODULE, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(MODULE, "EXPECTED_SOURCE_BYTES", source_bytes)
    monkeypatch.setattr(MODULE.base, "_load_mechanics", fake_mechanics)
    value = upstream_report(candidate, source_bytes)
    upstream = tmp_path / "upstream.json"
    upstream.write_bytes(MODULE.base.canonical_bytes(value))
    with pytest.raises(MODULE.AuthorityBindingError, match="schema/keyset drift"):
        MODULE.materialize_authoritative(
            candidate,
            upstream,
            tmp_path / "out.jsonl",
            tmp_path / "report.json",
            expected_upstream_report_sha256=str(value["report_sha256"]),
        )


def test_privacy_repair_implementation_drift_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, upstream, expected_upstream_sha = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(MODULE, "EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA", "0" * 40)
    output = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    with pytest.raises(MODULE.AuthorityBindingError, match="privacy repair implementation drift"):
        MODULE.materialize_authoritative(
            candidate,
            upstream,
            output,
            report,
            expected_upstream_report_sha256=expected_upstream_sha,
        )
    assert not output.exists()
    assert not report.exists()
