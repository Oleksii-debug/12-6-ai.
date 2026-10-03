from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.data import nbu_dedup_intake as mod

ROOT = Path(__file__).resolve().parents[1]


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _row(index: int, text: str) -> dict[str, object]:
    payload = text.encode()
    digest = hashlib.sha256(payload).hexdigest()
    pdf_sha = hashlib.sha256(f"pdf-{index}".encode()).hexdigest()
    document_url = f"https://bank.gov.ua/ua/legislation/Resolution_202609{index:02d}_x"
    pdf_url = f"https://bank.gov.ua/admin_uploads/law/{index}.pdf"
    provenance = {
        "document_url": document_url,
        "pdf_url": pdf_url,
        "pdf_sha256": pdf_sha,
        "text_sha256": digest,
    }
    record_id = hashlib.sha256(_canonical_line(provenance)).hexdigest()
    return {
        "schema": "12-6.d03-ua-nbu-pdftotext-record.v1",
        "record_id": record_id,
        "source_id": mod.SOURCE_ID,
        "family_id": mod.SOURCE_FAMILY,
        "document_url": document_url,
        "pdf_url": pdf_url,
        "pdf_sha256": pdf_sha,
        "text_sha256": digest,
        "text_bytes": len(payload),
        "extractor_backend": mod.EXTRACTOR_BACKEND,
        "extractor_version": mod.EXTRACTOR_VERSION,
        "extractor_arguments": list(mod.EXTRACTOR_ARGUMENTS),
        "text": text,
    }


def _evidence(rows: list[dict[str, object]], raw: bytes) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "12-6.d03-ua-nbu-pdftotext-materialization-evidence.v1",
        "status": "TEXT_MATERIALIZED_ZERO_CREDIT",
        "contract_identity_sha256": "1" * 64,
        "discovery_evidence_identity_sha256": "2" * 64,
        "pdf_pin_evidence_identity_sha256": "3" * 64,
        "source_id": mod.SOURCE_ID,
        "family_id": mod.SOURCE_FAMILY,
        "extractor_backend": mod.EXTRACTOR_BACKEND,
        "extractor_version": mod.EXTRACTOR_VERSION,
        "extractor_arguments": list(mod.EXTRACTOR_ARGUMENTS),
        "materialized_records": len(rows),
        "observed_text_bytes": sum(int(row["text_bytes"]) for row in rows),
        "text_artifact_bytes": len(raw),
        "text_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "records": [{k: v for k, v in row.items() if k != "text"} for row in rows],
        "raw_text_emitted_in_evidence": False,
        "text_artifact_is_training_authority": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    value["evidence_identity_sha256"] = mod._evidence_identity(value)
    return value


def _patch_authority(monkeypatch, rows: list[dict[str, object]], raw: bytes, evidence):
    monkeypatch.setattr(mod, "CANDIDATE_RECORDS", len(rows))
    monkeypatch.setattr(
        mod,
        "CANDIDATE_TEXT_BYTES",
        sum(int(row["text_bytes"]) for row in rows),
    )
    monkeypatch.setattr(mod, "CANDIDATE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        mod,
        "EVIDENCE_IDENTITY_SHA256",
        evidence["evidence_identity_sha256"],
    )


def _write_fixture(tmp_path: Path, monkeypatch):
    rows = [_row(1, "Постанова один."), _row(2, "Постанова два.")]
    raw = b"".join(_canonical_line(row) for row in rows)
    evidence = _evidence(rows, raw)
    _patch_authority(monkeypatch, rows, raw, evidence)
    candidate = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "evidence.json"
    candidate.write_bytes(raw)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, candidate, evidence_path


def test_projection_source_contains_no_escaped_mechanical_newlines() -> None:
    source = (ROOT / "src" / "twelve_six" / "data" / "nbu_dedup_intake.py").read_text(
        encoding="utf-8"
    )
    assert "receipt-visible.\\n    projection_identity" not in source


def test_projects_audited_nbu_records_one_for_one(tmp_path, monkeypatch) -> None:
    rows, candidate, evidence = _write_fixture(tmp_path, monkeypatch)
    result = mod.validate_and_project_nbu(candidate, evidence)

    assert result.sources is not None
    assert result.payloads is not None
    assert len(result.sources) == len(rows)
    assert len(result.payloads) == len(rows)
    first = result.sources[0]
    assert first["source_family"] == mod.SOURCE_FAMILY
    assert first["modality"] == "uk"
    assert first["stable_object_id"] == f"sha256:{rows[0]['text_sha256']}"
    assert first["declared_capacity_bytes"] == rows[0]["text_bytes"]
    assert result.payloads[first["source_id"]] == str(rows[0]["text"]).encode()
    assert result.receipt["execution_gate"]["canonical_global_dedup_executed"] is False
    assert result.receipt["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert result.receipt["projection"]["matcher_source_inventory_identity_sha256"] == mod._sha256(
        mod._canonical(list(result.sources))
    )


@pytest.mark.parametrize("bad", [0, 1, None, "false"])
def test_retain_payloads_requires_exact_bool(tmp_path, monkeypatch, bad) -> None:
    _, candidate, evidence = _write_fixture(tmp_path, monkeypatch)
    with pytest.raises(mod.NbuDedupIntakeError, match="retain_payloads must be exact bool"):
        mod.validate_and_project_nbu(candidate, evidence, retain_payloads=bad)


def test_receipt_is_text_free_and_payloads_can_be_dropped(tmp_path, monkeypatch) -> None:
    secret = "НЕ ВИВОДИТИ ТЕКСТ У RECEIPT"
    rows = [_row(1, secret)]
    raw = b"".join(_canonical_line(row) for row in rows)
    evidence = _evidence(rows, raw)
    _patch_authority(monkeypatch, rows, raw, evidence)
    candidate = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "evidence.json"
    candidate.write_bytes(raw)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")

    result = mod.validate_and_project_nbu(candidate, evidence_path, retain_payloads=False)
    assert result.sources is None
    assert result.payloads is None
    assert secret not in json.dumps(result.receipt, ensure_ascii=False)


def test_candidate_byte_drift_fails_before_projection(tmp_path, monkeypatch) -> None:
    _, candidate, evidence = _write_fixture(tmp_path, monkeypatch)
    candidate.write_bytes(candidate.read_bytes() + b" ")
    with pytest.raises(mod.NbuDedupIntakeError, match="candidate hash drift"):
        mod.validate_and_project_nbu(candidate, evidence)


def test_manifest_candidate_drift_is_rejected(tmp_path, monkeypatch) -> None:
    _rows, candidate, evidence_path = _write_fixture(tmp_path, monkeypatch)
    evidence = json.loads(evidence_path.read_text())
    evidence["records"][0]["pdf_sha256"] = "f" * 64
    evidence["evidence_identity_sha256"] = mod._evidence_identity(evidence)
    monkeypatch.setattr(
        mod,
        "EVIDENCE_IDENTITY_SHA256",
        evidence["evidence_identity_sha256"],
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(mod.NbuDedupIntakeError, match="manifest/candidate drift"):
        mod.validate_and_project_nbu(candidate, evidence_path)


def test_record_provenance_identity_drift_is_rejected(tmp_path, monkeypatch) -> None:
    rows, candidate, evidence_path = _write_fixture(tmp_path, monkeypatch)
    rows[0]["record_id"] = "f" * 64
    raw = b"".join(_canonical_line(row) for row in rows)
    evidence = _evidence(rows, raw)
    _patch_authority(monkeypatch, rows, raw, evidence)
    candidate.write_bytes(raw)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(mod.NbuDedupIntakeError, match="record provenance identity drift"):
        mod.validate_and_project_nbu(candidate, evidence_path)


def test_duplicate_candidate_record_id_is_rejected(tmp_path, monkeypatch) -> None:
    rows = [_row(1, "Один."), _row(2, "Два.")]
    rows[1] = dict(rows[0])
    raw = b"".join(_canonical_line(row) for row in rows)
    evidence = _evidence(rows, raw)
    _patch_authority(monkeypatch, rows, raw, evidence)
    candidate = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "evidence.json"
    candidate.write_bytes(raw)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(mod.NbuDedupIntakeError, match="duplicate NBU record id"):
        mod.validate_and_project_nbu(candidate, evidence_path)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("canonical_capacity_credit_bytes", 1, "truth boundary drift"),
        ("training_authorized_bytes", 1, "truth boundary drift"),
        ("tokenizer_fit_authorized", True, "truth boundary drift"),
        ("model_training_executed", True, "truth boundary drift"),
        ("final_test_accessed", True, "truth boundary drift"),
        ("paid_compute_used", True, "truth boundary drift"),
    ],
)
def test_evidence_truth_promotion_is_rejected(
    tmp_path, monkeypatch, field: str, bad: object, message: str
) -> None:
    _, candidate, evidence_path = _write_fixture(tmp_path, monkeypatch)
    evidence = json.loads(evidence_path.read_text())
    evidence[field] = bad
    evidence["evidence_identity_sha256"] = mod._evidence_identity(evidence)
    monkeypatch.setattr(
        mod,
        "EVIDENCE_IDENTITY_SHA256",
        evidence["evidence_identity_sha256"],
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(mod.NbuDedupIntakeError, match=message):
        mod.validate_and_project_nbu(candidate, evidence_path)


def test_evidence_bool_cannot_alias_required_zero(tmp_path, monkeypatch) -> None:
    _, candidate, evidence_path = _write_fixture(tmp_path, monkeypatch)
    evidence = json.loads(evidence_path.read_text())
    evidence["optimizer_updates"] = False
    evidence["evidence_identity_sha256"] = mod._evidence_identity(evidence)
    monkeypatch.setattr(
        mod,
        "EVIDENCE_IDENTITY_SHA256",
        evidence["evidence_identity_sha256"],
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(mod.NbuDedupIntakeError, match="truth type drift"):
        mod.validate_and_project_nbu(candidate, evidence_path)


def test_nonfinite_json_is_rejected(tmp_path, monkeypatch) -> None:
    _, candidate, evidence_path = _write_fixture(tmp_path, monkeypatch)
    raw = evidence_path.read_text().replace('"optimizer_updates": 0', '"optimizer_updates": 1e400')
    evidence_path.write_text(raw, encoding="utf-8")
    with pytest.raises(mod.NbuDedupIntakeError, match="non-finite JSON number"):
        mod.validate_and_project_nbu(candidate, evidence_path)


def test_projection_receipt_self_hash_is_exact(tmp_path, monkeypatch) -> None:
    _, candidate, evidence = _write_fixture(tmp_path, monkeypatch)
    result = mod.validate_and_project_nbu(candidate, evidence)
    receipt = deepcopy(result.receipt)
    identity = receipt.pop("receipt_identity_sha256")
    assert identity == mod._sha256(mod._canonical(receipt))
