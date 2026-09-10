from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rada_convergence",
    ROOT / "tools/converge_d03_rada_trees_language_quality_privacy.py",
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(mod.canonical_line(value))


def write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join(mod.canonical_line(row) for row in rows)
    path.write_bytes(payload)
    return payload


def self_hashed(core: dict) -> dict:
    return {**core, "report_sha256": mod.sha256_bytes(mod.canonical_bytes(core))}


def setup_authorities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = [
        {
            "record_id": "r1",
            "source_dataset": mod.EXPECTED_DATASET,
            "source_revision": mod.EXPECTED_REVISION,
            "source_family": mod.EXPECTED_FAMILY,
            "source_path": "texts/2020-01-01__a.txt",
            "session_date": "2020-01-01",
            "source_payload_sha256": "a" * 64,
            "source_payload_bytes": 10,
            "decoded_text_utf8_sha256": "b" * 64,
            "quality_privacy_complete": False,
            "language_quality_privacy_complete": False,
            "training_eligible": False,
            "evaluation_eligible": False,
        },
        {
            "record_id": "r2",
            "source_dataset": mod.EXPECTED_DATASET,
            "source_revision": mod.EXPECTED_REVISION,
            "source_family": mod.EXPECTED_FAMILY,
            "source_path": "texts/2020-01-02__b.txt",
            "session_date": "2020-01-02",
            "source_payload_sha256": "c" * 64,
            "source_payload_bytes": 20,
            "decoded_text_utf8_sha256": "d" * 64,
            "quality_privacy_complete": False,
            "language_quality_privacy_complete": False,
            "training_eligible": False,
            "evaluation_eligible": False,
        },
    ]
    candidate = tmp_path / "candidate.jsonl"
    candidate_raw = write_jsonl(candidate, rows)
    inventory, total = mod.rights_inventory(rows)
    monkeypatch.setattr(mod, "EXPECTED_RECORDS", 2)
    monkeypatch.setattr(mod, "EXPECTED_SOURCE_BYTES", total)
    monkeypatch.setattr(mod, "EXPECTED_RIGHTS_INVENTORY", inventory)
    monkeypatch.setattr(mod, "EXPECTED_LANGUAGE_DECISION_INVENTORY", "e" * 64)

    language_core = {
        "schema_version": "12-6.d03-rada-trees-language-gate-report.v1",
        "source": {"dataset": mod.EXPECTED_DATASET, "dataset_revision": mod.EXPECTED_REVISION},
        "parent_provenance_rights": {"accepted_path_inventory_sha256": inventory},
        "language_result": {
            "language_pass_members": 2,
            "language_reject_members": 0,
            "language_pass_bytes": total,
            "language_decision_inventory_sha256": "e" * 64,
        },
    }
    language = self_hashed(language_core)
    monkeypatch.setattr(mod, "EXPECTED_LANGUAGE_REPORT", language["report_sha256"])
    language_path = tmp_path / "language.json"
    write_json(language_path, language)

    qp_rows = []
    for row in rows[:1]:
        out = dict(row)
        out["quality_privacy_complete"] = True
        qp_rows.append(out)
    qp_path = tmp_path / "qp.jsonl"
    qp_raw = write_jsonl(qp_path, qp_rows)
    qp_core = {
        "schema_version": "12-6.d03-rada-trees-quality-privacy-gate.v2",
        "input_candidate_jsonl_sha256": mod.sha256_bytes(candidate_raw),
        "input_records": 2,
        "output_records": 1,
        "output_jsonl_sha256": mod.sha256_bytes(qp_raw),
        "claim_boundary": {"language_authority_bound": False, "training_authorized_bytes": 0},
    }
    qp_report = self_hashed(qp_core)
    qp_report_path = tmp_path / "qp-report.json"
    write_json(qp_report_path, qp_report)
    return rows, candidate, language_path, qp_path, qp_report_path


def test_convergence_binds_authorities_and_keeps_zero_credit(tmp_path, monkeypatch):
    _, candidate, language, qp, qp_report = setup_authorities(tmp_path, monkeypatch)
    out = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    result = mod.converge(candidate, qp, qp_report, language, out, report)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["language_quality_privacy_complete"] is True
    assert row["training_eligible"] is False
    assert row["evaluation_eligible"] is False
    assert result["claim_boundary"]["training_authorized_bytes"] == 0
    assert result["claim_boundary"]["global_dedup_complete"] is False


def test_convergence_rejects_quality_privacy_input_substitution(tmp_path, monkeypatch):
    _, candidate, language, qp, qp_report_path = setup_authorities(tmp_path, monkeypatch)
    report = json.loads(qp_report_path.read_text(encoding="utf-8"))
    report["input_candidate_jsonl_sha256"] = "0" * 64
    core = dict(report)
    core.pop("report_sha256")
    report["report_sha256"] = mod.sha256_bytes(mod.canonical_bytes(core))
    write_json(qp_report_path, report)
    with pytest.raises(mod.ConvergenceError, match="input identity mismatch"):
        mod.converge(candidate, qp, qp_report_path, language, tmp_path / "o", tmp_path / "r")


def test_convergence_rejects_source_identity_mutation(tmp_path, monkeypatch):
    _, candidate, language, qp_path, qp_report_path = setup_authorities(tmp_path, monkeypatch)
    row = json.loads(qp_path.read_text(encoding="utf-8"))
    row["source_payload_sha256"] = "f" * 64
    qp_raw = write_jsonl(qp_path, [row])
    report = json.loads(qp_report_path.read_text(encoding="utf-8"))
    report["output_jsonl_sha256"] = mod.sha256_bytes(qp_raw)
    core = dict(report)
    core.pop("report_sha256")
    report["report_sha256"] = mod.sha256_bytes(mod.canonical_bytes(core))
    write_json(qp_report_path, report)
    with pytest.raises(mod.ConvergenceError, match="source identity mutation"):
        mod.converge(candidate, qp_path, qp_report_path, language, tmp_path / "o", tmp_path / "r")


def test_convergence_rejects_partial_language_authority(tmp_path, monkeypatch):
    _, candidate, language_path, qp, qp_report = setup_authorities(tmp_path, monkeypatch)
    report = json.loads(language_path.read_text(encoding="utf-8"))
    report["language_result"]["language_reject_members"] = 1
    core = dict(report)
    core.pop("report_sha256")
    report["report_sha256"] = mod.sha256_bytes(mod.canonical_bytes(core))
    monkeypatch.setattr(mod, "EXPECTED_LANGUAGE_REPORT", report["report_sha256"])
    write_json(language_path, report)
    with pytest.raises(mod.ConvergenceError, match="language rejects present"):
        mod.converge(candidate, qp, qp_report, language_path, tmp_path / "o", tmp_path / "r")
