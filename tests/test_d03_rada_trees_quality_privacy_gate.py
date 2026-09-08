from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_d03_rada_trees_quality_privacy_gate.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_quality_privacy", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakePrivacyResult:
    def __init__(self, action: str) -> None:
        self.action = action

    def evidence(self) -> dict[str, Any]:
        return {
            "input_sha256": "a" * 64,
            "input_bytes": 12,
            "action": self.action,
            "detector_counts": {},
        }


def record(record_id: str = "r1", text: str = "Український парламентський текст.") -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source_family": MODULE.EXPECTED_FAMILY,
        "source_dataset": MODULE.EXPECTED_DATASET,
        "source_revision": MODULE.EXPECTED_REVISION,
        "source_archive": "rada_xtag_texts.7z",
        "source_path": f"texts/{record_id}.txt",
        "session_date": "2020-01-01",
        "source_payload_sha256": "b" * 64,
        "source_payload_bytes": len(text.encode("utf-8")),
        "decoded_encoding": "utf-8",
        "decoded_text_utf8_sha256": "c" * 64,
        "decoded_text_utf8_bytes": len(text.encode("utf-8")),
        "rights_scope_status": "RIGHTS_SCOPE_SUPPORTED_DATED_PARLIAMENT_TRANSCRIPT_CANDIDATE",
        "attribution_required": True,
        "language_quality_privacy_complete": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(MODULE.canonical_bytes(row) for row in rows))


def fake_mechanics(*, quality_status: str = "RETAIN_ALL", privacy_action: str = "ALLOW"):
    def quality(record_id: str, text: str, mode: str) -> dict[str, Any]:
        assert record_id
        assert text
        assert mode == "uk"
        return {
            "record_id": record_id,
            "status": quality_status,
            "retained_utf8_bytes": len(text.encode("utf-8")) if quality_status == "RETAIN_ALL" else 0,
            "rejected_utf8_bytes": 0 if quality_status == "RETAIN_ALL" else len(text.encode("utf-8")),
        }

    def privacy(text: str) -> FakePrivacyResult:
        assert text
        return FakePrivacyResult(privacy_action)

    def policy() -> dict[str, Any]:
        return {"policy_sha256": "d" * 64}

    return quality, privacy, policy


def test_gate_retains_only_fully_allowed_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "clean.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(candidate, [record()])
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: fake_mechanics())

    result = MODULE.run_gate(candidate, output, report)

    assert result["input_records"] == 1
    assert result["output_records"] == 1
    assert result["claim_boundary"]["training_authorized_bytes"] == 0
    assert result["claim_boundary"]["unique_causal_loss_positions_authorized"] == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["language_quality_privacy_complete"] is True
    assert row["quality_gate_status"] == "RETAIN_ALL"
    assert row["privacy_gate_action"] == "ALLOW"
    assert row["training_eligible"] is False
    assert row["evaluation_eligible"] is False


def test_partial_quality_is_held_until_window_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "clean.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(candidate, [record()])
    monkeypatch.setattr(
        MODULE,
        "_load_mechanics",
        lambda: fake_mechanics(quality_status="RETAIN_PARTIAL"),
    )

    result = MODULE.run_gate(candidate, output, report)

    assert result["output_records"] == 0
    assert result["quality"]["statuses"] == {"RETAIN_PARTIAL": 1}
    assert result["quality"]["partial_records_are_held_not_materialized"] is True
    assert output.read_bytes() == b""


def test_privacy_redact_is_held_without_inventing_redaction_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "clean.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(candidate, [record()])
    monkeypatch.setattr(
        MODULE,
        "_load_mechanics",
        lambda: fake_mechanics(privacy_action="REDACT"),
    )

    result = MODULE.run_gate(candidate, output, report)

    assert result["output_records"] == 0
    assert result["privacy"]["actions"] == {"REDACT": 1}
    assert result["privacy"]["blocked_records"] == 1
    assert result["privacy"]["redact_records_held_not_materialized"] == 1


def test_input_with_premature_training_credit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "clean.jsonl"
    report = tmp_path / "report.json"
    row = record()
    row["training_eligible"] = True
    write_jsonl(candidate, [row])
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: fake_mechanics())

    with pytest.raises(MODULE.GateError, match="premature training credit"):
        MODULE.run_gate(candidate, output, report)


def test_missing_incumbent_mechanics_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = MODULE.importlib.import_module

    def missing(name: str) -> ModuleType:
        if name in {MODULE.QUALITY_MODULE, MODULE.PRIVACY_MODULE}:
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(MODULE.importlib, "import_module", missing)
    with pytest.raises(MODULE.GateError, match="not composed"):
        MODULE._load_mechanics()
