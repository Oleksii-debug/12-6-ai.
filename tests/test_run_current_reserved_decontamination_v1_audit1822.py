from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_current_reserved_decontamination_v1 as runner


def test_publish_race_preserves_concurrent_empty_destination(
    tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    original = runner._rename_directory_no_replace

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        original(source, destination)

    monkeypatch.setattr(runner, "_rename_directory_no_replace", race)
    files = {
        runner.REPORT_NAME: b"{}\n",
        runner.EVIDENCE_NAME: b"{}\n",
        runner.RECEIPT_NAME: b"{}\n",
    }

    with pytest.raises(FileExistsError):
        runner._publish_bundle(output_dir, files)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_receipt_rejects_self_resealed_unknown_top_level_field() -> None:
    report: dict[str, object] = {}
    evidence: dict[str, object] = {}
    receipt = runner.build_run_receipt(
        implementation_git_sha="a" * 40,
        input_file_sha256={
            "training_records_jsonl": "1" * 64,
            "training_handoff_json": "2" * 64,
            "evaluation_records_jsonl": "3" * 64,
            "reserved_binding_json": "4" * 64,
        },
        report=report,
        evidence=evidence,
        report_file_bytes=runner._file_bytes(report),
        evidence_file_bytes=runner._file_bytes(evidence),
    )
    receipt["unexpected_authority"] = "forged-but-self-hashed"
    body = dict(receipt)
    body.pop("receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = runner._sha256_bytes(
        runner._canonical_bytes(body)
    )

    with pytest.raises(ValueError, match="top-level key set drift"):
        runner.verify_run_receipt(receipt, report, evidence)
