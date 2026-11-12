"""S15 genuine EVAL233 final-test custody and candidate DATA232 decontamination."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_real_eval233_decontamination_v1 as real

ROOT = Path(__file__).resolve().parents[1]


def test_real_frozen_final_test_custody_and_nonrelease(tmp_path: Path) -> None:
    report = real.stage(ROOT, tmp_path / "real-final")
    assert report["decision"] == (
        "REAL_EVAL233_FINAL_CUSTODY_DECONTAMINATION_CANDIDATE_ONLY"
    )
    assert report["real_final_test_record_count"] == 16
    assert report["real_final_test_custody_verified"] is True
    assert report["real_final_test_outcomes_read"] is False
    assert report["real_final_test_payload_read_for_decontamination_only"] is True
    assert report["selection_payload_scanned"] is False
    assert report["physical_training_candidate_count"] > 0
    assert report["physical_training_candidate_count"] <= 15
    assert len(report["real_eval233_resolver_identity_sha256"]) == 64
    assert len(report["data232_report_sha256"]) == 64
    assert report["physical_s3_s9_admitted"] is False
    assert report["tokenizer_fit_authorized"] is False
    assert report["training_corpus_authorized"] is False
    assert report["production_release_authorized"] is False
    assert report["terminal_done"] is False
    assert json.loads((tmp_path / "real-final" / real.OUTPUT).read_bytes()) == report
    assert real.stage(ROOT, tmp_path / "real-final") == report


@pytest.mark.parametrize("alter", [
    "source_release", "source_admission", "selection_blob", "selection_train",
    "selection_final_outcomes",
])
def test_real_final_authority_escalation_fail_closed(
    monkeypatch: pytest.MonkeyPatch, alter: str,
) -> None:
    source = {
        "source_family_count": 3,
        "document_identity_count": 15,
        "real_evaluation_custody_established": False,
        "production_release_authorized": False,
    }
    if alter == "source_release":
        source["production_release_authorized"] = True
    elif alter == "source_admission":
        source["real_evaluation_custody_established"] = True
    monkeypatch.setattr(real.physical, "inspect", lambda _: source)
    if alter not in {"source_release", "source_admission"}:
        original = real.books.read_checked

        def altered(root: Path, filename: str) -> bytes:
            raw = original(root, filename)
            if filename != real.SELECTION:
                return raw
            if alter == "selection_blob":
                return b"{}\n"
            manifest = json.loads(raw)
            if alter == "selection_train":
                manifest["usage_contract"]["may_train"] = True
            else:
                manifest["final_test_firewall"]["outcomes_read_by_eval303"] = True
            return real.books.canonical(manifest)

        monkeypatch.setattr(real.books, "read_checked", altered)
    # Use the pinned repository corpus: a temporary directory lacks EVAL-303
    # source evidence and would fail for an unrelated missing-file reason.
    with pytest.raises(real.RealFinalDenied):
        real.inspect(ROOT)


def test_readback_denies_tampered_final_custody_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"schema_version": real.SCHEMA,
              "production_release_authorized": False}
    monkeypatch.setattr(real, "inspect", lambda _: report)
    path = tmp_path / "out"
    real.stage(tmp_path, path)
    (path / real.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(real.RealFinalDenied, match="immutable"):
        real.stage(tmp_path, path)


def test_final_custody_destination_symlink_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dst = tmp_path / "dest"
    dst.mkdir()
    link = tmp_path / "link"
    link.symlink_to(dst, target_is_directory=True)
    monkeypatch.setattr(real, "inspect", lambda _: pytest.fail("read source"))
    with pytest.raises(real.RealFinalDenied, match="symlink"):
        real.stage(tmp_path, link)
