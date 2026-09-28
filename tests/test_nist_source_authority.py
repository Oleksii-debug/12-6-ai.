from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.data import nist_source_authority as nist


def test_current_main_nist_authority_is_zero_credit_and_fail_closed():
    result = nist.validate_nist_source_authority()
    assert result["family_id"] == "en.usgov.nist.technical-series"
    assert result["publication_count"] == 3
    assert result["normalized_source_bytes"] == 59358
    assert result["raw_source_bytes"] == 2620454
    assert result["canonical_capacity_credit_bytes"] == 0
    assert result["authorized_optimized_target_exposure"] == 0
    assert result["tokenizer_fit_authorized"] is False
    assert result["training_executed"] is False
    assert result["learned_weights_created"] is False
    assert result["final_test_outcomes_read"] is False
    assert result["paid_compute_used"] is False
    assert result["foreign_pretrained_weights"] is False
    assert result["current_corpus_external_llm_free_claimed_by_this_authority"] is False
    assert result["downstream_global_dedup_required"] is True
    assert result["evaluation_separately_admitted"] is False


def _mutate_json(tmp_path: Path, monkeypatch, mutate):
    data = json.loads(nist.SEAL_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(nist, "SEAL_PATH", path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("terminal_payload_sha256", "0" * 64),
        lambda d: d["family"].__setitem__("family_id", "other"),
        lambda d: d["family"].__setitem__("publication_count", True),
        lambda d: d["admit"][0].__setitem__("raw_bytes", False),
        lambda d: d["rights"].__setitem__("evaluation", "ALLOWED"),
        lambda d: d.__setitem__("corpus_integration", "INTEGRATED"),
    ],
)
def test_tampered_terminal_authority_is_rejected(tmp_path, monkeypatch, mutate):
    _mutate_json(tmp_path, monkeypatch, mutate)
    with pytest.raises(nist.NistAuthorityError):
        nist.validate_nist_source_authority()


def test_duplicate_json_key_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "seal.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    monkeypatch.setattr(nist, "SEAL_PATH", path)
    with pytest.raises(nist.NistAuthorityError, match="duplicate JSON key"):
        nist.validate_nist_source_authority()


def test_nonfinite_json_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "seal.json"
    path.write_text('{"schema_version":NaN}', encoding="utf-8")
    monkeypatch.setattr(nist, "SEAL_PATH", path)
    with pytest.raises(nist.NistAuthorityError, match="non-finite"):
        nist.validate_nist_source_authority()


def test_overflow_nonfinite_json_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "seal.json"
    path.write_text('{"schema_version":1e400}', encoding="utf-8")
    monkeypatch.setattr(nist, "SEAL_PATH", path)
    with pytest.raises(nist.NistAuthorityError, match="non-finite"):
        nist.validate_nist_source_authority()
