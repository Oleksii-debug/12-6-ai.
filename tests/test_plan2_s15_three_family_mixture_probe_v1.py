"""S15 exact multi-family S9 mixture mechanics may not grant real authority."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_mixture_probe_v1 as probe

ROOT = Path(__file__).resolve().parents[1]


def test_real_three_family_s9_mechanics_all_15_documents(tmp_path: Path) -> None:
    result = probe.stage(ROOT, tmp_path / "mixture")
    assert result["decision"] == (
        "THREE_REAL_FAMILIES_S9_MIXTURE_MECHANICS_NOT_ADMISSION"
    )
    assert result["source_family_count"] == 3
    assert result["physical_source_count"] == 5
    assert result["selected_physical_document_count"] == 15
    assert result["source_excluded_counts"] == {}
    assert result["source_contributions"]["source"]
    assert len(result["source_contributions"]["source"]) == 5
    assert result["token_count_is_proxy_only"] is True
    assert result["real_s8_authority_issued"] is False
    assert result["physical_s3_s9_admitted"] is False
    assert result["training_corpus_authorized"] is False
    assert result["tokenizer_fit_authorized"] is False
    assert result["production_release_authorized"] is False
    assert result["terminal_done"] is False
    assert json.loads((tmp_path / "mixture" / probe.OUTPUT).read_bytes()) == result
    assert probe.stage(ROOT, tmp_path / "mixture") == result


@pytest.mark.parametrize("name", [
    "family", "missing_source", "privacy", "DATA232", "promotion",
])
def test_false_multisource_corpus_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    source = {
        "source_family_count": 3,
        "document_identity_count": 15,
        "g06_rejected_record_ids": [],
        "data232_excluded_record_count": 0,
        "data232_quarantined_source_family_count": 0,
        "training_corpus_authorized": False,
        "physical_s3_s9_multi_family_admitted": False,
    }
    if name == "family":
        source["source_family_count"] = 2
    elif name == "missing_source":
        source["document_identity_count"] = 14
    elif name == "privacy":
        source["g06_rejected_record_ids"] = ["excluded"]
    elif name == "DATA232":
        source["data232_quarantined_source_family_count"] = 1
    else:
        source["training_corpus_authorized"] = True
    monkeypatch.setattr(probe.combined, "inspect", lambda _: source)
    with pytest.raises(probe.MixProbeDenied, match="privacy/rights"):
        probe.inspect(tmp_path)


def test_resigned_mixture_receipt_tamper_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"schema_version": probe.SCHEMA,
            "training_corpus_authorized": False}
    monkeypatch.setattr(probe, "inspect", lambda _: body)
    path = tmp_path / "out"
    probe.stage(tmp_path, path)
    (path / probe.OUTPUT).write_bytes(
        b'{"training_corpus_authorized":true}\n'
    )
    with pytest.raises(probe.MixProbeDenied, match="immutable"):
        probe.stage(tmp_path, path)


def test_symlinked_publication_denied(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "real"
    dest.mkdir()
    link = tmp_path / "link"
    link.symlink_to(dest, target_is_directory=True)
    monkeypatch.setattr(probe, "inspect", lambda _: pytest.fail("unsafe read"))
    with pytest.raises(probe.MixProbeDenied, match="symlink"):
        probe.stage(tmp_path, link)
