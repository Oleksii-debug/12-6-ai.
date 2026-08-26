from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data import cross_source_capacity_audit_v4 as v4


def _nist_row(pdf: bytes, normalized: bytes) -> dict[str, object]:
    return {
        "source_id": "en.usgov.nist.probe",
        "source_family": "en.usgov.nist.technical-series",
        "stable_origin_id": "publisher:nist:technical-series",
        "stable_object_id": "doi:probe",
        "modality": "en",
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": "terminal-probe",
        "declared_capacity_bytes": len(normalized),
        "expected_raw_bytes": len(pdf),
        "expected_raw_sha256": hashlib.sha256(pdf).hexdigest(),
        "acquisition_url": "https://example.invalid/probe.pdf",
        "origin_key": "nist:probe",
        "comparison_normalization": v4.NIST_POLICY,
        "expected_comparison_bytes": len(normalized),
        "expected_comparison_sha256": hashlib.sha256(normalized).hexdigest(),
        "pdf_start_page": 9,
        "expected_extractor_version_prefix": "pdftotext version 24.02.0",
    }


def test_nist_normalization_matches_sealed_policy_primitives() -> None:
    text = "  Header  \r\ncontact@example.org\r\n\r\n\r\nFullwidth: ＡＢＣ  \fTail\n"
    normalized = v4._normalize_nist_extracted(text)
    assert normalized.decode("utf-8") == "Header\n<EMAIL>\n\nFullwidth: ABC\nTail\n"


def test_nist_materialization_binds_pdf_and_normalized_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = b"%PDF-frozen-terminal-probe"
    extracted = "NIST body\ncontact@example.org\n\nTechnical prose\n"
    normalized = v4._normalize_nist_extracted(extracted)
    row = _nist_row(pdf, normalized)

    monkeypatch.setattr(v4, "_pdftotext_version", lambda: "pdftotext version 24.02.0")

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        Path(args[-1]).write_text(extracted, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(v4.subprocess, "run", fake_run)
    payload, evidence = v4._materialize_nist(row, pdf)

    assert payload == normalized
    assert evidence["upstream_raw_sha256"] == hashlib.sha256(pdf).hexdigest()
    assert evidence["materialized_sha256"] == hashlib.sha256(normalized).hexdigest()


def test_nist_materialization_fails_closed_on_extractor_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = b"%PDF-frozen-terminal-probe"
    normalized = v4._normalize_nist_extracted("Technical prose\n")
    row = _nist_row(pdf, normalized)
    monkeypatch.setattr(v4, "_pdftotext_version", lambda: "pdftotext version 25.06.0")

    with pytest.raises(v4.CrossSourceV4Error, match="extractor drift"):
        v4._materialize_nist(row, pdf)


def test_successor_config_adds_one_independent_english_family_and_exact_capacity() -> None:
    config = json.loads(
        Path("configs/data/next100_076_cross_source_dedup_v4.json").read_text(encoding="utf-8")
    )
    rows = config["additional_sources"]
    assert {row["source_family"] for row in rows} == {"en.usgov.nist.technical-series"}
    assert sum(row["declared_capacity_bytes"] for row in rows) == 59_358
    assert config["expected_pre_dedup_projection"] == {
        "base_declared_capacity_bytes": 243970,
        "added_nist_declared_capacity_bytes": 59358,
        "converged_declared_capacity_bytes": 303328,
        "english_declared_capacity_bytes": 144151,
        "english_independent_family_floor": 2,
        "note": "Projection only. Canonical successor report must establish actual cross-source dedup capacity and family accounting.",
    }


def test_expand_preserves_immutable_base_and_maps_nist_to_effective_text_payload() -> None:
    pdf = b"%PDF-probe"
    normalized = v4._normalize_nist_extracted("Technical prose\n")
    late = _nist_row(pdf, normalized)
    base = {
        "schema_version": v3.INVENTORY_SCHEMA,
        "sources": [
            {
                "source_id": "en.base",
                "source_family": "en.base",
                "stable_origin_id": "base-origin",
                "stable_object_id": "base-object",
                "modality": "en",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": "base",
                "declared_capacity_bytes": 3,
                "expected_raw_bytes": 3,
                "expected_raw_sha256": hashlib.sha256(b"abc").hexdigest(),
                "acquisition_url": "https://example.invalid/base",
                "origin_key": "base-origin-key",
            }
        ],
        "lineage_edges": [],
    }
    inventory = {
        "schema_version": v4.SCHEMA,
        "worker_id": "test",
        "local_free_only": True,
        "model_training_executed": False,
        "base_inventory": {
            "acquisition_url": "https://example.invalid/base.json",
            "expected_git_blob_sha1": "a" * 40,
            "authority_head_sha": "b" * 40,
        },
        "additional_sources": [late],
        "additional_lineage_edges": [],
    }

    expanded = v4._expand_inventory(inventory, base)
    assert expanded["sources"][0] == base["sources"][0]
    mapped = expanded["sources"][1]
    assert mapped["expected_raw_bytes"] == len(normalized)
    assert mapped["expected_raw_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert "comparison_normalization" not in mapped
