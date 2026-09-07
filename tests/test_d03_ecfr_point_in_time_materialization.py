from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import materialize_d03_ecfr_point_in_time as materializer
import validate_d03_ecfr_point_in_time_contract as contract

CONFIG_PATH = ROOT / "configs/data/d03_ecfr_point_in_time_materialization_v1.json"


def _config() -> dict[str, object]:
    return materializer.load_contract(CONFIG_PATH)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_contract_accepts_exact_parent_and_zero_credit_boundary() -> None:
    config = _config()
    assert config["contract_identity_sha256"] == contract.EXPECTED_CONTRACT_IDENTITY
    assert config["objects"] == [
        {
            "date": "2026-08-06",
            "title": 5,
            "url": "https://www.ecfr.gov/api/versioner/v1/full/2026-08-06/title-5.xml",
        }
    ]
    assert config["claim_boundary"]["training_authorized_bytes"] == 0
    assert config["claim_boundary"]["paid_compute_used"] is False


def test_contract_rejects_reserved_title_even_with_recomputed_self_hash(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["objects"][0]["title"] = 35
    config["objects"][0]["url"] = (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-08-06/title-35.xml"
    )
    config["contract_identity_sha256"] = contract.contract_identity(config)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(materializer.MaterializationError):
        materializer.load_contract(path)


def test_redirect_must_preserve_https_host_path_and_query() -> None:
    expected = "https://www.ecfr.gov/api/versioner/v1/full/2026-08-06/title-5.xml"
    materializer._validate_final_url(
        expected,
        expected,
        ["www.ecfr.gov", "ecfr.gov"],
    )
    with pytest.raises(materializer.MaterializationError):
        materializer._validate_final_url(
            expected,
            "https://evil.example/api/versioner/v1/full/2026-08-06/title-5.xml",
            ["www.ecfr.gov", "ecfr.gov"],
        )
    with pytest.raises(materializer.MaterializationError):
        materializer._validate_final_url(
            expected,
            "https://www.ecfr.gov/api/versioner/v1/full/2026-08-06/title-6.xml",
            ["www.ecfr.gov", "ecfr.gov"],
        )
    with pytest.raises(materializer.MaterializationError):
        materializer._validate_final_url(
            expected,
            "https://www.ecfr.gov:444/api/versioner/v1/full/2026-08-06/title-5.xml",
            ["www.ecfr.gov", "ecfr.gov"],
        )


def test_double_fetch_requires_same_bytes_url_and_content_type() -> None:
    good = {
        "size_bytes": 123,
        "sha256": "a" * 64,
        "final_url": "https://www.ecfr.gov/x.xml",
        "content_type": "application/xml",
    }
    materializer._require_identical(good, copy.deepcopy(good))
    for field, value in (
        ("size_bytes", 124),
        ("sha256", "b" * 64),
        ("final_url", "https://ecfr.gov/x.xml"),
        ("content_type", "text/xml"),
    ):
        bad = copy.deepcopy(good)
        bad[field] = value
        with pytest.raises(materializer.MaterializationError):
            materializer._require_identical(good, bad)


def test_xml_text_identity_collapses_unicode_whitespace_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_text(
        "<ROOT>  Alpha\n <P>Beta\tGamma</P>  </ROOT>",
        encoding="utf-8",
    )
    observed = materializer.extract_text_identity(path)
    expected = "Alpha Beta Gamma"
    assert observed["normalized_text_sha256"] == _sha(expected)
    assert observed["normalized_text_utf8_bytes"] == len(expected.encode("utf-8"))
    assert observed["normalized_text_characters"] == len(expected)


def test_xml_doctype_and_entity_declarations_fail_closed(tmp_path: Path) -> None:
    for name, payload in (
        ("doctype.xml", '<!DOCTYPE root SYSTEM "x"><root>Alpha</root>'),
        ("entity.xml", '<!DOCTYPE root [<!ENTITY x "Alpha">]><root>&x;</root>'),
    ):
        path = tmp_path / name
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(materializer.MaterializationError):
            materializer.extract_text_identity(path)


def test_malformed_or_empty_character_data_xml_fails_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<root><x></root>", encoding="utf-8")
    with pytest.raises(materializer.MaterializationError):
        materializer.extract_text_identity(malformed)

    empty = tmp_path / "empty.xml"
    empty.write_text("<root><x /></root>", encoding="utf-8")
    with pytest.raises(materializer.MaterializationError):
        materializer.extract_text_identity(empty)


def test_materialize_uses_two_fetches_and_emits_text_free_zero_credit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    calls: list[Path] = []
    xml = b"<ROOT> Alpha <P>Beta</P> Gamma </ROOT>"
    digest = hashlib.sha256(xml).hexdigest()

    def fake_download(url: str, destination: Path, network, *, timeout: float):
        del network, timeout
        destination.write_bytes(xml)
        calls.append(destination)
        return {
            "url": url,
            "final_url": url,
            "content_type": "application/xml",
            "size_bytes": len(xml),
            "sha256": digest,
        }

    monkeypatch.setattr(materializer, "_download_once", fake_download)
    report = materializer.materialize(config, tmp_path / "work")
    materializer.verify_report(config, report)

    assert len(calls) == 2
    assert report["aggregate"]["object_count"] == 1
    assert report["aggregate"]["raw_bytes"] == len(xml)
    assert report["objects"][0]["two_byte_identical_acquisitions"] is True
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["claim_boundary"]["authorized_unique_loss_positions"] == 0
    assert report["raw_text_emitted_in_report"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert "Alpha" not in serialized
    assert "Beta" not in serialized


def test_report_verification_rejects_training_or_rights_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    xml = b"<ROOT>Alpha Beta</ROOT>"
    digest = hashlib.sha256(xml).hexdigest()

    def fake_download(url: str, destination: Path, network, *, timeout: float):
        del network, timeout
        destination.write_bytes(xml)
        return {
            "url": url,
            "final_url": url,
            "content_type": "application/xml",
            "size_bytes": len(xml),
            "sha256": digest,
        }

    monkeypatch.setattr(materializer, "_download_once", fake_download)
    report = materializer.materialize(config, tmp_path / "work")

    for field, value in (
        ("training_authorized_bytes", 1),
        ("rights_and_provenance_complete", True),
        ("paid_compute_used", True),
    ):
        mutated = copy.deepcopy(report)
        mutated["claim_boundary"][field] = value
        mutated["report_identity_sha256"] = materializer._report_identity(mutated)
        with pytest.raises(materializer.MaterializationError):
            materializer.verify_report(config, mutated)
