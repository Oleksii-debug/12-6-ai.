"""S15 D03 historical raw snapshots: real-normalization and fail-closed tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_d03_physical_sources_v1 as d03
from tools import plan2_public_domain_books_v1 as books

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_d03_physical_12_source_rebuild(tmp_path: Path) -> None:
    result = d03.stage(ROOT, tmp_path / "d03")
    assert result["decision"] == "D03_PHYSICAL_UK_SOURCE_CANDIDATE_NOT_S3_S9_RELEASE"
    assert result["physical_family_count"] == 2
    assert result["physical_record_count"] == 12
    assert result["normalized_source_bytes"] == 48675
    assert result["physical_source_families"] == sorted(d03.FAMILIES.values())
    assert len({r["normalized_sha256"] for r in result["records"]}) == 12
    assert len({r["source_family"] for r in result["records"]}) == 2
    assert isinstance(result["g06_rejected_record_ids"], list)
    assert set(result["g06_rejected_record_ids"]).issubset(
        {item["record_id"] for item in result["records"]}
    )
    assert result["final_test_outcomes_read"] is False
    assert result["production_release_authorized"] is False
    assert result["tokenizer_fit_authorized"] is False
    assert result["training_corpus_authorized"] is False
    assert result["terminal_done"] is False
    assert json.loads((tmp_path / "d03" / d03.OUTPUT).read_bytes()) == result
    assert d03.stage(ROOT, tmp_path / "d03") == result


def test_all_physical_normalizers_reproduce_sealed_d03_sha() -> None:
    config = json.loads(books.read_checked(ROOT, d03.CONFIG))
    assert len(config["objects"]) == 12
    checks = 0
    for source in config["objects"]:
        raw = books.read_checked(ROOT, source["physical_path"])
        actual = (d03.normalize_php(raw)
                  if source["family"] == "php"
                  else d03.normalize_rust(raw))
        assert books.sha(actual) == source["normalized_expected_sha256"]
        assert len(actual) == source["normalized_expected_bytes"]
        checks += 1
    assert checks == 12


@pytest.mark.parametrize("alter", ["source", "license", "seal", "rights"])
def test_physical_d03_identity_or_permission_tampering_denied(
    monkeypatch: pytest.MonkeyPatch, alter: str,
) -> None:
    original = d03.books.read_checked

    def changed(root: Path, path: str) -> bytes:
        if alter == "source" and path.endswith("language/basic-syntax.xml"):
            return b"forged"
        if alter == "license" and path.endswith("licenses/rust/LICENSE-MIT"):
            return b"forged"
        if alter == "seal" and path == d03.SEALS:
            return b'{"source": "forged"}\n'
        if alter == "rights" and path == d03.CONFIG:
            config = json.loads(original(root, path))
            config["training_corpus_authorized"] = True
            return books.canonical(config)
        return original(root, path)

    monkeypatch.setattr(d03.books, "read_checked", changed)
    with pytest.raises(d03.D03SourceDenied):
        d03.inspect(ROOT)


def test_d03_source_publication_tamper_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    receipt = {"decision": "D03_PHYSICAL_UK_SOURCE_CANDIDATE_NOT_S3_S9_RELEASE",
               "production_release_authorized": False}
    monkeypatch.setattr(d03, "inspect", lambda _root: receipt)
    d03.stage(tmp_path, out)
    (out / d03.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(d03.D03SourceDenied, match="immutable"):
        d03.stage(tmp_path, out)


def test_d03_source_symlink_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "source"
    out.mkdir()
    link = tmp_path / "link"
    link.symlink_to(out, target_is_directory=True)
    monkeypatch.setattr(d03, "inspect", lambda _: pytest.fail("read unsafe source"))
    with pytest.raises(d03.D03SourceDenied, match="symlink"):
        d03.stage(tmp_path, link)
