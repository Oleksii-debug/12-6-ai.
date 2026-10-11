"""S15 five physical-source rights: source grants cannot authorize corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_five_source_rights_v1 as rights

ROOT = Path(__file__).resolve().parents[1]


def test_five_real_source_level_rights_custody(tmp_path: Path) -> None:
    proof = rights.stage(ROOT, tmp_path / "source-rights")
    assert proof["source_count"] == 5
    assert proof["canonical_source_family_count"] == 3
    assert proof["source_level_license_training_permission_evidenced"] is True
    assert proof["source_level_release_conditions_require_attribution"] is True
    assert proof["source_inventory_status"] == "candidate"
    assert proof["s3_s9_physical_admission"] is False
    assert proof["tokenizer_fit_authorized"] is False
    assert proof["training_corpus_authorized"] is False
    assert proof["production_release_authorized"] is False
    assert proof["terminal_done"] is False
    assert len(proof["source_inventory_sha256"]) == 64
    assert len(proof["source_rights_catalog_sha256"]) == 64
    assert proof["source_inventory_sha256"] != proof["source_rights_catalog_sha256"]
    assert json.loads((tmp_path / "source-rights" / rights.OUTPUT).read_bytes()) == proof
    assert rights.stage(ROOT, tmp_path / "source-rights") == proof


@pytest.mark.parametrize("field", [
    "source_family_count", "document_identity_count",
    "training_corpus_authorized", "physical_s3_s9_multi_family_admitted",
])
def test_forged_upstream_source_promotion_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    receipt = {
        "source_family_count": 3,
        "document_identity_count": 15,
        "training_corpus_authorized": False,
        "physical_s3_s9_multi_family_admitted": False,
    }
    receipt[field] = (
        2 if field == "source_family_count"
        else 14 if field == "document_identity_count" else True
    )
    monkeypatch.setattr(rights.combined, "inspect", lambda _: receipt)
    with pytest.raises(rights.RightsBoundaryDenied, match="physical source cohort"):
        rights.inspect(tmp_path)


def test_changed_source_license_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = rights.books.read_checked

    def forged(root: Path, path: str) -> bytes:
        if path == rights.BOOK_RIGHTS:
            return b"forged public-domain rights"
        return original(root, path)

    monkeypatch.setattr(rights.books, "read_checked", forged)
    with pytest.raises((rights.RightsBoundaryDenied, ValueError)):
        rights.inspect(ROOT)


def test_changed_immutable_rights_publication_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {"schema_version": rights.SCHEMA,
              "training_corpus_authorized": False}
    monkeypatch.setattr(rights, "inspect", lambda _: report)
    destination = tmp_path / "published"
    rights.stage(tmp_path, destination)
    (destination / rights.OUTPUT).write_bytes(
        b'{"training_corpus_authorized":true}\n'
    )
    with pytest.raises(rights.RightsBoundaryDenied, match="immutable"):
        rights.stage(tmp_path, destination)


def test_source_rights_symlink_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dst = tmp_path / "real"
    dst.mkdir()
    link = tmp_path / "link"
    link.symlink_to(dst, target_is_directory=True)
    monkeypatch.setattr(rights, "inspect", lambda _: pytest.fail("unsafe read"))
    with pytest.raises(rights.RightsBoundaryDenied, match="symlink"):
        rights.stage(tmp_path, link)
