"""Real, pinned public-domain snapshots are verified; training is NOT released."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_public_domain_books_v1 as books

ROOT = Path(__file__).resolve().parents[1]


def test_three_independent_real_books_rebuild_and_restart(tmp_path: Path) -> None:
    first = books.stage(ROOT, tmp_path / "first")
    repeat = books.stage(ROOT, tmp_path / "first")
    clean = books.stage(ROOT, tmp_path / "second")
    assert first == repeat == clean
    assert first["physical_source_families"] == 1
    assert first["physical_document_families"] == 3
    assert first["physical_source_bytes"] == 1_265_481
    assert first["g06_counts"]["records"] == 3
    assert len(first["g06_execution_identity_sha256"]) == 64
    assert len(first["g06_input_root_sha256"]) == 64
    assert len(first["g06_privacy_policy_git_blob"]) == 40
    assert isinstance(first["g06_rejected_record_ids"], list)
    assert first["training_corpus_authorized"] is False
    assert first["rights_boundary"] == (
        "ORIGINAL_WORK_SOURCE_PERMISSIONS_ONLY_NOT_CORPUS"
    )
    for field in (
        "source_inventory_sha256", "source_rights_catalog_sha256",
        "source_training_rights_receipt_sha256",
        "source_release_rights_receipt_sha256",
    ):
        assert len(first[field]) == 64
    assert first["tokenizer_fit_authorized"] is False
    assert first["production_release_authorized"] is False
    assert len({item["source_family"] for item in first["books"]}) == 1
    assert len({item["document_family"] for item in first["books"]}) == 3
    assert json.loads((tmp_path / "first" / books.OUTPUT).read_bytes()) == first
    assert first["manifest_sha256"] == books.sha(books.canonical({
        k: v for k, v in first.items() if k != "manifest_sha256"
    }))


def test_text_member_tamper_denied_before_training(monkeypatch) -> None:
    original = books.read_checked

    def corrupted(root: Path, name: str) -> bytes:
        raw = original(root, name)
        if name.endswith("pride-and-prejudice.en.txt"):
            return raw[:-1] + b"X"
        return raw

    monkeypatch.setattr(books, "read_checked", corrupted)
    with pytest.raises(books.BookCohortDenied, match="SHA-256 or length"):
        books.inspect(ROOT)


def test_catalog_modification_denied_by_pinned_git_blob(monkeypatch) -> None:
    original = books.read_checked

    def corrupted(root: Path, name: str) -> bytes:
        raw = original(root, name)
        if name == books.CATALOG:
            return raw + b" "
        return raw

    monkeypatch.setattr(books, "read_checked", corrupted)
    with pytest.raises(books.BookCohortDenied, match="Git blob changed"):
        books.inspect(ROOT)


def test_missing_physical_member_fails_closed(monkeypatch) -> None:
    original = books.read_checked

    def missing(root: Path, name: str) -> bytes:
        if name.endswith("alice-in-wonderland.en.txt"):
            raise books.BookCohortDenied("source missing")
        return original(root, name)

    monkeypatch.setattr(books, "read_checked", missing)
    with pytest.raises(books.BookCohortDenied, match="source missing"):
        books.inspect(ROOT)


def test_existing_audit_report_refuses_tamper(tmp_path: Path) -> None:
    folder = tmp_path / "report"
    books.stage(ROOT, folder)
    target = folder / books.OUTPUT
    target.write_bytes(b'{"training_corpus_authorized":true}\n')
    with pytest.raises(books.BookCohortDenied, match="immutable"):
        books.stage(ROOT, folder)


def test_symlink_destination_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "external"
    symlink.symlink_to(real, target_is_directory=True)
    with pytest.raises(books.BookCohortDenied, match="symlink"):
        books.stage(ROOT, symlink)


def test_traversal_and_symlinked_source_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(books.BookCohortDenied, match="escaped or ambiguous"):
        books.read_checked(ROOT, "../missing-secret")
    link = tmp_path / "linked"
    link.symlink_to(ROOT / books.CATALOG)
    with pytest.raises(books.BookCohortDenied, match="symlink"):
        books.read_checked(tmp_path, "linked")


def test_missing_or_modified_source_rights_evidence_is_denied(monkeypatch) -> None:
    original = books.read_checked

    def changed_rights(root: Path, name: str) -> bytes:
        raw = original(root, name)
        return raw + b"tampered" if name == books.RIGHTS_EVIDENCE else raw

    monkeypatch.setattr(books, "read_checked", changed_rights)
    with pytest.raises(books.BookCohortDenied, match="rights evidence changed"):
        books.inspect(ROOT)


def test_training_rights_authority_cannot_promote_corpus(monkeypatch) -> None:
    real_receipt = books.rights.materialization_receipt

    def invented_authority(*args, **kwargs):
        result = real_receipt(*args, **kwargs)
        return {**result, "training_corpus_authorized": True}

    monkeypatch.setattr(books.rights, "materialization_receipt",
                        invented_authority)
    with pytest.raises(books.BookCohortDenied, match="promoted training"):
        books.inspect(ROOT)


def test_broken_g06_privacy_identity_cannot_be_published(monkeypatch) -> None:
    original = books.g06.build_privacy_execution_authority

    def corrupted(*args, **kwargs):
        receipt = original(*args, **kwargs)
        return {**receipt, "execution_identity_sha256": "0" * 64}

    monkeypatch.setattr(books.g06, "build_privacy_execution_authority", corrupted)
    with pytest.raises(ValueError, match="identity|hash|drift|mismatch"):
        books.inspect(ROOT)
