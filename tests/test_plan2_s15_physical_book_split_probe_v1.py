"""S15 physical-book cluster mechanics and negative/restart evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_physical_book_split_probe_v1 as probe

ROOT = Path(__file__).resolve().parents[1]


def test_three_real_books_form_isolated_document_clusters(tmp_path: Path) -> None:
    out = tmp_path / "book-probe"
    first = probe.stage(ROOT, out)
    assert first == probe.stage(ROOT, out)
    assert first == probe.stage(ROOT, tmp_path / "clean-build")
    assert first["physical_document_clusters"] == 3
    assert first["canonical_source_families"] == 1
    assert first["physical_book_span_count"] == 6
    assert first["verified_physical_bytes"] == 1_265_481
    assert set(first["assigned_document_splits"].values()) == {
        "train", "validation", "test"
    }
    assert first["decision"] == "S10_MECHANICS_ONLY_NOT_PHYSICAL_S9_ADMISSION"
    assert first["physical_s9_admitted"] is False
    for key in ("training_corpus_authorized", "production_release_authorized",
                "tokenizer_fit_authorized", "real_final_test_accessed"):
        assert first[key] is False
    assert json.loads((out / probe.OUTPUT).read_bytes()) == first
    assert (out / probe.OUTPUT).read_bytes() == (
        tmp_path / "clean-build" / probe.OUTPUT).read_bytes()


def test_changed_physical_book_bytes_denied(monkeypatch) -> None:
    original = probe.books.read_checked

    def tamper(root: Path, name: str) -> bytes:
        raw = original(root, name)
        if name.endswith("alice-in-wonderland.en.txt"):
            return raw[:-1] + b"X"
        return raw

    monkeypatch.setattr(probe.books, "read_checked", tamper)
    with pytest.raises(ValueError, match="SHA-256 or length"):
        probe.inspect(ROOT)


def test_no_second_fake_source_family(monkeypatch) -> None:
    original = probe.books.inspect

    def promoted(root: Path):
        return {**original(root), "physical_source_families": 3}

    monkeypatch.setattr(probe.books, "inspect", promoted)
    with pytest.raises(probe.BookSplitProbeDenied, match="identity"):
        probe.inspect(ROOT)


def test_split_cannot_promote_real_book_training(monkeypatch) -> None:
    original = probe.split.build_cluster_split

    def illicit(*args, **kwargs):
        return {**original(*args, **kwargs),
                "training_corpus_authorized": True}

    monkeypatch.setattr(probe.split, "build_cluster_split", illicit)
    with pytest.raises(probe.BookSplitProbeDenied, match="release authority"):
        probe.inspect(ROOT)


def test_changed_partition_denied(monkeypatch) -> None:
    original = probe.split.build_cluster_split

    def contaminated(*args, **kwargs):
        receipt = original(*args, **kwargs)
        return {**receipt, "record_assignments": {
            key: "test" for key in receipt["record_assignments"]
        }}

    monkeypatch.setattr(probe.split, "build_cluster_split", contaminated)
    with pytest.raises(probe.BookSplitProbeDenied, match="partition"):
        probe.inspect(ROOT)


def test_immutable_probe_refuses_changed_report(tmp_path: Path) -> None:
    target_dir = tmp_path / "report"
    probe.stage(ROOT, target_dir)
    path = target_dir / probe.OUTPUT
    path.write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(probe.BookSplitProbeDenied, match="immutable"):
        probe.stage(ROOT, target_dir)


def test_symlink_probe_destination_refused(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(probe, "inspect", lambda _:
                        pytest.fail("must not inspect before symlink check"))
    with pytest.raises(probe.BookSplitProbeDenied, match="symlink"):
        probe.stage(ROOT, symlink)


def test_probe_has_no_effect_on_rights_policy(tmp_path: Path) -> None:
    before = (ROOT / "configs/data/plan2_rights_admissibility_v1.json").read_bytes()
    probe.stage(ROOT, tmp_path / "report")
    after = (ROOT / "configs/data/plan2_rights_admissibility_v1.json").read_bytes()
    assert before == after
