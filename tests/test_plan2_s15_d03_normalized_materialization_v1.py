"""Physical D03 normalized texts publish only immutable candidates, not model data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_d03_normalized_materialization_v1 as physical

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_12_normalized_real_files_and_restart(tmp_path: Path) -> None:
    dest = tmp_path / "physical"
    proof = physical.stage(ROOT, dest)
    assert proof["normalized_member_count"] == 12
    assert proof["normalized_total_bytes"] == 48675
    assert proof["physical_s3_s9_admitted"] is False
    assert proof["training_corpus_authorized"] is False
    assert proof["production_release_authorized"] is False
    assert proof["tokenizer_fit_authorized"] is False
    assert proof["terminal_done"] is False
    assert json.loads((dest / physical.OUTPUT).read_bytes()) == proof
    seen = set()
    for item in proof["normalized_members"]:
        body = (dest / item["path"]).read_bytes()
        assert len(body) == item["normalized_bytes"]
        assert physical.books.sha(body) == item["normalized_sha256"]
        assert item["record_id"] not in seen
        seen.add(item["record_id"])
    assert physical.stage(ROOT, dest) == proof


def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(physical, "build", lambda _root: (
        {"schema_version": physical.SCHEMA,
         "production_release_authorized": False},
        {"normalized/php/test.xml.txt": b"hello\n"},
    ))


def test_changed_physical_byte_refuses_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch)
    out = tmp_path / "out"
    physical.stage(tmp_path, out)
    (out / "normalized/php/test.xml.txt").write_bytes(b"changed\n")
    with pytest.raises(physical.D03MaterializationDenied, match="immutable"):
        physical.stage(tmp_path, out)


def test_orphan_normalized_member_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch)
    out = tmp_path / "out"
    physical.stage(tmp_path, out)
    (out / "orphan.txt").write_bytes(b"orphan\n")
    with pytest.raises(physical.D03MaterializationDenied, match="unexpected"):
        physical.stage(tmp_path, out)


def test_symlink_output_refused_without_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "safe"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(physical, "build", lambda _: pytest.fail("source read"))
    with pytest.raises(physical.D03MaterializationDenied, match="symlink"):
        physical.stage(tmp_path, link)
