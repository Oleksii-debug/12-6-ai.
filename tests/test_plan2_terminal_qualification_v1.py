"""Section-15 qualification must report real candidate limits, never synthetic DONE."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_terminal_qualification_v1 as terminal

ROOT = Path(__file__).resolve().parents[1]


def test_real_candidate_and_fixture_clean_rebuild_are_not_release(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    report = terminal.stage(ROOT, out)
    assert report["reproducible_clean_builds"] == 2
    assert report["decision"] == "COMPONENT_AUDIT_ONLY_NOT_TERMINAL"
    assert report["terminal_done"] is False
    assert report["production_release_authorized"] is False
    assert report["physical_corpus_training_authorized"] is False
    assert report["evidence"]["physical_split"] == "DENIED_SINGLE_SOURCE_FAMILY"
    assert report["evidence"]["physical_source_family_count"] == 1
    assert report["evidence"]["synthetic_target_count"] > 0
    assert len(report["evidence"]["member_sha256"]) >= 7
    assert len(report["blocking_gates"]) == 5
    published = out / terminal.OUTPUT
    assert json.loads(published.read_bytes()) == report
    assert terminal._digest(terminal._canonical({
        k: v for k, v in report.items() if k != "audit_sha256"
    })) == report["audit_sha256"]


def test_immutable_readback_refuses_tampering(tmp_path: Path, monkeypatch) -> None:
    report = {
        "schema_version": terminal.SCHEMA,
        "terminal_done": False,
        "production_release_authorized": False,
    }
    monkeypatch.setattr(terminal, "audit", lambda _: report)
    directory = tmp_path / "audit"
    assert terminal.stage(ROOT, directory) == report
    target = directory / terminal.OUTPUT
    target.write_bytes(b'{"terminal_done":true}\n')
    with pytest.raises(terminal.QualificationDenied, match="immutable"):
        terminal.stage(ROOT, directory)
    assert target.read_bytes() == b'{"terminal_done":true}\n'


def test_symlink_publication_refused_before_processing(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    monkeypatch.setattr(terminal, "audit", lambda _: pytest.fail("unsafe audit"))
    with pytest.raises(terminal.QualificationDenied, match="symlink"):
        terminal.stage(ROOT, link)


def test_missing_physical_source_does_not_fallback_to_fixture(monkeypatch) -> None:
    def no_source(*_args, **_kwargs):
        raise terminal.physical.Plan2MaterializationError("source missing")

    monkeypatch.setattr(terminal.physical, "stage_candidate_cohort", no_source)
    with pytest.raises(terminal.physical.Plan2MaterializationError, match="missing"):
        terminal.audit(ROOT)


def test_unexpected_split_error_is_not_misclassified(tmp_path: Path, monkeypatch) -> None:
    def broken_split(*_args, **_kwargs):
        raise terminal.split.Plan2SplitError("unexpected integrity failure")

    monkeypatch.setattr(terminal.split, "stage_candidate", broken_split)
    with pytest.raises(terminal.QualificationDenied, match="unexpected physical split"):
        terminal.audit(ROOT)


def test_changed_physical_family_count_refuses_stale_audit(monkeypatch) -> None:
    def multiple_families(*_args, **_kwargs):
        return {
            "training_corpus_authorized": False,
            "contributions": {"family": {"document-a": {}, "document-b": {}}},
        }

    monkeypatch.setattr(terminal.mixture, "stage_mixture", multiple_families)
    with pytest.raises(terminal.QualificationDenied, match="source-family count changed"):
        terminal.audit(ROOT)
