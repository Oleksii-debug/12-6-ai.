"""S16 physical source admission is a bounded, nonrelease checkpoint."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.plan2_s16_source_rights_privacy_v1 import (
    OUTPUT,
    SourceAdmissionDenied,
    inspect,
    stage,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verified() -> dict:
    return inspect(ROOT)


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    shutil.copytree(ROOT / "configs/data", root / "configs/data")
    shutil.copytree(ROOT / "data/external", root / "data/external")
    return root


def test_real_source_rights_privacy_are_bound_without_corpus_promotion(verified: dict) -> None:
    assert verified["source_level_rights_verified"] is True
    assert verified["physical_privacy_executed"] is True
    assert verified["fixture_decontamination_executed"] is True
    assert verified["physical_record_count"] == 15
    assert len(verified["source_ids"]) == 5
    assert all(verified[key] is False for key in (
        "independent_ua_en_holdouts_verified", "cross_language_semantic_dedup_verified",
        "s3_s9_corpus_admitted", "training_corpus_authorized",
        "tokenizer_fit_authorized", "production_release_authorized",
        "plan9_binding_authorized", "terminal_done",
    ))


def test_immutable_restart_and_clean_rebuild(tmp_path: Path, verified: dict) -> None:
    first = stage(ROOT, tmp_path / "first")
    assert first == verified
    assert stage(ROOT, tmp_path / "first") == first
    assert stage(ROOT, tmp_path / "second") == first
    assert (tmp_path / "first" / OUTPUT).read_bytes() == (tmp_path / "second" / OUTPUT).read_bytes()
    (tmp_path / "first" / OUTPUT).write_bytes(b"{}\n")
    with pytest.raises(SourceAdmissionDenied, match="immutable"):
        stage(ROOT, tmp_path / "first")


@pytest.mark.parametrize("target", [
    "data/external/rights-evidence/plan2-books/public-domain-original-works-v1.txt",
    "data/external/snapshots/plan2-public-domain-books-v1/pride-and-prejudice.en.txt",
    "data/external/snapshots/plan2-s15-d03-ua-v1/licenses/rust/LICENSE-MIT",
    "data/external/snapshots/plan2-s15-d03-ua-v1/raw/rust/ch01-01-installation.md",
])
def test_changed_license_or_physical_member_denied(tmp_path: Path, target: str) -> None:
    root = _source_tree(tmp_path)
    path = root / target
    path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(SourceAdmissionDenied, match="rights/privacy evidence denied"):
        inspect(root)


def test_missing_ua_source_denied(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    (root / "data/external/snapshots/plan2-s15-d03-ua-v1/raw/php/language/context.xml").unlink()
    with pytest.raises(SourceAdmissionDenied, match="rights/privacy evidence denied"):
        inspect(root)


def test_no_manifest_on_source_failure(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    (root / "data/external/rights-evidence/plan2-books/public-domain-original-works-v1.txt").unlink()
    destination = tmp_path / "output"
    with pytest.raises(SourceAdmissionDenied):
        stage(root, destination)
    assert not destination.exists()
