"""S15 real-source token fit stays non-release and preserves train/holdout isolation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import plan2_s15_physical_tokenizer_candidate_v1 as realfit

ROOT = Path(__file__).resolve().parents[1]


def test_real_training_document_is_fitted_but_not_released(tmp_path: Path) -> None:
    proof = realfit.stage(ROOT, tmp_path / "fit")
    assert proof["train_record_count"] > 0
    assert proof["train_document_bytes"] > 100_000
    assert proof["byte_roundtrip_verified"] is True
    assert proof["sampled_only"] is False
    assert proof["frozen"] is True
    assert proof["production_release_authorized"] is False
    assert proof["training_corpus_authorized"] is False
    assert proof["tokenizer_fit_authorized"] is False
    assert proof["physical_s9_admitted"] is False
    assert proof["terminal_done"] is False
    assert len(proof["physical_split_manifest_sha256"]) == 64
    assert proof["physical_split_manifest_sha256"] != proof["physical_split_probe_sha256"]
    assert len(proof["manifest_sha256"]) == 64
    assert realfit.stage(ROOT, tmp_path / "fit") == proof


def test_line_segmentation_reconstructs_unicode_and_has_disjoint_ids() -> None:
    source = "en.physical.book"
    text = ("Це довгий тест \N{SNOWMAN}\n" * 2000) + "ending"
    rows = realfit.segments(source, text)
    assert len(rows) > 1
    assert "".join(r["text"] for r in rows) == text
    assert len({r["record_id"] for r in rows}) == len(rows)
    assert all(r["record_id"].startswith(source + ":r5") for r in rows)
    assert all(len(r["text"].encode("utf-8")) <= realfit.MAX_BYTES for r in rows)
    assert all(r["record_id"] not in (source + ":r00000000",
                                     source + ":r00000001") for r in rows)


def test_overlong_single_line_fails_closed() -> None:
    with pytest.raises(realfit.PhysicalFitDenied, match="oversized"):
        realfit.segments("en.book", "a" * (realfit.MAX_BYTES + 1))


@pytest.mark.parametrize("field", [
    "physical_heldout_decontamination_clean", "final_test_outcomes_read",
    "terminal_done", "physical_s9_admitted",
])
def test_forged_heldout_authority_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    receipt = {
        "physical_heldout_decontamination_clean": True,
        "final_test_outcomes_read": False,
        "terminal_done": False,
        "physical_s9_admitted": False,
    }
    receipt[field] = not receipt[field]
    monkeypatch.setattr(realfit.heldout, "inspect", lambda _root: receipt)
    with pytest.raises(realfit.PhysicalFitDenied, match="heldout"):
        realfit.inspect(tmp_path)


def test_immutable_fit_receipt_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"schema_version": realfit.SCHEMA,
               "production_release_authorized": False}
    monkeypatch.setattr(realfit, "inspect", lambda _: receipt)
    target = tmp_path / "fit"
    realfit.stage(tmp_path, target)
    (target / realfit.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(realfit.PhysicalFitDenied, match="immutable"):
        realfit.stage(tmp_path, target)
