"""S15 train-only physical tokenizer never fits heldout/evaluation text."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import plan2_s15_three_family_train_bpe_v1 as candidate

ROOT = Path(__file__).resolve().parents[1]


def test_real_three_family_train_only_bpe_roundtrip(tmp_path: Path) -> None:
    result = candidate.stage(ROOT, tmp_path / "model")
    assert result["decision"] == (
        "THREE_FAMILY_TRAIN_ONLY_BPE_CANDIDATE_NOT_PRODUCTION"
    )
    assert result["source_train_document_count"] > 0
    assert result["source_train_document_count"] <= 15
    assert result["source_train_utf8_bytes"] > 0
    assert result["training_segment_count"] >= result["source_train_document_count"]
    assert result["all_train_segments_byte_roundtrip"] is True
    assert result["heldout_payloads_fitted"] is False
    assert result["real_evaluation_outcomes_read"] is False
    assert result["target_vocab_size"] == 32768
    assert result["candidate_vocab_size"] <= 388
    assert result["production_target_vocab_frozen"] is False
    assert result["training_corpus_authorized"] is False
    assert result["tokenizer_fit_authorized"] is False
    assert result["production_release_authorized"] is False
    assert result["terminal_done"] is False
    assert len(result["manifest_sha256"]) == 64
    assert candidate.stage(ROOT, tmp_path / "model") == result


@pytest.mark.parametrize("key", [
    "source_families_total", "heldout_document_count",
    "heldout_plaintext_materialized", "training_corpus_authorized",
    "tokenizer_fit_authorized", "physical_s9_admitted",
])
def test_upstream_training_or_holdout_promotion_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    source = {
        "source_families_total": 3,
        "train_document_count": 1,
        "heldout_document_count": 2,
        "heldout_plaintext_materialized": False,
        "physical_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
    }
    source[key] = (
        2 if key == "source_families_total"
        else 0 if key == "heldout_document_count"
        else True
    )
    monkeypatch.setattr(candidate.training, "build", lambda _: (
        source, {"train/valid.utf8": b"real source"}
    ))
    with pytest.raises(candidate.TrainBpeDenied, match="heldout-isolated"):
        candidate.inspect(tmp_path)


def test_model_receipt_tamper_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = {"decision": "THREE_FAMILY_TRAIN_ONLY_BPE_CANDIDATE_NOT_PRODUCTION",
             "production_release_authorized": False}
    monkeypatch.setattr(candidate, "inspect", lambda _: proof)
    target = tmp_path / "fit"
    candidate.stage(tmp_path, target)
    (target / candidate.OUTPUT).write_bytes(
        b'{"production_release_authorized":true}\n'
    )
    with pytest.raises(candidate.TrainBpeDenied, match="immutable"):
        candidate.stage(tmp_path, target)


def test_symlink_fit_output_denied_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(candidate, "inspect", lambda _: pytest.fail("unsafe source"))
    with pytest.raises(candidate.TrainBpeDenied, match="symlink"):
        candidate.stage(tmp_path, link)
