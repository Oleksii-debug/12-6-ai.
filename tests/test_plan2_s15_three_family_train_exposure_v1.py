"""S15 verify three-family S13/S14 exact physical exposure and isolation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_s15_three_family_train_exposure_v1 as train

ROOT = Path(__file__).resolve().parents[1]


def test_real_3_family_train_shards_have_causal_replay(tmp_path: Path) -> None:
    receipt = train.stage(ROOT, tmp_path / "physical")
    assert receipt["decision"] == (
        "THREE_REAL_FAMILIES_S13_S14_PACKED_AND_REPLAYED_NOT_RELEASE"
    )
    assert receipt["train_document_count"] > 0
    assert receipt["heldout_document_count"] > 0
    assert receipt["block_count"] > 0
    assert receipt["target_count"] > 0
    assert receipt["actual_loss_targets_are_32k_tokenizer_based"] is True
    assert receipt["production_language_balance_approved"] is False
    assert set(receipt["actual_loss_targets_by_language"]) == {"en", "uk"}
    assert sum(receipt["actual_loss_targets_by_language"].values()) == (
        receipt["target_count"]
    )
    assert len(receipt["actual_loss_targets_by_source_family"]) == 3
    assert sum(receipt["actual_loss_targets_by_source_family"].values()) == (
        receipt["target_count"]
    )
    assert receipt["physical_train_byte_count"] > 0
    assert receipt["heldout_payloads_exposed"] is False
    assert receipt["real_final_test_outcomes_read"] is False
    assert receipt["plan9_optimizer_permission"] is False
    assert receipt["physical_s9_admitted"] is False
    assert receipt["training_corpus_authorized"] is False
    assert receipt["production_release_authorized"] is False
    assert receipt["terminal_done"] is False
    for key in (
        "source_s10_split_sha256", "frozen_s12_candidate_sha256",
        "physical_s13_shard_sha256", "physical_s14_exposure_sha256",
        "independent_s13_s14_readback_sha256", "ordered_exposure_chain_sha256",
    ):
        assert len(receipt[key]) == 64
    assert json.loads(
        (tmp_path / "physical" / train.OUTPUT).read_bytes()
    ) == receipt


def test_unexpected_combined_member_denied_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "orphan.txt").write_bytes(b"unexpected")
    fake = {
        "source_train_partition_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "train_document_count": 1,
        "heldout_document_count": 1,
        "heldout_plaintext_materialized": False,
        "production_release_authorized": False,
        "training_corpus_authorized": False,
        "cluster_split_sha256": "3" * 64,
        "tokenizer_manifest_sha256": "4" * 64,
        "target_count": 2,
        "block_count": 1,
        "manifest_sha256": "5" * 64,
    }
    monkeypatch.setattr(train.train_packing, "build", lambda _: (fake, {}))
    monkeypatch.setattr(train.exposure, "build_from_packed", lambda *_: (
        {
            "packing_manifest_sha256": "5" * 64,
            "cluster_split_sha256": "3" * 64,
            "tokenizer_manifest_sha256": "4" * 64,
            "target_count": 2,
            "block_count": 1,
            "production_release_authorized": False,
            "optimizer_effect_authorized": False,
        }, {},
    ))
    with pytest.raises(train.TrainExposureDenied, match="unexpected"):
        train.stage(tmp_path, destination)


def test_symlinked_output_denied_before_expensive_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(train.train_packing, "build",
                        lambda _: pytest.fail("read unsafe input"))
    with pytest.raises(train.TrainExposureDenied, match="symlink"):
        train.stage(tmp_path, link)


def test_precomputed_32k_fit_is_forwarded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"schema_version": "candidate", "manifest_sha256": "a" * 64}
    calls = []

    def deny_after_capture(root: Path, *, fitted: dict) -> None:
        calls.append((root, fitted))
        raise train.TrainExposureDenied("precomputed receipt recognized")

    monkeypatch.setattr(train.train_packing, "build", deny_after_capture)
    with pytest.raises(train.TrainExposureDenied, match="precomputed receipt"):
        train.stage(tmp_path, tmp_path / "destination", fitted=identity)
    assert calls == [(tmp_path, identity)]


def test_tampered_precomputed_32k_fitter_receipt_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import plan2_s15_three_family_train_packing_v1 as packer

    monkeypatch.setattr(packer.training, "build",
                        lambda _: ({"source_families_total": 3}, {}))
    monkeypatch.setattr(packer.fit, "inspect",
                        lambda _: pytest.fail("precomputed receipt must avoid refit"))
    forged = {"schema_version": packer.fit.SCHEMA,
              "manifest_sha256": "0" * 64,
              "actual_vocab_size": 32768}
    with pytest.raises(packer.ThreeFamilyPackingDenied, match="tampered"):
        packer.build(tmp_path, fitted=forged)
