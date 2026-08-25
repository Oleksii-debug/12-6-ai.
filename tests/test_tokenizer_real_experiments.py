from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.tokenization import real_experiments as real


def _rehash(report: dict[str, object]) -> None:
    report.pop("evidence_sha256", None)
    report["evidence_sha256"] = real._sha256_bytes(real._canonical_json(report).encode())


def _truthful_report() -> dict[str, object]:
    source_sha = "a" * 40
    train_ids = [f"train-{index}" for index in range(10)]
    algorithm = {
        "status": "PASS",
        "training_input": {
            "split": "train",
            "records": 10,
            "record_ids": train_ids,
            "validation_used_for_training": False,
        },
        "repeated_build_identity_equal": True,
        "held_out": {
            "strict_round_trip_all": True,
            "unknown_tokens": 0,
        },
        "locked_for_s1": False,
    }
    report: dict[str, object] = {
        "schema": real.SCHEMA,
        "authority": real.AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "observed_head_sha": source_sha,
        },
        "dataset": {
            "train_validation_record_overlap": [],
            "representative_s1_corpus": False,
            "external_sources_training_approved": False,
        },
        "algorithms": {
            "bpe": copy.deepcopy(algorithm),
            "unigram": copy.deepcopy(algorithm),
        },
        "gates": {
            "exact_source_binding": "PASS",
            "hash_locked_experiment_runtime": "PASS",
            "same_train_corpus_for_algorithms": "PASS",
            "train_validation_separation": "PASS",
            "real_bpe_execution": "PASS",
            "real_unigram_execution": "PASS",
            "repeatable_artifact_identity": "PASS",
            "held_out_strict_round_trip": "PASS",
            "held_out_zero_unknown_tokens": "PASS",
            "representative_s1_corpus": "NOT_TESTED",
            "external_source_rights_approval": "NOT_TESTED",
            "s1_tokenizer_freeze": "NOT_TESTED",
            "model_quality": "NOT_TESTED",
        },
        "decision": {
            "status": "NO_FREEZE_CONTROLLED_MECHANICS_ONLY",
            "winner": None,
        },
        "truth_boundary": {
            "canonical_s0_tokenizer_unchanged": True,
            "s1_tokenizer_frozen": False,
            "external_sources_training_approved": False,
            "representative_corpus_claimed": False,
            "model_quality_claimed": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "audit_pass_claimed": False,
            "candidate_or_stable_claimed": False,
        },
    }
    _rehash(report)
    return report


def test_experiment_lock_is_one_exact_hash_pinned_package() -> None:
    evidence = real.verify_experiment_lock()
    assert evidence["version"] == "0.23.1"
    assert evidence["wheel_sha256"] == real.TOKENIZERS_WHEEL_SHA256
    assert evidence["install_policy"] == "pip --require-hashes --only-binary=:all: --no-deps"


def test_dataset_contract_keeps_controlled_train_and_validation_disjoint() -> None:
    dataset = real._dataset_contract()
    assert len(dataset["train_records"]) == 10
    assert len(dataset["validation_records"]) == 2
    assert dataset["train_validation_overlap"] == []
    assert set(dataset["train_record_ids"]).isdisjoint(dataset["validation_record_ids"])


def test_bpe_and_unigram_manifests_bind_same_train_only_corpus() -> None:
    dataset = real._dataset_contract()
    bpe = real._training_manifest("bpe", dataset, vocab_size=512)
    unigram = real._training_manifest("unigram", dataset, vocab_size=512)
    for manifest in (bpe, unigram):
        assert [item.path for item in manifest.corpus_files] == [
            "data/s0/packaged/train.jsonl"
        ]
        assert manifest.dataset_id == dataset["dataset_id"]
        assert manifest.dataset_manifest_sha256 == dataset["dataset_manifest_sha256"]
        assert all("validation" not in item.path for item in manifest.corpus_files)
    assert bpe.algorithm == "bpe"
    assert unigram.algorithm == "unigram"
    assert bpe.sha256 != unigram.sha256


def test_truthful_controlled_mechanics_report_validates() -> None:
    report = _truthful_report()
    assert real.validate_report(report, expected_source_sha="a" * 40) is report


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["algorithms"]["bpe"]["training_input"].__setitem__(
            "validation_used_for_training", True
        ),
        lambda report: report["dataset"].__setitem__("representative_s1_corpus", True),
        lambda report: report["gates"].__setitem__("representative_s1_corpus", "PASS"),
        lambda report: report["decision"].__setitem__("winner", "bpe"),
        lambda report: report["algorithms"]["unigram"].__setitem__("locked_for_s1", True),
        lambda report: report["truth_boundary"].__setitem__(
            "external_sources_training_approved", True
        ),
    ],
)
def test_validator_rejects_overclaim_even_after_rehash(mutator: object) -> None:
    report = _truthful_report()
    mutator(report)
    _rehash(report)
    with pytest.raises(real.TokenizerEvidenceError):
        real.validate_report(report)


def test_validator_rejects_missing_unigram_even_after_rehash() -> None:
    report = _truthful_report()
    del report["algorithms"]["unigram"]
    _rehash(report)
    with pytest.raises(real.TokenizerEvidenceError, match="exactly BPE and Unigram"):
        real.validate_report(report)


def test_validator_rejects_self_hash_tamper() -> None:
    report = _truthful_report()
    report["decision"]["winner"] = "bpe"
    with pytest.raises(real.TokenizerEvidenceError, match="self-hash"):
        real.validate_report(report)


def test_floating_or_wrong_hash_experiment_lock_is_rejected(tmp_path: Path) -> None:
    floating = tmp_path / "floating.txt"
    floating.write_text("tokenizers>=0.23\n", encoding="utf-8")
    with pytest.raises(real.TokenizerEvidenceError, match="exactly the admitted wheel hash"):
        real.verify_experiment_lock(floating)

    wrong_hash = tmp_path / "wrong.txt"
    wrong_hash.write_text("tokenizers==0.23.1 --hash=sha256:" + "0" * 64 + "\n")
    with pytest.raises(real.TokenizerEvidenceError, match="exactly the admitted wheel hash"):
        real.verify_experiment_lock(wrong_hash)


def test_evidence_json_round_trip_keeps_validator_authority(tmp_path: Path) -> None:
    report = _truthful_report()
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert real.validate_report(loaded)["authority"] == real.AUTHORITY
