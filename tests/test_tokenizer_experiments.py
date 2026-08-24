from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.tokenization.byte import BYTE_VOCAB_HASH, ByteTokenizer
from twelve_six.tokenization.experiments import (
    CONTROLLED_SAMPLES,
    TokenizerTrainingManifest,
    corpus_sha256,
    measure_tokenizer,
    ordered_vocab_sha256,
    vocabulary_parameter_cost,
)


def _manifest() -> TokenizerTrainingManifest:
    texts = ("one", "два")
    return TokenizerTrainingManifest(
        experiment_id="s1-controlled-bpe",
        stage="S1",
        algorithm="bpe",
        backend_library="tokenizers",
        backend_version="test-1.0",
        requested_vocab_size=512,
        training_corpus_sha256=corpus_sha256(texts),
        training_document_count=len(texts),
    )


def test_s0_byte_controlled_metrics_and_round_trip_are_exact() -> None:
    tokenizer = ByteTokenizer()
    metrics = {item.category: item for item in measure_tokenizer(tokenizer)}
    assert set(metrics) == set(CONTROLLED_SAMPLES)
    assert all(item.round_trip_rate == 1.0 for item in metrics.values())
    assert metrics["en"].fertility_tokens_per_code_point == 1.0
    assert metrics["uk"].fertility_tokens_per_code_point > metrics["en"].fertility_tokens_per_code_point
    assert metrics["code"].tokens == metrics["code"].utf8_bytes
    assert tokenizer.identity.vocab_sha256 == BYTE_VOCAB_HASH


def test_vocab_parameter_cost_is_explicit_for_tied_and_untied_heads() -> None:
    assert vocabulary_parameter_cost(256, 20) == 5_120
    assert vocabulary_parameter_cost(512, 48) == 24_576
    assert vocabulary_parameter_cost(512, 48, tied_embeddings=False) == 49_152
    assert vocabulary_parameter_cost(512, 48, lm_head_bias=True) == 25_088


def test_training_manifest_identity_is_stable_and_changes_on_semantic_drift() -> None:
    manifest = _manifest()
    assert manifest.identity_sha256 == _manifest().identity_sha256
    assert replace(manifest, requested_vocab_size=1024).identity_sha256 != manifest.identity_sha256
    assert replace(manifest, backend_version="test-2.0").identity_sha256 != manifest.identity_sha256


def test_training_manifest_rejects_backend_algorithm_mismatch() -> None:
    with pytest.raises(ValueError, match="must use maintained library"):
        TokenizerTrainingManifest(
            experiment_id="bad",
            stage="S1",
            algorithm="bpe",
            backend_library="sentencepiece",
            backend_version="1",
            requested_vocab_size=512,
            training_corpus_sha256=corpus_sha256(("x",)),
            training_document_count=1,
        )


def test_ordered_vocab_hash_detects_silent_token_id_drift() -> None:
    vocab = {"a": 0, "b": 1, "c": 2}
    original = ordered_vocab_sha256(vocab)
    assert ordered_vocab_sha256({"c": 2, "a": 0, "b": 1}) == original
    assert ordered_vocab_sha256({"a": 1, "b": 0, "c": 2}) != original
    with pytest.raises(ValueError, match="dense"):
        ordered_vocab_sha256({"a": 0, "b": 2})
