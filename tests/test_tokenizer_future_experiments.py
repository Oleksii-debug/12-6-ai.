from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.tokenization import ByteTokenizer
from twelve_six.tokenization import experiments as exp
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerExperimentDependencyError,
    TokenizerExperimentError,
    TokenizerProbe,
    TokenizerTrainingManifest,
    measure_probe,
    ordered_vocab_sha256,
    summarize_by_language,
    vocabulary_parameter_cost,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _manifest(
    *,
    algorithm: str = "bpe",
    files: tuple[CorpusFileIdentity, ...] | None = None,
    vocab_size: int = 4096,
) -> TokenizerTrainingManifest:
    return TokenizerTrainingManifest(
        experiment_id=f"fixture-{algorithm}-v1",
        algorithm=algorithm,
        tokenizers_version="0.22.1",
        dataset_id="fixture",
        dataset_manifest_sha256=SHA_A,
        corpus_files=files
        or (
            CorpusFileIdentity("train/a.jsonl", SHA_B, 123),
            CorpusFileIdentity("train/b.jsonl", SHA_C, 456),
        ),
        vocab_size=vocab_size,
        min_frequency=2 if algorithm == "bpe" else None,
    )


def test_training_manifest_is_order_stable_but_identity_sensitive() -> None:
    first = _manifest()
    reversed_files = tuple(reversed(first.corpus_files))
    second = _manifest(files=reversed_files)
    assert first.sha256 == second.sha256

    changed = TokenizerTrainingManifest(
        experiment_id=first.experiment_id,
        algorithm=first.algorithm,
        tokenizers_version=first.tokenizers_version,
        dataset_id=first.dataset_id,
        dataset_manifest_sha256=SHA_C,
        corpus_files=first.corpus_files,
        vocab_size=first.vocab_size,
        min_frequency=first.min_frequency,
    )
    assert first.sha256 != changed.sha256
    assert first.to_dict()["canonical_s0_unchanged"] is True
    assert first.to_dict()["promotion_allowed"] is False


def test_training_manifest_rejects_ambiguous_inputs() -> None:
    duplicate = CorpusFileIdentity("train/a.jsonl", SHA_B, 123)
    with pytest.raises(TokenizerExperimentError, match="unique"):
        _manifest(files=(duplicate, duplicate))
    with pytest.raises(TokenizerExperimentError, match="Unigram"):
        TokenizerTrainingManifest(
            experiment_id="bad",
            algorithm="unigram",
            tokenizers_version="0.22.1",
            dataset_id="fixture",
            dataset_manifest_sha256=SHA_A,
            corpus_files=(duplicate,),
            vocab_size=1024,
            min_frequency=2,
        )


class _Vocab:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def test_ordered_vocab_hash_detects_silent_token_id_drift() -> None:
    original = _Vocab({"a": 0, "b": 1, "c": 2})
    drifted = _Vocab({"a": 1, "b": 0, "c": 2})
    assert ordered_vocab_sha256(original) != ordered_vocab_sha256(drifted)

    with pytest.raises(TokenizerExperimentError, match="contiguous"):
        ordered_vocab_sha256(_Vocab({"a": 0, "b": 2}))


def _d03_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in (
        Path("data/s0/packaged/train.jsonl"),
        Path("data/s0/packaged/validation.jsonl"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    return records


def test_s0_byte_fertility_roundtrip_and_unicode_coverage_on_controlled_d03_data() -> None:
    tokenizer = ByteTokenizer()
    results = []
    for record in _d03_records():
        results.append(
            measure_probe(
                tokenizer,
                TokenizerProbe(
                    name=str(record["id"]),
                    language=str(record["language"]),
                    category="d03-controlled",
                    text=str(record["text"]),
                ),
            )
        )

    summary = summarize_by_language(results)
    assert summary["en"]["probes"] == 6
    assert summary["en"]["codepoints"] == 811
    assert summary["en"]["tokens"] == 811
    assert summary["en"]["fertility_tokens_per_codepoint"] == 1.0
    assert summary["uk"]["probes"] == 6
    assert summary["uk"]["codepoints"] == 811
    assert summary["uk"]["tokens"] == 1515
    assert summary["uk"]["fertility_tokens_per_codepoint"] == pytest.approx(1515 / 811)
    assert all(result.round_trip_exact for result in results)
    assert all(result.unknown_tokens == 0 for result in results)

    evidence = json.loads(
        Path("configs/tokenizers/s0_controlled_metrics_v1.json").read_text(encoding="utf-8")
    )
    assert evidence["tokenizer"]["config_sha256"] == tokenizer.identity.config_sha256
    assert evidence["tokenizer"]["vocab_sha256"] == tokenizer.identity.vocab_sha256
    assert evidence["language_aggregate"]["en"]["tokens"] == summary["en"]["tokens"]
    assert evidence["language_aggregate"]["uk"]["tokens"] == summary["uk"]["tokens"]
    assert evidence["language_aggregate"]["uk"]["fertility_tokens_per_codepoint"] == pytest.approx(
        summary["uk"]["fertility_tokens_per_codepoint"]
    )


@pytest.mark.parametrize(
    ("name", "language", "category", "text"),
    [
        ("code-python", "code", "code", "def add(a: int, b: int) -> int:\\n    return a + b\\n"),
        (
            "unicode-mixed",
            "multi",
            "unicode",
            "Україна 🇺🇦 — naïve café; 数学; مرحبا; é",
        ),
    ],
)
def test_s0_byte_code_and_unicode_probes_are_lossless(
    name: str,
    language: str,
    category: str,
    text: str,
) -> None:
    result = measure_probe(
        ByteTokenizer(),
        TokenizerProbe(name=name, language=language, category=category, text=text),
    )
    assert result.round_trip_exact
    assert result.tokens == len(text.encode("utf-8"))
    assert result.unknown_tokens == 0


def test_vocabulary_parameter_cost_exposes_embedding_budget() -> None:
    s0 = vocabulary_parameter_cost(vocab_size=256, d_model=20, tied_lm_head=True)
    assert s0.embedding_parameters == 5120
    assert s0.lm_head_parameters == 0
    assert s0.total_vocabulary_parameters == 5120

    s4_candidate = vocabulary_parameter_cost(vocab_size=32768, d_model=768, tied_lm_head=True)
    assert s4_candidate.total_vocabulary_parameters == 25_165_824

    untied = vocabulary_parameter_cost(vocab_size=32768, d_model=768, tied_lm_head=False)
    assert untied.total_vocabulary_parameters == 50_331_648


def test_hf_experiment_runtime_is_lazy_and_fails_closed_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(exp.importlib, "import_module", _missing)
    with pytest.raises(TokenizerExperimentDependencyError, match="optional experiment runtime"):
        exp.train_hf_tokenizer(_manifest(), ["some training text"])
