from __future__ import annotations

import pytest

from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.tokenization.scale_metrics import measure_tokenizer, vocabulary_parameter_cost


class CodepointTokenizer:
    pad_id = None
    bos_id = None
    eos_id = None
    vocab_size = 65_536
    version = "test-codepoint"

    @property
    def identity(self) -> object:
        raise NotImplementedError

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        assert not add_bos and not add_eos
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del skip_special_tokens, errors
        return "".join(chr(token_id) for token_id in token_ids)


def test_byte_measurement_is_exact_raw_utf8_baseline() -> None:
    texts = ["Україна і AI", "def f(x): return x + 1", ""]
    result = measure_tokenizer(ByteTokenizer(), texts, context_length=16).to_dict()

    assert result["tokens"] == result["utf8_bytes"]
    assert result["relative_sequence_length_vs_raw_byte"] == 1.0
    assert result["relative_dense_attention_pairs_vs_raw_byte"] == 1.0
    assert result["roundtrip_failures"] == 0
    assert result["empty_documents"] == 1
    assert result["documents"] == 3
    assert result["documents_over_context"] >= 1


def test_codepoint_candidate_exposes_utf8_sequence_compression() -> None:
    text = "українська"
    result = measure_tokenizer(CodepointTokenizer(), [text], context_length=64).to_dict()

    assert result["tokens"] == len(text)
    assert result["utf8_bytes"] == len(text.encode("utf-8"))
    assert 0.0 < result["relative_sequence_length_vs_raw_byte"] < 1.0
    assert (
        result["relative_dense_attention_pairs_vs_raw_byte"]
        < result["relative_sequence_length_vs_raw_byte"]
    )
    assert result["roundtrip_failures"] == 0


def test_vocabulary_cost_exposes_small_model_embedding_pressure() -> None:
    result = vocabulary_parameter_cost(
        vocab_size=8192,
        d_model=320,
        tied_embeddings=True,
        target_parameters=20_613_440,
    )

    assert result["embedding_parameters"] == 8192 * 320
    assert result["baseline_embedding_parameters"] == 256 * 320
    assert result["embedding_parameter_delta_vs_byte_vocab"] == (8192 - 256) * 320
    assert 0.12 < result["embedding_share_of_target_parameters"] < 0.13


def test_untied_embeddings_double_vocabulary_cost() -> None:
    tied = vocabulary_parameter_cost(vocab_size=4096, d_model=256, tied_embeddings=True)
    untied = vocabulary_parameter_cost(vocab_size=4096, d_model=256, tied_embeddings=False)
    assert untied["embedding_parameters"] == 2 * tied["embedding_parameters"]


def test_measurement_rejects_invalid_context() -> None:
    with pytest.raises(ValueError, match="context_length"):
        measure_tokenizer(ByteTokenizer(), ["x"], context_length=1)
