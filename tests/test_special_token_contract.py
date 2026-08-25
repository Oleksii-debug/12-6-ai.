from __future__ import annotations

import hashlib
from itertools import pairwise

import pytest

from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.transformers_llama import llama_config_dict
from twelve_six.model import ModelSpec
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    EXPERIMENTAL_CONFIG_SHA256,
    EXPERIMENTAL_EOS_ID,
    EXPERIMENTAL_TOKENIZER_VERSION,
    EXPERIMENTAL_VOCAB_SHA256,
    EXPERIMENTAL_VOCAB_SIZE,
    ByteTokenizer,
    ExperimentalByteEosTokenizer,
    TokenizerCompatibilityError,
    apply_to_llama_config,
    contract_artifact_bytes,
    contract_payload,
    hf_special_token_ids,
    require_tokenizer_identity,
    vocab_artifact_bytes,
)


def _training_pairs(examples):
    pairs = []
    for example in examples:
        for index, keep in enumerate(example.loss_mask):
            if keep:
                pairs.append((example.input_ids[index], example.labels[index + 1]))
    return pairs


def test_contract_has_exact_hashes_and_preserves_frozen_s0_identity() -> None:
    tokenizer = ExperimentalByteEosTokenizer()
    s0 = ByteTokenizer()

    assert BYTE_TOKENIZER_VERSION == "s0-byte-v1"
    assert BYTE_TOKENIZER_HASH == (
        "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
    )
    assert BYTE_VOCAB_HASH == (
        "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
    )
    assert EXPERIMENTAL_TOKENIZER_VERSION == "exp-byte-eos-v1"
    assert EXPERIMENTAL_EOS_ID == 256
    assert EXPERIMENTAL_VOCAB_SIZE == 257
    assert hashlib.sha256(contract_artifact_bytes()).hexdigest() == EXPERIMENTAL_CONFIG_SHA256
    assert hashlib.sha256(vocab_artifact_bytes()).hexdigest() == EXPERIMENTAL_VOCAB_SHA256
    assert tokenizer.identity.special_tokens == {"eos": 256}
    assert s0.identity.special_tokens == {}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "ASCII bytes",
        "Українська мова",
        "tabs\tnewlines\n",
        "naïve — 你好 — 😀",
    ],
)
def test_byte_ids_remain_exact_and_round_trip(text: str) -> None:
    s0 = ByteTokenizer()
    tokenizer = ExperimentalByteEosTokenizer()
    assert tokenizer.encode(text) == s0.encode(text) == list(text.encode("utf-8"))
    encoded = tokenizer.encode(text, add_eos=True)
    assert encoded[:-1] == s0.encode(text)
    assert encoded[-1] == EXPERIMENTAL_EOS_ID
    assert tokenizer.decode(encoded) == text
    assert tokenizer.oov_count(text) == 0


def test_no_bos_pad_unk_or_chat_semantics_exist() -> None:
    tokenizer = ExperimentalByteEosTokenizer()
    contract = contract_payload()
    assert tokenizer.bos_id is None
    assert tokenizer.pad_id is None
    assert tokenizer.unk_id is None
    assert contract["absent_special_tokens"] == ["bos", "pad", "unk"]
    assert contract["instruction_tokens"] == []
    assert contract["system_tokens"] == []
    assert contract["hf_bridge"]["chat_template"] is None
    with pytest.raises(ValueError, match="no BOS"):
        tokenizer.encode("x", add_bos=True)


def test_cross_document_packing_has_explicit_eos_boundaries() -> None:
    tokenizer = ExperimentalByteEosTokenizer()
    records = [
        TextRecord("a", "ab", "train"),
        TextRecord("b", "cd", "train"),
    ]
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=8,
            cross_document=True,
            add_eos=True,
        )
    )
    stream = tokenizer.encode("ab", add_eos=True) + tokenizer.encode("cd", add_eos=True)
    assert _training_pairs(examples) == list(pairwise(stream))
    assert stream == [97, 98, 256, 99, 100, 256]


def test_batch_padding_remains_mask_defined_not_semantic_pad() -> None:
    tokenizer = ExperimentalByteEosTokenizer()
    [example] = list(
        iter_packed_examples(
            [TextRecord("a", "xy", "train")],
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    assert tokenizer.pad_id is None
    assert example.input_ids == (120, 121, 0, 0)
    assert example.attention_mask == (1, 1, 0, 0)
    assert example.loss_mask == (1, 0, 0, 0)
    assert example.labels[2:] == (-100, -100)


class _ImmediateEosBackend:
    eos_token_id = EXPERIMENTAL_EOS_ID
    max_context_tokens = 16

    def __init__(self) -> None:
        self.tokenizer = ExperimentalByteEosTokenizer()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)

    def next_token_logits(self, input_ids):
        logits = [0.0] * EXPERIMENTAL_VOCAB_SIZE
        logits[EXPERIMENTAL_EOS_ID] = 10.0
        return logits


def test_generation_stops_on_experimental_eos() -> None:
    result = generate(_ImmediateEosBackend(), "x", GenerationConfig(max_new_tokens=8))
    assert result.generated_token_ids == (EXPERIMENTAL_EOS_ID,)
    assert result.stop_reason == "eos"
    assert result.text == ""


def test_empty_context_remains_fail_closed_without_untrained_bos() -> None:
    with pytest.raises(ValueError, match="zero tokens"):
        generate(_ImmediateEosBackend(), "")


def test_s0_checkpoint_identity_cannot_be_reinterpreted_as_experimental() -> None:
    tokenizer = ExperimentalByteEosTokenizer()
    with pytest.raises(TokenizerCompatibilityError):
        require_tokenizer_identity(
            tokenizer,
            expected_version=BYTE_TOKENIZER_VERSION,
            expected_config_sha256=BYTE_TOKENIZER_HASH,
            expected_vocab_sha256=BYTE_VOCAB_HASH,
            expected_vocab_size=256,
        )


def test_hf_mapping_sets_only_eos_and_rejects_s0_sized_model() -> None:
    mapping = dict(hf_special_token_ids())
    assert mapping == {
        "bos_token_id": None,
        "eos_token_id": 256,
        "pad_token_id": None,
        "unk_token_id": None,
    }

    s0_spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=16,
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        rope_rotary_dim=8,
    )
    base = llama_config_dict(s0_spec)
    assert base["bos_token_id"] is None
    assert base["eos_token_id"] is None
    assert base["pad_token_id"] is None
    with pytest.raises(ValueError, match="vocab_size=257"):
        apply_to_llama_config(base)

    migrated = dict(base)
    migrated["vocab_size"] = EXPERIMENTAL_VOCAB_SIZE
    mapped = apply_to_llama_config(migrated)
    assert mapped["bos_token_id"] is None
    assert mapped["eos_token_id"] == EXPERIMENTAL_EOS_ID
    assert mapped["pad_token_id"] is None
    assert "chat_template" not in mapped


def test_hf_mapping_refuses_conflicting_or_chat_semantics() -> None:
    with pytest.raises(ValueError, match="conflicting eos_token_id"):
        apply_to_llama_config(
            {
                "vocab_size": EXPERIMENTAL_VOCAB_SIZE,
                "eos_token_id": 7,
            }
        )
    with pytest.raises(ValueError, match="chat_template"):
        apply_to_llama_config(
            {
                "vocab_size": EXPERIMENTAL_VOCAB_SIZE,
                "chat_template": "{{ messages }}",
            }
        )
