from __future__ import annotations

import copy

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.tokenization.base_contract import (
    assert_checkpoint_compatible,
    assert_runtime_contract,
    deterministic_artifact_proof,
    hf_transformers_token_mapping,
    load_research_base_token_contract,
)


def _compatible_manifest() -> dict:
    contract = load_research_base_token_contract()
    return {
        "identity": {
            "tokenizer_hash": contract["checkpoint_compatibility"][
                "required_tokenizer_config_sha256"
            ],
            "tokenizer_vocab_hash": contract["checkpoint_compatibility"][
                "required_tokenizer_vocab_sha256"
            ],
            "model_spec": {"vocab_size": 256},
            "training_config": {
                "data": {
                    "packing_version": contract["checkpoint_compatibility"][
                        "required_packing_version_for_training_resume"
                    ]
                }
            },
        }
    }


def test_contract_runtime_and_artifact_identity_are_stable() -> None:
    contract = load_research_base_token_contract()
    assert_runtime_contract(contract=contract)
    a = deterministic_artifact_proof()
    b = deterministic_artifact_proof()
    assert a == b
    assert a["byte_identical_reloads"] is True


def test_no_special_tokens_and_hf_mapping_are_explicit() -> None:
    contract = load_research_base_token_contract()
    assert contract["ordinary_vocabulary"]["id_range"] == [0, 255]
    assert contract["special_tokens"] == {
        "bos_token_id": None,
        "chat_token_ids": [],
        "eod_token_id": None,
        "eos_token_id": None,
        "instruction_token_ids": [],
        "pad_token_id": None,
        "system_token_ids": [],
        "unk_token_id": None,
    }
    mapping = hf_transformers_token_mapping(contract)
    assert mapping["vocab_size"] == 256
    assert mapping["bos_token_id"] is None
    assert mapping["eos_token_id"] is None
    assert mapping["pad_token_id"] is None
    assert mapping["unk_token_id"] is None
    assert mapping["added_tokens"] == []
    assert mapping["special_tokens_map"] == {}


def test_byte_ids_roundtrip_and_empty_generation_seed_is_absent() -> None:
    tok = ByteTokenizer()
    text = "UA Україна / EN English / code: x += 1\n"
    ids = tok.encode(text)
    assert ids == list(text.encode("utf-8"))
    assert tok.decode(ids) == text
    assert tok.encode("") == []
    assert tok.bos_id is None
    assert tok.eos_id is None
    assert tok.pad_id is None
    with pytest.raises(ValueError, match="no EOS"):
        tok.encode(text, add_eos=True)


def test_document_isolation_and_mask_only_padding() -> None:
    tok = ByteTokenizer()
    records = [
        TextRecord("a", "ab", "train"),
        TextRecord("b", "cd", "train"),
    ]
    examples = list(
        iter_packed_examples(
            records,
            tok,
            expected_split="train",
            sequence_length=8,
            cross_document=False,
        )
    )
    assert len(examples) == 2
    assert all(len(example.record_ids) == 1 for example in examples)
    for example in examples:
        for token_id, attention, label in zip(
            example.input_ids, example.attention_mask, example.labels
        ):
            if attention == 0:
                assert token_id == 0
                assert label == -100

    with pytest.raises(ValueError, match="requires an explicit EOS"):
        list(
            iter_packed_examples(
                records,
                tok,
                expected_split="train",
                sequence_length=8,
                cross_document=True,
            )
        )


def test_checkpoint_contract_accepts_only_exact_byte_noeos_lineage() -> None:
    compatible = _compatible_manifest()
    assert_checkpoint_compatible(compatible)

    wrong_tokenizer = copy.deepcopy(compatible)
    wrong_tokenizer["identity"]["tokenizer_hash"] = "0" * 64
    with pytest.raises(CheckpointCompatibilityError, match="tokenizer_hash"):
        assert_checkpoint_compatible(wrong_tokenizer)

    eos_sized = copy.deepcopy(compatible)
    eos_sized["identity"]["model_spec"]["vocab_size"] = 257
    with pytest.raises(CheckpointCompatibilityError, match="vocab_size"):
        assert_checkpoint_compatible(eos_sized)

    wrong_packing = copy.deepcopy(compatible)
    wrong_packing["identity"]["training_config"]["data"]["packing_version"] = "other"
    with pytest.raises(CheckpointCompatibilityError, match="packing_version"):
        assert_checkpoint_compatible(wrong_packing)
