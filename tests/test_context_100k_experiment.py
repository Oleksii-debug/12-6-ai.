from __future__ import annotations

import json
from pathlib import Path

import torch

from twelve_six.context_100k_candidate import (
    TARGET_OPTIMIZED_TOKENS,
    _state_digest,
    research_100k_spec,
    shared_trainer_config,
)
from twelve_six.context_scaling import ContextPackingSpec
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.packing import DEFAULT_SEQUENCE_LENGTH, PACKING_VERSION
from twelve_six.tokenization import BYTE_TOKENIZER_HASH


def test_model17_contexts_keep_parameter_geometry_fixed() -> None:
    c128 = research_100k_spec(128)
    c256 = research_100k_spec(256)
    assert c128.parameter_count() == c256.parameter_count() == 95_568
    assert c128.max_seq_len == 128
    assert c256.max_seq_len == 256
    left = c128.to_dict(); right = c256.to_dict()
    left.pop("max_seq_len"); right.pop("max_seq_len")
    assert left == right
    assert c128.identity_sha256() != c256.identity_sha256()


def test_model17_initial_trainable_tensors_are_identical() -> None:
    torch.manual_seed(1337)
    model128 = TwelveSixDecoder(research_100k_spec(128), InitSpec())
    digest128 = _state_digest(model128)
    torch.manual_seed(1337)
    model256 = TwelveSixDecoder(research_100k_spec(256), InitSpec())
    assert digest128 == _state_digest(model256)


def test_model17_packing_identity_is_distinct_without_mutating_s0() -> None:
    p128 = ContextPackingSpec(sequence_length=128)
    p256 = ContextPackingSpec(sequence_length=256)
    assert p128.identity_sha256(tokenizer_config_sha256=BYTE_TOKENIZER_HASH) != p256.identity_sha256(tokenizer_config_sha256=BYTE_TOKENIZER_HASH)
    assert DEFAULT_SEQUENCE_LENGTH == 128
    assert PACKING_VERSION == "s0-byte-pack-v1"


def test_model17_optimizer_and_config_contract() -> None:
    cfg = shared_trainer_config()
    assert cfg.learning_rate == 3e-4
    assert cfg.betas == (0.9, 0.95)
    assert cfg.weight_decay == 0.0
    assert cfg.scheduler == "constant"
    assert cfg.precision == "fp32"
    assert TARGET_OPTIMIZED_TOKENS == 32_768

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs/context/model17_100k_128_vs_256.experimental.json").read_text(encoding="utf-8"))
    assert payload["research_vehicle"]["expected_parameters"] == 95_568
    assert payload["controls"]["optimized_causal_tokens_per_condition"] == 32_768
    assert payload["controls"]["canonical_s0_packing_identity_modified"] is False
