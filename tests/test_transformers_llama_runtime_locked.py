from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.inference.transformers_llama import (
    convert_state_dict_to_llama,
    llama_config_dict,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _build_pair():
    assert transformers.__version__ == "5.15.0"
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    torch.manual_seed(1337)
    source = TwelveSixDecoder(stage.model, stage.init).eval()
    config = LlamaConfig(**llama_config_dict(stage.model))
    target = LlamaForCausalLM(config).eval()
    incompatible = target.load_state_dict(
        convert_state_dict_to_llama(stage.model, source.state_dict()),
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    return stage, source, target


def _assert_runtime_parity(source, target, ids: list[int]) -> None:
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        source_logits = source(input_ids).logits
        target_logits = target(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).logits
    torch.testing.assert_close(source_logits, target_logits, atol=1e-5, rtol=1e-5)
    assert torch.equal(
        source_logits[:, -1].argmax(dim=-1),
        target_logits[:, -1].argmax(dim=-1),
    )


def test_locked_transformers_515_executes_12_6_weights_with_logit_parity() -> None:
    stage, source, target = _build_pair()

    assert target.config.bos_token_id is None
    assert target.config.eos_token_id is None
    assert target.config.pad_token_id is None
    assert target.config.vocab_size == stage.model.vocab_size == 256
    assert target.config.max_position_embeddings == stage.model.max_seq_len

    probes = [
        list(b"Hello"),
        list("Привіт".encode("utf-8")),
        list(b"def add(a, b):\n    return a + b\n"),
        [index % 256 for index in range(stage.model.max_seq_len)],
    ]
    for probe in probes:
        _assert_runtime_parity(source, target, probe)


def test_locked_transformers_greedy_generation_matches_first_party() -> None:
    stage, source, target = _build_pair()
    prompt = list("Україна".encode("utf-8"))
    source_ids = list(prompt)
    target_ids = list(prompt)

    for _ in range(8):
        with torch.no_grad():
            source_logits = source(torch.tensor([source_ids], dtype=torch.long)).logits[:, -1]
            target_logits = target(
                input_ids=torch.tensor([target_ids], dtype=torch.long),
                use_cache=False,
                return_dict=True,
            ).logits[:, -1]
        torch.testing.assert_close(source_logits, target_logits, atol=1e-5, rtol=1e-5)
        source_next = int(source_logits.argmax(dim=-1).item())
        target_next = int(target_logits.argmax(dim=-1).item())
        assert source_next == target_next
        source_ids.append(source_next)
        target_ids.append(target_next)
        if len(source_ids) == stage.model.max_seq_len:
            break

    assert source_ids == target_ids


def test_locked_transformers_runtime_is_random_init_weight_conversion_only() -> None:
    stage, source, target = _build_pair()
    converted = convert_state_dict_to_llama(stage.model, source.state_dict())
    target_state = target.state_dict()

    for name, value in converted.items():
        torch.testing.assert_close(target_state[name], value, atol=0, rtol=0)

    # Construction uses LlamaConfig + our converted tensors only: no from_pretrained,
    # AutoModel, model hub identifier, chat template, or special-token insertion.
    assert target.config.architectures == ["LlamaForCausalLM"]
    assert target.config.bos_token_id is None
    assert target.config.eos_token_id is None
    assert target.config.pad_token_id is None
