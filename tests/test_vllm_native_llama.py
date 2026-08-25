from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load as load_safetensors_bytes

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint
from twelve_six.inference.transformers_llama import (
    convert_state_dict_to_llama,
    llama_config_dict,
)
from twelve_six.inference.vllm_native_llama import (
    VllmNativeLlamaBackend,
    VllmRuntimeError,
    materialize_vllm_llama_directory,
    verify_vllm_llama_directory,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


def _spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=128,
        d_model=20,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=10,
        d_ff=56,
        rope_rotary_dim=10,
    )


def _checkpoint_identity(spec: ModelSpec) -> CheckpointIdentity:
    tokenizer = ByteTokenizer()
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "training": {"context_length": spec.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=7,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
    )


def _materialized_fixture(tmp_path: Path):
    spec = _spec()
    torch.manual_seed(7)
    model = TwelveSixDecoder(spec)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=model,
        identity=_checkpoint_identity(spec),
    )
    source_export = export_hf_directory(
        checkpoint,
        tmp_path / "hf-source",
        hf_config={"model_type": "twelve_six"},
    )
    target = materialize_vllm_llama_directory(
        source_export,
        tmp_path / "vllm-llama",
    )
    return spec, checkpoint, source_export, target


def test_materializes_exact_verified_export_bytes_into_standard_llama(tmp_path: Path):
    spec, _checkpoint, source_export, target = _materialized_fixture(tmp_path)

    provenance = verify_vllm_llama_directory(target)
    assert provenance["source_model_spec"] == spec.to_dict()
    assert provenance["execution_contract"] == {
        "vllm_implementation": "BUILTIN_LLAMA",
        "skip_tokenizer_init": True,
        "prompt_input": "TOKEN_IDS",
        "tokenizer_owner": "12-6.s0-byte-v1",
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
    }
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert config == llama_config_dict(spec)
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"

    source_state = load_safetensors_bytes((source_export / "model.safetensors").read_bytes())
    expected = convert_state_dict_to_llama(spec, source_state)
    actual = load_safetensors_bytes((target / "model.safetensors").read_bytes())
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name

    assert not torch.equal(
        source_state["blocks.0.attn.q_proj.weight"],
        actual["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        source_state["blocks.0.attn.v_proj.weight"],
        actual["model.layers.0.self_attn.v_proj.weight"],
    )


def test_materialization_is_deterministic_for_same_verified_export(tmp_path: Path):
    _spec_value, _checkpoint, source_export, target = _materialized_fixture(tmp_path)
    second = materialize_vllm_llama_directory(source_export, tmp_path / "vllm-llama-2")

    for name in ("config.json", "model.safetensors", "12-6-vllm-runtime.json"):
        assert (target / name).read_bytes() == (second / name).read_bytes()


def test_verifier_rejects_target_weight_tamper(tmp_path: Path):
    _spec_value, _checkpoint, _source_export, target = _materialized_fixture(tmp_path)
    weights = target / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(VllmRuntimeError, match="weights hash mismatch"):
        verify_vllm_llama_directory(target)


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEngine:
    def __init__(self, vocab_size: int, sampled_token: int):
        self.vocab_size = vocab_size
        self.sampled_token = sampled_token
        self.last_prompts = None
        self.last_params = None

    def generate(self, prompts, params, use_tqdm=False):
        assert use_tqdm is False
        self.last_prompts = prompts
        self.last_params = params
        raw = {
            token_id: SimpleNamespace(logprob=float(token_id))
            for token_id in range(self.vocab_size)
        }
        output = SimpleNamespace(
            token_ids=[self.sampled_token],
            logprobs=[raw],
        )
        return [SimpleNamespace(outputs=[output])]


def test_backend_uses_token_ids_and_returns_full_raw_logit_vector():
    spec = _spec()
    backend = VllmNativeLlamaBackend.__new__(VllmNativeLlamaBackend)
    backend.spec = spec
    backend.tokenizer = ByteTokenizer()
    backend.max_context_tokens = spec.max_seq_len
    backend._SamplingParams = _SamplingParams
    backend._engine = _FakeEngine(spec.vocab_size, spec.vocab_size - 1)

    logits = backend.next_token_logits([65, 66, 67])

    assert len(logits) == spec.vocab_size
    assert logits[-1] == float(spec.vocab_size - 1)
    assert backend._engine.last_prompts == [{"prompt_token_ids": [65, 66, 67]}]
    assert backend._engine.last_params.kwargs == {
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": -1,
        "detokenize": False,
        "ignore_eos": True,
    }


def test_backend_rejects_over_context_before_calling_vllm():
    spec = _spec()
    backend = VllmNativeLlamaBackend.__new__(VllmNativeLlamaBackend)
    backend.spec = spec
    backend.tokenizer = ByteTokenizer()
    backend.max_context_tokens = spec.max_seq_len
    backend._SamplingParams = _SamplingParams
    backend._engine = _FakeEngine(spec.vocab_size, spec.vocab_size - 1)

    with pytest.raises(ValueError, match="exceed model context"):
        backend.next_token_logits([0] * (spec.max_seq_len + 1))
    assert backend._engine.last_prompts is None
