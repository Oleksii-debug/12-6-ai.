from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint
from twelve_six.inference.llama_runtime_export import (
    RUNTIME_CONFIG_NAME,
    RUNTIME_PROVENANCE_NAME,
    RUNTIME_WEIGHTS_NAME,
    materialize_standard_llama_directory,
    verify_standard_llama_directory,
)
from twelve_six.inference.transformers_llama import llama_config_dict
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
        hf_config=llama_config_dict(spec),
    )
    target = materialize_vllm_llama_directory(
        source_export,
        tmp_path / "vllm-llama",
    )
    return spec, checkpoint, source_export, target


def test_vllm_materialization_is_exact_incumbent_standard_llama_export(tmp_path: Path):
    spec, _checkpoint, source_export, target = _materialized_fixture(tmp_path)

    binding = verify_vllm_llama_directory(target)
    runtime = verify_standard_llama_directory(target)
    assert binding["checkpoint_id"] == runtime["source_checkpoint_id"]
    assert binding["source_model_spec"] == spec.to_dict()
    assert binding["model_spec_sha256"] == spec.identity_sha256()
    assert binding["parameter_count"] == spec.parameter_count()
    assert binding["execution_contract"] == {
        "vllm_implementation": "BUILTIN_LLAMA",
        "skip_tokenizer_init": True,
        "prompt_input": "TOKEN_IDS",
        "trust_remote_code": False,
    }
    assert binding["runtime_provenance_file"] == RUNTIME_PROVENANCE_NAME
    assert {path.name for path in target.iterdir()} == {
        RUNTIME_CONFIG_NAME,
        RUNTIME_WEIGHTS_NAME,
        RUNTIME_PROVENANCE_NAME,
    }
    config = json.loads((target / RUNTIME_CONFIG_NAME).read_text(encoding="utf-8"))
    assert config == llama_config_dict(spec)

    direct = materialize_standard_llama_directory(
        source_export,
        tmp_path / "incumbent-direct",
    )
    for name in (RUNTIME_CONFIG_NAME, RUNTIME_WEIGHTS_NAME, RUNTIME_PROVENANCE_NAME):
        assert (target / name).read_bytes() == (direct / name).read_bytes()


def test_verifier_rejects_incumbent_runtime_weight_tamper(tmp_path: Path):
    _spec_value, _checkpoint, _source_export, target = _materialized_fixture(tmp_path)
    weights = target / RUNTIME_WEIGHTS_NAME
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(VllmRuntimeError, match="runtime export weights hash mismatch"):
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
