from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from tools.prepare_vllm_learned_checkpoint import prepare
from tools.validate_vllm_learned_parity import _collect_raw_trace


def _learned_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=16,
        d_ff=64,
        rope_rotary_dim=16,
    )


def _learned_checkpoint(tmp_path: Path) -> tuple[Path, CheckpointIdentity]:
    spec = _learned_spec()
    tokenizer = ByteTokenizer()
    identity = CheckpointIdentity(
        git_sha="d" * 40,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": spec.max_seq_len},
        },
        seed=1337,
        precision="fp32",
        step=3,
        tokens_seen=96,
        optimizer={"name": "AdamW"},
        scheduler=None,
    )
    torch.manual_seed(1337)
    checkpoint = tmp_path / "source"
    save_checkpoint(checkpoint, model=TwelveSixDecoder(spec), identity=identity)
    return checkpoint, identity


def test_prepare_learned_checkpoint_binds_existing_export_and_tokenizer(tmp_path: Path):
    checkpoint, identity = _learned_checkpoint(tmp_path)
    tokenizer = ByteTokenizer()
    manifest_id = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )["checkpoint_id"]
    args = SimpleNamespace(
        checkpoint=checkpoint,
        output_root=tmp_path / "prepared",
        expected_checkpoint_id=manifest_id,
        expected_model_spec_sha256=_learned_spec().identity_sha256(),
        expected_tokenizer_config_sha256=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_sha256=tokenizer.identity.vocab_sha256,
        source_repository="Oleksii-debug/12-6-ai.",
        source_artifact_id=123,
        source_artifact_name="unit-learned",
        source_artifact_digest="sha256:" + "a" * 64,
        source_artifact_head_sha="d" * 40,
    )

    evidence = prepare(args)

    assert evidence["learned_checkpoint"]["checkpoint_id"] == manifest_id
    assert evidence["learned_checkpoint"]["step"] == identity.step
    assert evidence["learned_checkpoint"]["tokens_seen"] == identity.tokens_seen
    assert evidence["standard_llama_export"]["qk_rope_basis_conversion"] == "INCUMBENT_D07_EXACT"
    assert evidence["tokenizer"]["owner"] == "12-6_CANONICAL_OUTSIDE_VLLM"
    assert evidence["tokenizer"]["chat_template_used"] is False
    assert evidence["execution_contract"]["vllm_implementation"] == "BUILTIN_LLAMA"
    assert "validate_vllm_learned_parity.py" in evidence["gpu_parity_command"]
    assert (tmp_path / "prepared" / "source-checkpoint" / "weights.safetensors").is_file()
    assert (tmp_path / "prepared" / "vllm-model" / "model.safetensors").is_file()


class _TraceBackend:
    eos_token_id = None
    max_context_tokens = 5

    def encode(self, text: str) -> list[int]:
        assert text == "probe"
        return [1, 2]

    def decode(self, token_ids) -> str:
        return ",".join(str(value) for value in token_ids)

    def next_token_logits(self, input_ids):
        winner = len(input_ids) % 4
        values = [0.0, 0.0, 0.0, 0.0]
        values[winner] = 10.0
        return values


def test_raw_trace_retains_exact_logits_and_greedy_sequence():
    reference = _TraceBackend()
    candidate = _TraceBackend()

    trace = _collect_raw_trace(reference, candidate, ["probe"], max_new_tokens=8)

    assert len(trace) == 1
    item = trace[0]
    assert item["prompt_token_ids"] == [1, 2]
    assert item["reference_greedy_token_sequence"] == [2, 3, 0]
    assert item["vllm_greedy_token_sequence"] == [2, 3, 0]
    assert item["reference_decoded_continuation"] == "2,3,0"
    assert item["vllm_decoded_continuation"] == "2,3,0"
    assert item["stopped_at_context_boundary"] is True
    assert len(item["steps"]) == 3
    for step in item["steps"]:
        assert step["reference_raw_logits"] == step["vllm_raw_logits"]
        assert len(step["reference_raw_logits"]) == 4
        assert len(step["reference_raw_logits_sha256"]) == 64
        assert len(step["vllm_raw_logits_sha256"]) == 64
