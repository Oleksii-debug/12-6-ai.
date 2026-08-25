from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint
from twelve_six.inference.transformers_llama import convert_state_dict_to_llama, llama_config_dict
from twelve_six.inference.transformers_llama_runtime import (
    TRANSFORMERS_VERSION,
    assert_transformers_llama_parity,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="dedicated hash-locked Transformers runtime is not installed",
)


def _trained_s0_export(tmp_path: Path) -> tuple[Path, Path]:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    seed = 20260825
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer_config = TrainerConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        max_steps=2,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    trainer = Trainer(model, trainer_config, device="cpu")
    token_ids = tokenizer.encode("12-6 Transformers runtime fixture")
    batch = torch.tensor([token_ids], dtype=torch.long)
    for _ in range(2):
        metrics = trainer.train_microbatch({"input_ids": batch, "labels": batch})
        assert metrics.optimizer_stepped is True

    source_sha = os.environ.get("RUNTIME24_SOURCE_SHA", "a" * 40)
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise ValueError("RUNTIME24_SOURCE_SHA must be a lowercase 40-character Git SHA")
    identity = CheckpointIdentity(
        git_sha=source_sha,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": stage.model.max_seq_len},
        },
        seed=seed,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "learning_rate": trainer_config.learning_rate},
        scheduler={"name": "constant"},
    )
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=model, identity=identity)
    export = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config=llama_config_dict(stage.model),
    )
    return checkpoint, export


def test_real_trained_export_executes_in_transformers_with_full_parity(tmp_path: Path) -> None:
    checkpoint, export = _trained_s0_export(tmp_path)
    evidence = assert_transformers_llama_parity(
        checkpoint,
        export,
        ("Hello, 12-6.", "Привіт, Україно.", "def f(x): return x + 1"),
        max_new_tokens=4,
    )

    assert evidence["passed"] is True
    assert evidence["transformers_version"] == TRANSFORMERS_VERSION
    assert evidence["architecture"] == "LlamaForCausalLM"
    assert evidence["source_git_sha"] == os.environ.get("RUNTIME24_SOURCE_SHA", "a" * 40)
    assert evidence["tensor_mapping"]["strict_load"] == "PASS"
    assert evidence["rope"]["exact"] is True
    assert evidence["rope"]["max_abs_error"] == 0.0
    assert evidence["logits"]["max_abs_error"] <= evidence["logits"]["atol"] + (
        evidence["logits"]["rtol"] * 1.0
    )
    assert evidence["logits"]["steps_compared"] >= 4
    assert all(row["tokenizer_exact_utf8_bytes"] for row in evidence["prompts"])
    assert all(row["greedy_exact"] for row in evidence["prompts"])
    assert all(row["decode_exact"] for row in evidence["prompts"])
    assert evidence["context"]["boundary_logit_parity"] is True
    assert evidence["context"]["boundary_generation_emits_zero_tokens"] is True
    assert evidence["context"]["over_context_reference_rejected"] is True
    assert evidence["context"]["over_context_transformers_rejected"] is True
    assert evidence["truth_boundary"]["pretrained_weights_used"] is False
    assert evidence["truth_boundary"]["pretrained_model_api_used"] is False
    print(f"RUNTIME24_EVIDENCE_SHA={evidence['evidence_sha256']}")
    print(f"RUNTIME24_WEIGHTS_SHA={evidence['executed_weights_sha256']}")
    print(f"RUNTIME24_CONFIG_SHA={evidence['executed_config_sha256']}")
    print(f"RUNTIME24_MAX_ABS={evidence['logits']['max_abs_error']:.12g}")
    print(f"RUNTIME24_MAX_REL={evidence['logits']['max_rel_error']:.12g}")


def test_transformers_mapping_is_not_s0_shape_specific() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415

    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=128,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        d_ff=256,
        rope_rotary_dim=32,
    )
    torch.manual_seed(99)
    source = TwelveSixDecoder(spec).eval()
    converted = convert_state_dict_to_llama(spec, source.state_dict())
    target = LlamaForCausalLM(LlamaConfig.from_dict(llama_config_dict(spec))).eval()
    incompatible = target.load_state_dict(converted, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert target.config.num_hidden_layers == 3
    assert target.config.num_attention_heads == 4
    assert target.config.num_key_value_heads == 2
    assert sum(parameter.numel() for parameter in target.parameters()) == spec.parameter_count()

    ids = torch.tensor([[index * 17 % spec.vocab_size for index in range(31)]], dtype=torch.long)
    with torch.no_grad():
        reference = source(ids).logits
        candidate = target(input_ids=ids, use_cache=False, return_dict=True).logits
    torch.testing.assert_close(reference, candidate, atol=1e-5, rtol=1e-5)
    assert torch.equal(reference[:, -1].argmax(dim=-1), candidate[:, -1].argmax(dim=-1))
    print(f"RUNTIME24_GQA_MAX_ABS={(reference - candidate).abs().max().item():.12g}")
