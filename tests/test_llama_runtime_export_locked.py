from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint
from twelve_six.inference.llama_runtime_export import (
    materialize_standard_llama_directory,
    verify_standard_llama_directory,
)
from twelve_six.inference.transformers_llama import llama_config_dict
from twelve_six.model import TwelveSixDecoder, load_stage_config

transformers = pytest.importorskip("transformers")
from transformers import LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def _identity(stage) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 0},
        seed=1337,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
    )


def test_standard_llama_runtime_export_loads_via_from_pretrained(tmp_path: Path) -> None:
    assert transformers.__version__ == "5.15.0"
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    torch.manual_seed(1337)
    source = TwelveSixDecoder(stage.model, stage.init).eval()

    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=source, identity=_identity(stage))
    canonical_export = export_hf_directory(
        checkpoint,
        tmp_path / "canonical-export",
        hf_config=llama_config_dict(stage.model),
    )
    runtime_dir = materialize_standard_llama_directory(
        canonical_export,
        tmp_path / "llama-runtime",
    )
    provenance = verify_standard_llama_directory(runtime_dir)

    assert provenance["source_checkpoint_id"]
    assert provenance["model_spec_sha256"] == stage.model.identity_sha256()
    assert provenance["parameter_count"] == 10_140
    assert provenance["target_architecture"] == "LlamaForCausalLM"
    assert provenance["foreign_pretrained_weights"] is False
    assert provenance["model_downloaded"] is False

    target = LlamaForCausalLM.from_pretrained(
        runtime_dir,
        local_files_only=True,
    ).eval()
    for ids in (
        list(b"Hello"),
        list("Привіт".encode()),
        list(b"def f(x): return x + 1"),
    ):
        tokens = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            reference = source(tokens).logits
            candidate = target(input_ids=tokens, use_cache=False, return_dict=True).logits
        torch.testing.assert_close(reference, candidate, atol=1e-5, rtol=1e-5)
        assert torch.equal(
            reference[:, -1].argmax(dim=-1),
            candidate[:, -1].argmax(dim=-1),
        )


def test_standard_llama_runtime_export_is_immutable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    source = TwelveSixDecoder(stage.model, stage.init)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=source, identity=_identity(stage))
    canonical_export = export_hf_directory(
        checkpoint,
        tmp_path / "canonical-export",
        hf_config=llama_config_dict(stage.model),
    )
    runtime_dir = materialize_standard_llama_directory(
        canonical_export,
        tmp_path / "llama-runtime",
    )

    with pytest.raises(FileExistsError):
        materialize_standard_llama_directory(canonical_export, runtime_dir)

    weights = runtime_dir / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="weights hash mismatch"):
        verify_standard_llama_directory(runtime_dir)
