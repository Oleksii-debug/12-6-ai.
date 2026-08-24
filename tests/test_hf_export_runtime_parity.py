from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.checkpoint.hf_export import export_hf_directory
from twelve_six.inference.hf_export_parity import (
    assert_hf_export_parity,
    collect_hf_export_parity_evidence,
    load_hf_style_export_backend,
    main,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]


def _trained_s0_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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
    token_ids = tokenizer.encode("12-6 export parity training fixture")
    batch_ids = torch.tensor([token_ids], dtype=torch.long)
    for _ in range(2):
        metrics = trainer.train_microbatch({"input_ids": batch_ids, "labels": batch_ids})
        assert metrics.optimizer_stepped is True
    assert trainer.optimizer_step == 2

    identity = CheckpointIdentity(
        git_sha="a" * 40,
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
        hf_config={
            "model_type": "twelve_six",
            "architectures": ["TwelveSixDecoder"],
            "vocab_size": stage.model.vocab_size,
            "max_position_embeddings": stage.model.max_seq_len,
            "compatibility_status": "12-6-native-only-transformers-not-claimed",
        },
    )
    return checkpoint, export


def test_real_trained_s0_export_has_zero_tolerance_logit_token_decode_parity(
    tmp_path: Path,
):
    checkpoint, export = _trained_s0_checkpoint(tmp_path)
    evidence = assert_hf_export_parity(
        checkpoint,
        export,
        ("12-6", "Україна", "code: x = 1"),
        max_new_tokens=4,
    )

    assert evidence["passed"] is True
    assert evidence["reference_weights_sha256"] == evidence["candidate_weights_sha256"]
    assert evidence["parity"]["passed"] is True
    assert evidence["parity"]["steps_compared"] > 0
    assert evidence["parity"]["max_abs_error"] == 0.0
    assert evidence["parity"]["max_rel_error"] == 0.0
    assert evidence["parity"]["atol"] == 0.0
    assert evidence["parity"]["rtol"] == 0.0
    assert evidence["compatibility"] == {
        "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
        "runtime_logit_token_decode_parity": "PASS",
        "tolerance": "EXACT_ZERO",
        "transformers_architecture": "NOT_CLAIMED",
    }
    assert evidence["promotion_authority"] is False
    assert len(evidence["evidence_sha256"]) == 64
    assert len(evidence["prompt_sha256"]) == 3
    assert "12-6" not in json.dumps(evidence["prompt_sha256"])


def test_export_backend_uses_verified_snapshot_and_rejects_candidate_tamper(tmp_path: Path):
    checkpoint, export = _trained_s0_checkpoint(tmp_path)
    backend = load_hf_style_export_backend(checkpoint, export)
    assert backend.diagnostics()["parameter_count"] == 10140
    assert backend.diagnostics()["vocab_size"] == 256

    weights = export / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError):
        load_hf_style_export_backend(checkpoint, export)


def test_export_parity_fails_closed_on_attestation_or_config_drift(tmp_path: Path):
    checkpoint, export = _trained_s0_checkpoint(tmp_path)
    config = export / "config.json"
    config.write_text('{"model_type":"different"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError):
        collect_hf_export_parity_evidence(checkpoint, export, ["12-6"])


def test_export_parity_cli_emits_machine_json_without_prompt_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    checkpoint, export = _trained_s0_checkpoint(tmp_path)
    output = tmp_path / "parity-evidence.json"
    code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--export",
            str(export),
            "--prompt",
            "private-ish prompt fixture",
            "--max-new-tokens",
            "2",
            "--output",
            str(output),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["passed"] is True
    assert "private-ish prompt fixture" not in captured.out
    assert "private-ish prompt fixture" not in output.read_text(encoding="utf-8")
