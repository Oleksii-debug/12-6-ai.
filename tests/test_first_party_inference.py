from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    bind_checkpoint_identity,
    hash_json,
    save_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.cli import main as cli_main
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.openai_compat import completion_response
from twelve_six.inference.parity import compare_backends
from twelve_six.inference.sampling import greedy_token
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"


def _first_training_text() -> str:
    path = ROOT / "data/s0/packaged/train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return str(json.loads(line)["text"])
    raise AssertionError("S0 packaged training fixture is empty")


def _trained_s0_checkpoint(
    tmp_path: Path,
) -> tuple[TwelveSixDecoder, ByteTokenizer, Path, CheckpointIdentity]:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(_first_training_text())[:64]
    seed = 20260824
    trainer_config = TrainerConfig(
        learning_rate=1e-2,
        max_steps=1,
        seed=seed,
        precision="fp32",
        deterministic_algorithms=True,
    )

    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")
    batch_ids = torch.tensor([token_ids], dtype=torch.long)
    metrics = trainer.train_microbatch({"input_ids": batch_ids, "labels": batch_ids})
    assert metrics.optimizer_stepped is True

    model_spec = stage.model.to_dict()
    run_manifest = {
        "schema_version": 1,
        "run_id": "s0-d05-first-party-inference-test",
        "stage": "S0",
        "run_kind": "integrated_training",
        "state": "RUNNING",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": "1" * 40,
            "branch_or_tag": "d05-test-fixture",
            "modelspec_sha256": hash_json(model_spec),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": "s0-tiny-controlled-v1",
        },
        "training": {
            "seed": seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {"name": "AdamW", "lr": trainer_config.learning_rate},
            "scheduler": {"name": trainer_config.scheduler},
            "context_length": stage.model.max_seq_len,
            "global_batch_tokens": len(token_ids),
            "target_steps": 1,
            "target_tokens": len(token_ids),
            "checkpoint_interval_steps": 1,
        },
    }
    identity = bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=model_spec,
        tokenizer_identity=tokenizer.identity.to_dict(),
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
    )
    checkpoint = tmp_path / "trained-s0"
    save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    return model, tokenizer, checkpoint, identity


def _fixture_identity(
    spec: ModelSpec,
    tokenizer: ByteTokenizer,
    *,
    tokenizer_hash: str | None = None,
    context_length: int | None = None,
) -> CheckpointIdentity:
    declared_context = spec.max_seq_len if context_length is None else context_length
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer_hash or tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "run_id": "first-party-compat-fixture",
            "training": {"context_length": declared_context},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=1,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "fixture"},
        scheduler=None,
    )


def test_real_trained_reload_greedy_sampling_logits_and_parity(tmp_path: Path) -> None:
    model, tokenizer, checkpoint, identity = _trained_s0_checkpoint(tmp_path)
    direct = S0TorchInferenceBackend(model, tokenizer)
    reloaded = load_first_party_backend(checkpoint)
    prompt = "12-6"
    input_ids = tokenizer.encode(prompt)

    assert reloaded.next_token_logits(input_ids) == pytest.approx(
        direct.next_token_logits(input_ids),
        rel=0.0,
        abs=0.0,
    )

    greedy = GenerationConfig(max_new_tokens=4, sample=False, seed=17)
    assert generate(reloaded, prompt, greedy) == generate(direct, prompt, greedy)

    sampled = GenerationConfig(
        max_new_tokens=6,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        seed=17,
    )
    first_sample = generate(reloaded, prompt, sampled)
    second_sample = generate(reloaded, prompt, sampled)
    assert first_sample == second_sample
    assert first_sample == generate(direct, prompt, sampled)

    parity = compare_backends(
        direct,
        reloaded,
        ("12-6", "Base"),
        max_new_tokens=4,
        atol=0.0,
        rtol=0.0,
    )
    assert parity.passed is True
    assert parity.steps_compared > 0
    assert parity.max_abs_error == 0.0
    assert parity.max_rel_error == 0.0

    diagnostics = reloaded.diagnostics()
    assert diagnostics["git_sha"] == identity.git_sha
    assert diagnostics["model_spec_sha256"] == hash_json(identity.model_spec)
    assert diagnostics["tokenizer_config_sha256"] == tokenizer.identity.config_sha256
    assert diagnostics["tokenizer_vocab_sha256"] == tokenizer.identity.vocab_sha256
    assert diagnostics["parameter_count"] == 10_140
    assert diagnostics["max_context_tokens"] == 128
    assert reloaded.eos_token_id is None


def test_stop_context_decode_and_token_validation_on_first_party_backend(tmp_path: Path) -> None:
    _, _, checkpoint, _ = _trained_s0_checkpoint(tmp_path)
    backend = load_first_party_backend(checkpoint)
    prompt = "A"
    first_token = greedy_token(backend.next_token_logits(backend.encode(prompt)))

    stopped = generate(
        backend,
        prompt,
        GenerationConfig(max_new_tokens=8, stop_token_ids=(first_token,)),
    )
    assert stopped.generated_token_ids == (first_token,)
    assert stopped.stop_reason == "stop_token"

    first_text = backend.decode([first_token])
    text_stopped = generate(
        backend,
        prompt,
        GenerationConfig(max_new_tokens=8, stop_strings=(first_text,)),
    )
    assert text_stopped.stop_reason == "stop_string"
    assert text_stopped.text == ""

    context_full = generate(backend, "A" * 128, GenerationConfig(max_new_tokens=1))
    assert context_full.generated_token_ids == ()
    assert context_full.stop_reason == "context_limit"

    with pytest.raises(ValueError, match="max_context_tokens"):
        generate(backend, "A" * 129, GenerationConfig(max_new_tokens=1))
    with pytest.raises(ValueError, match="outside vocabulary"):
        backend.next_token_logits([256])
    assert backend.decode([255]) == "\ufffd"


def test_corrupt_tokenizer_vocab_and_context_states_fail_closed(tmp_path: Path) -> None:
    _, tokenizer, checkpoint, _ = _trained_s0_checkpoint(tmp_path)

    corrupt = tmp_path / "corrupt"
    shutil.copytree(checkpoint, corrupt)
    weights = corrupt / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_first_party_backend(corrupt)

    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    wrong_tokenizer = tmp_path / "wrong-tokenizer"
    save_checkpoint(
        wrong_tokenizer,
        model=TwelveSixDecoder(stage.model, stage.init),
        identity=_fixture_identity(stage.model, tokenizer, tokenizer_hash="f" * 64),
    )
    with pytest.raises(CheckpointCompatibilityError, match="tokenizer config"):
        load_first_party_backend(wrong_tokenizer)

    wrong_context = tmp_path / "wrong-context"
    save_checkpoint(
        wrong_context,
        model=TwelveSixDecoder(stage.model, stage.init),
        identity=_fixture_identity(
            stage.model,
            tokenizer,
            context_length=stage.model.max_seq_len + 1,
        ),
    )
    with pytest.raises(CheckpointCompatibilityError, match="context_length"):
        load_first_party_backend(wrong_context)

    bad_spec_payload = stage.model.to_dict()
    bad_spec_payload["vocab_size"] = 255
    bad_spec = ModelSpec.from_dict(bad_spec_payload)
    wrong_vocab = tmp_path / "wrong-vocab"
    save_checkpoint(
        wrong_vocab,
        model=TwelveSixDecoder(bad_spec, stage.init),
        identity=_fixture_identity(bad_spec, tokenizer),
    )
    with pytest.raises(CheckpointCompatibilityError, match="vocabulary size"):
        load_first_party_backend(wrong_vocab)


def test_cli_defaults_to_first_party_and_supports_json_and_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, tokenizer, checkpoint, _ = _trained_s0_checkpoint(tmp_path)
    code = cli_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "12-6",
            "--greedy",
            "--max-new-tokens",
            "2",
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["prompt_token_ids"] == tokenizer.encode("12-6")
    assert payload["backend"]["backend"] == "first_party_torch"
    assert "backend: kind=first_party_torch" in captured.err
    assert "generation: mode=greedy" in captured.err
    assert "\x1b[" not in captured.out + captured.err

    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin prompt"))
    code = cli_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--max-new-tokens",
            "1",
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["prompt_token_ids"] == tokenizer.encode("stdin prompt")


def test_openai_completion_handoff_preserves_raw_base_semantics(tmp_path: Path) -> None:
    _, _, checkpoint, _ = _trained_s0_checkpoint(tmp_path)
    backend = load_first_party_backend(checkpoint)
    payload = {
        "model": "12-6-base",
        "prompt": "12-6",
        "max_tokens": 3,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 19,
    }
    response = completion_response(
        backend,
        payload,
        response_id="cmpl-test",
        created=123,
        model_name="12-6-base-s0",
    )
    direct = generate(
        backend,
        "12-6",
        GenerationConfig(max_new_tokens=3, sample=False, seed=19),
    )
    choices = response["choices"]
    usage = response["usage"]
    assert isinstance(choices, list)
    assert isinstance(usage, dict)
    assert choices[0]["text"] == direct.text
    assert usage["prompt_tokens"] == len(direct.prompt_token_ids)
    assert usage["completion_tokens"] == len(direct.generated_token_ids)
    assert response["model"] == "12-6-base-s0"

    with pytest.raises(ValueError, match="chat/messages"):
        completion_response(backend, {"messages": [], "prompt": "ignored"})
    with pytest.raises(ValueError, match="stream=true"):
        completion_response(backend, {"prompt": "x", "stream": True})
