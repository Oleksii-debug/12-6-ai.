from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.inference import first_party
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _identity(
    *,
    stage: object,
    tokenizer: ByteTokenizer,
    git_hex: str,
    dataset_hex: str,
    run_hex: str,
    step: int,
    tokens_seen: int,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=git_hex * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_hex * 64,
        run_manifest_hash=run_hex * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": stage.model.max_seq_len},
        },
        seed=17,
        precision="fp32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _fill_model(model: TwelveSixDecoder, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def _state(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def test_first_party_loader_keeps_identity_and_weights_on_one_verified_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    checkpoint = tmp_path / "checkpoint"
    replacement = tmp_path / "replacement"

    model_a = TwelveSixDecoder(stage.model, stage.init)
    model_b = TwelveSixDecoder(stage.model, stage.init)
    _fill_model(model_a, 0.03125)
    _fill_model(model_b, -0.0625)
    state_a = _state(model_a)
    state_b = _state(model_b)

    manifest_a = save_checkpoint(
        checkpoint,
        model=model_a,
        identity=_identity(
            stage=stage,
            tokenizer=tokenizer,
            git_hex="a",
            dataset_hex="d",
            run_hex="e",
            step=3,
            tokens_seen=96,
        ),
    )
    manifest_b = save_checkpoint(
        replacement,
        model=model_b,
        identity=_identity(
            stage=stage,
            tokenizer=tokenizer,
            git_hex="b",
            dataset_hex="c",
            run_hex="1",
            step=9,
            tokens_seen=288,
        ),
    )
    assert manifest_a["checkpoint_id"] != manifest_b["checkpoint_id"]

    real_prepare = first_party.prepare_checkpoint_load
    real_load_verified = first_party.load_verified_checkpoint
    prepared: list[object] = []
    consumed: list[object] = []

    def capture_prepare(path: Path):
        verified = real_prepare(path)
        prepared.append(verified)
        return verified

    def swap_path_then_load(verified, **kwargs):
        consumed.append(verified)
        shutil.rmtree(checkpoint)
        shutil.copytree(replacement, checkpoint)
        return real_load_verified(verified, **kwargs)

    monkeypatch.setattr(first_party, "prepare_checkpoint_load", capture_prepare)
    monkeypatch.setattr(first_party, "load_verified_checkpoint", swap_path_then_load)

    backend = first_party.load_first_party_backend(checkpoint)

    assert len(prepared) == 1
    assert len(consumed) == 1
    assert consumed[0] is prepared[0]

    loaded_state = _state(backend.model)
    assert loaded_state.keys() == state_a.keys() == state_b.keys()
    for name in loaded_state:
        torch.testing.assert_close(loaded_state[name], state_a[name], rtol=0, atol=0)
    assert any(not torch.equal(loaded_state[name], state_b[name]) for name in loaded_state)

    diagnostics = backend.diagnostics()
    assert diagnostics["checkpoint_id"] == manifest_a["checkpoint_id"]
    assert diagnostics["git_sha"] == "a" * 40
    assert diagnostics["dataset_manifest_sha256"] == "d" * 64
    assert diagnostics["run_manifest_sha256"] == "e" * 64
    assert diagnostics["step"] == 3
    assert diagnostics["tokens_seen"] == 96

    # The source pathname now resolves to a different fully valid checkpoint.
    # The loaded backend must nevertheless remain bound to the already verified A bytes.
    disk_manifest = verify_checkpoint(checkpoint)
    assert disk_manifest["checkpoint_id"] == manifest_b["checkpoint_id"]
    assert disk_manifest["identity"]["git_sha"] == "b" * 40


def test_first_party_loader_rejects_external_model_spec_mismatch_before_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    checkpoint = tmp_path / "checkpoint"
    model = TwelveSixDecoder(stage.model, stage.init)
    save_checkpoint(
        checkpoint,
        model=model,
        identity=_identity(
            stage=stage,
            tokenizer=tokenizer,
            git_hex="a",
            dataset_hex="d",
            run_hex="e",
            step=3,
            tokens_seen=96,
        ),
    )

    def forbid_model_allocation(*args, **kwargs):
        raise AssertionError("model allocation must not occur after external ModelSpec mismatch")

    monkeypatch.setattr(first_party, "TwelveSixDecoder", forbid_model_allocation)
    with pytest.raises(
        CheckpointCompatibilityError,
        match="externally expected identity",
    ):
        first_party.load_first_party_backend(
            checkpoint,
            expected_model_spec_sha256="0" * 64,
        )
