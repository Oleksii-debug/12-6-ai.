"""Integrity-valid incomplete-RNG rejection for the D05/D02 restore boundary.

The independent regression cases were adopted into PR #2628 alongside the
canonical same-lineage fail-closed repair; execution still requires CI.
No real corpus, tokenizer fitting or trained-model evidence is used.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    core,
    progress_trainer,
    trainer_adapter,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


@pytest.fixture
def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d02-incomplete-rng-regression", "width": 3},
        parameter_count=12,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _seal_checkpoint(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: CheckpointIdentity,
    missing: str | None,
) -> tuple[dict[str, Any], TrainerConfig]:
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    captured = core.capture_rng_state()
    deliberately_incomplete = dict(captured)
    if missing == "cuda":
        deliberately_incomplete["torch"] = dict(captured["torch"])
        deliberately_incomplete["torch"].pop("cuda")
    elif missing == "warn_only":
        deliberately_incomplete["torch"] = dict(captured["torch"])
        deliberately_incomplete["torch"].pop("deterministic_warn_only")
    elif missing is not None:
        deliberately_incomplete.pop(missing)
    # Produce a completely re-signed checkpoint through the production writer.
    # There is no checksum tampering, mocked decoder or unverified file input.
    with monkeypatch.context() as patch:
        patch.setattr(core, "capture_rng_state", lambda: deliberately_incomplete)
        trainer_adapter.save_trainer_checkpoint(
            path, model=source_model, trainer=source, identity=identity,
        )
    core.verify_checkpoint(path)
    _, decoded = core._decode_verified_state(core.prepare_checkpoint_load(path))
    if missing == "cuda":
        assert "cuda" not in decoded["rng"]["torch"]
    elif missing == "warn_only":
        assert "deterministic_warn_only" not in decoded["rng"]["torch"]
    elif missing is not None:
        assert missing not in decoded["rng"]
    return captured, config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("missing", ["python", "numpy", "cuda"])
def test_incomplete_but_verified_rng_rejected_before_model_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    missing: str,
) -> None:
    checkpoint = tmp_path / "sealed-incomplete-rng"
    _, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, missing)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    before = [p.detach().clone() for p in model.parameters()]

    def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("missing RNG stream must reject before model materialization")

    monkeypatch.setattr(loader, "_prepare_model_weights", forbidden_materialization)
    with pytest.raises(CheckpointCompatibilityError, match="RNG|rng|Python|NumPy"):
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
    for current, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(current.detach(), initial, rtol=0, atol=0)
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("missing", ["python", "numpy", "cuda"])
def test_explicit_rng_opt_out_retains_existing_checkpoint_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    missing: str,
) -> None:
    checkpoint = tmp_path / "sealed-opt-out"
    _, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, missing)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    result = loader.load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, restore_rng=False,
    )
    assert result.manifest["identity"]["step"] == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_complete_verified_checkpoint_replays_python_numpy_and_torch_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        checkpoint = tmp_path / "sealed-complete"
        saved, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, None)
        py_probe = random.Random()
        py_probe.setstate(saved["python"])
        np_probe = np.random.RandomState()
        np_probe.set_state(saved["numpy"])
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(saved["torch"]["cpu"])
        expected = (
            py_probe.random(), np_probe.random_sample(),
            torch.rand((), generator=torch_probe).item(),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        random.seed(1199)
        np.random.seed(1199)
        torch.manual_seed(1199)
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        assert actual == expected
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _seal_warn_only_mismatch(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: CheckpointIdentity,
) -> TrainerConfig:
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    captured = core.capture_rng_state()
    mismatched = dict(captured)
    mismatched["torch"] = dict(captured["torch"])
    mismatched["torch"]["deterministic_warn_only"] = (
        not config.deterministic_warn_only
    )
    with monkeypatch.context() as patch:
        patch.setattr(core, "capture_rng_state", lambda: mismatched)
        trainer_adapter.save_trainer_checkpoint(
            path, model=source_model, trainer=source, identity=identity,
        )
    core.verify_checkpoint(path)
    return config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_warn_only_mismatch_rejected_before_model_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-warn-only-mismatch"
    config = _seal_warn_only_mismatch(checkpoint, monkeypatch, checkpoint_identity)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "warn-only mismatch must reject before model materialization"
        )

    monkeypatch.setattr(loader, "_prepare_model_weights", forbidden_materialization)
    with pytest.raises(
        CheckpointCompatibilityError, match="deterministic_warn_only",
    ):
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )

    for current, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(current.detach(), initial, rtol=0, atol=0)
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_legacy_warn_only_omission_remains_canonical_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-legacy-warn-only"
    _, config = _seal_checkpoint(
        checkpoint, monkeypatch, checkpoint_identity, "warn_only",
    )
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)

    result = loader.load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, restore_rng=True,
    )

    assert result.manifest["identity"]["step"] == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
