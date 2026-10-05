"""D05 must pin native TrainerConfig values across the full restore boundary.

Synthetic CPU coverage only; this grants no corpus, training, or learned-weight evidence.
"""
from __future__ import annotations

import random
from dataclasses import asdict, fields, make_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, core
from twelve_six.checkpoint import progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "restore-binding-config-drift", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=919,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def _source() -> Trainer:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    return source


@pytest.mark.parametrize(
    ("loader", "final_restore_call"),
    [
        (trainer_adapter, 6),
        (progress_trainer, 3),
    ],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_process_state_rollback_cannot_redefine_native_config_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    final_restore_call: int,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "config-boundary-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    initial_max_steps = target.config.max_steps
    assert not torch.equal(initial_weights, source.model.weight.detach())

    original_restore = loader._restore_preapply_process_state
    original_bind = loader._bind_model_state_loader
    restore_calls = 0
    model_applications: list[bool] = []

    def restore_then_mutate(*args: Any, **kwargs: Any) -> Any:
        nonlocal restore_calls
        result = original_restore(*args, **kwargs)
        restore_calls += 1
        if restore_calls == final_restore_call:
            object.__setattr__(
                target.config,
                "max_steps",
                target.config.max_steps + 1,
            )
        return result

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    monkeypatch.setattr(
        loader,
        "_restore_preapply_process_state",
        restore_then_mutate,
    )
    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer
        else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="canonical trainer config changed during checkpoint restore",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert restore_calls == final_restore_call
    assert model_applications == []
    assert target.config.max_steps == initial_max_steps + 1
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_entry_config_snapshot_refuses_effectful_slot_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    original_slot = TrainerConfig.__dict__["max_steps"]
    hook_calls: list[bool] = []
    checkpoint_reads: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    class EffectfulSlot:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            hook_calls.append(True)
            random.random()
            np.random.random()
            torch.rand(1)
            return original_slot.__get__(instance, owner)

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("effectful config descriptor reached checkpoint I/O")

    monkeypatch.setattr(TrainerConfig, "max_steps", EffectfulSlot())
    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="config fields must remain inert slots",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert hook_calls == []
    assert checkpoint_reads == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False



@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_entry_config_snapshot_requires_exact_native_config_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=919, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    ImpostorConfig = make_dataclass(
        "TrainerConfig",
        [(field.name, field.type) for field in fields(TrainerConfig)],
        namespace={"__module__": "twelve_six.training.config"},
        frozen=True,
        slots=True,
    )
    target.config = ImpostorConfig(**asdict(target.config))
    assert type(target.config).__module__ == "twelve_six.training.config"
    assert type(target.config).__name__ == "TrainerConfig"

    checkpoint_reads: list[bool] = []

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("spoofed native config reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="must remain canonical TrainerConfig",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert checkpoint_reads == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False
