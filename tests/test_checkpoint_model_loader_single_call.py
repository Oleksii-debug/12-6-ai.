"""A model loader must never be called twice after partial checkpoint application.

Synthetic CPU fixtures: no training credit, optimizer steps or trained weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class _FailsAfterPartialApply:
    def __init__(self) -> None:
        self.calls = 0
        self.value: dict[str, int] | None = None

    def load_state_dict(self, state: dict[str, int], strict: bool = True) -> None:
        self.calls += 1
        self.value = dict(state)
        if self.calls == 1:
            raise TypeError("injected TypeError after partial model apply")


def test_strict_loader_typeerror_is_not_retried() -> None:
    model = _FailsAfterPartialApply()
    with pytest.raises(TypeError, match="after partial model apply"):
        core._apply_model_weights(model, {"weight": 4}, True)
    assert model.calls == 1
    assert model.value == {"weight": 4}


def test_legacy_loader_without_strict_still_loads_once() -> None:
    class Legacy:
        calls = 0

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.calls += 1
            self.value = state

    model = Legacy()
    core._apply_model_weights(model, {"weight": 4}, False)
    assert model.calls == 1
    assert model.value == {"weight": 4}


def test_keyword_forwarder_receives_explicit_strict() -> None:
    class Forwarder:
        calls = 0

        def load_state_dict(self, state: dict[str, int], **kwargs: Any) -> None:
            self.calls += 1
            self.value = state
            self.strict = kwargs["strict"]

    model = Forwarder()
    core._apply_model_weights(model, {"weight": 4}, False)
    assert model.calls == 1
    assert model.strict is False


def test_positional_only_strict_argument_is_honored() -> None:
    class PositionalOnly:
        calls = 0

        def load_state_dict(self, state: dict[str, int], strict: bool = True, /) -> None:
            self.calls += 1
            self.strict = strict

    model = PositionalOnly()
    core._apply_model_weights(model, {}, False)
    assert model.calls == 1
    assert model.strict is False


def test_unknown_signature_fails_before_model_mutation() -> None:
    class Opaque:
        calls = 0

        @property
        def __signature__(self) -> object:
            raise ValueError("signature unavailable")

        def __call__(self, _state: object) -> None:
            self.calls += 1

    class Model:
        def __init__(self) -> None:
            self.load_state_dict = Opaque()

    model = Model()
    with pytest.raises(CheckpointCompatibilityError, match="signature unavailable"):
        core._apply_model_weights(model, {}, True)
    assert model.load_state_dict.calls == 0


def test_ambiguous_positional_strict_fails_before_model_mutation() -> None:
    class Model:
        calls = 0

        def load_state_dict(
            self, state: dict[str, int], other: object = "default",
            strict: bool = True, /,
        ) -> None:
            self.calls += 1

    model = Model()
    with pytest.raises(CheckpointCompatibilityError, match="positional strict"):
        core._apply_model_weights(model, {}, False)
    assert model.calls == 0


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_canonical_d02_partial_typeerror_poison_and_single_call(
    tmp_path: Path, loader: Any,
) -> None:
    config = TrainerConfig(max_steps=3, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config, device="cpu")
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d02-one-shot-model-load", "width": 3},
        parameter_count=12,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    checkpoint = tmp_path / "sealed-source"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint, model=source_model, trainer=source, identity=identity,
    )
    core.verify_checkpoint(checkpoint)

    target_model = torch.nn.Linear(3, 3)
    target = Trainer(target_model, config, device="cpu")
    native_load = target_model.load_state_dict
    calls = 0

    def injected_load(state: Any, strict: bool = True) -> Any:
        nonlocal calls
        calls += 1
        result = native_load(state, strict=strict)
        if calls == 1:
            raise TypeError("injected post-apply model TypeError")
        return result

    target_model.load_state_dict = injected_load  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="injected post-apply model TypeError"):
        loader.load_trainer_checkpoint(
            checkpoint, model=target_model, trainer=target,
            restore_rng=False,
        )
    assert calls == 1
    assert target.optimizer_step == 0
    assert target.tokens_seen == 0
    assert target._failure_reason == "checkpoint_restore_apply_failed"
    assert target._update_incomplete is True
