"""Do not remap same-shaped AdamW moments across different named model parameters.

Synthetic optimizer transitions verify the D02/D05 contract, not corpus training.
"""

from __future__ import annotations

import copy
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
from twelve_six.training import Trainer, TrainerConfig


class _TwoNamedParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.ones(3))
        self.right = torch.nn.Parameter(torch.full((3,), 2.0))


def _trainer(
    model: _TwoNamedParameters,
    config: TrainerConfig,
    *,
    reverse: bool = False,
    multiple_groups: bool = False,
) -> Trainer:
    if multiple_groups:
        groups = [
            {"params": [model.right if reverse else model.left]},
            {"params": [model.left if reverse else model.right]},
        ]
        optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    else:
        params = [model.right, model.left] if reverse else [model.left, model.right]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    return Trainer(model, config, optimizer=optimizer, device="cpu")


def _saved_nonzero_checkpoint(
    directory: Path, *, multiple_groups: bool,
) -> tuple[_TwoNamedParameters, Trainer, TrainerConfig]:
    config = TrainerConfig(seed=703, max_steps=3)
    model = _TwoNamedParameters()
    source = _trainer(model, config, multiple_groups=multiple_groups)
    model.left.grad = torch.ones(3)
    model.right.grad = torch.full((3,), 9.0)
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    # Explicitly synthetic accounting; these are not admitted data exposures.
    source.micro_step = 1
    source.optimizer_step = 1
    source.tokens_seen = 3
    source.assert_checkpoint_safe()
    sealed_state = source.state_dict()
    names = [group["param_names"] for group in sealed_state.optimizer["param_groups"]]
    assert names == (
        [["left"], ["right"]] if multiple_groups else [["left", "right"]]
    )
    assert not torch.equal(
        source.optimizer.state[model.left]["exp_avg"],
        source.optimizer.state[model.right]["exp_avg"],
    )
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "optimizer-named-order", "width": 3},
        parameter_count=6,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    trainer_adapter.save_trainer_checkpoint(
        directory, model=model, trainer=source, identity=identity,
    )
    core.verify_checkpoint(directory)
    return model, source, config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("multiple_groups", [False, True], ids=["one-group", "two-groups"])
def test_reordered_same_shape_optimizer_rejected_before_weight_application(
    tmp_path: Path,
    loader: Any,
    multiple_groups: bool,
) -> None:
    checkpoint = tmp_path / "named checkpoint"
    _, source, config = _saved_nonzero_checkpoint(
        checkpoint, multiple_groups=multiple_groups,
    )
    assert source.optimizer_step == 1
    target_model = _TwoNamedParameters()
    target = _trainer(target_model, config, reverse=True, multiple_groups=multiple_groups)
    initial = {
        name: value.detach().clone() for name, value in target_model.state_dict().items()
    }
    extra = {"expected_step": 1, "expected_tokens_seen": 3} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="optimizer parameter order"):
        loader.load_trainer_checkpoint(
            checkpoint, model=target_model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    for name, before in initial.items():
        torch.testing.assert_close(target_model.state_dict()[name], before, rtol=0, atol=0)
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("multiple_groups", [False, True], ids=["one-group", "two-groups"])
def test_same_named_order_preserves_adamw_moments_and_next_update(
    tmp_path: Path,
    loader: Any,
    multiple_groups: bool,
) -> None:
    checkpoint = tmp_path / "named complete"
    source_model, source, config = _saved_nonzero_checkpoint(
        checkpoint, multiple_groups=multiple_groups,
    )
    target_model = _TwoNamedParameters()
    target = _trainer(target_model, config, multiple_groups=multiple_groups)
    extra = {"expected_step": 1, "expected_tokens_seen": 3} if (
        loader is progress_trainer
    ) else {}
    loader.load_trainer_checkpoint(
        checkpoint, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 3)
    for name in ("left", "right"):
        src = getattr(source_model, name)
        dst = getattr(target_model, name)
        torch.testing.assert_close(src, dst, rtol=0, atol=0)
        for key, value in source.optimizer.state[src].items():
            torch.testing.assert_close(target.optimizer.state[dst][key], value, rtol=0, atol=0)
        src.grad = torch.full_like(src, 0.25 if name == "left" else 0.5)
        dst.grad = torch.full_like(dst, 0.25 if name == "left" else 0.5)
    source.optimizer.step()
    target.optimizer.step()
    for name in ("left", "right"):
        torch.testing.assert_close(
            getattr(source_model, name), getattr(target_model, name), rtol=0, atol=0,
        )


@pytest.mark.parametrize("multiple_groups", [False, True])
def test_direct_d02_refuses_legacy_unnamed_state_without_poisoning(
    tmp_path: Path,
    multiple_groups: bool,
) -> None:
    checkpoint = tmp_path / "named source"
    _, source, config = _saved_nonzero_checkpoint(
        checkpoint, multiple_groups=multiple_groups,
    )
    state = copy.deepcopy(source.state_dict())
    for group in state.optimizer["param_groups"]:
        del group["param_names"]
    model = _TwoNamedParameters()
    target = _trainer(model, config, multiple_groups=multiple_groups)
    with pytest.raises(ValueError, match="legacy unnamed optimizer state"):
        target.load_state_dict(state)
    assert not target.optimizer.state
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_tied_parameter_uses_first_named_identity() -> None:
    model = _TwoNamedParameters()
    model.alias_of_left = model.left
    trainer = _trainer(model, TrainerConfig(seed=703, max_steps=1))
    groups = trainer._optimizer_parameter_name_groups()
    assert groups == [["left", "right"]]
