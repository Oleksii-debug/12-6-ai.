"""Serialized PyTorch optimizer IDs must agree with authoritative named slots.

These are small synthetic CPU transitions, never admitted corpus/training evidence.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
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
from twelve_six.training import Trainer, TrainerConfig


class _TwoSameShapeParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.right = torch.nn.Parameter(torch.tensor([-0.1, 0.2, -0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.left + 0.5 * self.right
        return logits.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture
def preserve_ambient_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


def _trainer(model: _TwoSameShapeParameters, *, separate_groups: bool) -> Trainer:
    config = TrainerConfig(seed=703, max_steps=3)
    if separate_groups:
        groups = [{"params": [model.left]}, {"params": [model.right]}]
    else:
        groups = [{"params": [model.left, model.right]}]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    return Trainer(model, config, optimizer=optimizer, device="cpu")


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "optimizer-id-binding", "width": 3},
        parameter_count=6,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _corrupt_ids(state: dict[str, Any], *, separate_groups: bool, attack: str) -> None:
    groups = state["optimizer"]["param_groups"]
    assert [group["param_names"] for group in groups] == (
        [["left"], ["right"]] if separate_groups else [["left", "right"]]
    )
    if attack == "permutation":
        if separate_groups:
            groups[0]["params"][0], groups[1]["params"][0] = (
                groups[1]["params"][0], groups[0]["params"][0]
            )
        else:
            groups[0]["params"].reverse()
    elif attack == "boolean-alias":
        groups[0]["params"][0] = False
    elif attack == "float-alias":
        groups[0]["params"][0] = 0.0
    else:
        raise AssertionError(f"unknown test attack: {attack}")
    # Keep the named binding and state map intact: old checks accepted these.
    assert [group["param_names"] for group in groups] == (
        [["left"], ["right"]] if separate_groups else [["left", "right"]]
    )


@pytest.mark.parametrize("separate_groups", [False, True], ids=["one", "two"])
@pytest.mark.parametrize("attack", ["permutation", "boolean-alias", "float-alias"])
def test_direct_d02_rejects_mismatched_serialized_ids_before_optimizer_apply(
    separate_groups: bool,
    attack: str,
    preserve_ambient_state: Any,
) -> None:
    model = _TwoSameShapeParameters()
    source = _trainer(model, separate_groups=separate_groups)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    valid = asdict(source.state_dict())
    bad = copy.deepcopy(valid)
    _corrupt_ids(bad, separate_groups=separate_groups, attack=attack)

    target_model = _TwoSameShapeParameters()
    target = _trainer(target_model, separate_groups=separate_groups)
    before = {name: value.clone() for name, value in target_model.state_dict().items()}
    with pytest.raises(ValueError, match="optimizer parameter ID/order differs"):
        target.load_state_dict(bad)
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    for name, value in before.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("separate_groups", [False, True], ids=["one", "two"])
@pytest.mark.parametrize("attack", ["permutation", "boolean-alias", "float-alias"])
def test_sealed_id_alias_or_permutation_refused_before_model_apply_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    separate_groups: bool,
    attack: str,
    preserve_ambient_state: Any,
) -> None:
    source_model = _TwoSameShapeParameters()
    source = _trainer(source_model, separate_groups=separate_groups)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert (source.micro_step, source.optimizer_step, source.tokens_seen) == (1, 1, 2)
    valid_state = asdict(source.state_dict())
    assert not torch.equal(
        source.optimizer.state[source_model.left]["exp_avg"],
        source.optimizer.state[source_model.right]["exp_avg"],
    )
    bad_state = copy.deepcopy(valid_state)
    _corrupt_ids(bad_state, separate_groups=separate_groups, attack=attack)

    invalid = tmp_path / f"invalid-ids-{attack}-дані"
    valid = tmp_path / f"valid-ids-{attack}-дані"
    for path, payload in ((invalid, bad_state), (valid, valid_state)):
        core.save_checkpoint(path, model=source_model, trainer_state=payload, identity=_identity())
        core.verify_checkpoint(path)

    target_model = _TwoSameShapeParameters()
    target = _trainer(target_model, separate_groups=separate_groups)
    before = {name: value.clone() for name, value in target_model.state_dict().items()}
    touched: list[bool] = []

    def reject_model_apply(*args: Any, **kwargs: Any) -> None:
        touched.append(True)
        raise AssertionError("unbound serialized parameter ID reached model application")

    monkeypatch.setattr(
        loader,
        "_bind_model_state_loader",
        lambda *args, **kwargs: reject_model_apply,
    )
    extra = {"expected_step": 1, "expected_tokens_seen": 2} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="optimizer parameter order/identity"):
        loader.load_trainer_checkpoint(
            invalid, model=target_model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    assert touched == []
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    for name, value in before.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    for name in ("left", "right"):
        src = getattr(source_model, name)
        dst = getattr(target_model, name)
        torch.testing.assert_close(src, dst, rtol=0, atol=0)
        for key, value in source.optimizer.state[src].items():
            torch.testing.assert_close(target.optimizer.state[dst][key], value, rtol=0, atol=0)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert (source.optimizer_step, target.optimizer_step) == (2, 2)
    for name in ("left", "right"):
        torch.testing.assert_close(
            getattr(source_model, name), getattr(target_model, name), rtol=0, atol=0,
        )


@pytest.mark.parametrize("separate_groups", [False, True], ids=["one", "two"])
def test_id_preflight_never_reexports_an_effectful_live_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    separate_groups: bool,
) -> None:
    """Validate canonical ID order without invoking a second state export."""
    source_model = _TwoSameShapeParameters()
    source = _trainer(source_model, separate_groups=separate_groups)
    state = source.state_dict().optimizer
    target_model = _TwoSameShapeParameters()
    target = _trainer(target_model, separate_groups=separate_groups)
    before = {name: value.clone() for name, value in target_model.state_dict().items()}
    called: list[bool] = []

    def forbidden_export() -> None:
        called.append(True)
        with torch.no_grad():
            target_model.left.add_(1.0)
        raise AssertionError("preflight called effectful optimizer.state_dict")

    monkeypatch.setattr(target.optimizer, "state_dict", forbidden_export)
    target._require_optimizer_state_parameter_order(state)
    assert called == []
    for name, value in before.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("separate_groups", [False, True], ids=["one", "two"])
@pytest.mark.parametrize("alias", [False, 0.0], ids=["bool-zero", "float-zero"])
def test_direct_d02_rejects_noncanonical_state_map_keys(
    separate_groups: bool,
    alias: Any,
    preserve_ambient_state: Any,
) -> None:
    source_model = _TwoSameShapeParameters()
    source = _trainer(source_model, separate_groups=separate_groups)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    valid = asdict(source.state_dict())
    invalid = copy.deepcopy(valid)
    invalid["optimizer"]["state"][alias] = invalid["optimizer"]["state"].pop(0)
    assert list(invalid["optimizer"]["state"]) != list(valid["optimizer"]["state"])
    assert invalid["optimizer"]["param_groups"] == valid["optimizer"]["param_groups"]

    target_model = _TwoSameShapeParameters()
    target = _trainer(target_model, separate_groups=separate_groups)
    with pytest.raises(ValueError, match="optimizer state parameter ID is noncanonical"):
        target.load_state_dict(invalid)
    assert target._failure_reason is None and target._update_incomplete is False
    assert not target.optimizer.state
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("separate_groups", [False, True], ids=["one", "two"])
@pytest.mark.parametrize("alias", [False, 0.0], ids=["bool-zero", "float-zero"])
def test_sealed_noncanonical_state_map_key_refused_without_model_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    separate_groups: bool,
    alias: Any,
    preserve_ambient_state: Any,
) -> None:
    source_model = _TwoSameShapeParameters()
    source = _trainer(source_model, separate_groups=separate_groups)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    correct = asdict(source.state_dict())
    bad = copy.deepcopy(correct)
    bad["optimizer"]["state"][alias] = bad["optimizer"]["state"].pop(0)
    assert [type(key) for key in bad["optimizer"]["state"]] != [
        type(key) for key in correct["optimizer"]["state"]
    ]

    invalid = tmp_path / "invalid-state-key-дані"
    valid = tmp_path / "valid-state-key-дані"
    for location, payload in ((invalid, bad), (valid, correct)):
        core.save_checkpoint(
            location, model=source_model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(location)

    target_model = _TwoSameShapeParameters()
    target = _trainer(target_model, separate_groups=separate_groups)
    before = {name: value.clone() for name, value in target_model.state_dict().items()}
    touched: list[bool] = []

    def forbidden_apply(*args: Any, **kwargs: Any) -> None:
        touched.append(True)
        raise AssertionError("noncanonical optimizer ID reached model application")

    monkeypatch.setattr(
        loader,
        "_bind_model_state_loader",
        lambda *args, **kwargs: forbidden_apply,
    )
    extra = {"expected_step": 1, "expected_tokens_seen": 2} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(
        CheckpointCompatibilityError, match="optimizer parameter order/identity",
    ):
        loader.load_trainer_checkpoint(
            invalid, model=target_model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    assert touched == []
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    for name, value in before.items():
        torch.testing.assert_close(target_model.state_dict()[name], value, rtol=0, atol=0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    for name in ("left", "right"):
        src = getattr(source_model, name)
        dst = getattr(target_model, name)
        for key, value in source.optimizer.state[src].items():
            torch.testing.assert_close(target.optimizer.state[dst][key], value, rtol=0, atol=0)
