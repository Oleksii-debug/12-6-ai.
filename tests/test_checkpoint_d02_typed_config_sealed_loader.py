"""A checksum-valid but mistyped D02 checkpoint must fail before model apply.

Both public D05 loaders are exercised with physical sealed files and a
same-instance clean retry. Tiny CPU tensors are not admitted training evidence.
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


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "typed-config-preapply", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )



def _assert_cuda_rng_unchanged(before: list[torch.Tensor] | None) -> None:
    """Require invalid checkpoint preflight not to consume GPU RNG."""
    if before is None:
        return
    after = torch.cuda.get_rng_state_all()
    assert len(after) == len(before)
    for actual, previous in zip(after, before, strict=True):
        torch.testing.assert_close(actual, previous, rtol=0, atol=0)


def _assert_live_rng_matches(expected: dict[str, Any]) -> None:
    """Require actual stream states, not only deterministic model outputs."""
    assert random.getstate() == expected["python"]
    actual_numpy = np.random.get_state()
    wanted_numpy = expected["numpy"]
    assert actual_numpy[0] == wanted_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], wanted_numpy[1])
    assert actual_numpy[2:] == wanted_numpy[2:]
    torch.testing.assert_close(
        torch.get_rng_state(), expected["torch"]["cpu"], rtol=0, atol=0,
    )
    assert (
        torch.are_deterministic_algorithms_enabled()
        == expected["torch"]["deterministic_algorithms"]
    )
    if torch.cuda.is_available():
        for actual, wanted in zip(
            torch.cuda.get_rng_state_all(), expected["torch"]["cuda"], strict=True,
        ):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def _advance_rng_for_replay_probe() -> dict[str, Any]:
    """Create a visible difference from the checkpoint's persisted RNG states."""
    random.random()
    np.random.random()
    torch.rand(())
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.rand((), device=f"cuda:{device}")
    return core.capture_rng_state()


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("field", "alias", "config_betas"),
    [
        ("deterministic_algorithms", 1, (0.9, 0.95)),
        ("seed", 703.0, (0.9, 0.95)),
        ("betas", (0, 0.95), (0.0, 0.95)),
    ],
    ids=["bool-as-int", "seed-as-float", "nested-beta-alias"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "replay"])
def test_sealed_mistyped_config_refused_before_model_apply_then_valid_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    loader: Any,
    field: str,
    alias: Any,
    config_betas: tuple[float, float],
    restore_rng: bool,
) -> None:
    config = TrainerConfig(seed=703, max_steps=2, betas=config_betas)
    source_model = _TinyLogits()
    source = Trainer(source_model, config, device="cpu")
    with torch.no_grad():
        source_model.weight.add_(1.0)
    state = asdict(source.state_dict())
    invalid_state = copy.deepcopy(state)
    invalid_state["config"][field] = alias
    assert invalid_state["config"] == state["config"]
    # The comparison used before #2653 accepted this metadata despite type drift.
    if field == "betas":
        assert type(invalid_state["config"][field][0]) is not float
    else:
        assert type(invalid_state["config"][field]) is not type(state["config"][field])

    invalid_path = tmp_path / "sealed-invalid з пробілами"
    valid_path = tmp_path / "sealed-valid з пробілами"
    for path, payload in ((invalid_path, invalid_state), (valid_path, state)):
        core.save_checkpoint(
            path, model=source_model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(path)  # Internally valid; semantically incompatible.

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    before_weights = target_model.weight.detach().clone()
    application_calls: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def forbidden_model_apply(*args: Any, **kwargs: Any) -> None:
        application_calls.append(True)
        raise AssertionError("model application reached before config validation")

    monkeypatch.setattr(loader, "_apply_model_weights", forbidden_model_apply)
    extra = {"expected_step": 0, "expected_tokens_seen": 0} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="trainer config mismatch"):
        loader.load_trainer_checkpoint(
            invalid_path, model=target_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **extra,
        )

    assert application_calls == []
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert not target.optimizer.state
    assert target_model.weight.grad is None
    torch.testing.assert_close(target_model.weight, before_weights, rtol=0, atol=0)
    # Failed pre-application decoding must not advance future RNG draws.
    assert random.getstate() == py_before
    after_numpy = np.random.get_state()
    assert after_numpy[0] == np_before[0]
    np.testing.assert_array_equal(after_numpy[1], np_before[1])
    assert after_numpy[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    _assert_cuda_rng_unchanged(cuda_before)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    # Decode the trusted bytes, then deliberately drift all live RNG streams.
    saved_rng = core._decode_verified_state(
        core.prepare_checkpoint_load(valid_path)
    )[1]["rng"]
    drifted_rng = _advance_rng_for_replay_probe()
    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target_model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **extra,
    )
    _assert_live_rng_matches(saved_rng if restore_rng else drifted_rng)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)

    # A real D02 next-step transition is equivalent after the clean physical retry.
    source.train_microbatch(_BATCH)
    target.train_microbatch(_BATCH)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    for source_state, target_state in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True,
    ):
        for name in source_state:
            torch.testing.assert_close(
                target_state[name], source_state[name], rtol=0, atol=0,
            )


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("field", ["micro_step", "optimizer_step", "tokens_seen"])
@pytest.mark.parametrize("alias", [False, 0.0], ids=["bool-zero", "float-zero"])
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "replay"])
def test_sealed_counter_alias_refused_before_weight_apply_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    loader: Any,
    field: str,
    alias: Any,
    restore_rng: bool,
) -> None:
    """Persist representable aliases, not just Python-only int subclasses."""
    config = TrainerConfig(seed=703, max_steps=2)
    source_model = _TinyLogits()
    source = Trainer(source_model, config, device="cpu")
    with torch.no_grad():
        source_model.weight.add_(1.0)
    valid_state = asdict(source.state_dict())
    invalid_state = copy.deepcopy(valid_state)
    invalid_state[field] = alias
    assert invalid_state[field] == valid_state[field]
    assert type(invalid_state[field]) is not int

    invalid_path = tmp_path / f"invalid-{field}-дані"
    valid_path = tmp_path / f"valid-{field}-дані"
    for path, payload in ((invalid_path, invalid_state), (valid_path, valid_state)):
        core.save_checkpoint(
            path, model=source_model, trainer_state=payload, identity=_identity(),
        )
        core.verify_checkpoint(path)

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    original_weight = target_model.weight.detach().clone()
    reached_apply: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def forbid_model_apply(*args: Any, **kwargs: Any) -> None:
        reached_apply.append(True)
        raise AssertionError("counter type rejection must precede model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_apply)
    expected = {"expected_step": 0, "expected_tokens_seen": 0} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(
        CheckpointCompatibilityError, match=f"trainer {field} must be a non-negative integer",
    ):
        loader.load_trainer_checkpoint(
            invalid_path, model=target_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **expected,
        )
    assert reached_apply == []
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target._failure_reason is None and target._update_incomplete is False
    assert not target.optimizer.state and target_model.weight.grad is None
    torch.testing.assert_close(target_model.weight, original_weight, rtol=0, atol=0)
    assert random.getstate() == py_before
    after_np = np.random.get_state()
    assert after_np[0] == np_before[0]
    np.testing.assert_array_equal(after_np[1], np_before[1])
    assert after_np[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    _assert_cuda_rng_unchanged(cuda_before)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    # Decode the trusted bytes, then deliberately drift all live RNG streams.
    saved_rng = core._decode_verified_state(
        core.prepare_checkpoint_load(valid_path)
    )[1]["rng"]
    drifted_rng = _advance_rng_for_replay_probe()
    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target_model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **expected,
    )
    _assert_live_rng_matches(saved_rng if restore_rng else drifted_rng)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    source.train_microbatch(_BATCH)
    target.train_microbatch(_BATCH)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("micro_step", True),
        ("optimizer_step", True),
        ("micro_step", 1.0),
        ("optimizer_step", 1.0),
        ("tokens_seen", 2.0),
        ("micro_step", np.int64(1)),
        ("optimizer_step", np.int64(1)),
        ("tokens_seen", np.int64(2)),
    ],
    ids=[
        "micro-bool", "optimizer-bool", "micro-float", "optimizer-float",
        "tokens-float", "micro-numpy-int", "optimizer-numpy-int", "tokens-numpy-int",
    ],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "replay"])
def test_sealed_nonzero_counter_alias_cannot_remap_adamw_or_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    loader: Any,
    field: str,
    alias: Any,
    restore_rng: bool,
) -> None:
    """Refuse a semantically invalid nonzero checkpoint before model mutation."""
    from dataclasses import replace

    config = TrainerConfig(seed=703, max_steps=2)
    source_model = _TinyLogits()
    source = Trainer(source_model, config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert (source.micro_step, source.optimizer_step, source.tokens_seen) == (1, 1, 2)
    state = asdict(source.state_dict())
    bad_state = copy.deepcopy(state)
    bad_state[field] = alias
    assert bad_state[field] == state[field] and type(bad_state[field]) is not int

    identity = replace(_identity(), step=1, tokens_seen=2)
    invalid = tmp_path / f"invalid-committed-{field}-дані"
    valid = tmp_path / f"valid-committed-{field}-дані"
    for location, payload in ((invalid, bad_state), (valid, state)):
        core.save_checkpoint(
            location, model=source_model, trainer_state=payload, identity=identity,
        )
        core.verify_checkpoint(location)

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    weights_before = target_model.weight.detach().clone()
    model_applications: list[bool] = []
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def refuse_model_apply(*args: Any, **kwargs: Any) -> None:
        model_applications.append(True)
        raise AssertionError("invalid committed counter reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", refuse_model_apply)
    expected = {"expected_step": 1, "expected_tokens_seen": 2} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(
        CheckpointCompatibilityError, match=f"trainer {field} must be a non-negative integer",
    ):
        loader.load_trainer_checkpoint(
            invalid, model=target_model, trainer=target,
            strict_model=False, restore_rng=restore_rng, **expected,
        )
    assert model_applications == []
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    assert target_model.weight.grad is None
    torch.testing.assert_close(target_model.weight, weights_before, rtol=0, atol=0)
    assert random.getstate() == py_before
    after_numpy = np.random.get_state()
    assert after_numpy[0] == np_before[0]
    np.testing.assert_array_equal(after_numpy[1], np_before[1])
    assert after_numpy[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    _assert_cuda_rng_unchanged(cuda_before)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    # Decode the trusted bytes, then deliberately drift all live RNG streams.
    saved_rng = core._decode_verified_state(
        core.prepare_checkpoint_load(valid)
    )[1]["rng"]
    drifted_rng = _advance_rng_for_replay_probe()
    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid, model=target_model, trainer=target,
        strict_model=False, restore_rng=restore_rng, **expected,
    )
    _assert_live_rng_matches(saved_rng if restore_rng else drifted_rng)
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    for source_state, target_state in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True,
    ):
        for key in source_state:
            torch.testing.assert_close(source_state[key], target_state[key], rtol=0, atol=0)

    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    assert (source.optimizer_step, target.optimizer_step) == (2, 2)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
