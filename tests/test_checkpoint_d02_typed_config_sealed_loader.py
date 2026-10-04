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

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
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
def test_sealed_mistyped_config_refused_before_model_apply_then_valid_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state: Any,
    loader: Any,
    field: str,
    alias: Any,
    config_betas: tuple[float, float],
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
            strict_model=False, restore_rng=False, **extra,
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
    assert policy_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
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
