"""D02 trainer resume must reject Python numeric/bool aliases before mutation.

Synthetic CPU engineering cases; no corpus, optimizer admission or learned-model credit.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig


class _TinyModel(torch.nn.Module):
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
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("deterministic_algorithms", 1),  # True == 1, but not the same configuration.
        ("deterministic_warn_only", 0),  # False == 0.
        ("seed", 703.0),  # 703 == 703.0.
        ("max_steps", 2.0),
        ("weight_decay", 0),  # 0.0 == 0.
        ("betas", (0.9, 0.95, 0.0)),  # Structural mismatch.
        ("betas", [0.9, 0.95]),  # Same numbers, wrong sequence type.
        ("gradient_clip_norm", 1),  # 1 == 1.0.
    ],
)
def test_mistyped_or_malformed_config_fails_before_optimizer_load(
    monkeypatch: pytest.MonkeyPatch,
    preserve_ambient_state,
    field: str,
    alias: object,
) -> None:
    config = TrainerConfig(seed=703, max_steps=2)
    saved = Trainer(_TinyModel(), config).state_dict()
    invalid = copy.deepcopy(saved.config)
    invalid[field] = alias
    target = Trainer(_TinyModel(), config)
    weights_before = target.model.weight.detach().clone()
    optimizer_load_calls: list[bool] = []

    def forbidden_load(value: object) -> None:
        optimizer_load_calls.append(True)
        raise AssertionError("config preflight must refuse before optimizer I/O")

    monkeypatch.setattr(target.optimizer, "load_state_dict", forbidden_load)
    with pytest.raises(ValueError, match="trainer config mismatch"):
        target.load_state_dict(replace(saved, config=invalid))

    assert optimizer_load_calls == []
    assert target._failure_reason is None
    assert target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight.detach(), weights_before, rtol=0, atol=0)
    monkeypatch.undo()
    target.load_state_dict(saved)
    assert target.state_dict().optimizer_step == 0



def test_nested_numeric_alias_is_rejected_before_mutation(preserve_ambient_state) -> None:
    config = TrainerConfig(seed=703, max_steps=2, betas=(0.0, 0.95))
    saved = Trainer(_TinyModel(), config).state_dict()
    invalid = copy.deepcopy(saved.config)
    invalid["betas"] = (0, 0.95)  # Python considers these tuples equal.
    target = Trainer(_TinyModel(), config)
    with pytest.raises(ValueError, match="trainer config mismatch"):
        target.load_state_dict(replace(saved, config=invalid))
    assert target._failure_reason is None
    assert target._update_incomplete is False

@pytest.mark.parametrize("variant", ["missing", "unexpected"])
def test_config_key_set_is_exact_before_optimizer_load(
    preserve_ambient_state, variant: str,
) -> None:
    config = TrainerConfig(seed=703, max_steps=2)
    saved = Trainer(_TinyModel(), config).state_dict()
    invalid = copy.deepcopy(saved.config)
    if variant == "missing":
        del invalid["seed"]
    else:
        invalid["unknown_policy"] = False
    target = Trainer(_TinyModel(), config)
    with pytest.raises(ValueError, match="trainer config mismatch"):
        target.load_state_dict(replace(saved, config=invalid))
    assert target._failure_reason is None
    assert target._update_incomplete is False


def test_correctly_typed_nonzero_resume_preserves_next_adamw_step(
    preserve_ambient_state,
) -> None:
    config = TrainerConfig(seed=703, max_steps=2)
    source = Trainer(_TinyModel(), config)
    source.train_microbatch(_BATCH)
    saved = source.state_dict()
    model_state = copy.deepcopy(source.model.state_dict())
    target = Trainer(_TinyModel(), config)
    target.model.load_state_dict(model_state)
    target.load_state_dict(saved)
    source.train_microbatch(_BATCH)
    target.train_microbatch(_BATCH)
    torch.testing.assert_close(source.model.weight, target.model.weight, rtol=0, atol=0)
    for source_state, target_state in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True,
    ):
        for name in source_state:
            torch.testing.assert_close(source_state[name], target_state[name], rtol=0, atol=0)
    assert source.state_dict().optimizer_step == target.state_dict().optimizer_step == 2
