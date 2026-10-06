"""D02 policy mismatch must reject before checkpoint I/O in both public loaders.

Tiny CPU-only regression fixtures; no training, corpus, or GPU acceptance.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointCompatibilityError, progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


@pytest.fixture
def preserve_process_state():
    """Return RNG/policy globals to the caller after each adversarial case."""
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0], warn_only=policy_before[1],
        )


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("enabled", "warn_only"),
    [(False, True), (True, False)],
    ids=["enabled-drift", "warn-only-drift"],
)
def test_canonical_d02_policy_drift_rejected_before_any_checkpoint_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_process_state: Any,
    loader: Any,
    enabled: bool,
    warn_only: bool,
) -> None:
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=2,
            seed=703,
            deterministic_algorithms=True,
            deterministic_warn_only=True,
        ),
        device="cpu",
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    attempted_reads: list[Path] = []

    class CheckpointReadReached(Exception):
        pass

    def reached_read(directory: str | Path) -> None:
        attempted_reads.append(Path(directory))
        raise CheckpointReadReached()

    monkeypatch.setattr(loader, "prepare_checkpoint_load", reached_read)
    checkpoint = tmp_path / "missing checkpoint з пробілами"
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    with pytest.raises(CheckpointCompatibilityError, match="deterministic"):
        loader.load_trainer_checkpoint(checkpoint, model=model, trainer=trainer)
    assert attempted_reads == []
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
    assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (0, 0, 0)
    for name, initial in before.items():
        torch.testing.assert_close(model.state_dict()[name], initial, rtol=0, atol=0)

    # Correcting the pure precondition must allow a clean same-instance retry.
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.raises(CheckpointReadReached):
        loader.load_trainer_checkpoint(checkpoint, model=model, trainer=trainer)
    assert attempted_reads == [checkpoint]
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
