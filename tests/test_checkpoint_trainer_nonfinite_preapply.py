"""A checksum-valid non-finite AdamW state must fail before model mutation.

Tiny CPU examples test checkpoint safety, not authorized corpus training.
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
    cpu_before = torch.get_rng_state()
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
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "optimizer-nonfinite-preapply", "width": 3},
        parameter_count=3,
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


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("moment", "invalid"),
    [("exp_avg", float("nan")), ("exp_avg_sq", float("inf")), ("step", float("nan"))],
    ids=["moment-nan", "squared-moment-infinity", "step-nan"],
)
def test_sealed_nonfinite_optimizer_refused_before_apply_with_clean_valid_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    moment: str,
    invalid: float,
    preserve_ambient_state: Any,
) -> None:
    config = TrainerConfig(seed=703, max_steps=3)
    source_model = _TinyLogits()
    source = Trainer(source_model, config, device="cpu")
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert (source.micro_step, source.optimizer_step, source.tokens_seen) == (1, 1, 2)

    correct = asdict(source.state_dict())
    bad = copy.deepcopy(correct)
    bad_moment = bad["optimizer"]["state"][0][moment]
    assert isinstance(bad_moment, torch.Tensor)
    bad_moment.reshape(-1)[0] = invalid
    assert not torch.isfinite(bad_moment).all().item()

    invalid_path = tmp_path / f"invalid-{moment}-дані з пробілами"
    valid_path = tmp_path / f"valid-{moment}-дані з пробілами"
    for path, payload in ((invalid_path, bad), (valid_path, correct)):
        core.save_checkpoint(path, model=source_model, trainer_state=payload, identity=_identity())
        core.verify_checkpoint(path)  # Hash-valid, semantically unsafe moments.

    target_model = _TinyLogits()
    target = Trainer(target_model, config, device="cpu")
    before = target_model.weight.detach().clone()
    applied: list[bool] = []

    def forbid_model_apply(*args: Any, **kwargs: Any) -> None:
        applied.append(True)
        raise AssertionError("non-finite optimizer reached model application")

    monkeypatch.setattr(loader, "_apply_model_weights", forbid_model_apply)
    extra = {"expected_step": 1, "expected_tokens_seen": 2} if (
        loader is progress_trainer
    ) else {}
    with pytest.raises(CheckpointCompatibilityError, match="non-finite or invalid numeric"):
        loader.load_trainer_checkpoint(
            invalid_path, model=target_model, trainer=target,
            strict_model=False, restore_rng=False, **extra,
        )
    assert applied == []
    torch.testing.assert_close(target_model.weight, before, rtol=0, atol=0)
    assert not target.optimizer.state
    assert target._failure_reason is None and target._update_incomplete is False
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)

    monkeypatch.undo()
    loader.load_trainer_checkpoint(
        valid_path, model=target_model, trainer=target,
        strict_model=False, restore_rng=False, **extra,
    )
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (1, 1, 2)
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
    for src_state, dst_state in zip(
        source.optimizer.state.values(), target.optimizer.state.values(), strict=True,
    ):
        for name in src_state:
            torch.testing.assert_close(src_state[name], dst_state[name], rtol=0, atol=0)
    assert source.train_microbatch(_BATCH).optimizer_stepped
    assert target.train_microbatch(_BATCH).optimizer_stepped
    torch.testing.assert_close(target_model.weight, source_model.weight, rtol=0, atol=0)
