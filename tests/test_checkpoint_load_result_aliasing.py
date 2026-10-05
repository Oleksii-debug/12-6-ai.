from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, progress_trainer, trainer_adapter
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
        model_spec={"kind": "load-result-aliasing", "width": 3},
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


def _optimizer_exp_avg(state: Any) -> torch.Tensor:
    assert isinstance(state, dict)
    slots = state["state"]
    assert isinstance(slots, dict)
    for slot in slots.values():
        if isinstance(slot, dict) and isinstance(slot.get("exp_avg"), torch.Tensor):
            return slot["exp_avg"]
    raise AssertionError("AdamW exp_avg state unavailable")


def test_trainer_load_state_dict_owns_mutable_component_payloads() -> None:
    source = _source()
    state = asdict(source.state_dict())
    target = Trainer(_TinyLogits(), source.config, device="cpu")
    target.load_state_dict(state)

    supplied_exp_avg = _optimizer_exp_avg(state["optimizer"])
    live_exp_avg = next(iter(target.optimizer.state.values()))["exp_avg"]
    live_exp_avg_before = live_exp_avg.detach().clone()
    assert supplied_exp_avg.data_ptr() != live_exp_avg.data_ptr()

    supplied_exp_avg.mul_(0)
    torch.testing.assert_close(live_exp_avg, live_exp_avg_before, rtol=0, atol=0)

    supplied_scheduler = state["scheduler"]
    assert isinstance(supplied_scheduler, dict)
    supplied_last_lr = supplied_scheduler["_last_lr"]
    assert isinstance(supplied_last_lr, list)
    live_last_lr_before = list(target.scheduler._last_lr)
    supplied_last_lr[0] = float(supplied_last_lr[0]) + 321.0
    assert target.scheduler._last_lr == live_last_lr_before


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_load_result_state_cannot_mutate_live_trainer(
    tmp_path: Path,
    loader: Any,
) -> None:
    source = _source()
    checkpoint = tmp_path / "return-state-alias-дані з пробілами"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source.model,
        trainer=source,
        identity=_identity(),
    )

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )
    result = loader.load_trainer_checkpoint(
        checkpoint,
        model=target.model,
        trainer=target,
        restore_rng=False,
        **extra,
    )

    returned_optimizer = result.trainer_state["optimizer"]
    assert isinstance(returned_optimizer, dict)
    returned_exp_avg = _optimizer_exp_avg(returned_optimizer)
    live_exp_avg = next(iter(target.optimizer.state.values()))["exp_avg"]
    live_exp_avg_before = live_exp_avg.detach().clone()
    assert returned_exp_avg.data_ptr() != live_exp_avg.data_ptr()

    returned_exp_avg.add_(123.0)
    torch.testing.assert_close(live_exp_avg, live_exp_avg_before, rtol=0, atol=0)

    returned_scheduler = result.trainer_state["scheduler"]
    assert isinstance(returned_scheduler, dict)
    returned_last_lr = returned_scheduler["_last_lr"]
    assert isinstance(returned_last_lr, list)
    live_last_lr_before = list(target.scheduler._last_lr)
    returned_last_lr[0] = float(returned_last_lr[0]) + 123.0
    assert target.scheduler._last_lr == live_last_lr_before
