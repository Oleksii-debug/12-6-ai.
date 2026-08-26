from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.training import Trainer, TrainerConfig


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "trainer-target-preflight-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=17,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


@pytest.mark.parametrize(
    ("target_flag", "target_value"),
    [
        ("_failure_reason", "synthetic poisoned trainer"),
        ("_update_incomplete", True),
    ],
)
def test_production_trainer_rejects_unloadable_target_before_model_or_rng_mutation(
    tmp_path: Path,
    target_flag: str,
    target_value: object,
) -> None:
    torch = pytest.importorskip("torch")
    config = TrainerConfig(max_steps=1, seed=17)

    source_model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        source_model.weight.fill_(3.0)
    source_trainer = Trainer(source_model, config)

    checkpoint = tmp_path / "trainer-target-preflight"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    target_model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        target_model.weight.fill_(-2.0)
    target_trainer = Trainer(target_model, config)
    setattr(target_trainer, target_flag, target_value)
    before_model = {
        name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()
    }

    # Make the live RNG state intentionally distinct from the checkpoint RNG state.
    # A preflight rejection must happen before load_verified_checkpoint can restore it.
    torch.manual_seed(123456)
    before_torch_rng = torch.random.get_rng_state().clone()

    with pytest.raises(CheckpointCompatibilityError, match="fresh trainer"):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            restore_rng=True,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), before_torch_rng)
