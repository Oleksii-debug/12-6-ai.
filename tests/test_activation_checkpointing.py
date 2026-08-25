from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.distributed.activation_checkpointing import (
    apply_activation_checkpointing,
    checkpoint_block_indices,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.loss import causal_lm_loss


def _stage():
    return load_stage_config(Path(__file__).parents[1] / "configs/stages/s1_100k.json")


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _forward_backward(state_dict, policy: str):
    stage = _stage()
    model = TwelveSixDecoder(stage.model, stage.init)
    model.load_state_dict(state_dict)
    plan = apply_activation_checkpointing(model, policy)  # type: ignore[arg-type]
    input_ids = torch.randint(
        0,
        stage.model.vocab_size,
        (2, 32),
        generator=torch.Generator().manual_seed(143),
        dtype=torch.long,
    )
    output = model(input_ids)
    loss = causal_lm_loss(output.logits, input_ids)
    loss.backward()
    gradients = {
        _canonical_parameter_name(name): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.logits.detach().clone(), gradients, plan


def test_policy_block_selection_is_deterministic() -> None:
    assert checkpoint_block_indices(6, "none") == ()
    assert checkpoint_block_indices(6, "every_other_block") == (0, 2, 4)
    assert checkpoint_block_indices(6, "per_block") == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError, match="unsupported activation checkpoint policy"):
        checkpoint_block_indices(6, "unknown")  # type: ignore[arg-type]


def test_checkpoint_wrappers_preserve_state_dict_keys() -> None:
    stage = _stage()
    model = TwelveSixDecoder(stage.model, stage.init)
    expected = tuple(model.state_dict().keys())
    plan = apply_activation_checkpointing(model, "per_block")
    assert plan.checkpointed_blocks == stage.model.n_layers
    assert tuple(model.state_dict().keys()) == expected


def test_checkpointed_forward_and_gradients_match_fp32_control() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260826)
    stage = _stage()
    control = TwelveSixDecoder(stage.model, stage.init)
    state_dict = {name: tensor.detach().clone() for name, tensor in control.state_dict().items()}

    reference_logits, reference_gradients, _ = _forward_backward(state_dict, "none")
    for policy in ("every_other_block", "per_block"):
        logits, gradients, plan = _forward_backward(state_dict, policy)
        assert plan.checkpointed_blocks > 0
        torch.testing.assert_close(logits, reference_logits, rtol=0.0, atol=0.0)
        assert gradients.keys() == reference_gradients.keys()
        for name in reference_gradients:
            torch.testing.assert_close(
                gradients[name],
                reference_gradients[name],
                rtol=0.0,
                atol=0.0,
            )


def test_checkpointing_is_not_applied_twice() -> None:
    stage = _stage()
    model = TwelveSixDecoder(stage.model, stage.init)
    apply_activation_checkpointing(model, "per_block")
    with pytest.raises(RuntimeError, match="applied exactly once"):
        apply_activation_checkpointing(model, "per_block")
