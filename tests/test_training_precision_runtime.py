from __future__ import annotations

import math
import random

import pytest
import torch
from torch import nn

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig, resolve_precision_runtime


class SideEffectProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.to_called = False

    def to(self, *args, **kwargs):
        self.to_called = True
        return super().to(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, time = input_ids.shape
        return torch.zeros(batch, time, 4) * self.weight


def test_fp16_cpu_rejected_before_trainer_side_effects() -> None:
    model = SideEffectProbe()
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="available CUDA"):
        Trainer(model, TrainerConfig(max_steps=1, precision="fp16"), device="cpu")

    assert model.to_called is False
    assert random.getstate() == python_rng_before
    torch.testing.assert_close(torch.get_rng_state(), torch_rng_before)
    assert next(iter(model.parameters())).grad is None


def test_cuda_fp32_unavailable_rejected_before_trainer_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SideEffectProbe()

    with pytest.raises(ValueError, match="fp32 CUDA training requires an available CUDA"):
        Trainer(model, TrainerConfig(max_steps=1, precision="fp32"), device="cuda")

    assert model.to_called is False


def test_explicit_cuda_index_must_be_visible_before_trainer_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    model = SideEffectProbe()

    with pytest.raises(ValueError, match="cuda:1 is not visible"):
        Trainer(model, TrainerConfig(max_steps=1, precision="fp32"), device="cuda:1")

    assert model.to_called is False


def test_precision_runtime_contract_is_machine_readable() -> None:
    fp32 = resolve_precision_runtime("fp32", "cpu")
    bf16 = resolve_precision_runtime("bf16", "cpu")

    assert fp32.to_dict() == {
        "requested": "fp32",
        "device_type": "cpu",
        "parameter_dtype": "float32",
        "optimizer_master_dtype": "float32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "grad_scaler_enabled": False,
        "grad_scaler_device": None,
    }
    assert bf16.to_dict() == {
        "requested": "bf16",
        "device_type": "cpu",
        "parameter_dtype": "float32",
        "optimizer_master_dtype": "float32",
        "autocast_enabled": True,
        "autocast_dtype": "bfloat16",
        "grad_scaler_enabled": False,
        "grad_scaler_device": None,
    }


def test_cuda_fp16_policy_uses_fp32_master_weights_and_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    runtime = resolve_precision_runtime("fp16", "cuda")

    assert runtime.parameter_dtype == "float32"
    assert runtime.optimizer_master_dtype == "float32"
    assert runtime.autocast_dtype == "float16"
    assert runtime.grad_scaler_enabled is True
    assert runtime.grad_scaler_device == "cuda"


def test_cuda_bf16_capability_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(ValueError, match="native bf16 support"):
        resolve_precision_runtime("bf16", "cuda")


def test_cuda_bf16_probe_excludes_emulation_when_runtime_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def probe(*, including_emulation: bool = True) -> bool:
        calls.append(including_emulation)
        return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", probe)

    runtime = resolve_precision_runtime("bf16", "cuda")

    assert runtime.autocast_dtype == "bfloat16"
    assert calls == [False]


def test_mixed_precision_rejects_downcast_master_weights_before_device_move() -> None:
    model = SideEffectProbe().bfloat16()

    with pytest.raises(ValueError, match="requires FP32 model parameters"):
        Trainer(model, TrainerConfig(max_steps=1, precision="bf16"), device="cpu")

    assert model.to_called is False


def test_real_s0_10k_cpu_bf16_step_is_finite_and_updates_weights() -> None:
    stage = load_stage_config("configs/stages/s0_10k.json")
    torch.manual_seed(1337)
    model = TwelveSixDecoder(stage.model, stage.init)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            weight_decay=0.0,
            max_steps=2,
            scheduler="constant",
            gradient_clip_norm=1.0,
            precision="bf16",
            seed=1337,
        ),
        device="cpu",
    )
    assert trainer.precision_runtime.to_dict()["autocast_dtype"] == "bfloat16"
    assert trainer.precision_runtime.grad_scaler_enabled is False
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}

    batch = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.long,
        )
    }
    metrics = [trainer.train_microbatch(batch) for _ in range(2)]

    assert trainer.optimizer_step == 2
    assert trainer.tokens_seen == 28
    assert all(math.isfinite(item.loss) for item in metrics)
    assert all(item.grad_norm is not None and math.isfinite(item.grad_norm) for item in metrics)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    state_dtypes = {
        value.dtype
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel() > 1
    }
    assert state_dtypes == {torch.float32}
