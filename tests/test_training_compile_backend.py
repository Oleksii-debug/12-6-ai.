from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from twelve_six.training.compile_backend import (
    CompileTrainingConfig,
    CompiledTrainer,
    break_even_step_count,
    build_training_backend,
    compilation_runtime_audit,
    explain_model_graph,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=1,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=None,
        precision="fp32",
        seed=7,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def test_compile_config_is_opt_in_and_eager_remains_canonical() -> None:
    model = _ToyLM()
    trainer = build_training_backend(model, _trainer_config())
    assert type(trainer) is Trainer
    assert not hasattr(trainer, "compile_config")
    assert CompileTrainingConfig().enabled is False


def test_eager_dynamo_backend_preserves_one_update_and_state_keys() -> None:
    torch.manual_seed(11)
    eager_model = _ToyLM()
    compiled_model = copy.deepcopy(eager_model)
    eager = Trainer(eager_model, _trainer_config(), device="cpu")
    compiled = build_training_backend(
        compiled_model,
        _trainer_config(),
        compile_config=CompileTrainingConfig(
            enabled=True,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        ),
        device="cpu",
    )
    assert isinstance(compiled, CompiledTrainer)
    assert tuple(eager.model.state_dict()) == tuple(compiled.model.state_dict())

    batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)}
    eager_metrics = eager.train_microbatch(batch)
    compiled_metrics = compiled.train_microbatch(batch)

    assert eager_metrics == compiled_metrics
    for eager_parameter, compiled_parameter in zip(
        eager.model.parameters(),
        compiled.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            eager_parameter,
            compiled_parameter,
            rtol=0.0,
            atol=0.0,
        )


def test_graph_diagnostics_report_break_count_for_traceable_model() -> None:
    torch.manual_seed(13)
    model = _ToyLM()
    diagnostics = explain_model_graph(
        model,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    )
    if not diagnostics.supported:
        pytest.skip("torch._dynamo.explain unavailable")
    assert diagnostics.graph_count == 1
    assert diagnostics.graph_break_count == 0
    assert diagnostics.break_reasons == ()


def test_runtime_audit_is_bounded_and_never_claims_paid_compute() -> None:
    audit = compilation_runtime_audit()
    assert audit["torch"] == torch.__version__
    assert audit["paid_compute"] is False
    assert isinstance(audit["dynamo_backends"], list)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "eager_first_seconds": 1.0,
                "eager_steady_seconds": 1.0,
                "compiled_first_seconds": 5.0,
                "compiled_steady_seconds": 0.5,
            },
            9,
        ),
        (
            {
                "eager_first_seconds": 1.0,
                "eager_steady_seconds": 1.0,
                "compiled_first_seconds": 2.0,
                "compiled_steady_seconds": 1.1,
            },
            None,
        ),
        (
            {
                "eager_first_seconds": 1.0,
                "eager_steady_seconds": 1.0,
                "compiled_first_seconds": 0.8,
                "compiled_steady_seconds": 0.5,
            },
            1,
        ),
    ],
)
def test_break_even_step_count(values: dict[str, float], expected: int | None) -> None:
    assert break_even_step_count(**values) == expected
