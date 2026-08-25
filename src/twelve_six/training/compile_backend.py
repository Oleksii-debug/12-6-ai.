"""Opt-in torch.compile training backend that preserves canonical Trainer semantics."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import TrainerConfig
from .trainer import Trainer


@dataclass(frozen=True, slots=True)
class CompileTrainingConfig:
    """Transient compilation controls; disabled unless explicitly opted in."""

    enabled: bool = False
    backend: str = "inductor"
    fullgraph: bool = True
    dynamic: bool | None = False
    mode: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.fullgraph, bool):
            raise TypeError("fullgraph must be bool")
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise TypeError("dynamic must be bool or None")
        if self.mode is not None and (not isinstance(self.mode, str) or not self.mode.strip()):
            raise ValueError("mode must be None or a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompileGraphDiagnostics:
    supported: bool
    graph_count: int | None
    graph_break_count: int | None
    op_count: int | None
    break_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "graph_count": self.graph_count,
            "graph_break_count": self.graph_break_count,
            "op_count": self.op_count,
            "break_reasons": list(self.break_reasons),
        }


def compilation_runtime_audit() -> dict[str, Any]:
    """Return bounded runtime metadata relevant to torch.compile decisions."""

    dynamo = getattr(torch, "_dynamo", None)
    backends: list[str] = []
    if dynamo is not None and hasattr(dynamo, "list_backends"):
        try:
            backends = sorted(str(value) for value in dynamo.list_backends())
        except Exception:
            backends = []

    cpu_capability: str | None = None
    cpu_backend = getattr(torch.backends, "cpu", None)
    if cpu_backend is not None and hasattr(cpu_backend, "get_cpu_capability"):
        try:
            cpu_capability = str(cpu_backend.get_cpu_capability())
        except Exception:
            cpu_capability = None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_compile_available": callable(getattr(torch, "compile", None)),
        "module_compile_available": callable(getattr(nn.Module, "compile", None)),
        "dynamo_available": dynamo is not None,
        "dynamo_backends": backends,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_capability": cpu_capability,
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "paid_compute": False,
    }


def explain_model_graph(model: nn.Module, input_ids: Tensor) -> CompileGraphDiagnostics:
    """Capture Dynamo graph-break diagnostics without requiring Inductor code generation."""

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None or not hasattr(dynamo, "explain"):
        return CompileGraphDiagnostics(
            supported=False,
            graph_count=None,
            graph_break_count=None,
            op_count=None,
            break_reasons=(),
        )

    if hasattr(dynamo, "reset"):
        dynamo.reset()
    explanation = dynamo.explain(model)(input_ids)
    reasons = tuple(str(value) for value in getattr(explanation, "break_reasons", ()) or ())
    graph_count = getattr(explanation, "graph_count", None)
    graph_break_count = getattr(explanation, "graph_break_count", None)
    op_count = getattr(explanation, "op_count", None)
    return CompileGraphDiagnostics(
        supported=True,
        graph_count=None if graph_count is None else int(graph_count),
        graph_break_count=None if graph_break_count is None else int(graph_break_count),
        op_count=None if op_count is None else int(op_count),
        break_reasons=reasons,
    )


class CompiledTrainer(Trainer):
    """Trainer variant that compiles only the model call path in-place.

    Optimizer construction, loss semantics, gradient normalization/clipping,
    counters, failure poisoning, scheduler behavior, and checkpoint state remain
    inherited from the canonical eager Trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        compile_config: CompileTrainingConfig,
        device: str | torch.device = "cpu",
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        if not compile_config.enabled:
            raise ValueError("CompiledTrainer requires compile_config.enabled=true")
        if not callable(getattr(nn.Module, "compile", None)):
            raise RuntimeError("this PyTorch runtime does not provide nn.Module.compile()")

        super().__init__(
            model,
            config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.compile_config = compile_config

        parameter_ids_before = tuple(id(parameter) for parameter in self.model.parameters())
        state_keys_before = tuple(self.model.state_dict().keys())
        kwargs: dict[str, Any] = {
            "backend": compile_config.backend,
            "fullgraph": compile_config.fullgraph,
            "dynamic": compile_config.dynamic,
        }
        if compile_config.mode is not None:
            kwargs["mode"] = compile_config.mode
        self.model.compile(**kwargs)

        parameter_ids_after = tuple(id(parameter) for parameter in self.model.parameters())
        state_keys_after = tuple(self.model.state_dict().keys())
        if parameter_ids_after != parameter_ids_before:
            raise RuntimeError("torch.compile changed canonical parameter object identities")
        if state_keys_after != state_keys_before:
            raise RuntimeError("torch.compile changed canonical model state_dict keys")


def build_training_backend(
    model: nn.Module,
    trainer_config: TrainerConfig,
    *,
    compile_config: CompileTrainingConfig | None = None,
    device: str | torch.device = "cpu",
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> Trainer:
    """Construct eager by default and compiled only when explicitly enabled."""

    selected = CompileTrainingConfig() if compile_config is None else compile_config
    if not selected.enabled:
        return Trainer(
            model,
            trainer_config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return CompiledTrainer(
        model,
        trainer_config,
        compile_config=selected,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def break_even_step_count(
    *,
    eager_first_seconds: float,
    eager_steady_seconds: float,
    compiled_first_seconds: float,
    compiled_steady_seconds: float,
) -> int | None:
    """Return first step count where the measured compiled timing model wins."""

    values = (
        eager_first_seconds,
        eager_steady_seconds,
        compiled_first_seconds,
        compiled_steady_seconds,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timings must be finite positive seconds")
    if compiled_steady_seconds >= eager_steady_seconds:
        return None
    first_overhead = compiled_first_seconds - eager_first_seconds
    if first_overhead <= 0.0:
        return 1
    per_following_step_gain = eager_steady_seconds - compiled_steady_seconds
    return max(1, math.ceil(1.0 + first_overhead / per_following_step_gain))
