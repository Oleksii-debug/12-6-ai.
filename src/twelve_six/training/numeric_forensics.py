"""Bounded, privacy-safe diagnostics for poisoned numerical training states.

Failure details live here so Trainer only owns poisoning and transition points. The
module never logs raw training text, installs hooks, or changes model forward semantics.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

FailureKind = Literal["loss", "gradient", "update"]
ActivationHealthProvider = Callable[[], Mapping[str, Any] | None]

_MAX_AFFECTED_NAMES = 16
_MAX_ACTIVATION_HEALTH_ITEMS = 16
_BATCH_TENSOR_KEYS = ("input_ids", "labels", "target_ids", "loss_mask")


@dataclass(frozen=True, slots=True)
class NumericFailureDiagnostics:
    """Machine-readable forensic snapshot captured only on a non-finite failure."""

    schema_version: int
    kind: FailureKind
    micro_step: int
    optimizer_step: int
    tokens_seen: int
    batch_tokens: int
    pending_tokens: int
    precision: str
    device_type: str
    learning_rate: float
    gradient_norm: float | None
    gradient_norm_finite: bool | None
    affected_parameter_names: tuple[str, ...]
    affected_module_names: tuple[str, ...]
    affected_parameter_count: int
    affected_names_truncated: bool
    activation_health: dict[str, bool | int | float | str] | None
    batch_identity_sha256: str
    raw_training_text_logged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AffectedParameters:
    names: tuple[str, ...]
    total_count: int

    @property
    def truncated(self) -> bool:
        return self.total_count > len(self.names)


def batch_identity_sha256(batch: Mapping[str, Any]) -> str:
    """Hash only tensor-valued training fields; ignore arbitrary metadata/text."""

    digest = hashlib.sha256()
    found = False
    for key in _BATCH_TENSOR_KEYS:
        value = batch.get(key)
        if not isinstance(value, Tensor):
            continue
        found = True
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(key.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if not found:
        digest.update(b"no-training-tensors")
    return digest.hexdigest()


def _bounded_parameter_names(names: list[str], total_count: int) -> AffectedParameters:
    return AffectedParameters(tuple(names[:_MAX_AFFECTED_NAMES]), total_count)


def model_parameters_are_finite(model: nn.Module) -> bool:
    """Check all model parameters with at most one host synchronization per device."""

    finite_by_device: dict[torch.device, Tensor] = {}
    for parameter in model.parameters():
        finite = torch.isfinite(parameter.detach()).all()
        previous = finite_by_device.get(parameter.device)
        finite_by_device[parameter.device] = finite if previous is None else previous & finite
    return all(bool(finite.item()) for finite in finite_by_device.values())


def nonfinite_gradient_parameters(model: nn.Module) -> AffectedParameters:
    names: list[str] = []
    total = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None or torch.isfinite(gradient.detach()).all().item():
            continue
        total += 1
        if len(names) < _MAX_AFFECTED_NAMES:
            names.append(name)
    return _bounded_parameter_names(names, total)


def nonfinite_update_parameters(model: nn.Module, optimizer: Optimizer) -> AffectedParameters:
    """Name poisoned parameters and enrich them from optimizer tensor state."""

    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    affected: set[str] = set()

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all().item():
            affected.add(name)

    for parameter, state in optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None:
            continue
        for value in state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value.detach()).all().item():
                affected.add(name)
                break

    ordered = [name for name in parameter_names.values() if name in affected]
    return _bounded_parameter_names(ordered, len(affected))


def _module_names(parameter_names: tuple[str, ...]) -> tuple[str, ...]:
    modules: list[str] = []
    for parameter_name in parameter_names:
        module_name = parameter_name.rpartition(".")[0] or "<root>"
        if module_name not in modules:
            modules.append(module_name)
    return tuple(modules[:_MAX_AFFECTED_NAMES])


def _sanitize_activation_health(
    provider: ActivationHealthProvider | None,
) -> dict[str, bool | int | float | str] | None:
    if provider is None:
        return None
    try:
        payload = provider()
    except Exception:
        return {"provider_error": True}
    if payload is None:
        return None

    safe: dict[str, bool | int | float | str] = {}
    string_keys = sorted(key for key in payload if isinstance(key, str))
    for key in string_keys:
        if len(safe) >= _MAX_ACTIVATION_HEALTH_ITEMS:
            break
        if len(key) > 128:
            continue
        value = payload[key]
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, float):
            safe[key] = value if math.isfinite(value) else "nonfinite"
        elif isinstance(value, Tensor) and value.numel() == 1:
            scalar = float(value.detach().float().item())
            safe[key] = scalar if math.isfinite(scalar) else "nonfinite"
        # Provider strings and compound objects are intentionally excluded.
    return safe


def build_numeric_failure_diagnostics(
    *,
    kind: FailureKind,
    micro_step: int,
    optimizer_step: int,
    tokens_seen: int,
    batch_tokens: int,
    pending_tokens: int,
    precision: str,
    device_type: str,
    learning_rate: float,
    gradient_norm: float | None,
    gradient_norm_finite: bool | None,
    affected: AffectedParameters,
    batch: Mapping[str, Any],
    activation_health_provider: ActivationHealthProvider | None = None,
) -> NumericFailureDiagnostics:
    return NumericFailureDiagnostics(
        schema_version=1,
        kind=kind,
        micro_step=micro_step,
        optimizer_step=optimizer_step,
        tokens_seen=tokens_seen,
        batch_tokens=batch_tokens,
        pending_tokens=pending_tokens,
        precision=precision,
        device_type=device_type,
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        gradient_norm_finite=gradient_norm_finite,
        affected_parameter_names=affected.names,
        affected_module_names=_module_names(affected.names),
        affected_parameter_count=affected.total_count,
        affected_names_truncated=affected.truncated,
        activation_health=_sanitize_activation_health(activation_health_provider),
        batch_identity_sha256=batch_identity_sha256(batch),
    )
