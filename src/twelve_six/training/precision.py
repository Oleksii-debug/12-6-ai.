"""Fail-closed runtime precision capability resolution for D02 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import PrecisionMode


@dataclass(frozen=True, slots=True)
class PrecisionRuntime:
    """Resolved runtime behavior for one requested training precision mode.

    The requested config value alone is not evidence that the current device can
    execute that mode.  This contract is resolved before Trainer performs any
    model/device, RNG, optimizer, scheduler, or scaler mutation.
    """

    requested: PrecisionMode
    device_type: str
    autocast_enabled: bool
    autocast_dtype: str | None
    grad_scaler_enabled: bool

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def resolve_precision_runtime(
    precision: PrecisionMode,
    device: str | torch.device,
) -> PrecisionRuntime:
    """Resolve a requested precision to a proven runtime policy or fail closed."""
    resolved_device = torch.device(device)
    device_type = resolved_device.type

    if precision == "fp32":
        return PrecisionRuntime(
            requested=precision,
            device_type=device_type,
            autocast_enabled=False,
            autocast_dtype=None,
            grad_scaler_enabled=False,
        )

    if precision == "fp16":
        if device_type != "cuda" or not torch.cuda.is_available():
            raise ValueError("fp16 training requires an available CUDA device")
        return PrecisionRuntime(
            requested=precision,
            device_type=device_type,
            autocast_enabled=True,
            autocast_dtype="float16",
            grad_scaler_enabled=True,
        )

    if precision == "bf16":
        if device_type == "cpu":
            return PrecisionRuntime(
                requested=precision,
                device_type=device_type,
                autocast_enabled=True,
                autocast_dtype="bfloat16",
                grad_scaler_enabled=False,
            )
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("bf16 CUDA training requires an available CUDA device")
            probe = getattr(torch.cuda, "is_bf16_supported", None)
            if probe is None:
                raise RuntimeError(
                    "cannot prove CUDA bf16 support with this PyTorch runtime; "
                    "precision resolution fails closed"
                )
            if not probe():
                raise ValueError("current CUDA device does not report bf16 support")
            return PrecisionRuntime(
                requested=precision,
                device_type=device_type,
                autocast_enabled=True,
                autocast_dtype="bfloat16",
                grad_scaler_enabled=False,
            )
        raise ValueError(
            f"bf16 training is not verified for device type {device_type!r}; "
            "use fp32 or a proven CPU/CUDA bf16 runtime"
        )

    raise ValueError(f"unsupported precision: {precision!r}")


def autocast_dtype(runtime: PrecisionRuntime) -> torch.dtype:
    """Return the torch dtype for an autocast-enabled resolved runtime."""
    if not runtime.autocast_enabled:
        raise ValueError("resolved precision does not enable autocast")
    if runtime.autocast_dtype == "bfloat16":
        return torch.bfloat16
    if runtime.autocast_dtype == "float16":
        return torch.float16
    raise RuntimeError(f"invalid resolved autocast dtype: {runtime.autocast_dtype!r}")
