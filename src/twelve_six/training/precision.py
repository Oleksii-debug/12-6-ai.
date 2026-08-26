"""Fail-closed runtime precision capability resolution for D02 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .config import PrecisionMode


@dataclass(frozen=True, slots=True)
class PrecisionRuntime:
    """Resolved runtime behavior for one requested training precision mode.

    The requested config value alone is not evidence that the current device can
    execute that mode. This contract is resolved before Trainer performs model/device,
    RNG, optimizer, scheduler, or scaler mutation. Mixed-precision modes intentionally
    keep FP32 model parameters as optimizer master weights and lower only eligible
    compute through autocast.
    """

    requested: PrecisionMode
    device_type: str
    parameter_dtype: str
    optimizer_master_dtype: str
    autocast_enabled: bool
    autocast_dtype: str | None
    grad_scaler_enabled: bool
    grad_scaler_device: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def _require_supported_device(precision: PrecisionMode, device: torch.device) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"{precision} training is not verified for device type {device.type!r}; "
            "use an explicitly proven CPU or CUDA runtime"
        )
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise ValueError(f"{precision} CUDA training requires an available CUDA device")
    if device.index is not None:
        visible_devices = torch.cuda.device_count()
        if device.index >= visible_devices:
            raise ValueError(
                f"{precision} CUDA device {device} is not visible; "
                f"runtime reports {visible_devices} visible CUDA device(s)"
            )


def _cuda_native_bf16_supported(device: torch.device) -> bool:
    """Probe native BF16 on the requested CUDA device, excluding emulation when possible."""
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if probe is None:
        raise RuntimeError(
            "cannot prove CUDA bf16 support with this PyTorch runtime; "
            "precision resolution fails closed"
        )

    def _probe_current_device() -> bool:
        try:
            return bool(probe(including_emulation=False))
        except TypeError:
            # Older PyTorch releases exposed only the native-support form with no
            # including_emulation keyword.
            return bool(probe())

    if device.index is None:
        return _probe_current_device()
    with torch.cuda.device(device):
        return _probe_current_device()


def resolve_precision_runtime(
    precision: PrecisionMode,
    device: str | torch.device,
) -> PrecisionRuntime:
    """Resolve a requested precision to a proven runtime policy or fail closed."""
    resolved_device = torch.device(device)
    _require_supported_device(precision, resolved_device)
    device_type = resolved_device.type

    if precision == "fp32":
        return PrecisionRuntime(
            requested=precision,
            device_type=device_type,
            parameter_dtype="float32",
            optimizer_master_dtype="float32",
            autocast_enabled=False,
            autocast_dtype=None,
            grad_scaler_enabled=False,
            grad_scaler_device=None,
        )

    if precision == "fp16":
        if device_type != "cuda":
            raise ValueError(
                "fp16 training requires a CUDA device; an available CUDA device must be selected"
            )
        return PrecisionRuntime(
            requested=precision,
            device_type=device_type,
            parameter_dtype="float32",
            optimizer_master_dtype="float32",
            autocast_enabled=True,
            autocast_dtype="float16",
            grad_scaler_enabled=True,
            grad_scaler_device="cuda",
        )

    if precision == "bf16":
        if device_type == "cpu":
            return PrecisionRuntime(
                requested=precision,
                device_type=device_type,
                parameter_dtype="float32",
                optimizer_master_dtype="float32",
                autocast_enabled=True,
                autocast_dtype="bfloat16",
                grad_scaler_enabled=False,
                grad_scaler_device=None,
            )
        if not _cuda_native_bf16_supported(resolved_device):
            raise ValueError("current CUDA device does not report native bf16 support")
        return PrecisionRuntime(
            requested=precision,
            device_type=device_type,
            parameter_dtype="float32",
            optimizer_master_dtype="float32",
            autocast_enabled=True,
            autocast_dtype="bfloat16",
            grad_scaler_enabled=False,
            grad_scaler_device=None,
        )

    raise ValueError(f"unsupported precision: {precision!r}")


def validate_master_weight_semantics(model: nn.Module, runtime: PrecisionRuntime) -> None:
    """Require FP32 model parameters so AdamW updates true FP32 master weights.

    The Trainer uses native autocast rather than converting the module to bf16/fp16.
    Accepting an already down-cast model would silently change optimizer semantics and
    checkpoint precision identity, so that combination fails before device transfer or
    optimizer construction.
    """
    expected = torch.float32
    invalid = [
        f"{name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != expected
    ]
    if invalid:
        preview = ", ".join(invalid[:3])
        suffix = "" if len(invalid) <= 3 else f", ... ({len(invalid)} parameters total)"
        raise ValueError(
            f"{runtime.requested} training requires FP32 model parameters as optimizer "
            f"master weights before autocast; found {preview}{suffix}"
        )


def autocast_dtype(runtime: PrecisionRuntime) -> torch.dtype:
    """Return the torch dtype for an autocast-enabled resolved runtime."""
    if not runtime.autocast_enabled:
        raise ValueError("resolved precision does not enable autocast")
    if runtime.autocast_dtype == "bfloat16":
        return torch.bfloat16
    if runtime.autocast_dtype == "float16":
        return torch.float16
    raise RuntimeError(f"invalid resolved autocast dtype: {runtime.autocast_dtype!r}")
