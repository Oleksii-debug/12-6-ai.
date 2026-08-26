"""Fail-closed D05 load guards for checkpoint-v1."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np


def _torch_tensor(x: Any) -> bool:
    c = x.__class__
    return c.__module__.startswith("torch") and c.__name__ in {"Tensor", "Parameter"}


def _dtype_for_target(target: Any, core: Any) -> np.dtype[Any]:
    if isinstance(target, np.ndarray):
        return target.dtype
    if not _torch_tensor(target):
        raise core.CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")
    if str(target.dtype) == "torch.bfloat16":
        return np.dtype(np.uint16)
    try:
        return target.detach().cpu().new_empty((0,)).numpy().dtype
    except (TypeError, RuntimeError) as exc:
        raise core.CheckpointCompatibilityError(
            f"checkpoint cannot represent target dtype {target.dtype} losslessly"
        ) from exc


def _materialize(array: np.ndarray, target: Any, core: Any) -> Any:
    if tuple(array.shape) != tuple(target.shape):
        raise core.CheckpointCompatibilityError(
            f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
        )
    expected = _dtype_for_target(target, core)
    if array.dtype != expected:
        raise core.CheckpointCompatibilityError(
            f"dtype mismatch: checkpoint {array.dtype} vs target storage {expected}"
        )
    if isinstance(target, np.ndarray):
        return array.copy()
    torch = core.importlib.import_module("torch")
    tensor = torch.from_numpy(array.copy())
    if str(target.dtype) == "torch.bfloat16":
        tensor = tensor.view(torch.bfloat16)
    elif tensor.dtype != target.dtype:
        raise core.CheckpointCompatibilityError(
            f"dtype mismatch after decode: checkpoint {tensor.dtype} vs target {target.dtype}"
        )
    return tensor.to(device=target.device)


def _validate_scalars(identity: Any, core: Any) -> None:
    if not isinstance(identity, Mapping):
        raise core.CheckpointIntegrityError("manifest identity must be a mapping")
    pc = identity.get("parameter_count")
    if not isinstance(pc, int) or isinstance(pc, bool) or pc <= 0:
        raise core.CheckpointIntegrityError("identity.parameter_count must be a positive integer")
    for field in ("seed", "step", "tokens_seen"):
        value = identity.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise core.CheckpointIntegrityError(f"identity.{field} must be a non-negative integer")
    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise core.CheckpointIntegrityError("identity.precision must be a non-empty string")


def _shape(x: Any) -> tuple[int, ...] | None:
    if isinstance(x, np.ndarray) or _torch_tensor(x):
        return tuple(x.shape)
    return None


def _generic_compatible(src: Any, dst: Any, path: str, core: Any) -> None:
    ss, ds = _shape(src), _shape(dst)
    if ss is not None or ds is not None:
        if ss != ds:
            raise core.CheckpointCompatibilityError(
                f"optimizer state tensor shape mismatch at {path}: checkpoint {ss} vs target {ds}"
            )
        if isinstance(src, np.ndarray) and isinstance(dst, np.ndarray) and src.dtype != dst.dtype:
            raise core.CheckpointCompatibilityError(
                f"optimizer state dtype mismatch at {path}: checkpoint {src.dtype} vs target {dst.dtype}"
            )
        if _torch_tensor(src) and _torch_tensor(dst) and src.dtype != dst.dtype:
            raise core.CheckpointCompatibilityError(
                f"optimizer state dtype mismatch at {path}: checkpoint {src.dtype} vs target {dst.dtype}"
            )
        return
    if isinstance(src, Mapping) or isinstance(dst, Mapping):
        if not isinstance(src, Mapping) or not isinstance(dst, Mapping) or set(src) != set(dst):
            raise core.CheckpointCompatibilityError(f"optimizer state structure mismatch at {path}")
        for key in dst:
            _generic_compatible(src[key], dst[key], f"{path}.{key}", core)
        return
    seq = lambda x: isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))
    if seq(src) or seq(dst):
        if not seq(src) or not seq(dst) or len(src) != len(dst):
            raise core.CheckpointCompatibilityError(f"optimizer state sequence mismatch at {path}")
        for i, (a, b) in enumerate(zip(src, dst, strict=True)):
            _generic_compatible(a, b, f"{path}[{i}]", core)


def _parameter_state_compatible(value: Any, parameter: Any, path: str, core: Any) -> None:
    shape = _shape(value)
    if shape is not None:
        pshape = tuple(parameter.shape)
        if shape not in {(), pshape}:
            raise core.CheckpointCompatibilityError(
                f"optimizer state tensor shape mismatch at {path}: checkpoint {shape} vs parameter {pshape}"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _parameter_state_compatible(item, parameter, f"{path}.{key}", core)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, item in enumerate(value):
            _parameter_state_compatible(item, parameter, f"{path}[{i}]", core)


def _optimizer_preflight(optimizer: Any, state: Any, core: Any) -> None:
    if not isinstance(state, Mapping) or not hasattr(optimizer, "state_dict"):
        raise core.CheckpointCompatibilityError("optimizer state/target is not loadable")
    current = optimizer.state_dict()
    groups, saved = state.get("param_groups"), state.get("state")
    live_groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list) and isinstance(saved, Mapping) and isinstance(live_groups, list):
        if len(groups) != len(live_groups):
            raise core.CheckpointCompatibilityError("optimizer parameter-group count mismatch")
        by_id: dict[Any, Any] = {}
        for gi, (sg, lg) in enumerate(zip(groups, live_groups, strict=True)):
            ids, params = sg.get("params"), lg.get("params")
            if not isinstance(ids, list) or not isinstance(params, list) or len(ids) != len(params):
                raise core.CheckpointCompatibilityError(
                    f"optimizer param_group[{gi}] parameter count mismatch"
                )
            for pid, param in zip(ids, params, strict=True):
                if pid in by_id:
                    raise core.CheckpointCompatibilityError(f"duplicate optimizer parameter id {pid!r}")
                by_id[pid] = param
        extra = set(saved) - set(by_id)
        if extra:
            raise core.CheckpointCompatibilityError("optimizer state contains unreferenced parameters")
        for pid, per_state in saved.items():
            if not isinstance(per_state, Mapping):
                raise core.CheckpointCompatibilityError(f"optimizer state[{pid!r}] must be a mapping")
            for key, value in per_state.items():
                _parameter_state_compatible(value, by_id[pid], f"state[{pid!r}].{key}", core)
        return
    if not isinstance(current, Mapping):
        raise core.CheckpointCompatibilityError("optimizer.state_dict() must be a mapping")
    _generic_compatible(state, current, "optimizer", core)


def install_checkpoint_hardening(core: Any) -> None:
    """Install the NEXT100-075 fail-closed guards exactly once."""
    if getattr(core, "_NEXT100075_HARDENING_INSTALLED", False):
        return
    old_validate = core._validate_manifest_identity
    old_load = core.load_verified_checkpoint

    def validate(identity: Any) -> None:
        old_validate(identity)
        _validate_scalars(identity, core)

    def materialize(array: np.ndarray, target: Any) -> Any:
        return _materialize(array, target, core)

    def load(verified: Any, *, model: Any, optimizer: Any | None = None,
             scheduler: Any | None = None, strict_model: bool = True,
             restore_rng: bool = True, expected_git_sha: str | None = None,
             expected_model_spec_hash: str | None = None,
             expected_tokenizer_hash: str | None = None,
             expected_tokenizer_vocab_hash: str | None = None,
             expected_dataset_manifest_hash: str | None = None,
             expected_run_manifest_hash: str | None = None) -> Any:
        kwargs = dict(
            expected_git_sha=expected_git_sha,
            expected_model_spec_hash=expected_model_spec_hash,
            expected_tokenizer_hash=expected_tokenizer_hash,
            expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
            expected_dataset_manifest_hash=expected_dataset_manifest_hash,
            expected_run_manifest_hash=expected_run_manifest_hash,
        )
        core.assert_identity(
            verified.manifest,
            git_sha=expected_git_sha,
            model_spec_hash=expected_model_spec_hash,
            tokenizer_hash=expected_tokenizer_hash,
            tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
            dataset_manifest_hash=expected_dataset_manifest_hash,
            run_manifest_hash=expected_run_manifest_hash,
        )
        arrays, combined = core._decode_verified_state(verified)
        core._prepare_model_weights(model, arrays, strict_model)
        if optimizer is not None:
            opt_state = combined.get("optimizer")
            if opt_state is None:
                raise core.CheckpointCompatibilityError(
                    "optimizer was requested but checkpoint has no optimizer state"
                )
            _optimizer_preflight(optimizer, opt_state, core)
        if scheduler is not None and combined.get("scheduler") is None:
            raise core.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        if restore_rng:
            core._preflight_rng_state(combined["rng"])
        return old_load(
            verified, model=model, optimizer=optimizer, scheduler=scheduler,
            strict_model=strict_model, restore_rng=restore_rng, **kwargs
        )

    core._validate_manifest_identity = validate
    core._materialize_for_target = materialize
    core.load_verified_checkpoint = load
    core._NEXT100075_HARDENING_INSTALLED = True
