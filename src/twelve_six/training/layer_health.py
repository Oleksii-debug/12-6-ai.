"""Temporary per-layer health probes for 12-6 training diagnostics.

Hooks exist only inside explicit diagnostic windows. The probe performs a
side-effect-free forward/backward, records bounded health statistics, and removes
all hooks in ``finally`` without changing decoder forward semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.observability import TrainingObserver
from twelve_six.training.s0_evidence_contract import validate_locked_environment_evidence
from twelve_six.training.trainer import Trainer

SCHEMA_VERSION = "12-6.layer-health.v1"
AUTHORITY = "LOCAL_FREE_LAYER_HEALTH_DIAGNOSTIC_NOT_STAGE_OR_QUALITY_EVIDENCE"
REPOSITORY = "Oleksii-debug/12-6-ai."
DEFAULT_DIAGNOSTIC_STEPS = (0, 4, 16, 64)
_EXPECTED_PARAMETERS = (95_568, 467_808, 1_037_696)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(tensor: Tensor) -> dict[str, float | bool]:
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    if value.numel() == 0:
        return {
            "finite": finite,
            "mean": 0.0,
            "variance": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
        }
    return {
        "finite": finite,
        "mean": float(value.mean().item()),
        "variance": float(value.var(unbiased=False).item()),
        "rms": float(value.square().mean().sqrt().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _parameter_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().double()
        squared += float(grad.square().sum().item())
    return math.sqrt(squared)


def _global_grad_norm(model: nn.Module) -> float:
    return _parameter_grad_norm(tuple(model.parameters()))


def hook_count(model: nn.Module) -> int:
    """Return registered module-hook count for cleanup assertions."""
    total = 0
    for module in model.modules():
        total += len(module._forward_hooks)
        total += len(module._forward_pre_hooks)
        total += len(module._backward_hooks)
    return total


def _clone_existing_grads(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _restore_existing_grads(model: nn.Module, saved: Mapping[str, Tensor]) -> None:
    for name, parameter in model.named_parameters():
        value = saved.get(name)
        parameter.grad = None if value is None else value.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )


def _capture_rng_state() -> tuple[Tensor, list[Tensor] | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda


def _restore_rng_state(state: tuple[Tensor, list[Tensor] | None]) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _profile(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "first": None,
            "last": None,
            "minimum": None,
            "maximum": None,
            "endpoint_ratio": None,
            "max_to_min_ratio": None,
            "log_slope_per_layer": None,
        }
    first = float(values[0])
    last = float(values[-1])
    minimum = min(float(value) for value in values)
    maximum = max(float(value) for value in values)
    endpoint = last / first if first > 0.0 else None
    spread = maximum / minimum if minimum > 0.0 else None
    slope: float | None = None
    positive = [float(value) for value in values]
    if len(positive) > 1 and all(value > 0.0 for value in positive):
        x = torch.arange(len(positive), dtype=torch.float64)
        y = torch.tensor([math.log(value) for value in positive], dtype=torch.float64)
        x_centered = x - x.mean()
        denominator = float((x_centered * x_centered).sum().item())
        if denominator > 0.0:
            slope = float((x_centered * (y - y.mean())).sum().item() / denominator)
    return {
        "first": first,
        "last": last,
        "minimum": minimum,
        "maximum": maximum,
        "endpoint_ratio": endpoint,
        "max_to_min_ratio": spread,
        "log_slope_per_layer": slope,
    }


def detect_depth_health(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify severe depth trends with explicit engineering heuristics."""
    if not layers:
        raise ValueError("layers must be non-empty")
    residual = [float(layer["residual_out"]["rms"]) for layer in layers]
    attention_grad = [float(layer["gradient_norms"]["attention"]) for layer in layers]
    mlp_grad = [float(layer["gradient_norms"]["mlp"]) for layer in layers]
    norm_grad = [float(layer["gradient_norms"]["norm"]) for layer in layers]
    profiles = {
        "residual_rms": _profile(residual),
        "attention_grad_norm": _profile(attention_grad),
        "mlp_grad_norm": _profile(mlp_grad),
        "norm_grad_norm": _profile(norm_grad),
    }
    signals: list[str] = []
    residual_endpoint = profiles["residual_rms"]["endpoint_ratio"]
    residual_spread = profiles["residual_rms"]["max_to_min_ratio"]
    if residual_endpoint is not None and residual_endpoint >= 2.5:
        signals.append("RESIDUAL_DEPTH_EXPLOSION_ENDPOINT")
    if residual_endpoint is not None and residual_endpoint <= 0.4:
        signals.append("RESIDUAL_DEPTH_VANISHING_ENDPOINT")
    if residual_spread is not None and residual_spread >= 4.0:
        signals.append("RESIDUAL_DEPTH_SPREAD_LARGE")
    for name in ("attention_grad_norm", "mlp_grad_norm", "norm_grad_norm"):
        endpoint = profiles[name]["endpoint_ratio"]
        spread = profiles[name]["max_to_min_ratio"]
        label = name.upper()
        if endpoint is not None and endpoint >= 10.0:
            signals.append(f"{label}_DEPTH_EXPLOSION_ENDPOINT")
        if endpoint is not None and endpoint <= 0.1:
            signals.append(f"{label}_DEPTH_VANISHING_ENDPOINT")
        if spread is not None and spread >= 50.0:
            signals.append(f"{label}_DEPTH_SPREAD_LARGE")
    return {
        "status": "HEALTHY_NO_SEVERE_DEPTH_TREND" if not signals else "DEPTH_TREND_WARNING",
        "signals": signals,
        "profiles": profiles,
        "heuristics": {
            "residual_endpoint_explosion": 2.5,
            "residual_endpoint_vanishing": 0.4,
            "residual_max_to_min_warning": 4.0,
            "gradient_endpoint_explosion": 10.0,
            "gradient_endpoint_vanishing": 0.1,
            "gradient_max_to_min_warning": 50.0,
            "theoretical_thresholds_claimed": False,
        },
    }


def _register_layer_hooks(
    model: TwelveSixDecoder,
    activations: list[dict[str, Any]],
) -> list[Any]:
    handles: list[Any] = []
    for index, block in enumerate(model.blocks):
        record: dict[str, Any] = {"layer": index}
        activations.append(record)

        def residual_in_hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            target: dict[str, Any] = record,
        ) -> None:
            if inputs and isinstance(inputs[0], Tensor):
                target["residual_in"] = _tensor_stats(inputs[0])

        def residual_after_attn_hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            target: dict[str, Any] = record,
        ) -> None:
            if inputs and isinstance(inputs[0], Tensor):
                target["residual_after_attention"] = _tensor_stats(inputs[0])

        def output_hook(key: str, *, target: dict[str, Any] = record):
            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
            ) -> None:
                if isinstance(output, Tensor):
                    target[key] = _tensor_stats(output)

            return hook

        handles.append(block.register_forward_pre_hook(residual_in_hook))
        handles.append(block.attn_norm.register_forward_hook(output_hook("attention_norm_out")))
        handles.append(block.attn.register_forward_hook(output_hook("attention_out")))
        handles.append(block.mlp_norm.register_forward_pre_hook(residual_after_attn_hook))
        handles.append(block.mlp_norm.register_forward_hook(output_hook("mlp_norm_out")))
        handles.append(block.mlp.register_forward_hook(output_hook("mlp_out")))
        handles.append(block.register_forward_hook(output_hook("residual_out")))
    return handles


def capture_layer_health_window(
    model: TwelveSixDecoder,
    batch: Mapping[str, Tensor],
    *,
    label: str,
    optimizer_step: int,
    tokens_seen: int,
    gradient_clip_norm: float | None,
) -> dict[str, Any]:
    """Run one temporary-hook diagnostic forward/backward without training mutation."""
    if "input_ids" not in batch:
        raise KeyError("diagnostic batch must contain input_ids")
    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids)
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("diagnostic input_ids/labels must have identical [batch,time] shape")
    hook_count_before = hook_count(model)
    saved_grads = _clone_existing_grads(model)
    rng_state = _capture_rng_state()
    activations: list[dict[str, Any]] = []
    handles: list[Any] = []
    started = time.perf_counter()
    try:
        model.zero_grad(set_to_none=True)
        handles = _register_layer_hooks(model, activations)
        output = model(input_ids)
        loss = causal_lm_loss(output.logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("diagnostic layer-health loss is non-finite")
        loss.backward()
        for index, block in enumerate(model.blocks):
            norm_parameters = tuple(block.attn_norm.parameters()) + tuple(
                block.mlp_norm.parameters()
            )
            activations[index]["gradient_norms"] = {
                "attention": _parameter_grad_norm(tuple(block.attn.parameters())),
                "mlp": _parameter_grad_norm(tuple(block.mlp.parameters())),
                "norm": _parameter_grad_norm(norm_parameters),
                "attention_norm": _parameter_grad_norm(tuple(block.attn_norm.parameters())),
                "mlp_norm": _parameter_grad_norm(tuple(block.mlp_norm.parameters())),
            }
        global_grad_norm = _global_grad_norm(model)
        depth_health = detect_depth_health(activations)
        clip_factor = None
        would_clip = False
        if gradient_clip_norm is not None:
            clip_factor = min(1.0, float(gradient_clip_norm) / max(global_grad_norm, 1e-30))
            would_clip = global_grad_norm > float(gradient_clip_norm)
        return {
            "label": label,
            "optimizer_step": optimizer_step,
            "tokens_seen": tokens_seen,
            "loss": float(loss.detach().item()),
            "layers": activations,
            "global_gradient_norm": global_grad_norm,
            "clipping": {
                "configured_gradient_clip_norm": gradient_clip_norm,
                "would_clip_this_diagnostic_gradient": would_clip,
                "hypothetical_clip_factor": clip_factor,
                "clipping_applied_by_diagnostic": False,
            },
            "depth_health": depth_health,
            "init_spec": model.init_spec.to_dict(),
            "init_identity_sha256": model.init_spec.identity_sha256(),
            "residual_output_init_std": model.init_spec.residual_std(model.spec.n_layers),
            "diagnostic_seconds": time.perf_counter() - started,
            "hook_count_before": hook_count_before,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        _restore_existing_grads(model, saved_grads)
        _restore_rng_state(rng_state)
        hook_count_after = hook_count(model)
        if hook_count_after != hook_count_before:
            raise RuntimeError(
                "layer-health hooks leaked outside diagnostic window: "
                f"before={hook_count_before}, after={hook_count_after}"
            )


def _model_spec(
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    d_ff: int,
) -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=head_dim,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def controlled_layer_health_specs() -> tuple[ModelSpec, ...]:
    """Exact ~100K/~500K/~1M subset of the incumbent RESEARCH41 family."""
    specs = (
        _model_spec(d_model=48, n_layers=3, n_heads=4, head_dim=12, d_ff=128),
        _model_spec(d_model=96, n_layers=4, n_heads=6, head_dim=16, d_ff=256),
        _model_spec(d_model=128, n_layers=5, n_heads=8, head_dim=16, d_ff=352),
    )
    counts = tuple(spec.parameter_count() for spec in specs)
    if counts != _EXPECTED_PARAMETERS:
        raise RuntimeError(f"controlled layer-health family drift: {counts!r}")
    return specs


def _read_training_stream(root: Path) -> tuple[bytes, dict[str, Any]]:
    path = root / "data/s0/packaged/train.jsonl"
    tokenizer = ByteTokenizer()
    encoded: list[bytes] = []
    record_ids: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise TypeError(f"{path}:{line_number} must be an object")
        text = record.get("text")
        record_id = record.get("id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} missing text")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number} missing id")
        encoded.append(bytes(tokenizer.encode(text)))
        record_ids.append(record_id)
    if not encoded:
        raise ValueError("training fixture is empty")
    stream = b"\n".join(encoded) + b"\n"
    return stream, {
        "train_jsonl_sha256": _file_sha256(path),
        "record_ids": record_ids,
        "unique_stream_bytes": len(stream),
    }


def _make_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Tensor]:
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append(
            [stream[(start + offset) % len(stream)] for offset in range(sequence_length)]
        )
    input_ids = torch.tensor(rows, dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _trainer_config(*, max_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def run_layer_health_matrix(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    output_path: str | Path | None = None,
    seed: int = 1337,
    max_steps: int = 64,
    diagnostic_steps: Sequence[int] = DEFAULT_DIAGNOSTIC_STEPS,
    batch_size: int = 4,
    sequence_length: int = 64,
) -> dict[str, Any]:
    """Execute the fixed-control three-scale layer-health matrix."""
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be lowercase 40-hex")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    steps = tuple(int(step) for step in diagnostic_steps)
    if not steps or steps[0] != 0 or tuple(sorted(set(steps))) != steps:
        raise ValueError("diagnostic_steps must be unique, increasing, and begin at 0")
    if steps[-1] > max_steps:
        raise ValueError("diagnostic step exceeds max_steps")
    if batch_size <= 0 or sequence_length < 2 or sequence_length > 256:
        raise ValueError("invalid batch_size/sequence_length")
    root_path = Path(root).resolve()
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    stream, data_identity = _read_training_stream(root_path)
    init_spec = InitSpec()
    runs: list[dict[str, Any]] = []
    for spec in controlled_layer_health_specs():
        config = _trainer_config(max_steps=max_steps, seed=seed)
        run_identity = {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "probe_schema": SCHEMA_VERSION,
            "model_spec": spec.to_dict(),
            "model_identity_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init_spec.to_dict(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "training_config": asdict(config),
            "data": data_identity,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "diagnostic_steps": list(steps),
            "environment": dict(environment),
        }
        observer = TrainingObserver(
            run_identity,
            device="cpu",
            max_step_samples=max_steps + 1,
            gpu_sample_every_steps=max_steps + 1,
        )
        torch.manual_seed(seed)
        model = TwelveSixDecoder(spec, init_spec)
        trainer = Trainer(model, config, device="cpu")
        windows: list[dict[str, Any]] = []

        def capture(step: int) -> None:
            counters_before = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
            state_before = {
                name: parameter.detach().clone() for name, parameter in model.named_parameters()
            }
            window = capture_layer_health_window(
                model,
                _make_batch(
                    stream,
                    step=step,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                ),
                label="initialization" if step == 0 else f"optimizer_step_{step}",
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            counters_after = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
            if counters_after != counters_before:
                raise RuntimeError("diagnostic window mutated Trainer counters")
            if any(
                not torch.equal(parameter.detach(), state_before[name])
                for name, parameter in model.named_parameters()
            ):
                raise RuntimeError("diagnostic window mutated model parameters")
            window["trainer_counters_preserved"] = True
            window["model_parameters_preserved"] = True
            window["hooks_removed_cleanly"] = True
            windows.append(window)

        capture(0)
        for step in range(1, max_steps + 1):
            batch = _make_batch(
                stream,
                step=step - 1,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            observer.train_microbatch(trainer, batch, data_wait_seconds=0.0)
            if step in steps[1:]:
                capture(step)
        runs.append(
            {
                "parameter_count": spec.parameter_count(),
                "model_identity_sha256": spec.identity_sha256(),
                "n_layers": spec.n_layers,
                "d_model": spec.d_model,
                "windows": windows,
                "telemetry": observer.export(),
                "diagnostic_seconds_total": sum(
                    float(window["diagnostic_seconds"]) for window in windows
                ),
                "hook_cleanup_all_windows": all(
                    bool(window["hooks_removed_cleanly"]) for window in windows
                ),
            }
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "controls": {
            "seed": seed,
            "max_steps": max_steps,
            "diagnostic_steps": list(steps),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "init_spec": init_spec.to_dict(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "residual_branch_scale": init_spec.residual_branch_scale,
            "gradient_clip_norm": 1.0,
            "forward_semantics_modified": False,
            "hooks_active_outside_diagnostic_windows": False,
        },
        "data": data_identity,
        "runs": runs,
        "cross_scale": [
            {
                "parameter_count": run["parameter_count"],
                "initial_depth_health": run["windows"][0]["depth_health"],
                "final_depth_health": run["windows"][-1]["depth_health"],
                "initial_global_gradient_norm": run["windows"][0]["global_gradient_norm"],
                "final_global_gradient_norm": run["windows"][-1]["global_gradient_norm"],
                "final_would_clip": run["windows"][-1]["clipping"][
                    "would_clip_this_diagnostic_gradient"
                ],
            }
            for run in runs
        ],
        "interpretation_contract": {
            "init_relation": (
                "Current InitSpec uses Normal(0,0.02) and scales attention/MLP residual "
                "output projections by 0.02/sqrt(2L); residual-depth trends are evaluated "
                "against that incumbent scaling without changing it."
            ),
            "clipping_relation": (
                "Diagnostic gradients are pre-clip. The report records whether the current "
                "1.0 global-norm clip would engage and its hypothetical factor; the probe "
                "never applies clipping or an optimizer update."
            ),
            "depth_thresholds_are_engineering_heuristics": True,
        },
        "truth_boundary": {
            "paid_compute_used": False,
            "stage_or_architecture_frozen": False,
            "quality_or_capability_evidence": False,
            "representative_corpus_claim": False,
            "model_forward_modified": False,
            "optimizer_state_modified_by_diagnostics": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_layer_health_report(report, expected_source_sha=source_sha)
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def validate_layer_health_report(
    report: Mapping[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong layer-health schema")
    if report.get("authority") != AUTHORITY:
        raise ValueError("wrong layer-health authority")
    source = report.get("source")
    if not isinstance(source, Mapping) or source.get("repository") != REPOSITORY:
        raise ValueError("wrong layer-health repository")
    source_sha = source.get("git_sha")
    if not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ValueError("invalid layer-health source SHA")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("layer-health source SHA mismatch")
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("layer-health report self-hash mismatch")
    controls = report.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("layer-health controls missing")
    if controls.get("forward_semantics_modified") is not False:
        raise ValueError("forward-semantics overclaim")
    if controls.get("hooks_active_outside_diagnostic_windows") is not False:
        raise ValueError("hook-window contract weakened")
    if controls.get("gradient_clip_norm") != 1.0:
        raise ValueError("unexpected clipping control")
    if controls.get("residual_branch_scale") != "sqrt_2_layers":
        raise ValueError("unexpected InitSpec residual scaling")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("layer-health runs missing")
    if [run.get("parameter_count") for run in runs] != list(_EXPECTED_PARAMETERS):
        raise ValueError("layer-health controlled scales drifted")
    expected_steps = controls.get("diagnostic_steps")
    for run in runs:
        windows = run.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("layer-health windows missing")
        if [window.get("optimizer_step") for window in windows] != expected_steps:
            raise ValueError("layer-health diagnostic step coverage drifted")
        if len(windows[0].get("layers", [])) != run.get("n_layers"):
            raise ValueError("layer-health per-layer coverage mismatch")
        if run.get("hook_cleanup_all_windows") is not True:
            raise ValueError("layer-health hook cleanup failed")
        for window in windows:
            if window.get("hooks_removed_cleanly") is not True:
                raise ValueError("diagnostic hook leak reported")
            if window.get("trainer_counters_preserved") is not True:
                raise ValueError("diagnostic mutated Trainer counters")
            if window.get("model_parameters_preserved") is not True:
                raise ValueError("diagnostic mutated model parameters")
            if not math.isfinite(float(window.get("global_gradient_norm"))):
                raise ValueError("non-finite diagnostic global gradient norm")
            layers = window.get("layers")
            if not isinstance(layers, list):
                raise ValueError("layer records missing")
            for layer in layers:
                for key in (
                    "residual_in",
                    "attention_norm_out",
                    "attention_out",
                    "residual_after_attention",
                    "mlp_norm_out",
                    "mlp_out",
                    "residual_out",
                ):
                    stats = layer.get(key)
                    if not isinstance(stats, Mapping) or stats.get("finite") is not True:
                        raise ValueError(f"invalid activation statistics: {key}")
                    for field in ("variance", "rms"):
                        value = float(stats[field])
                        if not math.isfinite(value) or value < 0.0:
                            raise ValueError(f"invalid activation {field}: {key}")
                gradients = layer.get("gradient_norms")
                if not isinstance(gradients, Mapping):
                    raise ValueError("layer gradient norms missing")
                for key in ("attention", "mlp", "norm", "attention_norm", "mlp_norm"):
                    value = float(gradients[key])
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(f"invalid layer gradient norm: {key}")
    truth = report.get("truth_boundary")
    if not isinstance(truth, Mapping):
        raise ValueError("layer-health truth boundary missing")
    forbidden = (
        "paid_compute_used",
        "stage_or_architecture_frozen",
        "quality_or_capability_evidence",
        "representative_corpus_claim",
        "model_forward_modified",
        "optimizer_state_modified_by_diagnostics",
    )
    if any(truth.get(key) is not False for key in forbidden):
        raise ValueError("layer-health truth boundary weakened")
