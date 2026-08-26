"""PERF-148 bounded PyTorch Profiler evidence for the live S3 ~10M Trainer step.

The profiler is deliberately opt-in and external to the normal training loop. It
uses the incumbent Trainer/model, installs temporary record_function wrappers only
inside the profiler window, and restores every patched callable before returning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import types
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset

import twelve_six.training.trainer as trainer_module
from twelve_six.checkpoint.core import CheckpointIdentity
from twelve_six.checkpoint.trainer_adapter import save_trainer_checkpoint
from twelve_six.model import CausalSelfAttention, RMSNorm, SwiGLU, TwelveSixDecoder
from twelve_six.s3_engineering import (
    S3_CURRENT_CANDIDATE_ID,
    S3_CURRENT_EXPECTED_PARAMETERS,
    S3_CURRENT_MODEL_SHA256,
    s3_current_model_spec,
    s3_init_spec,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.perf148-10m-profiler.v1"
AUTHORITY = "LOCAL_FREE_PERFORMANCE_EVIDENCE_NOT_MODEL_OR_TRAINING_SEMANTICS"


class DeterministicBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, steps: int, batch_size: int, sequence_length: int, seed: int) -> None:
        self.steps = steps
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index)
        input_ids = torch.randint(
            0,
            256,
            (self.batch_size, self.sequence_length),
            generator=generator,
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_tensor(parameter).encode("ascii"))
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": cuda,
        "cuda_runtime": getattr(torch.version, "cuda", None),
    }
    if cuda:
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        payload["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
    return payload


def _trainer_config(*, max_steps: int, precision: str, seed: int) -> TrainerConfig:
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
        precision=precision,
        seed=seed,
        deterministic_algorithms=False,
        deterministic_warn_only=False,
    )


def _checkpoint_identity(config: TrainerConfig, trainer: Trainer, source_sha: str) -> CheckpointIdentity:
    tokenizer = ByteTokenizer()
    run_config = {"trainer": asdict(config), "perf148": True}
    label = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=s3_current_model_spec().to_dict(),
        parameter_count=S3_CURRENT_EXPECTED_PARAMETERS,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=label("PERF-148-controlled-synthetic-trace-v1"),
        run_manifest_hash=label("PERF-148-10M-profiler-v1"),
        training_config=run_config,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=label("PERF-148-local-free-runtime"),
    )


@contextmanager
def _module_regions(model: torch.nn.Module) -> Iterator[None]:
    handles: list[Any] = []
    active: dict[int, list[Any]] = {}

    def install(module: torch.nn.Module, label: str) -> None:
        def before(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
            ctx = record_function(label)
            ctx.__enter__()
            active.setdefault(id(module), []).append(ctx)

        def after(_module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            ctx = active[id(module)].pop()
            ctx.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))

    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            install(module, "perf148::forward_attention")
        elif isinstance(module, SwiGLU):
            install(module, "perf148::forward_mlp")
        elif isinstance(module, RMSNorm):
            install(module, "perf148::forward_rmsnorm")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        if any(active.values()):
            raise RuntimeError("PERF-148 module profiler region leaked")


@contextmanager
def _trainer_regions(trainer: Trainer) -> Iterator[None]:
    original_forward_loss = trainer._forward_loss
    original_normalize = trainer._normalize_gradients_and_norm
    original_lm_loss = trainer_module.causal_lm_loss
    original_pair_loss = trainer_module.causal_pair_loss
    original_autograd_backward = torch.autograd.backward
    original_clip = torch.nn.utils.clip_grad_norm_
    original_optimizer_step = trainer.optimizer.step

    def forward_loss(self: Trainer, *args: Any, **kwargs: Any) -> torch.Tensor:
        with record_function("perf148::forward_and_loss"):
            return original_forward_loss(*args, **kwargs)

    def normalize(self: Trainer, token_count: int) -> torch.Tensor:
        with record_function("perf148::gradient_normalize_and_norm"):
            return original_normalize(token_count)

    def lm_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        with record_function("perf148::loss"):
            return original_lm_loss(*args, **kwargs)

    def pair_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        with record_function("perf148::loss"):
            return original_pair_loss(*args, **kwargs)

    def backward(*args: Any, **kwargs: Any) -> Any:
        with record_function("perf148::backward"):
            return original_autograd_backward(*args, **kwargs)

    def clip(*args: Any, **kwargs: Any) -> Any:
        with record_function("perf148::gradient_clip"):
            return original_clip(*args, **kwargs)

    def optimizer_step(*args: Any, **kwargs: Any) -> Any:
        with record_function("perf148::optimizer_step"):
            return original_optimizer_step(*args, **kwargs)

    trainer._forward_loss = types.MethodType(forward_loss, trainer)
    trainer._normalize_gradients_and_norm = types.MethodType(normalize, trainer)
    trainer_module.causal_lm_loss = lm_loss
    trainer_module.causal_pair_loss = pair_loss
    torch.autograd.backward = backward
    torch.nn.utils.clip_grad_norm_ = clip
    trainer.optimizer.step = optimizer_step
    try:
        yield
    finally:
        trainer._forward_loss = original_forward_loss
        trainer._normalize_gradients_and_norm = original_normalize
        trainer_module.causal_lm_loss = original_lm_loss
        trainer_module.causal_pair_loss = original_pair_loss
        torch.autograd.backward = original_autograd_backward
        torch.nn.utils.clip_grad_norm_ = original_clip
        trainer.optimizer.step = original_optimizer_step


def _event_row(event: Any) -> dict[str, Any]:
    return {
        "key": str(event.key),
        "calls": int(event.count),
        "self_cpu_us": float(getattr(event, "self_cpu_time_total", 0.0)),
        "cpu_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        "self_device_us": float(getattr(event, "self_device_time_total", 0.0)),
        "device_total_us": float(getattr(event, "device_time_total", 0.0)),
        "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
        "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
        "self_device_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
        "device_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
        "input_shapes": getattr(event, "input_shapes", None),
    }


def _summarize(profiler: Any, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = [_event_row(event) for event in profiler.key_averages(group_by_input_shape=True)]
    regions = [row for row in events if row["key"].startswith("perf148::")]
    metric = "device_total_us" if device.type == "cuda" else "cpu_total_us"
    regions.sort(key=lambda row: row[metric], reverse=True)
    operators = [row for row in events if not row["key"].startswith("perf148::")]
    self_metric = "self_device_us" if device.type == "cuda" else "self_cpu_us"
    operators.sort(key=lambda row: row[self_metric], reverse=True)
    return regions, operators[:80]


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    source_sha = os.popen(f"git -C {repo_root} rev-parse HEAD").read().strip()
    if args.source_sha and source_sha != args.source_sha:
        raise RuntimeError(f"source SHA mismatch: expected {args.source_sha}, observed {source_sha}")

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "auto" else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"

    spec = s3_current_model_spec()
    if spec.parameter_count() != S3_CURRENT_EXPECTED_PARAMETERS:
        raise RuntimeError("S3 parameter binding drift")
    if spec.identity_sha256() != S3_CURRENT_MODEL_SHA256:
        raise RuntimeError("S3 model identity drift")
    if args.sequence_length > spec.max_seq_len:
        raise ValueError("sequence length exceeds S3 context")

    total_steps = args.warmup_steps + args.profile_steps
    config = _trainer_config(max_steps=total_steps, precision=precision, seed=args.seed)
    torch.manual_seed(args.seed)
    model = TwelveSixDecoder(spec, s3_init_spec())
    trainer = Trainer(model, config, device=device)
    initial_parameter_sha256 = _parameter_sha256(model)

    dataset = DeterministicBatchDataset(
        steps=total_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    iterator = iter(loader)

    warmup_wall: list[float] = []
    for _ in range(args.warmup_steps):
        batch = next(iterator)
        _sync(device)
        started = time.perf_counter()
        trainer.train_microbatch(batch)
        _sync(device)
        warmup_wall.append(time.perf_counter() - started)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)

    profiled_data_wait: list[float] = []
    profiled_step_wall: list[float] = []
    checkpoint_wall = 0.0
    checkpoint_bytes = 0
    batch_trace: list[str] = []

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        with _module_regions(model), _trainer_regions(trainer):
            for _ in range(args.profile_steps):
                _sync(device)
                started = time.perf_counter()
                with record_function("perf148::data_wait"):
                    batch = next(iterator)
                _sync(device)
                waited = time.perf_counter() - started
                batch_trace.append(_sha256_tensor(batch["input_ids"]))
                profiled_data_wait.append(waited)

                _sync(device)
                started = time.perf_counter()
                with record_function("perf148::trainer_step"):
                    trainer.train_microbatch(batch)
                _sync(device)
                profiled_step_wall.append(time.perf_counter() - started)

            trainer.assert_checkpoint_safe()
            with tempfile.TemporaryDirectory(prefix="perf148-checkpoint-") as temp_dir:
                checkpoint_dir = Path(temp_dir) / "checkpoint"
                identity = _checkpoint_identity(config, trainer, source_sha)
                _sync(device)
                started = time.perf_counter()
                with record_function("perf148::checkpoint_save"):
                    save_trainer_checkpoint(
                        checkpoint_dir,
                        model=model,
                        trainer=trainer,
                        identity=identity,
                    )
                _sync(device)
                checkpoint_wall = time.perf_counter() - started
                checkpoint_bytes = sum(
                    path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
                )

    final_parameter_sha256 = _parameter_sha256(model)
    regions, operators = _summarize(prof, device)
    metric = "device_total_us" if device.type == "cuda" else "cpu_total_us"
    leaf_keys = {
        "perf148::forward_attention",
        "perf148::forward_mlp",
        "perf148::forward_rmsnorm",
        "perf148::loss",
        "perf148::backward",
        "perf148::gradient_normalize_and_norm",
        "perf148::gradient_clip",
        "perf148::optimizer_step",
        "perf148::data_wait",
        "perf148::checkpoint_save",
    }
    bottlenecks = [row for row in regions if row["key"] in leaf_keys]
    bottlenecks.sort(key=lambda row: row[metric], reverse=True)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "swarm_worker_id": "PERF-148-10M-PROFILER",
        "paid_compute": False,
        "candidate": {
            "id": S3_CURRENT_CANDIDATE_ID,
            "parameters": spec.parameter_count(),
            "model_identity_sha256": spec.identity_sha256(),
            "context": spec.max_seq_len,
            "profile_shape": [args.batch_size, args.sequence_length],
        },
        "runtime": _runtime(),
        "execution": {
            "device": str(device),
            "precision": precision,
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
            "warmup_wall_seconds": warmup_wall,
            "data_wait_seconds": profiled_data_wait,
            "step_wall_seconds": profiled_step_wall,
            "checkpoint_wall_seconds": checkpoint_wall,
            "checkpoint_bytes": checkpoint_bytes,
            "tokens_seen": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
        },
        "trace": {
            "seed": args.seed,
            "data_seed": args.data_seed,
            "batch_sha256": batch_trace,
            "initial_parameter_sha256": initial_parameter_sha256,
            "final_parameter_sha256": final_parameter_sha256,
        },
        "regions": regions,
        "top_operators_by_self_time": operators,
        "top_bottlenecks": bottlenecks[:3],
        "profiler": {
            "activities": [activity.name for activity in activities],
            "record_shapes": True,
            "profile_memory": True,
            "with_flops": True,
            "short_explicit_window": True,
            "normal_hot_loop_logging_changed": False,
        },
        "cross_reference": {
            "perf59_torch_compile": "EXISTING_OPT_IN_WORK__DO_NOT_DUPLICATE",
            "native_gqa": "EXISTING_PERF94_WORK__DO_NOT_DUPLICATE_REPEAT_INTERLEAVE_FIX",
        },
        "truth_boundary": {
            "cpu_evidence_is_cpu_only": device.type == "cpu",
            "cuda_claimed_without_cuda": False,
            "model_semantics_changed": False,
            "training_objective_changed": False,
            "paid_compute_used": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.trace_output:
        prof.export_chrome_trace(str(args.trace_output.resolve()))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--data-seed", type=int, default=148000)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    print(json.dumps({
        "source_sha": report["source_sha"],
        "device": report["execution"]["device"],
        "parameters": report["candidate"]["parameters"],
        "step_wall_seconds": report["execution"]["step_wall_seconds"],
        "top_bottlenecks": report["top_bottlenecks"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
