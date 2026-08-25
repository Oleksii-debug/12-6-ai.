"""Measured LOCAL_FREE CPU profiling for the exact S0 implementation.

This module is deliberately additive: it composes existing model, data/packing,
training, checkpoint, and inference contracts without modifying their semantics.
Timing values are observations of one CI/local host, not capacity or SLA claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    detect_git_sha,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import (
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    batch_examples,
    collate_rows,
    iter_packed_examples,
    load_jsonl_records,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.s0_evidence import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    TRAIN_JSONL_SHA256,
    VALIDATION_JSONL_SHA256,
    run_s0_training_evidence,
)

SCHEMA_VERSION = "12-6.s0-cpu-profile.v1"
AUTHORITY = "LOCAL_FREE_CPU_PROFILE_NOT_CAPACITY_OR_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
_REQUIRED_PHASES = frozenset(
    {
        "seed_and_model_construction",
        "train_split_read_tokenize_pack",
        "forward_eval",
        "train_microbatch_forward_backward_update",
        "checkpoint_save",
        "checkpoint_verify",
        "checkpoint_load",
        "greedy_generation",
    }
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_sha(source_sha: str) -> None:
    if len(source_sha) != 40 or source_sha != source_sha.lower():
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")
    if any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")


def _rss_high_water_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _summarize_samples(
    wall_seconds: list[float],
    cpu_seconds: list[float],
) -> dict[str, Any]:
    if not wall_seconds or len(wall_seconds) != len(cpu_seconds):
        raise ValueError("timing samples must be non-empty and aligned")
    if any(value <= 0 or not math.isfinite(value) for value in wall_seconds):
        raise RuntimeError("wall timing produced a non-positive/non-finite sample")
    if any(value < 0 or not math.isfinite(value) for value in cpu_seconds):
        raise RuntimeError("CPU timing produced a negative/non-finite sample")
    return {
        "repetitions": len(wall_seconds),
        "wall_seconds": {
            "min": min(wall_seconds),
            "median": statistics.median(wall_seconds),
            "max": max(wall_seconds),
        },
        "process_cpu_seconds": {
            "min": min(cpu_seconds),
            "median": statistics.median(cpu_seconds),
            "max": max(cpu_seconds),
        },
    }


def _measure_prepared(
    prepare: Callable[[], Callable[[], Any]],
    *,
    repetitions: int,
    warmups: int = 0,
) -> dict[str, Any]:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be >= 1 and warmups must be >= 0")
    for _ in range(warmups):
        prepare()()
    wall: list[float] = []
    cpu: list[float] = []
    for _ in range(repetitions):
        operation = prepare()
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        operation()
        cpu.append((time.process_time_ns() - cpu_start) / 1_000_000_000)
        wall.append((time.perf_counter_ns() - wall_start) / 1_000_000_000)
    return _summarize_samples(wall, cpu)


def _tensor_batches(
    root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], int, int]:
    records = tuple(
        load_jsonl_records(root / f"data/s0/packaged/{split}.jsonl", split=split)
    )
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    if not records or not examples:
        raise RuntimeError(f"S0 {split} split must be non-empty")
    batches: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        batches.append(
            {
                "input_ids": torch.tensor(rows["input_ids"], dtype=torch.long),
                "labels": torch.tensor(rows["labels"], dtype=torch.long),
            }
        )
    return (
        batches,
        sum(example.num_loss_tokens for example in examples),
        sum(len(record.text.encode("utf-8")) for record in records),
    )


def _state_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_state_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_tensor_bytes(item) for item in value)
    return 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _training_config(seed: int, *, max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _checkpoint_identity(
    *,
    root: Path,
    source_sha: str,
    stage: Any,
    tokenizer: ByteTokenizer,
    trainer: Trainer,
    config: TrainerConfig,
) -> Any:
    environment_lock_hash = _sha256_file(root / "requirements/locks/index.json")
    run_manifest = {
        "schema_version": "12-6.w5-s0-profile-run.v1",
        "run_id": "W5-S0-CPU-PROFILE",
        "stage": "S0",
        "run_kind": "LOCAL_FREE_CPU_PROFILE",
        "candidate": {
            "git_sha": source_sha,
            "modelspec_sha256": stage.model.identity_sha256(),
            "initspec_sha256": stage.init.identity_sha256(),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "split_identity": TRAIN_JSONL_SHA256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
        },
        "training": {
            "seed": config.seed,
            "precision": config.precision,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            "scheduler": None,
        },
        "environment": {"lock_sha256": environment_lock_hash},
    }
    return bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=stage.model.to_dict(),
        init_spec=stage.init.to_dict(),
        tokenizer_identity=tokenizer.identity.to_dict(),
        packing_identity={
            "version": PACKING_VERSION,
            "config_sha256": PACKING_CONFIG_HASH,
        },
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        environment_lock_hash=environment_lock_hash,
    )


def run_s0_cpu_profile(
    root: str | Path,
    *,
    source_sha: str,
    seed: int = 1337,
    training_steps: int = 40,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Measure bounded S0 CPU mechanics without claiming machine capacity."""
    _validate_source_sha(source_sha)
    if training_steps < 1:
        raise ValueError("training_steps must be >= 1")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    root = Path(root).resolve()
    live_sha = detect_git_sha(root)
    if live_sha is None:
        raise RuntimeError("S0 profile requires a Git checkout to prove exact source identity")
    if live_sha != source_sha:
        raise RuntimeError(
            f"S0 profile source mismatch: declared={source_sha} checkout={live_sha}"
        )

    stage = load_stage_config(root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    if stage.expected_parameters != 10_140:
        raise RuntimeError("profiling requires the exact 10,140-parameter S0 stage")
    if stage.model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("S0 model/tokenizer vocabulary mismatch")

    rss_before = _rss_high_water_bytes()
    phases: dict[str, Any] = {}

    def prepare_construct() -> Callable[[], Any]:
        def operation() -> TwelveSixDecoder:
            torch.manual_seed(seed)
            return TwelveSixDecoder(stage.model, stage.init)

        return operation

    phases["seed_and_model_construction"] = _measure_prepared(
        prepare_construct,
        repetitions=repetitions,
    )

    packing_observation: dict[str, int] = {}

    def prepare_packing() -> Callable[[], Any]:
        def operation() -> None:
            batches, loss_tokens, source_bytes = _tensor_batches(
                root,
                split="train",
                tokenizer=tokenizer,
                batch_size=3,
            )
            packing_observation["batches"] = len(batches)
            packing_observation["loss_tokens"] = loss_tokens
            packing_observation["source_bytes"] = source_bytes

        return operation

    phases["train_split_read_tokenize_pack"] = _measure_prepared(
        prepare_packing,
        repetitions=repetitions,
    )
    phases["train_split_read_tokenize_pack"]["work"] = dict(packing_observation)

    train_batches, train_loss_tokens, _ = _tensor_batches(
        root,
        split="train",
        tokenizer=tokenizer,
        batch_size=3,
    )
    _, validation_loss_tokens, _ = _tensor_batches(
        root,
        split="validation",
        tokenizer=tokenizer,
        batch_size=3,
    )
    representative_batch = train_batches[0]

    torch.manual_seed(seed)
    forward_model = TwelveSixDecoder(stage.model, stage.init)
    forward_model.eval()
    input_ids = representative_batch["input_ids"]

    def prepare_forward() -> Callable[[], Any]:
        def operation() -> None:
            with torch.no_grad():
                _ = forward_model(input_ids).logits

        return operation

    phases["forward_eval"] = _measure_prepared(
        prepare_forward,
        repetitions=repetitions,
        warmups=2,
    )
    forward_tokens = int(input_ids.numel())
    forward_median = phases["forward_eval"]["wall_seconds"]["median"]
    phases["forward_eval"]["work"] = {
        "input_tokens": forward_tokens,
        "tokens_per_second_from_median": forward_tokens / forward_median,
    }

    one_step_config = _training_config(seed, max_steps=1)

    def prepare_train_step() -> Callable[[], Any]:
        torch.manual_seed(seed)
        model = TwelveSixDecoder(stage.model, stage.init)
        trainer = Trainer(model, one_step_config, device="cpu")

        def operation() -> None:
            metrics = trainer.train_microbatch(representative_batch)
            if not metrics.optimizer_stepped:
                raise RuntimeError("profile train step did not commit an optimizer update")

        return operation

    phases["train_microbatch_forward_backward_update"] = _measure_prepared(
        prepare_train_step,
        repetitions=repetitions,
        warmups=1,
    )
    scoreable_tokens = int(representative_batch["labels"][:, 1:].ne(-100).sum().item())
    train_median = phases["train_microbatch_forward_backward_update"]["wall_seconds"]["median"]
    phases["train_microbatch_forward_backward_update"]["work"] = {
        "optimized_tokens": scoreable_tokens,
        "tokens_per_second_from_median": scoreable_tokens / train_median,
        "semantic_breakdown": (
            "canonical Trainer.train_microbatch end-to-end; backward and optimizer update "
            "are covered but intentionally not reported as separately timed subphases"
        ),
    }

    checkpoint_config = _training_config(seed, max_steps=3)
    torch.manual_seed(seed)
    checkpoint_model = TwelveSixDecoder(stage.model, stage.init)
    checkpoint_trainer = Trainer(checkpoint_model, checkpoint_config, device="cpu")
    checkpoint_trainer.train_microbatch(train_batches[0])
    checkpoint_trainer.train_microbatch(train_batches[1])
    identity = _checkpoint_identity(
        root=root,
        source_sha=source_sha,
        stage=stage,
        tokenizer=tokenizer,
        trainer=checkpoint_trainer,
        config=checkpoint_config,
    )

    with tempfile.TemporaryDirectory(prefix="w5-s0-profile-") as temporary:
        temporary_root = Path(temporary)
        save_index = 0
        save_directories: list[Path] = []

        def prepare_save() -> Callable[[], Any]:
            nonlocal save_index
            destination = temporary_root / f"checkpoint-{save_index}"
            save_index += 1
            save_directories.append(destination)
            return lambda: save_trainer_checkpoint(
                destination,
                model=checkpoint_model,
                trainer=checkpoint_trainer,
                identity=identity,
            )

        phases["checkpoint_save"] = _measure_prepared(
            prepare_save,
            repetitions=max(1, min(repetitions, 3)),
        )
        checkpoint_dir = save_directories[-1]
        verified = verify_checkpoint(checkpoint_dir)
        checkpoint_bytes = _directory_bytes(checkpoint_dir)
        phases["checkpoint_save"]["work"] = {
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_id": verified["checkpoint_id"],
        }

        phases["checkpoint_verify"] = _measure_prepared(
            lambda: lambda: verify_checkpoint(checkpoint_dir),
            repetitions=repetitions,
            warmups=1,
        )
        phases["checkpoint_verify"]["work"] = {"checkpoint_bytes": checkpoint_bytes}

        def prepare_load() -> Callable[[], Any]:
            torch.manual_seed(seed)
            target_model = TwelveSixDecoder(stage.model, stage.init)
            target_trainer = Trainer(target_model, checkpoint_config, device="cpu")
            return lambda: load_trainer_checkpoint(
                checkpoint_dir,
                model=target_model,
                trainer=target_trainer,
                restore_rng=False,
                expected_git_sha=source_sha,
                expected_model_spec_hash=stage.model.identity_sha256(),
                expected_init_spec_hash=stage.init.identity_sha256(),
                expected_tokenizer_hash=tokenizer.identity.config_sha256,
                expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
                expected_dataset_manifest_hash=DATASET_MANIFEST_SHA256,
                expected_split_identity=TRAIN_JSONL_SHA256,
                expected_packing_hash=PACKING_CONFIG_HASH,
                expected_packing_version=PACKING_VERSION,
                expected_environment_lock_hash=_sha256_file(
                    root / "requirements/locks/index.json"
                ),
                expected_seed=seed,
            )

        phases["checkpoint_load"] = _measure_prepared(
            prepare_load,
            repetitions=max(1, min(repetitions, 3)),
        )
        phases["checkpoint_load"]["work"] = {"checkpoint_bytes": checkpoint_bytes}

    generation_backend = S0TorchInferenceBackend(checkpoint_model, tokenizer)
    generation_config = GenerationConfig(max_new_tokens=16, sample=False, seed=seed)
    generation_prompt = "12-6 profile:"

    def prepare_generation() -> Callable[[], Any]:
        return lambda: generate(
            generation_backend,
            generation_prompt,
            generation_config,
        )

    phases["greedy_generation"] = _measure_prepared(
        prepare_generation,
        repetitions=repetitions,
        warmups=1,
    )
    generation_result = generate(
        generation_backend,
        generation_prompt,
        generation_config,
    )
    generated_tokens = len(generation_result.generated_token_ids)
    generation_median = phases["greedy_generation"]["wall_seconds"]["median"]
    phases["greedy_generation"]["work"] = {
        "prompt_tokens": len(generation_result.prompt_token_ids),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second_from_median": (
            generated_tokens / generation_median if generated_tokens else 0.0
        ),
        "stop_reason": generation_result.stop_reason,
    }

    training_evidence = run_s0_training_evidence(
        root,
        source_sha=source_sha,
        seed=seed,
        max_steps=training_steps,
        batch_size=3,
    )
    training_runtime = training_evidence["runtime"]
    optimized_tokens = int(training_evidence["training"]["optimized_tokens"])
    full_training_wall = float(training_runtime["wall_seconds"])

    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in checkpoint_model.parameters()
    )
    model_buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in checkpoint_model.buffers()
    )
    optimizer_state_bytes = _state_tensor_bytes(checkpoint_trainer.optimizer.state_dict())

    tracemalloc.start()
    torch.manual_seed(seed)
    memory_probe_model = TwelveSixDecoder(stage.model, stage.init)
    memory_probe_trainer = Trainer(memory_probe_model, one_step_config, device="cpu")
    memory_probe_trainer.train_microbatch(representative_batch)
    python_current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del memory_probe_trainer, memory_probe_model
    rss_after = _rss_high_water_bytes()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S0",
        "identity": {
            "parameter_count": stage.expected_parameters,
            "modelspec_sha256": stage.model.identity_sha256(),
            "initspec_sha256": stage.init.identity_sha256(),
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "train_jsonl_sha256": TRAIN_JSONL_SHA256,
            "validation_jsonl_sha256": VALIDATION_JSONL_SHA256,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "packing_config_sha256": PACKING_CONFIG_HASH,
            "environment_lock_sha256": _sha256_file(
                root / "requirements/locks/index.json"
            ),
        },
        "host": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "git_checkout_verified": True,
        },
        "phases": phases,
        "full_training": {
            "steps": training_steps,
            "optimized_tokens": optimized_tokens,
            "wall_seconds": full_training_wall,
            "process_cpu_seconds": float(training_runtime["process_cpu_seconds"]),
            "tokens_per_second": optimized_tokens / full_training_wall,
            "training_evidence_sha256": training_evidence["evidence_sha256"],
            "validation_optimized_tokens": training_evidence["split_isolation"][
                "validation_optimized_tokens"
            ],
        },
        "memory_and_storage": {
            "model_parameter_bytes": model_parameter_bytes,
            "model_buffer_bytes": model_buffer_bytes,
            "optimizer_tensor_bytes_after_two_steps": optimizer_state_bytes,
            "python_tracemalloc_current_bytes": python_current,
            "python_tracemalloc_peak_bytes": python_peak,
            "process_rss_high_water_before_bytes": rss_before,
            "process_rss_high_water_after_bytes": rss_after,
            "note": (
                "tracemalloc measures Python allocations; RSS high-water includes native "
                "allocations but is process-global and monotonic on supported Unix hosts"
            ),
        },
        "dataset_work": {
            "train_loss_tokens_per_epoch": train_loss_tokens,
            "validation_loss_tokens": validation_loss_tokens,
        },
        "truth_boundary": {
            "paid_compute_authorized_or_used": False,
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_training": False,
            "gpu_or_cuda_profile": False,
            "distributed_profile": False,
            "mfu_measured": False,
            "capacity_or_sla_claim": False,
            "audit_or_promotion_claim": False,
            "backward_and_update_separately_timed": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    validate_s0_cpu_profile(payload, expected_source_sha=source_sha)
    return payload


def validate_s0_cpu_profile(
    payload: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> None:
    """Fail closed on drift, malformed metrics, tampering, or truth-boundary escalation."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected S0 profile schema_version")
    if payload.get("authority") != AUTHORITY:
        raise ValueError("unexpected S0 profile authority")
    if payload.get("repository") != REPOSITORY:
        raise ValueError("unexpected repository identity")
    source_sha = payload.get("source_sha")
    if not isinstance(source_sha, str):
        raise TypeError("source_sha is missing")
    _validate_source_sha(source_sha)
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("profile source_sha does not match expected exact head")

    identity = payload.get("identity")
    if not isinstance(identity, dict) or identity.get("parameter_count") != 10_140:
        raise ValueError("profile is not bound to the exact 10,140-parameter S0")
    for key, value in identity.items():
        if key == "parameter_count":
            continue
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"identity {key} must be a 64-hex SHA-256")
        if value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"identity {key} must be a 64-hex SHA-256")

    phases = payload.get("phases")
    if not isinstance(phases, dict) or not _REQUIRED_PHASES.issubset(phases):
        raise ValueError("required profiling phases are missing")
    for name in _REQUIRED_PHASES:
        phase = phases[name]
        if not isinstance(phase, dict) or int(phase.get("repetitions", 0)) < 1:
            raise ValueError(f"phase {name} has invalid repetitions")
        for clock in ("wall_seconds", "process_cpu_seconds"):
            values = phase.get(clock)
            if not isinstance(values, dict):
                raise TypeError(f"phase {name} is missing {clock}")
            triple = [values.get(item) for item in ("min", "median", "max")]
            if not all(isinstance(value, (int, float)) for value in triple):
                raise ValueError(f"phase {name} has non-numeric {clock}")
            if not all(math.isfinite(float(value)) for value in triple):
                raise ValueError(f"phase {name} has non-finite {clock}")
            if float(triple[0]) > float(triple[1]) or float(triple[1]) > float(triple[2]):
                raise ValueError(f"phase {name} has unordered {clock}")
            if clock == "wall_seconds" and float(triple[0]) <= 0:
                raise ValueError(f"phase {name} has non-positive wall time")
            if clock == "process_cpu_seconds" and float(triple[0]) < 0:
                raise ValueError(f"phase {name} has negative CPU time")

    full_training = payload.get("full_training")
    if not isinstance(full_training, dict):
        raise TypeError("full_training evidence is missing")
    if int(full_training.get("steps", 0)) < 1:
        raise ValueError("full_training steps must be positive")
    if int(full_training.get("optimized_tokens", 0)) <= 0:
        raise ValueError("full_training optimized_tokens must be positive")
    if int(full_training.get("validation_optimized_tokens", -1)) != 0:
        raise ValueError("validation tokens must remain unoptimized")
    for key in ("wall_seconds", "process_cpu_seconds", "tokens_per_second"):
        value = full_training.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"full_training {key} must be finite")
    if float(full_training["wall_seconds"]) <= 0:
        raise ValueError("full_training wall_seconds must be positive")
    if float(full_training["tokens_per_second"]) <= 0:
        raise ValueError("full_training tokens_per_second must be positive")

    truth = payload.get("truth_boundary")
    if not isinstance(truth, dict):
        raise TypeError("truth_boundary is missing")
    forbidden_true = {
        "paid_compute_authorized_or_used",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "gpu_or_cuda_profile",
        "distributed_profile",
        "mfu_measured",
        "capacity_or_sla_claim",
        "audit_or_promotion_claim",
        "backward_and_update_separately_timed",
    }
    if any(truth.get(field) is not False for field in forbidden_true):
        raise ValueError("profile truth boundary was escalated or became ambiguous")

    claimed_hash = payload.get("evidence_sha256")
    if not isinstance(claimed_hash, str):
        raise TypeError("evidence_sha256 is missing")
    unhashed = dict(payload)
    unhashed.pop("evidence_sha256", None)
    if claimed_hash != _canonical_hash(unhashed):
        raise ValueError("S0 profile evidence hash mismatch")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute the bounded LOCAL_FREE S0 CPU profile")
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--training-steps", type=int, default=40)
    run.add_argument("--repetitions", type=int, default=5)

    validate = subparsers.add_parser("validate", help="validate a saved profile report")
    validate.add_argument("report", type=Path)
    validate.add_argument("--source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        payload = run_s0_cpu_profile(
            args.root,
            source_sha=args.source_sha,
            seed=args.seed,
            training_steps=args.training_steps,
            repetitions=args.repetitions,
        )
        _write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "evidence_sha256": payload["evidence_sha256"],
                    "training_tokens_per_second": payload["full_training"][
                        "tokens_per_second"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("profile report must be a JSON object")
    validate_s0_cpu_profile(payload, expected_source_sha=args.source_sha)
    print(json.dumps({"valid": True, "evidence_sha256": payload["evidence_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
