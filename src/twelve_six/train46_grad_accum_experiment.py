"""LOCAL_FREE next-scale gradient-accumulation correctness evidence for TRAIN-46."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .model import InitSpec, TwelveSixDecoder
from .optimizer_dynamics_experiment import (
    BETA1,
    CLIP_NORM,
    EPS,
    LEARNING_RATE,
    SEED,
    SEQUENCE_LENGTH,
    WEIGHT_DECAY,
)
from .scaling_experiment import _byte_stream, _make_batch, _read_jsonl, controlled_specs
from .tokenization import ByteTokenizer
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.train46-grad-accum-correctness.v1"
AUTHORITY = "LOCAL_FREE_CPU_TRAINER_CORRECTNESS_EVIDENCE"
BETA2 = 0.95
EFFECTIVE_BATCH_SIZE = 4
FULL_MICROBATCH_SIZE = 4
ACCUM_MICROBATCH_SIZE = 1
EQUIVALENCE_UPDATES = 12
PARAMETER_MAX_ABS_TOL = 1e-5
PARAMETER_REL_L2_TOL = 1e-5
LOSS_ABS_TOL = 1e-6
OPTIMIZER_STATE_MAX_ABS_TOL = 1e-5


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _config(*, accumulation: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        eps=EPS,
        max_steps=EQUIVALENCE_UPDATES + 2,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=accumulation,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _load_train_stream(repo_root: Path) -> bytes:
    tokenizer = ByteTokenizer()
    records = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    return _byte_stream(records, tokenizer)


def _aligned_effective_batch(
    train_stream: bytes,
    *,
    update: int,
) -> dict[str, torch.Tensor]:
    inputs = _make_batch(
        train_stream,
        step=update,
        batch_size=EFFECTIVE_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)
    targets[:, -1] = 0
    loss_mask = torch.ones_like(inputs, dtype=torch.bool)
    loss_mask[:, -1] = False
    for row in range(EFFECTIVE_BATCH_SIZE):
        loss_mask[row, :row] = False
    return {"input_ids": inputs, "target_ids": targets, "loss_mask": loss_mask}


def _slice_batch(
    batch: dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _flatten_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = [state[key].detach().float().reshape(-1).cpu() for key in sorted(state)]
    return torch.cat(tensors)


def _model_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> tuple[float, float]:
    left_vector = _flatten_state(left)
    right_vector = _flatten_state(right)
    delta = right_vector - left_vector
    max_abs = float(torch.max(torch.abs(delta)).item())
    denominator = float(torch.linalg.vector_norm(left_vector).item())
    relative_l2 = (
        float(torch.linalg.vector_norm(delta).item()) / denominator
        if denominator
        else math.inf
    )
    return max_abs, relative_l2


def _optimizer_tensors(trainer: Trainer) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    state = trainer.optimizer.state_dict()["state"]
    for state_id in sorted(state):
        entry = state[state_id]
        for key in sorted(entry):
            value = entry[key]
            if isinstance(value, torch.Tensor):
                tensors.append(value.detach().float().reshape(-1).cpu())
    return tensors


def _max_tensor_diff(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for left_tensor, right_tensor in zip(left, right, strict=True):
        if left_tensor.shape != right_tensor.shape:
            return math.inf
        if left_tensor.numel():
            difference = torch.max(torch.abs(left_tensor - right_tensor)).item()
            maximum = max(maximum, float(difference))
    return maximum


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() == "Linux" else value


def _worker(
    *,
    repo_root: Path,
    microbatch_size: int,
    state_path: Path,
    metadata_path: Path,
) -> None:
    if EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch_size must divide effective batch size")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_stream = _load_train_stream(repo_root)
    spec = controlled_specs()[3]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    accumulation = EFFECTIVE_BATCH_SIZE // microbatch_size
    trainer = Trainer(model, _config(accumulation=accumulation), device="cpu")
    expected_tokens = 0
    trace: list[dict[str, Any]] = []
    measured_wall = 0.0
    for update in range(EQUIVALENCE_UPDATES):
        effective = _aligned_effective_batch(train_stream, update=update)
        expected_tokens += int(effective["loss_mask"].sum().item())
        final_metrics = None
        started = time.perf_counter()
        for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size):
            final_metrics = trainer.train_microbatch(
                _slice_batch(effective, offset, offset + microbatch_size)
            )
        measured_wall += time.perf_counter() - started
        if final_metrics is None or not final_metrics.optimizer_stepped:
            raise RuntimeError("effective update did not commit exactly once")
        trace.append(
            {
                "optimizer_step": trainer.optimizer_step,
                "valid_causal_tokens": trainer.tokens_seen,
                "update_loss": float(final_metrics.update_loss),
                "grad_norm": float(final_metrics.grad_norm),
            }
        )
    if trainer.optimizer_step != EQUIVALENCE_UPDATES:
        raise RuntimeError("optimizer-step count drift")
    if trainer.tokens_seen != expected_tokens:
        raise RuntimeError(
            f"valid-token accounting mismatch: observed={trainer.tokens_seen} "
            f"expected={expected_tokens}"
        )
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_tensors": _optimizer_tensors(trainer),
        },
        state_path,
    )
    metadata = {
        "parameters": spec.parameter_count(),
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "optimizer_steps": trainer.optimizer_step,
        "valid_causal_tokens": trainer.tokens_seen,
        "expected_valid_causal_tokens": expected_tokens,
        "measured_training_wall_seconds": measured_wall,
        "tokens_per_second": trainer.tokens_seen / measured_wall,
        "peak_rss_bytes": _peak_rss_bytes(),
        "trace": trace,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_worker_process(
    *,
    repo_root: Path,
    microbatch_size: int,
    state_path: Path,
    metadata_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "twelve_six.train46_grad_accum_experiment",
        "--repo-root",
        str(repo_root),
        "--worker-microbatch-size",
        str(microbatch_size),
        "--state-output",
        str(state_path),
        "--metadata-output",
        str(metadata_path),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _checkpoint_boundary_probe(repo_root: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_stream = _load_train_stream(repo_root)
    spec = controlled_specs()[3]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    config = _config(accumulation=4)
    trainer = Trainer(model, config, device="cpu")
    first_effective = _aligned_effective_batch(train_stream, update=0)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    first_metrics = trainer.train_microbatch(_slice_batch(first_effective, 0, 1))
    if first_metrics.optimizer_stepped:
        raise RuntimeError("optimizer stepped before accumulation boundary")
    partial_publish_blocked = False
    partial_error = ""
    try:
        trainer.state_dict()
    except RuntimeError as exc:
        partial_publish_blocked = "mid-accumulation" in str(exc)
        partial_error = str(exc)
    if not partial_publish_blocked:
        raise RuntimeError("mid-accumulation trainer state was publishable")
    precommit_max_abs, _ = _model_diff(
        before,
        {key: value.detach() for key, value in model.state_dict().items()},
    )
    if precommit_max_abs != 0.0:
        raise RuntimeError("model parameters changed before a committed optimizer step")

    committed_metrics = first_metrics
    for offset in (1, 2, 3):
        committed_metrics = trainer.train_microbatch(
            _slice_batch(first_effective, offset, offset + 1)
        )
    if not committed_metrics.optimizer_stepped or trainer.optimizer_step != 1:
        raise RuntimeError("accumulation group did not publish exactly one optimizer step")
    committed_model_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    committed_trainer_state = trainer.state_dict()
    committed_tokens = int(first_effective["loss_mask"].sum().item())
    if trainer.tokens_seen != committed_tokens:
        raise RuntimeError("committed token count drift")

    resumed_model = TwelveSixDecoder(spec, InitSpec())
    resumed_model.load_state_dict(committed_model_state, strict=True)
    resumed_trainer = Trainer(resumed_model, config, device="cpu")
    resumed_trainer.load_state_dict(committed_trainer_state)

    second_effective = _aligned_effective_batch(train_stream, update=1)
    for offset in range(EFFECTIVE_BATCH_SIZE):
        microbatch = _slice_batch(second_effective, offset, offset + 1)
        trainer.train_microbatch(microbatch)
        resumed_trainer.train_microbatch(microbatch)
    resume_max_abs, resume_relative_l2 = _model_diff(
        {key: value.detach() for key, value in model.state_dict().items()},
        {key: value.detach() for key, value in resumed_model.state_dict().items()},
    )
    expected_tokens = committed_tokens + int(second_effective["loss_mask"].sum().item())
    counters_match = (
        trainer.optimizer_step
        == resumed_trainer.optimizer_step
        == 2
        and trainer.tokens_seen
        == resumed_trainer.tokens_seen
        == expected_tokens
    )
    if not counters_match:
        raise RuntimeError("resume replay/omission detected in optimizer-step or token counters")
    return {
        "parameters": spec.parameter_count(),
        "accumulation_steps": 4,
        "partial_state_publish_blocked": partial_publish_blocked,
        "partial_publish_error": partial_error,
        "parameters_unchanged_before_commit": precommit_max_abs == 0.0,
        "committed_state_publish_succeeded": True,
        "committed_optimizer_step": committed_trainer_state.optimizer_step,
        "committed_valid_causal_tokens": committed_tokens,
        "resumed_optimizer_step_after_next_update": resumed_trainer.optimizer_step,
        "resumed_valid_causal_tokens_after_next_update": resumed_trainer.tokens_seen,
        "expected_valid_causal_tokens_after_next_update": expected_tokens,
        "resume_parameter_max_abs_diff_vs_uninterrupted": resume_max_abs,
        "resume_parameter_relative_l2_diff_vs_uninterrupted": resume_relative_l2,
        "duplicate_or_omitted_optimizer_step_detected": False,
        "duplicate_or_omitted_token_detected": False,
    }


def _equivalence_pass(
    *,
    token_match: bool,
    parameter_max_abs: float,
    parameter_relative_l2: float,
    max_loss_diff: float,
    optimizer_max_abs: float,
) -> bool:
    return (
        token_match
        and parameter_max_abs <= PARAMETER_MAX_ABS_TOL
        and parameter_relative_l2 <= PARAMETER_REL_L2_TOL
        and max_loss_diff <= LOSS_ABS_TOL
        and optimizer_max_abs <= OPTIMIZER_STATE_MAX_ABS_TOL
    )


def run_experiment(
    *,
    repo_root: Path,
    output_path: Path,
    expected_source_sha: str | None,
) -> dict[str, Any]:
    source_sha = _git_head(repo_root)
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise RuntimeError(
            f"source SHA mismatch: observed={source_sha} expected={expected_source_sha}"
        )
    with tempfile.TemporaryDirectory(prefix="train46-") as temporary_directory:
        temporary = Path(temporary_directory)
        outputs: dict[int, tuple[Path, Path]] = {}
        for microbatch_size in (FULL_MICROBATCH_SIZE, ACCUM_MICROBATCH_SIZE):
            state_path = temporary / f"state-{microbatch_size}.pt"
            metadata_path = temporary / f"metadata-{microbatch_size}.json"
            _run_worker_process(
                repo_root=repo_root,
                microbatch_size=microbatch_size,
                state_path=state_path,
                metadata_path=metadata_path,
            )
            outputs[microbatch_size] = (state_path, metadata_path)
        full_state = torch.load(
            outputs[FULL_MICROBATCH_SIZE][0],
            map_location="cpu",
            weights_only=True,
        )
        accumulated_state = torch.load(
            outputs[ACCUM_MICROBATCH_SIZE][0],
            map_location="cpu",
            weights_only=True,
        )
        full_metadata = json.loads(
            outputs[FULL_MICROBATCH_SIZE][1].read_text(encoding="utf-8")
        )
        accumulated_metadata = json.loads(
            outputs[ACCUM_MICROBATCH_SIZE][1].read_text(encoding="utf-8")
        )

    parameter_max_abs, parameter_relative_l2 = _model_diff(
        full_state["model_state"],
        accumulated_state["model_state"],
    )
    optimizer_max_abs = _max_tensor_diff(
        full_state["optimizer_tensors"],
        accumulated_state["optimizer_tensors"],
    )
    loss_diffs = [
        abs(float(full["update_loss"]) - float(accumulated["update_loss"]))
        for full, accumulated in zip(
            full_metadata["trace"],
            accumulated_metadata["trace"],
            strict=True,
        )
    ]
    max_loss_diff = max(loss_diffs)
    token_match = (
        full_metadata["valid_causal_tokens"]
        == accumulated_metadata["valid_causal_tokens"]
        == full_metadata["expected_valid_causal_tokens"]
        == accumulated_metadata["expected_valid_causal_tokens"]
    )
    equivalence = _equivalence_pass(
        token_match=token_match,
        parameter_max_abs=parameter_max_abs,
        parameter_relative_l2=parameter_relative_l2,
        max_loss_diff=max_loss_diff,
        optimizer_max_abs=optimizer_max_abs,
    )
    checkpoint = _checkpoint_boundary_probe(repo_root)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "precision": "fp32",
            "deterministic_algorithms": True,
            "paid_compute": False,
        },
        "controls": {
            "seed": SEED,
            "parameters": int(full_metadata["parameters"]),
            "learning_rate": LEARNING_RATE,
            "betas": [BETA1, BETA2],
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": CLIP_NORM,
            "scheduler": "constant",
            "warmup_steps": 0,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "valid_tokens_per_row": [63, 62, 61, 60],
            "valid_tokens_per_effective_batch": 246,
            "equivalence_updates": EQUIVALENCE_UPDATES,
            "aligned_targets": True,
            "variable_valid_tokens_per_microbatch": True,
        },
        "tolerances": {
            "final_parameter_max_abs": PARAMETER_MAX_ABS_TOL,
            "final_parameter_relative_l2": PARAMETER_REL_L2_TOL,
            "update_loss_abs": LOSS_ABS_TOL,
            "optimizer_state_max_abs": OPTIMIZER_STATE_MAX_ABS_TOL,
            "justification": (
                "deterministic fp32 execution changes reduction grouping when a fixed effective "
                "batch is partitioned; tolerances admit only rounding-scale drift"
            ),
        },
        "full_microbatch": full_metadata,
        "accumulated_microbatch": accumulated_metadata,
        "equivalence": {
            "valid_causal_token_counts_match": token_match,
            "max_update_loss_abs_diff": max_loss_diff,
            "final_parameter_max_abs_diff": parameter_max_abs,
            "final_parameter_relative_l2_diff": parameter_relative_l2,
            "optimizer_state_max_abs_diff": optimizer_max_abs,
            "pass": equivalence,
        },
        "resource_comparison": {
            "measurement_process_isolated": True,
            "timing_scope": "optimizer_work_only_excludes_batch_construction_and_state_serialization",
            "full_microbatch_tokens_per_second": full_metadata["tokens_per_second"],
            "accumulated_tokens_per_second": accumulated_metadata["tokens_per_second"],
            "throughput_ratio_accum_vs_full": (
                float(accumulated_metadata["tokens_per_second"])
                / float(full_metadata["tokens_per_second"])
            ),
            "full_microbatch_peak_rss_bytes": full_metadata["peak_rss_bytes"],
            "accumulated_peak_rss_bytes": accumulated_metadata["peak_rss_bytes"],
            "peak_rss_ratio_accum_vs_full": (
                float(accumulated_metadata["peak_rss_bytes"])
                / float(full_metadata["peak_rss_bytes"])
            ),
        },
        "checkpoint_boundary": checkpoint,
        "decision": {
            "gradient_accumulation_semantics_correct": equivalence,
            "partial_accumulation_checkpoint_publication_supported": False,
            "resume_only_from_committed_optimizer_boundaries": True,
            "duplicate_step_or_token_accounting_risk_closed": (
                equivalence
                and not checkpoint["duplicate_or_omitted_optimizer_step_detected"]
                and not checkpoint["duplicate_or_omitted_token_detected"]
            ),
        },
        "truth_boundary": {
            "single_process_cpu_only": True,
            "gpu_behavior_claimed": False,
            "distributed_training_redesigned": False,
            "quality_or_capability_claimed": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["decision"]["duplicate_step_or_token_accounting_risk_closed"]:
        raise RuntimeError("TRAIN-46 accumulation correctness did not close")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--worker-microbatch-size", type=int)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.worker_microbatch_size is not None:
        if args.state_output is None or args.metadata_output is None:
            parser.error("worker mode requires --state-output and --metadata-output")
        _worker(
            repo_root=repo_root,
            microbatch_size=args.worker_microbatch_size,
            state_path=args.state_output,
            metadata_path=args.metadata_output,
        )
        return 0
    if args.output is None:
        parser.error("--output is required outside worker mode")
    report = run_experiment(
        repo_root=repo_root,
        output_path=args.output,
        expected_source_sha=args.expected_source_sha,
    )
    print(
        "TRAIN46_EXECUTED_EVIDENCE="
        + json.dumps(
            {
                "source": report["source"],
                "controls": report["controls"],
                "equivalence": report["equivalence"],
                "resource_comparison": report["resource_comparison"],
                "checkpoint_boundary": report["checkpoint_boundary"],
                "decision": report["decision"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
