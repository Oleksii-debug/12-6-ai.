"""Execute TRAIN-46 next-scale gradient-accumulation correctness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.scaling_experiment import _byte_stream, _make_batch, _read_jsonl, controlled_specs
from twelve_six.schedule_batch_experiment import _max_tensor_diff, _trainer_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer

SCHEMA = "12-6.train46-grad-accum-correctness.v1"
AUTHORITY = "LOCAL_FREE_CPU_TRAINER_CORRECTNESS_EVIDENCE"
SEED = 1337
SEQUENCE_LENGTH = 64
EFFECTIVE_BATCH_SIZE = 8
FULL_MICROBATCH_SIZE = 8
ACCUM_MICROBATCH_SIZE = 2
EQUIVALENCE_UPDATES = 8
PARAMETER_MAX_ABS_TOL = 1e-5
PARAMETER_REL_L2_TOL = 1e-5
LOSS_ABS_TOL = 1e-6
OPTIMIZER_STATE_MAX_ABS_TOL = 1e-5


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_train_stream(repo_root: Path) -> bytes:
    tokenizer = ByteTokenizer()
    records = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    return _byte_stream(records, tokenizer)


def _aligned_effective_batch(train_stream: bytes, *, update: int) -> dict[str, torch.Tensor]:
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
        loss_mask[row, : row % 4] = False
    return {"input_ids": inputs, "target_ids": targets, "loss_mask": loss_mask}


def _slice_batch(batch: dict[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _flatten_model_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[key].detach().float().reshape(-1).cpu() for key in sorted(state)])


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
        raise ValueError("microbatch_size must divide effective batch")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_stream = _load_train_stream(repo_root)
    spec = controlled_specs()[-1]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    accumulation = EFFECTIVE_BATCH_SIZE // microbatch_size
    trainer = Trainer(
        model,
        _trainer_config(warmup_steps=0, accumulation=accumulation, scheduler="constant"),
        device="cpu",
    )
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    expected_tokens = 0
    for update in range(EQUIVALENCE_UPDATES):
        effective = _aligned_effective_batch(train_stream, update=update)
        expected_tokens += int(effective["loss_mask"].sum().item())
        final_metrics = None
        for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size):
            final_metrics = trainer.train_microbatch(
                _slice_batch(effective, offset, offset + microbatch_size)
            )
        if final_metrics is None or not final_metrics.optimizer_stepped:
            raise RuntimeError("effective update did not commit")
        trace.append(
            {
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "update_loss": float(final_metrics.update_loss),
                "grad_norm": float(final_metrics.grad_norm),
            }
        )
    elapsed = time.perf_counter() - started
    if trainer.tokens_seen != expected_tokens:
        raise RuntimeError(
            f"valid-token accounting mismatch: observed={trainer.tokens_seen} expected={expected_tokens}"
        )
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {"model_state": model_state, "optimizer_tensors": _optimizer_tensors(trainer)},
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
        "tokens_per_second": trainer.tokens_seen / elapsed,
        "training_wall_seconds": elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "trace": trace,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_worker_process(
    *, repo_root: Path, microbatch_size: int, state_path: Path, metadata_path: Path
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-microbatch-size",
        str(microbatch_size),
        "--repo-root",
        str(repo_root),
        "--state-output",
        str(state_path),
        "--metadata-output",
        str(metadata_path),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _model_max_abs_diff(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[float, float]:
    left_vector = _flatten_model_state(left)
    right_vector = _flatten_model_state(right)
    delta = right_vector - left_vector
    max_abs = float(torch.max(torch.abs(delta)).item())
    denominator = float(torch.linalg.vector_norm(left_vector).item())
    relative_l2 = float(torch.linalg.vector_norm(delta).item()) / denominator if denominator else math.inf
    return max_abs, relative_l2


def _checkpoint_boundary_probe(repo_root: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_stream = _load_train_stream(repo_root)
    spec = controlled_specs()[-1]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    config = _trainer_config(warmup_steps=0, accumulation=4, scheduler="constant")
    trainer = Trainer(model, config, device="cpu")
    effective = _aligned_effective_batch(train_stream, update=0)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    first = trainer.train_microbatch(_slice_batch(effective, 0, 2))
    if first.optimizer_stepped:
        raise RuntimeError("optimizer stepped before accumulation boundary")
    partial_publish_blocked = False
    partial_error = ""
    try:
        trainer.state_dict()
    except RuntimeError as exc:
        partial_publish_blocked = "mid-accumulation" in str(exc)
        partial_error = str(exc)
    if not partial_publish_blocked:
        raise RuntimeError("partial accumulation state was publishable")
    precommit_max_abs, _ = _model_max_abs_diff(
        initial_state,
        {key: value.detach() for key, value in model.state_dict().items()},
    )
    if precommit_max_abs != 0.0:
        raise RuntimeError("model parameters changed before committed optimizer step")

    for offset in (2, 4, 6):
        committed = trainer.train_microbatch(_slice_batch(effective, offset, offset + 2))
    if not committed.optimizer_stepped or trainer.optimizer_step != 1:
        raise RuntimeError("accumulation boundary did not commit exactly one optimizer step")
    committed_trainer_state = trainer.state_dict()
    committed_model_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    committed_tokens = int(effective["loss_mask"].sum().item())
    if trainer.tokens_seen != committed_tokens:
        raise RuntimeError("committed token count drift")

    fresh_model = TwelveSixDecoder(spec, InitSpec())
    fresh_model.load_state_dict(committed_model_state, strict=True)
    fresh_trainer = Trainer(fresh_model, config, device="cpu")
    fresh_trainer.load_state_dict(committed_trainer_state)

    next_effective = _aligned_effective_batch(train_stream, update=1)
    for offset in (0, 2, 4, 6):
        trainer.train_microbatch(_slice_batch(next_effective, offset, offset + 2))
        fresh_trainer.train_microbatch(_slice_batch(next_effective, offset, offset + 2))
    resume_max_abs, resume_relative_l2 = _model_max_abs_diff(
        {key: value.detach() for key, value in model.state_dict().items()},
        {key: value.detach() for key, value in fresh_model.state_dict().items()},
    )
    expected_after_resume = committed_tokens + int(next_effective["loss_mask"].sum().item())
    if trainer.tokens_seen != expected_after_resume or fresh_trainer.tokens_seen != expected_after_resume:
        raise RuntimeError("resume token accounting indicates replay or omission")
    if trainer.optimizer_step != 2 or fresh_trainer.optimizer_step != 2:
        raise RuntimeError("resume optimizer-step accounting indicates replay or omission")
    return {
        "scale_parameters": spec.parameter_count(),
        "accumulation_steps": 4,
        "partial_state_publish_blocked": partial_publish_blocked,
        "partial_publish_error": partial_error,
        "parameters_unchanged_before_commit": precommit_max_abs == 0.0,
        "committed_state_publish_succeeded": True,
        "committed_optimizer_step": int(committed_trainer_state["optimizer_step"]),
        "committed_tokens": committed_tokens,
        "resumed_optimizer_step_after_next_update": fresh_trainer.optimizer_step,
        "resumed_tokens_after_next_update": fresh_trainer.tokens_seen,
        "expected_tokens_after_next_update": expected_after_resume,
        "resume_parameter_max_abs_diff_vs_uninterrupted": resume_max_abs,
        "resume_parameter_relative_l2_diff_vs_uninterrupted": resume_relative_l2,
        "duplicate_or_omitted_optimizer_step_detected": False,
        "duplicate_or_omitted_token_detected": False,
    }


def run_experiment(*, repo_root: Path, output_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="train46-") as tmp:
        tmp_path = Path(tmp)
        outputs: dict[int, tuple[Path, Path]] = {}
        for microbatch_size in (FULL_MICROBATCH_SIZE, ACCUM_MICROBATCH_SIZE):
            state_path = tmp_path / f"state-{microbatch_size}.pt"
            metadata_path = tmp_path / f"metadata-{microbatch_size}.json"
            _run_worker_process(
                repo_root=repo_root,
                microbatch_size=microbatch_size,
                state_path=state_path,
                metadata_path=metadata_path,
            )
            outputs[microbatch_size] = (state_path, metadata_path)

        full_state = torch.load(outputs[FULL_MICROBATCH_SIZE][0], map_location="cpu", weights_only=True)
        accum_state = torch.load(outputs[ACCUM_MICROBATCH_SIZE][0], map_location="cpu", weights_only=True)
        full_meta = json.loads(outputs[FULL_MICROBATCH_SIZE][1].read_text(encoding="utf-8"))
        accum_meta = json.loads(outputs[ACCUM_MICROBATCH_SIZE][1].read_text(encoding="utf-8"))

    parameter_max_abs, parameter_relative_l2 = _model_max_abs_diff(
        full_state["model_state"], accum_state["model_state"]
    )
    optimizer_max_abs = _max_tensor_diff(
        full_state["optimizer_tensors"], accum_state["optimizer_tensors"]
    )
    loss_diffs = [
        abs(float(left["update_loss"]) - float(right["update_loss"]))
        for left, right in zip(full_meta["trace"], accum_meta["trace"], strict=True)
    ]
    max_loss_diff = max(loss_diffs)
    token_match = (
        full_meta["valid_causal_tokens"]
        == accum_meta["valid_causal_tokens"]
        == full_meta["expected_valid_causal_tokens"]
        == accum_meta["expected_valid_causal_tokens"]
    )
    equivalence_pass = (
        token_match
        and parameter_max_abs <= PARAMETER_MAX_ABS_TOL
        and parameter_relative_l2 <= PARAMETER_REL_L2_TOL
        and max_loss_diff <= LOSS_ABS_TOL
        and optimizer_max_abs <= OPTIMIZER_STATE_MAX_ABS_TOL
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
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
            "scale_parameters": int(full_meta["parameters"]),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "equivalence_updates": EQUIVALENCE_UPDATES,
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "scheduler": "constant",
            "aligned_targets": True,
            "deliberately_variable_valid_tokens_per_microbatch": True,
        },
        "tolerances": {
            "parameter_max_abs": PARAMETER_MAX_ABS_TOL,
            "parameter_relative_l2": PARAMETER_REL_L2_TOL,
            "update_loss_abs": LOSS_ABS_TOL,
            "optimizer_state_max_abs": OPTIMIZER_STATE_MAX_ABS_TOL,
            "justification": (
                "fp32 reductions are deterministic within each run but partitioning a batch changes "
                "floating-point summation order; tolerances admit only rounding-scale drift"
            ),
        },
        "full_microbatch": full_meta,
        "accumulated_microbatch": accum_meta,
        "equivalence": {
            "valid_causal_token_counts_match": token_match,
            "max_update_loss_abs_diff": max_loss_diff,
            "final_parameter_max_abs_diff": parameter_max_abs,
            "final_parameter_relative_l2_diff": parameter_relative_l2,
            "optimizer_state_max_abs_diff": optimizer_max_abs,
            "pass": equivalence_pass,
        },
        "resource_comparison": {
            "measurement_process_isolated": True,
            "full_microbatch_tokens_per_second": full_meta["tokens_per_second"],
            "accumulated_tokens_per_second": accum_meta["tokens_per_second"],
            "full_microbatch_peak_rss_bytes": full_meta["peak_rss_bytes"],
            "accumulated_peak_rss_bytes": accum_meta["peak_rss_bytes"],
            "throughput_ratio_accum_vs_full": (
                float(accum_meta["tokens_per_second"]) / float(full_meta["tokens_per_second"])
            ),
            "peak_rss_ratio_accum_vs_full": (
                float(accum_meta["peak_rss_bytes"]) / float(full_meta["peak_rss_bytes"])
            ),
        },
        "checkpoint_boundary": _checkpoint_boundary_probe(repo_root),
        "decision": {
            "gradient_accumulation_semantics_correct": equivalence_pass,
            "partial_accumulation_checkpoint_publication_supported": False,
            "resume_only_from_committed_optimizer_boundaries": True,
            "duplicate_step_or_token_accounting_risk_closed": equivalence_pass,
        },
        "truth_boundary": {
            "single_process_cpu_only": True,
            "distributed_training_redesigned": False,
            "gpu_behavior_claimed": False,
            "quality_or_capability_claimed": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not equivalence_pass:
        raise RuntimeError("TRAIN-46 gradient accumulation equivalence failed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
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
    report = run_experiment(repo_root=repo_root, output_path=args.output)
    print(
        json.dumps(
            {
                "equivalence_pass": report["equivalence"]["pass"],
                "checkpoint_boundary": report["checkpoint_boundary"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
