from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.checkpoint.cadence import choose_cadence, estimate_cadence, robust_seconds
from twelve_six.checkpoint.core import hash_json, verify_checkpoint
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.fixed_token_efficiency import (
    PACKING_SHA256,
    PACKING_VERSION,
    _checkpoint_identity,
    _control_bundle,
    _directory_bytes,
    _make_pair_batch,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH
from twelve_six.training import Trainer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (0.25, 1.0, 5.0, 30.0, 60.0)
MODEL_CASES = ((0, "~100K"), (3, "~1M"))
SEQUENCE_LENGTH = 64
BATCH_SIZE = 4
CAPACITY = BATCH_SIZE * SEQUENCE_LENGTH
TIMING_WARMUP_STEPS = 2
TIMING_STEPS = 7
CHECKPOINT_REPEATS = 3
SEED = 1337
TORCH_THREADS = 2


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _new_model_and_trainer(bundle: Mapping[str, Any]) -> tuple[TwelveSixDecoder, Trainer]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    return model, trainer


def _full_pair_batch(bundle: Mapping[str, Any], *, causal_offset: int) -> Mapping[str, torch.Tensor]:
    return _make_pair_batch(
        bundle["data"]["train_stream"],
        causal_offset=causal_offset,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        valid_pairs=CAPACITY,
    )


def _bound_load(
    path: Path,
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    bundle: Mapping[str, Any],
    source_sha: str,
    restore_rng: bool,
) -> None:
    load_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        restore_rng=restore_rng,
        expected_git_sha=source_sha,
        expected_model_spec_hash=bundle["spec"].identity_sha256(),
        expected_init_spec_hash=bundle["init_spec"].identity_sha256(),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=bundle["data"]["dataset_manifest_hash"],
        expected_split_identity=bundle["data"]["split_identity"],
        expected_packing_hash=PACKING_SHA256,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=bundle["run_manifest_hash"],
        expected_training_config_hash=hash_json(bundle["training_config"]),
        expected_seed=SEED,
    )


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _recursive_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _model_digest(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _benchmark_model(
    *,
    source_sha: str,
    model_index: int,
    label: str,
    targets: tuple[float, ...],
) -> dict[str, Any]:
    total_steps = TIMING_WARMUP_STEPS + TIMING_STEPS
    final_tokens = total_steps * CAPACITY
    bundle = _control_bundle(
        repo_root=ROOT,
        source_sha=source_sha,
        model_index=model_index,
        final_tokens=final_tokens,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SEED,
    )
    model, trainer = _new_model_and_trainer(bundle)

    step_samples: list[float] = []
    for step in range(total_steps):
        batch = _full_pair_batch(bundle, causal_offset=trainer.tokens_seen)
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        elapsed = time.perf_counter() - started
        if metrics.tokens != CAPACITY:
            raise AssertionError("aligned pair batch did not optimize the full step capacity")
        if step >= TIMING_WARMUP_STEPS:
            step_samples.append(elapsed)

    identity = _checkpoint_identity(source_sha=source_sha, bundle=bundle, trainer=trainer)
    with tempfile.TemporaryDirectory(prefix=f"train56-{model_index}-") as temp_root:
        root = Path(temp_root)
        checkpoint_paths: list[Path] = []
        save_samples: list[float] = []
        for index in range(CHECKPOINT_REPEATS):
            path = root / f"checkpoint-{index}"
            started = time.perf_counter()
            save_trainer_checkpoint(path, model=model, trainer=trainer, identity=identity)
            save_samples.append(time.perf_counter() - started)
            checkpoint_paths.append(path)

        verify_samples: list[float] = []
        for path in checkpoint_paths:
            started = time.perf_counter()
            verify_checkpoint(path)
            verify_samples.append(time.perf_counter() - started)

        fresh_load_samples: list[float] = []
        load_apply_samples: list[float] = []
        for index, path in enumerate(checkpoint_paths):
            started = time.perf_counter()
            fresh_model, fresh_trainer = _new_model_and_trainer(bundle)
            before_load = time.perf_counter()
            _bound_load(
                path,
                model=fresh_model,
                trainer=fresh_trainer,
                bundle=bundle,
                source_sha=source_sha,
                restore_rng=False,
            )
            finished = time.perf_counter()
            load_apply_samples.append(finished - before_load)
            fresh_load_samples.append(finished - started)
            if fresh_trainer.optimizer_step != trainer.optimizer_step:
                raise AssertionError(
                    f"fresh load {index} restored a different optimizer step"
                )

        checkpoint_bytes = _directory_bytes(checkpoint_paths[0])

    step_seconds = robust_seconds(step_samples)
    checkpoint_seconds = robust_seconds(save_samples)
    estimates = [
        estimate_cadence(
            step_seconds=step_seconds,
            checkpoint_seconds=checkpoint_seconds,
            max_recompute_seconds=target,
        )
        for target in targets
    ]
    selected = choose_cadence(estimates, max_overhead_percent=5.0)
    actual_parameters = bundle["spec"].parameter_count()
    return {
        "label": label,
        "model_index": model_index,
        "actual_parameters": actual_parameters,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "valid_causal_tokens_per_step": CAPACITY,
        "precision": bundle["trainer_config"].precision,
        "step_samples_s": step_samples,
        "optimizer_step_median_s": step_seconds,
        "checkpoint_save_verify_samples_s": save_samples,
        "checkpoint_save_verify_median_s": checkpoint_seconds,
        "checkpoint_explicit_verify_samples_s": verify_samples,
        "checkpoint_explicit_verify_median_s": robust_seconds(verify_samples),
        "fresh_load_samples_s": fresh_load_samples,
        "fresh_load_median_s": robust_seconds(fresh_load_samples),
        "load_apply_samples_s": load_apply_samples,
        "load_apply_median_s": robust_seconds(load_apply_samples),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_bytes_per_parameter": checkpoint_bytes / actual_parameters,
        "cadence_targets": [item.to_dict() for item in estimates],
        "selected_cadence": selected.to_dict(),
    }


def _equivalence_run(
    *,
    source_sha: str,
    interval_steps: int,
) -> dict[str, Any]:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be > 0")
    total_steps = interval_steps * 2 + 1
    bundle = _control_bundle(
        repo_root=ROOT,
        source_sha=source_sha,
        model_index=0,
        final_tokens=total_steps * CAPACITY,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SEED,
    )

    control_model, control_trainer = _new_model_and_trainer(bundle)
    for _ in range(total_steps):
        control_trainer.train_microbatch(
            _full_pair_batch(bundle, causal_offset=control_trainer.tokens_seen)
        )

    interrupted_model, interrupted_trainer = _new_model_and_trainer(bundle)
    for _ in range(interval_steps):
        interrupted_trainer.train_microbatch(
            _full_pair_batch(bundle, causal_offset=interrupted_trainer.tokens_seen)
        )

    with tempfile.TemporaryDirectory(prefix="train56-equivalence-") as temp_root:
        checkpoint_path = Path(temp_root) / f"step-{interval_steps:08d}"
        save_trainer_checkpoint(
            checkpoint_path,
            model=interrupted_model,
            trainer=interrupted_trainer,
            identity=_checkpoint_identity(
                source_sha=source_sha,
                bundle=bundle,
                trainer=interrupted_trainer,
            ),
        )
        del interrupted_model
        del interrupted_trainer

        resumed_model, resumed_trainer = _new_model_and_trainer(bundle)
        _bound_load(
            checkpoint_path,
            model=resumed_model,
            trainer=resumed_trainer,
            bundle=bundle,
            source_sha=source_sha,
            restore_rng=True,
        )
        if resumed_trainer.optimizer_step != interval_steps:
            raise AssertionError("resume did not restore the interrupted optimizer step")

        for _ in range(interval_steps, total_steps):
            resumed_trainer.train_microbatch(
                _full_pair_batch(bundle, causal_offset=resumed_trainer.tokens_seen)
            )

    control_state = control_trainer.state_dict()
    resumed_state = resumed_trainer.state_dict()
    model_equal = _recursive_equal(control_model.state_dict(), resumed_model.state_dict())
    trainer_equal = _recursive_equal(asdict(control_state), asdict(resumed_state))
    if not model_equal or not trainer_equal:
        raise AssertionError("interrupted/resumed final state differs from uninterrupted control")

    return {
        "model_parameters": bundle["spec"].parameter_count(),
        "checkpoint_interval_steps": interval_steps,
        "checkpoint_interval_tokens": interval_steps * CAPACITY,
        "interruption_step": interval_steps,
        "total_optimizer_steps": total_steps,
        "control_optimizer_step": control_trainer.optimizer_step,
        "resumed_optimizer_step": resumed_trainer.optimizer_step,
        "control_tokens_seen": control_trainer.tokens_seen,
        "resumed_tokens_seen": resumed_trainer.tokens_seen,
        "model_state_exact_equal": model_equal,
        "trainer_state_exact_equal": trainer_equal,
        "control_model_sha256": _model_digest(control_model),
        "resumed_model_sha256": _model_digest(resumed_model),
    }


def _transfer_guidance(stages: list[dict[str, Any]]) -> dict[str, Any]:
    small, large = stages
    return {
        "measured_parameter_range": [small["actual_parameters"], large["actual_parameters"]],
        "parameter_ratio_large_over_small": (
            large["actual_parameters"] / small["actual_parameters"]
        ),
        "step_time_ratio_large_over_small": (
            large["optimizer_step_median_s"] / small["optimizer_step_median_s"]
        ),
        "checkpoint_time_ratio_large_over_small": (
            large["checkpoint_save_verify_median_s"]
            / small["checkpoint_save_verify_median_s"]
        ),
        "checkpoint_bytes_ratio_large_over_small": (
            large["checkpoint_bytes"] / small["checkpoint_bytes"]
        ),
        "later_campaign_rule": (
            "Remeasure median optimizer-step and incumbent save+verify latency at the exact "
            "10M/100M batch geometry, then choose interval=floor(max_recompute_seconds/step_seconds) "
            "and require synchronous checkpoint overhead <=5%. Do not extrapolate step time as linear."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAIN-56 checkpoint-cadence experiment")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/train56/checkpoint_cadence_runtime.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    source_sha = _git_sha()
    stages = [
        _benchmark_model(
            source_sha=source_sha,
            model_index=model_index,
            label=label,
            targets=DEFAULT_TARGETS,
        )
        for model_index, label in MODEL_CASES
    ]
    selected_small = stages[0]["selected_cadence"]
    equivalence = _equivalence_run(
        source_sha=source_sha,
        interval_steps=int(selected_small["interval_steps"]),
    )

    report = {
        "schema_version": 1,
        "worker_id": "TRAIN-56-CKPT-CADENCE",
        "git_sha": source_sha,
        "host_scope": "github-actions-ubuntu-cpu",
        "checkpoint_format": "incumbent 12-6-checkpoint v1",
        "checkpoint_save_note": (
            "save timing is end-to-end and already includes incumbent internal verification "
            "before atomic publication"
        ),
        "measurement_assumptions": {
            "model_family": "current RESEARCH41 fixed-vocabulary controlled_specs indices 0 and 3",
            "data": "current fixed_token_efficiency control data and aligned causal-pair packing",
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "valid_causal_tokens_per_step": CAPACITY,
            "gradient_accumulation_steps": 1,
            "precision": "fp32",
            "scheduler": "constant",
            "seed": SEED,
            "torch_threads": TORCH_THREADS,
            "timing_warmup_steps": TIMING_WARMUP_STEPS,
            "timing_measured_steps": TIMING_STEPS,
            "checkpoint_repeats": CHECKPOINT_REPEATS,
            "cadence_targets_seconds": DEFAULT_TARGETS,
            "selected_policy": (
                "tightest maximum-recompute target with <=5% synchronous checkpoint overhead"
            ),
        },
        "stages": stages,
        "transfer_guidance": _transfer_guidance(stages),
        "interrupted_equivalence": equivalence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TRAIN56_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TRAIN56_REPORT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
