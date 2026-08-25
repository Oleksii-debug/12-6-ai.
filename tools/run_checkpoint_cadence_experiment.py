from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.checkpoint.cadence import choose_cadence, estimate_cadence, robust_seconds
from twelve_six.checkpoint.core import CheckpointIdentity, sha256_file, verify_checkpoint
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (0.25, 1.0, 5.0, 30.0, 60.0)
SEQ_LEN = 64
BATCH_SIZE = 2
TIMING_WARMUP_STEPS = 2
TIMING_STEPS = 7
CHECKPOINT_REPEATS = 3
SEED = 560056


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_corpus_bytes() -> bytes:
    path = ROOT / "data/s0/packaged/train.jsonl"
    chunks: list[bytes] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        text = json.loads(raw_line)["text"]
        chunks.append(text.encode("utf-8") + b"\n")
    corpus = b"".join(chunks)
    if not corpus:
        raise RuntimeError("training corpus fixture is empty")
    return corpus


def _batch_at(corpus: bytes, *, step: int, vocab_size: int) -> Mapping[str, torch.Tensor]:
    required = BATCH_SIZE * SEQ_LEN
    start = step * required
    repeated = corpus * ((start + required) // len(corpus) + 1)
    window = repeated[start : start + required]
    values = [value % vocab_size for value in window]
    tensor = torch.tensor(values, dtype=torch.long).view(BATCH_SIZE, SEQ_LEN)
    return {"input_ids": tensor}


def _trainer_config(*, max_steps: int) -> TrainerConfig:
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
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _new_model_and_trainer(stage_path: Path, *, max_steps: int) -> tuple[Any, Trainer, Any]:
    stage = load_stage_config(stage_path)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, _trainer_config(max_steps=max_steps), device="cpu")
    return model, trainer, stage


def _identity(*, model: Any, trainer: Trainer, stage: Any, run_tag: str) -> CheckpointIdentity:
    tokenizer_path = ROOT / "configs/s0/tokenizer_byte_v1.json"
    dataset_manifest_path = ROOT / "data/s0/packaged/manifest.json"
    trainer_config = asdict(trainer.config)
    run_manifest_hash = _sha256_bytes(
        json.dumps(
            {
                "worker": "TRAIN-56-CKPT-CADENCE",
                "run_tag": run_tag,
                "stage": stage.stage,
                "trainer": trainer_config,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQ_LEN,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return CheckpointIdentity(
        git_sha=_git_sha(),
        model_spec=stage.model.to_dict(),
        parameter_count=count_trainable_parameters(model),
        tokenizer_hash=sha256_file(tokenizer_path),
        tokenizer_vocab_hash=_sha256_bytes(bytes(range(256))),
        dataset_manifest_hash=sha256_file(dataset_manifest_path),
        run_manifest_hash=run_manifest_hash,
        training_config={"trainer": trainer_config, "purpose": "train56_checkpoint_cadence"},
        seed=trainer.config.seed,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": trainer.config.learning_rate,
            "betas": trainer.config.betas,
            "eps": trainer.config.eps,
            "weight_decay": trainer.config.weight_decay,
        },
        scheduler=None,
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
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


def _model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _benchmark_stage(stage_path: Path, targets: tuple[float, ...]) -> dict[str, Any]:
    corpus = _load_corpus_bytes()
    total_steps = TIMING_WARMUP_STEPS + TIMING_STEPS
    model, trainer, stage = _new_model_and_trainer(stage_path, max_steps=total_steps)

    step_samples: list[float] = []
    for step in range(total_steps):
        started = time.perf_counter()
        trainer.train_microbatch(_batch_at(corpus, step=step, vocab_size=stage.model.vocab_size))
        elapsed = time.perf_counter() - started
        if step >= TIMING_WARMUP_STEPS:
            step_samples.append(elapsed)

    identity = _identity(model=model, trainer=trainer, stage=stage, run_tag="timing")
    with tempfile.TemporaryDirectory(prefix=f"train56-{stage.stage.lower()}-") as temp_root:
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
        for index in range(CHECKPOINT_REPEATS):
            started = time.perf_counter()
            fresh_model, fresh_trainer, _ = _new_model_and_trainer(
                stage_path, max_steps=total_steps
            )
            before_load = time.perf_counter()
            load_trainer_checkpoint(
                checkpoint_paths[index],
                model=fresh_model,
                trainer=fresh_trainer,
                restore_rng=False,
            )
            finished = time.perf_counter()
            load_apply_samples.append(finished - before_load)
            fresh_load_samples.append(finished - started)
            if fresh_trainer.optimizer_step != trainer.optimizer_step:
                raise AssertionError("fresh-loaded optimizer step differs from saved trainer")

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
    return {
        "stage": stage.stage,
        "target_parameters": stage.target_parameters,
        "actual_parameters": count_trainable_parameters(model),
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "precision": trainer.config.precision,
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
        "checkpoint_bytes_per_parameter": checkpoint_bytes / count_trainable_parameters(model),
        "cadence_targets": [item.to_dict() for item in estimates],
        "selected_cadence": selected.to_dict(),
    }


def _equivalence_run(stage_path: Path, interval_steps: int) -> dict[str, Any]:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be > 0")
    corpus = _load_corpus_bytes()
    total_steps = interval_steps * 2 + 1

    control_model, control_trainer, control_stage = _new_model_and_trainer(
        stage_path, max_steps=total_steps
    )
    for step in range(total_steps):
        control_trainer.train_microbatch(
            _batch_at(corpus, step=step, vocab_size=control_stage.model.vocab_size)
        )

    interrupted_model, interrupted_trainer, stage = _new_model_and_trainer(
        stage_path, max_steps=total_steps
    )
    for step in range(interval_steps):
        interrupted_trainer.train_microbatch(
            _batch_at(corpus, step=step, vocab_size=stage.model.vocab_size)
        )

    with tempfile.TemporaryDirectory(prefix="train56-equivalence-") as temp_root:
        checkpoint_path = Path(temp_root) / f"step-{interval_steps:08d}"
        save_trainer_checkpoint(
            checkpoint_path,
            model=interrupted_model,
            trainer=interrupted_trainer,
            identity=_identity(
                model=interrupted_model,
                trainer=interrupted_trainer,
                stage=stage,
                run_tag="equivalence-interruption",
            ),
        )

        # The original objects are intentionally discarded from the resumed path.
        del interrupted_model
        del interrupted_trainer

        resumed_model, resumed_trainer, _ = _new_model_and_trainer(
            stage_path, max_steps=total_steps
        )
        load_trainer_checkpoint(
            checkpoint_path,
            model=resumed_model,
            trainer=resumed_trainer,
            restore_rng=True,
        )
        if resumed_trainer.optimizer_step != interval_steps:
            raise AssertionError("resume did not restore the interrupted optimizer step")

        for step in range(interval_steps, total_steps):
            resumed_trainer.train_microbatch(
                _batch_at(corpus, step=step, vocab_size=stage.model.vocab_size)
            )

    control_state = control_trainer.state_dict()
    resumed_state = resumed_trainer.state_dict()
    model_equal = _recursive_equal(control_model.state_dict(), resumed_model.state_dict())
    trainer_equal = _recursive_equal(asdict(control_state), asdict(resumed_state))
    if not model_equal or not trainer_equal:
        raise AssertionError("interrupted/resumed final state differs from uninterrupted control")

    return {
        "stage": stage.stage,
        "checkpoint_interval_steps": interval_steps,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="TRAIN-56 checkpoint-cadence experiment")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/train56/checkpoint_cadence_runtime.json",
    )
    args = parser.parse_args()

    stage_paths = (
        ROOT / "configs/stages/s1_100k.json",
        ROOT / "configs/stages/s2_1m.json",
    )
    stages = [_benchmark_stage(path, DEFAULT_TARGETS) for path in stage_paths]
    selected_s1 = stages[0]["selected_cadence"]
    equivalence = _equivalence_run(
        stage_paths[0],
        interval_steps=int(selected_s1["interval_steps"]),
    )

    ratio = {
        "parameter_ratio_s2_over_s1": stages[1]["actual_parameters"] / stages[0]["actual_parameters"],
        "step_time_ratio_s2_over_s1": stages[1]["optimizer_step_median_s"] / stages[0]["optimizer_step_median_s"],
        "checkpoint_time_ratio_s2_over_s1": stages[1]["checkpoint_save_verify_median_s"] / stages[0]["checkpoint_save_verify_median_s"],
        "checkpoint_bytes_ratio_s2_over_s1": stages[1]["checkpoint_bytes"] / stages[0]["checkpoint_bytes"],
    }
    report = {
        "schema_version": 1,
        "worker_id": "TRAIN-56-CKPT-CADENCE",
        "git_sha": _git_sha(),
        "host_scope": "github-actions-ubuntu-cpu",
        "checkpoint_format": "incumbent 12-6-checkpoint v1",
        "checkpoint_save_note": "save timing is end-to-end and already includes incumbent internal verification before atomic publication",
        "measurement_assumptions": {
            "data": "repository data/s0/packaged/train.jsonl encoded as UTF-8 byte token ids",
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LEN,
            "gradient_accumulation_steps": 1,
            "precision": "fp32",
            "scheduler": "constant",
            "timing_warmup_steps": TIMING_WARMUP_STEPS,
            "timing_measured_steps": TIMING_STEPS,
            "checkpoint_repeats": CHECKPOINT_REPEATS,
            "cadence_targets_seconds": DEFAULT_TARGETS,
            "selected_policy": "tightest recompute target with <=5% synchronous checkpoint overhead",
        },
        "stages": stages,
        "scaling_observation": ratio,
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
