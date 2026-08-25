"""SCALE-141: scientifically bounded learned-10M fallback/continuation campaign.

There is intentionally no path that silently treats an S3 mechanics checkpoint as a
learned incumbent.  When no verified learned-10M checkpoint exists, this module runs
the strongest LOCAL_FREE fallback on the retained DATA-25 corpus while preserving the
accepted S3 architecture/optimizer lineage and D05 checkpoint recovery semantics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.milestone100_first_learned import (
    EXPECTED_CORPUS_ID,
    _build_corpus,
    _eval_examples,
    _locks,
    _read_json,
    _require_head,
    _rows,
    _state_hash,
    _write_json,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_VERSION, TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.scale141-10m-learned-fallback.v1"
AUTHORITY = "LOCAL_FREE_LEARNED_10M_EXPERIMENT_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
STAGE_CONFIG = Path("configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json")
PREPARED_RUN_CONFIG = Path("configs/runs/s3_10m_scale03_gpu_pilot.json")
EXPECTED_MODEL_SHA = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_INIT_SHA = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_PARAMETERS = 10_000_640
TRAIN_CORPUS_BYTES = 20_000_775
SEQ = 1024
BATCH = 1
MAX_STEPS = 2000
RESUME_STEP = 1000
EXPECTED_TOKENS_PER_STEP = BATCH * (SEQ - 1)
EXPECTED_OPTIMIZED_TOKENS = MAX_STEPS * EXPECTED_TOKENS_PER_STEP
EVAL_STEPS = (0, 500, 1000, 1500, 2000)
SEED = 20260825
LR = 3e-4
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
CADENCE_TARGET_SECONDS = 5.0
CADENCE_OVERHEAD_FRACTION = 0.05
CALIBRATION_STEPS = 3
HELDOUT_WINDOWS_PER_MODALITY = 32
TRAIN_WINDOWS_PER_MODALITY = 8
MIXTURE = (
    "uk", "en", "uk", "code", "en",
    "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk",
    "en", "uk", "code", "en", "uk",
)
PROMPTS = {"uk": "Українська мова ", "en": "The training corpus ", "code": "def stable_"}
SCALING_FIT = {
    "source": "RESEARCH41 controlled 95,568->1,037,696 parameter family",
    "source_pr": 162,
    "form": "log(loss)=b0+bN*log(N)+bT*log(T)",
    "b0": 5.30258616173305,
    "bN": -0.1175,
    "bT": -0.2666,
    "r_squared_in_box": 0.9628,
    "out_of_domain_for_10m": True,
}


class Scale141Error(RuntimeError):
    pass


def _append(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _model(repo: Path) -> tuple[ModelSpec, InitSpec, dict[str, Any]]:
    stage = load_stage_config(repo / STAGE_CONFIG)
    spec = stage.model
    init = stage.init
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise Scale141Error(f"S3 parameter drift: {spec.parameter_count()}")
    if spec.identity_sha256() != EXPECTED_MODEL_SHA:
        raise Scale141Error("S3 ModelSpec identity drift")
    if init.identity_sha256() != EXPECTED_INIT_SHA:
        raise Scale141Error("S3 InitSpec identity drift")
    prepared = _read_json(repo / PREPARED_RUN_CONFIG)
    candidate = prepared["candidate"]
    training = prepared["training"]
    if candidate["model_spec_sha256"] != EXPECTED_MODEL_SHA:
        raise Scale141Error("prepared S3 run no longer names exact candidate")
    expected_optimizer = {
        "optimizer": "AdamW", "learning_rate": LR, "betas": [0.9, 0.95],
        "eps": 1e-8, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": CLIP_NORM,
    }
    for key, value in expected_optimizer.items():
        if training[key] != value:
            raise Scale141Error(f"prepared S3 optimizer drift: {key}")
    return spec, init, prepared


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=MAX_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _packed(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, split: str, stratum: str):
    records = (
        TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
        for row in _rows(corpus, manifest, split, stratum)
    )
    yield from iter_packed_examples(
        records, tok, expected_split=split, sequence_length=SEQ, cross_document=False,
    )


def _steps_by_stratum(steps: int) -> dict[str, int]:
    result = {"uk": 0, "en": 0, "code": 0}
    for index in range(steps):
        result[MIXTURE[index % len(MIXTURE)]] += 1
    return result


def _train_iters(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, completed_steps: int):
    result = {s: _packed(corpus, manifest, tok, "train", s) for s in ("uk", "en", "code")}
    for stratum, count in _steps_by_stratum(completed_steps).items():
        for _ in range(count * BATCH):
            try:
                next(result[stratum])
            except StopIteration as exc:
                raise Scale141Error(f"{stratum} exhausted while restoring data position") from exc
    return result


def _next_batch(iterator: Iterator[Any]) -> dict[str, torch.Tensor]:
    examples = []
    for _ in range(BATCH):
        try:
            examples.append(next(iterator))
        except StopIteration as exc:
            raise Scale141Error("training corpus exhausted before target budget") from exc
    return {
        "input_ids": torch.tensor([x.input_ids for x in examples], dtype=torch.long),
        "labels": torch.tensor([x.labels for x in examples], dtype=torch.long),
    }


def _fixed_eval(model: TwelveSixDecoder, corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, *, split: str, windows: int):
    before = _state_hash(model)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by_modality: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for stratum in ("uk", "en", "code"):
                examples = []
                for example in _packed(corpus, manifest, tok, split, stratum):
                    examples.append(example)
                    if len(examples) == windows:
                        break
                if len(examples) != windows:
                    raise Scale141Error(f"insufficient {split}/{stratum} fixed eval windows")
                nll, tokens = _eval_examples(model, examples)
                loss = nll / tokens
                by_modality[stratum] = {
                    "loss": loss,
                    "bits_per_byte": loss / math.log(2.0),
                    "predicted_byte_tokens": tokens,
                    "windows": windows,
                }
                total_nll += nll
                total_tokens += tokens
    finally:
        model.train(was_training)
    if _state_hash(model) != before:
        raise Scale141Error("evaluation mutated model state")
    loss = total_nll / total_tokens
    return {
        "split": split,
        "loss": loss,
        "bits_per_byte": loss / math.log(2.0),
        "predicted_byte_tokens": total_tokens,
        "by_modality": by_modality,
        "non_mutation_passed": True,
    }


def _generation(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    cfg = GenerationConfig(max_new_tokens=64, sample=False)
    outputs = {}
    for name, prompt in PROMPTS.items():
        result = generate(backend, prompt, cfg)
        outputs[name] = {
            "prompt": prompt,
            "generated_token_ids": list(result.generated_token_ids),
            "text": result.text,
            "stop_reason": result.stop_reason,
        }
    return {"decoding": "greedy", "outputs": outputs}


def _scaling_prediction(tokens: int) -> dict[str, Any] | None:
    if tokens <= 0:
        return None
    log_loss = (
        SCALING_FIT["b0"]
        + SCALING_FIT["bN"] * math.log(EXPECTED_PARAMETERS)
        + SCALING_FIT["bT"] * math.log(tokens)
    )
    loss = math.exp(log_loss)
    return {
        "predicted_loss": loss,
        "predicted_bits_per_byte": loss / math.log(2.0),
        "fit": dict(SCALING_FIT),
        "interpretation": "OUT_OF_DOMAIN_DIAGNOSTIC_NOT_A_SUCCESS_GATE",
    }


def _select_cadence(step_seconds: list[float], save_seconds: list[float]) -> dict[str, Any]:
    if not step_seconds or not save_seconds:
        raise Scale141Error("cadence calibration requires measured step and save timings")
    median_step = statistics.median(step_seconds)
    median_save = statistics.median(save_seconds)
    if median_step <= 0.0 or median_save <= 0.0:
        raise Scale141Error("cadence calibration produced non-positive timing")
    lost_work_bound = max(1, int(CADENCE_TARGET_SECONDS // median_step))
    overhead_bound = max(1, math.ceil(median_save / (CADENCE_OVERHEAD_FRACTION * median_step)))
    feasible = overhead_bound <= lost_work_bound
    cadence = lost_work_bound if feasible else overhead_bound
    return {
        "schema": "12-6.scale141-cadence-runtime.v1",
        "policy_source": "TRAIN-56 default 5-second lost-work envelope, remeasured at exact 10M geometry",
        "median_optimizer_step_seconds": median_step,
        "median_save_verify_seconds": median_save,
        "lost_work_target_seconds": CADENCE_TARGET_SECONDS,
        "max_checkpoint_overhead_fraction": CADENCE_OVERHEAD_FRACTION,
        "lost_work_bound_steps": lost_work_bound,
        "minimum_overhead_bound_steps": overhead_bound,
        "checkpoint_every_optimizer_steps": cadence,
        "five_second_and_overhead_constraints_jointly_feasible": feasible,
        "projected_lost_work_seconds": cadence * median_step,
        "projected_checkpoint_overhead_fraction": median_save / (cadence * median_step),
        "timing_is_telemetry_not_run_identity": True,
    }


def _run_manifest(source_sha: str, spec: ModelSpec, init: InitSpec, tok: ByteTokenizer, manifest: dict[str, Any], cfg: TrainerConfig, locks: dict[str, Any], prepared: dict[str, Any]):
    value = {
        "schema": "12-6.scale141-run-manifest.v1",
        "source_sha": source_sha,
        "authority": AUTHORITY,
        "fallback_reason": "NO_VERIFIED_LEARNED_10M_INCUMBENT_FOUND; S3 artifacts before SCALE-141 are mechanics/prepared-only",
        "model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "init_spec": init.to_dict(),
        "init_spec_sha256": init.identity_sha256(),
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
        },
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "corpus_train_byte_tokens": manifest["by_split"]["train"]["byte_tokens"],
        "packing": {"version": PACKING_VERSION, "sequence_length": SEQ, "cross_document": False},
        "trainer_config": asdict(cfg),
        "micro_batch_size": BATCH,
        "max_steps": MAX_STEPS,
        "expected_tokens_per_step": EXPECTED_TOKENS_PER_STEP,
        "target_optimized_tokens": EXPECTED_OPTIMIZED_TOKENS,
        "target_corpus_fraction": EXPECTED_OPTIMIZED_TOKENS / TRAIN_CORPUS_BYTES,
        "scheduled_evaluation_steps": list(EVAL_STEPS),
        "fresh_process_resume_step": RESUME_STEP,
        "mixture_pattern": list(MIXTURE),
        "fixed_probe_windows": {"heldout_per_modality": HELDOUT_WINDOWS_PER_MODALITY, "train_per_modality": TRAIN_WINDOWS_PER_MODALITY},
        "prepared_s3_lineage": {
            "path": PREPARED_RUN_CONFIG.as_posix(),
            "schema": prepared["schema"],
            "candidate": prepared["candidate"],
            "optimizer": {k: prepared["training"][k] for k in ("optimizer", "learning_rate", "betas", "eps", "weight_decay", "gradient_clip_norm")},
        },
        "local_free_adaptation": {
            "device": "cpu",
            "precision": "fp32",
            "reason": "No authorized free CUDA device is asserted by this campaign; never silently spend or silently downgrade an accelerator run.",
            "prepared_gpu_microbatch_and_accumulation_are_not_claimed_as_executed": True,
        },
        "environment_lock_sha256": locks["combined_sha256"],
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
        "paid_compute": False,
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _identity(source_sha: str, spec: ModelSpec, tok: ByteTokenizer, manifest: dict[str, Any], run: dict[str, Any], cfg: TrainerConfig, trainer: Trainer, locks: dict[str, Any]):
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256,
        tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=manifest["corpus_identity_sha256"],
        run_manifest_hash=run["identity_sha256"],
        training_config={"trainer": asdict(cfg), "packing": run["packing"], "corpus_identity_sha256": manifest["corpus_identity_sha256"]},
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "learning_rate": cfg.learning_rate, "betas": list(cfg.betas), "eps": cfg.eps, "weight_decay": cfg.weight_decay},
        scheduler=None,
        environment_lock_hash=locks["combined_sha256"],
    )


def _save(path: Path, source_sha: str, spec: ModelSpec, tok: ByteTokenizer, manifest: dict[str, Any], run: dict[str, Any], cfg: TrainerConfig, trainer: Trainer, locks: dict[str, Any]):
    save_trainer_checkpoint(path, model=trainer.model, trainer=trainer, identity=_identity(source_sha, spec, tok, manifest, run, cfg, trainer, locks), overwrite=True)
    checked = verify_checkpoint(path)
    return {"step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen, "checkpoint_id": checked["checkpoint_id"]}


def _interval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"optimizer_steps": 0}
    losses = [r["loss"] for r in rows]
    grads = [r["grad_norm"] for r in rows if r["grad_norm"] is not None]
    clips = sum(bool(r["clipped"]) for r in rows)
    throughputs = [r["tokens_per_second"] for r in rows if r["tokens_per_second"] is not None]
    memories = [r["rss_peak_bytes"] for r in rows if r["rss_peak_bytes"] is not None]
    ordered = sorted(grads)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    return {
        "optimizer_steps": len(rows),
        "train_loss_mean": statistics.fmean(losses),
        "train_bits_per_byte_mean": statistics.fmean(losses) / math.log(2.0),
        "grad_norm_mean": statistics.fmean(grads) if grads else None,
        "grad_norm_p95": p95,
        "grad_norm_max": max(grads) if grads else None,
        "clip_count": clips,
        "clip_rate": clips / len(rows),
        "train_tokens_per_second_mean": statistics.fmean(throughputs) if throughputs else None,
        "rss_peak_bytes": max(memories) if memories else None,
        "scheduled_step_update_norm": rows[-1].get("update_norm") if rows[-1]["optimizer_step"] in EVAL_STEPS else None,
        "scheduled_step_relative_update_norm": rows[-1].get("relative_update_norm") if rows[-1]["optimizer_step"] in EVAL_STEPS else None,
    }


def _eval_point(model: TwelveSixDecoder, checkpoint: Path, corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, trainer: Trainer, interval_rows: list[dict[str, Any]]):
    heldout = _fixed_eval(model, corpus, manifest, tok, split="validation", windows=HELDOUT_WINDOWS_PER_MODALITY)
    train_probe = _fixed_eval(model, corpus, manifest, tok, split="train", windows=TRAIN_WINDOWS_PER_MODALITY)
    gap = heldout["bits_per_byte"] - train_probe["bits_per_byte"]
    prediction = _scaling_prediction(trainer.tokens_seen)
    if prediction is not None:
        prediction["observed_loss"] = heldout["loss"]
        prediction["observed_bits_per_byte"] = heldout["bits_per_byte"]
        prediction["loss_residual_observed_minus_predicted"] = heldout["loss"] - prediction["predicted_loss"]
    return {
        "optimizer_step": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "heldout": heldout,
        "train_probe": train_probe,
        "memorization": {
            "metric": "fixed_train_probe_advantage_over_fixed_heldout_bpb",
            "heldout_minus_train_bits_per_byte": gap,
            "interpretation": "EXPOSURE_DIAGNOSTIC_NOT_A_PRIVACY_LEAKAGE_CLAIM",
        },
        "training_interval": _interval_summary(interval_rows),
        "raw_base_generation": _generation(checkpoint),
        "scaling_law_diagnostic": prediction,
    }


def _common(repo: Path, source_sha: str, out: Path, build: bool):
    _require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = _build_corpus(repo, out) if build else _read_json(out / "corpus-manifest.json")
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Scale141Error("DATA-25 corpus identity drift")
    if manifest["by_split"]["train"]["byte_tokens"] != TRAIN_CORPUS_BYTES:
        raise Scale141Error("DATA-25 train size drift")
    if EXPECTED_OPTIMIZED_TOKENS >= TRAIN_CORPUS_BYTES:
        raise Scale141Error("fallback budget would recycle corpus; explicit re-review required")
    tok = ByteTokenizer()
    spec, init, prepared = _model(repo)
    cfg = _trainer_config()
    locks = _locks(repo)
    run = _run_manifest(source_sha, spec, init, tok, manifest, cfg, locks, prepared)
    if build:
        _write_json(out / "run-manifest.json", run)
        _write_json(out / "machine-phase1.json", {"source_sha": source_sha, "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(), "pid": os.getpid(), "paid_compute": False})
    else:
        if _read_json(out / "run-manifest.json") != run:
            raise Scale141Error("run manifest changed across fresh process")
        _write_json(out / "machine-resume.json", {"source_sha": source_sha, "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count(), "pid": os.getpid(), "paid_compute": False})
    return manifest, tok, spec, init, cfg, locks, run


def _train_transition(observer: TrainingObserver, trainer: Trainer, batch: dict[str, torch.Tensor], *, scheduled: bool) -> dict[str, Any]:
    before = None
    before_norm = None
    if scheduled:
        before = [p.detach().clone() for p in trainer.model.parameters()]
        before_norm = math.sqrt(sum(float(torch.sum(p.detach().float() ** 2).item()) for p in trainer.model.parameters()))
    batch, wait = batch, 0.0
    metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
    observation = observer.step_samples[-1]
    update_norm = None
    relative = None
    if before is not None:
        update_sq = 0.0
        for parameter, old in zip(trainer.model.parameters(), before, strict=True):
            delta = parameter.detach().float() - old.float()
            update_sq += float(torch.sum(delta * delta).item())
        update_norm = math.sqrt(update_sq)
        relative = update_norm / before_norm if before_norm and before_norm > 0.0 else None
    loss = metrics.update_loss if metrics.update_loss is not None else metrics.loss
    return {
        "optimizer_step": metrics.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "tokens": metrics.tokens,
        "loss": loss,
        "grad_norm": metrics.grad_norm,
        "learning_rate": metrics.learning_rate,
        "clipped": metrics.grad_norm is not None and metrics.grad_norm > CLIP_NORM,
        "step_seconds": observation.step_seconds,
        "tokens_per_second": observation.train_tokens_per_second,
        "rss_peak_bytes": observation.memory.process_rss_peak_bytes,
        "update_norm": update_norm,
        "relative_update_norm": relative,
    }


def phase1(repo: Path, source_sha: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, spec, init, cfg, locks, run = _common(repo, source_sha, out, True)
    corpus = out / "corpus-a"
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS:
        raise Scale141Error("runtime parameter count mismatch")
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(run, device="cpu", max_step_samples=2048)
    checkpoint0 = _save(out / "checkpoint-0000", source_sha, spec, tok, manifest, run, cfg, trainer, locks)
    scheduled: dict[str, Any] = {"0": _eval_point(model, out / "checkpoint-0000", corpus, manifest, tok, trainer, [])}
    initial_state = _state_hash(model)
    its = _train_iters(corpus, manifest, tok, 0)
    rows: list[dict[str, Any]] = []
    curve = out / "train-curve.jsonl"
    if curve.exists():
        curve.unlink()
    calibration_step_seconds: list[float] = []
    calibration_save_seconds: list[float] = []
    cadence = None
    for index in range(RESUME_STEP):
        stratum = MIXTURE[index % len(MIXTURE)]
        row = _train_transition(observer, trainer, _next_batch(its[stratum]), scheduled=(trainer.optimizer_step + 1 in EVAL_STEPS))
        row["stratum"] = stratum
        rows.append(row)
        _append(curve, row)
        if row["tokens"] != EXPECTED_TOKENS_PER_STEP:
            raise Scale141Error(f"unexpected optimized tokens/step: {row['tokens']}")
        if trainer.optimizer_step <= CALIBRATION_STEPS:
            calibration_step_seconds.append(row["step_seconds"])
            import time
            started = time.perf_counter()
            _save(out / "recovery-latest", source_sha, spec, tok, manifest, run, cfg, trainer, locks)
            calibration_save_seconds.append(time.perf_counter() - started)
            if trainer.optimizer_step == CALIBRATION_STEPS:
                cadence = _select_cadence(calibration_step_seconds, calibration_save_seconds)
                _write_json(out / "cadence-runtime.json", cadence)
        if trainer.optimizer_step in (500, 1000):
            path = out / f"checkpoint-{trainer.optimizer_step:04d}"
            _save(path, source_sha, spec, tok, manifest, run, cfg, trainer, locks)
            start = 0 if trainer.optimizer_step == 500 else 500
            interval = [r for r in rows if start < r["optimizer_step"] <= trainer.optimizer_step]
            scheduled[str(trainer.optimizer_step)] = _eval_point(model, path, corpus, manifest, tok, trainer, interval)
        elif cadence is not None and trainer.optimizer_step % cadence["checkpoint_every_optimizer_steps"] == 0:
            _save(out / "recovery-latest", source_sha, spec, tok, manifest, run, cfg, trainer, locks)
    if trainer.optimizer_step != RESUME_STEP:
        raise Scale141Error("phase1 stop boundary failed")
    result = {
        "schema": "12-6.scale141-phase1.v1",
        "source_sha": source_sha,
        "process": {"pid": os.getpid(), "python_executable": sys.executable},
        "model": {"parameter_count": EXPECTED_PARAMETERS, "model_spec_sha256": spec.identity_sha256(), "init_spec_sha256": init.identity_sha256(), "initial_state_sha256": initial_state},
        "checkpoint0": checkpoint0,
        "cadence": cadence,
        "scheduled": scheduled,
        "observer": observer.summary(),
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    result["identity_sha256"] = hash_json(result)
    _write_json(out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path):
    manifest, tok, spec, init, cfg, locks, run = _common(repo, source_sha, out, False)
    corpus = out / "corpus-a"
    p1 = _read_json(out / "phase1.json")
    cadence = _read_json(out / "cadence-runtime.json")
    integrity = verify_checkpoint(out / "checkpoint-1000")
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    loaded = load_trainer_checkpoint(
        out / "checkpoint-1000", model=model, trainer=trainer, strict_model=True, restore_rng=True,
        expected_git_sha=source_sha, expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"]:
        raise Scale141Error("resume run-manifest mismatch")
    if trainer.optimizer_step != RESUME_STEP or trainer.tokens_seen != RESUME_STEP * EXPECTED_TOKENS_PER_STEP:
        raise Scale141Error("resume checkpoint state mismatch")
    heldout_recheck = _fixed_eval(model, corpus, manifest, tok, split="validation", windows=HELDOUT_WINDOWS_PER_MODALITY)
    prior_bpb = p1["scheduled"]["1000"]["heldout"]["bits_per_byte"]
    if not math.isclose(heldout_recheck["bits_per_byte"], prior_bpb, rel_tol=0.0, abs_tol=1e-9):
        raise Scale141Error("held-out metric changed after verified fresh-process reload")
    if os.getpid() == p1["process"]["pid"]:
        raise Scale141Error("resume did not occur in a fresh process")
    observer = TrainingObserver(run, device="cpu", max_step_samples=2048)
    its = _train_iters(corpus, manifest, tok, RESUME_STEP)
    rows: list[dict[str, Any]] = []
    scheduled = dict(p1["scheduled"])
    first_resumed = None
    for index in range(RESUME_STEP, MAX_STEPS):
        stratum = MIXTURE[index % len(MIXTURE)]
        row = _train_transition(observer, trainer, _next_batch(its[stratum]), scheduled=(trainer.optimizer_step + 1 in EVAL_STEPS))
        row["stratum"] = stratum
        rows.append(row)
        _append(out / "train-curve.jsonl", row)
        first_resumed = first_resumed or trainer.optimizer_step
        if trainer.optimizer_step in (1500, 2000):
            path = out / f"checkpoint-{trainer.optimizer_step:04d}"
            _save(path, source_sha, spec, tok, manifest, run, cfg, trainer, locks)
            start = trainer.optimizer_step - 500
            interval = [r for r in rows if start < r["optimizer_step"] <= trainer.optimizer_step]
            scheduled[str(trainer.optimizer_step)] = _eval_point(model, path, corpus, manifest, tok, trainer, interval)
        elif trainer.optimizer_step % cadence["checkpoint_every_optimizer_steps"] == 0:
            _save(out / "recovery-latest", source_sha, spec, tok, manifest, run, cfg, trainer, locks)
    if first_resumed != 1001 or trainer.optimizer_step != MAX_STEPS:
        raise Scale141Error("fresh-process continuation boundary failed")
    if trainer.tokens_seen != EXPECTED_OPTIMIZED_TOKENS:
        raise Scale141Error(f"optimized-token target mismatch: {trainer.tokens_seen}")
    final_integrity = verify_checkpoint(out / "checkpoint-2000")
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "incumbent_search_decision": "FALLBACK_NO_VERIFIED_LEARNED_10M",
        "model": {"parameter_count": EXPECTED_PARAMETERS, "model_spec_sha256": spec.identity_sha256(), "init_spec_sha256": init.identity_sha256()},
        "tokenizer": run["tokenizer"],
        "corpus": {"identity_sha256": manifest["corpus_identity_sha256"], "train_byte_tokens": TRAIN_CORPUS_BYTES, "optimized_tokens": trainer.tokens_seen, "fraction_of_one_train_corpus": trainer.tokens_seen / TRAIN_CORPUS_BYTES, "corpus_replay": False},
        "optimizer": run["prepared_s3_lineage"]["optimizer"],
        "run_manifest_identity_sha256": run["identity_sha256"],
        "cadence": cadence,
        "checkpoint_integrity": {"pre_resume_checkpoint_id": integrity["checkpoint_id"], "final_checkpoint_id": final_integrity["checkpoint_id"]},
        "fresh_process_resume": {"phase1_pid": p1["process"]["pid"], "resume_pid": os.getpid(), "loaded_step": RESUME_STEP, "first_resumed_step": first_resumed, "heldout_bpb_before_stop": prior_bpb, "heldout_bpb_after_reload": heldout_recheck["bits_per_byte"], "metric_recheck_passed": True},
        "scheduled": scheduled,
        "scaling_fit": SCALING_FIT,
        "observer_phase1": p1["observer"],
        "observer_resume": observer.summary(),
        "success": {
            "exact_10m_geometry": spec.parameter_count() == EXPECTED_PARAMETERS,
            "verified_checkpoint_before_continuation": True,
            "heldout_metric_rechecked_before_continuation": True,
            "fresh_process_resume": True,
            "optimized_token_target": trainer.tokens_seen == EXPECTED_OPTIMIZED_TOKENS,
            "no_corpus_replay": trainer.tokens_seen < TRAIN_CORPUS_BYTES,
            "paid_compute": False,
            "scientific_fallback_executed": True,
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(out / "report.json", report)
    return report


def validate(path: Path, expected_source_sha: str | None = None):
    report = _read_json(path)
    if report.get("schema") != SCHEMA:
        raise Scale141Error("unexpected report schema")
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise Scale141Error("report source SHA mismatch")
    if report["model"]["parameter_count"] != EXPECTED_PARAMETERS:
        raise Scale141Error("report parameter count mismatch")
    if report["corpus"]["corpus_replay"] is not False:
        raise Scale141Error("report admits corpus replay")
    required = ("exact_10m_geometry", "verified_checkpoint_before_continuation", "heldout_metric_rechecked_before_continuation", "fresh_process_resume", "optimized_token_target", "no_corpus_replay", "scientific_fallback_executed")
    if not all(report["success"].get(key) is True for key in required):
        raise Scale141Error("one or more required scientific gates failed")
    if report["success"].get("paid_compute") is not False:
        raise Scale141Error("paid compute boundary violated")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase1", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--repo-root", type=Path, default=Path("."))
        p.add_argument("--source-sha", required=True)
        p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("report", type=Path)
    p.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "phase1":
        value = phase1(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
    elif args.command == "resume":
        value = resume(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
    else:
        value = validate(args.report, args.expected_source_sha)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
