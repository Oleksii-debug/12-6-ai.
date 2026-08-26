"""RECOVER-170: execute the failed TRAIN-41 long ~100K experiment on DATA-25.

This is an additive recovery runner.  It preserves TRAIN-41's learned-Base model,
initializer, optimizer, seed and batch/sequence geometry while replacing only the
historical tiny S0 fixture with the accepted compatible DATA-25 project corpus.
The primary budget remains 2,097,152 actual optimized causal targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six import milestone150_learned_base_ladder as ladder
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
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.recover170-train41-long-100k.v1"
REPORT_SCHEMA = "12-6.recover170-train41-long-100k-report.v1"
AUTHORITY = "LOCAL_FREE_LONG_100K_DATA25_EVIDENCE_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
SEMANTIC_INCUMBENT_SHA = "55c452e651ce1254d2bd21c3ec7746ee26ac6ee7"
EXPECTED_PARAMETERS = 95_568
EXPECTED_MODEL_SPEC_SHA256 = (
    "4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470"
)
EXPECTED_INIT_SPEC_SHA256 = (
    "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
)
EXPECTED_CORPUS_ID = m100.EXPECTED_CORPUS_ID
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
SEED = 1337
LEARNING_RATE = 3e-4
SAFETY_MAX_STEPS = 12_000
FINAL_TOKENS = 2_097_152
RESUME_TOKENS = 1_048_576
EVALUATION_BUDGETS = (
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    1_572_864,
    2_097_152,
)
GENERATION_BUDGETS = (
    0,
    16_384,
    65_536,
    262_144,
    1_048_576,
    1_572_864,
    2_097_152,
)
CHECKPOINT_BUDGETS = (65_536, 262_144, 1_048_576, 1_572_864, 2_097_152)
GENERATION_PROMPTS = {"en": "The ", "uk": "Україна ", "code": "def "}
MAX_NEW_TOKENS = 32
OVERFIT_RISE_BPB = 0.01
PROFILE_ID = "linux-x86_64-cuda-training"


class RecoveryError(RuntimeError):
    pass


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _self_hashed(value: dict[str, Any], key: str = "identity_sha256") -> dict[str, Any]:
    normalized = _json_normalize(value)
    if not isinstance(normalized, dict):
        raise RecoveryError("self-hashed payload must normalize to an object")
    normalized[key] = hash_json(normalized)
    return normalized


def _check_self_hash(value: dict[str, Any], key: str = "identity_sha256") -> None:
    expected = value.get(key)
    unsigned = dict(value)
    unsigned.pop(key, None)
    if not isinstance(expected, str) or expected != hash_json(unsigned):
        raise RecoveryError(f"{key} mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _configure_data_geometry() -> None:
    # The DATA-25 helpers are deliberately reused with TRAIN-41's original geometry.
    m100.SEQ = SEQUENCE_LENGTH
    m100.BATCH = BATCH_SIZE


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=SAFETY_MAX_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _model_truth() -> tuple[Any, Any]:
    spec = ladder.model_spec("100k")
    init = ladder.init_spec()
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise RecoveryError("TRAIN-41 parameter-count drift")
    if spec.identity_sha256() != EXPECTED_MODEL_SPEC_SHA256:
        raise RecoveryError("TRAIN-41 ModelSpec identity drift")
    if init.identity_sha256() != EXPECTED_INIT_SPEC_SHA256:
        raise RecoveryError("InitSpec identity drift")
    return spec, init


def _validate_preflight(path: Path, source_sha: str) -> dict[str, Any]:
    evidence = _read_json(path)
    if evidence.get("source_sha") != source_sha:
        raise RecoveryError("purpose-environment evidence source SHA mismatch")
    if evidence.get("profile_id") != PROFILE_ID:
        raise RecoveryError("wrong purpose-environment profile")
    verification = evidence.get("verification")
    if not isinstance(verification, dict):
        raise RecoveryError("purpose-environment verification is absent")
    required = ("registry_validation", "exact_hash_install", "project_wheel_install", "runtime_probe")
    if any(verification.get(key) != "PASS" for key in required):
        raise RecoveryError("purpose-environment preflight is not fully PASS")
    expected_hash = evidence.get("evidence_sha256")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    if expected_hash != hash_json(unsigned):
        raise RecoveryError("purpose-environment evidence self-hash mismatch")
    evidence["file_sha256"] = _sha256_file(path)
    return evidence


def _evaluation_identity(tok: ByteTokenizer, manifest: dict[str, Any]) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema": "12-6.recover170-evaluation-identity.v1",
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "split": "validation",
            "strata": ["uk", "en", "code"],
            "metric": "autoregressive_cross_entropy_and_bits_per_raw_utf8_byte",
            "tokenizer": {
                "version": tok.identity.version,
                "config_sha256": tok.identity.config_sha256,
                "vocab_sha256": tok.identity.vocab_sha256,
                "vocab_size": tok.identity.vocab_size,
            },
            "packing": {
                "version": m100.PACKING_VERSION,
                "sequence_length": SEQUENCE_LENGTH,
                "cross_document": False,
            },
        }
    )


def _run_manifest(
    source_sha: str,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    cfg: TrainerConfig,
    locks: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    spec, init = _model_truth()
    return _self_hashed(
        {
            "schema": SCHEMA,
            "authority": AUTHORITY,
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "semantic_incumbent": {
                "experiment": "TRAIN-41",
                "source_sha": SEMANTIC_INCUMBENT_SHA,
                "preserved": [
                    "ModelSpec",
                    "InitSpec",
                    "byte tokenizer",
                    "AdamW recipe",
                    "seed",
                    "batch_size",
                    "sequence_length",
                    "2,097,152 causal-target frontier",
                ],
                "intentional_change": (
                    "historical S0 fixture -> accepted compatible DATA-25 project corpus"
                ),
            },
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
            "evaluation_identity_sha256": eval_identity["identity_sha256"],
            "trainer_config": asdict(cfg),
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "mixture_pattern": list(m100.MIXTURE),
            "optimized_token_frontier": FINAL_TOKENS,
            "resume_token_frontier": RESUME_TOKENS,
            "evaluation_budgets": list(EVALUATION_BUDGETS),
            "generation_budgets": list(GENERATION_BUDGETS),
            "checkpoint_budgets": list(CHECKPOINT_BUDGETS),
            "environment_lock_sha256": locks["combined_sha256"],
            "purpose_environment": {
                "profile_id": preflight["profile_id"],
                "profile_sha256": preflight["profile"]["profile_sha256"],
                "purpose_index_sha256": preflight["purpose_index"]["index_sha256"],
                "evidence_sha256": preflight["evidence_sha256"],
                "file_sha256": preflight["file_sha256"],
            },
            "truth_boundary": {
                "foreign_pretrained_weights": False,
                "sft": False,
                "rlhf": False,
                "dpo": False,
                "paid_compute": False,
                "instruction_following_claim": False,
                "alignment_claim": False,
                "production_readiness_claim": False,
                "intelligence_claim": False,
                "representative_external_corpus_claim": False,
                "data25_is_project_authored": True,
            },
        }
    )


def _checkpoint_identity(
    source_sha: str,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
) -> CheckpointIdentity:
    spec, _ = _model_truth()
    training_config = {
        "trainer": _json_normalize(asdict(cfg)),
        "data": {
            "tokenizer_version": tok.identity.version,
            "packing_version": m100.PACKING_VERSION,
            "packing_sequence_length": SEQUENCE_LENGTH,
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "evaluation_identity_sha256": eval_identity["identity_sha256"],
        },
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256,
        tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=manifest["corpus_identity_sha256"],
        run_manifest_hash=run["identity_sha256"],
        training_config=training_config,
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=locks["combined_sha256"],
    )


def _save_checkpoint(
    out: Path,
    name: str,
    source_sha: str,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
) -> dict[str, Any]:
    destination = out / "checkpoints" / name
    started = time.perf_counter()
    save_trainer_checkpoint(
        destination,
        model=trainer.model,
        trainer=trainer,
        identity=_checkpoint_identity(
            source_sha, tok, manifest, eval_identity, run, cfg, trainer, locks
        ),
        overwrite=True,
    )
    checked = verify_checkpoint(destination)
    return {
        "directory": destination.relative_to(out).as_posix(),
        "checkpoint_id": checked["checkpoint_id"],
        "optimizer_step": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "save_verify_wall_seconds": time.perf_counter() - started,
    }


def _generation(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, sample=False)
    outputs: dict[str, Any] = {}
    for name, prompt in GENERATION_PROMPTS.items():
        result = generate(backend, prompt, config)
        outputs[name] = {
            "prompt": prompt,
            "prompt_token_ids": list(result.prompt_token_ids),
            "generated_token_ids": list(result.generated_token_ids),
            "text": result.text,
            "stop_reason": result.stop_reason,
        }
    return {
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "backend_diagnostics": backend.diagnostics(),
        "outputs": outputs,
    }


def _generation_from_state(
    out: Path,
    label: str,
    source_sha: str,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
) -> dict[str, Any]:
    record = _save_checkpoint(
        out,
        f"transient-generation-{label}",
        source_sha,
        tok,
        manifest,
        eval_identity,
        run,
        cfg,
        trainer,
        locks,
    )
    path = out / record["directory"]
    try:
        result = _generation(path)
    finally:
        shutil.rmtree(path)
    result["optimizer_step"] = trainer.optimizer_step
    result["optimized_tokens"] = trainer.tokens_seen
    return result


def _evaluate(
    model: TwelveSixDecoder,
    corpus: Path,
    manifest: dict[str, Any],
    tok: ByteTokenizer,
) -> dict[str, Any]:
    before = m100._state_hash(model)
    started = time.perf_counter()
    result = m100._evaluate(model, corpus, manifest, tok)
    result["wall_seconds"] = time.perf_counter() - started
    if result.get("non_mutation_passed") is not True:
        raise RecoveryError("held-out evaluation did not prove non-mutation")
    if before != m100._state_hash(model):
        raise RecoveryError("held-out evaluation mutated model state")
    return result


def _new_tracking() -> dict[str, Any]:
    return {
        "evaluation_points": [],
        "generation_snapshots": [],
        "checkpoint_records": [],
        "training_intervals": [],
        "gradient_norms": [],
        "clip_count": 0,
        "training_wall_seconds": 0.0,
        "checkpoint_wall_seconds": 0.0,
        "evaluation_wall_seconds": 0.0,
        "generation_wall_seconds": 0.0,
        "interval_nll": 0.0,
        "interval_tokens": 0,
        "total_training_tokens": 0,
    }


def _record_eval(
    tracking: dict[str, Any],
    requested_tokens: int,
    trainer: Trainer,
    evaluation: dict[str, Any],
) -> None:
    interval_tokens = int(tracking["interval_tokens"])
    interval_nll = float(tracking["interval_nll"])
    train_loss = interval_nll / interval_tokens if interval_tokens else None
    train_bpb = train_loss / math.log(2.0) if train_loss is not None else None
    heldout_bpb = float(evaluation["bits_per_byte"])
    tracking["evaluation_points"].append(
        {
            "requested_optimized_tokens": requested_tokens,
            "actual_optimized_tokens": trainer.tokens_seen,
            "optimizer_step": trainer.optimizer_step,
            "training_loss_since_previous_eval": train_loss,
            "training_bpb_since_previous_eval": train_bpb,
            "heldout": evaluation,
            "heldout_minus_training_bpb": (
                heldout_bpb - train_bpb if train_bpb is not None else None
            ),
        }
    )
    tracking["evaluation_wall_seconds"] += float(evaluation["wall_seconds"])
    tracking["interval_nll"] = 0.0
    tracking["interval_tokens"] = 0


def _overfit_analysis(points: list[dict[str, Any]]) -> dict[str, Any]:
    learned = [point for point in points if int(point["actual_optimized_tokens"]) > 0]
    if not learned:
        return {"status": "NO_LEARNED_EVALUATIONS"}
    best = min(learned, key=lambda point: float(point["heldout"]["bits_per_byte"]))
    best_bpb = float(best["heldout"]["bits_per_byte"])
    onset = None
    previous_train = None
    best_so_far = float("inf")
    for point in learned:
        heldout = float(point["heldout"]["bits_per_byte"])
        train = point["training_bpb_since_previous_eval"]
        if train is not None:
            train = float(train)
        if (
            previous_train is not None
            and train is not None
            and train < previous_train
            and heldout >= best_so_far + OVERFIT_RISE_BPB
        ):
            onset = {
                "actual_optimized_tokens": point["actual_optimized_tokens"],
                "optimizer_step": point["optimizer_step"],
                "heldout_bpb": heldout,
                "training_bpb": train,
                "best_prior_heldout_bpb": best_so_far,
                "criterion": (
                    "heldout BPB >= best prior + 0.01 while interval training BPB improves"
                ),
            }
            break
        best_so_far = min(best_so_far, heldout)
        if train is not None:
            previous_train = train
    return {
        "status": "OVERFIT_PROXY_ONSET_DETECTED" if onset else "NO_PROXY_ONSET_DETECTED",
        "definition": (
            "diagnostic generalization-gap proxy only; not a privacy or memorization-extraction claim"
        ),
        "best_heldout": {
            "actual_optimized_tokens": best["actual_optimized_tokens"],
            "optimizer_step": best["optimizer_step"],
            "bits_per_byte": best_bpb,
        },
        "onset": onset,
    }


def _next_thresholds(values: tuple[int, ...], completed_tokens: int) -> list[int]:
    return [value for value in values if value > completed_tokens]


def _crossed(pending: list[int], tokens: int) -> list[int]:
    result: list[int] = []
    while pending and pending[0] <= tokens:
        result.append(pending.pop(0))
    return result


def _machine(source_sha: str) -> dict[str, Any]:
    return {
        "source_sha": source_sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "pid": os.getpid(),
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "peak_rss_bytes": _rss_bytes(),
    }


def prepare(
    repo: Path,
    source_sha: str,
    out: Path,
    environment_evidence: Path,
) -> dict[str, Any]:
    m100._require_head(repo, source_sha)
    _configure_data_geometry()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = m100._build_corpus(repo, out)
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RecoveryError("DATA-25 corpus identity drift")
    if manifest["train_validation_content_overlap"] != 0:
        raise RecoveryError("DATA-25 train/validation leakage")
    tok = ByteTokenizer()
    if tok.identity.vocab_size != 256 or tok.identity.special_tokens:
        raise RecoveryError("canonical byte-tokenizer identity drift")
    cfg = _trainer_config()
    locks = m100._locks(repo)
    preflight = _validate_preflight(environment_evidence, source_sha)
    eval_identity = _evaluation_identity(tok, manifest)
    run = _run_manifest(source_sha, tok, manifest, eval_identity, cfg, locks, preflight)
    _write_json(out / "evaluation-identity.json", eval_identity)
    _write_json(out / "run-manifest.json", run)
    _write_json(out / "environment-preflight.json", preflight)
    _write_json(out / "machine-prepare.json", _machine(source_sha))
    return run


def _load_common(repo: Path, source_sha: str, out: Path) -> tuple[Any, ...]:
    m100._require_head(repo, source_sha)
    _configure_data_geometry()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = _read_json(out / "corpus-manifest.json")
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RecoveryError("persisted DATA-25 corpus identity drift")
    tok = ByteTokenizer()
    eval_identity = _read_json(out / "evaluation-identity.json")
    _check_self_hash(eval_identity)
    cfg = _trainer_config()
    locks = m100._locks(repo)
    preflight = _read_json(out / "environment-preflight.json")
    run = _run_manifest(source_sha, tok, manifest, eval_identity, cfg, locks, preflight)
    persisted = _read_json(out / "run-manifest.json")
    _check_self_hash(persisted)
    if persisted != run:
        raise RecoveryError("run manifest changed between fresh processes")
    return manifest, tok, eval_identity, cfg, locks, run


def _process_crossings(
    *,
    out: Path,
    source_sha: str,
    corpus: Path,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
    tracking: dict[str, Any],
    pending_evals: list[int],
    pending_generations: list[int],
    pending_checkpoints: list[int],
) -> None:
    for target in _crossed(pending_evals, trainer.tokens_seen):
        evaluation = _evaluate(trainer.model, corpus, manifest, tok)
        _record_eval(tracking, target, trainer, evaluation)
    for target in _crossed(pending_checkpoints, trainer.tokens_seen):
        record = _save_checkpoint(
            out,
            f"checkpoint-t{target:07d}",
            source_sha,
            tok,
            manifest,
            eval_identity,
            run,
            cfg,
            trainer,
            locks,
        )
        record["requested_optimized_tokens"] = target
        record["overshoot_tokens"] = trainer.tokens_seen - target
        tracking["checkpoint_wall_seconds"] += record["save_verify_wall_seconds"]
        tracking["checkpoint_records"].append(record)
    for target in _crossed(pending_generations, trainer.tokens_seen):
        started = time.perf_counter()
        snapshot = _generation_from_state(
            out,
            f"t{target:07d}",
            source_sha,
            tok,
            manifest,
            eval_identity,
            run,
            cfg,
            trainer,
            locks,
        )
        snapshot["requested_optimized_tokens"] = target
        snapshot["overshoot_tokens"] = trainer.tokens_seen - target
        tracking["generation_wall_seconds"] += time.perf_counter() - started
        tracking["generation_snapshots"].append(snapshot)


def _train_until(
    *,
    out: Path,
    source_sha: str,
    corpus: Path,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_identity: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
    tracking: dict[str, Any],
    stop_tokens: int,
) -> None:
    pending_evals = _next_thresholds(EVALUATION_BUDGETS, trainer.tokens_seen)
    pending_generations = _next_thresholds(GENERATION_BUDGETS, trainer.tokens_seen)
    pending_checkpoints = _next_thresholds(CHECKPOINT_BUDGETS, trainer.tokens_seen)
    its = m100._train_iters(corpus, manifest, tok, trainer.optimizer_step)
    batches = {stratum: m100._batches(iterator) for stratum, iterator in its.items()}
    observer = TrainingObserver(run, device="cpu", max_step_samples=2048)
    curve = out / "train-curve.jsonl"

    while trainer.tokens_seen < stop_tokens:
        if trainer.optimizer_step >= SAFETY_MAX_STEPS:
            raise RecoveryError("safety max_steps reached before causal-token frontier")
        stratum = m100.MIXTURE[trainer.optimizer_step % len(m100.MIXTURE)]
        batch = next(batches[stratum])
        before_tokens = trainer.tokens_seen
        started = time.perf_counter()
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=0.0)
        step_seconds = time.perf_counter() - started
        actual_tokens = trainer.tokens_seen - before_tokens
        if actual_tokens <= 0 or actual_tokens != int(metrics.tokens):
            raise RecoveryError("Trainer causal-token accounting mismatch")
        loss = metrics.update_loss if metrics.update_loss is not None else metrics.loss
        if not math.isfinite(float(loss)):
            raise RecoveryError("non-finite training loss")
        tracking["training_wall_seconds"] += step_seconds
        tracking["total_training_tokens"] += actual_tokens
        tracking["interval_nll"] += float(loss) * actual_tokens
        tracking["interval_tokens"] += actual_tokens
        grad_norm = float(metrics.grad_norm)
        if not math.isfinite(grad_norm):
            raise RecoveryError("non-finite gradient norm")
        tracking["gradient_norms"].append(grad_norm)
        if grad_norm > 1.0:
            tracking["clip_count"] += 1
        _append_jsonl(
            curve,
            {
                "optimizer_step": trainer.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "step_optimized_tokens": actual_tokens,
                "stratum": stratum,
                "loss": float(loss),
                "bits_per_byte": float(loss) / math.log(2.0),
                "grad_norm": grad_norm,
                "learning_rate": float(metrics.learning_rate),
                "step_wall_seconds": step_seconds,
                "optimized_tokens_per_second": actual_tokens / step_seconds,
            },
        )
        _process_crossings(
            out=out,
            source_sha=source_sha,
            corpus=corpus,
            tok=tok,
            manifest=manifest,
            eval_identity=eval_identity,
            run=run,
            cfg=cfg,
            trainer=trainer,
            locks=locks,
            tracking=tracking,
            pending_evals=pending_evals,
            pending_generations=pending_generations,
            pending_checkpoints=pending_checkpoints,
        )


def phase1(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    manifest, tok, eval_identity, cfg, locks, run = _load_common(repo, source_sha, out)
    spec, init = _model_truth()
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    tracking = _new_tracking()
    corpus = out / "corpus-a"

    initial_eval = _evaluate(model, corpus, manifest, tok)
    _record_eval(tracking, 0, trainer, initial_eval)
    started = time.perf_counter()
    initial_generation = _generation_from_state(
        out,
        "t0000000",
        source_sha,
        tok,
        manifest,
        eval_identity,
        run,
        cfg,
        trainer,
        locks,
    )
    initial_generation["requested_optimized_tokens"] = 0
    initial_generation["overshoot_tokens"] = 0
    tracking["generation_wall_seconds"] += time.perf_counter() - started
    tracking["generation_snapshots"].append(initial_generation)

    _train_until(
        out=out,
        source_sha=source_sha,
        corpus=corpus,
        tok=tok,
        manifest=manifest,
        eval_identity=eval_identity,
        run=run,
        cfg=cfg,
        trainer=trainer,
        locks=locks,
        tracking=tracking,
        stop_tokens=RESUME_TOKENS,
    )
    resume_records = [
        item
        for item in tracking["checkpoint_records"]
        if item["requested_optimized_tokens"] == RESUME_TOKENS
    ]
    if len(resume_records) != 1:
        raise RecoveryError("mandatory resume checkpoint was not retained exactly once")
    resume_record = resume_records[0]
    boundary_eval = next(
        item
        for item in tracking["evaluation_points"]
        if item["requested_optimized_tokens"] == RESUME_TOKENS
    )
    boundary_generation = next(
        item
        for item in tracking["generation_snapshots"]
        if item["requested_optimized_tokens"] == RESUME_TOKENS
    )
    result = _self_hashed(
        {
            "schema": "12-6.recover170-phase1.v1",
            "source_sha": source_sha,
            "pid": os.getpid(),
            "optimizer_step": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "resume_frontier": RESUME_TOKENS,
            "resume_overshoot_tokens": trainer.tokens_seen - RESUME_TOKENS,
            "resume_checkpoint": resume_record,
            "resume_boundary_evaluation": boundary_eval,
            "resume_boundary_generation": boundary_generation,
            "tracking": tracking,
            "machine": _machine(source_sha),
        }
    )
    _write_json(out / "phase1.json", result)
    return result


def _assert_eval_equal(left: dict[str, Any], right: dict[str, Any]) -> float:
    values = [
        abs(float(left["bits_per_byte"]) - float(right["bits_per_byte"])),
        abs(float(left["loss"]) - float(right["loss"])),
    ]
    for stratum in ("uk", "en", "code"):
        values.append(
            abs(
                float(left["by_stratum"][stratum]["bits_per_byte"])
                - float(right["by_stratum"][stratum]["bits_per_byte"])
            )
        )
    drift = max(values)
    if drift > 1e-12:
        raise RecoveryError(f"fresh-process resume held-out drift {drift} exceeds 1e-12")
    return drift


def resume(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    manifest, tok, eval_identity, cfg, locks, run = _load_common(repo, source_sha, out)
    phase1_result = _read_json(out / "phase1.json")
    _check_self_hash(phase1_result)
    if int(phase1_result["pid"]) == os.getpid():
        raise RecoveryError("resume must execute in a fresh process")
    spec, init = _model_truth()
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    resume_path = out / phase1_result["resume_checkpoint"]["directory"]
    load_started = time.perf_counter()
    loaded = load_trainer_checkpoint(
        resume_path,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    load_seconds = time.perf_counter() - load_started
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"]:
        raise RecoveryError("resume checkpoint run-manifest identity mismatch")
    if trainer.tokens_seen != int(phase1_result["optimized_tokens"]):
        raise RecoveryError("fresh Trainer did not restore exact causal-token counter")
    if trainer.optimizer_step != int(phase1_result["optimizer_step"]):
        raise RecoveryError("fresh Trainer did not restore exact optimizer step")

    corpus = out / "corpus-a"
    fresh_eval = _evaluate(model, corpus, manifest, tok)
    drift = _assert_eval_equal(
        phase1_result["resume_boundary_evaluation"]["heldout"], fresh_eval
    )
    fresh_generation = _generation(resume_path)
    prior_outputs = phase1_result["resume_boundary_generation"]["outputs"]
    if fresh_generation["outputs"] != prior_outputs:
        raise RecoveryError("fresh-process resume generation parity failed")

    tracking = phase1_result["tracking"]
    tracking["checkpoint_wall_seconds"] += load_seconds
    _train_until(
        out=out,
        source_sha=source_sha,
        corpus=corpus,
        tok=tok,
        manifest=manifest,
        eval_identity=eval_identity,
        run=run,
        cfg=cfg,
        trainer=trainer,
        locks=locks,
        tracking=tracking,
        stop_tokens=FINAL_TOKENS,
    )
    final_records = [
        item
        for item in tracking["checkpoint_records"]
        if item["requested_optimized_tokens"] == FINAL_TOKENS
    ]
    if len(final_records) != 1:
        raise RecoveryError("final retained checkpoint is missing")
    final_checkpoint = final_records[0]
    checked = verify_checkpoint(out / final_checkpoint["directory"])
    if checked["checkpoint_id"] != final_checkpoint["checkpoint_id"]:
        raise RecoveryError("final checkpoint identity drift")

    gradients = [float(value) for value in tracking["gradient_norms"]]
    training_seconds = float(tracking["training_wall_seconds"])
    exact_tokens = int(trainer.tokens_seen)
    report = _self_hashed(
        {
            "schema": REPORT_SCHEMA,
            "authority": AUTHORITY,
            "source": {
                "repository": REPOSITORY,
                "git_sha": source_sha,
                "semantic_incumbent_sha": SEMANTIC_INCUMBENT_SHA,
            },
            "model": {
                "spec": spec.to_dict(),
                "model_spec_sha256": spec.identity_sha256(),
                "parameter_count": spec.parameter_count(),
                "init_spec": init.to_dict(),
                "init_spec_sha256": init.identity_sha256(),
                "random_initialization": True,
            },
            "truth_model": {
                "tokenizer": run["tokenizer"],
                "corpus_identity_sha256": manifest["corpus_identity_sha256"],
                "evaluation_identity": eval_identity,
                "data_change_from_train41": (
                    "S0 fixture replaced by compatible project-authored DATA-25"
                ),
                "representative_external_corpus_claim": False,
            },
            "run": {
                "run_manifest": run,
                "run_manifest_sha256": run["identity_sha256"],
                "trainer_config": _json_normalize(asdict(cfg)),
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "frontier_requested_optimized_tokens": FINAL_TOKENS,
                "frontier_actual_optimized_tokens": exact_tokens,
                "frontier_overshoot_tokens": exact_tokens - FINAL_TOKENS,
                "optimizer_steps": trainer.optimizer_step,
                "exact_causal_token_accounting": "Trainer.tokens_seen",
            },
            "evaluation": {
                "points": tracking["evaluation_points"],
                "overfit_memorization_proxy": _overfit_analysis(
                    tracking["evaluation_points"]
                ),
            },
            "checkpoints": {
                "retained": tracking["checkpoint_records"],
                "final": final_checkpoint,
            },
            "resume": {
                "status": "PASS",
                "phase1_pid": phase1_result["pid"],
                "resume_pid": os.getpid(),
                "fresh_process": int(phase1_result["pid"]) != os.getpid(),
                "checkpoint_id": phase1_result["resume_checkpoint"]["checkpoint_id"],
                "optimized_tokens": phase1_result["optimized_tokens"],
                "optimizer_step": phase1_result["optimizer_step"],
                "heldout_max_abs_drift": drift,
                "generation_token_parity": "PASS",
                "load_wall_seconds": load_seconds,
            },
            "generation": {
                "raw_base_snapshots": tracking["generation_snapshots"],
                "final_first_party_snapshot": _generation(out / final_checkpoint["directory"]),
            },
            "systems": {
                "training_wall_seconds": training_seconds,
                "end_to_end_measured_region_seconds": (
                    training_seconds
                    + float(tracking["evaluation_wall_seconds"])
                    + float(tracking["generation_wall_seconds"])
                    + float(tracking["checkpoint_wall_seconds"])
                ),
                "optimized_tokens_per_training_second": (
                    exact_tokens / training_seconds if training_seconds > 0 else None
                ),
                "peak_rss_bytes": _rss_bytes(),
                "gradient_norm": {
                    "min": min(gradients),
                    "max": max(gradients),
                    "mean": sum(gradients) / len(gradients),
                },
                "clip_count": int(tracking["clip_count"]),
                "clip_rate": int(tracking["clip_count"]) / trainer.optimizer_step,
                "machine_phase1": phase1_result["machine"],
                "machine_resume": _machine(source_sha),
            },
            "environment_preflight": _read_json(out / "environment-preflight.json"),
            "truth_boundary": run["truth_boundary"],
        }
    )
    _write_json(out / "report.json", report)
    return report


def validate(report_path: Path, expected_source_sha: str) -> dict[str, Any]:
    report = _read_json(report_path)
    _check_self_hash(report)
    if report.get("schema") != REPORT_SCHEMA:
        raise RecoveryError("wrong recovery report schema")
    if report["source"]["git_sha"] != expected_source_sha:
        raise RecoveryError("recovery report source SHA mismatch")
    if report["model"]["parameter_count"] != EXPECTED_PARAMETERS:
        raise RecoveryError("recovery report parameter-count mismatch")
    if report["model"]["model_spec_sha256"] != EXPECTED_MODEL_SPEC_SHA256:
        raise RecoveryError("recovery report ModelSpec identity mismatch")
    if report["truth_model"]["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RecoveryError("recovery report corpus identity mismatch")
    if report["resume"]["status"] != "PASS" or not report["resume"]["fresh_process"]:
        raise RecoveryError("fresh-process resume is not PASS")
    actual = int(report["run"]["frontier_actual_optimized_tokens"])
    overshoot = int(report["run"]["frontier_overshoot_tokens"])
    if actual < FINAL_TOKENS or overshoot < 0 or overshoot >= BATCH_SIZE * SEQUENCE_LENGTH:
        raise RecoveryError("invalid final causal-token frontier accounting")
    truth = report["truth_boundary"]
    forbidden_true = (
        "foreign_pretrained_weights",
        "sft",
        "rlhf",
        "dpo",
        "paid_compute",
        "instruction_following_claim",
        "alignment_claim",
        "production_readiness_claim",
        "intelligence_claim",
        "representative_external_corpus_claim",
    )
    if any(truth.get(key) is not False for key in forbidden_true):
        raise RecoveryError("truth boundary contains an unsupported claim")
    final_path = report_path.parent / report["checkpoints"]["final"]["directory"]
    checked = verify_checkpoint(final_path)
    if checked["checkpoint_id"] != report["checkpoints"]["final"]["checkpoint_id"]:
        raise RecoveryError("final checkpoint no longer verifies")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, required=True)
    prepare_parser.add_argument("--source-sha", required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--environment-evidence", type=Path, required=True)

    for name in ("phase1", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--source-sha", required=True)
        command.add_argument("--output-dir", type=Path, required=True)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.repo_root.resolve(),
            args.source_sha,
            args.output_dir.resolve(),
            args.environment_evidence.resolve(),
        )
    elif args.command == "phase1":
        result = phase1(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
    elif args.command == "resume":
        result = resume(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
    else:
        result = validate(args.report.resolve(), args.expected_source_sha)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
