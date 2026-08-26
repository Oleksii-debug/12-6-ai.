"""MILESTONE-150 coherent learned Base ladder over one DATA-25 truth model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import resource
import struct
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.learned-base-ladder.v1"
RUN_SCHEMA = "12-6.learned-base-ladder-run.v1"
EVAL_SCHEMA = "12-6.learned-base-ladder-evaluation-identity.v1"
AUTHORITY = "LOCAL_FREE_LEARNED_BASE_LADDER_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
BRANCH = "milestone150/learned-base-ladder-v1-20260826"

EXPECTED_CORPUS_ID = m100.EXPECTED_CORPUS_ID
SEQ = m100.SEQ
BATCH = m100.BATCH
MAX_STEPS = m100.MAX_STEPS
RESUME_STEP = m100.RESUME_STEP
CHECKPOINT_STEPS = tuple(sorted(m100.CHECKPOINT_STEPS))
SEED = m100.SEED
LR = m100.LR
STRATA = ("uk", "en", "code")
VERIFY_TOL = 1e-7

SCALE_SPECS: dict[str, dict[str, Any]] = {
    "100k": {
        "expected_parameters": 95_568,
        "expected_model_spec_sha256": "4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470",
        "provenance": "S1/RESEARCH-41 controlled family; vocab 512->256 for canonical s0-byte-v1",
        "model": {
            "schema_version": 1,
            "vocab_size": 256,
            "max_seq_len": 256,
            "d_model": 48,
            "n_layers": 3,
            "n_heads": 4,
            "n_kv_heads": 4,
            "head_dim": 12,
            "d_ff": 128,
            "activation": "swiglu",
            "norm_kind": "rmsnorm",
            "norm_placement": "pre",
            "norm_eps": 1e-5,
            "position_embedding": "rope",
            "rope_theta": 10_000.0,
            "rope_rotary_dim": 12,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "final_norm": True,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
        },
    },
    "500k": {
        "expected_parameters": 467_808,
        "expected_model_spec_sha256": "208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a",
        "provenance": "RESEARCH-41 fixed-control family incumbent",
        "model": {
            "schema_version": 1,
            "vocab_size": 256,
            "max_seq_len": 256,
            "d_model": 96,
            "n_layers": 4,
            "n_heads": 6,
            "n_kv_heads": 6,
            "head_dim": 16,
            "d_ff": 256,
            "activation": "swiglu",
            "norm_kind": "rmsnorm",
            "norm_placement": "pre",
            "norm_eps": 1e-5,
            "position_embedding": "rope",
            "rope_theta": 10_000.0,
            "rope_rotary_dim": 16,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "final_norm": True,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
        },
    },
    "1m": {
        "expected_parameters": 1_037_696,
        "expected_model_spec_sha256": "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07",
        "provenance": "RESEARCH-41 fixed-control family incumbent",
        "model": {
            "schema_version": 1,
            "vocab_size": 256,
            "max_seq_len": 256,
            "d_model": 128,
            "n_layers": 5,
            "n_heads": 8,
            "n_kv_heads": 8,
            "head_dim": 16,
            "d_ff": 352,
            "activation": "swiglu",
            "norm_kind": "rmsnorm",
            "norm_placement": "pre",
            "norm_eps": 1e-5,
            "position_embedding": "rope",
            "rope_theta": 10_000.0,
            "rope_rotary_dim": 16,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "final_norm": True,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
        },
    },
}

SCALE_ORDER = ("100k", "500k", "1m")


class LadderError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LadderError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _self_hashed(payload: dict[str, Any], key: str = "identity_sha256") -> dict[str, Any]:
    value = dict(payload)
    value[key] = hash_json(value)
    return value


def _check_self_hash(payload: dict[str, Any], key: str = "identity_sha256") -> None:
    expected = payload.get(key)
    unsigned = dict(payload)
    unsigned.pop(key, None)
    if not isinstance(expected, str) or expected != hash_json(unsigned):
        raise LadderError(f"{key} mismatch")


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB. The retained workflow executes on Ubuntu.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def model_spec(scale: str) -> ModelSpec:
    try:
        cfg = SCALE_SPECS[scale]
    except KeyError as exc:
        raise LadderError(f"unknown scale: {scale}") from exc
    spec = ModelSpec.from_dict(dict(cfg["model"]))
    if spec.parameter_count() != cfg["expected_parameters"]:
        raise LadderError(
            f"{scale} parameter count drift: {spec.parameter_count()} != {cfg['expected_parameters']}"
        )
    if spec.identity_sha256() != cfg["expected_model_spec_sha256"]:
        raise LadderError(f"{scale} ModelSpec semantic identity drift")
    return spec


def init_spec() -> InitSpec:
    init = InitSpec()
    expected = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
    if init.identity_sha256() != expected:
        raise LadderError("InitSpec identity drift")
    return init


def trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LR,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=MAX_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def evaluation_identity(tok: ByteTokenizer, manifest: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema": EVAL_SCHEMA,
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "split": "validation",
        "strata_order": list(STRATA),
        "metric": "autoregressive_cross_entropy_nats_and_bits_per_raw_utf8_byte",
        "target_mask": "labels[:,1:] != -100",
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
            "normalization": tok.identity.normalization,
            "encoding": tok.identity.encoding,
            "special_tokens": dict(tok.identity.special_tokens),
        },
        "packing": {
            "version": m100.PACKING_VERSION,
            "sequence_length": SEQ,
            "cross_document": False,
        },
    }
    return _self_hashed(value)


def _common_truth(
    repo: Path, source_sha: str, out: Path, *, build: bool
) -> tuple[dict[str, Any], ByteTokenizer, dict[str, Any]]:
    m100._require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if build:
        manifest = m100._build_corpus(repo, out)
    else:
        manifest = _read_json(out / "corpus-manifest.json")
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise LadderError("DATA-25 corpus identity mismatch")
    if manifest["train_validation_content_overlap"] != 0:
        raise LadderError("DATA-25 train/validation leakage")
    tok = ByteTokenizer()
    if tok.identity.vocab_size != 256 or tok.identity.special_tokens:
        raise LadderError("canonical byte-tokenizer truth model drift")
    eval_id = evaluation_identity(tok, manifest)
    return manifest, tok, eval_id


def prepare(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, eval_id = _common_truth(repo, source_sha, out, build=True)
    for scale in SCALE_ORDER:
        model_spec(scale)
    truth = _self_hashed(
        {
            "schema": "12-6.learned-base-ladder-truth.v1",
            "source_sha": source_sha,
            "repository": REPOSITORY,
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "tokenizer": {
                "version": tok.identity.version,
                "config_sha256": tok.identity.config_sha256,
                "vocab_sha256": tok.identity.vocab_sha256,
                "vocab_size": tok.identity.vocab_size,
                "special_tokens": dict(tok.identity.special_tokens),
            },
            "evaluation_identity": eval_id,
            "packing": {
                "version": m100.PACKING_VERSION,
                "sequence_length": SEQ,
                "cross_document": False,
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
                "external_real_world_training_data_present": False,
                "representative_external_corpus_claim": False,
            },
            "scale_order": list(SCALE_ORDER),
            "ten_million_status": "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE",
        }
    )
    _write_json(out / "ladder-truth.json", truth)
    _write_json(out / "machine-prepare.json", m100._machine(source_sha, m100._locks(repo)))
    return truth


def _run_manifest(
    source_sha: str,
    scale: str,
    spec: ModelSpec,
    init: InitSpec,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    eval_id: dict[str, Any],
    cfg: TrainerConfig,
    locks: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": RUN_SCHEMA,
        "source_sha": source_sha,
        "scale": scale,
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
        "evaluation_identity_sha256": eval_id["identity_sha256"],
        "packing": {
            "version": m100.PACKING_VERSION,
            "sequence_length": SEQ,
            "cross_document": False,
        },
        "trainer_config": asdict(cfg),
        "batch_size": BATCH,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "mixture_pattern": list(m100.MIXTURE),
        "environment_lock_sha256": locks["combined_sha256"],
        "foreign_pretrained_weights": False,
        "sft": False,
        "rlhf": False,
        "dpo": False,
        "paid_compute": False,
    }
    # Bind the manifest to its persisted JSON semantics before hashing. Dataclass
    # tuples (notably AdamW betas) serialize as JSON arrays; normalizing here keeps
    # strict phase1/resume equality while preserving the fail-closed identity check.
    persisted_value = json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    if not isinstance(persisted_value, dict):
        raise LadderError("run manifest must normalize to a JSON object")
    return _self_hashed(persisted_value)


def _checkpoint_identity(
    source_sha: str,
    spec: ModelSpec,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
) -> CheckpointIdentity:
    training_config = {
        "trainer": asdict(cfg),
        "data": {
            "tokenizer_version": tok.identity.version,
            "packing_version": m100.PACKING_VERSION,
            "packing_sequence_length": SEQ,
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "evaluation_identity_sha256": run["evaluation_identity_sha256"],
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
    scale_out: Path,
    source_sha: str,
    spec: ModelSpec,
    tok: ByteTokenizer,
    manifest: dict[str, Any],
    run: dict[str, Any],
    cfg: TrainerConfig,
    trainer: Trainer,
    locks: dict[str, Any],
) -> dict[str, Any]:
    step = trainer.optimizer_step
    if step not in CHECKPOINT_STEPS:
        raise LadderError(f"unexpected checkpoint step {step}")
    path = scale_out / f"checkpoint-{step:04d}"
    save_trainer_checkpoint(
        path,
        model=trainer.model,
        trainer=trainer,
        identity=_checkpoint_identity(
            source_sha, spec, tok, manifest, run, cfg, trainer, locks
        ),
        overwrite=True,
    )
    checked = verify_checkpoint(path)
    return {
        "step": step,
        "tokens_seen": trainer.tokens_seen,
        "checkpoint_id": checked["checkpoint_id"],
    }


def _eval_record(
    model: TwelveSixDecoder,
    corpus: Path,
    manifest: dict[str, Any],
    tok: ByteTokenizer,
    step: int,
) -> dict[str, Any]:
    result = m100._evaluate(model, corpus, manifest, tok)
    result["step"] = step
    return result


def _training_curve_stats(scale_out: Path) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in (scale_out / "train-curve.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != MAX_STEPS:
        raise LadderError(f"expected {MAX_STEPS} train rows, got {len(rows)}")
    first100 = sum(float(r["loss"]) for r in rows[:100]) / 100
    last100 = sum(float(r["loss"]) for r in rows[-100:]) / 100
    return {"first100_mean_loss": first100, "last100_mean_loss": last100}


def phase1(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    started = time.perf_counter()
    manifest, tok, eval_id = _common_truth(repo, source_sha, out, build=False)
    truth = _read_json(out / "ladder-truth.json")
    _check_self_hash(truth)
    if truth["evaluation_identity"]["identity_sha256"] != eval_id["identity_sha256"]:
        raise LadderError("common evaluation identity changed after prepare")

    spec = model_spec(scale)
    init = init_spec()
    cfg = trainer_config()
    locks = m100._locks(repo)
    run = _run_manifest(source_sha, scale, spec, init, tok, manifest, eval_id, cfg, locks)

    scale_out = out / scale
    scale_out.mkdir(parents=True, exist_ok=True)
    _write_json(scale_out / "run-manifest.json", run)
    _write_json(scale_out / "machine-phase1.json", m100._machine(source_sha, locks))

    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    if sum(p.numel() for p in model.parameters()) != spec.parameter_count():
        raise LadderError("runtime parameter count mismatch")
    random_hash = m100._state_hash(model)
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    corpus = out / "corpus-a"

    evaluations: dict[str, Any] = {}
    evaluations["0"] = observer.measure_region(
        "evaluation",
        f"{scale}-heldout-0",
        lambda: _eval_record(model, corpus, manifest, tok, 0),
        optimizer_step=0,
        tokens_seen=0,
    )
    checkpoints = [
        _save_checkpoint(scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks)
    ]
    initial_generation = m100._generation(scale_out / "checkpoint-0000")

    curve = scale_out / "train-curve.jsonl"
    if curve.exists():
        curve.unlink()

    its = m100._train_iters(corpus, manifest, tok, 0)
    batches = {s: m100._batches(it) for s, it in its.items()}
    for i in range(RESUME_STEP):
        stratum = m100.MIXTURE[i % len(m100.MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        m100._append(
            curve,
            {
                "optimizer_step": metrics.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "stratum": stratum,
                "tokens": metrics.tokens,
                "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
                "grad_norm": metrics.grad_norm,
                "learning_rate": metrics.learning_rate,
            },
        )
        if metrics.optimizer_step in (250, 500):
            checkpoints.append(
                observer.measure_region(
                    "checkpoint",
                    f"{scale}-save-{metrics.optimizer_step}",
                    lambda: _save_checkpoint(
                        scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks
                    ),
                    optimizer_step=trainer.optimizer_step,
                    tokens_seen=trainer.tokens_seen,
                )
            )
            evaluations[str(metrics.optimizer_step)] = observer.measure_region(
                "evaluation",
                f"{scale}-heldout-{metrics.optimizer_step}",
                lambda: _eval_record(model, corpus, manifest, tok, trainer.optimizer_step),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            )

    if trainer.optimizer_step != RESUME_STEP:
        raise LadderError("phase1 did not stop at resume step")

    result = _self_hashed(
        {
            "schema": "12-6.learned-base-ladder-phase1.v1",
            "source_sha": source_sha,
            "scale": scale,
            "process_pid": os.getpid(),
            "model": {
                "spec": spec.to_dict(),
                "spec_sha256": spec.identity_sha256(),
                "parameter_count": spec.parameter_count(),
                "init_spec": init.to_dict(),
                "init_spec_sha256": init.identity_sha256(),
                "random_initialization": True,
                "random_init_state_sha256": random_hash,
                "geometry_provenance": SCALE_SPECS[scale]["provenance"],
            },
            "evaluation_identity_sha256": eval_id["identity_sha256"],
            "evaluations": evaluations,
            "checkpoints": checkpoints,
            "initial_generation": initial_generation,
            "optimizer_step": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "observability": observer.summary(),
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": _rss_bytes(),
        }
    )
    _write_json(scale_out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    started = time.perf_counter()
    manifest, tok, eval_id = _common_truth(repo, source_sha, out, build=False)
    spec = model_spec(scale)
    init = init_spec()
    cfg = trainer_config()
    locks = m100._locks(repo)
    run = _run_manifest(source_sha, scale, spec, init, tok, manifest, eval_id, cfg, locks)
    scale_out = out / scale

    persisted_run = _read_json(scale_out / "run-manifest.json")
    _check_self_hash(persisted_run)
    if persisted_run != run:
        raise LadderError("run manifest changed between phase1 and resume")
    p1 = _read_json(scale_out / "phase1.json")
    _check_self_hash(p1)

    _write_json(scale_out / "machine-resume.json", m100._machine(source_sha, locks))
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    loaded = load_trainer_checkpoint(
        scale_out / "checkpoint-0500",
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"]:
        raise LadderError("resume checkpoint run-manifest mismatch")
    if trainer.optimizer_step != RESUME_STEP:
        raise LadderError("resume checkpoint did not restore step 500")

    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    corpus = out / "corpus-a"
    its = m100._train_iters(corpus, manifest, tok, RESUME_STEP)
    batches = {s: m100._batches(it) for s, it in its.items()}
    first_resumed = None
    checkpoints: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = dict(p1["evaluations"])

    for i in range(RESUME_STEP, MAX_STEPS):
        stratum = m100.MIXTURE[i % len(m100.MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        first_resumed = first_resumed or metrics.optimizer_step
        m100._append(
            scale_out / "train-curve.jsonl",
            {
                "optimizer_step": metrics.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "stratum": stratum,
                "tokens": metrics.tokens,
                "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
                "grad_norm": metrics.grad_norm,
                "learning_rate": metrics.learning_rate,
            },
        )
        if metrics.optimizer_step in (750, 1000):
            checkpoints.append(
                observer.measure_region(
                    "checkpoint",
                    f"{scale}-save-{metrics.optimizer_step}",
                    lambda: _save_checkpoint(
                        scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks
                    ),
                    optimizer_step=trainer.optimizer_step,
                    tokens_seen=trainer.tokens_seen,
                )
            )
            evaluations[str(metrics.optimizer_step)] = observer.measure_region(
                "evaluation",
                f"{scale}-heldout-{metrics.optimizer_step}",
                lambda: _eval_record(model, corpus, manifest, tok, trainer.optimizer_step),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            )

    if first_resumed != 501 or trainer.optimizer_step != MAX_STEPS:
        raise LadderError(
            f"fresh resume transition invalid: first={first_resumed} final={trainer.optimizer_step}"
        )

    stats = _training_curve_stats(scale_out)
    if not stats["last100_mean_loss"] < stats["first100_mean_loss"]:
        raise LadderError("training loss did not decrease")

    initial_bpb = float(evaluations["0"]["bits_per_byte"])
    final_bpb = float(evaluations[str(MAX_STEPS)]["bits_per_byte"])
    if not final_bpb < initial_bpb:
        raise LadderError("final held-out BPB did not improve from random initialization")

    learned = [
        (int(step), float(value["bits_per_byte"]))
        for step, value in evaluations.items()
        if int(step) > 0
    ]
    best_step, best_bpb = min(learned, key=lambda item: (item[1], item[0]))
    best_path = scale_out / f"checkpoint-{best_step:04d}"
    final_path = scale_out / f"checkpoint-{MAX_STEPS:04d}"
    best_generation = m100._generation(best_path)
    final_generation = m100._generation(final_path)

    all_manifests = {
        str(step): verify_checkpoint(scale_out / f"checkpoint-{step:04d}")
        for step in CHECKPOINT_STEPS
    }
    final_manifest = all_manifests[str(MAX_STEPS)]
    best_manifest = all_manifests[str(best_step)]

    phase1_wall = float(p1["wall_seconds"])
    resume_wall = time.perf_counter() - started
    report = _self_hashed(
        {
            "schema": "12-6.learned-base-ladder-scale-report.v1",
            "authority": AUTHORITY,
            "source": {"repository": REPOSITORY, "branch": BRANCH, "git_sha": source_sha},
            "scale": scale,
            "model": p1["model"],
            "tokenizer": {
                "version": tok.identity.version,
                "config_sha256": tok.identity.config_sha256,
                "vocab_sha256": tok.identity.vocab_sha256,
                "vocab_size": tok.identity.vocab_size,
                "special_tokens": dict(tok.identity.special_tokens),
            },
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "evaluation_identity_sha256": eval_id["identity_sha256"],
            "training": {
                "trainer_config": asdict(cfg),
                "optimizer": "AdamW",
                "batch_size": BATCH,
                "sequence_length": SEQ,
                "optimized_tokens": trainer.tokens_seen,
                "first100_mean_loss": stats["first100_mean_loss"],
                "last100_mean_loss": stats["last100_mean_loss"],
                "loss_decreased": True,
            },
            "evaluation": {
                "checkpoints": evaluations,
                "initial_bits_per_byte": initial_bpb,
                "final_bits_per_byte": final_bpb,
                "best_bits_per_byte": best_bpb,
                "best_step": best_step,
                "by_stratum_final": evaluations[str(MAX_STEPS)]["by_stratum"],
                "by_stratum_best": evaluations[str(best_step)]["by_stratum"],
                "all_non_mutating": all(
                    bool(value["non_mutation_passed"]) for value in evaluations.values()
                ),
            },
            "checkpoints": {
                "all_steps": list(CHECKPOINT_STEPS),
                "best_checkpoint": f"checkpoint-{best_step:04d}",
                "best_checkpoint_id": best_manifest["checkpoint_id"],
                "final_checkpoint": f"checkpoint-{MAX_STEPS:04d}",
                "final_checkpoint_id": final_manifest["checkpoint_id"],
            },
            "resume": {
                "loaded_step": loaded.manifest["identity"]["step"],
                "first_resumed_optimizer_step": first_resumed,
                "phase1_pid": p1["process_pid"],
                "resume_pid": os.getpid(),
                "fresh_process": p1["process_pid"] != os.getpid(),
                "passed": p1["process_pid"] != os.getpid() and first_resumed == 501,
            },
            "generation": {
                "random_init": p1["initial_generation"],
                "best_checkpoint": best_generation,
                "final_checkpoint": final_generation,
            },
            "compute": {
                "device": "cpu",
                "phase1_wall_seconds": phase1_wall,
                "resume_wall_seconds": resume_wall,
                "total_train_wall_seconds": phase1_wall + resume_wall,
                "phase1_peak_rss_bytes": p1["peak_rss_bytes"],
                "resume_peak_rss_bytes": _rss_bytes(),
                "peak_rss_bytes": max(int(p1["peak_rss_bytes"]), _rss_bytes()),
                "optimized_tokens_per_wall_second": trainer.tokens_seen
                / max(phase1_wall + resume_wall, 1e-12),
                "phase1_observability": p1["observability"],
                "resume_observability": observer.summary(),
            },
            "truth_boundary": {
                "learned_from_random_initialization": True,
                "foreign_pretrained_weights": False,
                "sft": False,
                "rlhf": False,
                "dpo": False,
                "paid_compute": False,
                "instruction_following_claim": False,
                "alignment_claim": False,
                "production_readiness_claim": False,
                "intelligence_claim": False,
                "external_real_world_training_data_present": False,
                "representative_external_corpus_claim": False,
            },
            "fresh_verification": {"status": "PENDING"},
        }
    )
    _write_json(scale_out / "report.preverify.json", report)
    return report


def _logits_snapshot(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    outputs: dict[str, Any] = {}
    for name, prompt in m100.PROMPTS.items():
        ids = backend.encode(prompt)
        logits = list(backend.next_token_logits(ids))
        if len(logits) != 256 or not all(math.isfinite(float(x)) for x in logits):
            raise LadderError("first-party logits are invalid")
        packed = b"".join(struct.pack("<f", float(x)) for x in logits)
        ranked = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))[:8]
        outputs[name] = {
            "prompt": prompt,
            "input_ids": ids,
            "logits_float32_sha256": hashlib.sha256(packed).hexdigest(),
            "argmax_token_id": ranked[0],
            "top8_token_ids": ranked,
        }
    return {"backend_diagnostics": backend.diagnostics(), "outputs": outputs}


def _compare_eval(recorded: dict[str, Any], fresh: dict[str, Any]) -> None:
    for key in ("loss", "bits_per_byte", "perplexity"):
        if abs(float(recorded[key]) - float(fresh[key])) > VERIFY_TOL:
            raise LadderError(f"fresh evaluation mismatch for {key}")
    if int(recorded["predicted_byte_tokens"]) != int(fresh["predicted_byte_tokens"]):
        raise LadderError("fresh evaluation target-token count mismatch")
    for stratum in STRATA:
        rb = recorded["by_stratum"][stratum]
        fb = fresh["by_stratum"][stratum]
        for key in ("loss", "bits_per_byte", "perplexity"):
            if abs(float(rb[key]) - float(fb[key])) > VERIFY_TOL:
                raise LadderError(f"fresh {stratum} evaluation mismatch for {key}")
        if int(rb["predicted_byte_tokens"]) != int(fb["predicted_byte_tokens"]):
            raise LadderError(f"fresh {stratum} target-token count mismatch")


def verify_scale(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    manifest, tok, eval_id = _common_truth(repo, source_sha, out, build=False)
    scale_out = out / scale
    report = _read_json(scale_out / "report.preverify.json")
    _check_self_hash(report)
    run = _read_json(scale_out / "run-manifest.json")
    _check_self_hash(run)

    if report["evaluation_identity_sha256"] != eval_id["identity_sha256"]:
        raise LadderError("scale report evaluation identity mismatch")
    if report["corpus_identity_sha256"] != manifest["corpus_identity_sha256"]:
        raise LadderError("scale report corpus identity mismatch")

    best_step = int(report["evaluation"]["best_step"])
    steps = sorted({best_step, MAX_STEPS})
    fresh: dict[str, Any] = {}

    for step in steps:
        checkpoint = scale_out / f"checkpoint-{step:04d}"
        checked = verify_checkpoint(checkpoint)
        identity = checked["identity"]
        spec = model_spec(scale)
        if identity["git_sha"] != source_sha:
            raise LadderError("checkpoint source SHA mismatch")
        if identity["model_spec_hash"] != spec.identity_sha256():
            raise LadderError("checkpoint ModelSpec identity mismatch")
        if int(identity["parameter_count"]) != spec.parameter_count():
            raise LadderError("checkpoint parameter count mismatch")
        if identity["tokenizer_hash"] != tok.identity.config_sha256:
            raise LadderError("checkpoint tokenizer identity mismatch")
        if identity["tokenizer_vocab_hash"] != tok.identity.vocab_sha256:
            raise LadderError("checkpoint tokenizer vocab identity mismatch")
        if identity["dataset_manifest_hash"] != manifest["corpus_identity_sha256"]:
            raise LadderError("checkpoint corpus identity mismatch")
        if identity["run_manifest_hash"] != run["identity_sha256"]:
            raise LadderError("checkpoint run-manifest identity mismatch")
        if int(identity["step"]) != step:
            raise LadderError("checkpoint step identity mismatch")

        logits1 = _logits_snapshot(checkpoint)
        logits2 = _logits_snapshot(checkpoint)
        if logits1 != logits2:
            raise LadderError("first-party logits are not reproducible")

        backend = load_first_party_backend(checkpoint)
        fresh_eval = m100._evaluate(backend.model, out / "corpus-a", manifest, tok)
        if not fresh_eval["non_mutation_passed"]:
            raise LadderError("fresh evaluation mutated checkpoint model")
        recorded_eval = report["evaluation"]["checkpoints"][str(step)]
        _compare_eval(recorded_eval, fresh_eval)

        generation = m100._generation(checkpoint)
        key = "best_checkpoint" if step == best_step else "final_checkpoint"
        expected_generation = report["generation"][key]
        if generation != expected_generation:
            if step == MAX_STEPS and best_step == MAX_STEPS:
                if generation != report["generation"]["best_checkpoint"]:
                    raise LadderError("fresh generation mismatch")
            else:
                raise LadderError("fresh generation mismatch")

        fresh[str(step)] = {
            "checkpoint_id": checked["checkpoint_id"],
            "checkpoint_identity": identity,
            "first_party_logits": logits1,
            "evaluation": fresh_eval,
            "generation": generation,
        }

    verified_report = dict(report)
    verified_report["fresh_verification"] = {
        "status": "PASS",
        "verified_steps": steps,
        "checkpoint_load": True,
        "first_party_logits": True,
        "evaluation_non_mutation": True,
        "checkpoint_identity": True,
        "generation": True,
        "reproducibility_manifest_validation": True,
        "evidence": fresh,
    }
    verified_report.pop("identity_sha256", None)
    verified_report = _self_hashed(verified_report)
    _write_json(scale_out / "report.json", verified_report)
    return verified_report


def _quality_rank(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        reports,
        key=lambda r: (
            float(r["evaluation"]["best_bits_per_byte"]),
            int(r["model"]["parameter_count"]),
        ),
    )
    return [
        {
            "rank": i + 1,
            "scale": r["scale"],
            "best_bits_per_byte": r["evaluation"]["best_bits_per_byte"],
            "best_step": r["evaluation"]["best_step"],
        }
        for i, r in enumerate(ordered)
    ]


def _efficiency_rank(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in reports:
        initial = float(r["evaluation"]["initial_bits_per_byte"])
        best = float(r["evaluation"]["best_bits_per_byte"])
        tokens = int(r["training"]["optimized_tokens"])
        rows.append(
            {
                "scale": r["scale"],
                "optimized_tokens_per_wall_second": float(
                    r["compute"]["optimized_tokens_per_wall_second"]
                ),
                "bpb_reduction_per_million_optimized_tokens": (initial - best)
                / max(tokens / 1_000_000.0, 1e-12),
            }
        )
    rows.sort(
        key=lambda x: (
            -x["bpb_reduction_per_million_optimized_tokens"],
            -x["optimized_tokens_per_wall_second"],
        )
    )
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def finalize(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    m100._require_head(repo, source_sha)
    truth = _read_json(out / "ladder-truth.json")
    _check_self_hash(truth)
    reports = []
    for scale in SCALE_ORDER:
        r = _read_json(out / scale / "report.json")
        _check_self_hash(r)
        if r["fresh_verification"]["status"] != "PASS":
            raise LadderError(f"{scale} has no fresh verification PASS")
        if r["source"]["git_sha"] != source_sha:
            raise LadderError(f"{scale} source SHA mismatch")
        if r["corpus_identity_sha256"] != truth["corpus_identity_sha256"]:
            raise LadderError(f"{scale} corpus identity mismatch")
        if r["evaluation_identity_sha256"] != truth["evaluation_identity"]["identity_sha256"]:
            raise LadderError(f"{scale} evaluation identity mismatch")
        if r["tokenizer"]["config_sha256"] != truth["tokenizer"]["config_sha256"]:
            raise LadderError(f"{scale} tokenizer identity mismatch")
        reports.append(r)

    by_scale = {r["scale"]: r for r in reports}
    scaling = []
    for left, right in itertools.pairwise(SCALE_ORDER):
        a = float(by_scale[left]["evaluation"]["best_bits_per_byte"])
        b = float(by_scale[right]["evaluation"]["best_bits_per_byte"])
        scaling.append(
            {
                "from": left,
                "to": right,
                "from_parameters": by_scale[left]["model"]["parameter_count"],
                "to_parameters": by_scale[right]["model"]["parameter_count"],
                "best_bpb_absolute_change": b - a,
                "best_bpb_relative_change": (b - a) / a,
                "quality_improved": b < a,
            }
        )

    ladder = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "branch": BRANCH, "git_sha": source_sha},
        "truth_model": truth,
        "minimum_comparable_ladder_complete": True,
        "scales": {r["scale"]: r for r in reports},
        "ten_million": {
            "status": "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE",
            "included_in_rankings": False,
            "reason": (
                "Existing ~10M evidence proves model/trainer/runtime mechanics on controlled "
                "synthetic or non-frozen data, not a genuinely learned checkpoint evaluated "
                "under this DATA-25/tokenizer/evaluation identity."
            ),
            "unsupported_fields_absent": [
                "held_out_bpb",
                "ua_en_code_breakdown",
                "best_checkpoint",
                "final_checkpoint",
                "raw_base_generation",
            ],
        },
        "rankings": {
            "quality": _quality_rank(reports),
            "efficiency": _efficiency_rank(reports),
            "scaling_improvement": scaling,
        },
        "claims": {
            "learned_base_ladder": True,
            "intelligence": False,
            "production_readiness": False,
            "alignment": False,
            "instruction_following": False,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "paid_compute": False,
            "representative_external_corpus": False,
        },
    }
    ladder = _self_hashed(ladder, "report_sha256")
    _write_json(out / "ladder-report.json", ladder)
    return ladder


def validate_ladder(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    _check_self_hash(report, "report_sha256")
    if report["schema"] != SCHEMA or report["authority"] != AUTHORITY:
        raise LadderError("ladder schema/authority mismatch")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise LadderError("ladder source SHA mismatch")
    if report["minimum_comparable_ladder_complete"] is not True:
        raise LadderError("minimum 100K/500K/1M comparable ladder incomplete")
    if set(report["scales"]) != set(SCALE_ORDER):
        raise LadderError("unexpected learned ladder scale set")
    identities = {r["evaluation_identity_sha256"] for r in report["scales"].values()}
    corpora = {r["corpus_identity_sha256"] for r in report["scales"].values()}
    tokenizers = {
        (r["tokenizer"]["config_sha256"], r["tokenizer"]["vocab_sha256"])
        for r in report["scales"].values()
    }
    if len(identities) != 1 or len(corpora) != 1 or len(tokenizers) != 1:
        raise LadderError("ladder common truth identity gate failed")
    for scale, r in report["scales"].items():
        if r["fresh_verification"]["status"] != "PASS":
            raise LadderError(f"{scale} fresh verification missing")
        if r["truth_boundary"]["foreign_pretrained_weights"] is not False:
            raise LadderError(f"{scale} foreign pretrained weights truth weakened")
        if not r["resume"]["passed"]:
            raise LadderError(f"{scale} resume evidence failed")
    if report["ten_million"]["status"] != "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE":
        raise LadderError("10M must remain incomplete without comparable learned evidence")
    for forbidden in ("intelligence", "production_readiness", "alignment", "instruction_following"):
        if report["claims"][forbidden] is not False:
            raise LadderError(f"unsupported claim enabled: {forbidden}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=Path("."))
    common.add_argument("--source-sha", required=True)
    common.add_argument("--output-dir", type=Path, required=True)

    sub.add_parser("prepare", parents=[common])
    for name in ("phase1", "resume", "verify-scale"):
        q = sub.add_parser(name, parents=[common])
        q.add_argument("--scale", choices=SCALE_ORDER, required=True)
    sub.add_parser("finalize", parents=[common])

    q = sub.add_parser("validate")
    q.add_argument("report", type=Path)
    q.add_argument("--expected-source-sha")

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = prepare(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
        print(json.dumps({"prepared": True, "truth_identity": result["identity_sha256"]}, indent=2))
    elif args.cmd == "phase1":
        result = phase1(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve(), args.scale)
        print(json.dumps({"scale": args.scale, "phase": "phase1", "step": result["optimizer_step"], "optimized_tokens": result["optimized_tokens"]}, indent=2))
    elif args.cmd == "resume":
        result = resume(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve(), args.scale)
        print(json.dumps({"scale": args.scale, "phase": "resume", "best_step": result["evaluation"]["best_step"], "best_bpb": result["evaluation"]["best_bits_per_byte"], "final_bpb": result["evaluation"]["final_bits_per_byte"]}, indent=2))
    elif args.cmd == "verify-scale":
        result = verify_scale(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve(), args.scale)
        print(json.dumps({"scale": args.scale, "fresh_verification": result["fresh_verification"]["status"], "verified_steps": result["fresh_verification"]["verified_steps"]}, indent=2))
    elif args.cmd == "finalize":
        result = finalize(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve())
        print(json.dumps({"minimum_comparable_ladder_complete": result["minimum_comparable_ladder_complete"], "quality_rank": result["rankings"]["quality"], "ten_million": result["ten_million"]["status"], "report_sha256": result["report_sha256"]}, indent=2))
    else:
        result = validate_ladder(args.report, args.expected_source_sha)
        print(json.dumps({"validation": "PASS", "report_sha256": result["report_sha256"], "ten_million": result["ten_million"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
