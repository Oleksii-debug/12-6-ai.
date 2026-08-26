"""RESEARCH-192 clean 1M -> 3M -> 10M fixed-recipe scaling transfer."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.research192-scaling-transfer.v2"
ARM_SCHEMA = "12-6.research192-scaling-transfer-arm.v2"
AUTHORITY = "LOCAL_FREE_FIXED_RECIPE_SCALING_TRANSFER_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
BRANCH = "research192/one-three-ten-million-20260826"
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_EVALUATION_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
CHECKPOINT_STEPS = (18, 70, 139)
EXPECTED_TOKEN_BUDGETS = {18: 17_125, 70: 66_417, 139: 131_938}
MIDPOINT_STEP = 70
FINAL_STEP = 139
PAIRED_SEEDS = (1337, 1338)
ARM_MATRIX = (("1m", 1337), ("1m", 1338), ("3m", 1337), ("3m", 1338), ("10m", 1337))
M150_PRODUCER = {
    "source_sha": "5838cd16869dcfcf762368d8673eddf52d51b7e3",
    "workflow_run_id": 32937411703,
    "artifact_id": 9595677772,
    "artifact_name": "milestone150-learned-base-ladder-v1",
    "artifact_sha256": "c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71",
    "ladder_report_sha256": "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a",
    "one_m_report_identity_sha256": "1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738",
}
LEARN191_GEOMETRY = {
    "pr": 348,
    "source_sha": "a75920cef8bde37a8c590e34095be83c97b75f1d",
    "model_spec_sha256": "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc",
    "parameters": 3_213_120,
    "nominal_targets": [16_632, 65_772, 131_292],
    "role": "geometry/budget preregistration authority; checkpoint reuse requires terminal artifact",
}


def _model_payload(d: int, layers: int, heads: int, ff: int) -> dict[str, Any]:
    if d % heads:
        raise ValueError("d_model must divide evenly across heads")
    hd = d // heads
    return {
        "schema_version": 1, "vocab_size": 256, "max_seq_len": 256,
        "d_model": d, "n_layers": layers, "n_heads": heads, "n_kv_heads": heads,
        "head_dim": hd, "d_ff": ff, "activation": "swiglu", "norm_kind": "rmsnorm",
        "norm_placement": "pre", "norm_eps": 1e-5, "position_embedding": "rope",
        "rope_theta": 10_000.0, "rope_rotary_dim": hd, "attention_bias": False,
        "mlp_bias": False, "attention_dropout": 0.0, "final_norm": True,
        "tie_word_embeddings": True, "lm_head_bias": False,
    }


SCALE_SPECS: dict[str, dict[str, Any]] = {
    "1m": {
        "expected_parameters": 1_037_696,
        "expected_model_spec_sha256": "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07",
        "model": _model_payload(128, 5, 8, 352),
        "provenance": "accepted MILESTONE-150 fixed-control incumbent geometry",
    },
    "3m": {
        "expected_parameters": 3_213_120,
        "expected_model_spec_sha256": "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc",
        "model": _model_payload(192, 7, 12, 528),
        "provenance": "LEARN-191 bridge geometry: RESEARCH-138 target, fixed-family 2.75x FFN continuation",
    },
    "10m": {
        "expected_parameters": 10_000_640,
        "expected_model_spec_sha256": "f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53",
        "model": _model_payload(256, 12, 16, 736),
        "provenance": "10M MHA/context-256 control; matches S3 parameter count without importing S3 GQA/runtime changes",
    },
}


class Research192Error(RuntimeError):
    pass


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Research192Error(f"{path} must contain a JSON object")
    return value


def writej(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def appendj(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def selfhash(payload: dict[str, Any], key: str = "identity_sha256") -> dict[str, Any]:
    value = dict(payload)
    value[key] = hash_json(value)
    return value


def spec_for(scale: str) -> ModelSpec:
    if scale not in SCALE_SPECS:
        raise Research192Error(f"unknown scale {scale}")
    cfg = SCALE_SPECS[scale]
    spec = ModelSpec.from_dict(dict(cfg["model"]))
    if spec.parameter_count() != cfg["expected_parameters"]:
        raise Research192Error(f"{scale} parameter-count drift")
    if spec.identity_sha256() != cfg["expected_model_spec_sha256"]:
        raise Research192Error(f"{scale} ModelSpec identity drift")
    return spec


def init_spec() -> InitSpec:
    value = InitSpec()
    if value.identity_sha256() != INIT_SPEC_SHA256:
        raise Research192Error("InitSpec drift")
    return value


def trainer_config(seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8,
        max_steps=FINAL_STEP, warmup_steps=0, scheduler="constant",
        gradient_accumulation_steps=1, gradient_clip_norm=1.0, precision="fp32",
        seed=seed, deterministic_algorithms=True, deterministic_warn_only=False,
    )


def common(repo: Path, source_sha: str, out: Path, build: bool):
    m100._require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = m100._build_corpus(repo, out) if build else readj(out / "corpus-manifest.json")
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Research192Error("DATA-25 corpus identity drift")
    if manifest["train_validation_content_overlap"] != 0:
        raise Research192Error("train/validation leakage")
    tok = ByteTokenizer()
    if tok.identity.vocab_size != 256 or tok.identity.special_tokens:
        raise Research192Error("canonical byte tokenizer drift")
    eval_id = m150.evaluation_identity(tok, manifest)
    if eval_id["identity_sha256"] != EXPECTED_EVALUATION_ID:
        raise Research192Error("M150 evaluation identity drift")
    train_bytes = int(manifest["by_split"]["train"]["byte_tokens"])
    if EXPECTED_TOKEN_BUDGETS[FINAL_STEP] / train_bytes >= 0.01:
        raise Research192Error("source-exposure ceiling exceeded")
    return manifest, tok, eval_id


def run_manifest(source_sha: str, scale: str, seed: int, manifest, tok, eval_id, spec, init, cfg, locks):
    payload = {
        "schema": "12-6.research192-run-manifest.v2",
        "worker_id": "RESEARCH-192-ONE-THREE-TEN-MILLION",
        "source_sha": source_sha, "scale": scale, "seed": seed,
        "model_spec": spec.to_dict(), "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(), "init_spec": init.to_dict(),
        "init_spec_sha256": init.identity_sha256(),
        "tokenizer": {"version": tok.identity.version, "config_sha256": tok.identity.config_sha256,
                      "vocab_sha256": tok.identity.vocab_sha256, "vocab_size": tok.identity.vocab_size,
                      "special_tokens": dict(tok.identity.special_tokens)},
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "evaluation_identity_sha256": eval_id["identity_sha256"],
        "packing": {"version": m100.PACKING_VERSION, "sequence_length": 128, "cross_document": False},
        "trainer_config": asdict(cfg), "batch_size": 8, "mixture_pattern": list(m100.MIXTURE),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "exact_optimized_token_budgets": {str(k): v for k, v in EXPECTED_TOKEN_BUDGETS.items()},
        "midpoint_resume_step": MIDPOINT_STEP,
        "environment_lock_sha256": locks["combined_sha256"],
        "random_initialization": True, "foreign_pretrained_weights": False, "sft": False,
        "rlhf": False, "dpo": False, "paid_compute": False,
    }
    persisted = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return selfhash(persisted)


def checkpoint_identity(source_sha, spec, tok, manifest, run, cfg, trainer, locks):
    return CheckpointIdentity(
        git_sha=source_sha, model_spec=spec.to_dict(), parameter_count=spec.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256, tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=manifest["corpus_identity_sha256"], run_manifest_hash=run["identity_sha256"],
        training_config={"trainer": asdict(cfg), "evaluation_identity_sha256": run["evaluation_identity_sha256"]},
        seed=cfg.seed, precision=cfg.precision, step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "learning_rate": cfg.learning_rate, "betas": list(cfg.betas),
                   "eps": cfg.eps, "weight_decay": cfg.weight_decay},
        scheduler=None, environment_lock_hash=locks["combined_sha256"],
    )


def checkpoint_dir(scale_out: Path, step: int) -> Path:
    return scale_out / f"checkpoint-{step:04d}"


def save_checkpoint(scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks):
    step = trainer.optimizer_step
    if step not in CHECKPOINT_STEPS:
        raise Research192Error(f"unpreregistered checkpoint step {step}")
    expected_tokens = EXPECTED_TOKEN_BUDGETS[step]
    if trainer.tokens_seen != expected_tokens:
        raise Research192Error(f"hidden token advantage at step {step}: {trainer.tokens_seen} != {expected_tokens}")
    path = checkpoint_dir(scale_out, step)
    save_trainer_checkpoint(path, model=trainer.model, trainer=trainer,
                            identity=checkpoint_identity(source_sha, spec, tok, manifest, run, cfg, trainer, locks),
                            overwrite=True)
    checked = verify_checkpoint(path)
    return {"optimizer_step": step, "optimized_tokens": trainer.tokens_seen,
            "checkpoint_id": checked["checkpoint_id"], "path": path.name}


def evalpoint(model, corpus, manifest, tok, trainer, metrics, segment_nll, segment_tokens, wall_seconds):
    heldout = m100._evaluate(model, corpus, manifest, tok)
    online_bpb = float(metrics.update_loss if metrics.update_loss is not None else metrics.loss) / math.log(2.0)
    segment_bpb = segment_nll / math.log(2.0) / segment_tokens
    return {
        "optimizer_step": trainer.optimizer_step, "optimized_tokens": trainer.tokens_seen,
        "heldout": heldout, "checkpoint_training_bpb": online_bpb,
        "segment_mean_training_bpb": segment_bpb,
        "generalization_gap_bpb": float(heldout["bits_per_byte"]) - online_bpb,
        "gradient_norm_pre_clip": metrics.grad_norm,
        "clip_active": bool(metrics.grad_norm is not None and metrics.grad_norm > 1.0),
        "wall_seconds_end_to_end_in_process": wall_seconds,
        "peak_rss_bytes": m150._rss_bytes(),
    }


def prepare(repo: Path, source_sha: str, out: Path, scale: str, seed: int):
    if (scale, seed) not in ARM_MATRIX:
        raise Research192Error("arm is outside preregistered matrix")
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, eval_id = common(repo, source_sha, out, True)
    spec, init, cfg, locks = spec_for(scale), init_spec(), trainer_config(seed), m100._locks(repo)
    run = run_manifest(source_sha, scale, seed, manifest, tok, eval_id, spec, init, cfg, locks)
    writej(out / "run-manifest.json", run)
    truth = selfhash({
        "schema": "12-6.research192-arm-truth.v2", "source_sha": source_sha,
        "scale": scale, "seed": seed, "parameter_count": spec.parameter_count(),
        "model_spec_sha256": spec.identity_sha256(), "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "evaluation_identity_sha256": eval_id["identity_sha256"],
        "checkpoint_steps": list(CHECKPOINT_STEPS), "optimized_token_budgets": EXPECTED_TOKEN_BUDGETS,
        "source_exposure_fraction_final": EXPECTED_TOKEN_BUDGETS[FINAL_STEP] / int(manifest["by_split"]["train"]["byte_tokens"]),
        "hidden_token_advantage_allowed": False,
    })
    writej(out / "truth.json", truth)
    return truth


def _loaded(repo, source_sha, out, scale, seed):
    manifest, tok, eval_id = common(repo, source_sha, out, False)
    spec, init, cfg, locks = spec_for(scale), init_spec(), trainer_config(seed), m100._locks(repo)
    run = run_manifest(source_sha, scale, seed, manifest, tok, eval_id, spec, init, cfg, locks)
    if readj(out / "run-manifest.json") != run:
        raise Research192Error("run manifest changed after prepare")
    return manifest, tok, eval_id, spec, init, cfg, locks, run


def phase1(repo: Path, source_sha: str, out: Path, scale: str, seed: int):
    started = time.perf_counter()
    manifest, tok, _eval_id, spec, init, cfg, locks, run = _loaded(repo, source_sha, out, scale, seed)
    scale_out = out / scale
    scale_out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    random_hash = m100._state_hash(model)
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(run, device="cpu", max_step_samples=256)
    curve = scale_out / "train-curve.jsonl"
    curve.unlink(missing_ok=True)
    corpus = out / "corpus-a"
    its = m100._train_iters(corpus, manifest, tok, 0)
    batches = {s: m100._batches(it) for s, it in its.items()}
    points, checkpoints = [], []
    segment_nll = 0.0
    segment_tokens = 0
    for i in range(MIDPOINT_STEP):
        stratum = m100.MIXTURE[i % len(m100.MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        loss = float(metrics.update_loss if metrics.update_loss is not None else metrics.loss)
        segment_nll += loss * int(metrics.tokens)
        segment_tokens += int(metrics.tokens)
        appendj(curve, {"optimizer_step": trainer.optimizer_step, "optimized_tokens": trainer.tokens_seen,
                        "stratum": stratum, "training_bpb": loss / math.log(2.0),
                        "gradient_norm_pre_clip": metrics.grad_norm, "learning_rate": metrics.learning_rate})
        if trainer.optimizer_step in (CHECKPOINT_STEPS[0], MIDPOINT_STEP):
            checkpoints.append(save_checkpoint(scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks))
            points.append(evalpoint(model, corpus, manifest, tok, trainer, metrics, segment_nll, segment_tokens,
                                    time.perf_counter() - started))
            segment_nll = 0.0
            segment_tokens = 0
    if trainer.optimizer_step != MIDPOINT_STEP or trainer.tokens_seen != EXPECTED_TOKEN_BUDGETS[MIDPOINT_STEP]:
        raise Research192Error("phase1 midpoint ledger mismatch")
    result = selfhash({
        "schema": "12-6.research192-phase1.v2", "source_sha": source_sha, "scale": scale, "seed": seed,
        "process_pid": os.getpid(), "random_init_state_sha256": random_hash,
        "optimizer_step": trainer.optimizer_step, "optimized_tokens": trainer.tokens_seen,
        "points": points, "checkpoints": checkpoints, "observability": observer.summary(),
        "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": m150._rss_bytes(),
    })
    writej(scale_out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path, scale: str, seed: int):
    started = time.perf_counter()
    manifest, tok, _eval_id, spec, init, cfg, locks, run = _loaded(repo, source_sha, out, scale, seed)
    scale_out = out / scale
    p1 = readj(scale_out / "phase1.json")
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    load_trainer_checkpoint(checkpoint_dir(scale_out, MIDPOINT_STEP), model=model, trainer=trainer,
        strict_model=True, restore_rng=True, expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(), expected_tokenizer_hash=tok.identity.config_sha256,
        expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"], expected_run_manifest_hash=run["identity_sha256"])
    if os.getpid() == int(p1["process_pid"]):
        raise Research192Error("midpoint resume did not cross a process boundary")
    if trainer.optimizer_step != MIDPOINT_STEP or trainer.tokens_seen != EXPECTED_TOKEN_BUDGETS[MIDPOINT_STEP]:
        raise Research192Error("restored midpoint ledger mismatch")
    observer = TrainingObserver(run, device="cpu", max_step_samples=256)
    corpus = out / "corpus-a"
    its = m100._train_iters(corpus, manifest, tok, MIDPOINT_STEP)
    batches = {s: m100._batches(it) for s, it in its.items()}
    curve = scale_out / "train-curve.jsonl"
    segment_nll = 0.0
    segment_tokens = 0
    metrics = None
    for i in range(MIDPOINT_STEP, FINAL_STEP):
        stratum = m100.MIXTURE[i % len(m100.MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        loss = float(metrics.update_loss if metrics.update_loss is not None else metrics.loss)
        segment_nll += loss * int(metrics.tokens)
        segment_tokens += int(metrics.tokens)
        appendj(curve, {"optimizer_step": trainer.optimizer_step, "optimized_tokens": trainer.tokens_seen,
                        "stratum": stratum, "training_bpb": loss / math.log(2.0),
                        "gradient_norm_pre_clip": metrics.grad_norm, "learning_rate": metrics.learning_rate})
    if metrics is None or trainer.optimizer_step != FINAL_STEP or trainer.tokens_seen != EXPECTED_TOKEN_BUDGETS[FINAL_STEP]:
        raise Research192Error("final ledger mismatch")
    checkpoint = save_checkpoint(scale_out, source_sha, spec, tok, manifest, run, cfg, trainer, locks)
    point = evalpoint(model, corpus, manifest, tok, trainer, metrics, segment_nll, segment_tokens,
                      float(p1["wall_seconds"]) + time.perf_counter() - started)
    result = selfhash({
        "schema": "12-6.research192-resume.v2", "source_sha": source_sha, "scale": scale, "seed": seed,
        "process_pid": os.getpid(), "phase1_process_pid": p1["process_pid"], "fresh_process_resume_passed": True,
        "optimizer_step": trainer.optimizer_step, "optimized_tokens": trainer.tokens_seen,
        "point": point, "checkpoint": checkpoint, "observability": observer.summary(),
        "resume_wall_seconds": time.perf_counter() - started,
        "total_train_eval_wall_seconds": float(p1["wall_seconds"]) + time.perf_counter() - started,
        "peak_rss_bytes": max(int(p1["peak_rss_bytes"]), m150._rss_bytes()),
    })
    writej(scale_out / "resume.json", result)
    return result


def verify(repo: Path, source_sha: str, out: Path, scale: str, seed: int):
    manifest, tok, _eval_id, spec, init, cfg, _locks, run = _loaded(repo, source_sha, out, scale, seed)
    scale_out = out / scale
    resume_report = readj(scale_out / "resume.json")
    proof = {"schema": "12-6.research192-fresh-verify.v2", "source_sha": source_sha,
             "scale": scale, "seed": seed, "process_pid": os.getpid(), "checkpoints": []}
    for step in CHECKPOINT_STEPS:
        path = checkpoint_dir(scale_out, step)
        checked = verify_checkpoint(path)
        torch.manual_seed(seed)
        model = TwelveSixDecoder(spec, init)
        trainer = Trainer(model, cfg, device="cpu")
        load_trainer_checkpoint(path, model=model, trainer=trainer, strict_model=True, restore_rng=False,
            expected_git_sha=source_sha, expected_model_spec_hash=spec.identity_sha256(),
            expected_tokenizer_hash=tok.identity.config_sha256, expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,
            expected_dataset_manifest_hash=manifest["corpus_identity_sha256"], expected_run_manifest_hash=run["identity_sha256"])
        if trainer.optimizer_step != step or trainer.tokens_seen != EXPECTED_TOKEN_BUDGETS[step]:
            raise Research192Error("fresh verification token/step mismatch")
        before = m100._state_hash(model)
        ev = m100._evaluate(model, out / "corpus-a", manifest, tok)
        after = m100._state_hash(model)
        if before != after or not ev["non_mutation_passed"]:
            raise Research192Error("fresh evaluation mutated retained model")
        proof["checkpoints"].append({"optimizer_step": step, "optimized_tokens": trainer.tokens_seen,
            "checkpoint_id": checked["checkpoint_id"], "heldout_bpb": ev["bits_per_byte"],
            "model_state_sha256": before})
    if os.getpid() in {int(resume_report["process_pid"]), int(resume_report["phase1_process_pid"])}:
        raise Research192Error("verification was not a fresh process")
    proof["status"] = "PASS"
    proof = selfhash(proof)
    writej(scale_out / "fresh-verify.json", proof)
    return proof


def summarize(out: Path, scale: str, seed: int):
    p1 = readj(out / scale / "phase1.json")
    res = readj(out / scale / "resume.json")
    proof = readj(out / scale / "fresh-verify.json")
    run = readj(out / "run-manifest.json")
    if proof["status"] != "PASS" or not res["fresh_process_resume_passed"]:
        raise Research192Error("arm integrity proof failed")
    points = {str(p["optimizer_step"]): p for p in [*p1["points"], res["point"]]}
    if set(map(int, points)) != set(CHECKPOINT_STEPS):
        raise Research192Error("arm checkpoint set mismatch")
    arm = selfhash({
        "schema": ARM_SCHEMA, "authority": AUTHORITY, "scale": scale, "seed": seed,
        "source_sha": run["source_sha"], "parameter_count": run["parameter_count"],
        "model_spec_sha256": run["model_spec_sha256"], "init_spec_sha256": run["init_spec_sha256"],
        "corpus_identity_sha256": run["corpus_identity_sha256"],
        "evaluation_identity_sha256": run["evaluation_identity_sha256"], "tokenizer": run["tokenizer"],
        "packing": run["packing"], "trainer_config": run["trainer_config"], "batch_size": run["batch_size"],
        "points": points, "fresh_process_resume": True, "fresh_verification": "PASS",
        "parameter_bytes_fp32": int(run["parameter_count"]) * 4,
        "peak_rss_bytes": max(int(p1["peak_rss_bytes"]), int(res["peak_rss_bytes"])),
        "total_train_eval_wall_seconds": float(res["total_train_eval_wall_seconds"]),
    })
    writej(out / "research192-arm.json", arm)
    return arm


def _non_size(arm: dict[str, Any]):
    cfg = dict(arm["trainer_config"])
    cfg.pop("seed", None)
    return {"corpus": arm["corpus_identity_sha256"], "evaluation": arm["evaluation_identity_sha256"],
            "tokenizer": arm["tokenizer"], "packing": arm["packing"], "trainer_except_seed": cfg,
            "batch_size": arm["batch_size"]}


def compare(arms_root: Path, m150_root: Path, out: Path):
    arms = [readj(p) for p in sorted(arms_root.rglob("research192-arm.json"))]
    by = {(a["scale"], int(a["seed"])): a for a in arms}
    if set(by) != set(ARM_MATRIX):
        raise Research192Error(f"incomplete arm matrix: {sorted(by)}")
    ref = _non_size(by[("1m", 1337)])
    for arm in arms:
        if _non_size(arm) != ref:
            raise Research192Error("non-size variable drift")
        for step in CHECKPOINT_STEPS:
            if int(arm["points"][str(step)]["optimized_tokens"]) != EXPECTED_TOKEN_BUDGETS[step]:
                raise Research192Error("hidden token advantage detected")

    incumbent_report = readj(m150_root / "1m" / "report.json")
    ladder = readj(m150_root / "ladder-report.json")
    if (incumbent_report["identity_sha256"] != M150_PRODUCER["one_m_report_identity_sha256"] or
            ladder["report_sha256"] != M150_PRODUCER["ladder_report_sha256"]):
        raise Research192Error("accepted M150 incumbent identity mismatch")
    rows_by_step = {}
    for step in CHECKPOINT_STEPS:
        rows = []
        for scale, seed in ARM_MATRIX:
            arm = by[(scale, seed)]
            point = arm["points"][str(step)]
            h = point["heldout"]
            wall = float(point["wall_seconds_end_to_end_in_process"])
            if step == FINAL_STEP:
                wall = float(arm["total_train_eval_wall_seconds"])
            n = int(arm["parameter_count"])
            t = int(point["optimized_tokens"])
            rows.append({
                "scale": scale, "seed": seed, "parameter_count": n, "optimized_tokens": t,
                "heldout_bpb": float(h["bits_per_byte"]),
                "ua_bpb": float(h["by_stratum"]["uk"]["bits_per_byte"]),
                "en_bpb": float(h["by_stratum"]["en"]["bits_per_byte"]),
                "code_bpb": float(h["by_stratum"]["code"]["bits_per_byte"]),
                "training_bpb": float(point["checkpoint_training_bpb"]),
                "segment_mean_training_bpb": float(point["segment_mean_training_bpb"]),
                "generalization_gap_bpb": float(point["generalization_gap_bpb"]),
                "compute_proxy_6nt": 6 * n * t, "wall_seconds": wall,
                "parameter_bytes_fp32": int(arm["parameter_bytes_fp32"]),
                "peak_rss_bytes": int(arm["peak_rss_bytes"]),
                "throughput_optimized_tokens_per_wall_second": t / max(wall, 1e-12),
            })
        rows_by_step[str(step)] = {"optimized_tokens": EXPECTED_TOKEN_BUDGETS[step], "rows": rows}

    paired = {}
    for step in CHECKPOINT_STEPS:
        ds = []
        for seed in PAIRED_SEEDS:
            a = float(by[("1m", seed)]["points"][str(step)]["heldout"]["bits_per_byte"])
            b = float(by[("3m", seed)]["points"][str(step)]["heldout"]["bits_per_byte"])
            ds.append(a - b)
        paired[str(step)] = {"paired_seeds": list(PAIRED_SEEDS), "bpb_improvements_1m_to_3m": ds,
                             "mean_bpb_improvement": sum(ds) / len(ds),
                             "direction_consistent": all(x > 0 for x in ds) or all(x < 0 for x in ds),
                             "promotion_authority": False}
    efficiency = []
    for step in CHECKPOINT_STEPS:
        t = EXPECTED_TOKEN_BUDGETS[step]
        for left, right in (("1m", "3m"), ("3m", "10m")):
            a, b = by[(left, 1337)], by[(right, 1337)]
            improvement = (float(a["points"][str(step)]["heldout"]["bits_per_byte"]) -
                           float(b["points"][str(step)]["heldout"]["bits_per_byte"]))
            dn = int(b["parameter_count"]) - int(a["parameter_count"])
            dc = 6 * dn * t
            efficiency.append({"optimizer_step": step, "optimized_tokens": t, "seed": 1337,
                "from": left, "to": right, "heldout_bpb_improvement": improvement,
                "added_parameters": dn, "improvement_per_added_parameter": improvement / dn,
                "incremental_compute_proxy_6_delta_n_t": dc, "improvement_per_incremental_compute": improvement / dc})
    result = selfhash({
        "schema": SCHEMA, "authority": AUTHORITY, "source": {"repository": REPOSITORY, "branch": BRANCH},
        "frozen_non_size_recipe": ref, "scale_specs": SCALE_SPECS,
        "common_optimizer_steps": list(CHECKPOINT_STEPS), "common_optimized_token_budgets": EXPECTED_TOKEN_BUDGETS,
        "hidden_token_advantage": False, "arm_matrix": [{"scale": s, "seed": z} for s, z in ARM_MATRIX],
        "learn191_geometry_authority": LEARN191_GEOMETRY, "m150_incumbent_anchor": M150_PRODUCER,
        "checkpoints": rows_by_step, "paired_1m_3m": paired, "pairwise_efficiency": efficiency,
        "definitions": {"compute_proxy": "6*N*T using actual optimized targets only",
                        "training_bpb": "optimizer-step minibatch NLL / ln(2)",
                        "generalization_gap_bpb": "heldout BPB - checkpoint training BPB",
                        "parameter_memory": "FP32 trainable parameter bytes; peak RSS reported separately",
                        "throughput": "actual optimized tokens divided by cumulative end-to-end train/eval/checkpoint wall time"},
        "truth_boundary": {"local_free_only": True, "paid_compute": False, "foreign_pretrained_weights": False,
                           "sft": False, "rlhf": False, "dpo": False, "stage_promotion": False,
                           "representative_external_corpus_claim": False, "universal_scaling_law_claim": False},
    })
    out.mkdir(parents=True, exist_ok=True)
    writej(out / "research192-scaling-comparison.json", result)
    return result


def validate_static_contract():
    for scale in SCALE_SPECS:
        spec = spec_for(scale)
        if spec.n_heads != spec.n_kv_heads or spec.max_seq_len != 256 or spec.head_dim != 16:
            raise Research192Error(f"{scale} left fixed MHA/context/head family")
    if SCALE_SPECS["3m"]["expected_model_spec_sha256"] != LEARN191_GEOMETRY["model_spec_sha256"]:
        raise Research192Error("3M geometry diverged from LEARN-191")
    if EXPECTED_TOKEN_BUDGETS != {18: 17_125, 70: 66_417, 139: 131_938}:
        raise Research192Error("optimizer-boundary token budgets drifted")
    if EXPECTED_TOKEN_BUDGETS[FINAL_STEP] / 20_000_775 >= 0.01:
        raise Research192Error("source-exposure ceiling invalid")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "phase1", "resume", "verify", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("--repo", type=Path, default=Path("."))
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--scale", choices=sorted(SCALE_SPECS), required=True)
        q.add_argument("--seed", type=int, required=True)
        if name != "summarize":
            q.add_argument("--source-sha", required=True)
    q = sub.add_parser("compare")
    q.add_argument("--arms-root", type=Path, required=True)
    q.add_argument("--m150-root", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    sub.add_parser("validate-static")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "validate-static":
        validate_static_contract()
        return 0
    if a.cmd == "compare":
        compare(a.arms_root, a.m150_root, a.out)
        return 0
    if a.cmd == "prepare":
        prepare(a.repo, a.source_sha, a.out, a.scale, a.seed)
    elif a.cmd == "phase1":
        phase1(a.repo, a.source_sha, a.out, a.scale, a.seed)
    elif a.cmd == "resume":
        resume(a.repo, a.source_sha, a.out, a.scale, a.seed)
    elif a.cmd == "verify":
        verify(a.repo, a.source_sha, a.out, a.scale, a.seed)
    elif a.cmd == "summarize":
        summarize(a.out, a.scale, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
