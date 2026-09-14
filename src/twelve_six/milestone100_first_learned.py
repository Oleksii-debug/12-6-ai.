"""MILESTONE-100 first learned ~100K 12-6 Base convergence run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.data.corpus_v01 import verify_rebuild
from twelve_six.evaluation import perplexity_from_nll, relative_loss_improvement
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_VERSION, TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.milestone100-first-learned-base.v1"
AUTHORITY = "LOCAL_FREE_LEARNED_BASE_EXPERIMENT_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
CORPUS_CONFIG = Path("configs/data/corpus_v01.json")
RETAINED_CORPUS_MANIFEST = Path("data/corpus/v0.1/manifest.json")
STAGE_CONFIG = Path("configs/stages/s1_100k.json")
RUNTIME_LOCK = Path("requirements/locks/linux-x86_64/runtime.lock.txt")
TOOLCHAIN_LOCK = Path("requirements/locks/linux-x86_64/toolchain.lock.txt")
SEQ = 128
BATCH = 8
MAX_STEPS = 1000
RESUME_STEP = 500
CHECKPOINT_STEPS = {0, 250, 500, 750, 1000}
SEED = 1337
LR = 3e-4
MIXTURE = (
    "uk", "en", "uk", "code", "en",
    "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk",
    "en", "uk", "code", "en", "uk",
)
PROMPTS = {"uk": "Українська мова ", "en": "The training corpus ", "code": "def stable_"}


class MilestoneError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MilestoneError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _require_head(repo: Path, source_sha: str) -> None:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise MilestoneError("source_sha must be lowercase full 40-hex")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise MilestoneError(f"exact-head mismatch: {actual} != {source_sha}")


def _locks(repo: Path) -> dict[str, Any]:
    files = {p.as_posix(): sha256_file(repo / p) for p in (RUNTIME_LOCK, TOOLCHAIN_LOCK)}
    return {"files": files, "combined_sha256": hash_json(files)}


def _model(repo: Path) -> tuple[ModelSpec, InitSpec, dict[str, Any]]:
    stage = load_stage_config(repo / STAGE_CONFIG)
    payload = stage.model.to_dict()
    if payload["vocab_size"] != 512:
        raise MilestoneError("canonical S1 vocab drifted")
    payload["vocab_size"] = 256
    spec = ModelSpec.from_dict(payload)
    if spec.parameter_count() != 95_568:
        raise MilestoneError(f"expected 95,568 parameters, got {spec.parameter_count()}")
    init = InitSpec()
    return spec, init, {
        "source_stage_config": STAGE_CONFIG.as_posix(),
        "source_model_spec_sha256": stage.model.identity_sha256(),
        "source_expected_parameters": stage.expected_parameters,
        "only_geometry_change": "vocab_size:512->256 to bind canonical s0-byte-v1",
    }


def _trainer_config() -> TrainerConfig:
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


def _build_corpus(repo: Path, out: Path) -> dict[str, Any]:
    retained = _read_json(repo / RETAINED_CORPUS_MANIFEST)
    built = verify_rebuild(repo / CORPUS_CONFIG, out / "corpus-a", out / "corpus-b")
    if built != retained:
        raise MilestoneError("DATA-25 rebuild differs from retained manifest")
    if built["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise MilestoneError("DATA-25 corpus identity drift")
    if built["train_validation_content_overlap"] != 0:
        raise MilestoneError("DATA-25 validation leakage")
    truth = built["truth_boundary"]
    if truth["contains_external_training_data"] is not False:
        raise MilestoneError("external corpus truth boundary changed; re-audit required")
    _write_json(out / "corpus-manifest.json", built)
    return built


def _rows(corpus: Path, manifest: dict[str, Any], split: str, stratum: str) -> Iterator[dict[str, Any]]:
    for shard in manifest["shards"]:
        path = corpus / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise MilestoneError(f"shard hash mismatch: {shard['path']}")
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("split") == split and row.get("stratum") == stratum:
                    yield row


def _packed(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, split: str, stratum: str):
    records = (
        TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
        for row in _rows(corpus, manifest, split, stratum)
    )
    yield from iter_packed_examples(
        records,
        tok,
        expected_split=split,
        sequence_length=SEQ,
        cross_document=False,
    )


def _steps_by_stratum(steps: int) -> dict[str, int]:
    result = {"uk": 0, "en": 0, "code": 0}
    for i in range(steps):
        result[MIXTURE[i % len(MIXTURE)]] += 1
    return result


def _train_iters(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer, completed_steps: int):
    result = {s: _packed(corpus, manifest, tok, "train", s) for s in ("uk", "en", "code")}
    for stratum, steps in _steps_by_stratum(completed_steps).items():
        for _ in range(steps * BATCH):
            try:
                next(result[stratum])
            except StopIteration as exc:
                raise MilestoneError(f"{stratum} exhausted while restoring data position") from exc
    return result


def _batches(iterator):
    while True:
        examples = []
        for _ in range(BATCH):
            try:
                examples.append(next(iterator))
            except StopIteration as exc:
                raise MilestoneError("training corpus exhausted before max_steps") from exc
        yield {
            "input_ids": torch.tensor([x.input_ids for x in examples], dtype=torch.long),
            "labels": torch.tensor([x.labels for x in examples], dtype=torch.long),
        }


def _state_hash(model: TwelveSixDecoder) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode() + b"\0")
        h.update(str(value.dtype).encode() + b"\0")
        h.update(str(tuple(value.shape)).encode() + b"\0")
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _eval_examples(model: TwelveSixDecoder, examples) -> tuple[float, int]:
    ids = torch.tensor([x.input_ids for x in examples], dtype=torch.long)
    labels = torch.tensor([x.labels for x in examples], dtype=torch.long)
    logits = model(ids).logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    tokens = int(targets.ne(-100).sum().item())
    nll = F.cross_entropy(
        logits.reshape(-1, model.spec.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(nll.item()), tokens


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer):
    before = _state_hash(model)
    training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by_stratum = {}
    try:
        for stratum in ("uk", "en", "code"):
            nll_sum = 0.0
            tokens = 0
            pending = []
            for example in _packed(corpus, manifest, tok, "validation", stratum):
                pending.append(example)
                if len(pending) == 32:
                    n, t = _eval_examples(model, pending)
                    nll_sum += n
                    tokens += t
                    pending = []
            if pending:
                n, t = _eval_examples(model, pending)
                nll_sum += n
                tokens += t
            if tokens <= 0:
                raise MilestoneError(f"no held-out target bytes for {stratum}")
            loss = nll_sum / tokens
            by_stratum[stratum] = {
                "loss": loss,
                "bits_per_byte": nll_sum / math.log(2.0) / tokens,
                "perplexity": perplexity_from_nll(loss),
                "predicted_byte_tokens": tokens,
            }
            total_nll += nll_sum
            total_tokens += tokens
    finally:
        model.train(training)
    after = _state_hash(model)
    if after != before:
        raise MilestoneError("evaluation mutated model state")
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "bits_per_byte": total_nll / math.log(2.0) / total_tokens,
        "perplexity": perplexity_from_nll(loss),
        "predicted_byte_tokens": total_tokens,
        "by_stratum": by_stratum,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
    }


def _run_manifest(source_sha, spec, init, tok, manifest, cfg, locks):
    value = {
        "schema": "12-6.milestone100-run-manifest.v1",
        "source_sha": source_sha,
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
        "packing": {"version": PACKING_VERSION, "sequence_length": SEQ, "cross_document": False},
        "trainer_config": asdict(cfg),
        "batch_size": BATCH,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": sorted(CHECKPOINT_STEPS),
        "mixture_pattern": list(MIXTURE),
        "environment_lock_sha256": locks["combined_sha256"],
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
        "paid_compute": False,
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _identity(source_sha, spec, tok, manifest, run, cfg, trainer, locks):
    training_config = {
        "trainer": asdict(cfg),
        "data": {
            "tokenizer_version": tok.identity.version,
            "packing_version": PACKING_VERSION,
            "packing_sequence_length": SEQ,
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
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


def _save(out, source_sha, spec, tok, manifest, run, cfg, trainer, locks):
    step = trainer.optimizer_step
    if step not in CHECKPOINT_STEPS:
        raise MilestoneError(f"unexpected checkpoint step {step}")
    path = out / f"checkpoint-{step:04d}"
    save_trainer_checkpoint(
        path,
        model=trainer.model,
        trainer=trainer,
        identity=_identity(source_sha, spec, tok, manifest, run, cfg, trainer, locks),
        overwrite=True,
    )
    checked = verify_checkpoint(path)
    return {"step": step, "tokens_seen": trainer.tokens_seen, "checkpoint_id": checked["checkpoint_id"]}


def _generation(checkpoint: Path):
    backend = load_first_party_backend(checkpoint)
    cfg = GenerationConfig(max_new_tokens=48, sample=False)
    outputs = {}
    for name, prompt in PROMPTS.items():
        result = generate(backend, prompt, cfg)
        outputs[name] = {
            "prompt": prompt,
            "generated_token_ids": list(result.generated_token_ids),
            "text": result.text,
            "stop_reason": result.stop_reason,
        }
    return {"backend_diagnostics": backend.diagnostics(), "decoding": "greedy", "outputs": outputs}


def _machine(source_sha, locks):
    return {
        "schema": "12-6.milestone100-machine-manifest.v1",
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
        "environment_locks": locks,
        "paid_compute": False,
    }


def _common(repo: Path, source_sha: str, out: Path, build: bool):
    _require_head(repo, source_sha)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = _build_corpus(repo, out) if build else _read_json(out / "corpus-manifest.json")
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise MilestoneError("persisted corpus identity mismatch")
    tok = ByteTokenizer()
    spec, init, geometry = _model(repo)
    cfg = _trainer_config()
    locks = _locks(repo)
    run = _run_manifest(source_sha, spec, init, tok, manifest, cfg, locks)
    if build:
        _write_json(out / "run-manifest.json", run)
        _write_json(out / "machine-manifest-phase1.json", _machine(source_sha, locks))
    else:
        if _read_json(out / "run-manifest.json") != run:
            raise MilestoneError("run manifest changed between processes")
        _write_json(out / "machine-manifest-resume.json", _machine(source_sha, locks))
    return manifest, tok, spec, init, geometry, cfg, locks, run


def phase1(repo: Path, source_sha: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, spec, init, geometry, cfg, locks, run = _common(repo, source_sha, out, True)
    corpus = out / "corpus-a"
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    runtime_params = sum(p.numel() for p in model.parameters())
    if runtime_params != spec.parameter_count():
        raise MilestoneError("runtime parameter count mismatch")
    random_hash = _state_hash(model)
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    initial = observer.measure_region("evaluation", "heldout-init", lambda: _evaluate(model, corpus, manifest, tok), optimizer_step=0, tokens_seen=0)
    events = [_save(out, source_sha, spec, tok, manifest, run, cfg, trainer, locks)]
    generation0 = _generation(out / "checkpoint-0000")
    its = _train_iters(corpus, manifest, tok, 0)
    batches = {s: _batches(it) for s, it in its.items()}
    curve = out / "train-curve.jsonl"
    if curve.exists():
        curve.unlink()
    for i in range(RESUME_STEP):
        stratum = MIXTURE[i % len(MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        _append(curve, {
            "optimizer_step": metrics.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "stratum": stratum,
            "tokens": metrics.tokens,
            "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
            "grad_norm": metrics.grad_norm,
            "learning_rate": metrics.learning_rate,
        })
        if metrics.optimizer_step in (250, 500):
            events.append(observer.measure_region(
                "checkpoint",
                f"save-{metrics.optimizer_step}",
                lambda: _save(out, source_sha, spec, tok, manifest, run, cfg, trainer, locks),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            ))
    if trainer.optimizer_step != RESUME_STEP:
        raise MilestoneError("phase1 did not stop at step 500")
    middle = observer.measure_region("evaluation", "heldout-step500", lambda: _evaluate(model, corpus, manifest, tok), optimizer_step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen)
    result = {
        "schema": "12-6.milestone100-phase1.v1",
        "source_sha": source_sha,
        "process": {"pid": os.getpid(), "python_executable": sys.executable},
        "model": {
            "spec": spec.to_dict(),
            "spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "runtime_parameter_count": runtime_params,
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "random_initialization": True,
            "random_init_state_sha256": random_hash,
            "geometry_provenance": geometry,
        },
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "experimental_bpe_status": "REJECTED_STALE_FOR_FINAL_VERTICAL",
            "experimental_bpe_reason": "TOK-37 was selected on DATA-10; current packing and first-party inference are byte-tokenizer-bound.",
        },
        "initial_heldout": initial,
        "step500_heldout": middle,
        "initial_generation": generation0,
        "checkpoints": events,
        "observer": observer.summary(),
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
    }
    result["identity_sha256"] = hash_json(result)
    _write_json(out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path):
    manifest, tok, spec, init, _, cfg, locks, run = _common(repo, source_sha, out, False)
    corpus = out / "corpus-a"
    p1 = _read_json(out / "phase1.json")
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    loaded = load_trainer_checkpoint(
        out / "checkpoint-0500",
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
        raise MilestoneError("resume run-manifest identity mismatch")
    if trainer.optimizer_step != RESUME_STEP:
        raise MilestoneError("checkpoint did not restore step 500")
    observer = TrainingObserver(run, device="cpu", max_step_samples=1024)
    its = _train_iters(corpus, manifest, tok, RESUME_STEP)
    batches = {s: _batches(it) for s, it in its.items()}
    curve_path = out / "train-curve.jsonl"
    events = []
    first_resumed = None
    for i in range(RESUME_STEP, MAX_STEPS):
        stratum = MIXTURE[i % len(MIXTURE)]
        batch, wait = observer.measure_next(batches[stratum])
        metrics = observer.train_microbatch(trainer, batch, data_wait_seconds=wait)
        first_resumed = first_resumed or metrics.optimizer_step
        _append(curve_path, {
            "optimizer_step": metrics.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "stratum": stratum,
            "tokens": metrics.tokens,
            "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
            "grad_norm": metrics.grad_norm,
            "learning_rate": metrics.learning_rate,
        })
        if metrics.optimizer_step in (750, 1000):
            events.append(observer.measure_region(
                "checkpoint",
                f"save-{metrics.optimizer_step}",
                lambda: _save(out, source_sha, spec, tok, manifest, run, cfg, trainer, locks),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
            ))
    if first_resumed != 501 or trainer.optimizer_step != MAX_STEPS:
        raise MilestoneError(f"fresh resume transition invalid: first={first_resumed} final={trainer.optimizer_step}")
    final_eval = observer.measure_region("evaluation", "heldout-final", lambda: _evaluate(model, corpus, manifest, tok), optimizer_step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen)
    final_generation = _generation(out / "checkpoint-1000")
    curve = [json.loads(x) for x in curve_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(curve) != MAX_STEPS:
        raise MilestoneError(f"expected 1000 train points, got {len(curve)}")
    first100 = sum(float(x["loss"]) for x in curve[:100]) / 100
    last100 = sum(float(x["loss"]) for x in curve[-100:]) / 100
    bpb0 = float(p1["initial_heldout"]["bits_per_byte"])
    bpb1 = float(final_eval["bits_per_byte"])
    rel = relative_loss_improvement(float(p1["initial_heldout"]["loss"]), float(final_eval["loss"]))
    if not last100 < first100:
        raise MilestoneError("train loss did not decrease")
    if not bpb1 < bpb0 or not rel > 0:
        raise MilestoneError("held-out BPB/loss did not improve")
    token_by_stratum = {s: sum(int(x["tokens"]) for x in curve if x["stratum"] == s) for s in ("uk", "en", "code")}
    checkpoint_manifests = {str(s): verify_checkpoint(out / f"checkpoint-{s:04d}") for s in sorted(CHECKPOINT_STEPS)}
    final_checkpoint = checkpoint_manifests["1000"]
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha, "branch": "milestone100/first-learned-base-20260826"},
        "runtime": {
            "phase1_machine": _read_json(out / "machine-manifest-phase1.json"),
            "resume_machine": _read_json(out / "machine-manifest-resume.json"),
            "fresh_process_resume": {
                "phase1_pid": p1["process"]["pid"],
                "resume_pid": os.getpid(),
                "separate_cli_invocations_required": True,
                "checkpoint_loaded_step": loaded.manifest["identity"]["step"],
                "first_resumed_optimizer_step": first_resumed,
                "passed": True,
            },
        },
        "truth_boundary": {
            "genuinely_optimized_from_random_initialization": True,
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
            "broad_intelligence_claim": False,
            "external_real_world_training_data_present": False,
            "external_source_diversity_representative": False,
            "required_real_representative_corpus_gate": "FAIL",
            "milestone_promotion_authority": False,
        },
        "data": {
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "corpus_version": manifest["corpus_version"],
            "train": manifest["by_split"]["train"],
            "validation": manifest["by_split"]["validation"],
            "train_validation_content_overlap": manifest["train_validation_content_overlap"],
            "external_training_eligible_sources": manifest["external_training_eligible_sources"],
            "truth_boundary": manifest["truth_boundary"],
            "rebuild_twice_exact": True,
        },
        "tokenizer": p1["tokenizer"],
        "packing": {"version": PACKING_VERSION, "sequence_length": SEQ, "cross_document": False, "incumbent_reused_without_edit": True},
        "model": p1["model"],
        "training": {
            "trainer_config": asdict(cfg),
            "optimizer": "AdamW",
            "max_steps": MAX_STEPS,
            "batch_size": BATCH,
            "optimized_tokens": trainer.tokens_seen,
            "optimized_tokens_by_stratum": token_by_stratum,
            "optimized_token_mixture_fraction": {s: token_by_stratum[s] / trainer.tokens_seen for s in token_by_stratum},
            "first100_mean_loss": first100,
            "last100_mean_loss": last100,
            "train_loss_decreased": True,
            "phase1_observability": p1["observer"],
            "resume_observability": observer.summary(),
            "train_curve_path": "train-curve.jsonl",
        },
        "evaluation": {
            "initial": p1["initial_heldout"],
            "step500": p1["step500_heldout"],
            "final": final_eval,
            "initial_bits_per_byte": bpb0,
            "final_bits_per_byte": bpb1,
            "heldout_bits_per_byte_decreased": True,
            "relative_heldout_loss_improvement": rel,
            "evaluation_non_mutation": (
                p1["initial_heldout"]["non_mutation_passed"]
                and p1["step500_heldout"]["non_mutation_passed"]
                and final_eval["non_mutation_passed"]
            ),
        },
        "generation": {"before_training": p1["initial_generation"], "after_training": final_generation, "first_party_runtime": True},
        "checkpoints": {
            "steps": sorted(CHECKPOINT_STEPS),
            "phase1_events": p1["checkpoints"],
            "resume_events": events,
            "fresh_process_resume_from": "checkpoint-0500",
            "retained_exact_checkpoint": "checkpoint-1000",
            "retained_checkpoint_id": final_checkpoint["checkpoint_id"],
            "retained_checkpoint_identity": final_checkpoint["identity"],
        },
        "component_selection": {
            "model": "current convergence ModelSpec/TwelveSixDecoder",
            "experimental_tokenizer": "TOK-37 BPE rejected as stale for DATA-25 vertical",
            "selected_tokenizer": "canonical s0-byte-v1",
            "corpus": "DATA-25 corpus V0.1 selective exact intake, rebuilt twice on exact head",
            "streaming_packing": "incumbent split-safe iter_packed_examples",
            "trainer_optimizer": "D02 Trainer + AdamW",
            "observability": "TRAIN-29 TrainingObserver",
            "checkpoint_resume": "D05 save/load_trainer_checkpoint",
            "heldout_evaluation": "D06 metric primitives + byte-exact held-out NLL/BPB",
            "first_party_inference": "D07 load_first_party_backend + generate",
        },
        "reproduction": {
            "phase1_command": "PYTHONPATH=src python -m twelve_six.milestone100_first_learned phase1 --repo-root . --source-sha \"$(git rev-parse HEAD)\" --output-dir milestone100-evidence",
            "resume_command": "PYTHONPATH=src python -m twelve_six.milestone100_first_learned resume --repo-root . --source-sha \"$(git rev-parse HEAD)\" --output-dir milestone100-evidence",
        },
        "success": {
            "genuinely_learned_base_artifact": True,
            "all_runtime_training_checkpoint_eval_generation_gates": True,
            "real_representative_corpus_gate": False,
            "overall_requested_milestone": "PARTIAL_FAIL_CLOSED",
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(out / "report.json", report)
    return report


def validate(path: Path, expected_source_sha: str | None = None):
    r = _read_json(path)
    supplied = r["report_sha256"]
    unsigned = dict(r)
    unsigned.pop("report_sha256")
    if supplied != hash_json(unsigned):
        raise MilestoneError("report self-hash mismatch")
    if r["schema"] != SCHEMA or r["authority"] != AUTHORITY:
        raise MilestoneError("report schema/authority mismatch")
    if expected_source_sha and r["source"]["git_sha"] != expected_source_sha:
        raise MilestoneError("report source mismatch")
    if r["model"]["parameter_count"] != 95_568:
        raise MilestoneError("parameter gate failed")
    if not r["training"]["train_loss_decreased"]:
        raise MilestoneError("train-loss gate failed")
    if not r["evaluation"]["heldout_bits_per_byte_decreased"]:
        raise MilestoneError("held-out BPB gate failed")
    if not r["evaluation"]["evaluation_non_mutation"]:
        raise MilestoneError("eval mutation gate failed")
    if not r["runtime"]["fresh_process_resume"]["passed"]:
        raise MilestoneError("resume gate failed")
    if r["truth_boundary"]["external_real_world_training_data_present"] is not False:
        raise MilestoneError("corpus truth boundary weakened")
    if r["success"]["overall_requested_milestone"] != "PARTIAL_FAIL_CLOSED":
        raise MilestoneError("full milestone must remain fail-closed on real corpus gate")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("phase1", "resume"):
        q = sub.add_parser(name)
        q.add_argument("--repo-root", type=Path, default=Path("."))
        q.add_argument("--source-sha", required=True)
        q.add_argument("--output-dir", type=Path, required=True)
    q = sub.add_parser("validate")
    q.add_argument("report", type=Path)
    q.add_argument("--expected-source-sha")
    a = p.parse_args(argv)
    if a.cmd == "phase1":
        r = phase1(a.repo_root.resolve(), a.source_sha, a.output_dir.resolve())
        print(json.dumps({"phase": "phase1", "step": r["optimizer_step"], "tokens": r["tokens_seen"], "initial_bpb": r["initial_heldout"]["bits_per_byte"], "step500_bpb": r["step500_heldout"]["bits_per_byte"]}, indent=2))
    elif a.cmd == "resume":
        r = resume(a.repo_root.resolve(), a.source_sha, a.output_dir.resolve())
        print(json.dumps({"phase": "resume", "parameters": r["model"]["parameter_count"], "tokens": r["training"]["optimized_tokens"], "initial_bpb": r["evaluation"]["initial_bits_per_byte"], "final_bpb": r["evaluation"]["final_bits_per_byte"], "checkpoint_id": r["checkpoints"]["retained_checkpoint_id"], "overall": r["success"]["overall_requested_milestone"]}, indent=2))
    else:
        validate(a.report, a.expected_source_sha)
        print("MILESTONE-100 report validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
