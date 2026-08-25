"""MILESTONE-100 first defensible learned 12-6 Base experiment.

This is an orchestration layer over accepted incumbents. It does not own model,
tokenizer training, packing, Trainer, observability, checkpoint, or generation
semantics. The three phases are intentionally separate processes in CI so resume
is exercised from a fresh Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    environment_snapshot,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.data.corpus_v02 import build_corpus
from twelve_six.evaluation import perplexity_from_nll, relative_loss_improvement
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.generation import generate
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, count_trainable_parameters
from twelve_six.packing import (
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    DeterministicMixtureSampler,
    TextRecord,
    batch_examples,
    collate_rows,
    iter_packed_examples,
)
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    HFTokenizerAdapter,
    TokenizerArtifactIdentity,
    TokenizerTrainingManifest,
    ordered_vocab_sha256,
)
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.loss import causal_pair_loss
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.milestone100-first-learned-base.v1"
MODEL_NAME = "MILESTONE100_BPE512_107856P"
SEED = 1337
SEQUENCE_LENGTH = 128
BATCH_SIZE = 8
FINAL_STEPS = 1536
RESUME_STEP = 768
CHECKPOINT_EVERY = 256
EVAL_DOCS_PER_STRATUM = 64
EVAL_BATCH_SIZE = 8
MIXTURE_WEIGHTS = {"uk": 45.0, "en": 35.0, "code": 20.0}
GENERATION_PROMPTS = ("Україна ", "The ", "def ")
GENERATION_NEW_TOKENS = 32
ORIGIN_REAL = "REAL_EXTERNAL"
ORIGIN_PROJECT = "PROJECT_AUTHORED"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(dict(value)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha() -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40:
        raise RuntimeError("exact Git source SHA unavailable")
    return value


def _model_spec() -> ModelSpec:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=512,
        max_seq_len=256,
        d_model=48,
        n_layers=3,
        n_heads=4,
        n_kv_heads=4,
        head_dim=12,
        d_ff=128,
        rope_rotary_dim=12,
    )
    if spec.parameter_count() != 107_856:
        raise RuntimeError(f"MILESTONE-100 parameter identity drift: {spec.parameter_count()}")
    return spec


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=FINAL_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _runtime_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "safetensors", "tokenizers", "datatrove", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    core = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": packages,
    }
    return {**core, "runtime_identity_sha256": hash_json(core)}


def _load_tokenizer(corpus_dir: Path) -> HFTokenizerAdapter:
    from tokenizers import Tokenizer

    artifact = _read_json(corpus_dir / "tokenizer" / "artifact.json")
    raw = artifact["training_manifest"]
    manifest = TokenizerTrainingManifest(
        experiment_id=str(raw["experiment_id"]),
        algorithm=str(raw["algorithm"]),
        tokenizers_version=str(raw["library_version"]),
        dataset_id=str(raw["dataset_id"]),
        dataset_manifest_sha256=str(raw["dataset_manifest_sha256"]),
        corpus_files=tuple(
            CorpusFileIdentity(
                path=str(item["path"]),
                sha256=str(item["sha256"]),
                byte_count=int(item["byte_count"]),
            )
            for item in raw["corpus_files"]
        ),
        vocab_size=int(raw["vocab_size"]),
        min_frequency=int(raw["min_frequency"]),
        normalization=str(raw["normalization"]),
        pre_tokenizer=str(raw["pre_tokenizer"]),
        decoder=str(raw["decoder"]),
        special_tokens=tuple(str(item) for item in raw["special_tokens"]),
    )
    identity = TokenizerArtifactIdentity(
        algorithm=str(artifact["algorithm"]),
        tokenizers_version=str(artifact["tokenizers_version"]),
        training_manifest_sha256=str(artifact["training_manifest_sha256"]),
        tokenizer_json_sha256=str(artifact["tokenizer_json_sha256"]),
        vocab_sha256=str(artifact["vocab_sha256"]),
        vocab_size=int(artifact["vocab_size"]),
        special_tokens=tuple(
            (str(item[0]), int(item[1])) for item in artifact["special_tokens"]
        ),
    )
    if identity.config_sha256 != artifact["config_sha256"]:
        raise RuntimeError("persisted BPE config identity mismatch")
    runtime = Tokenizer.from_file(str(corpus_dir / "tokenizer" / "tokenizer.json"))
    if _sha256_bytes(runtime.to_str().encode("utf-8")) != identity.tokenizer_json_sha256:
        raise RuntimeError("persisted BPE tokenizer JSON semantic hash mismatch")
    if ordered_vocab_sha256(runtime) != identity.vocab_sha256:
        raise RuntimeError("persisted BPE vocabulary identity mismatch")
    tokenizer = HFTokenizerAdapter(runtime, manifest, identity)
    if tokenizer.vocab_size != 512:
        raise RuntimeError("MILESTONE-100 requires BPE vocabulary size 512")
    return tokenizer


def _iter_shard_rows(corpus_dir: Path) -> Iterator[dict[str, Any]]:
    manifest = _read_json(corpus_dir / "manifest.json")
    for shard in manifest["shards"]:
        path = corpus_dir / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"corpus shard hash mismatch: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError("corpus shard row must be object")
                yield row


def _origin_map(corpus_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _iter_shard_rows(corpus_dir):
        record_id = str(row["record_id"])
        origin = str(row["origin_class"])
        if record_id in mapping:
            raise RuntimeError(f"duplicate final corpus record id: {record_id}")
        mapping[record_id] = origin
    return mapping


def _records(
    corpus_dir: Path,
    *,
    split: str,
    stratum: str,
    origin: str | None = None,
) -> Iterator[TextRecord]:
    for row in _iter_shard_rows(corpus_dir):
        if row["split"] != split or row["stratum"] != stratum:
            continue
        if origin is not None and row["origin_class"] != origin:
            continue
        yield TextRecord(str(row["record_id"]), str(row["text"]), split)


def _stratum_examples(
    corpus_dir: Path,
    tokenizer: HFTokenizerAdapter,
    stratum: str,
) -> Iterator[Any]:
    while True:
        emitted = 0
        for origin in (ORIGIN_REAL, ORIGIN_PROJECT):
            stream = iter_packed_examples(
                _records(corpus_dir, split="train", stratum=stratum, origin=origin),
                tokenizer,
                expected_split="train",
                sequence_length=SEQUENCE_LENGTH,
                fill_token_id=tokenizer.unk_id,
                ignore_index=-100,
                add_bos=False,
                add_eos=False,
                cross_document=False,
            )
            for example in stream:
                emitted += 1
                yield example
        if emitted == 0:
            raise RuntimeError(f"train stratum has no packed examples: {stratum}")


def _mixture_examples(
    corpus_dir: Path,
    tokenizer: HFTokenizerAdapter,
    *,
    start_example: int,
) -> Iterator[Any]:
    if start_example < 0:
        raise ValueError("start_example must be non-negative")
    sampler = DeterministicMixtureSampler(MIXTURE_WEIGHTS, seed=SEED)
    streams = {name: _stratum_examples(corpus_dir, tokenizer, name) for name in MIXTURE_WEIGHTS}
    consumed = Counter(sampler.source_for_step(index) for index in range(start_example))
    for name, count in consumed.items():
        deque = streams[name]
        for _ in range(count):
            next(deque)
    for index in itertools.count(start_example):
        name = sampler.source_for_step(index)
        yield next(streams[name])


def _batch_iterator(
    corpus_dir: Path,
    tokenizer: HFTokenizerAdapter,
    *,
    start_step: int,
) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    examples = _mixture_examples(
        corpus_dir,
        tokenizer,
        start_example=start_step * BATCH_SIZE,
    )
    origins = _origin_map(corpus_dir)
    for group in batch_examples(examples, batch_size=BATCH_SIZE, drop_last=False):
        rows = collate_rows(group, target_mode="target_ids")
        batch = {key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()}
        origin_tokens: Counter[str] = Counter()
        record_ids: list[str] = []
        for example in group:
            if len(example.record_ids) != 1:
                raise RuntimeError("isolated-document packing unexpectedly mixed record ids")
            record_id = example.record_ids[0]
            record_ids.append(record_id)
            origin_tokens[origins[record_id]] += int(example.num_loss_tokens)
        metadata = {
            "record_ids": record_ids,
            "origin_tokens": dict(origin_tokens),
            "batch_identity_sha256": hash_json(
                {
                    "record_ids": record_ids,
                    "loss_tokens": [int(example.num_loss_tokens) for example in group],
                }
            ),
        }
        yield batch, metadata


def _peek_batch_identity(
    corpus_dir: Path,
    tokenizer: HFTokenizerAdapter,
    *,
    start_step: int,
) -> str:
    _batch, metadata = next(_batch_iterator(corpus_dir, tokenizer, start_step=start_step))
    return str(metadata["batch_identity_sha256"])


def _select_evaluation_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for stratum in MIXTURE_WEIGHTS:
        candidates = [
            row
            for row in _iter_shard_rows(corpus_dir)
            if row["split"] == "validation" and row["stratum"] == stratum
        ]
        candidates.sort(
            key=lambda row: (
                0 if row["origin_class"] == ORIGIN_REAL else 1,
                str(row["record_id"]),
            )
        )
        take = candidates[:EVAL_DOCS_PER_STRATUM]
        if not take:
            raise RuntimeError(f"held-out validation stratum is empty: {stratum}")
        selected.extend(take)
    return selected


def _evaluate(
    model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter,
    corpus_dir: Path,
) -> dict[str, Any]:
    selected = _select_evaluation_rows(corpus_dir)
    records = [
        TextRecord(str(row["record_id"]), str(row["text"]), "validation") for row in selected
    ]
    examples = iter_packed_examples(
        records,
        tokenizer,
        expected_split="validation",
        sequence_length=SEQUENCE_LENGTH,
        fill_token_id=tokenizer.unk_id,
        ignore_index=-100,
        add_bos=False,
        add_eos=False,
        cross_document=False,
    )
    before_hash = _state_hash(model)
    was_training = model.training
    total_nll = 0.0
    total_loss_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for group in batch_examples(examples, batch_size=EVAL_BATCH_SIZE, drop_last=False):
                rows = collate_rows(group, target_mode="target_ids")
                input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
                target_ids = torch.tensor(rows["target_ids"], dtype=torch.long)
                loss_mask = torch.tensor(rows["loss_mask"], dtype=torch.long)
                logits = model(input_ids).logits
                loss = causal_pair_loss(logits, target_ids, loss_mask=loss_mask)
                tokens = int(loss_mask.bool().sum().item())
                total_nll += float(loss.item()) * tokens
                total_loss_tokens += tokens
    finally:
        model.train(was_training)
    after_hash = _state_hash(model)
    if before_hash != after_hash:
        raise RuntimeError("held-out evaluation mutated model state")
    if total_loss_tokens <= 0:
        raise RuntimeError("held-out evaluation produced no loss tokens")
    total_bytes = sum(int(row["utf8_bytes"]) for row in selected)
    mean_token_nll = total_nll / total_loss_tokens
    bpb = total_nll / (total_bytes * math.log(2.0))
    identity = hash_json(
        [
            {
                "record_id": row["record_id"],
                "content_sha256": row["content_sha256"],
                "stratum": row["stratum"],
            }
            for row in selected
        ]
    )
    return {
        "evaluation_identity_sha256": identity,
        "documents": len(selected),
        "documents_by_stratum": dict(Counter(str(row["stratum"]) for row in selected)),
        "documents_by_origin": dict(Counter(str(row["origin_class"]) for row in selected)),
        "utf8_bytes": total_bytes,
        "loss_tokens": total_loss_tokens,
        "total_nll_nats": total_nll,
        "mean_token_nll": mean_token_nll,
        "token_perplexity": perplexity_from_nll(mean_token_nll),
        "bpb": bpb,
        "bpb_definition": "total next-token NLL bits divided by exact UTF-8 bytes of fixed held-out documents",
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "non_mutating": before_hash == after_hash,
    }


class _BPEFirstPartyBackend:
    """BPE decode shim around the incumbent first-party Torch logits backend."""

    eos_token_id = None

    def __init__(self, model: TwelveSixDecoder, tokenizer: HFTokenizerAdapter) -> None:
        self._delegate = S0TorchInferenceBackend(model, tokenizer)  # type: ignore[arg-type]
        self._tokenizer = tokenizer
        self.max_context_tokens = model.spec.max_seq_len

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True, errors="strict")

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self._delegate.next_token_logits(input_ids)


def _generation(model: TwelveSixDecoder, tokenizer: HFTokenizerAdapter) -> dict[str, Any]:
    backend = _BPEFirstPartyBackend(model, tokenizer)
    outputs = []
    for prompt in GENERATION_PROMPTS:
        result = generate(
            backend,
            prompt,
            GenerationConfig(max_new_tokens=GENERATION_NEW_TOKENS, sample=False, seed=0),
        )
        outputs.append(
            {
                "prompt": prompt,
                "prompt_token_ids": list(result.prompt_token_ids),
                "generated_token_ids": list(result.generated_token_ids),
                "text": result.text,
                "stop_reason": result.stop_reason,
            }
        )
    return {
        "backend": "incumbent_S0TorchInferenceBackend_with_experimental_BPE_decode_shim",
        "greedy": True,
        "max_new_tokens": GENERATION_NEW_TOKENS,
        "outputs": outputs,
        "durable_first_party_loader_compatibility": (
            "NOT_PROMOTED: canonical load_first_party_backend remains byte-tokenizer-only"
        ),
    }


def _bundle(corpus_dir: Path) -> dict[str, Any]:
    source_sha = _source_sha()
    corpus_manifest = _read_json(corpus_dir / "manifest.json")
    tokenizer = _load_tokenizer(corpus_dir)
    spec = _model_spec()
    init_spec = InitSpec()
    config = _trainer_config()
    runtime = _runtime_identity()
    training_config = {
        "schema_version": SCHEMA,
        "model_name": MODEL_NAME,
        "init_spec_sha256": init_spec.identity_sha256(),
        "trainer": asdict(config),
        "training": {"context_length": spec.max_seq_len},
        "data": {
            "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"],
            "split_identity": corpus_manifest["split_identity_sha256"],
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
            "packing_compatibility_boundary": (
                "incumbent mechanics reused with BPE-512; incumbent packing identity string remains byte-specific"
            ),
            "tokenizer_version": tokenizer.identity.version,
            "mixture_weights": MIXTURE_WEIGHTS,
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
        },
        "resume_step": RESUME_STEP,
        "checkpoint_every_steps": CHECKPOINT_EVERY,
        "final_steps": FINAL_STEPS,
    }
    run_manifest = {
        "schema_version": SCHEMA,
        "source_sha": source_sha,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "dataset_manifest_file_sha256": sha256_file(corpus_dir / "manifest.json"),
        "training_config": training_config,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    return {
        "source_sha": source_sha,
        "corpus_manifest": corpus_manifest,
        "dataset_manifest_hash": sha256_file(corpus_dir / "manifest.json"),
        "tokenizer": tokenizer,
        "spec": spec,
        "init_spec": init_spec,
        "trainer_config": config,
        "training_config": training_config,
        "runtime": runtime,
        "run_manifest": run_manifest,
        "run_manifest_hash": hash_json(run_manifest),
    }


def _checkpoint_identity(bundle: Mapping[str, Any], trainer: Trainer) -> CheckpointIdentity:
    config: TrainerConfig = bundle["trainer_config"]
    tokenizer: HFTokenizerAdapter = bundle["tokenizer"]
    spec: ModelSpec = bundle["spec"]
    return CheckpointIdentity(
        git_sha=str(bundle["source_sha"]),
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=str(bundle["dataset_manifest_hash"]),
        run_manifest_hash=str(bundle["run_manifest_hash"]),
        training_config=bundle["training_config"],
        seed=SEED,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": list(config.betas),
            "eps": config.eps,
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        scheduler={"kind": config.scheduler, "warmup_steps": config.warmup_steps},
        environment_lock_hash=str(bundle["runtime"]["runtime_identity_sha256"]),
    )


def _save_checkpoint(
    output: Path,
    bundle: Mapping[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
    state: dict[str, Any],
) -> None:
    destination = output / "checkpoints" / f"step-{trainer.optimizer_step:06d}"
    if destination.exists():
        raise FileExistsError(destination)
    model_hash = _state_hash(model)
    manifest = save_trainer_checkpoint(
        destination,
        model=model,
        trainer=trainer,
        identity=_checkpoint_identity(bundle, trainer),
    )
    verify_checkpoint(destination)
    state["checkpoints"].append(
        {
            "step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "path": str(destination.relative_to(output)),
            "checkpoint_id": manifest["checkpoint_id"],
            "model_state_sha256": model_hash,
            "manifest_sha256": sha256_file(destination / "manifest.json"),
        }
    )


def _load_checkpoint(
    output: Path,
    bundle: Mapping[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
    record: Mapping[str, Any],
) -> None:
    tokenizer: HFTokenizerAdapter = bundle["tokenizer"]
    spec: ModelSpec = bundle["spec"]
    config: TrainerConfig = bundle["trainer_config"]
    path = output / str(record["path"])
    load_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=str(bundle["source_sha"]),
        expected_model_spec_hash=spec.identity_sha256(),
        expected_init_spec_hash=str(bundle["init_spec"].identity_sha256()),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=str(bundle["dataset_manifest_hash"]),
        expected_split_identity=str(bundle["corpus_manifest"]["split_identity_sha256"]),
        expected_packing_hash=PACKING_CONFIG_HASH,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=str(bundle["run_manifest_hash"]),
        expected_training_config_hash=hash_json(bundle["training_config"]),
        expected_environment_lock_hash=str(bundle["runtime"]["runtime_identity_sha256"]),
        expected_seed=config.seed,
    )
    if trainer.optimizer_step != int(record["step"]):
        raise RuntimeError("restored optimizer step mismatch")
    if trainer.tokens_seen != int(record["tokens_seen"]):
        raise RuntimeError("restored token counter mismatch")
    if _state_hash(model) != record["model_state_sha256"]:
        raise RuntimeError("restored model state hash mismatch")


def _new_model(bundle: Mapping[str, Any]) -> TwelveSixDecoder:
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    if count_trainable_parameters(model) != bundle["spec"].parameter_count():
        raise RuntimeError("constructed model parameter count mismatch")
    return model


def _train_phase(
    *,
    output: Path,
    corpus_dir: Path,
    bundle: Mapping[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
    state: dict[str, Any],
    stop_step: int,
    phase: str,
) -> dict[str, Any]:
    observer = TrainingObserver(
        {
            "schema_version": SCHEMA,
            "source_sha": bundle["source_sha"],
            "run_manifest_sha256": bundle["run_manifest_hash"],
            "phase": phase,
            "start_optimizer_step": trainer.optimizer_step,
            "stop_optimizer_step": stop_step,
        },
        device="cpu",
        max_step_samples=2048,
    )
    stream = _batch_iterator(corpus_dir, bundle["tokenizer"], start_step=trainer.optimizer_step)
    iterator = iter(stream)
    while trainer.optimizer_step < stop_step:
        (batch, metadata), wait_seconds = observer.measure_next(iterator)
        metrics = observer.train_microbatch(
            trainer,
            batch,
            data_wait_seconds=wait_seconds,
        )
        if not metrics.optimizer_stepped or metrics.update_loss is None:
            raise RuntimeError("MILESTONE-100 expects one optimizer update per microbatch")
        state["loss_trace"].append(
            {
                "step": metrics.optimizer_step,
                "loss": metrics.loss,
                "update_loss": metrics.update_loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": metrics.grad_norm,
                "tokens": metrics.tokens,
            }
        )
        for origin, tokens in metadata["origin_tokens"].items():
            state["optimized_tokens_by_origin"][origin] = (
                int(state["optimized_tokens_by_origin"].get(origin, 0)) + int(tokens)
            )
        if (
            metrics.optimizer_step % CHECKPOINT_EVERY == 0
            or metrics.optimizer_step == stop_step
        ):
            observer.measure_region(
                "checkpoint",
                f"save-step-{metrics.optimizer_step}",
                lambda: _save_checkpoint(output, bundle, model, trainer, state),
                optimizer_step=trainer.optimizer_step,
                tokens_seen=trainer.tokens_seen,
                synchronize_cuda=False,
            )
            _write_json(output / "state.json", state)
    summary = observer.summary()
    _write_json(output / f"observability-{phase}.json", summary)
    return summary


def prepare(corpus_config: Path, corpus_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    if not (corpus_dir / "manifest.json").exists():
        build_corpus(corpus_config, corpus_dir)
    bundle = _bundle(corpus_dir)
    model = _new_model(bundle)
    initial_hash = _state_hash(model)
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    baseline_eval = _evaluate(model, bundle["tokenizer"], corpus_dir)
    baseline_generation = _generation(model, bundle["tokenizer"])
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source_sha": bundle["source_sha"],
        "run_manifest_sha256": bundle["run_manifest_hash"],
        "dataset_manifest_hash": bundle["dataset_manifest_hash"],
        "corpus_identity_sha256": bundle["corpus_manifest"]["corpus_identity_sha256"],
        "model_spec": bundle["spec"].to_dict(),
        "model_spec_sha256": bundle["spec"].identity_sha256(),
        "parameter_count": bundle["spec"].parameter_count(),
        "init_spec": bundle["init_spec"].to_dict(),
        "init_spec_sha256": bundle["init_spec"].identity_sha256(),
        "initialization": {
            "kind": "RANDOM_FROM_SCRATCH",
            "seed": SEED,
            "model_state_sha256": initial_hash,
            "checkpoint_loaded_before_baseline": False,
            "foreign_pretrained_weights": False,
        },
        "tokenizer": bundle["tokenizer"].identity.to_dict(),
        "packing": {
            "version": PACKING_VERSION,
            "sha256": PACKING_CONFIG_HASH,
            "compatibility_boundary": bundle["training_config"]["data"][
                "packing_compatibility_boundary"
            ],
        },
        "trainer_config": asdict(bundle["trainer_config"]),
        "runtime": bundle["runtime"],
        "baseline_evaluation": baseline_eval,
        "baseline_generation": baseline_generation,
        "loss_trace": [],
        "optimized_tokens_by_origin": {ORIGIN_REAL: 0, ORIGIN_PROJECT: 0},
        "checkpoints": [],
        "prepare_process_uuid": str(uuid.uuid4()),
        "resume_process_uuid": None,
        "verify_process_uuid": None,
        "fresh_process_resume": False,
        "resume_data_cursor_verified": False,
        "resume_checkpoint_verified": False,
    }
    _write_json(output / "run-manifest.json", bundle["run_manifest"])
    _write_json(output / "state.json", state)
    _train_phase(
        output=output,
        corpus_dir=corpus_dir,
        bundle=bundle,
        model=model,
        trainer=trainer,
        state=state,
        stop_step=RESUME_STEP,
        phase="prepare",
    )
    if trainer.optimizer_step != RESUME_STEP:
        raise RuntimeError("prepare phase did not reach resume boundary")
    state["expected_resume_batch_identity_sha256"] = _peek_batch_identity(
        corpus_dir,
        bundle["tokenizer"],
        start_step=RESUME_STEP,
    )
    _write_json(output / "state.json", state)


def resume(corpus_dir: Path, output: Path) -> None:
    state = _read_json(output / "state.json")
    bundle = _bundle(corpus_dir)
    if bundle["source_sha"] != state["source_sha"]:
        raise RuntimeError("resume source SHA drift")
    current_uuid = str(uuid.uuid4())
    if current_uuid == state["prepare_process_uuid"]:
        raise RuntimeError("fresh process proof unexpectedly collided")
    model = _new_model(bundle)
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    record = next(item for item in state["checkpoints"] if int(item["step"]) == RESUME_STEP)
    _load_checkpoint(output, bundle, model, trainer, record)
    state["resume_checkpoint_verified"] = True
    next_identity = _peek_batch_identity(
        corpus_dir,
        bundle["tokenizer"],
        start_step=trainer.optimizer_step,
    )
    if next_identity != state["expected_resume_batch_identity_sha256"]:
        raise RuntimeError("fresh-process resume data cursor identity mismatch")
    state["resume_data_cursor_verified"] = True
    state["resume_process_uuid"] = current_uuid
    state["fresh_process_resume"] = True
    _write_json(output / "state.json", state)
    _train_phase(
        output=output,
        corpus_dir=corpus_dir,
        bundle=bundle,
        model=model,
        trainer=trainer,
        state=state,
        stop_step=FINAL_STEPS,
        phase="resume",
    )
    if trainer.optimizer_step != FINAL_STEPS:
        raise RuntimeError("resume phase did not reach final step")
    _write_json(output / "state.json", state)


def _checkpoint_file_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def verify(corpus_dir: Path, output: Path) -> None:
    state = _read_json(output / "state.json")
    bundle = _bundle(corpus_dir)
    process_uuid = str(uuid.uuid4())
    if process_uuid in {state["prepare_process_uuid"], state["resume_process_uuid"]}:
        raise RuntimeError("verify phase must be a fresh process")
    final_record = next(
        item for item in state["checkpoints"] if int(item["step"]) == FINAL_STEPS
    )
    model = _new_model(bundle)
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    _load_checkpoint(output, bundle, model, trainer, final_record)
    final_eval = _evaluate(model, bundle["tokenizer"], corpus_dir)
    final_generation = _generation(model, bundle["tokenizer"])
    state["verify_process_uuid"] = process_uuid
    state["final_evaluation"] = final_eval
    state["final_generation"] = final_generation

    losses = [float(item["update_loss"]) for item in state["loss_trace"]]
    if len(losses) != FINAL_STEPS:
        raise RuntimeError(f"expected {FINAL_STEPS} optimizer losses, got {len(losses)}")
    train_before = statistics.fmean(losses[:64])
    train_after = statistics.fmean(losses[-64:])
    baseline_bpb = float(state["baseline_evaluation"]["bpb"])
    final_bpb = float(final_eval["bpb"])
    if not train_after < train_before:
        raise RuntimeError(f"train loss did not decrease: {train_before} -> {train_after}")
    if not final_bpb < baseline_bpb:
        raise RuntimeError(f"held-out BPB did not decrease: {baseline_bpb} -> {final_bpb}")
    if state["baseline_evaluation"]["evaluation_identity_sha256"] != final_eval[
        "evaluation_identity_sha256"
    ]:
        raise RuntimeError("held-out evaluation identity changed between before/after")
    if not state["fresh_process_resume"] or not state["resume_checkpoint_verified"]:
        raise RuntimeError("fresh-process resume proof is incomplete")
    if not state["resume_data_cursor_verified"]:
        raise RuntimeError("resume data cursor was not verified")
    if len(state["checkpoints"]) < 6:
        raise RuntimeError("multiple retained checkpoint requirement not met")
    if int(state["optimized_tokens_by_origin"].get(ORIGIN_REAL, 0)) <= 0:
        raise RuntimeError("actual optimized-token trace never consumed real external data")

    final_checkpoint = output / str(final_record["path"])
    checkpoint_hashes = _checkpoint_file_hashes(final_checkpoint)
    generation_changed = any(
        before["generated_token_ids"] != after["generated_token_ids"]
        for before, after in zip(
            state["baseline_generation"]["outputs"],
            final_generation["outputs"],
            strict=True,
        )
    )
    machine = {
        **environment_snapshot(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "github_actions": bool(os.environ.get("GITHUB_ACTIONS")),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "execution_class": "LOCAL_FREE_GITHUB_HOSTED_PUBLIC_REPO",
        "paid_compute": False,
    }
    evidence_core = {
        "schema_version": SCHEMA,
        "status": "PASS_LEARNED_BASE_ARTIFACT_WITH_EXPLICIT_CORPUS_LIMITATION",
        "source_sha": state["source_sha"],
        "branch": "milestone100/data101-first-learned-base-20260826",
        "model": {
            "name": MODEL_NAME,
            "model_spec_sha256": state["model_spec_sha256"],
            "parameter_count": state["parameter_count"],
            "random_initialization": state["initialization"],
        },
        "corpus": {
            "corpus_identity_sha256": state["corpus_identity_sha256"],
            "dataset_manifest_hash": state["dataset_manifest_hash"],
            "real_external_share_of_one_pass_supply": bundle["corpus_manifest"][
                "optimized_token_supply"
            ]["real_external_share"],
            "real_external_tokens_actually_optimized": state["optimized_tokens_by_origin"][
                ORIGIN_REAL
            ],
            "project_authored_tokens_actually_optimized": state[
                "optimized_tokens_by_origin"
            ][ORIGIN_PROJECT],
            "broad_representativeness_claimed": False,
            "real_external_code_available": bundle["corpus_manifest"]["truth_boundary"][
                "real_external_code_available"
            ],
            "limitation": bundle["corpus_manifest"]["truth_boundary"][
                "real_source_pool_limitation"
            ],
        },
        "tokenizer": state["tokenizer"],
        "packing": state["packing"],
        "training": {
            "optimizer": "AdamW",
            "trainer_config": state["trainer_config"],
            "optimizer_steps": FINAL_STEPS,
            "tokens_seen": trainer.tokens_seen,
            "train_loss_first_64_mean": train_before,
            "train_loss_last_64_mean": train_after,
            "train_loss_relative_improvement": relative_loss_improvement(
                train_before, train_after
            ),
        },
        "heldout_evaluation": {
            "before": state["baseline_evaluation"],
            "after": final_eval,
            "bpb_decreased": True,
            "bpb_delta": final_bpb - baseline_bpb,
            "evaluation_non_mutation": final_eval["non_mutating"]
            and state["baseline_evaluation"]["non_mutating"],
        },
        "checkpoint_resume": {
            "checkpoint_count": len(state["checkpoints"]),
            "checkpoints": state["checkpoints"],
            "fresh_process_resume": state["fresh_process_resume"],
            "resume_checkpoint_verified": state["resume_checkpoint_verified"],
            "resume_data_cursor_verified": state["resume_data_cursor_verified"],
            "retained_final_checkpoint": str(final_record["path"]),
            "retained_final_checkpoint_id": final_record["checkpoint_id"],
            "retained_final_checkpoint_file_hashes": checkpoint_hashes,
        },
        "generation": {
            "before": state["baseline_generation"],
            "after": final_generation,
            "greedy_sequence_changed": generation_changed,
        },
        "observability": {
            "prepare": _read_json(output / "observability-prepare.json"),
            "resume": _read_json(output / "observability-resume.json"),
        },
        "machine_manifest": machine,
        "reproduction": {
            "exact_source_sha": state["source_sha"],
            "commands": [
                "PYTHONPATH=src python -m twelve_six.data.corpus_v02 configs/data/corpus_v02.json --output-dir data/build/milestone100-corpus",
                "PYTHONPATH=src python tools/run_milestone100_first_learned_base.py prepare --corpus-config configs/data/corpus_v02.json --corpus-dir data/build/milestone100-corpus --output reports/milestone100/run",
                "PYTHONPATH=src python tools/run_milestone100_first_learned_base.py resume --corpus-dir data/build/milestone100-corpus --output reports/milestone100/run",
                "PYTHONPATH=src python tools/run_milestone100_first_learned_base.py verify --corpus-dir data/build/milestone100-corpus --output reports/milestone100/run",
            ],
        },
        "truth_boundary": {
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
            "broad_intelligence_claim": False,
            "stage_promotion": "NOT_PERFORMED",
            "durable_BPE_first_party_loader_promoted": False,
            "learned_base_artifact_claim": True,
        },
    }
    evidence = {**evidence_core, "evidence_sha256": hash_json(evidence_core)}
    state["final_evidence_sha256"] = evidence["evidence_sha256"]
    _write_json(output / "state.json", state)
    _write_json(output / "evidence.json", evidence)
    _write_json(output / "machine-manifest.json", machine)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--corpus-config", type=Path, required=True)
    prepare_parser.add_argument("--corpus-dir", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--corpus-dir", type=Path, required=True)
    resume_parser.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--corpus-dir", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.corpus_config, args.corpus_dir, args.output)
    elif args.command == "resume":
        resume(args.corpus_dir, args.output)
    else:
        verify(args.corpus_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
