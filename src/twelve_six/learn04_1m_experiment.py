"""LEARN-04 controlled learned-Base experiment around one million parameters.

This experiment composes RESEARCH41 fixed geometry/optimizer controls with DATA-10's
repeatable project-authored UK/EN/code ByteLevel-BPE evidence.  It is intentionally
LOCAL_FREE and non-promoting.  External representative pretraining data is not yet
training-approved, so the corpus is recycled and every report carries that boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .packing.scale_contracts import MixturePlan, MixtureSource
from .tokenization.experiments import (
    CorpusFileIdentity,
    HFTokenizerAdapter,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.learn04-real-1m.v1"
AUTHORITY = "LOCAL_FREE_PROJECT_AUTHORED_BASE_EXPERIMENT_NOT_REPRESENTATIVE_CORPUS_OR_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
RESEARCH41_PARENT_SHA = "9ff78ea31c34fd434015d5bc512596ce5dac766a"
DATA10_PARENT_SHA = "077205ef2b1662a5029bc77b8fc762078cabeb17"
DATA10_RECIPE_PATH = "configs/data/multilingual_uk_en_code_v1.experimental.json"
DATA10_CORPUS_PATH = "data/synthetic/data10/uk-en-code-train.txt"
DATA10_CORPUS_SHA256 = "059f04e01d6fc6b8224b373b08efbb37f09d546de35ed510afdb4587ebdb6012"
TOKENIZERS_VERSION = "0.23.1"
TOKENIZER_EXPECTED_VOCAB = 472
TOKENIZER_REQUESTED_VOCAB = 512
TOKENIZER_ALGORITHM = "bpe"
TOKENIZER_MIN_FREQUENCY = 2
MIXTURE_WEIGHTS = {"uk": 45, "en": 35, "code": 20}
PACKING_VERSION = "learn04-deterministic-mixture-seq64-v1"
GENERATION_PROMPT = "Українська модель learns code: "
GENERATION_NEW_TOKENS = 24
DEFAULT_TOKEN_BUDGETS = (4_096, 16_384, 65_536, 262_144)
DEFAULT_RESUME_BUDGET = 65_536
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQUENCE_LENGTH = 64
DEFAULT_SEED = 1337
DEFAULT_CURVE_INTERVAL_STEPS = 8
_HEX = frozenset("0123456789abcdef")

TRAIN_RECORDS: tuple[tuple[str, str, str], ...] = (
    ("uk", "uk-1", "Українська мова має відмінки, дієвідмінювання і словотвір. Ці дані потрібні для базового передтренування моделі."),
    ("uk", "uk-2", "Дослідники працюють із текстами різних жанрів, щоб модель бачила слова у називному, родовому, давальному, знахідному та орудному відмінках."),
    ("uk", "uk-3", "Київ, Львів і Ужгород мають різні мовні контексти; ґрунтовний корпус повинен містити літери ґ, ї, є, і та природні апострофи."),
    ("en", "en-1", "The training corpus contains English prose with varied syntax and vocabulary so the base model learns next-token statistics rather than instructions."),
    ("en", "en-2", "These records test deterministic data selection, source provenance, deduplication, and restart behavior for a universal language model."),
    ("en", "en-3", "Data quality includes valid encoding, stable normalization, explicit source rights, and strict separation from held-out evaluation material."),
    ("code", "code-1", "def stable_hash(value: str) -> str:\n    return hashlib.sha256(value.encode('utf-8')).hexdigest()\n"),
    ("code", "code-2", "class Counter:\n    def __init__(self):\n        self.value = 0\n    def increment(self):\n        self.value += 1\n        return self.value\n"),
    ("code", "code-3", "SELECT source_id, COUNT(*) FROM records\nWHERE split = 'train'\nGROUP BY source_id ORDER BY source_id;\n"),
)

VALIDATION_RECORDS: tuple[tuple[str, str, str], ...] = (
    ("uk", "uk-cases", "книга книги книзі книгу книгою; учень учня учневі учнем"),
    ("uk", "uk-verbs", "працювати працюю працюєш працює працюємо працюють; прочитати прочитають"),
    ("uk", "uk-orthography", "п'ять, об'єкт, м'який, під'їзд, ґанок, їжак, Європа, Україна"),
    ("en", "en", "The multilingual base model compares token fertility on unseen English."),
    ("code", "code", "for index, item in enumerate(records):\n    assert item.split == 'train'\n"),
    ("multi", "unicode", "Україна — Kyiv — naïve café — λ = 3.14 — 😀"),
)

_GEOMETRIES: tuple[dict[str, int], ...] = (
    {"d_model": 48, "n_layers": 3, "n_heads": 4, "n_kv_heads": 4, "head_dim": 12, "d_ff": 128},
    {"d_model": 72, "n_layers": 4, "n_heads": 6, "n_kv_heads": 6, "head_dim": 12, "d_ff": 192},
    {"d_model": 96, "n_layers": 4, "n_heads": 6, "n_kv_heads": 6, "head_dim": 16, "d_ff": 256},
    {"d_model": 128, "n_layers": 5, "n_heads": 8, "n_kv_heads": 8, "head_dim": 16, "d_ff": 352},
)
_EXPECTED_COUNTS_VOCAB472 = (105_936, 283_464, 488_544, 1_065_344)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _require_source_sha(source_sha: str) -> None:
    if len(source_sha) != 40 or source_sha != source_sha.lower() or any(ch not in _HEX for ch in source_sha):
        raise ValueError("source_sha must be a lowercase 40-hex Git SHA")


def _joined_training_text() -> str:
    return "\n".join(text for _, _, text in TRAIN_RECORDS) + "\n"


def _validation_text() -> str:
    return "\n".join(text for _, _, text in VALIDATION_RECORDS) + "\n"


def _verify_data_boundary(repo_root: Path) -> dict[str, Any]:
    corpus_path = repo_root / DATA10_CORPUS_PATH
    recipe_path = repo_root / DATA10_RECIPE_PATH
    joined = _joined_training_text()
    if corpus_path.read_text(encoding="utf-8") != joined:
        raise RuntimeError("DATA-10 project-authored training corpus bytes drifted")
    corpus_sha = _sha_text(joined)
    if corpus_sha != DATA10_CORPUS_SHA256:
        raise RuntimeError("DATA-10 project-authored training corpus SHA-256 drifted")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    local = recipe.get("local_mechanics_corpus", {})
    admission = recipe.get("source_admission", {})
    if local.get("sha256") != corpus_sha or local.get("representative_corpus") is not False:
        raise RuntimeError("DATA-10 recipe/corpus truth boundary drifted")
    if admission.get("external_sources_training_approved_at_recipe_creation") != 0:
        raise RuntimeError("external-data approval changed; re-review rather than silently consuming it")
    train_hashes = {_sha_text(text) for _, _, text in TRAIN_RECORDS}
    validation_hashes = {_sha_text(text) for _, _, text in VALIDATION_RECORDS}
    overlap = sorted(train_hashes & validation_hashes)
    if overlap:
        raise RuntimeError("training and validation texts overlap exactly")
    return {
        "recipe_path": DATA10_RECIPE_PATH,
        "recipe_sha256": sha256_file(recipe_path),
        "corpus_path": DATA10_CORPUS_PATH,
        "corpus_sha256": corpus_sha,
        "corpus_bytes": len(joined.encode("utf-8")),
        "training_records": len(TRAIN_RECORDS),
        "validation_records": len(VALIDATION_RECORDS),
        "exact_train_validation_overlap": overlap,
        "external_sources_training_approved": 0,
        "project_authored_synthetic": True,
        "representative_corpus": False,
    }


def _tokenizer_manifest(repo_root: Path) -> TokenizerTrainingManifest:
    del repo_root
    joined = _joined_training_text()
    corpus_sha = _sha_text(joined)
    dataset_sha = _canonical_hash({
        "dataset_id": "data10-project-authored-uk-en-code-v1",
        "records": [record_id for _, record_id, _ in TRAIN_RECORDS],
        "corpus_sha256": corpus_sha,
    })
    return TokenizerTrainingManifest(
        experiment_id="data10-bpe-512-v1",
        algorithm=TOKENIZER_ALGORITHM,
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id="data10-project-authored-uk-en-code-v1",
        dataset_manifest_sha256=dataset_sha,
        corpus_files=(CorpusFileIdentity(DATA10_CORPUS_PATH, corpus_sha, len(joined.encode("utf-8"))),),
        vocab_size=TOKENIZER_REQUESTED_VOCAB,
        min_frequency=TOKENIZER_MIN_FREQUENCY,
    )


def _build_tokenizer(repo_root: Path) -> HFTokenizerAdapter:
    _verify_data_boundary(repo_root)
    manifest = _tokenizer_manifest(repo_root)
    texts = tuple(text for _, _, text in TRAIN_RECORDS)
    first = train_hf_tokenizer(manifest, texts)
    second = train_hf_tokenizer(manifest, texts)
    if first.vocab_size != TOKENIZER_EXPECTED_VOCAB:
        raise RuntimeError(f"DATA-10 BPE vocabulary drift: {first.vocab_size} != {TOKENIZER_EXPECTED_VOCAB}")
    if first.artifact_identity.config_sha256 != second.artifact_identity.config_sha256:
        raise RuntimeError("DATA-10 BPE artifact identity is no longer repeatable")
    if first.identity.vocab_sha256 != second.identity.vocab_sha256:
        raise RuntimeError("DATA-10 BPE ordered vocabulary is no longer repeatable")
    return first


def _model_spec(geometry: Mapping[str, int], *, vocab_size: int) -> ModelSpec:
    return ModelSpec(
        schema_version=1, vocab_size=vocab_size, max_seq_len=256,
        d_model=int(geometry["d_model"]), n_layers=int(geometry["n_layers"]),
        n_heads=int(geometry["n_heads"]), n_kv_heads=int(geometry["n_kv_heads"]),
        head_dim=int(geometry["head_dim"]), d_ff=int(geometry["d_ff"]),
        activation="swiglu", norm_kind="rmsnorm", norm_placement="pre", norm_eps=1e-5,
        position_embedding="rope", rope_theta=10_000.0,
        rope_rotary_dim=int(geometry["head_dim"]), attention_bias=False, mlp_bias=False,
        attention_dropout=0.0, final_norm=True, tie_word_embeddings=True, lm_head_bias=False,
    )


def controlled_specs(vocab_size: int) -> tuple[ModelSpec, ...]:
    specs = tuple(_model_spec(geometry, vocab_size=vocab_size) for geometry in _GEOMETRIES)
    if vocab_size == TOKENIZER_EXPECTED_VOCAB:
        counts = tuple(spec.parameter_count() for spec in specs)
        if counts != _EXPECTED_COUNTS_VOCAB472:
            raise RuntimeError(f"LEARN-04 parameter family drift: {counts!r}")
    return specs


def _trainer_config(*, max_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8,
        max_steps=max_steps, warmup_steps=0, scheduler="constant",
        gradient_accumulation_steps=1, gradient_clip_norm=1.0,
        precision="fp32", seed=seed, deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _stream_by_stratum(tokenizer: HFTokenizerAdapter) -> dict[str, tuple[int, ...]]:
    streams: dict[str, tuple[int, ...]] = {}
    for stratum in ("uk", "en", "code"):
        text = "\n".join(value for name, _, value in TRAIN_RECORDS if name == stratum) + "\n"
        encoded = tuple(tokenizer.encode(text))
        if len(encoded) < DEFAULT_SEQUENCE_LENGTH:
            raise RuntimeError(f"{stratum} tokenizer stream unexpectedly too short")
        streams[stratum] = encoded
    return streams


def _mixture_plan(tokenizer: HFTokenizerAdapter) -> MixturePlan:
    sources = []
    for name in ("uk", "en", "code"):
        payload = [{"record_id": rid, "text_sha256": _sha_text(text)} for stratum, rid, text in TRAIN_RECORDS if stratum == name]
        sources.append(MixtureSource(name, _canonical_hash(payload), MIXTURE_WEIGHTS[name]))
    packing_hash = _canonical_hash({
        "version": PACKING_VERSION,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "row_stride": DEFAULT_SEQUENCE_LENGTH - 1,
        "cross_stratum_rows": False,
    })
    return MixturePlan(
        plan_id="learn04-uk-en-code-bpe-token-budget-v1",
        tokenizer_config_sha256=tokenizer.identity.config_sha256,
        tokenizer_vocab_sha256=tokenizer.identity.vocab_sha256,
        packing_config_sha256=packing_hash,
        sources=tuple(sources), seed=126, num_shards=1,
    )


def _schedule(plan: MixturePlan, total_rows: int) -> tuple[tuple[str, int], ...]:
    counts = {source.name: 0 for source in plan.sources}
    result = []
    for sample_index in range(total_rows):
        source = plan.source_for_sample(sample_index)
        result.append((source, counts[source]))
        counts[source] += 1
    return tuple(result)


def _make_batch(
    streams: Mapping[str, Sequence[int]], schedule: Sequence[tuple[str, int]], *,
    step: int, batch_size: int, sequence_length: int,
) -> torch.Tensor:
    rows: list[list[int]] = []
    stride = sequence_length - 1
    for row_index in range(step * batch_size, (step + 1) * batch_size):
        source, occurrence = schedule[row_index]
        stream = streams[source]
        start = (occurrence * stride) % len(stream)
        rows.append([int(stream[(start + offset) % len(stream)]) for offset in range(sequence_length)])
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def _validation_metrics(model: TwelveSixDecoder, tokenizer: HFTokenizerAdapter) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    text = _validation_text()
    ids = tokenizer.encode(text)
    total_nll = 0.0
    total_targets = 0
    start = 0
    while start < len(ids) - 1:
        chunk = ids[start : start + model.spec.max_seq_len]
        if len(chunk) < 2:
            break
        input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
            input_ids[:, 1:].reshape(-1), reduction="sum",
        )
        total_nll += float(loss.item())
        total_targets += len(chunk) - 1
        start += model.spec.max_seq_len - 1
    model.train(was_training)
    if total_targets != len(ids) - 1:
        raise RuntimeError("validation chunking failed to score each post-boundary token exactly once")
    utf8_bytes = len(text.encode("utf-8"))
    return {
        "loss_per_token": total_nll / total_targets,
        "nll_nats": total_nll,
        "target_tokens": total_targets,
        "utf8_bytes": utf8_bytes,
        "boundary_conditioned_bpb": total_nll / (math.log(2.0) * utf8_bytes),
    }


@torch.no_grad()
def _greedy_generation(model: TwelveSixDecoder, tokenizer: HFTokenizerAdapter) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    prompt_ids = tokenizer.encode(GENERATION_PROMPT)
    all_ids = list(prompt_ids)
    generated: list[int] = []
    for _ in range(GENERATION_NEW_TOKENS):
        context = all_ids[-model.spec.max_seq_len :]
        logits = model(torch.tensor(context, dtype=torch.long).unsqueeze(0)).logits
        token_id = int(torch.argmax(logits[0, -1]).item())
        generated.append(token_id)
        all_ids.append(token_id)
    model.train(was_training)
    return {
        "prompt": GENERATION_PROMPT,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "full_text": tokenizer.decode(all_ids, skip_special_tokens=False),
        "max_new_tokens": GENERATION_NEW_TOKENS,
        "sampling": "greedy",
    }


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parameter_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _movement(model: TwelveSixDecoder, baseline: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    delta_sq = base_sq = 0.0
    changed = total = 0
    for name, parameter in model.named_parameters():
        before = baseline[name].float()
        delta = parameter.detach().float() - before
        delta_sq += float(torch.sum(delta * delta).item())
        base_sq += float(torch.sum(before * before).item())
        changed += int(torch.count_nonzero(delta).item())
        total += delta.numel()
    delta_l2, base_l2 = math.sqrt(delta_sq), math.sqrt(base_sq)
    return {
        "delta_l2": delta_l2,
        "baseline_weight_l2": base_l2,
        "relative_delta_l2": delta_l2 / max(base_l2, 1e-30),
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
    }


def _single_step_update_ratio(model: TwelveSixDecoder, before: Mapping[str, torch.Tensor]) -> float:
    delta_sq = before_sq = 0.0
    for name, parameter in model.named_parameters():
        prior = before[name].float()
        delta = parameter.detach().float() - prior
        delta_sq += float(torch.sum(delta * delta).item())
        before_sq += float(torch.sum(prior * prior).item())
    return math.sqrt(delta_sq) / max(math.sqrt(before_sq), 1e-30)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values), "min": min(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "max": max(values),
    }


def _budget_steps(token_budgets: Sequence[int], tokens_per_step: int) -> tuple[int, ...]:
    return tuple(math.ceil(budget / tokens_per_step) for budget in token_budgets)


def _checkpoint_point(
    *, requested_budget: int, trainer: Trainer, model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter, metrics: Any, initial_snapshot: Mapping[str, torch.Tensor],
    elapsed: float, clip_count: int, grad_norms: Sequence[float], update_ratios: Sequence[float],
) -> dict[str, Any]:
    return {
        "requested_token_budget": requested_budget,
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "last_train_loss": float(metrics.loss),
        "last_grad_norm": float(metrics.grad_norm),
        "validation": _validation_metrics(model, tokenizer),
        "clip_count_cumulative": clip_count,
        "clip_frequency_cumulative": clip_count / max(trainer.optimizer_step, 1),
        "gradient_norm_summary_cumulative": _summary(grad_norms),
        "sampled_update_ratio_summary_cumulative": _summary(update_ratios),
        "optimizer_state_tensor_bytes": _tensor_bytes(trainer.optimizer.state_dict()),
        "parameter_movement_from_init": _movement(model, initial_snapshot),
        "elapsed_training_seconds": elapsed,
        "optimized_tokens_per_second": trainer.tokens_seen / max(elapsed, 1e-12),
        "generation": _greedy_generation(model, tokenizer),
    }


def _run_range(
    *, model: TwelveSixDecoder, trainer: Trainer, tokenizer: HFTokenizerAdapter,
    streams: Mapping[str, Sequence[int]], schedule: Sequence[tuple[str, int]],
    start_step: int, end_step: int, batch_size: int, sequence_length: int,
    token_budgets: Sequence[int], curve_interval_steps: int,
    initial_snapshot: Mapping[str, torch.Tensor], prior_curve: list[dict[str, Any]] | None = None,
    prior_checkpoints: list[dict[str, Any]] | None = None,
    prior_grad_norms: list[float] | None = None,
    prior_update_ratios: list[float] | None = None, prior_clip_count: int = 0,
    prior_elapsed: float = 0.0,
) -> dict[str, Any]:
    curve = list(prior_curve or [])
    checkpoints = list(prior_checkpoints or [])
    grad_norms = list(prior_grad_norms or [])
    update_ratios = list(prior_update_ratios or [])
    clip_count = int(prior_clip_count)
    tokens_per_step = batch_size * (sequence_length - 1)
    budget_steps = _budget_steps(token_budgets, tokens_per_step)
    checkpoint_steps = frozenset(budget_steps)
    completed = {int(point["requested_token_budget"]) for point in checkpoints}
    started = time.perf_counter()
    for step in range(start_step, end_step):
        sampled = step == 0 or (step + 1) % curve_interval_steps == 0 or (step + 1) in checkpoint_steps
        before_step = _parameter_snapshot(model) if sampled else None
        batch = _make_batch(streams, schedule, step=step, batch_size=batch_size, sequence_length=sequence_length)
        metrics = trainer.train_microbatch({"input_ids": batch})
        if not metrics.optimizer_stepped or metrics.grad_norm is None:
            raise RuntimeError("LEARN-04 requires one committed optimizer update per batch")
        grad_norm = float(metrics.grad_norm)
        if not math.isfinite(metrics.loss) or not math.isfinite(grad_norm):
            raise RuntimeError("non-finite training telemetry")
        grad_norms.append(grad_norm)
        if trainer.config.gradient_clip_norm is not None and grad_norm > trainer.config.gradient_clip_norm:
            clip_count += 1
        update_ratio = None
        if before_step is not None:
            update_ratio = _single_step_update_ratio(model, before_step)
            update_ratios.append(update_ratio)
        elapsed = prior_elapsed + time.perf_counter() - started
        if sampled:
            curve.append({
                "optimized_tokens": trainer.tokens_seen,
                "optimizer_step": trainer.optimizer_step,
                "train_loss": float(metrics.loss), "grad_norm": grad_norm,
                "gradient_clipped": bool(trainer.config.gradient_clip_norm is not None and grad_norm > trainer.config.gradient_clip_norm),
                "sampled_parameter_update_ratio": update_ratio,
                "learning_rate": float(metrics.learning_rate),
                "elapsed_training_seconds": elapsed,
            })
        for budget, budget_step in zip(token_budgets, budget_steps, strict=True):
            if budget not in completed and trainer.optimizer_step >= budget_step:
                checkpoints.append(_checkpoint_point(
                    requested_budget=budget, trainer=trainer, model=model, tokenizer=tokenizer,
                    metrics=metrics, initial_snapshot=initial_snapshot, elapsed=elapsed,
                    clip_count=clip_count, grad_norms=grad_norms, update_ratios=update_ratios,
                ))
                completed.add(budget)
    return {
        "curve": curve, "checkpoints": checkpoints, "gradient_norms": grad_norms,
        "sampled_update_ratios": update_ratios, "clip_count": clip_count,
        "elapsed_training_seconds": prior_elapsed + time.perf_counter() - started,
    }


def _identity(
    *, source_sha: str, spec: ModelSpec, init_spec: InitSpec,
    tokenizer: HFTokenizerAdapter, plan: MixturePlan, trainer: Trainer, repo_root: Path,
) -> CheckpointIdentity:
    dataset_hash = _canonical_hash({
        "data10_recipe_sha256": sha256_file(repo_root / DATA10_RECIPE_PATH),
        "train_corpus_sha256": DATA10_CORPUS_SHA256,
        "validation_sha256": _sha_text(_validation_text()),
        "validation_used_for_training": False,
    })
    split_identity = f"learn04-project-authored-train:{DATA10_CORPUS_SHA256}"
    environment_hash = sha256_file(repo_root / "requirements/locks/index.json")
    training_config = {
        "schema": SCHEMA, "authority": AUTHORITY,
        "init_spec_sha256": init_spec.identity_sha256(),
        "training": asdict(trainer.config),
        "data": {
            "dataset_manifest_sha256": dataset_hash,
            "split_identity": split_identity,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "packing_sha256": plan.packing_config_sha256,
            "packing_version": PACKING_VERSION,
            "mixture_plan_sha256": plan.sha256,
        },
        "environment": {"lock_sha256": environment_hash},
    }
    run_manifest = {
        "schema": SCHEMA, "source_sha": source_sha,
        "model_spec_sha256": spec.identity_sha256(), "parameters": spec.parameter_count(),
        "tokenizer_sha256": tokenizer.identity.config_sha256,
        "dataset_manifest_sha256": dataset_hash, "mixture_plan_sha256": plan.sha256,
        "step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen,
    }
    return CheckpointIdentity(
        git_sha=source_sha, model_spec=spec.to_dict(), parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_hash, run_manifest_hash=hash_json(run_manifest),
        training_config=training_config, seed=trainer.config.seed,
        precision=trainer.config.precision, step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "lr": trainer.config.learning_rate, "betas": list(trainer.config.betas), "eps": trainer.config.eps, "weight_decay": trainer.config.weight_decay},
        scheduler={"name": trainer.config.scheduler}, environment_lock_hash=environment_hash,
    )


def _initial_state(spec: ModelSpec, init_spec: InitSpec, config: TrainerConfig, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    snapshot = _parameter_snapshot(model)
    trainer = Trainer(model, config, device="cpu")
    return model, trainer, snapshot


def _state_digest(model: TwelveSixDecoder) -> str:
    return _canonical_hash({
        name: hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
        for name, tensor in model.state_dict().items()
    })


def stage1(
    *, repo_root: Path, source_sha: str, output_dir: Path,
    token_budgets: tuple[int, ...] = DEFAULT_TOKEN_BUDGETS,
    resume_budget: int = DEFAULT_RESUME_BUDGET, batch_size: int = DEFAULT_BATCH_SIZE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH, seed: int = DEFAULT_SEED,
    curve_interval_steps: int = DEFAULT_CURVE_INTERVAL_STEPS, torch_threads: int = 2,
) -> dict[str, Any]:
    repo_root, output_dir = repo_root.resolve(), output_dir.resolve()
    _require_source_sha(source_sha)
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("stage1 exact-checkout mismatch")
    if tuple(sorted(set(token_budgets))) != token_budgets or resume_budget not in token_budgets or resume_budget == token_budgets[-1]:
        raise ValueError("invalid token budget/resume controls")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    data_boundary = _verify_data_boundary(repo_root)
    tokenizer = _build_tokenizer(repo_root)
    specs = controlled_specs(tokenizer.vocab_size)
    plan = _mixture_plan(tokenizer)
    streams = _stream_by_stratum(tokenizer)
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps = math.ceil(token_budgets[-1] / tokens_per_step)
    resume_step = math.ceil(resume_budget / tokens_per_step)
    schedule = _schedule(plan, max_steps * batch_size)
    config = _trainer_config(max_steps=max_steps, seed=seed)
    init_spec = InitSpec()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_records = []
    for index, spec in enumerate(specs):
        model, trainer, initial_snapshot = _initial_state(spec, init_spec, config, seed)
        record: dict[str, Any] = {
            "model_spec": spec.to_dict(), "model_identity_sha256": spec.identity_sha256(),
            "parameters": spec.parameter_count(),
            "initial_validation": _validation_metrics(model, tokenizer),
            "initial_generation": _greedy_generation(model, tokenizer),
        }
        end_step = max_steps if index < 3 else resume_step
        active_budgets = token_budgets if index < 3 else tuple(b for b in token_budgets if b <= resume_budget)
        run = _run_range(
            model=model, trainer=trainer, tokenizer=tokenizer, streams=streams,
            schedule=schedule, start_step=0, end_step=end_step, batch_size=batch_size,
            sequence_length=sequence_length, token_budgets=active_budgets,
            curve_interval_steps=curve_interval_steps, initial_snapshot=initial_snapshot,
        )
        record.update(run)
        record["optimizer_state_tensor_bytes"] = _tensor_bytes(trainer.optimizer.state_dict())
        if index == 3:
            identity = _identity(source_sha=source_sha, spec=spec, init_spec=init_spec, tokenizer=tokenizer, plan=plan, trainer=trainer, repo_root=repo_root)
            checkpoint_dir = output_dir / "resume-checkpoint"
            started = time.perf_counter()
            manifest = save_trainer_checkpoint(checkpoint_dir, model=model, trainer=trainer, identity=identity)
            record["resume_checkpoint"] = {
                "path": "resume-checkpoint", "checkpoint_id": manifest["checkpoint_id"],
                "format": manifest["format"], "format_version": manifest["format_version"],
                "step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen,
                "directory_bytes": _directory_bytes(checkpoint_dir),
                "save_seconds": time.perf_counter() - started,
                "run_manifest_hash": identity.run_manifest_hash,
                "training_config_hash": hash_json(identity.training_config),
                "dataset_manifest_hash": identity.dataset_manifest_hash,
                "environment_lock_hash": identity.environment_lock_hash,
                "model_state_sha256": _state_digest(model),
            }
        model_records.append(record)
    payload: dict[str, Any] = {
        "schema": SCHEMA, "phase": "stage1", "authority": AUTHORITY, "source_sha": source_sha,
        "process": {"pid": os.getpid(), "process_token": uuid.uuid4().hex, "python": platform.python_version(), "torch": torch.__version__, "torch_threads": torch_threads},
        "parents": {"research41_fixed_control_sha": RESEARCH41_PARENT_SHA, "data10_multilingual_sha": DATA10_PARENT_SHA},
        "data_boundary": data_boundary,
        "tokenizer": {
            "version": tokenizer.identity.version, "algorithm": TOKENIZER_ALGORITHM,
            "requested_vocab_size": TOKENIZER_REQUESTED_VOCAB, "actual_vocab_size": tokenizer.vocab_size,
            "config_sha256": tokenizer.identity.config_sha256, "vocab_sha256": tokenizer.identity.vocab_sha256,
            "artifact_repeatable": True, "training_manifest_sha256": _tokenizer_manifest(repo_root).sha256,
            "validation_used_for_tokenizer_training": False, "frozen": False,
        },
        "controls": {
            "canonical_base": "random_init", "model_max_seq_len": 256,
            "training_sequence_length": sequence_length, "batch_size": batch_size,
            "tokens_per_optimizer_step": tokens_per_step, "token_budgets": list(token_budgets),
            "resume_budget": resume_budget, "seed": seed, "optimizer": asdict(config),
            "mixture_weights_post_tokenization_loss_token_budget": MIXTURE_WEIGHTS,
            "mixture_plan_sha256": plan.sha256, "packing_sha256": plan.packing_config_sha256,
            "packing_version": PACKING_VERSION, "curve_interval_steps": curve_interval_steps,
        },
        "models": model_records,
    }
    payload["stage1_sha256"] = _canonical_hash(payload)
    (output_dir / "stage1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _comparison(models: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    common = sorted(set.intersection(*[{int(p["requested_token_budget"]) for p in m["checkpoints"]} for m in models]))
    rows = []
    for budget in common:
        points = [(int(m["parameters"]), next(p for p in m["checkpoints"] if int(p["requested_token_budget"]) == budget)) for m in models]
        baseline_loss = float(points[0][1]["validation"]["loss_per_token"])
        baseline_bpb = float(points[0][1]["validation"]["boundary_conditioned_bpb"])
        rows.append({
            "requested_token_budget": budget,
            "models": [{
                "parameters": parameters, "optimized_tokens": int(point["optimized_tokens"]),
                "validation_loss": float(point["validation"]["loss_per_token"]),
                "boundary_conditioned_bpb": float(point["validation"]["boundary_conditioned_bpb"]),
                "validation_loss_reduction_vs_smallest": baseline_loss - float(point["validation"]["loss_per_token"]),
                "bpb_reduction_vs_smallest": baseline_bpb - float(point["validation"]["boundary_conditioned_bpb"]),
            } for parameters, point in points],
        })
    return rows


def stage2(*, repo_root: Path, source_sha: str, output_dir: Path, torch_threads: int = 2) -> dict[str, Any]:
    repo_root, output_dir = repo_root.resolve(), output_dir.resolve()
    _require_source_sha(source_sha)
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("stage2 exact-checkout mismatch")
    stage1_payload = json.loads((output_dir / "stage1.json").read_text(encoding="utf-8"))
    supplied_hash = stage1_payload.pop("stage1_sha256")
    if supplied_hash != _canonical_hash(stage1_payload):
        raise RuntimeError("stage1 evidence self-hash mismatch")
    stage1_payload["stage1_sha256"] = supplied_hash
    producer_token = str(stage1_payload["process"]["process_token"])
    process_token = uuid.uuid4().hex
    if process_token == producer_token:
        raise RuntimeError("fresh-process marker unexpectedly matched producer")
    controls = stage1_payload["controls"]
    token_budgets = tuple(int(v) for v in controls["token_budgets"])
    resume_budget = int(controls["resume_budget"])
    batch_size, sequence_length, seed = int(controls["batch_size"]), int(controls["training_sequence_length"]), int(controls["seed"])
    curve_interval_steps = int(controls["curve_interval_steps"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = _build_tokenizer(repo_root)
    if tokenizer.identity.config_sha256 != stage1_payload["tokenizer"]["config_sha256"]:
        raise RuntimeError("tokenizer identity changed between processes")
    large_spec = controlled_specs(tokenizer.vocab_size)[-1]
    plan = _mixture_plan(tokenizer)
    if plan.sha256 != controls["mixture_plan_sha256"]:
        raise RuntimeError("mixture plan changed between processes")
    streams = _stream_by_stratum(tokenizer)
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps, resume_step = math.ceil(token_budgets[-1] / tokens_per_step), math.ceil(resume_budget / tokens_per_step)
    schedule = _schedule(plan, max_steps * batch_size)
    config, init_spec = _trainer_config(max_steps=max_steps, seed=seed), InitSpec()
    random.seed(seed); torch.manual_seed(seed)
    init_reference_model = TwelveSixDecoder(large_spec, init_spec)
    initial_snapshot = _parameter_snapshot(init_reference_model)
    del init_reference_model
    model = TwelveSixDecoder(large_spec, init_spec)
    trainer = Trainer(model, config, device="cpu")
    large_stage1 = stage1_payload["models"][-1]
    checkpoint_meta = large_stage1["resume_checkpoint"]
    load_started = time.perf_counter()
    load_trainer_checkpoint(
        output_dir / checkpoint_meta["path"], model=model, trainer=trainer, restore_rng=True,
        expected_git_sha=source_sha, expected_model_spec_hash=large_spec.identity_sha256(),
        expected_init_spec_hash=init_spec.identity_sha256(),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=checkpoint_meta["dataset_manifest_hash"],
        expected_split_identity=f"learn04-project-authored-train:{DATA10_CORPUS_SHA256}",
        expected_packing_hash=plan.packing_config_sha256, expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=checkpoint_meta["run_manifest_hash"],
        expected_training_config_hash=checkpoint_meta["training_config_hash"],
        expected_environment_lock_hash=checkpoint_meta["environment_lock_hash"], expected_seed=seed,
    )
    load_seconds = time.perf_counter() - load_started
    if trainer.optimizer_step != resume_step or _state_digest(model) != checkpoint_meta["model_state_sha256"]:
        raise RuntimeError("fresh-process checkpoint reload equality failed")
    loaded_step, loaded_tokens = trainer.optimizer_step, trainer.tokens_seen
    run = _run_range(
        model=model, trainer=trainer, tokenizer=tokenizer, streams=streams, schedule=schedule,
        start_step=resume_step, end_step=max_steps, batch_size=batch_size, sequence_length=sequence_length,
        token_budgets=token_budgets, curve_interval_steps=curve_interval_steps,
        initial_snapshot=initial_snapshot, prior_curve=list(large_stage1["curve"]),
        prior_checkpoints=list(large_stage1["checkpoints"]),
        prior_grad_norms=[float(x) for x in large_stage1["gradient_norms"]],
        prior_update_ratios=[float(x) for x in large_stage1["sampled_update_ratios"]],
        prior_clip_count=int(large_stage1["clip_count"]),
        prior_elapsed=float(large_stage1["elapsed_training_seconds"]),
    )
    large_record = dict(large_stage1)
    large_record.update(run)
    large_record["optimizer_state_tensor_bytes"] = _tensor_bytes(trainer.optimizer.state_dict())
    large_record["fresh_process_resume"] = {
        "producer_pid": int(stage1_payload["process"]["pid"]), "producer_process_token": producer_token,
        "consumer_pid": os.getpid(), "consumer_process_token": process_token,
        "distinct_process_token": process_token != producer_token,
        "checkpoint_load_seconds": load_seconds, "model_state_exact_after_load": True,
        "optimizer_step_after_load": loaded_step, "loaded_tokens_seen": loaded_tokens,
    }
    final_identity = _identity(source_sha=source_sha, spec=large_spec, init_spec=init_spec, tokenizer=tokenizer, plan=plan, trainer=trainer, repo_root=repo_root)
    final_checkpoint_dir = output_dir / "final-checkpoint"
    save_started = time.perf_counter()
    final_manifest = save_trainer_checkpoint(final_checkpoint_dir, model=model, trainer=trainer, identity=final_identity)
    large_record["final_checkpoint"] = {
        "path": "final-checkpoint", "checkpoint_id": final_manifest["checkpoint_id"],
        "step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen,
        "directory_bytes": _directory_bytes(final_checkpoint_dir),
        "save_seconds": time.perf_counter() - save_started,
    }
    models = [dict(item) for item in stage1_payload["models"][:-1]] + [large_record]
    final_point = large_record["checkpoints"][-1]
    report: dict[str, Any] = {
        "schema": SCHEMA, "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha, "parents": stage1_payload["parents"]},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "torch_threads": torch_threads, "paid_compute": False},
        "data": stage1_payload["data_boundary"], "tokenizer": stage1_payload["tokenizer"],
        "controls": controls, "models": models,
        "matched_checkpoint_comparison": _comparison(models),
        "one_million_summary": {
            "parameters": int(large_record["parameters"]), "optimized_tokens": int(final_point["optimized_tokens"]),
            "final_train_loss": float(final_point["last_train_loss"]),
            "initial_validation_loss": float(large_record["initial_validation"]["loss_per_token"]),
            "final_validation_loss": float(final_point["validation"]["loss_per_token"]),
            "initial_boundary_conditioned_bpb": float(large_record["initial_validation"]["boundary_conditioned_bpb"]),
            "final_boundary_conditioned_bpb": float(final_point["validation"]["boundary_conditioned_bpb"]),
            "gradient_norm_summary": _summary([float(v) for v in large_record["gradient_norms"]]),
            "clip_count": int(large_record["clip_count"]),
            "clip_frequency": int(large_record["clip_count"]) / max(int(final_point["optimizer_steps"]), 1),
            "sampled_parameter_update_ratio_summary": _summary([float(v) for v in large_record["sampled_update_ratios"]]),
            "optimizer_state_tensor_bytes": int(large_record["optimizer_state_tensor_bytes"]),
            "training_wall_seconds": float(large_record["elapsed_training_seconds"]),
            "optimized_tokens_per_second": int(final_point["optimized_tokens"]) / max(float(large_record["elapsed_training_seconds"]), 1e-12),
            "resume_checkpoint_bytes": int(large_record["resume_checkpoint"]["directory_bytes"]),
            "resume_checkpoint_save_seconds": float(large_record["resume_checkpoint"]["save_seconds"]),
            "resume_checkpoint_load_seconds": load_seconds,
            "final_checkpoint_bytes": int(large_record["final_checkpoint"]["directory_bytes"]),
            "final_checkpoint_save_seconds": float(large_record["final_checkpoint"]["save_seconds"]),
            "generation_checkpoints": [
                {"label": "initialization", "optimized_tokens": 0, "snapshot": large_record["initial_generation"]},
                *[{"label": f"tokens-{point['requested_token_budget']}", "optimized_tokens": int(point["optimized_tokens"]), "snapshot": point["generation"]} for point in large_record["checkpoints"]],
            ],
        },
        "truth_boundary": {
            "base_pretraining": True, "foreign_pretrained_weights_used": False,
            "instruction_or_sft_used": False, "paid_compute_used": False,
            "external_training_sources_approved": 0, "representative_corpus": False,
            "project_authored_corpus_recycled": True, "tokenizer_frozen": False,
            "quality_or_general_capability_claim": False, "stage_promotion_authority": False,
            "scientific_interpretation": "Observed held-out behavior is valid only for the project-authored DATA-10 UK/EN/code fixture and repeated token trajectory. It is a real learned Base optimization experiment, not representative-corpus pretraining.",
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    (output_dir / "learn04-real-1m-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_report(report: Mapping[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("LEARN-04 report schema/authority mismatch")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("LEARN-04 report source SHA mismatch")
    if report["runtime"].get("paid_compute") is not False or report["data"].get("representative_corpus") is not False:
        raise ValueError("LEARN-04 truth boundary weakened")
    if report["tokenizer"].get("actual_vocab_size") != TOKENIZER_EXPECTED_VOCAB:
        raise ValueError("LEARN-04 tokenizer vocabulary drift")
    models = report.get("models")
    if not isinstance(models, list) or [int(model["parameters"]) for model in models] != list(_EXPECTED_COUNTS_VOCAB472):
        raise ValueError("LEARN-04 model family drift")
    for model in models:
        if [int(point["requested_token_budget"]) for point in model["checkpoints"]] != list(DEFAULT_TOKEN_BUDGETS):
            raise ValueError("LEARN-04 matched checkpoint grid incomplete")
        for point in model["checkpoints"]:
            if not math.isfinite(float(point["validation"]["loss_per_token"])) or not math.isfinite(float(point["validation"]["boundary_conditioned_bpb"])):
                raise ValueError("non-finite held-out evidence")
    resume = models[-1].get("fresh_process_resume", {})
    if resume.get("distinct_process_token") is not True or resume.get("model_state_exact_after_load") is not True:
        raise ValueError("fresh-process resume evidence missing")
    truth = report["truth_boundary"]
    for key in ("foreign_pretrained_weights_used", "instruction_or_sft_used", "paid_compute_used", "representative_corpus", "tokenizer_frozen", "quality_or_general_capability_claim", "stage_promotion_authority"):
        if truth.get(key) is not False:
            raise ValueError(f"truth boundary weakened: {key}")
    supplied = report.get("report_sha256")
    unsigned = dict(report); unsigned.pop("report_sha256", None)
    if supplied != _canonical_hash(unsigned):
        raise ValueError("LEARN-04 report self-hash mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("stage1", "stage2"):
        sub = subs.add_parser(name)
        sub.add_argument("--repo-root", type=Path, default=Path("."))
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--torch-threads", type=int, default=2)
    validate = subs.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stage1":
        payload = stage1(repo_root=args.repo_root, source_sha=args.source_sha, output_dir=args.output_dir, torch_threads=args.torch_threads)
        print(json.dumps({"phase": "stage1", "stage1_sha256": payload["stage1_sha256"], "parameters": [model["parameters"] for model in payload["models"]]}, sort_keys=True))
        return 0
    if args.command == "stage2":
        report = stage2(repo_root=args.repo_root, source_sha=args.source_sha, output_dir=args.output_dir, torch_threads=args.torch_threads)
        validate_report(report, expected_source_sha=args.source_sha)
        print(json.dumps(report["one_million_summary"], sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
