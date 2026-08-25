"""Genuine ~100K LOCAL_FREE Base-training experiment on the live DATA-10 corpus.

This is research evidence only. It trains random-init 12-6 weights from scratch on the
currently training-eligible project-authored UK/EN/code corpus. External approved
training sources are intentionally zero, so no representative-language claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
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
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.learn01-real-100k.v1"
AUTHORITY = "LOCAL_FREE_REAL_TRAINING_RESEARCH_EVIDENCE_NOT_STAGE_FREEZE"
REPOSITORY = "Oleksii-debug/12-6-ai."
CORPUS_PATH = Path("data/synthetic/data10/uk-en-code-train.txt")
RECIPE_PATH = Path("configs/data/multilingual_uk_en_code_v1.experimental.json")
TOKENIZER_LOCK_PATH = Path("requirements/experiments/tokenizers-linux-x86_64.lock.txt")
RUNTIME_LOCK_PATH = Path("requirements/locks/linux-x86_64/runtime.lock.txt")
TOOLCHAIN_LOCK_PATH = Path("requirements/locks/linux-x86_64/toolchain.lock.txt")
EXPECTED_CORPUS_SHA256 = "059f04e01d6fc6b8224b373b08efbb37f09d546de35ed510afdb4587ebdb6012"
TOKENIZERS_VERSION = "0.23.1"
MIXTURE_PATTERN = (
    "uk", "en", "uk", "code", "en",
    "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk",
    "en", "uk", "code", "en", "uk",
)
HOLDOUT_INDICES = (2, 5, 16)
GENERATION_STEPS = frozenset({0, 250, 500, 1000})
_HEX40 = frozenset("0123456789abcdef")


class Learn01Error(RuntimeError):
    """Fail-closed experiment error."""


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        encoding="utf-8",
    ).strip()


def _require_source_sha(source_sha: str, repo_root: Path) -> None:
    if len(source_sha) != 40 or any(ch not in _HEX40 for ch in source_sha):
        raise Learn01Error("source_sha must be lowercase full 40-hex Git SHA")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise Learn01Error(
            f"exact-checkout mismatch: expected {source_sha}, observed {observed}"
        )


def _read_recipe(repo_root: Path) -> dict[str, Any]:
    recipe = json.loads((repo_root / RECIPE_PATH).read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise Learn01Error("DATA-10 recipe must be a JSON object")
    local = recipe.get("local_mechanics_corpus")
    source_admission = recipe.get("source_admission")
    if not isinstance(local, dict) or not isinstance(source_admission, dict):
        raise Learn01Error("DATA-10 recipe is missing corpus/admission contracts")
    if local.get("authority") != "PROJECT_AUTHORED_SYNTHETIC_ONLY":
        raise Learn01Error("current corpus is not explicitly project-authored synthetic")
    if local.get("path") != CORPUS_PATH.as_posix():
        raise Learn01Error("DATA-10 local corpus path drifted")
    if local.get("sha256") != EXPECTED_CORPUS_SHA256:
        raise Learn01Error("DATA-10 recipe corpus identity drifted")
    if source_admission.get("external_sources_training_approved_at_recipe_creation") != 0:
        raise Learn01Error("external-source truth boundary changed; re-audit before this experiment")
    if source_admission.get("project_authored_synthetic_allowed") is not True:
        raise Learn01Error("DATA-10 does not permit project-authored synthetic training data")
    return recipe


def _split_corpus(repo_root: Path) -> dict[str, Any]:
    corpus_path = repo_root / CORPUS_PATH
    if sha256_file(corpus_path) != EXPECTED_CORPUS_SHA256:
        raise Learn01Error("DATA-10 corpus bytes do not match the declared recipe identity")
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 17:
        raise Learn01Error(f"expected 17 DATA-10 corpus lines, observed {len(lines)}")
    if any(not line for line in lines):
        raise Learn01Error("DATA-10 corpus contains an unexpected empty line")

    strata: dict[str, list[tuple[int, str]]] = {"uk": [], "en": [], "code": []}
    for index, text in enumerate(lines):
        name = "uk" if index < 3 else "en" if index < 6 else "code"
        strata[name].append((index, text))

    heldout = {index: lines[index] for index in HOLDOUT_INDICES}
    train_by_stratum: dict[str, list[str]] = {"uk": [], "en": [], "code": []}
    validation_by_stratum: dict[str, list[str]] = {"uk": [], "en": [], "code": []}
    for name, records in strata.items():
        for index, text in records:
            target = validation_by_stratum if index in heldout else train_by_stratum
            target[name].append(text)

    if {name: len(items) for name, items in validation_by_stratum.items()} != {
        "uk": 1, "en": 1, "code": 1
    }:
        raise Learn01Error("held-out split must contain exactly one record per stratum")
    if any(not values for values in train_by_stratum.values()):
        raise Learn01Error("every training stratum must remain non-empty")

    train_entries = [
        {"stratum": name, "text": text}
        for name in ("uk", "en", "code")
        for text in train_by_stratum[name]
    ]
    validation_entries = [
        {"stratum": name, "text": text}
        for name in ("uk", "en", "code")
        for text in validation_by_stratum[name]
    ]
    train_hashes = {_canonical_hash(item) for item in train_entries}
    validation_hashes = {_canonical_hash(item) for item in validation_entries}
    if train_hashes & validation_hashes:
        raise Learn01Error("train/held-out record overlap detected")

    manifest = {
        "schema": "12-6.learn01-corpus-split.v1",
        "source": CORPUS_PATH.as_posix(),
        "source_sha256": EXPECTED_CORPUS_SHA256,
        "source_authority": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
        "external_sources_approved": 0,
        "holdout_line_indices_zero_based": list(HOLDOUT_INDICES),
        "train": [
            {
                "stratum": item["stratum"],
                "utf8_bytes": len(item["text"].encode("utf-8")),
                "sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
            }
            for item in train_entries
        ],
        "validation": [
            {
                "stratum": item["stratum"],
                "utf8_bytes": len(item["text"].encode("utf-8")),
                "sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
            }
            for item in validation_entries
        ],
    }
    manifest["identity_sha256"] = _canonical_hash(manifest)
    return {
        "manifest": manifest,
        "train_by_stratum": train_by_stratum,
        "validation_by_stratum": validation_by_stratum,
        "all_train_texts": [item["text"] for item in train_entries],
        "all_validation_texts": [item["text"] for item in validation_entries],
    }


def _train_tokenizer(split: dict[str, Any], output_dir: Path):
    train_text = "\n".join(split["all_train_texts"]) + "\n"
    derived_train_path = output_dir / "train-split.txt"
    derived_train_path.write_text(train_text, encoding="utf-8")
    train_sha = sha256_file(derived_train_path)
    split_manifest_hash = str(split["manifest"]["identity_sha256"])
    manifest = TokenizerTrainingManifest(
        experiment_id="learn01-real-100k-bpe-v1",
        algorithm="bpe",
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id="data10-project-authored-uk-en-code-learn01-split-v1",
        dataset_manifest_sha256=split_manifest_hash,
        corpus_files=(
            CorpusFileIdentity(
                "learn01/train-split.txt",
                train_sha,
                len(derived_train_path.read_bytes()),
            ),
        ),
        vocab_size=512,
        min_frequency=2,
    )
    first = train_hf_tokenizer(manifest, split["all_train_texts"])
    second = train_hf_tokenizer(manifest, split["all_train_texts"])
    if first.artifact_identity != second.artifact_identity:
        raise Learn01Error("BPE tokenizer artifact identity is not repeatable")
    if first.vocab_size < 257 or first.vocab_size > 512:
        raise Learn01Error(f"unexpected BPE vocabulary size: {first.vocab_size}")

    validation_token_ids: list[list[int]] = []
    unknowns = 0
    for text in split["all_validation_texts"]:
        ids = first.encode(text)
        validation_token_ids.append(ids)
        unknowns += sum(token_id == first.unk_id for token_id in ids)
        if first.decode(ids, skip_special_tokens=False, errors="strict") != text:
            raise Learn01Error("BPE tokenizer failed strict held-out round trip")
    if unknowns:
        raise Learn01Error(f"BPE tokenizer emitted {unknowns} held-out unknown tokens")

    tokenizer_json = first._tokenizer.to_str()  # experiment artifact retention
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_json, encoding="utf-8")
    if sha256_file(tokenizer_path) != first.artifact_identity.tokenizer_json_sha256:
        raise Learn01Error("retained tokenizer JSON does not match artifact identity")

    byte_baseline = sum(len(text.encode("utf-8")) for text in split["all_validation_texts"])
    heldout_tokens = sum(len(ids) for ids in validation_token_ids)
    return first, manifest, {
        "algorithm": "bytelevel_bpe",
        "requested_vocab_size": 512,
        "actual_vocab_size": first.vocab_size,
        "version": TOKENIZERS_VERSION,
        "config_sha256": first.artifact_identity.config_sha256,
        "vocab_sha256": first.artifact_identity.vocab_sha256,
        "tokenizer_json_sha256": first.artifact_identity.tokenizer_json_sha256,
        "training_manifest_sha256": manifest.sha256,
        "repeat_build_identity_equal": True,
        "heldout_unknown_tokens": unknowns,
        "heldout_strict_roundtrip": True,
        "heldout_tokens": heldout_tokens,
        "heldout_utf8_bytes": byte_baseline,
        "heldout_token_reduction_vs_bytes": 1.0 - heldout_tokens / byte_baseline,
    }


def _model_spec(repo_root: Path, vocab_size: int) -> tuple[ModelSpec, dict[str, Any]]:
    canonical_s1 = load_stage_config(repo_root / "configs/stages/s1_100k.json")
    payload = canonical_s1.model.to_dict()
    payload["vocab_size"] = vocab_size
    spec = ModelSpec.from_dict(payload)
    if not 95_000 <= spec.parameter_count() <= 110_000:
        raise Learn01Error(
            f"evidence-bound BPE S1 geometry left ~100K envelope: {spec.parameter_count()}"
        )
    geometry = {
        "canonical_s1_model_identity_sha256": canonical_s1.model.identity_sha256(),
        "canonical_s1_expected_parameters": canonical_s1.expected_parameters,
        "canonical_s1_vocab_size": canonical_s1.model.vocab_size,
        "only_experimental_geometry_change": "vocab_size",
    }
    return spec, geometry


def _trainer_config(*, max_steps: int, seed: int, learning_rate: float) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _encoded_stream(tokenizer: Any, texts: list[str]) -> list[int]:
    stream = tokenizer.encode("\n".join(texts) + "\n")
    if len(stream) < 2:
        raise Learn01Error("training stratum encoded to fewer than two tokens")
    return stream


def _make_batch(
    stream: list[int],
    *,
    occurrence: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    width = batch_size * sequence_length
    base = (occurrence * width) % len(stream)
    rows = [
        [stream[(base + row * sequence_length + offset) % len(stream)]
         for offset in range(sequence_length)]
        for row in range(batch_size)
    ]
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def _evaluate_texts(
    model: TwelveSixDecoder,
    tokenizer: Any,
    texts_by_stratum: dict[str, list[str]],
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_predicted_bytes = 0
    by_stratum: dict[str, dict[str, float | int]] = {}

    for stratum in ("uk", "en", "code"):
        stratum_nll = 0.0
        stratum_tokens = 0
        stratum_bytes = 0
        for text in texts_by_stratum[stratum]:
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                raise Learn01Error(f"{stratum} evaluation record has fewer than two tokens")
            start = 0
            while start < len(ids) - 1:
                chunk = ids[start : start + model.spec.max_seq_len]
                if len(chunk) < 2:
                    break
                input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
                logits = model(input_ids).logits
                nll = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                targets = chunk[1:]
                predicted_text = tokenizer.decode(
                    targets, skip_special_tokens=False, errors="strict"
                )
                predicted_bytes = len(predicted_text.encode("utf-8"))
                token_count = len(targets)
                stratum_nll += float(nll.item())
                stratum_tokens += token_count
                stratum_bytes += predicted_bytes
                start += model.spec.max_seq_len - 1
        if stratum_tokens <= 0 or stratum_bytes <= 0:
            raise Learn01Error(f"{stratum} evaluation has no measurable targets")
        by_stratum[stratum] = {
            "loss": stratum_nll / stratum_tokens,
            "bits_per_byte": stratum_nll / math.log(2.0) / stratum_bytes,
            "tokens": stratum_tokens,
            "predicted_utf8_bytes": stratum_bytes,
        }
        total_nll += stratum_nll
        total_tokens += stratum_tokens
        total_predicted_bytes += stratum_bytes

    model.train(was_training)
    return {
        "loss": total_nll / total_tokens,
        "bits_per_byte": total_nll / math.log(2.0) / total_predicted_bytes,
        "tokens": total_tokens,
        "predicted_utf8_bytes": total_predicted_bytes,
        "by_stratum": by_stratum,
    }


def _weight_norm(model: TwelveSixDecoder) -> float:
    total = sum(
        float(torch.sum(parameter.detach().float() ** 2).item())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return math.sqrt(total)


def _snapshot_parameters(model: TwelveSixDecoder) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def _update_weight_ratio(
    model: TwelveSixDecoder, before: list[torch.Tensor], before_norm: float
) -> float:
    squared = 0.0
    index = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().float() - before[index].float()
        squared += float(torch.sum(delta * delta).item())
        index += 1
    if index != len(before):
        raise Learn01Error("trainable parameter set changed during optimization")
    return math.sqrt(squared) / max(before_norm, 1e-30)


@torch.no_grad()
def _generation_snapshot(
    model: TwelveSixDecoder,
    tokenizer: Any,
    *,
    optimizer_step: int,
    optimized_tokens: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    prompts = {
        "uk": "Українська ",
        "en": "The training ",
        "code": "def ",
    }
    outputs: dict[str, Any] = {}
    for name, prompt in prompts.items():
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        generated = model.generate(input_ids, max_new_tokens=48, do_sample=False)
        ids = generated[0].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False, errors="strict")
        outputs[name] = {
            "prompt": prompt,
            "token_ids": ids,
            "text": text,
            "replacement_characters": text.count("\ufffd"),
        }
    model.train(was_training)
    return {
        "optimizer_step": optimizer_step,
        "optimized_tokens": optimized_tokens,
        "decoding": "greedy",
        "max_new_tokens": 48,
        "outputs": outputs,
    }


def _lock_identity(repo_root: Path) -> dict[str, str]:
    paths = (TOOLCHAIN_LOCK_PATH, RUNTIME_LOCK_PATH, TOKENIZER_LOCK_PATH)
    hashes = {path.as_posix(): sha256_file(repo_root / path) for path in paths}
    return {
        "combined_sha256": _canonical_hash(hashes),
        **hashes,
    }


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    tokenizer: Any,
    split_manifest_hash: str,
    run_manifest_hash: str,
    config: TrainerConfig,
    trainer: Trainer,
    environment_lock_hash: str,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.artifact_identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.artifact_identity.vocab_sha256,
        dataset_manifest_hash=split_manifest_hash,
        run_manifest_hash=run_manifest_hash,
        training_config=asdict(config),
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=environment_lock_hash,
    )


def run_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    output_dir: Path,
    max_steps: int = 1000,
    batch_size: int = 8,
    sequence_length: int = 64,
    eval_every: int = 50,
    seed: int = 1337,
    learning_rate: float = 3e-4,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if max_steps < 4:
        raise ValueError("max_steps must be >= 4")
    if batch_size <= 0 or not 2 <= sequence_length <= 256:
        raise ValueError("invalid batch_size/sequence_length")
    if eval_every <= 0 or max_steps % eval_every != 0:
        raise ValueError("eval_every must divide max_steps")
    if max_steps != 1000 and GENERATION_STEPS - {0, max_steps}:
        # Custom smoke runs simply snapshot init/final; canonical run keeps four snapshots.
        generation_steps = frozenset({0, max_steps})
    else:
        generation_steps = GENERATION_STEPS
    _require_source_sha(source_sha, repo_root)
    _read_recipe(repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    split = _split_corpus(repo_root)
    split_manifest_path = output_dir / "split-manifest.json"
    split_manifest_path.write_text(
        json.dumps(split["manifest"], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_split_path = output_dir / "validation-split.txt"
    validation_split_path.write_text(
        "\n".join(split["all_validation_texts"]) + "\n", encoding="utf-8"
    )

    tokenizer, tokenizer_manifest, tokenizer_evidence = _train_tokenizer(split, output_dir)
    spec, s1_geometry = _model_spec(repo_root, tokenizer.vocab_size)
    init_spec = InitSpec()
    config = _trainer_config(
        max_steps=max_steps, seed=seed, learning_rate=learning_rate
    )
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")
    lock_identity = _lock_identity(repo_root)

    streams = {
        name: _encoded_stream(tokenizer, split["train_by_stratum"][name])
        for name in ("uk", "en", "code")
    }
    unique_train_tokens = {name: len(stream) for name, stream in streams.items()}
    occurrence = {"uk": 0, "en": 0, "code": 0}
    optimized_tokens_by_stratum = {"uk": 0, "en": 0, "code": 0}
    tokens_per_step = batch_size * (sequence_length - 1)

    run_manifest = {
        "schema": "12-6.learn01-run-manifest.v1",
        "source_sha": source_sha,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "tokenizer_config_sha256": tokenizer.artifact_identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.artifact_identity.vocab_sha256,
        "tokenizer_training_manifest_sha256": tokenizer_manifest.sha256,
        "dataset_manifest_sha256": split["manifest"]["identity_sha256"],
        "trainer_config": asdict(config),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "mixture_pattern": list(MIXTURE_PATTERN),
        "environment_lock_sha256": lock_identity["combined_sha256"],
        "paid_compute": False,
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
    }
    run_manifest_hash = _canonical_hash(run_manifest)

    initial_train_eval = _evaluate_texts(
        model, tokenizer, split["train_by_stratum"]
    )
    initial_validation = _evaluate_texts(
        model, tokenizer, split["validation_by_stratum"]
    )
    eval_curve: list[dict[str, Any]] = [
        {
            "optimizer_step": 0,
            "optimized_tokens": 0,
            "train": initial_train_eval,
            "validation": initial_validation,
        }
    ]
    generation_snapshots = [
        _generation_snapshot(
            model, tokenizer, optimizer_step=0, optimized_tokens=0
        )
    ]
    train_curve: list[dict[str, Any]] = []
    checkpoint_events: list[dict[str, Any]] = []
    best_validation_loss = float(initial_validation["loss"])
    best_step = 0
    best_tokens = 0
    best_checkpoint_dir = output_dir / "best-checkpoint"

    train_wall = 0.0
    for step_index in range(max_steps):
        stratum = MIXTURE_PATTERN[step_index % len(MIXTURE_PATTERN)]
        batch = _make_batch(
            streams[stratum],
            occurrence=occurrence[stratum],
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        occurrence[stratum] += 1
        before_norm = _weight_norm(model)
        before = _snapshot_parameters(model)
        started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        elapsed = time.perf_counter() - started
        train_wall += elapsed
        optimized_tokens_by_stratum[stratum] += metrics.tokens
        ratio = _update_weight_ratio(model, before, before_norm)
        train_curve.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "stratum": stratum,
                "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
                "grad_norm": metrics.grad_norm,
                "update_weight_ratio": ratio,
                "learning_rate": metrics.learning_rate,
                "step_seconds": elapsed,
                "tokens_per_second": metrics.tokens / elapsed,
                "cumulative_train_tokens_per_second": trainer.tokens_seen / train_wall,
            }
        )

        if metrics.optimizer_step in generation_steps and metrics.optimizer_step != 0:
            generation_snapshots.append(
                _generation_snapshot(
                    model,
                    tokenizer,
                    optimizer_step=metrics.optimizer_step,
                    optimized_tokens=trainer.tokens_seen,
                )
            )

        if metrics.optimizer_step % eval_every == 0:
            train_eval = _evaluate_texts(
                model, tokenizer, split["train_by_stratum"]
            )
            validation = _evaluate_texts(
                model, tokenizer, split["validation_by_stratum"]
            )
            point = {
                "optimizer_step": metrics.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "train": train_eval,
                "validation": validation,
            }
            eval_curve.append(point)
            if float(validation["loss"]) < best_validation_loss:
                best_validation_loss = float(validation["loss"])
                best_step = trainer.optimizer_step
                best_tokens = trainer.tokens_seen
                identity = _checkpoint_identity(
                    source_sha=source_sha,
                    spec=spec,
                    tokenizer=tokenizer,
                    split_manifest_hash=str(split["manifest"]["identity_sha256"]),
                    run_manifest_hash=run_manifest_hash,
                    config=config,
                    trainer=trainer,
                    environment_lock_hash=lock_identity["combined_sha256"],
                )
                checkpoint_started = time.perf_counter()
                save_trainer_checkpoint(
                    best_checkpoint_dir,
                    model=model,
                    trainer=trainer,
                    identity=identity,
                    overwrite=True,
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                checkpoint_events.append(
                    {
                        "kind": "best",
                        "optimizer_step": trainer.optimizer_step,
                        "optimized_tokens": trainer.tokens_seen,
                        "validation_loss": best_validation_loss,
                        "seconds": checkpoint_seconds,
                    }
                )

    if trainer.optimizer_step != max_steps:
        raise Learn01Error("training did not reach configured max_steps")
    if best_step <= 0 or not best_validation_loss < float(initial_validation["loss"]):
        raise Learn01Error(
            "held-out validation did not improve over random initialization"
        )

    final_checkpoint_dir = output_dir / "final-checkpoint"
    final_identity = _checkpoint_identity(
        source_sha=source_sha,
        spec=spec,
        tokenizer=tokenizer,
        split_manifest_hash=str(split["manifest"]["identity_sha256"]),
        run_manifest_hash=run_manifest_hash,
        config=config,
        trainer=trainer,
        environment_lock_hash=lock_identity["combined_sha256"],
    )
    final_checkpoint_started = time.perf_counter()
    save_trainer_checkpoint(
        final_checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=final_identity,
        overwrite=True,
    )
    final_checkpoint_seconds = time.perf_counter() - final_checkpoint_started
    checkpoint_events.append(
        {
            "kind": "final",
            "optimizer_step": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "validation_loss": float(eval_curve[-1]["validation"]["loss"]),
            "seconds": final_checkpoint_seconds,
        }
    )
    verify_checkpoint(final_checkpoint_dir)
    verify_checkpoint(best_checkpoint_dir)

    # Prove a fresh model+trainer can reload the strongest retained checkpoint.
    reload_model = TwelveSixDecoder(spec, init_spec)
    reload_trainer = Trainer(reload_model, config, device="cpu")
    reload_started = time.perf_counter()
    load_result = load_trainer_checkpoint(
        best_checkpoint_dir,
        model=reload_model,
        trainer=reload_trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tokenizer.artifact_identity.config_sha256,
        expected_dataset_manifest_hash=str(split["manifest"]["identity_sha256"]),
    )
    reload_seconds = time.perf_counter() - reload_started
    reloaded_validation = _evaluate_texts(
        reload_model, tokenizer, split["validation_by_stratum"]
    )
    if reload_trainer.optimizer_step != best_step or reload_trainer.tokens_seen != best_tokens:
        raise Learn01Error("reloaded trainer counters do not match best checkpoint")
    if not math.isclose(
        float(reloaded_validation["loss"]),
        best_validation_loss,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise Learn01Error("best checkpoint reload changed held-out validation loss")

    final_validation = eval_curve[-1]["validation"]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "base_pr": 173,
            "base_head_sha": "077205ef2b1662a5029bc77b8fc762078cabeb17",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
            "environment_locks": lock_identity,
        },
        "truth_boundary": {
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "external_training_sources_approved": 0,
            "project_authored_synthetic_only": True,
            "representative_corpus_evidence": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
        },
        "data": {
            "recipe_path": RECIPE_PATH.as_posix(),
            "corpus_path": CORPUS_PATH.as_posix(),
            "corpus_sha256": EXPECTED_CORPUS_SHA256,
            "split_manifest_sha256": split["manifest"]["identity_sha256"],
            "train_records": len(split["all_train_texts"]),
            "validation_records": len(split["all_validation_texts"]),
            "heldout_line_indices_zero_based": list(HOLDOUT_INDICES),
            "unique_train_utf8_bytes": sum(
                len(text.encode("utf-8")) for text in split["all_train_texts"]
            ),
            "validation_utf8_bytes": sum(
                len(text.encode("utf-8")) for text in split["all_validation_texts"]
            ),
            "unique_train_tokens_by_stratum": unique_train_tokens,
            "recycled_training_stream": True,
            "scientific_boundary": (
                "The only currently admitted corpus is tiny project-authored synthetic "
                "UK/EN/code mechanics data. It is deliberately recycled for optimization "
                "evidence; language quality and broad generalization are not established."
            ),
        },
        "tokenizer": tokenizer_evidence,
        "model": {
            "model_spec": spec.to_dict(),
            "model_identity_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init_spec.to_dict(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "random_init": True,
            "s1_geometry_reused": True,
            "s1_geometry_source": s1_geometry,
            "canonical_s1_freeze": False,
        },
        "training": {
            "trainer_config": asdict(config),
            "run_manifest_sha256": run_manifest_hash,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens_per_optimizer_step": tokens_per_step,
            "optimized_tokens": trainer.tokens_seen,
            "optimized_tokens_by_stratum": optimized_tokens_by_stratum,
            "optimized_token_mixture_fraction": {
                name: optimized_tokens_by_stratum[name] / trainer.tokens_seen
                for name in ("uk", "en", "code")
            },
            "train_wall_seconds_excluding_eval_checkpoint_generation": train_wall,
            "optimized_tokens_per_train_second": trainer.tokens_seen / train_wall,
            "train_curve": train_curve,
            "eval_curve": eval_curve,
        },
        "validation_gate": {
            "initial_loss": float(initial_validation["loss"]),
            "initial_bits_per_byte": float(initial_validation["bits_per_byte"]),
            "best_loss": best_validation_loss,
            "best_optimizer_step": best_step,
            "best_optimized_tokens": best_tokens,
            "final_loss": float(final_validation["loss"]),
            "final_bits_per_byte": float(final_validation["bits_per_byte"]),
            "absolute_best_loss_improvement": float(initial_validation["loss"]) - best_validation_loss,
            "relative_best_loss_improvement": (
                float(initial_validation["loss"]) - best_validation_loss
            ) / float(initial_validation["loss"]),
            "required_validation_improvement": True,
            "passed": True,
        },
        "generation_snapshots": generation_snapshots,
        "checkpoint": {
            "events": checkpoint_events,
            "best_directory": "best-checkpoint",
            "final_directory": "final-checkpoint",
            "best_reload_seconds": reload_seconds,
            "best_reload_optimizer_step": reload_trainer.optimizer_step,
            "best_reload_tokens_seen": reload_trainer.tokens_seen,
            "best_reload_validation_loss": float(reloaded_validation["loss"]),
            "best_manifest_format": load_result.manifest.get("format"),
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise Learn01Error("report schema/authority mismatch")
    source = report.get("source")
    if not isinstance(source, dict):
        raise Learn01Error("report source must be an object")
    if expected_source_sha is not None and source.get("git_sha") != expected_source_sha:
        raise Learn01Error("report source SHA mismatch")
    model = report.get("model")
    tokenizer = report.get("tokenizer")
    training = report.get("training")
    gate = report.get("validation_gate")
    truth = report.get("truth_boundary")
    checkpoint = report.get("checkpoint")
    for value, name in (
        (model, "model"),
        (tokenizer, "tokenizer"),
        (training, "training"),
        (gate, "validation_gate"),
        (truth, "truth_boundary"),
        (checkpoint, "checkpoint"),
    ):
        if not isinstance(value, dict):
            raise Learn01Error(f"report {name} must be an object")
    if not 95_000 <= int(model["parameter_count"]) <= 110_000:
        raise Learn01Error("report model left ~100K parameter envelope")
    if model.get("random_init") is not True or model.get("canonical_s1_freeze") is not False:
        raise Learn01Error("model truth boundary drifted")
    if tokenizer.get("repeat_build_identity_equal") is not True:
        raise Learn01Error("tokenizer repeatability was not proven")
    if int(tokenizer.get("heldout_unknown_tokens", -1)) != 0:
        raise Learn01Error("held-out tokenizer unknown-token gate failed")
    if gate.get("passed") is not True or gate.get("required_validation_improvement") is not True:
        raise Learn01Error("validation improvement gate did not pass")
    if not float(gate["best_loss"]) < float(gate["initial_loss"]):
        raise Learn01Error("best held-out loss did not improve")
    if float(checkpoint["best_reload_validation_loss"]) != float(gate["best_loss"]):
        if not math.isclose(
            float(checkpoint["best_reload_validation_loss"]),
            float(gate["best_loss"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise Learn01Error("checkpoint reload validation drift")
    if truth.get("external_training_sources_approved") != 0:
        raise Learn01Error("external corpus truth boundary drifted")
    for key in (
        "foreign_pretrained_weights",
        "instruction_tuning",
        "representative_corpus_evidence",
        "stage_freeze",
        "promotion_authority",
        "paid_compute_authority",
    ):
        if truth.get(key) is not False:
            raise Learn01Error(f"truth boundary {key} was weakened")
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unsigned):
        raise Learn01Error("report self-hash mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--max-steps", type=int, default=1000)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--sequence-length", type=int, default=64)
    run.add_argument("--eval-every", type=int, default=50)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--torch-threads", type=int, default=2)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            eval_every=args.eval_every,
            seed=args.seed,
            learning_rate=args.learning_rate,
            torch_threads=args.torch_threads,
        )
        validate_report(report, expected_source_sha=args.source_sha)
        summary = {
            "parameter_count": report["model"]["parameter_count"],
            "optimized_tokens": report["training"]["optimized_tokens"],
            "initial_validation_loss": report["validation_gate"]["initial_loss"],
            "best_validation_loss": report["validation_gate"]["best_loss"],
            "final_validation_loss": report["validation_gate"]["final_loss"],
            "tokens_per_second": report["training"]["optimized_tokens_per_train_second"],
            "report_sha256": report["report_sha256"],
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("report must be a JSON object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
