"""DATA-106 boundary study plus MILESTONE-100 LOCAL_FREE learned-Base runner.

The runner composes incumbent DATA-21/22 intake, D04 packing, D01 model,
D02 Trainer, D05 checkpoints, and D07 first-party generation. It is evidence
orchestration, not a replacement training/data/inference framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
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
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.data.source_intake import run_bounded_intake
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.generation import generate
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import (
    TextRecord,
    batch_examples,
    collate_rows,
    iter_packed_examples,
)
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from twelve_six.tokenization.special_tokens import (
    EXPERIMENTAL_CONFIG_SHA256,
    EXPERIMENTAL_TOKENIZER_VERSION,
    EXPERIMENTAL_VOCAB_SHA256,
    ExperimentalByteEosTokenizer,
)
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.data106-document-boundaries.v1"
MILESTONE_SCHEMA = "12-6.milestone100-first-learned-base.v1"
AUTHORITY = "LOCAL_FREE_BASE_PRETRAINING_EVIDENCE_NOT_PROMOTION"
SEED = 1337
SEQ = 64
BATCH = 4
COMPARE_TOKENS = 16_384
PHASE1_TOKENS = 65_536
FINAL_TOKENS = 131_072
CHECKPOINTS = (32_768, 65_536, 98_304, 131_072)
PROMPTS = ("Україна ", "The ", "def ")
TRAIN_SUFFIXES = ("d23314.htm", "8-typography.rst")
VALID_SUFFIXES = ("9-metadata.rst",)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hash(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _spec(vocab: int) -> ModelSpec:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=vocab,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        head_dim=16,
        d_ff=256,
        rope_rotary_dim=16,
    )
    expected = 467_808 if vocab == 256 else 467_904
    if spec.parameter_count() != expected:
        raise RuntimeError("LEARN03 468K geometry drifted")
    return spec


def _trainer_config(max_steps: int) -> TrainerConfig:
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


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _trainer_hash(trainer: Trainer) -> str:
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value, default=str, sort_keys=True).encode())

    visit(trainer.state_dict())
    return digest.hexdigest()


def _common_init(vocab: int) -> TwelveSixDecoder:
    init = InitSpec()
    random.seed(SEED)
    torch.manual_seed(SEED)
    base = TwelveSixDecoder(_spec(256), init)
    if vocab == 256:
        return base
    torch.manual_seed(SEED)
    expanded = TwelveSixDecoder(_spec(257), init)
    source = base.state_dict()
    target = expanded.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for name, dst in target.items():
        src = source[name]
        if src.shape == dst.shape:
            merged[name] = src
        elif (
            src.ndim >= 1
            and dst.shape[0] == src.shape[0] + 1
            and dst.shape[1:] == src.shape[1:]
        ):
            value = dst.clone()
            value[: src.shape[0]].copy_(src)
            merged[name] = value
        else:
            raise RuntimeError(f"unexpected vocab expansion shape: {name}")
    expanded.load_state_dict(merged)
    return expanded


def _prepare_corpus(root: Path, out: Path) -> dict[str, Any]:
    intake = out / "intake"
    if intake.exists():
        shutil.rmtree(intake)
    registry_path = root / "configs/data/external_source_candidates_ua_en_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest = run_bounded_intake(
        registry,
        intake,
        max_download_bytes=2_000_000,
        max_normalized_chars=50_000,
    )
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for row in manifest["records"]:
        if row.get("status") != "ACCEPTED":
            continue
        url = str(row["acquisition_url"])
        if any(url.endswith(suffix) for suffix in TRAIN_SUFFIXES):
            target = train
        elif any(url.endswith(suffix) for suffix in VALID_SUFFIXES):
            target = valid
        else:
            raise RuntimeError(f"accepted object lacks predeclared split: {url}")
        text = (intake / row["text_path"]).read_text(encoding="utf-8").rstrip("\n")
        target.append(
            {
                "id": str(row["id"]),
                "text": text,
                "language": str(row["language"]),
                "content_sha256": str(row["content_sha256"]),
                "acquisition_url": url,
            }
        )
    if len(train) != 2 or len(valid) != 1:
        raise RuntimeError(f"expected 2 train/1 valid docs, got {len(train)}/{len(valid)}")
    train_ids = {row["id"] for row in train}
    valid_ids = {row["id"] for row in valid}
    if train_ids & valid_ids:
        raise RuntimeError("train/validation record overlap")
    split = {
        "schema": "12-6.data106-split.v1",
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "registry_identity_sha256": manifest["candidate_registry_identity_sha256"],
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(valid_ids),
        "train_content_sha256": sorted(row["content_sha256"] for row in train),
        "validation_content_sha256": sorted(row["content_sha256"] for row in valid),
        "train_bytes": sum(len(row["text"].encode()) for row in train),
        "validation_bytes": sum(len(row["text"].encode()) for row in valid),
        "validation_never_passed_to_training_packer": True,
        "reserved_split_boundary_crossing_possible": False,
        "representativeness_boundary": (
            "REAL_RIGHTS_APPROVED_BOUNDED_UK_EN_SAMPLE_NOT_BROAD_REPRESENTATIVE_CORPUS"
        ),
    }
    split["split_manifest_sha256"] = _hash(split)
    return {"manifest": manifest, "train": train, "valid": valid, "split": split}


def _reload_corpus(out: Path) -> dict[str, Any]:
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "intake/manifest.json").read_text(encoding="utf-8"))
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for row in manifest["records"]:
        if row.get("status") != "ACCEPTED":
            continue
        url = str(row["acquisition_url"])
        if any(url.endswith(suffix) for suffix in TRAIN_SUFFIXES):
            target = train
        elif any(url.endswith(suffix) for suffix in VALID_SUFFIXES):
            target = valid
        else:
            continue
        text = (out / "intake" / row["text_path"]).read_text(encoding="utf-8")
        target.append({"id": str(row["id"]), "text": text.rstrip("\n")})
    return {"train": train, "valid": valid, "split": comparison["corpus"]}


def _tokenizer(arm: str):
    if arm == "B_eos_packed":
        return ExperimentalByteEosTokenizer()
    return ByteTokenizer()


def _examples(arm: str, docs: list[dict[str, Any]]):
    tokenizer = _tokenizer(arm)
    records = [TextRecord(row["id"], row["text"], "train") for row in docs]
    eos = arm == "B_eos_packed"
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=SEQ,
            add_eos=eos,
            cross_document=eos,
        )
    )
    return tokenizer, examples


def _packing_stats(arm: str, tokenizer: Any, examples, docs) -> dict[str, Any]:
    source_bytes = sum(len(row["text"].encode()) for row in docs)
    unique = sum(len(tokenizer.encode(row["text"])) for row in docs)
    if tokenizer.eos_id is not None:
        unique += len(docs)
    actual = sum(sum(example.attention_mask) for example in examples)
    valid = sum(example.num_loss_tokens for example in examples)
    slots = len(examples) * SEQ
    loss_slots = len(examples) * (SEQ - 1)
    return {
        "source_bytes": source_bytes,
        "unique_tokens_including_eos": unique,
        "tokens_per_source_byte": unique / source_bytes,
        "packed_examples": len(examples),
        "packing_utilization": valid / loss_slots,
        "padding_waste_fraction": (slots - actual) / slots,
        "packed_input_tokens_per_source_byte": actual / source_bytes,
        "cross_document_attention_allowed": arm == "B_eos_packed",
        "attention_reset_at_eos": False,
    }


def _tensor_batch(batch, remaining: int | None = None) -> dict[str, torch.Tensor]:
    rows = collate_rows(batch, target_mode="target_ids")
    inputs = torch.tensor(rows["input_ids"], dtype=torch.long)
    targets = torch.tensor(rows["target_ids"], dtype=torch.long)
    mask = torch.tensor(rows["loss_mask"], dtype=torch.long)
    if remaining is not None:
        positions = torch.nonzero((targets != -100) & mask.bool(), as_tuple=False)
        for row, column in positions[remaining:].tolist():
            targets[row, column] = -100
            mask[row, column] = 0
    return {"input_ids": inputs, "target_ids": targets, "loss_mask": mask}


def _batches(examples, trainer: Trainer, target_tokens: int):
    batches = list(batch_examples(examples, batch_size=BATCH, drop_last=False))
    index = trainer.micro_step
    while trainer.tokens_seen < target_tokens:
        remaining = target_tokens - trainer.tokens_seen
        yield _tensor_batch(batches[index % len(batches)], remaining)
        index += 1


@torch.no_grad()
def _evaluate(model, tokenizer, docs) -> dict[str, Any]:
    before = _model_hash(model)
    was_training = model.training
    model.eval()
    nll = 0.0
    targets = 0
    eos_nll = 0.0
    eos_targets = 0
    try:
        for row in docs:
            ids = tokenizer.encode(row["text"])
            start = 0
            while start < len(ids) - 1:
                chunk = ids[start : start + model.spec.max_seq_len]
                if len(chunk) < 2:
                    break
                tensor = torch.tensor([chunk], dtype=torch.long)
                logits = model(tensor).logits
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    tensor[:, 1:].reshape(-1),
                    reduction="sum",
                )
                nll += float(loss.item())
                targets += len(chunk) - 1
                start += model.spec.max_seq_len - 1
            if tokenizer.eos_id is not None and ids:
                context = torch.tensor([ids[-model.spec.max_seq_len :]], dtype=torch.long)
                logits = model(context).logits[0, -1].unsqueeze(0)
                target = torch.tensor([tokenizer.eos_id])
                eos_nll += float(F.cross_entropy(logits, target).item())
                eos_targets += 1
    finally:
        model.train(was_training)
    after = _model_hash(model)
    if before != after or targets <= 0:
        raise RuntimeError("evaluation mutation or empty validation")
    return {
        "held_out_bpb": (nll / targets) / math.log(2.0),
        "mean_byte_loss_nats": nll / targets,
        "source_byte_targets": targets,
        "eos_boundary_loss_nats": eos_nll / eos_targets if eos_targets else None,
        "evaluation_model_state_non_mutation": True,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
    }


@torch.no_grad()
def _boundary_probe(model, tokenizer, docs) -> dict[str, Any]:
    first = tokenizer.encode(docs[0]["text"])
    second = tokenizer.encode(docs[1]["text"])
    eos_loss = None
    prefix = first
    if tokenizer.eos_id is not None:
        context = torch.tensor([first[-model.spec.max_seq_len :]], dtype=torch.long)
        logits = model(context).logits[0, -1].unsqueeze(0)
        eos_loss = float(
            F.cross_entropy(logits, torch.tensor([tokenizer.eos_id])).item()
        )
        prefix = [*first, tokenizer.eos_id]
    context = torch.tensor([prefix[-model.spec.max_seq_len :]], dtype=torch.long)
    logits = model(context).logits[0, -1].unsqueeze(0)
    transition = float(F.cross_entropy(logits, torch.tensor([second[0]])).item())
    return {
        "document_boundary_prediction_loss_nats": eos_loss,
        "cross_document_transition_loss_nats": transition,
        "conditioned_on_explicit_eos": tokenizer.eos_id is not None,
        "attention_reset_at_boundary": False,
    }


def _generations(model, tokenizer) -> list[dict[str, Any]]:
    backend = S0TorchInferenceBackend(model, tokenizer)
    backend.eos_token_id = tokenizer.eos_id
    snapshots = []
    for prompt in PROMPTS:
        result = generate(
            backend,
            prompt,
            GenerationConfig(max_new_tokens=32, sample=False),
        )
        snapshots.append(
            {
                "prompt": prompt,
                "token_ids": list(result.generated_token_ids),
                "text": result.text,
                "stop_reason": result.stop_reason,
                "backend": "D07/S0TorchInferenceBackend",
            }
        )
    return snapshots


def _identity(source_sha, spec, tokenizer, trainer, dataset_hash, run_hash):
    config = trainer.config
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=dataset_hash,
        run_manifest_hash=run_hash,
        training_config={
            "trainer": asdict(config),
            "data": {
                "sequence_length": SEQ,
                "batch_size": BATCH,
                "tokenizer_version": tokenizer.identity.version,
            },
        },
        seed=SEED,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        scheduler=None,
    )


def _run_arm(
    arm: str,
    train_docs,
    valid_docs,
    target_tokens: int,
    source_sha: str,
    dataset_hash: str,
    run_hash: str,
    checkpoint_root: Path | None = None,
    checkpoint_budgets=(),
    resume: Path | None = None,
):
    tokenizer, examples = _examples(arm, train_docs)
    valid_ids = {row["id"] for row in valid_docs}
    packed_ids = {rid for example in examples for rid in example.record_ids}
    overlap = sorted(valid_ids & packed_ids)
    if overlap:
        raise RuntimeError(f"validation entered training packing: {overlap}")
    model = _common_init(tokenizer.vocab_size)
    spec = model.spec
    max_steps = math.ceil(FINAL_TOKENS / (BATCH * (SEQ - 1))) + 16
    trainer = Trainer(model, _trainer_config(max_steps), device="cpu")
    if resume is not None:
        load_trainer_checkpoint(
            resume,
            model=model,
            trainer=trainer,
            restore_rng=True,
            expected_git_sha=source_sha,
            expected_model_spec_hash=spec.identity_sha256(),
            expected_tokenizer_hash=tokenizer.identity.config_sha256,
            expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
            expected_dataset_manifest_hash=dataset_hash,
            expected_run_manifest_hash=run_hash,
            expected_seed=SEED,
        )
    trainer_before = _trainer_hash(trainer)
    initial_eval = _evaluate(model, tokenizer, valid_docs)
    if trainer_before != _trainer_hash(trainer):
        raise RuntimeError("evaluation mutated trainer state")
    initial_generation = _generations(model, tokenizer)
    curve = []
    saved = []
    budgets = list(checkpoint_budgets)
    budget_index = 0
    started = time.perf_counter()
    for batch in _batches(examples, trainer, target_tokens):
        metrics = trainer.train_microbatch(batch)
        if metrics.update_loss is None or metrics.grad_norm is None:
            continue
        curve.append(
            {
                "step": metrics.optimizer_step,
                "tokens": trainer.tokens_seen,
                "loss": metrics.update_loss,
                "grad_norm": metrics.grad_norm,
                "would_clip": metrics.grad_norm > 1.0,
            }
        )
        while budget_index < len(budgets) and trainer.tokens_seen >= budgets[budget_index]:
            if checkpoint_root is None:
                raise RuntimeError("checkpoint root missing")
            budget = budgets[budget_index]
            path = checkpoint_root / f"tokens-{budget}"
            save_trainer_checkpoint(
                path,
                model=model,
                trainer=trainer,
                identity=_identity(
                    source_sha,
                    spec,
                    tokenizer,
                    trainer,
                    dataset_hash,
                    run_hash,
                ),
            )
            saved.append(
                {
                    "budget": budget,
                    "tokens": trainer.tokens_seen,
                    "step": trainer.optimizer_step,
                    "path": str(path),
                    "manifest_sha256": sha256_file(path / "manifest.json"),
                }
            )
            budget_index += 1
    elapsed = time.perf_counter() - started
    if trainer.tokens_seen != target_tokens:
        raise RuntimeError("optimized valid-token budget drift")
    trainer_before = _trainer_hash(trainer)
    final_eval = _evaluate(model, tokenizer, valid_docs)
    if trainer_before != _trainer_hash(trainer):
        raise RuntimeError("final evaluation mutated trainer state")
    losses = [float(row["loss"]) for row in curve]
    window = min(8, len(losses))
    first_loss = sum(losses[:window]) / window
    last_loss = sum(losses[-window:]) / window
    grads = [float(row["grad_norm"]) for row in curve]
    report = {
        "arm": arm,
        "seed": SEED,
        "random_initialization": True,
        "model_spec": spec.to_dict(),
        "parameter_count": spec.parameter_count(),
        "parameter_delta_vs_467808": spec.parameter_count() - 467_808,
        "tokenizer": {
            "version": tokenizer.identity.version,
            "config_sha256": tokenizer.identity.config_sha256,
            "vocab_sha256": tokenizer.identity.vocab_sha256,
            "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id,
        },
        "packing": {
            **_packing_stats(arm, tokenizer, examples, train_docs),
            "reserved_validation_ids_seen": overlap,
            "no_reserved_split_boundary_crossing": True,
        },
        "optimized_valid_tokens": trainer.tokens_seen,
        "train_loss_first_window": first_loss,
        "train_loss_last_window": last_loss,
        "train_loss_decreased": last_loss < first_loss,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "held_out_bpb_decreased": (
            final_eval["held_out_bpb"] < initial_eval["held_out_bpb"]
        ),
        "boundary_probe": _boundary_probe(model, tokenizer, train_docs),
        "gradient_behavior": {
            "mean_grad_norm": sum(grads) / len(grads),
            "max_grad_norm": max(grads),
            "clip_frequency": sum(value > 1.0 for value in grads) / len(grads),
            "nonfinite_count": sum(not math.isfinite(value) for value in grads),
        },
        "throughput": {
            "valid_tokens_per_second": target_tokens / elapsed,
            "wall_seconds": elapsed,
        },
        "generation_before": initial_generation,
        "generation_after": _generations(model, tokenizer),
        "checkpoints": saved,
        "final_model_state_sha256": _model_hash(model),
        "final_trainer_state_sha256": _trainer_hash(trainer),
    }
    return report, model, trainer, tokenizer


def _machine(root: Path) -> dict[str, Any]:
    memory = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        parts = meminfo.read_text(encoding="utf-8").splitlines()[0].split()
        if len(parts) > 1:
            memory = int(parts[1]) * 1024
    return {
        "git_sha": _git_head(root),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "paid_compute": False,
    }


def compare(root: Path, out: Path) -> dict[str, Any]:
    source_sha = _git_head(root)
    corpus = _prepare_corpus(root, out)
    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "seed": SEED,
        "sequence_length": SEQ,
        "batch_size": BATCH,
        "optimized_valid_tokens_per_arm": COMPARE_TOKENS,
        "model_geometry": _spec(256).to_dict(),
        "optimizer": asdict(_trainer_config(999_999)),
        "train_ids": corpus["split"]["train_ids"],
        "validation_ids": corpus["split"]["validation_ids"],
        "tokenizer_incumbent": {
            "version": BYTE_TOKENIZER_VERSION,
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
        },
        "tokenizer_eos": {
            "version": EXPERIMENTAL_TOKENIZER_VERSION,
            "config_sha256": EXPERIMENTAL_CONFIG_SHA256,
            "vocab_sha256": EXPERIMENTAL_VOCAB_SHA256,
        },
    }
    run_hash = _hash(run_manifest)
    arms = []
    for arm in ("A_strict_isolation", "B_eos_packed", "C_incumbent_control"):
        result, _, _, _ = _run_arm(
            arm,
            corpus["train"],
            corpus["valid"],
            COMPARE_TOKENS,
            source_sha,
            corpus["split"]["split_manifest_sha256"],
            run_hash,
        )
        arms.append(result)
    a, b, c = arms
    ac_equal = (
        a["final_model_state_sha256"] == c["final_model_state_sha256"]
        and a["train_loss_last_window"] == c["train_loss_last_window"]
        and a["final_eval"] == c["final_eval"]
        and a["generation_after"] == c["generation_after"]
    )
    if not ac_equal:
        raise RuntimeError("A and incumbent-control C failed exact replay")
    eligible = [
        arm
        for arm in (a, b)
        if arm["train_loss_decreased"]
        and arm["held_out_bpb_decreased"]
        and arm["gradient_behavior"]["nonfinite_count"] == 0
    ]
    if eligible:
        winner = min(eligible, key=lambda arm: arm["final_eval"]["held_out_bpb"])
        rule = "LOWEST_HELD_OUT_BPB_AMONG_VALID_ARMS"
    else:
        winner = a
        rule = "INCONCLUSIVE_FAIL_CLOSED_TO_INCUMBENT"
    recommendation = {
        "selected_arm": winner["arm"],
        "decision_rule": rule,
        "a_c_replay_exact": True,
        "eos_is_attention_reset": False,
        "parameter_comparison": {
            "vocab256": 467_808,
            "vocab257": 467_904,
            "delta": 96,
            "delta_fraction": 96 / 467_808,
            "geometry_rebalanced": False,
        },
        "migration_rules": {
            "vocab256_to_vocab257_resume": "FORBIDDEN_FAIL_CLOSED",
            "reason": (
                "ModelSpec/tokenizer hashes and tied embedding shape change; start scratch."
            ),
            "eos_attention_reset": False,
            "chat_or_instruction_tokens": False,
        },
    }
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "paid_compute": False,
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
        "corpus": corpus["split"],
        "run_manifest": run_manifest,
        "run_manifest_sha256": run_hash,
        "arms": arms,
        "recommendation": recommendation,
        "truth_boundary": {
            "genuine_real_rights_approved_corpus": True,
            "representative_corpus_requirement_met": False,
            "reason": corpus["split"]["representativeness_boundary"],
            "broad_intelligence_claim": False,
        },
    }
    report["report_sha256"] = _hash(report)
    _write(out / "comparison.json", report)
    _write(out / "machine_manifest.json", _machine(root))
    return report


def phase1(root: Path, out: Path) -> dict[str, Any]:
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    corpus = _reload_corpus(out)
    selected = comparison["recommendation"]["selected_arm"]
    final_manifest = {
        "schema": MILESTONE_SCHEMA,
        "source_sha": _git_head(root),
        "selected_arm": selected,
        "seed": SEED,
        "phase1_tokens": PHASE1_TOKENS,
        "final_tokens": FINAL_TOKENS,
        "checkpoint_budgets": list(CHECKPOINTS),
        "comparison_sha256": comparison["report_sha256"],
        "dataset_sha256": corpus["split"]["split_manifest_sha256"],
    }
    run_hash = _hash(final_manifest)
    result, _, _, _ = _run_arm(
        selected,
        corpus["train"],
        corpus["valid"],
        PHASE1_TOKENS,
        _git_head(root),
        corpus["split"]["split_manifest_sha256"],
        run_hash,
        out / "final_checkpoints",
        (32_768, 65_536),
    )
    report = {
        "schema": MILESTONE_SCHEMA,
        "phase": "fresh_scratch_phase1",
        "final_run_manifest": final_manifest,
        "final_run_manifest_sha256": run_hash,
        "training": result,
    }
    _write(out / "phase1.json", report)
    return report


def resume(root: Path, out: Path) -> dict[str, Any]:
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    first = json.loads((out / "phase1.json").read_text(encoding="utf-8"))
    corpus = _reload_corpus(out)
    selected = comparison["recommendation"]["selected_arm"]
    source_sha = _git_head(root)
    run_hash = first["final_run_manifest_sha256"]
    result, model, trainer, tokenizer = _run_arm(
        selected,
        corpus["train"],
        corpus["valid"],
        FINAL_TOKENS,
        source_sha,
        corpus["split"]["split_manifest_sha256"],
        run_hash,
        out / "final_checkpoints",
        (98_304, 131_072),
        out / "final_checkpoints/tokens-65536",
    )
    retained = out / "final_checkpoints/tokens-131072"
    spec = _spec(tokenizer.vocab_size)
    fresh_model = TwelveSixDecoder(spec, InitSpec())
    max_steps = math.ceil(FINAL_TOKENS / (BATCH * (SEQ - 1))) + 16
    fresh_trainer = Trainer(fresh_model, _trainer_config(max_steps), device="cpu")
    load_trainer_checkpoint(
        retained,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=False,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=corpus["split"]["split_manifest_sha256"],
        expected_run_manifest_hash=run_hash,
        expected_seed=SEED,
    )
    reload_equal = (
        _model_hash(fresh_model) == _model_hash(model)
        and fresh_trainer.optimizer_step == trainer.optimizer_step
        and fresh_trainer.tokens_seen == trainer.tokens_seen
    )
    if not reload_equal:
        raise RuntimeError("retained checkpoint fresh reload mismatch")
    trainer_before = _trainer_hash(fresh_trainer)
    final_eval = _evaluate(fresh_model, tokenizer, corpus["valid"])
    if trainer_before != _trainer_hash(fresh_trainer):
        raise RuntimeError("retained evaluation mutated trainer")
    paths = [out / "final_checkpoints" / f"tokens-{n}" for n in CHECKPOINTS]
    if not all(path.exists() for path in paths):
        raise RuntimeError("required checkpoint set incomplete")
    milestone = {
        "schema": MILESTONE_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "selected_arm": selected,
        "random_initialization": True,
        "exact_parameter_count": spec.parameter_count(),
        "real_rights_approved_corpus": True,
        "representative_corpus_requirement_met": False,
        "representativeness_truth_boundary": (
            corpus["split"]["representativeness_boundary"]
        ),
        "versioned_tokenizer": {
            "version": tokenizer.identity.version,
            "config_sha256": tokenizer.identity.config_sha256,
            "vocab_sha256": tokenizer.identity.vocab_sha256,
            "vocab_size": tokenizer.vocab_size,
        },
        "train_loss_decreased": (
            result["train_loss_last_window"]
            < first["training"]["train_loss_first_window"]
        ),
        "initial_held_out_bpb": first["training"]["initial_eval"]["held_out_bpb"],
        "final_held_out_bpb": final_eval["held_out_bpb"],
        "held_out_bpb_decreased": (
            final_eval["held_out_bpb"]
            < first["training"]["initial_eval"]["held_out_bpb"]
        ),
        "multiple_checkpoints": [
            {
                "budget": budget,
                "path": str(path),
                "manifest_sha256": sha256_file(path / "manifest.json"),
            }
            for budget, path in zip(CHECKPOINTS, paths, strict=True)
        ],
        "fresh_process_resume_from_65536": True,
        "fresh_process_retained_reload_equal": reload_equal,
        "evaluation_non_mutation": final_eval["evaluation_model_state_non_mutation"],
        "generation_before_training": first["training"]["generation_before"],
        "generation_after_training": _generations(fresh_model, tokenizer),
        "retained_exact_checkpoint": {
            "path": str(retained),
            "manifest_sha256": sha256_file(retained / "manifest.json"),
            "model_state_sha256": _model_hash(fresh_model),
        },
        "optimized_valid_tokens_total": fresh_trainer.tokens_seen,
        "optimizer_steps_total": fresh_trainer.optimizer_step,
        "final_evaluation": final_eval,
        "final_run_manifest": first["final_run_manifest"],
        "final_run_manifest_sha256": run_hash,
        "reproduction_commands": [
            "python -m twelve_six.document_boundary_experiment compare --output artifacts/data106",
            "python -m twelve_six.document_boundary_experiment phase1 --output artifacts/data106",
            "python -m twelve_six.document_boundary_experiment resume --output artifacts/data106",
        ],
        "machine_manifest": _machine(root),
        "paid_compute": False,
        "foreign_pretrained_weights": False,
        "instruction_tuning": False,
        "broad_intelligence_claim": False,
    }
    milestone["milestone_report_sha256"] = _hash(milestone)
    _write(out / "milestone100.json", milestone)
    return milestone


def validate(out: Path) -> None:
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    milestone = json.loads((out / "milestone100.json").read_text(encoding="utf-8"))
    checks = {
        "control replay": comparison["recommendation"]["a_c_replay_exact"],
        "random init": milestone["random_initialization"],
        "train loss decrease": milestone["train_loss_decreased"],
        "held-out BPB decrease": milestone["held_out_bpb_decreased"],
        "resume": milestone["fresh_process_resume_from_65536"],
        "reload": milestone["fresh_process_retained_reload_equal"],
        "evaluation non-mutation": milestone["evaluation_non_mutation"],
        "four checkpoints": len(milestone["multiple_checkpoints"]) >= 4,
        "exact token budget": milestone["optimized_valid_tokens_total"] == FINAL_TOKENS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"evidence validation failed: {failed}")
    if milestone["representative_corpus_requirement_met"] is not False:
        raise ValueError("representativeness truth boundary was overstated")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "phase1", "resume", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path("."))
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    out = args.output.resolve()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.command == "compare":
        report = compare(root, out)
        print(json.dumps(report["recommendation"], indent=2, sort_keys=True))
    elif args.command == "phase1":
        report = phase1(root, out)
        print(json.dumps({"phase": report["phase"]}, indent=2))
    elif args.command == "resume":
        report = resume(root, out)
        print(
            json.dumps(
                {
                    "milestone_report_sha256": report["milestone_report_sha256"],
                    "selected_arm": report["selected_arm"],
                    "parameters": report["exact_parameter_count"],
                    "final_held_out_bpb": report["final_held_out_bpb"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        validate(out)
        print("DATA106/MILESTONE100 validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
