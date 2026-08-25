"""One LOCAL_FREE context condition for MODEL-17.

Reuses MODEL-36 ContextPackingSpec and the incumbent decoder/RoPE path. Canonical
S0 packing remains untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import detect_git_sha, sha256_file
from .context_scaling import ContextPackingSpec, measure_context_candidate_packing
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .packing import collate_rows, iter_packed_examples, load_jsonl_records
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_TOKENIZER_VERSION, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.model17-context-candidate.v1"
TARGET_OPTIMIZED_TOKENS = 32_768
SEED = 1337
TORCH_THREADS = 2


def research_100k_spec(context: int) -> ModelSpec:
    if context not in {128, 256}:
        raise ValueError("context must be 128 or 256")
    return ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=context,
        d_model=48, n_layers=3, n_heads=4, n_kv_heads=4, head_dim=12, d_ff=128,
        activation="swiglu", norm_kind="rmsnorm", norm_placement="pre", norm_eps=1e-5,
        position_embedding="rope", rope_theta=10_000.0, rope_rotary_dim=12,
        attention_bias=False, mlp_bias=False, attention_dropout=0.0,
        final_norm=True, tie_word_embeddings=True, lm_head_bias=False,
    )


def shared_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8,
        max_steps=10_000, warmup_steps=0, scheduler="constant",
        gradient_accumulation_steps=1, gradient_clip_norm=1.0,
        precision="fp32", seed=SEED,
        deterministic_algorithms=True, deterministic_warn_only=False,
    )


def _state_digest(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _packed(records, tokenizer: ByteTokenizer, split: str, context: int):
    return tuple(iter_packed_examples(
        records, tokenizer, expected_split=split, sequence_length=context,
        fill_token_id=0, ignore_index=-100, add_bos=False, add_eos=False,
        cross_document=False,
    ))


def _batch(example, keep_tokens: int | None = None) -> dict[str, torch.Tensor]:
    rows = collate_rows([example], target_mode="target_ids")
    batch = {name: torch.tensor(value, dtype=torch.long) for name, value in rows.items()}
    if keep_tokens is not None:
        positions = torch.nonzero(batch["loss_mask"][0], as_tuple=False).flatten()
        if not 0 < keep_tokens <= int(positions.numel()):
            raise ValueError("invalid keep_tokens")
        batch["loss_mask"][0, positions[keep_tokens:]] = 0
    return batch


@torch.no_grad()
def _heldout(model, records, tokenizer, context: int) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for example in _packed(records, tokenizer, "validation", context):
        batch = _batch(example)
        targets = batch["target_ids"].clone()
        targets[~batch["loss_mask"].bool()] = -100
        logits = model(batch["input_ids"]).logits
        total_nll += float(F.cross_entropy(
            logits.reshape(-1, model.spec.vocab_size), targets.reshape(-1),
            ignore_index=-100, reduction="sum",
        ).item())
        total_tokens += int(targets.ne(-100).sum().item())
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("held-out split produced no causal targets")
    nll = total_nll / total_tokens
    return {"context": context, "causal_tokens": total_tokens, "nll": nll, "bpb": nll / math.log(2.0)}


def _packing_metrics(records, tokenizer, split: str, context: int, manifest, source_path: Path):
    packing = ContextPackingSpec(sequence_length=context)
    measured = measure_context_candidate_packing(
        records, tokenizer, packing_spec=packing,
        dataset_id=str(manifest["dataset_id"]),
        dataset_identity_sha256=str(manifest["dataset_identity_sha256"]),
        source_jsonl_sha256=sha256_file(source_path), split=split,
    )
    lengths = [len(tokenizer.encode(record.text)) for record in records]
    expected_pairs = sum(max(length - 1, 0) for length in lengths)
    if measured.causal_loss_token_count != expected_pairs:
        raise RuntimeError("packing dropped or duplicated causal pairs")
    return {
        "packing_spec": packing.to_dict(),
        "packing_identity_sha256": packing.identity_sha256(tokenizer_config_sha256=tokenizer.identity.config_sha256),
        "measurement": measured.to_dict(),
        "causal_token_utilization": measured.causal_pair_utilization,
        "padding_token_slots": measured.packed_capacity_token_count - measured.packed_input_token_count,
        "tail_pair_waste": measured.causal_pair_capacity - measured.causal_loss_token_count,
        "documents_exceeding_context": sum(length > context for length in lengths),
        "documents_hard_truncated": 0,
        "source_causal_pairs_dropped": 0,
    }


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def run_candidate(repo_root: Path, source_sha: str, context: int, output: Path) -> dict[str, Any]:
    if detect_git_sha(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch")
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    val_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = tuple(load_jsonl_records(train_path, split="train"))
    val_records = tuple(load_jsonl_records(val_path, split="validation"))
    if {r.record_id for r in train_records} & {r.record_id for r in val_records}:
        raise RuntimeError("train/validation overlap")

    spec = research_100k_spec(context)
    if spec.parameter_count() != 95_568:
        raise RuntimeError("~100K geometry drifted")
    model = TwelveSixDecoder(spec, InitSpec())
    initial_weight_digest = _state_digest(model)
    trainer = Trainer(model, shared_trainer_config(), device="cpu")
    train_packing = _packing_metrics(train_records, tokenizer, "train", context, manifest, train_path)
    val_packing = _packing_metrics(val_records, tokenizer, "validation", context, manifest, val_path)
    train_examples = _packed(train_records, tokenizer, "train", context)

    initial_native = _heldout(model, val_records, tokenizer, context)
    initial_common128 = _heldout(model, val_records, tokenizer, 128)
    step_seconds: list[float] = []
    grad_norms: list[float] = []
    train_nll_sum = 0.0
    index = 0
    started = time.perf_counter()
    while trainer.tokens_seen < TARGET_OPTIMIZED_TOKENS:
        example = train_examples[index % len(train_examples)]
        index += 1
        keep = min(example.num_loss_tokens, TARGET_OPTIMIZED_TOKENS - trainer.tokens_seen)
        batch = _batch(example, keep)
        step_started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        step_seconds.append(time.perf_counter() - step_started)
        train_nll_sum += metrics.loss * metrics.tokens
        if metrics.grad_norm is not None:
            grad_norms.append(metrics.grad_norm)
    wall = time.perf_counter() - started
    if trainer.tokens_seen != TARGET_OPTIMIZED_TOKENS:
        raise RuntimeError("optimized-token accounting drift")

    final_native = _heldout(model, val_records, tokenizer, context)
    final_common128 = _heldout(model, val_records, tokenizer, 128)
    if final_native["causal_tokens"] != final_common128["causal_tokens"]:
        raise RuntimeError("held-out causal-token count drift")

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    report = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "context": context,
        "runtime": {"device": "cpu", "torch_threads": TORCH_THREADS, "paid_compute": False},
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "trainable_parameters": spec.parameter_count(),
        "initial_weight_digest_sha256": initial_weight_digest,
        "controls": {
            "tokenizer_version": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "optimizer": asdict(shared_trainer_config()),
            "init_spec": InitSpec().to_dict(), "seed": SEED,
        },
        "packing": {"train": train_packing, "validation": val_packing},
        "training": {
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
            "mean_train_nll": train_nll_sum / trainer.tokens_seen,
            "mean_train_bpb": (train_nll_sum / trainer.tokens_seen) / math.log(2.0),
            "last_train_nll": metrics.loss,
            "step_seconds": _summary(step_seconds),
            "wall_seconds": wall,
            "seconds_per_optimized_token": wall / trainer.tokens_seen,
            "optimized_tokens_per_second": trainer.tokens_seen / wall,
            "global_grad_norm": _summary(grad_norms),
            "peak_rss_bytes": rss,
        },
        "held_out": {
            "initial_native": initial_native, "initial_common_128": initial_common128,
            "final_native": final_native, "final_common_128": final_common128,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--context", type=int, choices=(128, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_candidate(args.repo_root.resolve(), args.source_sha, args.context, args.output)
    print(json.dumps({"context": report["context"], "training": report["training"], "held_out": report["held_out"], "packing": {"train": report["packing"]["train"]}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
