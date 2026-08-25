"""Controlled ~100K context 128-vs-256 training experiment.

The experiment reuses the incumbent ModelSpec/RoPE/cache implementation and the
versioned ContextPackingSpec. It never mutates canonical S0/S1 configs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.context_scaling import (
    ContextPackingSpec,
    context_probe_spec,
    measure_context_candidate_packing,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

SCHEMA = "12-6.context-100k-experiment.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_EVIDENCE_NOT_CANONICAL_CONTEXT_CHANGE"
IGNORE_INDEX = -100


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path, split: str) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        record_id = payload.get("id")
        text = payload.get("text")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number} invalid id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} invalid text")
        records.append(TextRecord(record_id=record_id, text=text, split=split))
    if not records:
        raise ValueError(f"{path} contains no records")
    return tuple(records)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _packing_stats(
    records: tuple[TextRecord, ...],
    tokenizer: ByteTokenizer,
    *,
    packing_spec: ContextPackingSpec,
    dataset_id: str,
    dataset_identity_sha256: str,
    source_jsonl_sha256: str,
    split: str,
) -> dict[str, Any]:
    measurement = measure_context_candidate_packing(
        records,
        tokenizer,
        packing_spec=packing_spec,
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity_sha256,
        source_jsonl_sha256=source_jsonl_sha256,
        split=split,
    )
    token_lengths = [len(tokenizer.encode(record.text)) for record in records]
    stride = packing_spec.sequence_length - 1
    multi_window_docs = sum(
        1 for length in token_lengths if length > packing_spec.sequence_length
    )
    tail_waste = measurement.packed_capacity_token_count - measurement.packed_input_token_count
    return {
        **measurement.to_dict(),
        "measurement_identity_sha256": measurement.identity_sha256(),
        "documents_truncated": 0,
        "documents_requiring_multiple_windows": multi_window_docs,
        "window_stride": stride,
        "tail_padding_tokens": tail_waste,
        "tail_padding_fraction": (
            tail_waste / measurement.packed_capacity_token_count
            if measurement.packed_capacity_token_count
            else 0.0
        ),
    }


def _tensorize_example(example: Any, *, take_loss_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    if take_loss_tokens <= 0 or take_loss_tokens > example.num_loss_tokens:
        raise ValueError("invalid take_loss_tokens")
    input_ids = torch.tensor(example.input_ids, dtype=torch.long).unsqueeze(0)
    labels = list(example.labels)
    keep = take_loss_tokens
    for target_index in range(1, len(labels)):
        if labels[target_index] == IGNORE_INDEX:
            continue
        if keep > 0:
            keep -= 1
        else:
            labels[target_index] = IGNORE_INDEX
    if keep != 0:
        raise RuntimeError("failed to select requested loss-token count")
    return input_ids, torch.tensor(labels, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    records: tuple[TextRecord, ...],
    tokenizer: ByteTokenizer,
    *,
    sequence_length: int,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    examples = iter_packed_examples(
        records,
        tokenizer,
        expected_split=records[0].split,
        sequence_length=sequence_length,
        fill_token_id=0,
        ignore_index=IGNORE_INDEX,
        add_bos=False,
        add_eos=False,
        cross_document=False,
    )
    for example in examples:
        input_ids = torch.tensor(example.input_ids, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(example.labels, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        nll = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        valid = int(shifted_labels.ne(IGNORE_INDEX).sum().item())
        total_nll += float(nll.item())
        total_tokens += valid
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("held-out split produced no causal targets")
    nats_per_byte = total_nll / total_tokens
    return {
        "loss_nats_per_byte": nats_per_byte,
        "bpb": nats_per_byte / math.log(2.0),
        "causal_tokens": total_tokens,
    }


def _grad_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum().cpu()
    return float(total.sqrt().item())


def run_context_condition(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    context_length: int,
    output_path: Path,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected MODEL-17 schema")
    contexts = tuple(int(value) for value in payload["contexts"])
    if contexts != (128, 256) or context_length not in contexts:
        raise ValueError("MODEL-17 requires exactly contexts 128 and 256")

    stage = load_stage_config(repo_root / payload["stage_config"])
    spec = context_probe_spec(stage.model, max_seq_len=context_length)
    if spec.parameter_count() != stage.model.parameter_count():
        raise RuntimeError("context candidate changed parameter count")
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = _load_records(train_path, "train")
    validation_records = _load_records(validation_path, "validation")
    train_ids = {record.record_id for record in train_records}
    validation_ids = {record.record_id for record in validation_records}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation overlap")

    packing_spec = ContextPackingSpec(sequence_length=context_length)
    train_packing = _packing_stats(
        train_records,
        tokenizer,
        packing_spec=packing_spec,
        dataset_id=manifest["dataset_id"],
        dataset_identity_sha256=manifest["dataset_identity_sha256"],
        source_jsonl_sha256=manifest["outputs"]["train.jsonl"],
        split="train",
    )
    validation_packing = _packing_stats(
        validation_records,
        tokenizer,
        packing_spec=packing_spec,
        dataset_id=manifest["dataset_id"],
        dataset_identity_sha256=manifest["dataset_identity_sha256"],
        source_jsonl_sha256=manifest["outputs"]["validation.jsonl"],
        split="validation",
    )

    training = payload["training"]
    seed = int(training["seed"])
    token_budget = int(training["optimized_token_budget"])
    tokens_per_update = int(training["optimized_tokens_per_update"])
    if token_budget <= 0 or tokens_per_update <= 0 or token_budget % tokens_per_update:
        raise ValueError("token budget must be a positive multiple of tokens per update")
    torch_threads = int(training["torch_threads"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, stage.init)
    initial_state_sha256 = _state_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    clip_norm = float(training["gradient_clip_norm"])
    initial_native = _evaluate(
        model, validation_records, tokenizer, sequence_length=context_length
    )
    initial_common_128 = _evaluate(
        model, validation_records, tokenizer, sequence_length=128
    )

    packed_train = list(
        iter_packed_examples(
            train_records,
            tokenizer,
            expected_split="train",
            sequence_length=context_length,
            fill_token_id=0,
            ignore_index=IGNORE_INDEX,
            add_bos=False,
            add_eos=False,
            cross_document=False,
        )
    )
    if not packed_train:
        raise RuntimeError("training packing emitted no examples")

    cursor = 0
    optimized_tokens = 0
    processed_input_tokens = 0
    optimizer_updates = 0
    training_loss_per_update: list[float] = []
    grad_norms: list[float] = []
    clip_factors: list[float] = []
    update_seconds: list[float] = []
    examples_per_update: list[int] = []

    model.train()
    while optimized_tokens < token_budget:
        optimizer.zero_grad(set_to_none=True)
        remaining_update = min(tokens_per_update, token_budget - optimized_tokens)
        target_update_tokens = remaining_update
        update_loss_sum = 0.0
        used_examples = 0
        started = time.perf_counter()
        while remaining_update > 0:
            example = packed_train[cursor % len(packed_train)]
            cursor += 1
            take = min(example.num_loss_tokens, remaining_update)
            input_ids, labels = _tensorize_example(example, take_loss_tokens=take)
            logits = model(input_ids).logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            if not bool(torch.isfinite(loss_sum).item()):
                raise RuntimeError("non-finite training loss")
            (loss_sum / target_update_tokens).backward()
            update_loss_sum += float(loss_sum.detach().item())
            remaining_update -= take
            processed_input_tokens += context_length
            used_examples += 1

        grad_norm = _grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        clip_factor = min(1.0, clip_norm / max(grad_norm, 1e-30))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        update_seconds.append(time.perf_counter() - started)
        grad_norms.append(grad_norm)
        clip_factors.append(clip_factor)
        examples_per_update.append(used_examples)
        training_loss_per_update.append(update_loss_sum / target_update_tokens)
        optimized_tokens += target_update_tokens
        optimizer_updates += 1

    final_native = _evaluate(
        model, validation_records, tokenizer, sequence_length=context_length
    )
    final_common_128 = _evaluate(
        model, validation_records, tokenizer, sequence_length=128
    )
    tail = training_loss_per_update[-min(8, len(training_loss_per_update)) :]
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "config_sha256": _sha256_file(config_path),
        "context_length": context_length,
        "canonical_stage": stage.stage,
        "canonical_stage_model_identity_sha256": stage.model.identity_sha256(),
        "candidate_model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "initial_state_sha256": initial_state_sha256,
        "tokenizer_identity": asdict(tokenizer.identity),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "packing_identity_sha256": packing_spec.identity_sha256(
            tokenizer_config_sha256=tokenizer.identity.config_sha256
        ),
        "train_packing": train_packing,
        "validation_packing": validation_packing,
        "training": {
            "optimized_tokens": optimized_tokens,
            "optimized_tokens_per_update": tokens_per_update,
            "optimizer_updates": optimizer_updates,
            "processed_input_tokens": processed_input_tokens,
            "processed_input_tokens_per_optimized_token": (
                processed_input_tokens / optimized_tokens
            ),
            "examples_per_update_mean": sum(examples_per_update) / len(examples_per_update),
            "training_loss_first": training_loss_per_update[0],
            "training_loss_last": training_loss_per_update[-1],
            "training_loss_tail_mean": sum(tail) / len(tail),
            "pre_clip_grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "pre_clip_grad_norm_max": max(grad_norms),
            "clip_fraction": sum(value < 1.0 for value in clip_factors) / len(clip_factors),
            "mean_update_seconds": sum(update_seconds) / len(update_seconds),
            "seconds_per_optimized_token": sum(update_seconds) / optimized_tokens,
        },
        "evaluation": {
            "initial_native": initial_native,
            "initial_common_128": initial_common_128,
            "final_native": final_native,
            "final_common_128": final_common_128,
            "native_minus_common_128_bpb": (
                float(final_native["bpb"]) - float(final_common_128["bpb"])
            ),
        },
        "memory": {
            "peak_rss_kib_isolated_process": peak_rss_kib,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
        "truth_boundary": {
            "paid_compute_used": False,
            "canonical_s0_packing_changed": False,
            "canonical_stage_config_changed": False,
            "dataset_is_tiny_project_fixture": True,
        },
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def compare_context_conditions(
    context_128: dict[str, Any],
    context_256: dict[str, Any],
) -> dict[str, Any]:
    if int(context_128["context_length"]) != 128 or int(context_256["context_length"]) != 256:
        raise ValueError("comparison requires 128 then 256")
    invariant_fields = (
        "source_sha",
        "parameters",
        "initial_state_sha256",
        "dataset_manifest_sha256",
        "dataset_identity_sha256",
    )
    for field in invariant_fields:
        if context_128[field] != context_256[field]:
            raise RuntimeError(f"context comparison invariant drift: {field}")
    t128 = context_128["training"]
    t256 = context_256["training"]
    if t128["optimized_tokens"] != t256["optimized_tokens"]:
        raise RuntimeError("optimized token counts differ")
    if t128["optimizer_updates"] != t256["optimizer_updates"]:
        raise RuntimeError("optimizer update counts differ")

    e128 = context_128["evaluation"]
    e256 = context_256["evaluation"]
    p128 = context_128["train_packing"]
    p256 = context_256["train_packing"]
    common_delta = float(e256["final_common_128"]["bpb"]) - float(
        e128["final_common_128"]["bpb"]
    )
    long_dependency_delta = float(e256["final_native"]["bpb"]) - float(
        e256["final_common_128"]["bpb"]
    )
    native_delta = float(e256["final_native"]["bpb"]) - float(e128["final_native"]["bpb"])
    utilization_delta = float(p256["causal_pair_utilization"]) - float(
        p128["causal_pair_utilization"]
    )
    compute_ratio = float(t256["seconds_per_optimized_token"]) / float(
        t128["seconds_per_optimized_token"]
    )

    primary = 256 if native_delta < 0.0 else 128
    recommendation = {
        "primary_100k_research_context_on_this_fixture": primary,
        "native_bpb_delta_256_minus_128": native_delta,
        "common_128_eval_bpb_delta_256train_minus_128train": common_delta,
        "within_256_model_native_minus_common128_bpb": long_dependency_delta,
        "packing_utilization_delta_256_minus_128": utilization_delta,
        "seconds_per_token_ratio_256_over_128": compute_ratio,
        "interpretation": (
            "native delta mixes trained-context and evaluation-context effects; "
            "common-128 delta isolates training under different packing/context at a common "
            "evaluation horizon; within-256 native-minus-common128 is the bounded estimate "
            "of benefit from seeing longer validation dependencies. Packing and compute "
            "terms are reported separately."
        ),
        "canonical_context_changed": False,
        "promotion_authorized": False,
    }
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": context_128["source_sha"],
        "conditions": {"128": context_128, "256": context_256},
        "recommendation": recommendation,
        "truth_boundary": {
            "current_corpus_is_tiny_controlled_fixture": True,
            "intended_large_corpus_not_available": True,
            "recommendation_is_primary_research_context_not_stage_freeze": True,
            "paid_compute_used": False,
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report
