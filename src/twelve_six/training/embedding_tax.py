"""Cross-scale embedding tax and real-tokenizer rebalanced training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization.experiments import train_hf_tokenizer
from twelve_six.tokenization.real_experiments import _dataset_contract, _training_manifest
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import Trainer
from twelve_six.vocabulary import rebalance_d_ff_for_vocabulary, vocabulary_cost

SCHEMA = "12-6.model15-embedding-tax.v1"
RETAINED_BPE = {
    "requested_vocab_size": 512,
    "actual_vocab_size": 472,
    "training_manifest_sha256": "5612fbff21c49d6a2215d031413d24f96078151a9de1a5d9b21976bb6cb8a23b",
    "tokenizer_json_sha256": "006c84fc0d05d3bedb5b0bceb587aab1631dd0295cc2063e97823c2121e08be0",
    "vocab_sha256": "b7c695b64993c468449d6143f289952f42c2bea8bc06265503a482d13ed78f6e",
    "config_sha256": "d511f76c308ff989e05de45718e6192a1e835a9886189c993f7999fbe3616d7d",
}
TARGETS = (100_000, 250_000, 500_000, 1_000_000, 10_000_000)


class EmbeddingTaxError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def geometry_profiles(root: Path) -> dict[int, ModelSpec]:
    s1 = load_stage_config(root / "configs/stages/s1_100k.json").model
    s2 = load_stage_config(root / "configs/stages/s2_1m.json").model
    s3 = load_stage_config(root / "configs/stages/s3_10m.json").model
    p250 = replace(
        s1,
        vocab_size=512,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=16,
        rope_rotary_dim=16,
        d_ff=160,
        max_seq_len=256,
    )
    p500 = replace(
        s1,
        vocab_size=512,
        d_model=96,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=24,
        rope_rotary_dim=24,
        d_ff=256,
        max_seq_len=256,
    )
    return {100_000: s1, 250_000: p250, 500_000: p500, 1_000_000: s2, 10_000_000: s3}


def tokenizer_candidates_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if int(report["requested_vocab_size"]) != RETAINED_BPE["requested_vocab_size"]:
        raise EmbeddingTaxError("unexpected requested tokenizer vocabulary")
    algorithms = report["algorithms"]
    bpe = algorithms["bpe"]
    unigram = algorithms["unigram"]
    bpe_artifact = bpe["artifact"]
    for field in (
        "training_manifest_sha256",
        "tokenizer_json_sha256",
        "vocab_sha256",
        "config_sha256",
    ):
        if str(bpe_artifact[field]) != RETAINED_BPE[field]:
            raise EmbeddingTaxError(f"retained BPE artifact drift: {field}")
    if int(bpe_artifact["vocab_size"]) != RETAINED_BPE["actual_vocab_size"]:
        raise EmbeddingTaxError("retained BPE actual vocabulary drift")
    byte_tokens = int(bpe["held_out"]["byte_baseline_tokens"])
    return [
        {
            "id": "byte-v1",
            "algorithm": "byte",
            "requested_vocab_size": 256,
            "actual_vocab_size": 256,
            "held_out_tokens": byte_tokens,
            "token_reduction_vs_bytes": 0.0,
            "repeatability_status": "PASS",
            "strict_round_trip": True,
            "unknown_tokens": 0,
            "artifact_identity": "canonical-s0-byte-v1",
        },
        {
            "id": "bpe-request512-actual472",
            "algorithm": "bpe",
            "requested_vocab_size": int(report["requested_vocab_size"]),
            "actual_vocab_size": int(bpe_artifact["vocab_size"]),
            "held_out_tokens": int(bpe["held_out"]["tokens"]),
            "token_reduction_vs_bytes": float(bpe["held_out"]["token_reduction_vs_bytes"]),
            "repeatability_status": str(bpe["repeatability_status"]),
            "strict_round_trip": bool(bpe["held_out"]["strict_round_trip_all"]),
            "unknown_tokens": int(bpe["held_out"]["unknown_tokens"]),
            "artifact_identity": dict(bpe_artifact),
        },
        {
            "id": "unigram-request512-actual497",
            "algorithm": "unigram",
            "requested_vocab_size": int(report["requested_vocab_size"]),
            "actual_vocab_size": int(unigram["artifact"]["vocab_size"]),
            "held_out_tokens": int(unigram["held_out"]["tokens"]),
            "token_reduction_vs_bytes": float(unigram["held_out"]["token_reduction_vs_bytes"]),
            "repeatability_status": str(unigram["repeatability_status"]),
            "strict_round_trip": bool(unigram["held_out"]["strict_round_trip_all"]),
            "unknown_tokens": int(unigram["held_out"]["unknown_tokens"]),
            "artifact_identity": dict(unigram["artifact"]),
        },
    ]


def allocation_row(
    template: ModelSpec,
    *,
    target: int,
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    vocab = int(tokenizer["actual_vocab_size"])
    tied = rebalance_d_ff_for_vocabulary(
        template,
        target_parameters=target,
        vocab_size=vocab,
        tied_lm_head=True,
        d_ff_alignment=8,
    )
    untied = rebalance_d_ff_for_vocabulary(
        template,
        target_parameters=target,
        vocab_size=vocab,
        tied_lm_head=False,
        d_ff_alignment=8,
    )
    tied_breakdown = tied.model.parameter_breakdown()
    untied_same_geometry_cost = vocabulary_cost(
        tied.model, vocab_size=vocab, tied_lm_head=False
    )
    same_geometry_untied_total = (
        tied.parameter_count + untied_same_geometry_cost.lm_head_weight_parameters
    )
    return {
        "target_parameters": target,
        "tokenizer_id": tokenizer["id"],
        "algorithm": tokenizer["algorithm"],
        "requested_vocab_size": tokenizer["requested_vocab_size"],
        "actual_vocab_size": vocab,
        "repeatability_status": tokenizer["repeatability_status"],
        "held_out_tokens": tokenizer["held_out_tokens"],
        "token_reduction_vs_bytes": tokenizer["token_reduction_vs_bytes"],
        "tied": {
            "parameter_count": tied.parameter_count,
            "target_delta": tied.target_delta,
            "model_identity_sha256": tied.model.identity_sha256(),
            "d_model": tied.model.d_model,
            "n_layers": tied.model.n_layers,
            "d_ff": tied.model.d_ff,
            "embedding_parameters": tied_breakdown["token_embedding"],
            "embedding_fraction": tied_breakdown["token_embedding"] / tied.parameter_count,
            "transformer_block_parameters": tied_breakdown["blocks_total"],
            "transformer_block_fraction": tied_breakdown["blocks_total"] / tied.parameter_count,
        },
        "hypothetical_untied_same_d_ff": {
            "parameter_count": same_geometry_untied_total,
            "extra_output_matrix_parameters": vocab * tied.model.d_model,
            "total_vocabulary_parameters": 2 * vocab * tied.model.d_model,
            "vocabulary_fraction": 2 * vocab * tied.model.d_model / same_geometry_untied_total,
        },
        "untied_rebalanced_near_target": {
            "parameter_count": untied.parameter_count,
            "target_delta": untied.target_delta,
            "d_ff": untied.model.d_ff,
            "model_identity_sha256": untied.model.identity_sha256(),
        },
        "d_ff_rebalancing": {
            "tied_d_ff": tied.model.d_ff,
            "untied_d_ff": untied.model.d_ff,
            "d_ff_recovered_by_tying": tied.model.d_ff - untied.model.d_ff,
        },
    }


def _pareto(rows: Sequence[Mapping[str, Any]], target: int) -> list[str]:
    eligible = [
        row for row in rows
        if row["target_parameters"] == target
        and row["repeatability_status"] == "PASS"
    ]
    frontier = []
    for row in eligible:
        dominated = False
        for other in eligible:
            if other is row:
                continue
            no_worse = (
                other["tied"]["embedding_fraction"] <= row["tied"]["embedding_fraction"]
                and other["held_out_tokens"] <= row["held_out_tokens"]
            )
            strict = (
                other["tied"]["embedding_fraction"] < row["tied"]["embedding_fraction"]
                or other["held_out_tokens"] < row["held_out_tokens"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(str(row["tokenizer_id"]))
    return sorted(frontier)


def _reproduce_bpe(report: Mapping[str, Any]):
    dataset = _dataset_contract()
    manifest = _training_manifest(
        "bpe", dataset, vocab_size=int(report["requested_vocab_size"])
    )
    train_texts = [str(record["text"]) for record in dataset["train_records"]]
    adapter = train_hf_tokenizer(manifest, train_texts)
    artifact = adapter.artifact_identity
    expected = report["algorithms"]["bpe"]["artifact"]
    observed = {
        "training_manifest_sha256": artifact.training_manifest_sha256,
        "tokenizer_json_sha256": artifact.tokenizer_json_sha256,
        "vocab_sha256": artifact.vocab_sha256,
        "config_sha256": artifact.config_sha256,
        "vocab_size": artifact.vocab_size,
    }
    for field, value in observed.items():
        if value != expected[field]:
            raise EmbeddingTaxError(f"reproduced BPE artifact mismatch: {field}")
    return adapter, dataset, observed


def _trace(texts: Sequence[str], tokenizer: Any, *, steps: int, length: int):
    batches = []
    for index in range(steps):
        ids = tokenizer.encode(texts[index % len(texts)])[:length]
        if len(ids) < 2:
            raise EmbeddingTaxError("tokenized training record shorter than two tokens")
        if max(ids) >= tokenizer.vocab_size:
            raise EmbeddingTaxError("tokenizer emitted id outside actual vocabulary")
        tensor = torch.tensor([ids], dtype=torch.long)
        batches.append({"input_ids": tensor, "labels": tensor.clone()})
    return batches


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, batches: Sequence[Mapping[str, torch.Tensor]]) -> float:
    model.eval()
    weighted = 0.0
    tokens = 0
    for batch in batches:
        labels = batch["labels"]
        n = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(model(batch["input_ids"]).logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite validation loss")
        weighted += float(loss.item()) * n
        tokens += n
    return weighted / tokens


def _train_scale(
    spec: ModelSpec,
    *,
    init_spec: Any,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    validation_batches: Sequence[Mapping[str, torch.Tensor]],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=len(train_batches),
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=seed,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    validation = [{"optimizer_step": 0, "loss": _evaluate(model, validation_batches)}]
    training = []
    times = []
    for step, batch in enumerate(train_batches, 1):
        start = time.perf_counter()
        metric = trainer.train_microbatch(batch)
        times.append(time.perf_counter() - start)
        if metric.grad_norm is None or not math.isfinite(metric.grad_norm):
            raise RuntimeError("non-finite/missing gradient norm")
        training.append(
            {
                "optimizer_step": metric.optimizer_step,
                "loss": metric.loss,
                "grad_norm": metric.grad_norm,
                "tokens": metric.tokens,
            }
        )
        if step % 2 == 0 or step == len(train_batches):
            validation.append({"optimizer_step": step, "loss": _evaluate(model, validation_batches)})
    return {
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "d_ff": spec.d_ff,
        "training_curve": training,
        "validation_curve": validation,
        "optimized_tokens": trainer.tokens_seen,
        "step_seconds_mean": sum(times) / len(times),
        "numerically_stable": all(
            math.isfinite(point["loss"]) and math.isfinite(point["grad_norm"])
            for point in training
        ) and all(math.isfinite(point["loss"]) for point in validation),
    }


def run(
    repo_root: str | Path,
    *,
    source_sha: str,
    tokenizer_report_path: str | Path,
    steps: int = 4,
    sequence_length: int = 96,
    seed: int = 20260825,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    report = json.loads(Path(tokenizer_report_path).read_text(encoding="utf-8"))
    tokenizers = tokenizer_candidates_from_report(report)
    profiles = geometry_profiles(root)
    rows = [
        allocation_row(profiles[target], target=target, tokenizer=tokenizer)
        for target in TARGETS
        for tokenizer in tokenizers
    ]
    bpe, dataset, reproduced = _reproduce_bpe(report)
    if bpe.vocab_size != 472:
        raise EmbeddingTaxError("training must bind actual BPE vocabulary 472")
    init_spec = load_stage_config(root / "configs/stages/s1_100k.json").init
    train_texts = [str(record["text"]) for record in dataset["train_records"]]
    validation_texts = [str(record["text"]) for record in dataset["validation_records"]]
    executed = {}
    for target in (100_000, 500_000, 1_000_000):
        allocation = next(
            row for row in rows
            if row["target_parameters"] == target
            and row["tokenizer_id"] == "bpe-request512-actual472"
        )
        template = profiles[target]
        rebalanced = rebalance_d_ff_for_vocabulary(
            template,
            target_parameters=target,
            vocab_size=bpe.vocab_size,
            tied_lm_head=True,
            d_ff_alignment=8,
        )
        if rebalanced.model.identity_sha256() != allocation["tied"]["model_identity_sha256"]:
            raise RuntimeError("allocation/training ModelSpec identity mismatch")
        length = min(sequence_length, rebalanced.model.max_seq_len)
        train_batches = _trace(train_texts, bpe, steps=steps, length=length)
        validation_batches = _trace(
            validation_texts, bpe, steps=len(validation_texts), length=length
        )
        executed[str(target)] = _train_scale(
            rebalanced.model,
            init_spec=init_spec,
            train_batches=train_batches,
            validation_batches=validation_batches,
            seed=seed,
        )
    bands = {
        "100000": [256, 512],
        "250000": [320, 768],
        "500000": [384, 1024],
        "1000000": [512, 1536],
        "10000000": [2048, 6144],
    }
    payload = {
        "schema": SCHEMA,
        "authority": "PARAMETER_ALLOCATION_AND_SHORT_LOCAL_FREE_TRAINING_NOT_TOKENIZER_FREEZE",
        "source_sha": source_sha,
        "live_tokenizer_report": {
            "source_sha": report["source"]["source_sha"],
            "evidence_sha256": report["evidence_sha256"],
            "requested_vocab_size": report["requested_vocab_size"],
            "candidates": tokenizers,
            "reproduced_bpe_artifact": reproduced,
        },
        "allocation_rows": rows,
        "pareto_by_scale": {str(target): _pareto(rows, target) for target in TARGETS},
        "executed_bpe472_rebalanced_training": executed,
        "recommended_vocabulary_search_bands": {
            "bands": bands,
            "basis": (
                "MODEL-37 measured 100K/1M/10M bands retained; 250K/500K are "
                "intermediate parameter-tax search bands, not tokenizer selections"
            ),
        },
        "claims": {
            "requested_vocab_treated_as_actual": False,
            "tokenizer_frozen_from_parameter_efficiency": False,
            "unigram_repeatability_block_respected": True,
            "paid_compute_used": False,
            "canonical_s0_changed": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tokenizer-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args(argv)
    payload = run(
        args.repo_root,
        source_sha=args.source_sha,
        tokenizer_report_path=args.tokenizer_report,
        steps=args.steps,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
