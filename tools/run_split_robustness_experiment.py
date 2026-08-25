"""Run DATA-36 split sensitivity evidence on project-authored eligible text.

This is a LOCAL_FREE mechanics experiment, not a production-corpus quality claim.
No benchmark/test example is present or optimized. Alternative validation sets are
all excluded from the shared training core before either candidate is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import batch_examples, collate_rows, iter_packed_examples
from twelve_six.packing.core import TextRecord
from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    bind_split_evidence,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    pairwise_ranking_stability,
    split_sensitivity,
    verify_split_family_manifest,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import Trainer

VARIANT_SEEDS = (
    "12-6-data36-v01",
    "12-6-data36-v02",
    "12-6-data36-v03",
    "12-6-data36-v04",
)


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _synthetic_eligible_corpus() -> list[SplitRecord]:
    """Build a deterministic UK/EN/code mechanics corpus with explicit cluster lineage."""

    records: list[SplitRecord] = []
    for cluster_index in range(90):
        modality = ("uk", "en", "code")[cluster_index % 3]
        source_id = f"project-authored-data36-{modality}-v1"
        cluster_id = f"cluster-{cluster_index:03d}"
        for member in range(2):
            if modality == "uk":
                text = (
                    f"Навчальний документ {cluster_index:03d}, варіант {member}. "
                    "Цей проєктний текст описує перевірку детермінованого поділу даних, "
                    "ізоляцію споріднених документів та чесне вимірювання мовної моделі. "
                    f"Контрольна ознака {cluster_index * 17 + member}."
                )
            elif modality == "en":
                text = (
                    f"Training document {cluster_index:03d}, variant {member}. "
                    "This project-authored text describes deterministic validation partitions, "
                    "near-duplicate cluster isolation, and unbiased small-model measurement. "
                    f"Control marker {cluster_index * 19 + member}."
                )
            else:
                text = (
                    f"def data36_example_{cluster_index}_{member}(value):\n"
                    f"    offset = {cluster_index * 3 + member}\n"
                    "    # Project-authored split robustness fixture.\n"
                    "    if value < 0:\n"
                    "        return -value + offset\n"
                    "    return value + offset\n"
                )
            record_id = f"data36::{cluster_index:03d}::{member}"
            records.append(
                SplitRecord(
                    id=record_id,
                    text=text,
                    source_id=source_id,
                    modality=modality,
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    near_duplicate_cluster_id=cluster_id,
                    training_eligible=True,
                    purpose="pretraining_eligible",
                )
            )
    return records


def _tensor_batches(
    records: list[SplitRecord],
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], int]:
    text_records = [TextRecord(record.id, record.text, split) for record in records]
    examples = list(
        iter_packed_examples(
            text_records,
            tokenizer,
            expected_split=split,
            sequence_length=128,
        )
    )
    batches: list[dict[str, torch.Tensor]] = []
    loss_tokens = 0
    for example in examples:
        loss_tokens += example.num_loss_tokens
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        batches.append(
            {
                "input_ids": torch.tensor(rows["input_ids"], dtype=torch.long),
                "labels": torch.tensor(rows["labels"], dtype=torch.long),
            }
        )
    if not batches or loss_tokens <= 0:
        raise RuntimeError(f"split {split!r} produced no scoreable batches")
    return batches, loss_tokens


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, batches: list[dict[str, torch.Tensor]]) -> tuple[float, int]:
    model.eval()
    weighted = 0.0
    tokens = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        scored = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite held-out loss")
        weighted += float(loss.item()) * scored
        tokens += scored
    if tokens <= 0:
        raise RuntimeError("zero scoreable held-out tokens")
    return weighted / tokens, tokens


def _experiment_primary_spec(root: Path) -> tuple[ModelSpec, Any]:
    """Reuse the S1 shape with the incumbent 256-byte vocabulary for a ~100K control."""

    s1 = load_stage_config(root / "configs/stages/s1_100k.json")
    payload = s1.model.to_dict()
    payload["vocab_size"] = 256
    payload["max_seq_len"] = 128
    spec = ModelSpec.from_dict(payload)
    if not 90_000 <= spec.parameter_count() <= 110_000:
        raise RuntimeError("DATA-36 primary model left the declared ~100K range")
    return spec, s1.init


def _candidate_specs(root: Path) -> dict[str, tuple[ModelSpec, Any]]:
    s0 = load_stage_config(root / "configs/stages/s0_10k.json")
    primary_spec, primary_init = _experiment_primary_spec(root)
    return {
        "base_10k": (s0.model, s0.init),
        "base_96k": (primary_spec, primary_init),
    }


def _train_candidate(
    name: str,
    spec: ModelSpec,
    init_spec: Any,
    train_batches: list[dict[str, torch.Tensor]],
    validation_batches: dict[str, list[dict[str, torch.Tensor]]],
    *,
    config: TrainerConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")
    initial_train_loss, initial_train_tokens = _evaluate(model, train_batches)
    result = trainer.run(islice(cycle(train_batches), config.max_steps))
    final_train_loss, final_train_tokens = _evaluate(model, train_batches)
    if not final_train_loss < initial_train_loss:
        raise RuntimeError(f"{name}: shared-train loss did not decrease")

    per_variant: list[dict[str, Any]] = []
    optimizer_step_before_eval = trainer.optimizer_step
    for variant_id in sorted(validation_batches):
        loss, tokens = _evaluate(model, validation_batches[variant_id])
        per_variant.append(
            {
                "variant_id": variant_id,
                "loss_nats_per_byte_token": loss,
                "bits_per_byte": loss / math.log(2.0),
                "scoreable_tokens": tokens,
            }
        )
    if trainer.optimizer_step != optimizer_step_before_eval:
        raise RuntimeError("validation evaluation changed optimizer step")

    bpb = [item["bits_per_byte"] for item in per_variant]
    return {
        "candidate_id": name,
        "parameter_count": spec.parameter_count(),
        "modelspec_sha256": spec.identity_sha256(),
        "initspec_sha256": init_spec.identity_sha256(),
        "optimizer_steps": result.optimizer_steps_completed,
        "optimized_tokens": result.tokens_consumed,
        "initial_shared_train_loss": initial_train_loss,
        "final_shared_train_loss": final_train_loss,
        "initial_shared_train_eval_tokens": initial_train_tokens,
        "final_shared_train_eval_tokens": final_train_tokens,
        "validation_optimized_tokens": 0,
        "per_variant": per_variant,
        "bpb_sensitivity": split_sensitivity(bpb),
    }


def run_experiment(root: Path, *, max_steps: int = 16, batch_size: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    records = _synthetic_eligible_corpus()
    corpus_sha = eligible_corpus_identity(records)
    dedup_sha = dedup_relations_identity(records)
    family_spec = SplitFamilySpec(
        eligible_corpus_sha256=corpus_sha,
        dedup_relations_sha256=dedup_sha,
        variant_seeds=VARIANT_SEEDS,
        validation_fraction=0.10,
    )
    family = build_split_family(records, family_spec)
    verify_split_family_manifest(records, family)

    by_id = {record.id: record for record in records}
    shared_train = [by_id[record_id] for record_id in family["shared_train_record_ids"]]
    validation_union = set(family["validation_union_record_ids"])
    if {record.id for record in shared_train} & validation_union:
        raise RuntimeError("shared train core contaminated by alternative validation")

    tokenizer = ByteTokenizer()
    train_batches, train_tokens = _tensor_batches(
        shared_train, split="shared_train_core", tokenizer=tokenizer, batch_size=batch_size
    )
    validation_batches: dict[str, list[dict[str, torch.Tensor]]] = {}
    variant_token_counts: dict[str, int] = {}
    for variant in family["variants"]:
        variant_id = variant["variant_id"]
        subset = [by_id[record_id] for record_id in variant["validation_record_ids"]]
        batches, tokens = _tensor_batches(
            subset,
            split=f"validation_{variant_id}",
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        validation_batches[variant_id] = batches
        variant_token_counts[variant_id] = tokens

    config = TrainerConfig(
        learning_rate=3e-2,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=1337,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    candidates = []
    for name, (spec, init_spec) in _candidate_specs(root).items():
        candidates.append(
            _train_candidate(
                name,
                spec,
                init_spec,
                train_batches,
                validation_batches,
                config=config,
            )
        )

    candidate_metrics = {
        item["candidate_id"]: [row["bits_per_byte"] for row in item["per_variant"]]
        for item in candidates
    }
    ranking = pairwise_ranking_stability(candidate_metrics, lower_is_better=True)
    primary = next(item for item in candidates if item["candidate_id"] == "base_96k")
    payload = {
        "authority": "LOCAL_FREE_PROJECT_AUTHORED_SYNTHETIC_MECHANICS_ONLY",
        "corpus": {
            "documents": len(records),
            "near_duplicate_clusters": len({r.near_duplicate_cluster_id for r in records}),
            "modalities": {
                modality: sum(record.modality == modality for record in records)
                for modality in ("uk", "en", "code")
            },
            "project_authored": True,
            "external_sources": False,
            "benchmark_or_test_records": 0,
        },
        "split_protocol": {
            "variant_count": len(family["variants"]),
            "validation_fraction_requested": family["validation_fraction_requested"],
            "shared_train_documents": family["shared_train_documents"],
            "validation_union_documents": family["validation_union_documents"],
            "shared_train_scoreable_tokens": train_tokens,
            "validation_scoreable_tokens": variant_token_counts,
            "cluster_straddles": family["cluster_straddles_across_variants"],
            "legacy_record_hash_risk_audit": family["legacy_record_hash_risk_audit"],
        },
        "training_protocol": {
            **asdict(config),
            "batch_size_examples": batch_size,
            "sequence_length": 128,
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "training_protocol_sha256": _canonical_hash(
                {
                    "trainer": asdict(config),
                    "batch_size_examples": batch_size,
                    "sequence_length": 128,
                    "tokenizer_config_sha256": tokenizer.identity.config_sha256,
                }
            ),
        },
        "candidates": candidates,
        "primary_approximately_100k": {
            "candidate_id": primary["candidate_id"],
            "parameter_count": primary["parameter_count"],
            "bpb_sensitivity": primary["bpb_sensitivity"],
        },
        "ranking_stability": ranking,
        "claims": {
            "benchmark_or_test_optimized": False,
            "split_metrics_used_to_tune_training": False,
            "all_alternative_validation_excluded_before_training": True,
            "validation_optimizer_steps": 0,
            "paid_compute_used": False,
            "production_corpus_quality_claim": False,
        },
        "recommendation": {
            "canonical_experimental_policy": (
                "Predeclare one cluster-aware canonical split identity for ordinary model iteration; "
                "for decision-critical small-model comparisons, evaluate on a frozen family of at "
                "least four cluster-aware variants while training on their validation-union-excluded "
                "shared core. Never select or tune to the best-looking split."
            ),
            "accept_single_lucky_split_conclusions": False,
        },
    }
    evidence = bind_split_evidence(payload, family)
    return family, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-family-output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    family, evidence = run_experiment(
        root, max_steps=args.max_steps, batch_size=args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.split_family_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.split_family_output.write_text(
        json.dumps(family, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evidence_sha256": evidence["evidence_sha256"],
                "split_family_identity_sha256": family["split_family_identity_sha256"],
                "primary": evidence["primary_approximately_100k"],
                "ranking_stability": evidence["ranking_stability"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
