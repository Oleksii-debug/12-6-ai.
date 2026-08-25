"""Equal-token LOCAL_FREE context-length experiment for the live ~100K S1 geometry.

This module consumes MODEL-36's context candidate packing contract. It does not
redefine canonical S0 packing or create another RoPE/cache implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .context_scaling import (
    ContextPackingSpec,
    context_probe_spec,
    measure_context_candidate_packing,
)
from .model import ModelSpec, TwelveSixDecoder, load_stage_config
from .packing import (
    DEFAULT_IGNORE_INDEX,
    PACKING_CONFIG_HASH,
    PACKING_VERSION,
    PackedCausalExample,
    TextRecord,
    iter_packed_examples,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig

SCHEMA_CONDITION = "12-6.model17-context-condition-evidence.v1"
SCHEMA_COMPARISON = "12-6.model17-context-comparison-evidence.v1"
AUTHORITY = "LOCAL_FREE_EQUAL_OPTIMIZED_TOKENS_NOT_CONTEXT_PROMOTION"
CONFIG_PATH = Path("configs/experiments/model17_context_100k_128_256.v1.json")


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_records(path: Path, *, split: str) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        record_id = value.get("id")
        text = value.get("text")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number} has invalid id")
        if not isinstance(text, str):
            raise ValueError(f"{path}:{line_number} has invalid text")
        records.append(TextRecord(record_id=record_id, text=text, split=split))
    if not records:
        raise ValueError(f"{path} contains no records")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{path} contains duplicate record ids")
    return tuple(records)


def _initial_parameter_fingerprint(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trainer_config(config: dict[str, Any]) -> TrainerConfig:
    controls = config["controls"]
    optimizer = controls["optimizer"]
    return TrainerConfig(
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["eps"]),
        max_steps=int(optimizer["max_steps_guard"]),
        warmup_steps=0,
        scheduler=str(optimizer["scheduler"]),
        gradient_accumulation_steps=1,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        precision=str(optimizer["precision"]),
        seed=int(controls["seed"]),
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _packed_examples(
    records: tuple[TextRecord, ...],
    tokenizer: ByteTokenizer,
    *,
    split: str,
    sequence_length: int,
) -> tuple[PackedCausalExample, ...]:
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=sequence_length,
            fill_token_id=0,
            ignore_index=DEFAULT_IGNORE_INDEX,
            add_bos=False,
            add_eos=False,
            cross_document=False,
        )
    )
    if not examples:
        raise RuntimeError(f"{split} produced no packed examples")
    return examples


def _packing_diagnostics(
    records: tuple[TextRecord, ...],
    examples: tuple[PackedCausalExample, ...],
    tokenizer: ByteTokenizer,
    *,
    sequence_length: int,
) -> dict[str, Any]:
    token_lengths = [len(tokenizer.encode(record.text)) for record in records]
    total_pairs = sum(max(0, length - 1) for length in token_lengths)
    emitted_pairs = sum(example.num_loss_tokens for example in examples)
    if emitted_pairs != total_pairs:
        raise RuntimeError(
            f"context packing dropped/duplicated causal pairs: {emitted_pairs} != {total_pairs}"
        )
    input_tokens = sum(sum(example.attention_mask) for example in examples)
    capacity_tokens = len(examples) * sequence_length
    pair_capacity = len(examples) * (sequence_length - 1)
    tail_waste = capacity_tokens - input_tokens
    pair_waste = pair_capacity - emitted_pairs
    long_context_targets = sum(
        max(0, sum(example.attention_mask) - 128) for example in examples
    ) if sequence_length > 128 else 0
    return {
        "document_count": len(records),
        "documents_split_across_windows": sum(length > sequence_length for length in token_lengths),
        "single_window_truncation_tokens_avoided": sum(
            max(0, length - sequence_length) for length in token_lengths
        ),
        "actual_causal_pairs_dropped_by_windowing": total_pairs - emitted_pairs,
        "packed_example_count": len(examples),
        "packed_input_token_count": input_tokens,
        "packed_capacity_token_count": capacity_tokens,
        "padding_tail_waste_tokens": tail_waste,
        "padding_tail_waste_fraction": tail_waste / capacity_tokens,
        "causal_loss_token_count": emitted_pairs,
        "causal_pair_capacity": pair_capacity,
        "causal_pair_waste_tokens": pair_waste,
        "causal_pair_utilization": emitted_pairs / pair_capacity,
        "targets_exposed_to_more_than_127_preceding_tokens": long_context_targets,
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
    }


@torch.no_grad()
def _evaluate_bpb(
    model: TwelveSixDecoder,
    examples: tuple[PackedCausalExample, ...],
) -> tuple[float, float, int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for example in examples:
        input_ids = torch.tensor(example.input_ids, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(example.labels, dtype=torch.long).unsqueeze(0)
        logits = model(input_ids).logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        total_nll += float(
            F.cross_entropy(
                logits.reshape(-1, model.spec.vocab_size),
                targets.reshape(-1),
                ignore_index=DEFAULT_IGNORE_INDEX,
                reduction="sum",
            ).item()
        )
        total_tokens += int(targets.ne(DEFAULT_IGNORE_INDEX).sum().item())
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("held-out evaluation produced no causal targets")
    mean_nll = total_nll / total_tokens
    return mean_nll, mean_nll / math.log(2.0), total_tokens


def _crop_labels_to_remaining(example: PackedCausalExample, remaining: int) -> torch.Tensor:
    labels = torch.tensor(example.labels, dtype=torch.long).unsqueeze(0)
    valid_positions = [
        index
        for index in range(1, labels.shape[1])
        if int(labels[0, index].item()) != DEFAULT_IGNORE_INDEX
    ]
    if remaining < len(valid_positions):
        for position in valid_positions[remaining:]:
            labels[0, position] = DEFAULT_IGNORE_INDEX
    return labels


def _peak_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)


def run_condition(
    *,
    repo_root: Path,
    source_sha: str,
    context: int,
    output_path: Path,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("MODEL-17 exact-checkout mismatch")
    config = json.loads((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "12-6.model17-context-100k-experiment.v1":
        raise ValueError("unexpected MODEL-17 config schema")
    contexts = [int(value) for value in config["contexts"]]
    if contexts != [128, 256] or context not in contexts:
        raise ValueError("MODEL-17 contexts must remain exactly 128 and 256")
    if PACKING_VERSION != "s0-byte-pack-v1" or PACKING_CONFIG_HASH != (
        "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
    ):
        raise RuntimeError("canonical S0 packing identity drifted")

    stage = load_stage_config(repo_root / str(config["derived_model"]))
    if stage.stage != "S1" or stage.model.parameter_count() != 107_856:
        raise RuntimeError("MODEL-17 live ~100K S1 geometry drifted")
    spec = context_probe_spec(stage.model, max_seq_len=context)
    if spec.parameter_count() != stage.model.parameter_count():
        raise RuntimeError("context condition changed trainable parameter count")

    controls = config["controls"]
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = _load_records(train_path, split="train")
    validation_records = _load_records(validation_path, split="validation")
    if {record.record_id for record in train_records} & {
        record.record_id for record in validation_records
    }:
        raise RuntimeError("MODEL-17 train/held-out record overlap")

    packing_spec = ContextPackingSpec(sequence_length=context)
    dataset_identity = str(manifest.get("dataset_identity_sha256") or _sha256_file(manifest_path))
    dataset_id = str(manifest.get("dataset_id") or "s0-packaged-controlled")
    train_measurement = measure_context_candidate_packing(
        train_records,
        tokenizer,
        packing_spec=packing_spec,
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity,
        source_jsonl_sha256=_sha256_file(train_path),
        split="train",
    )
    validation_measurement = measure_context_candidate_packing(
        validation_records,
        tokenizer,
        packing_spec=packing_spec,
        dataset_id=dataset_id,
        dataset_identity_sha256=dataset_identity,
        source_jsonl_sha256=_sha256_file(validation_path),
        split="validation",
    )
    train_examples = _packed_examples(
        train_records, tokenizer, split="train", sequence_length=context
    )
    validation_examples = _packed_examples(
        validation_records, tokenizer, split="validation", sequence_length=context
    )
    train_diagnostics = _packing_diagnostics(
        train_records, train_examples, tokenizer, sequence_length=context
    )
    validation_diagnostics = _packing_diagnostics(
        validation_records, validation_examples, tokenizer, sequence_length=context
    )

    seed = int(controls["seed"])
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, stage.init)
    initial_fingerprint = _initial_parameter_fingerprint(model)
    initial_nll, initial_bpb, held_out_tokens = _evaluate_bpb(model, validation_examples)
    trainer_config = _trainer_config(config)
    trainer = Trainer(model, trainer_config, device="cpu")
    budget = int(controls["optimized_tokens"])

    step_seconds: list[float] = []
    step_token_counts: list[int] = []
    update_losses: list[float] = []
    grad_norms: list[float] = []
    example_index = 0
    started = time.perf_counter()
    while trainer.tokens_seen < budget:
        example = train_examples[example_index % len(train_examples)]
        example_index += 1
        remaining = budget - trainer.tokens_seen
        labels = _crop_labels_to_remaining(example, remaining)
        input_ids = torch.tensor(example.input_ids, dtype=torch.long).unsqueeze(0)
        step_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": input_ids, "labels": labels})
        step_seconds.append(time.perf_counter() - step_started)
        step_token_counts.append(metrics.tokens)
        if metrics.update_loss is None or metrics.grad_norm is None or not metrics.optimizer_stepped:
            raise RuntimeError("MODEL-17 expects one optimizer update per packed example")
        update_losses.append(float(metrics.update_loss))
        grad_norms.append(float(metrics.grad_norm))
    train_wall_seconds = time.perf_counter() - started
    if trainer.tokens_seen != budget:
        raise RuntimeError(f"MODEL-17 optimized-token drift: {trainer.tokens_seen} != {budget}")

    final_nll, final_bpb, final_held_out_tokens = _evaluate_bpb(model, validation_examples)
    if final_held_out_tokens != held_out_tokens:
        raise RuntimeError("MODEL-17 held-out token accounting drift")
    last_count = min(8, len(update_losses))
    recent_weight = sum(step_token_counts[-last_count:])
    recent_train_nll = sum(
        loss * tokens
        for loss, tokens in zip(update_losses[-last_count:], step_token_counts[-last_count:])
    ) / recent_weight
    warm_steps = step_seconds[2:] if len(step_seconds) > 2 else step_seconds

    report: dict[str, Any] = {
        "schema": SCHEMA_CONDITION,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "paid_compute": False,
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
        "context": context,
        "model": {
            "live_s1_model_identity_sha256": stage.model.identity_sha256(),
            "condition_model_identity_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "model_spec": spec.to_dict(),
            "initial_parameter_fingerprint": initial_fingerprint,
        },
        "controls": {
            "seed": seed,
            "optimized_tokens": budget,
            "tokenizer_version": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "trainer_config": asdict(trainer_config),
            "train_jsonl_sha256": _sha256_file(train_path),
            "validation_jsonl_sha256": _sha256_file(validation_path),
            "dataset_manifest_sha256": _sha256_file(manifest_path),
            "context_packing_identity_sha256": packing_spec.identity_sha256(
                tokenizer_config_sha256=BYTE_TOKENIZER_HASH
            ),
        },
        "packing": {
            "train_measurement": train_measurement.to_dict(),
            "validation_measurement": validation_measurement.to_dict(),
            "train_diagnostics": train_diagnostics,
            "validation_diagnostics": validation_diagnostics,
        },
        "training": {
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
            "recent_training_nll": recent_train_nll,
            "recent_training_bpb": recent_train_nll / math.log(2.0),
            "mean_grad_norm": statistics.fmean(grad_norms),
            "train_wall_seconds": train_wall_seconds,
            "seconds_per_optimized_token": train_wall_seconds / trainer.tokens_seen,
            "median_step_seconds_after_warmup": statistics.median(warm_steps),
            "median_tokens_per_step": statistics.median(step_token_counts),
        },
        "held_out": {
            "causal_tokens": held_out_tokens,
            "initial_nll": initial_nll,
            "initial_bpb": initial_bpb,
            "final_nll": final_nll,
            "final_bpb": final_bpb,
        },
        "truth_boundary": {
            "canonical_s0_context_changed": False,
            "canonical_s0_packing_changed": False,
            "representative_intended_s1_corpus": False,
            "long_dependency_capability_claim": False,
            "context_promotion": False,
            "paid_compute": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def _validate_condition(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA_CONDITION or report.get("authority") != AUTHORITY:
        raise ValueError("MODEL-17 condition schema/authority mismatch")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("MODEL-17 condition source SHA mismatch")
    if int(report["context"]) not in {128, 256}:
        raise ValueError("unexpected MODEL-17 context")
    if int(report["model"]["parameter_count"]) != 107_856:
        raise ValueError("MODEL-17 parameter count drift")
    if int(report["training"]["optimized_tokens"]) != 16_128:
        raise ValueError("MODEL-17 optimized-token budget drift")
    if report["packing"]["train_diagnostics"]["actual_causal_pairs_dropped_by_windowing"] != 0:
        raise ValueError("MODEL-17 packing dropped causal pairs")
    truth = report["truth_boundary"]
    if any(truth.get(key) is not False for key in truth):
        raise ValueError("MODEL-17 truth boundary weakened")
    supplied = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied != _canonical_hash(unsigned):
        raise ValueError("MODEL-17 condition self-hash mismatch")


def compare_conditions(
    *,
    condition_128: dict[str, Any],
    condition_256: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    _validate_condition(condition_128)
    _validate_condition(condition_256)
    if condition_128["context"] != 128 or condition_256["context"] != 256:
        raise ValueError("MODEL-17 comparison requires ordered 128 and 256 conditions")
    if condition_128["source"] != condition_256["source"]:
        raise ValueError("MODEL-17 conditions came from different source heads")
    control_fields = (
        "seed",
        "optimized_tokens",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "train_jsonl_sha256",
        "validation_jsonl_sha256",
        "dataset_manifest_sha256",
        "trainer_config",
    )
    for field in control_fields:
        if condition_128["controls"][field] != condition_256["controls"][field]:
            raise ValueError(f"MODEL-17 control drift: {field}")
    if condition_128["model"]["parameter_count"] != condition_256["model"]["parameter_count"]:
        raise ValueError("MODEL-17 model parameter counts differ")
    if (
        condition_128["model"]["initial_parameter_fingerprint"]
        != condition_256["model"]["initial_parameter_fingerprint"]
    ):
        raise ValueError("MODEL-17 initial weights differ between contexts")

    bpb_delta = condition_256["held_out"]["final_bpb"] - condition_128["held_out"]["final_bpb"]
    if abs(bpb_delta) < 0.01:
        fixture_quality = "INCONCLUSIVE_WITHIN_0.01_BPB"
    elif bpb_delta < 0:
        fixture_quality = "CONTEXT_256_LOWER_HELD_OUT_BPB_ON_CONTROLLED_FIXTURE"
    else:
        fixture_quality = "CONTEXT_128_LOWER_HELD_OUT_BPB_ON_CONTROLLED_FIXTURE"

    util128 = condition_128["packing"]["train_diagnostics"]["causal_pair_utilization"]
    util256 = condition_256["packing"]["train_diagnostics"]["causal_pair_utilization"]
    sec128 = condition_128["training"]["seconds_per_optimized_token"]
    sec256 = condition_256["training"]["seconds_per_optimized_token"]
    rss128 = condition_128["runtime"]["peak_process_rss_bytes"]
    rss256 = condition_256["runtime"]["peak_process_rss_bytes"]
    long_targets = condition_256["packing"]["train_diagnostics"][
        "targets_exposed_to_more_than_127_preceding_tokens"
    ]

    report: dict[str, Any] = {
        "schema": SCHEMA_COMPARISON,
        "authority": AUTHORITY,
        "source": condition_128["source"],
        "conditions": {"128": condition_128, "256": condition_256},
        "controlled_equivalence": {
            "same_parameter_count": True,
            "same_initial_parameter_fingerprint": True,
            "same_tokenizer": True,
            "same_corpus": True,
            "same_optimizer": True,
            "same_optimized_token_count": True,
            "different_optimizer_step_count_allowed": True,
            "packing_identities_distinct": (
                condition_128["controls"]["context_packing_identity_sha256"]
                != condition_256["controls"]["context_packing_identity_sha256"]
            ),
        },
        "separation": {
            "longer_dependency_exposure": {
                "context_256_targets_with_more_than_127_preceding_tokens": long_targets,
                "interpretation": (
                    "This counts positions that can receive additional same-window history; it does "
                    "not prove the model learned to use that history."
                ),
            },
            "packing_efficiency": {
                "train_causal_pair_utilization_128": util128,
                "train_causal_pair_utilization_256": util256,
                "utilization_delta_256_minus_128": util256 - util128,
                "tail_waste_tokens_128": condition_128["packing"]["train_diagnostics"][
                    "padding_tail_waste_tokens"
                ],
                "tail_waste_tokens_256": condition_256["packing"]["train_diagnostics"][
                    "padding_tail_waste_tokens"
                ],
            },
            "compute_per_token": {
                "seconds_per_optimized_token_128": sec128,
                "seconds_per_optimized_token_256": sec256,
                "ratio_256_over_128": sec256 / sec128,
            },
            "memory": {
                "peak_process_rss_bytes_128": rss128,
                "peak_process_rss_bytes_256": rss256,
                "ratio_256_over_128": rss256 / rss128,
            },
            "quality": {
                "final_held_out_bpb_128": condition_128["held_out"]["final_bpb"],
                "final_held_out_bpb_256": condition_256["held_out"]["final_bpb"],
                "bpb_delta_256_minus_128": bpb_delta,
                "controlled_fixture_preference": fixture_quality,
            },
        },
        "recommendation": {
            "primary_100k_research_context": "KEEP_LIVE_S1_256_AS_INCUMBENT_PENDING_INTENDED_CORPUS_REPLICATION",
            "reason": (
                "The live S1 ModelSpec is already 256. This experiment can determine mechanics, "
                "packing, resource cost, and controlled-fixture loss, but the repository branch does "
                "not contain representative intended-S1 corpus evidence. Do not switch the primary "
                "research identity to 128 or claim a 256 long-dependency benefit from this fixture alone."
            ),
            "controlled_fixture_quality_signal": fixture_quality,
        },
        "truth_boundary": {
            "canonical_s0_context_changed": False,
            "canonical_s0_packing_changed": False,
            "representative_intended_s1_corpus": False,
            "long_dependency_capability_claim": False,
            "context_promotion": False,
            "paid_compute": False,
        },
    }
    if report["controlled_equivalence"]["packing_identities_distinct"] is not True:
        raise RuntimeError("MODEL-17 context packing identities are not distinct")
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def validate_comparison(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA_COMPARISON or report.get("authority") != AUTHORITY:
        raise ValueError("MODEL-17 comparison schema/authority mismatch")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("MODEL-17 comparison source SHA mismatch")
    _validate_condition(report["conditions"]["128"], expected_source_sha=expected_source_sha)
    _validate_condition(report["conditions"]["256"], expected_source_sha=expected_source_sha)
    equivalence = report["controlled_equivalence"]
    if not all(equivalence.values()):
        raise ValueError("MODEL-17 controlled equivalence weakened")
    if report["truth_boundary"]["representative_intended_s1_corpus"] is not False:
        raise ValueError("MODEL-17 may not claim intended-S1 corpus evidence")
    supplied = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied != _canonical_hash(unsigned):
        raise ValueError("MODEL-17 comparison self-hash mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    condition = sub.add_parser("condition")
    condition.add_argument("--repo-root", type=Path, default=Path("."))
    condition.add_argument("--source-sha", required=True)
    condition.add_argument("--context", type=int, required=True)
    condition.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--context-128", type=Path, required=True)
    compare.add_argument("--context-256", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--expected-source-sha")
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "condition":
        report = run_condition(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            context=args.context,
            output_path=args.output,
        )
        _validate_condition(report, expected_source_sha=args.source_sha)
        print(json.dumps({"context": report["context"], "training": report["training"], "held_out": report["held_out"]}, sort_keys=True))
        return 0
    if args.command == "compare":
        c128 = json.loads(args.context_128.read_text(encoding="utf-8"))
        c256 = json.loads(args.context_256.read_text(encoding="utf-8"))
        report = compare_conditions(condition_128=c128, condition_256=c256, output_path=args.output)
        validate_comparison(report, expected_source_sha=args.expected_source_sha)
        print("MODEL17_SUMMARY=" + json.dumps({"separation": report["separation"], "recommendation": report["recommendation"]}, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_comparison(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA_COMPARISON}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
