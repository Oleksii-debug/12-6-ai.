"""EVAL-133 project-owned English raw-LM diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .cloze import conditional_log_likelihood, text_log_likelihood
from .eval_reservations import (
    assert_training_text_not_reserved,
    canonical_json_sha256,
    load_all_reservations,
)
from .evaluation import BenchmarkRegistry, BenchmarkSpec
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .scaling_experiment import (
    _byte_stream,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    controlled_specs,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer

SCHEMA = "12-6.eval133-en-raw-results.v1"
SUITE_SCHEMA = "12-6.en-raw-minimal-pairs.v1"
SUITE_ID = "eval133-en-raw-v1"
SUITE_VERSION = "1.0.0"
SUITE_SHA256 = "f9e713ff336e6189f7aa0ddbb21303431ab2041b6700ed38243eaf65865805cb"
RESERVATION_SHA256 = "850e0c34fd6ab35d0829b3f78ff5e81fbcb8c1ee900f3e7f1b967ea23a8f2e40"
RESERVED_INDEX_SHA256 = "7c7ee6d7513b84eff5a5d6e16fcbd66cf290cc7cafc228ced81b3449fe87454f"
AUTHORITY = "LOCAL_FREE_MECHANISTIC_ENGLISH_LANGUAGE_PROBE_NOT_PROFICIENCY_OR_INTELLIGENCE"
TARGET_PARAMETERS = (95_568, 467_808, 1_037_696)
REQUESTED_TRAIN_TOKENS = 65_536
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
SEED = 1337
TORCH_THREADS = 2
CHANCE_ACCURACY = 0.5
PHENOMENA = (
    "subject_verb_agreement",
    "article_noun_compatibility",
    "pronoun_agreement",
    "basic_tense_consistency",
    "negation",
    "local_syntactic_dependencies",
    "simple_semantic_plausibility",
    "continuation_coherence",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(json.dumps(list(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())

    add(value)
    return digest.hexdigest()


def load_suite(root: Path) -> list[dict[str, str]]:
    path = root / "data/evaluation/eval133_en_raw_v1.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"id", "phenomenon", "context", "preferred", "dispreferred"}
    if len(items) != 32 or any(set(item) != required for item in items):
        raise ValueError("EVAL-133 suite schema/count drift")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate EVAL-133 item id")
    counts = {name: 0 for name in PHENOMENA}
    for item in items:
        if item["phenomenon"] not in counts:
            raise ValueError("unknown EVAL-133 phenomenon")
        counts[item["phenomenon"]] += 1
        if not item["preferred"].startswith(" ") or not item["dispreferred"].startswith(" "):
            raise ValueError("continuations must carry an explicit leading space")
        if any(ord(ch) > 127 for value in item.values() for ch in value):
            raise ValueError("EVAL-133 v1 must remain ASCII")
    if set(counts.values()) != {4}:
        raise ValueError("EVAL-133 phenomenon balance drift")
    payload = {
        "schema_version": SUITE_SCHEMA,
        "suite_id": SUITE_ID,
        "language": "en",
        "item_count": len(items),
        "items": items,
    }
    if canonical_json_sha256(payload) != SUITE_SHA256:
        raise ValueError("EVAL-133 immutable suite identity drift")
    return items


def benchmark_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=SUITE_ID,
        version=SUITE_VERSION,
        source_id=f"reserved-eval:{SUITE_SHA256}",
        held_out=True,
        allowed_uses=("evaluation", "research_diagnostic"),
        notes="Project-authored raw English cloze pairs reserved from all training uses.",
    )


def validate_reservation(root: Path, items: list[dict[str, str]]) -> dict[str, Any]:
    index = json.loads((root / "data/evaluation/reserved/index.json").read_text())
    if index.get("registry_sha256") != RESERVED_INDEX_SHA256:
        raise ValueError("reserved registry identity drift")
    matches = [
        row for row in load_all_reservations(root) if row.get("suite_id") == SUITE_ID
    ]
    if len(matches) != 1:
        raise ValueError("EVAL-133 reservation missing or duplicated")
    reservation = matches[0]
    if reservation.get("reservation_sha256") != RESERVATION_SHA256:
        raise ValueError("EVAL-133 reservation identity drift")
    if [row["id"] for row in reservation["items"]] != [row["id"] for row in items]:
        raise ValueError("EVAL-133 reservation does not cover every item")
    registry = BenchmarkRegistry([benchmark_spec()])
    if len(registry.training_collisions([benchmark_spec().source_id])) != 1:
        raise RuntimeError("D06 registry failed to reject reserved source for training")
    return {
        "reservation_id": reservation["reservation_id"],
        "reservation_sha256": RESERVATION_SHA256,
        "reserved_index_sha256": RESERVED_INDEX_SHA256,
        "benchmark_registry_manifest": registry.manifest(),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [float(row["margin"]) for row in rows]
    token = [float(row["token_margin"]) for row in rows]
    byte = [float(row["byte_margin"]) for row in rows]
    return {
        "items": len(rows),
        "chance_accuracy": CHANCE_ACCURACY,
        "accuracy": sum(x > 0 for x in raw) / len(rows),
        "correct": sum(x > 0 for x in raw),
        "ties": sum(x == 0 for x in raw),
        "mean_log_likelihood_margin": statistics.fmean(raw),
        "token_normalized_accuracy": sum(x > 0 for x in token) / len(rows),
        "mean_normalized_margin_nats_per_token": statistics.fmean(token),
        "byte_normalized_accuracy": sum(x > 0 for x in byte) / len(rows),
        "mean_normalized_margin_nats_per_utf8_byte": statistics.fmean(byte),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    xv = sum((x - xm) ** 2 for x in xs)
    yv = sum((y - ym) ** 2 for y in ys)
    if xv == 0 or yv == 0:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True)) / math.sqrt(xv * yv)


def evaluate_model(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    mode = model.training
    before = _state_hash(model.state_dict())
    rows: list[dict[str, Any]] = []
    contexts = []
    for item in items:
        good = conditional_log_likelihood(
            model, tokenizer, item["context"], item["preferred"]
        )
        bad = conditional_log_likelihood(
            model, tokenizer, item["context"], item["dispreferred"]
        )
        context = text_log_likelihood(
            model, tokenizer, item["context"], require_byte_tokenizer=True
        )
        margin = good.log_likelihood - bad.log_likelihood
        token_margin = (
            good.mean_log_likelihood_per_token - bad.mean_log_likelihood_per_token
        )
        byte_margin = (
            good.mean_log_likelihood_per_utf8_byte
            - bad.mean_log_likelihood_per_utf8_byte
        )
        delta_len = bad.target_tokens - good.target_tokens
        stratum = (
            "preferred_shorter"
            if delta_len > 0
            else "preferred_longer"
            if delta_len < 0
            else "equal_length"
        )
        rows.append(
            {
                "id": item["id"],
                "phenomenon": item["phenomenon"],
                "preferred_log_likelihood": good.log_likelihood,
                "dispreferred_log_likelihood": bad.log_likelihood,
                "margin": margin,
                "token_margin": token_margin,
                "byte_margin": byte_margin,
                "preferred_target_tokens": good.target_tokens,
                "dispreferred_target_tokens": bad.target_tokens,
                "preferred_utf8_bytes": good.target_utf8_bytes,
                "dispreferred_utf8_bytes": bad.target_utf8_bytes,
                "length_stratum": stratum,
                "raw_correct": margin > 0,
                "token_normalized_correct": token_margin > 0,
                "raw_vs_token_normalized_flip": (margin > 0) != (token_margin > 0),
                "context_bpb_scored_bytes": context.bits_per_scored_byte,
            }
        )
        contexts.append(context)
    if _state_hash(model.state_dict()) != before or model.training != mode:
        raise RuntimeError("EVAL-133 evaluation mutated model state/mode")

    result = _aggregate(rows)
    result["per_phenomenon"] = {
        name: _aggregate([row for row in rows if row["phenomenon"] == name])
        for name in PHENOMENA
    }
    total_bytes = sum(context.scored_utf8_bytes for context in contexts)
    result["context_bpb"] = {
        "aggregate": -sum(context.log_likelihood for context in contexts)
        / (math.log(2.0) * total_bytes),
        "mean_item": statistics.fmean(
            context.bits_per_scored_byte for context in contexts
        ),
        "median_item": statistics.median(
            context.bits_per_scored_byte for context in contexts
        ),
        "min_item": min(context.bits_per_scored_byte for context in contexts),
        "max_item": max(context.bits_per_scored_byte for context in contexts),
        "scored_utf8_bytes": total_bytes,
    }
    strata = ("preferred_shorter", "equal_length", "preferred_longer")
    by_length = {
        name: _aggregate([row for row in rows if row["length_stratum"] == name])
        for name in strata
    }
    length_advantage = [
        float(row["dispreferred_target_tokens"] - row["preferred_target_tokens"])
        for row in rows
    ]
    result["length_diagnostics"] = {
        "byte_tokenizer_token_equals_utf8_byte_for_suite": all(
            row["preferred_target_tokens"] == row["preferred_utf8_bytes"]
            and row["dispreferred_target_tokens"] == row["dispreferred_utf8_bytes"]
            for row in rows
        ),
        "by_length_stratum": by_length,
        "raw_vs_token_normalized_choice_flips": sum(
            row["raw_vs_token_normalized_flip"] for row in rows
        ),
        "raw_margin_vs_preferred_length_advantage_pearson": _pearson(
            length_advantage, [float(row["margin"]) for row in rows]
        ),
    }
    result["item_scores"] = rows
    return result


def _selected_specs() -> tuple[ModelSpec, ...]:
    result = tuple(
        spec for spec in controlled_specs() if spec.parameter_count() in TARGET_PARAMETERS
    )
    if tuple(spec.parameter_count() for spec in result) != TARGET_PARAMETERS:
        raise RuntimeError("EVAL-133 controlled model family drift")
    return result


def _save_checkpoint(
    path: Path,
    source_sha: str,
    model: TwelveSixDecoder,
    trainer: Trainer,
    controls_sha256: str,
) -> dict[str, Any]:
    trainer.assert_checkpoint_safe()
    payload = {
        "schema": "12-6.eval133-learned-checkpoint.v1",
        "source_sha": source_sha,
        "suite_sha256": SUITE_SHA256,
        "controls_sha256": controls_sha256,
        "model_spec": model.spec.to_dict(),
        "model_spec_sha256": model.spec.identity_sha256(),
        "init_spec_sha256": model.init_spec.identity_sha256(),
        "parameter_count": model.spec.parameter_count(),
        "tokenizer_id": BYTE_TOKENIZER_VERSION,
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "model_state": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": _sha_file(path),
        "bytes": path.stat().st_size,
        "model_state_sha256": _state_hash(payload["model_state"]),
    }


def run(
    root: Path,
    source_sha: str,
    output: Path,
    work_dir: Path,
    requested_train_tokens: int = REQUESTED_TRAIN_TOKENS,
    torch_threads: int = TORCH_THREADS,
) -> dict[str, Any]:
    root = root.resolve()
    if _git_head(root) != source_sha:
        raise RuntimeError("EVAL-133 exact-checkout mismatch")
    if requested_train_tokens <= 0 or torch_threads <= 0:
        raise ValueError("train tokens and torch threads must be positive")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    items = load_suite(root)
    reservation = validate_reservation(root, items)
    tokenizer = ByteTokenizer()

    train_path = root / "data/s0/packaged/train.jsonl"
    validation_path = root / "data/s0/packaged/validation.jsonl"
    manifest_path = root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    assert_training_text_not_reserved(root, [str(row["text"]) for row in train_records])
    assert_training_text_not_reserved(
        root, [str(row["text"]) for row in validation_records]
    )

    stream = _byte_stream(train_records, tokenizer)
    per_step = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
    max_steps = math.ceil(requested_train_tokens / per_step)
    trainer_config = _trainer_config(max_steps=max_steps, seed=SEED)
    manifest = json.loads(manifest_path.read_text())
    controls = {
        "family": "RESEARCH41_FIXED_BYTE_CONTROLLED",
        "tokenizer_id": BYTE_TOKENIZER_VERSION,
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
        "vocab_size": 256,
        "max_seq_len": 256,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "valid_causal_tokens_per_full_step": per_step,
        "requested_train_tokens": requested_train_tokens,
        "actual_train_tokens": max_steps * per_step,
        "seed": SEED,
        "optimizer": asdict(trainer_config),
        "train_jsonl_sha256": _sha_file(train_path),
        "validation_jsonl_sha256": _sha_file(validation_path),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
    }
    controls_sha256 = canonical_json_sha256(controls)
    model_results = []
    for spec in _selected_specs():
        random.seed(SEED)
        torch.manual_seed(SEED)
        model = TwelveSixDecoder(spec, InitSpec())
        trainer = Trainer(model, trainer_config, device="cpu")
        trainer_before = _state_hash(asdict(trainer.state_dict()))
        random_eval = evaluate_model(model, tokenizer, items)
        if _state_hash(asdict(trainer.state_dict())) != trainer_before:
            raise RuntimeError("random-init evaluation mutated Trainer state")

        for step in range(max_steps):
            batch = _make_batch(
                stream,
                step=step,
                batch_size=BATCH_SIZE,
                sequence_length=SEQUENCE_LENGTH,
            )
            metrics = trainer.train_microbatch({"input_ids": batch})
            if not metrics.optimizer_stepped:
                raise RuntimeError("expected one optimizer update per batch")
        if trainer.tokens_seen != max_steps * per_step:
            raise RuntimeError("optimized-token accounting drift")

        trainer_before = _state_hash(asdict(trainer.state_dict()))
        learned_eval = evaluate_model(model, tokenizer, items)
        if _state_hash(asdict(trainer.state_dict())) != trainer_before:
            raise RuntimeError("learned evaluation mutated Trainer state")
        checkpoint = _save_checkpoint(
            work_dir / "checkpoints" / f"{spec.parameter_count()}.pt",
            source_sha,
            model,
            trainer,
            controls_sha256,
        )
        model_results.append(
            {
                "parameters": spec.parameter_count(),
                "model_spec_sha256": spec.identity_sha256(),
                "optimized_tokens": trainer.tokens_seen,
                "optimizer_steps": trainer.optimizer_step,
                "random_init": random_eval,
                "learned": learned_eval,
                "checkpoint": checkpoint,
                "delta": {
                    "accuracy": learned_eval["accuracy"] - random_eval["accuracy"],
                    "mean_log_likelihood_margin": learned_eval[
                        "mean_log_likelihood_margin"
                    ]
                    - random_eval["mean_log_likelihood_margin"],
                    "context_bpb": learned_eval["context_bpb"]["aggregate"]
                    - random_eval["context_bpb"]["aggregate"],
                },
            }
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "paid_compute": False,
        },
        "suite": {
            "suite_id": SUITE_ID,
            "version": SUITE_VERSION,
            "suite_sha256": SUITE_SHA256,
            "items": len(items),
            "phenomena": list(PHENOMENA),
            "authorship": "PROJECT_AUTHORED_FOR_12_6",
            "chance_accuracy": CHANCE_ACCURACY,
            "instruction_following": False,
            "intelligence_benchmark": False,
            **reservation,
        },
        "controls": controls,
        "controls_sha256": controls_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "model_results": model_results,
        "interpretation_boundary": {
            "mechanistic_language_learning_probe": True,
            "broad_english_proficiency_claim": False,
            "instruction_following_claim": False,
            "intelligence_claim": False,
            "representative_corpus_claim": False,
            "training_fixture_recycled": True,
            "tokenization_note": (
                "ASCII plus canonical byte tokenization makes tokens equal UTF-8 bytes; "
                "raw, normalized, equal-length, and length-stratified results are reported."
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def validate_report(report: Mapping[str, Any], expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("EVAL-133 report schema/authority mismatch")
    if report["suite"]["suite_sha256"] != SUITE_SHA256:
        raise ValueError("EVAL-133 report suite identity mismatch")
    if report["source"]["paid_compute"] is not False:
        raise ValueError("EVAL-133 report is not LOCAL_FREE")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("EVAL-133 report source SHA mismatch")
    if [row["parameters"] for row in report["model_results"]] != list(TARGET_PARAMETERS):
        raise ValueError("EVAL-133 report model family mismatch")
    expected_tokens = report["controls"]["actual_train_tokens"]
    if any(row["optimized_tokens"] != expected_tokens for row in report["model_results"]):
        raise ValueError("EVAL-133 models are not token-comparable")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report["report_sha256"] != canonical_json_sha256(unsigned):
        raise ValueError("EVAL-133 report self hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--work-dir", type=Path, required=True)
    run_parser.add_argument("--requested-train-tokens", type=int, default=REQUESTED_TRAIN_TOKENS)
    run_parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "run":
        report = run(
            args.repo_root,
            args.source_sha,
            args.output,
            args.work_dir,
            args.requested_train_tokens,
            args.torch_threads,
        )
        print(
            json.dumps(
                [
                    {
                        "parameters": row["parameters"],
                        "random_accuracy": row["random_init"]["accuracy"],
                        "learned_accuracy": row["learned"]["accuracy"],
                        "learned_margin": row["learned"]["mean_log_likelihood_margin"],
                        "learned_context_bpb": row["learned"]["context_bpb"]["aggregate"],
                    }
                    for row in report["model_results"]
                ],
                indent=2,
            )
        )
        return 0
    report = json.loads(args.report.read_text())
    validate_report(report, args.expected_source_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
