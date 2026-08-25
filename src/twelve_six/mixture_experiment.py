"""DATA-34 fixed-control UK/EN/code mixture experiment.

The experiment varies only MixturePlan weights. Model geometry, tokenizer, optimizer,
seed, sequence geometry, loss-token budget and held-out material remain fixed.
Results are controlled LOCAL_FREE evidence, not a representative-corpus mixture freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .model import InitSpec, TwelveSixDecoder
from .packing.scale_contracts import MixturePlan, MixtureSource, RestartCursor
from .scaling_experiment import controlled_specs
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.data34-mixture-report.v1"
CONFIG_SCHEMA = "12-6.data34-mixture-experiment.v1"
AUTHORITY = "LOCAL_FREE_CONTROLLED_MIXTURE_EVIDENCE_NOT_CORPUS_FREEZE"
MODALITIES = ("uk", "en", "code")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid DATA-34 experiment config schema")
    controls = value.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("controls must be an object")
    if int(controls.get("expected_trainable_parameters", -1)) != 267_912:
        raise ValueError("DATA-34 control must remain the 267,912-parameter model")
    if controls.get("tokenizer") != BYTE_TOKENIZER_VERSION:
        raise ValueError("DATA-34 control must remain the canonical byte tokenizer")
    for field in ("requested_loss_tokens", "batch_size", "sequence_length", "torch_threads"):
        if int(controls.get(field, 0)) <= 0:
            raise ValueError(f"controls.{field} must be positive")
    if int(controls["sequence_length"]) > 256 or int(controls["sequence_length"]) < 2:
        raise ValueError("sequence_length must be within [2, 256]")

    mixtures = value.get("mixtures")
    if not isinstance(mixtures, list) or len(mixtures) != 4:
        raise ValueError("DATA-34 requires exactly four prespecified mixtures")
    ids: set[str] = set()
    incumbent_count = 0
    for mixture in mixtures:
        if not isinstance(mixture, dict) or not isinstance(mixture.get("id"), str):
            raise ValueError("each mixture requires an id")
        mixture_id = str(mixture["id"])
        if mixture_id in ids:
            raise ValueError("mixture ids must be unique")
        ids.add(mixture_id)
        weights = mixture.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(MODALITIES):
            raise ValueError("mixture weights must contain exactly uk/en/code")
        if any(not isinstance(weights[name], int) or int(weights[name]) <= 0 for name in MODALITIES):
            raise ValueError("mixture weights must be positive integers")
        if sum(int(weights[name]) for name in MODALITIES) != 100:
            raise ValueError("mixture weights must sum to 100")
        if mixture.get("role") == "incumbent":
            incumbent_count += 1
    if incumbent_count != 1:
        raise ValueError("exactly one mixture must be marked incumbent")

    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("final test must not be used for selection")
    if selection.get("incumbent_id") not in ids:
        raise ValueError("selection incumbent_id must name a configured mixture")
    threshold = float(selection.get("severe_regression_relative_threshold", -1.0))
    minimum_gain = float(selection.get("minimum_macro_improvement_relative", -1.0))
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= minimum_gain <= 1.0:
        raise ValueError("selection thresholds must be within [0, 1]")

    data = value.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    for split in ("selection_validation", "final_test"):
        split_value = data.get(split)
        if not isinstance(split_value, dict) or set(split_value) != set(MODALITIES):
            raise ValueError(f"{split} must contain exactly uk/en/code")
        for modality in MODALITIES:
            texts = split_value[modality]
            if not isinstance(texts, list) or not texts or not all(
                isinstance(text, str) and text for text in texts
            ):
                raise ValueError(f"{split}.{modality} must contain non-empty strings")
    return value


def _packing_identity(config: Mapping[str, Any]) -> str:
    controls = config["controls"]
    return _canonical_hash(
        {
            "id": "data34-source-conditioned-cyclic-byte-packing-v1",
            "batch_size": int(controls["batch_size"]),
            "sequence_length": int(controls["sequence_length"]),
            "target_alignment": "causal-shift-by-trainer",
            "source_window_stride": int(controls["sequence_length"]) - 1,
            "cross_source_rows": False,
        }
    )


def training_streams(
    repo_root: Path,
    config: Mapping[str, Any],
    tokenizer: ByteTokenizer,
) -> tuple[dict[str, bytes], dict[str, str], dict[str, Any]]:
    data = config["data"]
    train_path = repo_root / str(data["training_path"])
    raw = train_path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    streams: dict[str, bytes] = {}
    manifests: dict[str, str] = {}
    slices: dict[str, object] = {}
    for modality in MODALITIES:
        stratum = data["strata"][modality]
        start, end = [int(item) for item in stratum["line_range_1based_inclusive"]]
        if start <= 0 or end < start or end > len(lines):
            raise ValueError(f"invalid line range for {modality}")
        text = "".join(lines[start - 1 : end])
        encoded = bytes(tokenizer.encode(text))
        if len(encoded) < int(config["controls"]["sequence_length"]):
            raise ValueError(f"training stream {modality} is shorter than sequence_length")
        streams[modality] = encoded
        manifest_payload = {
            "training_file_sha256": _file_sha256(train_path),
            "training_path": str(data["training_path"]),
            "modality": modality,
            "line_range_1based_inclusive": [start, end],
            "utf8_bytes": len(text.encode("utf-8")),
        }
        manifests[modality] = _canonical_hash(manifest_payload)
        slices[modality] = manifest_payload
    metadata = {
        "training_path": str(data["training_path"]),
        "training_file_sha256": _file_sha256(train_path),
        "training_file_utf8_bytes": len(raw.encode("utf-8")),
        "strata": slices,
    }
    return streams, manifests, metadata


def build_plan(
    mixture: Mapping[str, Any],
    manifests: Mapping[str, str],
    *,
    config: Mapping[str, Any],
) -> MixturePlan:
    weights = mixture["weights"]
    return MixturePlan(
        plan_id=f"data34-{mixture['id']}-v1",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=_packing_identity(config),
        sources=tuple(
            MixtureSource(modality, manifests[modality], int(weights[modality]))
            for modality in MODALITIES
        ),
        seed=int(config["controls"]["seed"]),
        num_shards=1,
    )


def next_mixture_batch(
    plan: MixturePlan,
    cursor: RestartCursor,
    streams: Mapping[str, bytes],
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, RestartCursor, Counter[str]]:
    rows: list[list[int]] = []
    emitted: Counter[str] = Counter()
    current = cursor
    stride = sequence_length - 1
    for _ in range(batch_size):
        source, offset = current.next_source_and_offset(plan)
        stream = streams[source]
        start = (offset * stride) % len(stream)
        row = [stream[(start + index) % len(stream)] for index in range(sequence_length)]
        rows.append(row)
        current = current.advance(
            plan,
            source_name=source,
            emitted_sequences=1,
            emitted_loss_tokens=stride,
        )
        emitted[source] += stride
    return torch.tensor(rows, dtype=torch.long), current, emitted


def schedule_preview(
    plan: MixturePlan,
    *,
    requested_loss_tokens: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    tokens_per_step = batch_size * (sequence_length - 1)
    steps = math.ceil(requested_loss_tokens / tokens_per_step)
    samples = steps * batch_size
    counts: Counter[str] = Counter(plan.source_for_sample(index) for index in range(samples))
    loss_counts = {
        modality: counts[modality] * (sequence_length - 1) for modality in MODALITIES
    }
    actual = steps * tokens_per_step
    return {
        "optimizer_steps": steps,
        "scheduled_samples": samples,
        "actual_loss_tokens": actual,
        "loss_tokens_by_modality": loss_counts,
        "loss_token_share": {
            modality: loss_counts[modality] / actual for modality in MODALITIES
        },
    }


def _trainer_config(config: Mapping[str, Any], *, max_steps: int) -> TrainerConfig:
    controls = config["controls"]
    optimizer = controls["optimizer"]
    return TrainerConfig(
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        max_steps=max_steps,
        warmup_steps=int(optimizer["warmup_steps"]),
        scheduler=str(optimizer["scheduler"]),
        gradient_accumulation_steps=int(optimizer["gradient_accumulation_steps"]),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        precision=str(optimizer["precision"]),
        seed=int(controls["seed"]),
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


@torch.no_grad()
def evaluate_bpb(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    texts_by_modality: Mapping[str, list[str]],
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    modality_results: dict[str, dict[str, float | int]] = {}
    total_nll = 0.0
    total_targets = 0
    for modality in MODALITIES:
        nll = 0.0
        targets = 0
        for text in texts_by_modality[modality]:
            token_ids = tokenizer.encode(text)
            start = 0
            while start < len(token_ids) - 1:
                chunk = token_ids[start : start + model.spec.max_seq_len]
                if len(chunk) < 2:
                    break
                input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
                logits = model(input_ids).logits
                loss_sum = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                nll += float(loss_sum.item())
                targets += len(chunk) - 1
                start += model.spec.max_seq_len - 1
        if targets <= 0:
            raise RuntimeError(f"held-out split produced no target bytes for {modality}")
        bpb = nll / targets / math.log(2.0)
        modality_results[modality] = {
            "bpb": bpb,
            "target_bytes": targets,
            "nll_nats": nll,
        }
        total_nll += nll
        total_targets += targets
    model.train(was_training)
    bpbs = {modality: float(modality_results[modality]["bpb"]) for modality in MODALITIES}
    return {
        "by_modality": modality_results,
        "macro_bpb": sum(bpbs.values()) / len(MODALITIES),
        "micro_bpb": total_nll / total_targets / math.log(2.0),
        "incumbent_policy_weighted_bpb": (
            0.45 * bpbs["uk"] + 0.35 * bpbs["en"] + 0.20 * bpbs["code"]
        ),
        "total_target_bytes": total_targets,
    }


def run_one_mixture(
    *,
    config: Mapping[str, Any],
    mixture: Mapping[str, Any],
    streams: Mapping[str, bytes],
    manifests: Mapping[str, str],
) -> dict[str, Any]:
    controls = config["controls"]
    spec_index = int(controls["controlled_spec_index"])
    spec = controlled_specs()[spec_index]
    parameters = spec.parameter_count()
    if parameters != int(controls["expected_trainable_parameters"]):
        raise RuntimeError("controlled model geometry drifted")
    tokenizer = ByteTokenizer()
    batch_size = int(controls["batch_size"])
    sequence_length = int(controls["sequence_length"])
    requested = int(controls["requested_loss_tokens"])
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps = math.ceil(requested / tokens_per_step)
    plan = build_plan(mixture, manifests, config=config)
    preview = schedule_preview(
        plan,
        requested_loss_tokens=requested,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    seed = int(controls["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    init_spec = InitSpec()
    model = TwelveSixDecoder(spec, init_spec)
    trainer_config = _trainer_config(config, max_steps=max_steps)
    trainer = Trainer(model, trainer_config, device="cpu")
    initial_validation = evaluate_bpb(model, tokenizer, config["data"]["selection_validation"])

    cursor = RestartCursor.initial(plan)
    emitted: Counter[str] = Counter()
    last_metrics = None
    for _step in range(max_steps):
        batch, cursor, batch_emitted = next_mixture_batch(
            plan,
            cursor,
            streams,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        emitted.update(batch_emitted)
        last_metrics = trainer.train_microbatch({"input_ids": batch})

    actual_loss_tokens = max_steps * tokens_per_step
    if trainer.tokens_seen != actual_loss_tokens:
        raise RuntimeError(
            f"trainer loss-token accounting drift: {trainer.tokens_seen} != {actual_loss_tokens}"
        )
    if cursor.emitted_loss_tokens != actual_loss_tokens:
        raise RuntimeError("MixturePlan restart cursor loss-token accounting drifted")
    if sum(emitted.values()) != actual_loss_tokens:
        raise RuntimeError("per-modality loss-token accounting drifted")
    if dict(emitted) != preview["loss_tokens_by_modality"]:
        raise RuntimeError("executed mixture schedule differs from deterministic preview")

    validation = evaluate_bpb(model, tokenizer, config["data"]["selection_validation"])
    final_test = evaluate_bpb(model, tokenizer, config["data"]["final_test"])
    if last_metrics is None:
        raise RuntimeError("training produced no optimizer steps")
    return {
        "mixture_id": mixture["id"],
        "role": mixture["role"],
        "configured_weights": mixture["weights"],
        "mixture_plan_sha256": plan.sha256,
        "restart_cursor_sha256": cursor.sha256,
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": parameters,
        "requested_loss_tokens": requested,
        "optimized_loss_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "tokens_per_optimizer_step": tokens_per_step,
        "loss_tokens_by_modality": dict(sorted(emitted.items())),
        "loss_token_share": {
            modality: emitted[modality] / trainer.tokens_seen for modality in MODALITIES
        },
        "initial_validation": initial_validation,
        "selection_validation": validation,
        "final_test": final_test,
        "last_train_loss": float(last_metrics.loss),
        "last_grad_norm": last_metrics.grad_norm,
    }


def selection_decision(
    runs: list[dict[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(run["mixture_id"]): run for run in runs}
    incumbent_id = str(selection["incumbent_id"])
    incumbent = by_id[incumbent_id]
    incumbent_bpb = {
        modality: float(incumbent["selection_validation"]["by_modality"][modality]["bpb"])
        for modality in MODALITIES
    }
    threshold = float(selection["severe_regression_relative_threshold"])
    minimum_gain = float(selection["minimum_macro_improvement_relative"])
    comparisons: dict[str, Any] = {}
    eligible: list[dict[str, Any]] = []
    for run in runs:
        relative: dict[str, float] = {}
        severe: list[str] = []
        for modality in MODALITIES:
            candidate = float(run["selection_validation"]["by_modality"][modality]["bpb"])
            delta = candidate / incumbent_bpb[modality] - 1.0
            relative[modality] = delta
            if delta > threshold:
                severe.append(modality)
        macro = float(run["selection_validation"]["macro_bpb"])
        comparisons[str(run["mixture_id"])] = {
            "relative_validation_bpb_vs_incumbent": relative,
            "severe_regression_modalities": severe,
            "passes_regression_guard": not severe,
            "macro_validation_bpb": macro,
        }
        if not severe:
            eligible.append(run)
    if not eligible:
        raise RuntimeError("incumbent unexpectedly failed its own regression guard")
    best = min(eligible, key=lambda run: float(run["selection_validation"]["macro_bpb"]))
    incumbent_macro = float(incumbent["selection_validation"]["macro_bpb"])
    best_macro = float(best["selection_validation"]["macro_bpb"])
    relative_gain = (incumbent_macro - best_macro) / incumbent_macro
    if best["mixture_id"] != incumbent_id and relative_gain < minimum_gain:
        winner = incumbent
        retained_for_minimum_gain = True
    else:
        winner = best
        retained_for_minimum_gain = False
    return {
        "selection_split": selection["selection_split"],
        "test_used_for_selection": False,
        "primary_metric": selection["primary_metric"],
        "incumbent_id": incumbent_id,
        "winner_id": winner["mixture_id"],
        "best_guard_passing_id": best["mixture_id"],
        "best_guard_passing_relative_macro_gain_vs_incumbent": relative_gain,
        "minimum_macro_improvement_relative": minimum_gain,
        "retained_incumbent_for_minimum_gain": retained_for_minimum_gain,
        "severe_regression_relative_threshold": threshold,
        "comparisons": comparisons,
    }


def run_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if _HEX40.fullmatch(source_sha) is None:
        raise ValueError("source_sha must be lowercase 40-hex Git SHA")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed}")
    config = load_config(config_path)
    torch.set_num_threads(int(config["controls"]["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    streams, manifests, training_metadata = training_streams(repo_root, config, tokenizer)

    runs = [
        run_one_mixture(
            config=config,
            mixture=mixture,
            streams=streams,
            manifests=manifests,
        )
        for mixture in config["mixtures"]
    ]
    optimized_counts = {int(run["optimized_loss_tokens"]) for run in runs}
    if len(optimized_counts) != 1:
        raise RuntimeError("mixtures did not receive identical optimized loss-token counts")
    model_ids = {str(run["model_identity_sha256"]) for run in runs}
    if len(model_ids) != 1:
        raise RuntimeError("model identity changed across mixture runs")

    decision = selection_decision(runs, config["selection"])
    winner = next(run for run in runs if run["mixture_id"] == decision["winner_id"])
    incumbent = next(run for run in runs if run["mixture_id"] == decision["incumbent_id"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": int(config["controls"]["torch_threads"]),
            "paid_compute": False,
        },
        "config": {
            "path": str(config_path.relative_to(repo_root)),
            "sha256": _file_sha256(config_path),
            "semantic_sha256": _canonical_hash(config),
        },
        "fixed_controls": {
            **config["controls"],
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "packing_config_sha256": _packing_identity(config),
            "model_identity_sha256": next(iter(model_ids)),
            "actual_optimized_loss_tokens_each": next(iter(optimized_counts)),
        },
        "data": {
            **training_metadata,
            "stratum_manifest_sha256": manifests,
            "selection_validation_semantic_sha256": _canonical_hash(
                config["data"]["selection_validation"]
            ),
            "final_test_semantic_sha256": _canonical_hash(config["data"]["final_test"]),
            "authority": config["data"]["authority"],
        },
        "runs": runs,
        "decision": decision,
        "provisional_recommendation": {
            "mixture_id": winner["mixture_id"],
            "weights": winner["configured_weights"],
            "selection_validation_macro_bpb": winner["selection_validation"]["macro_bpb"],
            "final_test_macro_bpb_diagnostic_not_selection": winner["final_test"]["macro_bpb"],
            "incumbent_selection_validation_macro_bpb": incumbent["selection_validation"]["macro_bpb"],
            "scope": "provisional for the controlled 267,912-parameter fixture only",
        },
        "truth_boundary": config["truth_boundary"],
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: Mapping[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected DATA-34 report schema")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("report source SHA mismatch")
    payload = dict(report)
    observed_hash = str(payload.pop("report_sha256"))
    if observed_hash != _canonical_hash(payload):
        raise ValueError("DATA-34 report self-hash mismatch")
    if report["decision"]["test_used_for_selection"] is not False:
        raise ValueError("report illegally used final test for selection")
    optimized = {int(run["optimized_loss_tokens"]) for run in report["runs"]}
    if len(optimized) != 1:
        raise ValueError("report mixtures have unequal optimized-token counts")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/data34_mixture_268k_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    report = run_experiment(
        repo_root=repo_root,
        source_sha=args.source_sha,
        config_path=config_path,
        output_path=output_path,
    )
    validate_report(report, expected_source_sha=args.source_sha)
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "winner_id": report["decision"]["winner_id"],
                "weights": report["provisional_recommendation"]["weights"],
                "actual_optimized_loss_tokens_each": report["fixed_controls"][
                    "actual_optimized_loss_tokens_each"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
