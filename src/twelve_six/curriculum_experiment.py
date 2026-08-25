"""DATA-35 simple pretraining-curriculum experiment.

The incumbent MixturePlan + RestartCursor remains the only source scheduler.
DATA-35 materializes one finite incumbent source/offset trace and compares
permutations of that exact envelope multiset. Ordering changes; model, tokenizer,
optimizer recipe, optimized-token budget, final modality counts, and the exact
training-batch multiset do not.

This is LOCAL_FREE mechanics evidence on the tiny project-authored DATA-10
fixture, not representative-corpus evidence or promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F

from .checkpoint.core import hash_json, sha256_file
from .fixed_token_efficiency import _make_pair_batch, _trainer_config
from .model import InitSpec, TwelveSixDecoder
from .packing.scale_contracts import MixturePlan, MixtureSource, RestartCursor
from .scaling_experiment import controlled_specs
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.data35-curriculum-evidence.v1"
AUTHORITY = "LOCAL_FREE_SYNTHETIC_CURRICULUM_EVIDENCE_NOT_PROMOTION"
DATA10_SOURCE_SHA = "077205ef2b1662a5029bc77b8fc762078cabeb17"
EXPECTED_TRAIN_SHA256 = "059f04e01d6fc6b8224b373b08efbb37f09d546de35ed510afdb4587ebdb6012"
MODALITIES = ("uk", "en", "code")
Candidate = Literal[
    "fully_mixed",
    "quality_first_then_mixed",
    "ukrainian_first_then_mixed",
]

# Exact record reconstruction of the manifested 1,454-byte DATA-10 train file.
# The historical DATA-10 helper literals add terminal newlines to code records;
# those separators are not present in the manifested file, so DATA-35 binds to
# the file identity rather than reproducing that helper-literal inconsistency.
_TRAIN: tuple[tuple[str, str, str], ...] = (
    (
        "uk-1",
        "uk",
        "Українська мова має відмінки, дієвідмінювання і словотвір. Ці дані "
        "потрібні для базового передтренування моделі.",
    ),
    (
        "uk-2",
        "uk",
        "Дослідники працюють із текстами різних жанрів, щоб модель бачила слова "
        "у називному, родовому, давальному, знахідному та орудному відмінках.",
    ),
    (
        "uk-3",
        "uk",
        "Київ, Львів і Ужгород мають різні мовні контексти; ґрунтовний корпус "
        "повинен містити літери ґ, ї, є, і та природні апострофи.",
    ),
    (
        "en-1",
        "en",
        "The training corpus contains English prose with varied syntax and vocabulary "
        "so the base model learns next-token statistics rather than instructions.",
    ),
    (
        "en-2",
        "en",
        "These records test deterministic data selection, source provenance, "
        "deduplication, and restart behavior for a universal language model.",
    ),
    (
        "en-3",
        "en",
        "Data quality includes valid encoding, stable normalization, explicit source "
        "rights, and strict separation from held-out evaluation material.",
    ),
    (
        "code-1",
        "code",
        "def stable_hash(value: str) -> str:\n"
        "    return hashlib.sha256(value.encode('utf-8')).hexdigest()",
    ),
    (
        "code-2",
        "code",
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "    def increment(self):\n"
        "        self.value += 1\n"
        "        return self.value",
    ),
    (
        "code-3",
        "code",
        "SELECT source_id, COUNT(*) FROM records\n"
        "WHERE split = 'train'\n"
        "GROUP BY source_id ORDER BY source_id;",
    ),
)

# Existing DATA-10 held-out probes. They are never read by schedule construction.
_HELDOUT: dict[str, tuple[str, ...]] = {
    "uk": (
        "книга книги книзі книгу книгою; учень учня учневі учнем",
        "працювати працюю працюєш працює працюємо працюють; прочитати прочитають",
        "п'ять, об'єкт, м'який, під'їзд, ґанок, їжак, Європа, Україна",
    ),
    "en": (
        "The multilingual base model compares token fertility on unseen English.",
    ),
    "code": (
        "for index, item in enumerate(records):\n    assert item.split == 'train'\n",
    ),
}

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TraceEntry:
    sample_index: int
    source: str
    source_offset: int
    record_id: str
    record_sha256: str
    quality_score: float

    def identity(self) -> tuple[str, int, str, str]:
        return self.source, self.source_offset, self.record_id, self.record_sha256


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "12-6.data35-curriculum-config.v1":
        raise RuntimeError("DATA-35 config schema mismatch")
    return value


def _records_by_modality() -> dict[str, tuple[tuple[str, str], ...]]:
    result: dict[str, list[tuple[str, str]]] = {name: [] for name in MODALITIES}
    for record_id, modality, text in _TRAIN:
        result[modality].append((record_id, text))
    return {name: tuple(result[name]) for name in MODALITIES}


def _assert_current_train_snapshot(repo_root: Path) -> str:
    path = repo_root / "data/synthetic/data10/uk-en-code-train.txt"
    observed_sha = sha256_file(path)
    if observed_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"DATA-10 train fixture drift: {observed_sha} != {EXPECTED_TRAIN_SHA256}"
        )
    expected_text = "\n".join(text for _record_id, _modality, text in _TRAIN) + "\n"
    if path.read_text(encoding="utf-8") != expected_text:
        raise RuntimeError("DATA-10 record reconstruction does not match manifested fixture")
    train_hashes = {_sha_text(text) for _record_id, _modality, text in _TRAIN}
    heldout_hashes = {_sha_text(text) for texts in _HELDOUT.values() for text in texts}
    overlap = sorted(train_hashes & heldout_hashes)
    if overlap:
        raise RuntimeError(f"exact train/heldout text overlap: {overlap!r}")
    return observed_sha


def mechanics_quality_score(modality: str, text: str) -> float:
    """Training-only interpretable ordering proxy, never an eligibility decision."""
    encoded = text.encode("utf-8")
    nonspace = [char for char in text if not char.isspace()]
    length_term = min(len(encoded) / 180.0, 1.0)
    if modality in {"uk", "en"}:
        alphabetic_density = sum(char.isalpha() for char in nonspace) / max(len(nonspace), 1)
        words = [word.casefold() for word in _WORD_RE.findall(text)]
        type_ratio = len(set(words)) / max(len(words), 1)
        return 0.45 * alphabetic_density + 0.35 * type_ratio + 0.20 * length_term
    if modality != "code":
        raise ValueError(f"unknown modality {modality!r}")
    lines = text.splitlines()
    has_multiple_lines = float(len(lines) >= 2)
    has_structure = float(
        any(line.startswith((" ", "\t")) for line in lines)
        or "\nWHERE " in text
        or "\nGROUP BY " in text
    )
    structure = (has_multiple_lines + has_structure) / 2.0
    type_ratio = len(set(nonspace)) / max(len(nonspace), 1)
    return 0.40 * structure + 0.35 * type_ratio + 0.25 * length_term


def build_incumbent_plan(config: dict[str, Any]) -> MixturePlan:
    records = _records_by_modality()
    weights = config["data"]["mixture_weight_units"]
    sources: list[MixtureSource] = []
    for name in MODALITIES:
        manifest = hash_json(
            {
                "schema": "12-6.data35-source-manifest.v1",
                "source": name,
                "records": [
                    {"id": record_id, "sha256": _sha_text(text)}
                    for record_id, text in records[name]
                ],
            }
        )
        sources.append(MixtureSource(name, manifest, int(weights[name])))
    packing_hash = hash_json(
        {
            "version": "data35-research06-aligned-record-pairs-v1",
            "pair": "byte at cyclic record offset t predicts byte t+1",
            "batch_size": int(config["base_control"]["batch_size"]),
            "sequence_length": int(config["base_control"]["sequence_length"]),
            "curriculum_changes_batch_content": False,
        }
    )
    return MixturePlan(
        plan_id="data35-uk-en-code-incumbent-trace-v1",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=packing_hash,
        sources=tuple(sources),
        seed=int(config["data"]["mixture_plan_seed"]),
        num_shards=1,
    )


def materialize_incumbent_trace(
    plan: MixturePlan,
    *,
    steps: int,
    tokens_per_step: int,
) -> tuple[TraceEntry, ...]:
    if steps <= 0 or tokens_per_step <= 0:
        raise ValueError("steps and tokens_per_step must be positive")
    records = _records_by_modality()
    cursor = RestartCursor.initial(plan)
    result: list[TraceEntry] = []
    for sample_index in range(steps):
        source, source_offset = cursor.next_source_and_offset(plan)
        source_records = records[source]
        record_id, text = source_records[source_offset % len(source_records)]
        result.append(
            TraceEntry(
                sample_index=sample_index,
                source=source,
                source_offset=source_offset,
                record_id=record_id,
                record_sha256=_sha_text(text),
                quality_score=mechanics_quality_score(source, text),
            )
        )
        cursor = cursor.advance(
            plan,
            source_name=source,
            emitted_sequences=1,
            emitted_loss_tokens=tokens_per_step,
        )
    if cursor.emitted_sequences != steps:
        raise RuntimeError("MixturePlan sequence ledger drift")
    if cursor.emitted_loss_tokens != steps * tokens_per_step:
        raise RuntimeError("MixturePlan optimized-token ledger drift")
    return tuple(result)


def _assert_same_trace_multiset(
    reference: tuple[TraceEntry, ...],
    candidate: tuple[TraceEntry, ...],
) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError("curriculum changed trace length")
    if Counter(entry.identity() for entry in reference) != Counter(
        entry.identity() for entry in candidate
    ):
        raise RuntimeError("curriculum changed incumbent source/offset envelope multiset")


def _quality_prefix_with_fixed_source_slots(
    trace: tuple[TraceEntry, ...], prefix_steps: int
) -> tuple[TraceEntry, ...]:
    """Improve training-only quality while preserving the baseline prefix modality sequence."""
    source_slots = tuple(entry.source for entry in trace[:prefix_steps])
    quotas = Counter(source_slots)
    selected_by_source: dict[str, list[TraceEntry]] = {}
    selected_counts: Counter[tuple[str, int, str, str]] = Counter()
    for source in MODALITIES:
        ranked = sorted(
            (entry for entry in trace if entry.source == source),
            key=lambda item: (-item.quality_score, item.sample_index),
        )
        selected = ranked[: quotas[source]]
        if len(selected) != quotas[source]:
            raise RuntimeError(f"insufficient {source} envelopes for quality prefix")
        selected_by_source[source] = selected
        selected_counts.update(entry.identity() for entry in selected)

    next_index = Counter()
    prefix: list[TraceEntry] = []
    for source in source_slots:
        index = next_index[source]
        prefix.append(selected_by_source[source][index])
        next_index[source] += 1

    tail: list[TraceEntry] = []
    for entry in trace:
        identity = entry.identity()
        if selected_counts[identity]:
            selected_counts[identity] -= 1
        else:
            tail.append(entry)
    ordered = tuple(prefix) + tuple(tail)
    if tuple(entry.source for entry in ordered[:prefix_steps]) != source_slots:
        raise RuntimeError("quality curriculum changed baseline prefix modality sequence")
    return ordered


def order_trace(
    trace: tuple[TraceEntry, ...],
    *,
    candidate: Candidate,
    prefix_steps: int,
) -> tuple[TraceEntry, ...]:
    """Permute incumbent envelopes; never choose a new source or offset."""
    if not 0 < prefix_steps < len(trace):
        raise ValueError("prefix_steps must lie strictly inside the finite trace")
    if candidate == "fully_mixed":
        ordered = trace
    elif candidate == "quality_first_then_mixed":
        ordered = _quality_prefix_with_fixed_source_slots(trace, prefix_steps)
    elif candidate == "ukrainian_first_then_mixed":
        prefix = tuple(entry for entry in trace if entry.source == "uk")[:prefix_steps]
        if len(prefix) != prefix_steps:
            raise RuntimeError("incumbent trace has too few Ukrainian envelopes for prefix")
        selected = Counter(entry.identity() for entry in prefix)
        tail: list[TraceEntry] = []
        for entry in trace:
            identity = entry.identity()
            if selected[identity]:
                selected[identity] -= 1
            else:
                tail.append(entry)
        ordered = prefix + tuple(tail)
    else:
        raise ValueError(f"unknown curriculum candidate {candidate!r}")
    _assert_same_trace_multiset(trace, ordered)
    return ordered


def _order_identity(order: tuple[TraceEntry, ...]) -> str:
    return hash_json(
        [
            (entry.source, entry.source_offset, entry.record_id, entry.record_sha256)
            for entry in order
        ]
    )


def _trace_multiset_identity(trace: tuple[TraceEntry, ...]) -> str:
    return hash_json(sorted(entry.identity() for entry in trace))


def _texts_for_modality(modality: str) -> tuple[str, ...]:
    return tuple(text for _record_id, name, text in _TRAIN if name == modality)


@torch.no_grad()
def _bpb_texts(
    model: TwelveSixDecoder,
    texts: Iterable[str],
    tokenizer: ByteTokenizer,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_targets = 0
    for text in texts:
        token_ids = tokenizer.encode(text)
        start = 0
        while start < len(token_ids) - 1:
            chunk = token_ids[start : start + model.spec.max_seq_len]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids).logits
            total_nll += float(
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_targets += int(input_ids.shape[1] - 1)
            start += model.spec.max_seq_len - 1
    model.train(was_training)
    if total_targets <= 0:
        raise RuntimeError("BPB evaluation produced no targets")
    bpb = total_nll / total_targets / math.log(2.0)
    if not math.isfinite(bpb):
        raise RuntimeError("BPB evaluation is non-finite")
    return bpb, total_targets


def _evaluate_modalities(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    corpus: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    weighted_sum = 0.0
    total_targets = 0
    for modality in MODALITIES:
        bpb, targets = _bpb_texts(model, corpus[modality], tokenizer)
        result[modality] = {"bpb": bpb, "byte_targets": targets}
        weighted_sum += bpb * targets
        total_targets += targets
    result["aggregate"] = {
        "bpb": weighted_sum / total_targets,
        "byte_targets": total_targets,
    }
    return result


def _observed_interval_bpb(
    nll_sum: dict[str, float], token_count: dict[str, int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_nll = 0.0
    total_tokens = 0
    for modality in MODALITIES:
        tokens = int(token_count.get(modality, 0))
        nll = float(nll_sum.get(modality, 0.0))
        result[modality] = {
            "bpb": nll / tokens / math.log(2.0) if tokens else None,
            "optimized_tokens": tokens,
        }
        total_nll += nll
        total_tokens += tokens
    result["aggregate"] = {
        "bpb": total_nll / total_tokens / math.log(2.0) if total_tokens else None,
        "optimized_tokens": total_tokens,
    }
    return result


def _record_checkpoint(
    *,
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    optimized_tokens: int,
    interval_nll: dict[str, float],
    interval_tokens: dict[str, int],
) -> dict[str, Any]:
    train_reference = {name: _texts_for_modality(name) for name in MODALITIES}
    heldout = {name: _HELDOUT[name] for name in MODALITIES}
    return {
        "optimized_tokens": optimized_tokens,
        "train_observed_interval": _observed_interval_bpb(interval_nll, interval_tokens),
        "train_reference": _evaluate_modalities(model, tokenizer, train_reference),
        "heldout": _evaluate_modalities(model, tokenizer, heldout),
    }


def _entry_batch(
    entry: TraceEntry,
    *,
    tokenizer: ByteTokenizer,
    batch_size: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    records = {record_id: text for record_id, _modality, text in _TRAIN}
    text = records[entry.record_id]
    if _sha_text(text) != entry.record_sha256:
        raise RuntimeError("trace record content drift")
    capacity = batch_size * sequence_length
    return _make_pair_batch(
        bytes(tokenizer.encode(text)),
        causal_offset=entry.source_offset * capacity,
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_pairs=capacity,
    )


def run_candidate(
    *,
    order: tuple[TraceEntry, ...],
    seed: int,
    final_tokens: int,
    checkpoint_tokens: tuple[int, ...],
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    capacity = batch_size * sequence_length
    if final_tokens != len(order) * capacity:
        raise RuntimeError("finite trace does not exactly cover final optimized-token budget")
    if any(token % capacity for token in checkpoint_tokens):
        raise RuntimeError("checkpoint budgets must align to full DATA-35 batches")

    spec = controlled_specs()[1]
    if spec.parameter_count() != 267_912:
        raise RuntimeError("near-250K control ModelSpec drift")
    init_spec = InitSpec()
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer_config = _trainer_config(
        final_tokens=final_tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    trainer = Trainer(model, trainer_config, device="cpu")
    tokenizer = ByteTokenizer()
    checkpoints = [
        _record_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimized_tokens=0,
            interval_nll={},
            interval_tokens={},
        )
    ]
    interval_nll: dict[str, float] = defaultdict(float)
    interval_tokens: dict[str, int] = defaultdict(int)
    source_tokens: Counter[str] = Counter()
    checkpoint_set = set(checkpoint_tokens)

    for entry in order:
        before = trainer.tokens_seen
        metrics = trainer.train_microbatch(
            _entry_batch(
                entry,
                tokenizer=tokenizer,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        )
        if metrics.tokens != capacity or trainer.tokens_seen - before != capacity:
            raise RuntimeError("Trainer optimized-token ledger drift")
        interval_nll[entry.source] += float(metrics.loss) * int(metrics.tokens)
        interval_tokens[entry.source] += int(metrics.tokens)
        source_tokens[entry.source] += int(metrics.tokens)
        if trainer.tokens_seen in checkpoint_set:
            checkpoints.append(
                _record_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimized_tokens=trainer.tokens_seen,
                    interval_nll=interval_nll,
                    interval_tokens=interval_tokens,
                )
            )
            interval_nll = defaultdict(float)
            interval_tokens = defaultdict(int)

    if trainer.tokens_seen != final_tokens:
        raise RuntimeError(f"final token ledger drift: {trainer.tokens_seen} != {final_tokens}")
    if checkpoints[-1]["optimized_tokens"] != final_tokens:
        raise RuntimeError("final evaluation checkpoint missing")
    return {
        "seed": seed,
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "modality_optimized_tokens": dict(sorted(source_tokens.items())),
        "checkpoints": checkpoints,
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires values")
    return statistics.fmean(materialized)


def _final_heldout(run: dict[str, Any]) -> dict[str, Any]:
    return run["checkpoints"][-1]["heldout"]


def _decision(
    runs: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    baseline = runs["fully_mixed"]
    rules = config["decision_rule"]
    results: dict[str, Any] = {}
    accepted: list[str] = []
    for candidate in ("quality_first_then_mixed", "ukrainian_first_then_mixed"):
        candidate_runs = runs[candidate]
        deltas: list[float] = []
        modality_deltas: dict[str, list[float]] = {name: [] for name in MODALITIES}
        for baseline_run, candidate_run in zip(baseline, candidate_runs, strict=True):
            baseline_final = _final_heldout(baseline_run)
            candidate_final = _final_heldout(candidate_run)
            deltas.append(
                float(candidate_final["aggregate"]["bpb"])
                - float(baseline_final["aggregate"]["bpb"])
            )
            for modality in MODALITIES:
                modality_deltas[modality].append(
                    float(candidate_final[modality]["bpb"])
                    - float(baseline_final[modality]["bpb"])
                )
        mean_delta = _mean(deltas)
        mean_modality = {name: _mean(values) for name, values in modality_deltas.items()}
        seed_wins = sum(delta < 0.0 for delta in deltas)
        passes = (
            mean_delta <= float(rules["accept_if_mean_delta_bpb_at_most"])
            and seed_wins >= int(rules["minimum_seed_wins_out_of_3"])
            and max(mean_modality.values())
            <= float(rules["maximum_allowed_mean_modality_regression_bpb"])
        )
        results[candidate] = {
            "paired_final_aggregate_delta_bpb_by_seed": deltas,
            "mean_paired_final_aggregate_delta_bpb": mean_delta,
            "seed_wins": seed_wins,
            "mean_modality_delta_bpb": mean_modality,
            "passes_predeclared_rule": passes,
        }
        if passes:
            accepted.append(candidate)

    if accepted:
        recommendation = min(
            accepted,
            key=lambda name: results[name]["mean_paired_final_aggregate_delta_bpb"],
        )
        verdict = "ACCEPT_SIMPLE_CURRICULUM_PROVISIONALLY"
    else:
        recommendation = "fully_mixed"
        verdict = "REJECT_CURRICULUM_FOR_NEXT_SMALL_MODEL_CAMPAIGN"
    return {
        "verdict": verdict,
        "recommended_schedule": recommendation,
        "candidate_tests": results,
        "rule": rules,
    }


def run_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be exact lowercase 40-hex")
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("DATA-35 exact source checkout mismatch")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    config = _load_config(config_path)
    train_sha = _assert_current_train_snapshot(repo_root)
    batch_size = int(config["base_control"]["batch_size"])
    sequence_length = int(config["base_control"]["sequence_length"])
    final_tokens = int(config["budget"]["optimized_loss_tokens"])
    prefix_tokens = int(config["budget"]["curriculum_prefix_tokens"])
    capacity = batch_size * sequence_length
    if final_tokens % capacity or prefix_tokens % capacity:
        raise RuntimeError("DATA-35 budgets must align to full aligned-pair batches")
    steps = final_tokens // capacity
    prefix_steps = prefix_tokens // capacity
    checkpoint_tokens = tuple(int(value) for value in config["budget"]["checkpoints"])
    if checkpoint_tokens[-1] != final_tokens:
        raise RuntimeError("final checkpoint must equal final optimized-token budget")

    plan = build_incumbent_plan(config)
    trace = materialize_incumbent_trace(plan, steps=steps, tokens_per_step=capacity)
    candidates: tuple[Candidate, ...] = (
        "fully_mixed",
        "quality_first_then_mixed",
        "ukrainian_first_then_mixed",
    )
    orders = {
        candidate: order_trace(trace, candidate=candidate, prefix_steps=prefix_steps)
        for candidate in candidates
    }
    reference_multiset = Counter(entry.identity() for entry in trace)
    for order in orders.values():
        if Counter(entry.identity() for entry in order) != reference_multiset:
            raise RuntimeError("candidate changed exact training-envelope multiset")
    if tuple(entry.source for entry in orders["quality_first_then_mixed"][:prefix_steps]) != tuple(
        entry.source for entry in trace[:prefix_steps]
    ):
        raise RuntimeError("quality curriculum introduced a modality-ordering confound")

    schedule_summaries: dict[str, Any] = {}
    for candidate, order in orders.items():
        prefix = order[:prefix_steps]
        schedule_summaries[candidate] = {
            "order_sha256": _order_identity(order),
            "prefix_source_batches": dict(sorted(Counter(e.source for e in prefix).items())),
            "full_source_batches": dict(sorted(Counter(e.source for e in order).items())),
            "full_modality_optimized_tokens": {
                name: sum(entry.source == name for entry in order) * capacity
                for name in MODALITIES
            },
            "mean_prefix_quality_score": _mean(e.quality_score for e in prefix),
            "mean_full_quality_score": _mean(e.quality_score for e in order),
        }
    fixed_counts = {
        tuple(sorted(summary["full_modality_optimized_tokens"].items()))
        for summary in schedule_summaries.values()
    }
    if len(fixed_counts) != 1:
        raise RuntimeError("final modality token counts differ across candidates")

    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in candidates}
    for seed in (int(value) for value in config["paired_model_seeds"]):
        for candidate in candidates:
            runs[candidate].append(
                run_candidate(
                    order=orders[candidate],
                    seed=seed,
                    final_tokens=final_tokens,
                    checkpoint_tokens=checkpoint_tokens,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
            )

    modality_counts = runs["fully_mixed"][0]["modality_optimized_tokens"]
    for candidate_runs in runs.values():
        for candidate_run in candidate_runs:
            if candidate_run["modality_optimized_tokens"] != modality_counts:
                raise RuntimeError("executed final modality token counts are not identical")
            if candidate_run["optimized_tokens"] != final_tokens:
                raise RuntimeError("executed total optimized-token budget drift")

    spec = controlled_specs()[1]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "parent_control_sha": config["base_control"]["parent_sha"],
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "train_fixture_sha256": train_sha,
        "heldout_registry_sha256": hash_json(
            {"data10_source_sha": DATA10_SOURCE_SHA, "heldout": _HELDOUT}
        ),
        "heldout_used_for_schedule_selection": False,
        "model": {
            "parameters": spec.parameter_count(),
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": InitSpec().identity_sha256(),
        },
        "tokenizer": {
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
            "kind": "byte",
        },
        "optimizer_recipe": config["base_control"]["optimizer_recipe"],
        "precision": config["base_control"]["precision"],
        "total_optimized_tokens_per_run": final_tokens,
        "incumbent_mixture_plan_sha256": plan.sha256,
        "incumbent_trace_multiset_sha256": _trace_multiset_identity(trace),
        "schedule_summaries": schedule_summaries,
        "executed_modality_optimized_tokens": modality_counts,
        "paired_model_seeds": config["paired_model_seeds"],
        "runs": runs,
        "decision": _decision(runs, config),
        "truth_boundary": config["truth_boundary"],
        "paid_compute_used": False,
        "representative_corpus_evidence": False,
    }
    payload["report_sha256"] = hash_json(payload)
    return payload


def validate_report(payload: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("DATA-35 evidence schema mismatch")
    observed_hash = payload.get("report_sha256")
    body = dict(payload)
    body.pop("report_sha256", None)
    if observed_hash != hash_json(body):
        raise RuntimeError("DATA-35 report self-hash mismatch")
    if expected_source_sha is not None and payload.get("source_sha") != expected_source_sha:
        raise RuntimeError("DATA-35 report source SHA mismatch")
    if payload.get("paid_compute_used") is not False:
        raise RuntimeError("DATA-35 LOCAL_FREE truth boundary drift")
    final_counts = {
        tuple(sorted(summary["full_modality_optimized_tokens"].items()))
        for summary in payload["schedule_summaries"].values()
    }
    if len(final_counts) != 1:
        raise RuntimeError("DATA-35 report has unequal final modality token counts")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/data35_curriculum_v1.json"),
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--torch-threads", type=int, default=2)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args()

    if args.command == "run":
        payload = run_experiment(
            repo_root=args.repo_root,
            source_sha=args.source_sha,
            config_path=args.config,
            torch_threads=args.torch_threads,
        )
        validate_report(payload, expected_source_sha=args.source_sha)
        _write_report(args.output, payload)
        print(
            json.dumps(
                {
                    "report_sha256": payload["report_sha256"],
                    "decision": payload["decision"],
                    "modality_tokens": payload["executed_modality_optimized_tokens"],
                    "schedule_summaries": payload["schedule_summaries"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    validate_report(payload, expected_source_sha=args.expected_source_sha)
    print(
        json.dumps(
            {"report_sha256": payload["report_sha256"], "decision": payload["decision"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
