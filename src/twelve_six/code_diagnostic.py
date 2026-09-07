"""Reserved raw-Base code diagnostic scoring primitives for EVAL-134."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
SUITE_SCHEMA = "12-6.code-diagnostic-suite.v1"
REPORT_SCHEMA = "12-6.code-diagnostic-score.v1"


class DiagnosticError(ValueError):
    """Fail-closed diagnostic contract error."""


class TokenizerLike(Protocol):
    vocab_size: int

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class CodeProbe:
    id: str
    category: str
    language: str
    prefix: str
    completion: str
    distractors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all((self.id, self.category, self.language, self.prefix, self.completion)):
            raise DiagnosticError("probe metadata/prefix/completion must be non-empty")
        if not self.distractors:
            raise DiagnosticError(f"{self.id}: at least one distractor is required")
        if self.completion in self.distractors:
            raise DiagnosticError(f"{self.id}: completion duplicated as distractor")

    @property
    def choices(self) -> tuple[str, ...]:
        return (self.completion, *self.distractors)


@dataclass(frozen=True, slots=True)
class CompletionScore:
    text: str
    is_correct: bool
    log_likelihood_nats: float
    nll_nats: float
    nll_per_target_token: float
    nll_per_source_byte: float
    bits_per_source_byte: float
    target_tokens: int
    target_bytes: int
    byte_token_count: int
    tokenizer_tokens_per_source_byte: float
    exact_next_token: bool
    forced_boundary_matches_joint_encoding: bool


@dataclass(frozen=True, slots=True)
class ProbeScore:
    id: str
    category: str
    language: str
    choices: tuple[CompletionScore, ...]
    raw_correct: bool
    byte_normalized_correct: bool
    raw_log_likelihood_margin_nats: float
    byte_normalized_margin_nats_per_byte: float


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_suite(path: Path) -> tuple[CodeProbe, ...]:
    probes: list[CodeProbe] = []
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        probe = CodeProbe(
            id=row["id"],
            category=row["category"],
            language=row["language"],
            prefix=row["prefix"],
            completion=row["completion"],
            distractors=tuple(row["distractors"]),
        )
        if probe.id in ids:
            raise DiagnosticError(f"duplicate probe id at line {line_number}: {probe.id}")
        ids.add(probe.id)
        probes.append(probe)
    if not probes:
        raise DiagnosticError("empty diagnostic suite")
    return tuple(probes)


def suite_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_max_seq_len(model: torch.nn.Module) -> int:
    spec = getattr(model, "spec", None)
    value = getattr(spec, "max_seq_len", None)
    if not isinstance(value, int) or value <= 1:
        raise DiagnosticError("model.spec.max_seq_len is required")
    return value


def score_completion(
    model: torch.nn.Module,
    tokenizer: TokenizerLike,
    prefix: str,
    completion: str,
    *,
    is_correct: bool,
) -> CompletionScore:
    """Score an explicit tokenizer-boundary continuation without retokenizing prefix."""
    prefix_ids = tokenizer.encode(prefix)
    target_ids = tokenizer.encode(completion)
    if not prefix_ids or not target_ids:
        raise DiagnosticError("prefix and completion must each emit at least one token")
    ids = prefix_ids + target_ids
    if len(ids) > _model_max_seq_len(model):
        raise DiagnosticError("probe exceeds model context")
    if min(ids) < 0 or max(ids) >= tokenizer.vocab_size:
        raise DiagnosticError("tokenizer emitted out-of-vocabulary id")

    target_bytes = len(completion.encode("utf-8"))
    if target_bytes <= 0:
        raise DiagnosticError("completion must contain source bytes")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            sequence = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            logits = model(sequence).logits
            start = len(prefix_ids) - 1
            stop = len(ids) - 1
            selected = F.log_softmax(logits[0, start:stop, :], dim=-1)
            targets = torch.tensor(target_ids, dtype=torch.long).unsqueeze(1)
            token_logp = selected.gather(1, targets).squeeze(1)
            logp = float(token_logp.sum().item())
    finally:
        model.train(was_training)

    nll = -logp
    joint = tokenizer.encode(prefix + completion)
    forced = prefix_ids + target_ids
    return CompletionScore(
        text=completion,
        is_correct=is_correct,
        log_likelihood_nats=logp,
        nll_nats=nll,
        nll_per_target_token=nll / len(target_ids),
        nll_per_source_byte=nll / target_bytes,
        bits_per_source_byte=nll / (LN2 * target_bytes),
        target_tokens=len(target_ids),
        target_bytes=target_bytes,
        byte_token_count=target_bytes,
        tokenizer_tokens_per_source_byte=len(target_ids) / target_bytes,
        exact_next_token=len(target_ids) == 1,
        forced_boundary_matches_joint_encoding=joint == forced,
    )


def score_probe(
    model: torch.nn.Module, tokenizer: TokenizerLike, probe: CodeProbe
) -> ProbeScore:
    choices = tuple(
        score_completion(
            model,
            tokenizer,
            probe.prefix,
            choice,
            is_correct=index == 0,
        )
        for index, choice in enumerate(probe.choices)
    )
    correct = choices[0]
    alternatives = choices[1:]
    best_raw_alt = max(item.log_likelihood_nats for item in alternatives)
    best_norm_alt = min(item.nll_per_source_byte for item in alternatives)
    raw_winner = max(choices, key=lambda item: item.log_likelihood_nats)
    norm_winner = min(choices, key=lambda item: item.nll_per_source_byte)
    return ProbeScore(
        id=probe.id,
        category=probe.category,
        language=probe.language,
        choices=choices,
        raw_correct=raw_winner.is_correct,
        byte_normalized_correct=norm_winner.is_correct,
        raw_log_likelihood_margin_nats=correct.log_likelihood_nats - best_raw_alt,
        byte_normalized_margin_nats_per_byte=(
            best_norm_alt - correct.nll_per_source_byte
        ),
    )


def score_suite(
    model: torch.nn.Module,
    tokenizer: TokenizerLike,
    probes: Sequence[CodeProbe],
) -> tuple[ProbeScore, ...]:
    return tuple(score_probe(model, tokenizer, probe) for probe in probes)


def tokenizer_diagnostics(
    tokenizer: TokenizerLike,
    byte_tokenizer: TokenizerLike,
    probes: Sequence[CodeProbe],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        texts = [("correct", probe.completion)]
        texts.extend(
            (f"distractor_{index}", value)
            for index, value in enumerate(probe.distractors, 1)
        )
        for role, text in texts:
            source_bytes = len(text.encode("utf-8"))
            learned_ids = tokenizer.encode(text)
            byte_ids = byte_tokenizer.encode(text)
            rows.append(
                {
                    "probe_id": probe.id,
                    "role": role,
                    "source_bytes": source_bytes,
                    "bpe_tokens": len(learned_ids),
                    "byte_tokens": len(byte_ids),
                    "bpe_tokens_per_source_byte": len(learned_ids) / source_bytes,
                    "byte_tokens_per_source_byte": len(byte_ids) / source_bytes,
                    "bpe_roundtrip_boundary_matches_joint": (
                        tokenizer.encode(probe.prefix) + learned_ids
                        == tokenizer.encode(probe.prefix + text)
                    ),
                }
            )
    source_bytes = sum(row["source_bytes"] for row in rows)
    bpe_tokens = sum(row["bpe_tokens"] for row in rows)
    return {
        "rows": rows,
        "aggregate": {
            "source_bytes": source_bytes,
            "bpe_tokens": bpe_tokens,
            "byte_tokens": sum(row["byte_tokens"] for row in rows),
            "bpe_tokens_per_source_byte": bpe_tokens / source_bytes,
            "byte_tokens_per_source_byte": 1.0,
            "boundary_retokenization_changes": sum(
                not row["bpe_roundtrip_boundary_matches_joint"] for row in rows
            ),
        },
    }


def _summary_group(scores: Sequence[ProbeScore]) -> dict[str, float | int]:
    if not scores:
        raise DiagnosticError("cannot summarize empty group")
    return {
        "probes": len(scores),
        "raw_accuracy": sum(item.raw_correct for item in scores) / len(scores),
        "byte_normalized_accuracy": (
            sum(item.byte_normalized_correct for item in scores) / len(scores)
        ),
        "mean_raw_log_likelihood_margin_nats": (
            sum(item.raw_log_likelihood_margin_nats for item in scores) / len(scores)
        ),
        "mean_byte_normalized_margin_nats_per_byte": (
            sum(item.byte_normalized_margin_nats_per_byte for item in scores)
            / len(scores)
        ),
        "mean_correct_bits_per_source_byte": (
            sum(item.choices[0].bits_per_source_byte for item in scores) / len(scores)
        ),
    }


def summarize(scores: Sequence[ProbeScore]) -> dict[str, Any]:
    category: dict[str, list[ProbeScore]] = defaultdict(list)
    language: dict[str, list[ProbeScore]] = defaultdict(list)
    for score in scores:
        category[score.category].append(score)
        language[score.language].append(score)
    return {
        "overall": _summary_group(scores),
        "by_category": {
            name: _summary_group(group) for name, group in sorted(category.items())
        },
        "by_language": {
            name: _summary_group(group) for name, group in sorted(language.items())
        },
    }


def serializable_scores(scores: Iterable[ProbeScore]) -> list[dict[str, Any]]:
    return [asdict(score) for score in scores]
