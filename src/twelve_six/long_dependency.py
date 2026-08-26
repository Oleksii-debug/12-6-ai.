"""Reserved synthetic long-distance dependency diagnostics for raw 12-6 Base models.

The suite is tokenization-conditioned and evaluation-only. It never emits training
records and never asks a model to follow instructions. Each probe scores a single
next-token choice whose correct token recurs at an exact earlier token distance.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from twelve_six.inference.contracts import InferenceBackend

SUITE_SCHEMA = "12-6.eval135-long-dependency.v1"
SUITE_ID = "EVAL-135-LONG-DEPENDENCY-v1"
GENERATOR_VERSION = "eval135-token-conditioned-v1"
DEFAULT_DISTANCES = (32, 64, 128, 256, 512)
SHORT_CONTROL_DISTANCE = 16
DEFAULT_CASES_PER_FAMILY_DISTANCE = 16
DEFAULT_SEED = 135_001
FAMILIES = ("delayed_symbol", "key_value_recurrence", "natural_reference")
SOURCE_PURPOSE = "evaluation_test"
TRAINING_ALLOWED = False


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProbeCase:
    case_id: str
    role: str
    family: str
    dependency_distance: int
    source_index: int
    target_index: int
    prefix_ids: tuple[int, ...]
    target_id: int
    foil_id: int
    shuffled_prefix_ids: tuple[int, ...]
    truncate_tokens: int

    def __post_init__(self) -> None:
        if self.source_index != 0:
            raise ValueError("EVAL-135 v1 requires the dependency source at token index 0")
        if self.target_index != self.dependency_distance:
            raise ValueError("target index must equal the declared dependency distance")
        if len(self.prefix_ids) != self.target_index:
            raise ValueError("prefix length must equal the target index")
        if len(self.shuffled_prefix_ids) != len(self.prefix_ids):
            raise ValueError("shuffled control must preserve prefix length")
        if self.prefix_ids[0] != self.target_id:
            raise ValueError("full prefix must carry the correct distant source token")
        if self.shuffled_prefix_ids[0] != self.foil_id:
            raise ValueError("shuffled prefix must carry the foil distant source token")
        if not 0 < self.truncate_tokens < self.dependency_distance:
            raise ValueError("truncation must remove the dependency source")
        if self.target_id == self.foil_id:
            raise ValueError("target and foil must differ")

    @property
    def truncated_prefix_ids(self) -> tuple[int, ...]:
        return self.prefix_ids[-self.truncate_tokens :]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "family": self.family,
            "dependency_distance": self.dependency_distance,
            "source_index": self.source_index,
            "target_index": self.target_index,
            "prefix_ids": list(self.prefix_ids),
            "target_id": self.target_id,
            "foil_id": self.foil_id,
            "shuffled_prefix_ids": list(self.shuffled_prefix_ids),
            "truncate_tokens": self.truncate_tokens,
        }


@dataclass(frozen=True, slots=True)
class MaterializedSuite:
    suite_identity_sha256: str
    materialized_identity_sha256: str
    tokenizer_probe_signature_sha256: str
    cases: tuple[ProbeCase, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ItemScore:
    case_id: str
    role: str
    family: str
    dependency_distance: int
    condition: str
    context_tokens: int
    target_id: int
    foil_id: int
    target_nll: float
    target_bits: float
    pair_margin_nats: float
    pairwise_correct: bool
    top1_correct: bool


def suite_spec_payload(
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    cases_per_family_distance: int = DEFAULT_CASES_PER_FAMILY_DISTANCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    normalized_distances = tuple(int(value) for value in distances)
    if any(value <= SHORT_CONTROL_DISTANCE for value in normalized_distances):
        raise ValueError("primary distances must exceed the short-control distance")
    if sorted(set(normalized_distances)) != list(normalized_distances):
        raise ValueError("distances must be unique and increasing")
    if cases_per_family_distance <= 0:
        raise ValueError("cases_per_family_distance must be positive")
    return {
        "schema": SUITE_SCHEMA,
        "suite_id": SUITE_ID,
        "generator_version": GENERATOR_VERSION,
        "source_purpose": SOURCE_PURPOSE,
        "training_allowed": TRAINING_ALLOWED,
        "instruction_following": False,
        "task": "conditional_next_token_choice",
        "primary_distances_tokens": list(normalized_distances),
        "short_control_distance_tokens": SHORT_CONTROL_DISTANCE,
        "families": list(FAMILIES),
        "cases_per_family_distance": cases_per_family_distance,
        "seed": seed,
        "chance_pairwise_accuracy": 0.5,
        "distance_definition": "target_token_index_minus_dependency_source_token_index",
        "conditions": {
            "full": "all supported prefix tokens are visible",
            "truncated": "a suffix shorter than the dependency distance is visible",
            "shuffled": (
                "full positions are preserved but the distant source token is replaced by foil"
            ),
        },
        "truth_boundary": (
            "The suite measures in-range next-token dependency use only; it does not test "
            "instruction following and must not be used for training."
        ),
    }


def suite_identity_sha256(**kwargs: Any) -> str:
    return _canonical_hash(suite_spec_payload(**kwargs))


def _single_token_atoms(
    backend: InferenceBackend, literals: Iterable[str]
) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    seen_ids: set[int] = set()
    for literal in literals:
        token_ids = backend.encode(literal)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if token_id in seen_ids:
            continue
        selected.append((literal, token_id))
        seen_ids.add(token_id)
    return selected


def _encode(backend: InferenceBackend, text: str) -> tuple[int, ...]:
    token_ids = tuple(int(token_id) for token_id in backend.encode(text))
    if not token_ids:
        raise ValueError(f"probe fragment encoded to zero tokens: {text!r}")
    return token_ids


def _template_tokens(
    backend: InferenceBackend,
    family: str,
    variant: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    key = "abcd"[variant % 4]
    if family == "delayed_symbol":
        return _encode(backend, "|s|"), _encode(backend, "|s=")
    if family == "key_value_recurrence":
        return _encode(backend, f"|{key}|"), _encode(backend, f"|{key}=")
    if family == "natural_reference":
        name = ("na", "be", "ci", "do")[variant % 4]
        return _encode(backend, f" marks {name}."), _encode(backend, f" {name} marks ")
    raise ValueError(f"unknown family: {family}")


def materialize_suite(
    backend: InferenceBackend,
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    cases_per_family_distance: int = DEFAULT_CASES_PER_FAMILY_DISTANCE,
    seed: int = DEFAULT_SEED,
) -> MaterializedSuite:
    spec = suite_spec_payload(
        distances=distances,
        cases_per_family_distance=cases_per_family_distance,
        seed=seed,
    )
    value_atoms = _single_token_atoms(
        backend,
        "QZJXVKWRMPTF",
    )
    filler_atoms = _single_token_atoms(
        backend,
        ".,;:-_0123456789",
    )
    if len(value_atoms) < 4:
        raise ValueError("tokenizer must provide at least four distinct single-token value atoms")
    if len(filler_atoms) < 2:
        raise ValueError("tokenizer must provide at least two distinct single-token filler atoms")
    value_atoms = value_atoms[:8]
    value_ids = {token_id for _, token_id in value_atoms}
    filler_atoms = [item for item in filler_atoms if item[1] not in value_ids]
    if not filler_atoms:
        raise ValueError("value and filler token atoms must be disjoint")

    rng = random.Random(seed)
    cases: list[ProbeCase] = []
    all_distances = (SHORT_CONTROL_DISTANCE, *tuple(int(value) for value in distances))
    skipped_templates: list[dict[str, Any]] = []
    for distance in all_distances:
        role = "short_control" if distance == SHORT_CONTROL_DISTANCE else "long_dependency"
        for family in FAMILIES:
            for item_index in range(cases_per_family_distance):
                head, tail = _template_tokens(backend, family, item_index)
                required_fixed = 1 + len(head) + len(tail)
                filler_count = distance - required_fixed
                if filler_count < 0:
                    skipped_templates.append(
                        {"family": family, "distance": distance, "required_fixed": required_fixed}
                    )
                    continue
                target_slot = (item_index + distance // max(SHORT_CONTROL_DISTANCE, 1)) % len(
                    value_atoms
                )
                target_id = value_atoms[target_slot][1]
                foil_id = value_atoms[(target_slot + 1 + item_index % 3) % len(value_atoms)][1]
                if target_id == foil_id:
                    raise RuntimeError("balanced foil construction collapsed")
                filler_token_id = filler_atoms[rng.randrange(len(filler_atoms))][1]
                prefix = (
                    (target_id,)
                    + head
                    + (filler_token_id,) * filler_count
                    + tail
                )
                if len(prefix) != distance:
                    raise RuntimeError("exact dependency distance construction failed")
                shuffled = (foil_id,) + prefix[1:]
                truncate_tokens = max(4, distance // 2)
                truncate_tokens = min(truncate_tokens, distance - 1)
                case_id = f"{SUITE_ID}:{role}:{family}:d{distance}:i{item_index:02d}"
                cases.append(
                    ProbeCase(
                        case_id=case_id,
                        role=role,
                        family=family,
                        dependency_distance=distance,
                        source_index=0,
                        target_index=distance,
                        prefix_ids=prefix,
                        target_id=target_id,
                        foil_id=foil_id,
                        shuffled_prefix_ids=shuffled,
                        truncate_tokens=truncate_tokens,
                    )
                )

    if not cases:
        raise RuntimeError("no probe cases were materialized")
    signature_payload = {
        "value_atoms": value_atoms,
        "filler_atoms": filler_atoms,
        "fragment_encodings": {
            family: [
                list(_template_tokens(backend, family, variant)[0]),
                list(_template_tokens(backend, family, variant)[1]),
            ]
            for family in FAMILIES
            for variant in (0,)
        },
    }
    materialized_payload = {
        "suite_identity_sha256": _canonical_hash(spec),
        "tokenizer_probe_signature": signature_payload,
        "cases": [case.identity_payload() for case in cases],
    }
    metadata = {
        **spec,
        "materialized_case_count": len(cases),
        "skipped_template_count": len(skipped_templates),
        "skipped_templates": skipped_templates,
        "backend_max_context_tokens": int(backend.max_context_tokens),
    }
    return MaterializedSuite(
        suite_identity_sha256=_canonical_hash(spec),
        materialized_identity_sha256=_canonical_hash(materialized_payload),
        tokenizer_probe_signature_sha256=_canonical_hash(signature_payload),
        cases=tuple(cases),
        metadata=metadata,
    )


def _score_logits(
    logits: Sequence[float],
    *,
    target_id: int,
    foil_id: int,
) -> tuple[float, float, float, bool, bool]:
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("backend returned empty next-token logits")
    if not 0 <= target_id < len(values) or not 0 <= foil_id < len(values):
        raise ValueError("probe target/foil token is outside backend vocabulary")
    maximum = max(values)
    log_partition = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    target_nll = log_partition - values[target_id]
    margin = values[target_id] - values[foil_id]
    top1_id = max(range(len(values)), key=values.__getitem__)
    return (
        target_nll,
        target_nll / math.log(2.0),
        margin,
        margin > 0.0,
        top1_id == target_id,
    )


def _score_condition(
    backend: InferenceBackend,
    case: ProbeCase,
    condition: str,
) -> ItemScore:
    if condition == "full":
        prefix = case.prefix_ids
    elif condition == "truncated":
        prefix = case.truncated_prefix_ids
    elif condition == "shuffled":
        prefix = case.shuffled_prefix_ids
    else:
        raise ValueError(f"unknown condition: {condition}")
    if len(prefix) > backend.max_context_tokens:
        raise ValueError("attempted to score beyond backend max context")
    logits = backend.next_token_logits(prefix)
    target_nll, target_bits, margin, pairwise_correct, top1_correct = _score_logits(
        logits,
        target_id=case.target_id,
        foil_id=case.foil_id,
    )
    return ItemScore(
        case_id=case.case_id,
        role=case.role,
        family=case.family,
        dependency_distance=case.dependency_distance,
        condition=condition,
        context_tokens=len(prefix),
        target_id=case.target_id,
        foil_id=case.foil_id,
        target_nll=target_nll,
        target_bits=target_bits,
        pair_margin_nats=margin,
        pairwise_correct=pairwise_correct,
        top1_correct=top1_correct,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return sum(values) / len(values)


def _summarize(items: Sequence[ItemScore]) -> dict[str, Any]:
    return {
        "count": len(items),
        "mean_target_nll": _mean([item.target_nll for item in items]),
        "mean_target_bits": _mean([item.target_bits for item in items]),
        "mean_pair_margin_nats": _mean([item.pair_margin_nats for item in items]),
        "pairwise_accuracy": _mean([float(item.pairwise_correct) for item in items]),
        "top1_accuracy": _mean([float(item.top1_correct) for item in items]),
    }


def _condition_delta(
    full: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, float]:
    return {
        "target_nll_gain": control["mean_target_nll"] - full["mean_target_nll"],
        "pair_margin_gain_nats": (
            full["mean_pair_margin_nats"] - control["mean_pair_margin_nats"]
        ),
        "pairwise_accuracy_gain": full["pairwise_accuracy"] - control["pairwise_accuracy"],
    }


def score_suite(
    backend: InferenceBackend,
    suite: MaterializedSuite,
    *,
    model_label: str,
) -> dict[str, Any]:
    if backend.max_context_tokens <= 0:
        raise ValueError("backend max_context_tokens must be positive")
    scores: list[ItemScore] = []
    unsupported: dict[int, int] = defaultdict(int)
    for case in suite.cases:
        if case.dependency_distance > backend.max_context_tokens:
            unsupported[case.dependency_distance] += 1
            continue
        for condition in ("full", "truncated", "shuffled"):
            scores.append(_score_condition(backend, case, condition))

    grouped: dict[tuple[str, str, int, str], list[ItemScore]] = defaultdict(list)
    distance_grouped: dict[tuple[str, int, str], list[ItemScore]] = defaultdict(list)
    for item in scores:
        grouped[(item.role, item.family, item.dependency_distance, item.condition)].append(item)
        distance_grouped[(item.role, item.dependency_distance, item.condition)].append(item)

    by_family_distance: list[dict[str, Any]] = []
    for role in ("short_control", "long_dependency"):
        distances = sorted({item.dependency_distance for item in scores if item.role == role})
        for family in FAMILIES:
            for distance in distances:
                keys = {
                    condition: (role, family, distance, condition)
                    for condition in ("full", "truncated", "shuffled")
                }
                if any(not grouped[key] for key in keys.values()):
                    continue
                summaries = {
                    condition: _summarize(grouped[key])
                    for condition, key in keys.items()
                }
                by_family_distance.append(
                    {
                        "role": role,
                        "family": family,
                        "dependency_distance": distance,
                        "conditions": summaries,
                        "full_vs_truncated": _condition_delta(
                            summaries["full"], summaries["truncated"]
                        ),
                        "full_vs_shuffled": _condition_delta(
                            summaries["full"], summaries["shuffled"]
                        ),
                    }
                )

    by_distance: list[dict[str, Any]] = []
    for role in ("short_control", "long_dependency"):
        distances = sorted({item.dependency_distance for item in scores if item.role == role})
        for distance in distances:
            keys = {
                condition: (role, distance, condition)
                for condition in ("full", "truncated", "shuffled")
            }
            if any(not distance_grouped[key] for key in keys.values()):
                continue
            summaries = {
                condition: _summarize(distance_grouped[key])
                for condition, key in keys.items()
            }
            by_distance.append(
                {
                    "role": role,
                    "dependency_distance": distance,
                    "conditions": summaries,
                    "full_vs_truncated": _condition_delta(
                        summaries["full"], summaries["truncated"]
                    ),
                    "full_vs_shuffled": _condition_delta(
                        summaries["full"], summaries["shuffled"]
                    ),
                }
            )

    short_rows = [row for row in by_distance if row["role"] == "short_control"]
    short_pairwise = (
        short_rows[0]["conditions"]["full"]["pairwise_accuracy"] if short_rows else None
    )
    interpretation: dict[str, Any]
    if short_pairwise is None:
        interpretation = {
            "status": "short_control_unsupported",
            "usable_long_dependency_claim": False,
        }
    elif short_pairwise <= 0.55:
        interpretation = {
            "status": "probe_format_not_resolved_at_short_control",
            "usable_long_dependency_claim": False,
            "short_control_pairwise_accuracy": short_pairwise,
            "note": (
                "Long-distance failures are not isolated context failures because the model "
                "does not reliably solve the matched short control."
            ),
        }
    else:
        positive_distances: list[int] = []
        for row in by_distance:
            if row["role"] != "long_dependency":
                continue
            full = row["conditions"]["full"]
            trunc = row["full_vs_truncated"]
            shuffled = row["full_vs_shuffled"]
            if (
                full["pairwise_accuracy"] > 0.5
                and trunc["pair_margin_gain_nats"] > 0.0
                and shuffled["pair_margin_gain_nats"] > 0.0
            ):
                positive_distances.append(row["dependency_distance"])
        interpretation = {
            "status": "interpretable",
            "usable_long_dependency_claim": bool(positive_distances),
            "positive_signal_distances_tokens": positive_distances,
            "short_control_pairwise_accuracy": short_pairwise,
        }

    report: dict[str, Any] = {
        "schema": SUITE_SCHEMA,
        "suite_id": SUITE_ID,
        "suite_identity_sha256": suite.suite_identity_sha256,
        "materialized_identity_sha256": suite.materialized_identity_sha256,
        "tokenizer_probe_signature_sha256": suite.tokenizer_probe_signature_sha256,
        "model": {
            "label": model_label,
            "max_context_tokens": int(backend.max_context_tokens),
        },
        "evaluation": {
            "training_allowed": False,
            "instruction_following": False,
            "extrapolation_attempted": False,
            "conditions": ["full", "truncated", "shuffled"],
            "pairwise_chance_accuracy": 0.5,
            "scored_item_condition_count": len(scores),
            "unsupported_case_counts_by_distance": {
                str(distance): count for distance, count in sorted(unsupported.items())
            },
        },
        "by_family_distance": by_family_distance,
        "by_distance": by_distance,
        "interpretation": interpretation,
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != SUITE_SCHEMA or report.get("suite_id") != SUITE_ID:
        raise ValueError("unexpected EVAL-135 report identity")
    evaluation = report.get("evaluation", {})
    if evaluation.get("training_allowed") is not False:
        raise ValueError("EVAL-135 reports must remain evaluation-only")
    if evaluation.get("instruction_following") is not False:
        raise ValueError("EVAL-135 is not an instruction-following benchmark")
    if evaluation.get("extrapolation_attempted") is not False:
        raise ValueError("EVAL-135 must not score beyond trained/supported context")
    claimed = report.get("report_sha256")
    if not isinstance(claimed, str):
        raise ValueError("missing report hash")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if claimed != _canonical_hash(payload):
        raise ValueError("report hash mismatch")
