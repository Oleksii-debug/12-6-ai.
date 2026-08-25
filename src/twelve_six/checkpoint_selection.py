"""Preregistered experiment-level checkpoint selection for 12-6.

Selection is deliberately separated from training, final-test, and diagnostic
metrics.  A selection decision can consume only observations bound to one
immutable selection-validation purpose identity.  Final-test evidence is
attached after the decision and cannot affect the selected checkpoint or the
selection-decision identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

SCHEMA = "12-6.checkpoint-selection.v1"
PURPOSE_SCHEMA = "12-6.evaluation-purpose.v1"
RULE_SCHEMA = "12-6.checkpoint-selection-rule.v1"

PurposeKind = Literal[
    "training_metrics", "selection_validation", "final_test", "diagnostic_only"
]
_ALLOWED_PURPOSES = frozenset(
    {"training_metrics", "selection_validation", "final_test", "diagnostic_only"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class EvaluationPurpose:
    """Immutable identity for one evaluation use and one exact evidence suite."""

    purpose: PurposeKind
    suite_id: str
    suite_identity_sha256: str
    metric_names: tuple[str, ...]
    selection_eligible: bool
    schema: str = PURPOSE_SCHEMA
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema != PURPOSE_SCHEMA:
            raise ValueError("evaluation-purpose schema drift")
        if self.purpose not in _ALLOWED_PURPOSES:
            raise ValueError(f"unknown evaluation purpose: {self.purpose}")
        if not self.suite_id.strip():
            raise ValueError("suite_id is required")
        if (
            len(self.suite_identity_sha256) != 64
            or self.suite_identity_sha256 != self.suite_identity_sha256.lower()
            or any(ch not in "0123456789abcdef" for ch in self.suite_identity_sha256)
        ):
            raise ValueError("suite_identity_sha256 must be exact lowercase sha256")
        if not self.metric_names or any(not name.strip() for name in self.metric_names):
            raise ValueError("metric_names must be non-empty")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be unique")
        expected = self.purpose == "selection_validation"
        if self.selection_eligible is not expected:
            raise ValueError(
                "selection_eligible must be true only for selection_validation"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "purpose": self.purpose,
            "suite_id": self.suite_id,
            "suite_identity_sha256": self.suite_identity_sha256,
            "metric_names": list(self.metric_names),
            "selection_eligible": self.selection_eligible,
        }

    @property
    def identity_sha256(self) -> str:
        return hash_json(self.identity_payload())


@dataclass(frozen=True)
class CheckpointRef:
    """Stable experiment-level checkpoint identity and ordering metadata."""

    checkpoint_id: str
    ordinal: int
    optimizer_step: int
    optimized_tokens: int
    artifact_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id is required")
        for name, value in (
            ("ordinal", self.ordinal),
            ("optimizer_step", self.optimizer_step),
            ("optimized_tokens", self.optimized_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.artifact_identity_sha256 is not None and (
            len(self.artifact_identity_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.artifact_identity_sha256)
        ):
            raise ValueError("artifact_identity_sha256 must be lowercase sha256")


@dataclass(frozen=True)
class SelectionValidationObservation:
    """The only observation type accepted by the selector."""

    checkpoint_id: str
    purpose_identity_sha256: str
    metric_name: str
    value: float

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id is required")
        if (
            len(self.purpose_identity_sha256) != 64
            or self.purpose_identity_sha256 != self.purpose_identity_sha256.lower()
            or any(ch not in "0123456789abcdef" for ch in self.purpose_identity_sha256)
        ):
            raise ValueError("purpose_identity_sha256 must be exact lowercase sha256")
        if not self.metric_name.strip():
            raise ValueError("metric_name is required")
        _finite(self.value, "value")


@dataclass(frozen=True)
class MetricObservation:
    """Non-selection evidence retained for reporting after selection is frozen."""

    checkpoint_id: str
    purpose_identity_sha256: str
    metric_name: str
    value: float

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id is required")
        if (
            len(self.purpose_identity_sha256) != 64
            or self.purpose_identity_sha256 != self.purpose_identity_sha256.lower()
            or any(ch not in "0123456789abcdef" for ch in self.purpose_identity_sha256)
        ):
            raise ValueError("purpose_identity_sha256 must be exact lowercase sha256")
        if not self.metric_name.strip():
            raise ValueError("metric_name is required")
        _finite(self.value, "value")


@dataclass(frozen=True)
class SelectionRule:
    """Preregistered rule. Defaults are the canonical EVAL-139 v1 policy."""

    metric_name: str = "bpb"
    direction: Literal["minimize"] = "minimize"
    smoother: Literal["trailing_median"] = "trailing_median"
    smoothing_window: int = 3
    minimum_improvement: float = 0.01
    schema: str = RULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RULE_SCHEMA:
            raise ValueError("selection-rule schema drift")
        if not self.metric_name.strip():
            raise ValueError("metric_name is required")
        if self.direction != "minimize":
            raise ValueError("v1 supports minimize only")
        if self.smoother != "trailing_median":
            raise ValueError("v1 supports trailing_median only")
        if (
            not isinstance(self.smoothing_window, int)
            or isinstance(self.smoothing_window, bool)
            or self.smoothing_window < 1
            or self.smoothing_window % 2 == 0
        ):
            raise ValueError("smoothing_window must be a positive odd integer")
        threshold = _finite(self.minimum_improvement, "minimum_improvement")
        if threshold < 0:
            raise ValueError("minimum_improvement must be >= 0")

    @property
    def identity_sha256(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class SelectionDecision:
    selected_checkpoint_id: str
    checkpoint_registry: tuple[CheckpointRef, ...]
    selection_observations: tuple[SelectionValidationObservation, ...]
    trace: tuple[Mapping[str, Any], ...]
    selection_purpose: EvaluationPurpose
    rule: SelectionRule
    absolute_posthoc_best_validation_checkpoint_id: str
    final_checkpoint_id: str
    decision_identity_sha256: str
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "selection_decision_sha256": self.decision_identity_sha256,
            "selection_purpose": {
                **self.selection_purpose.identity_payload(),
                "purpose_identity_sha256": self.selection_purpose.identity_sha256,
            },
            "selection_rule": {
                **asdict(self.rule),
                "rule_identity_sha256": self.rule.identity_sha256,
            },
            "checkpoint_registry": [asdict(item) for item in self.checkpoint_registry],
            "selection_observations": [asdict(item) for item in self.selection_observations],
            "retained_checkpoint_ids": [item.checkpoint_id for item in self.checkpoint_registry],
            "delete_unselected_checkpoints": False,
            "trace": [dict(item) for item in self.trace],
            "posthoc_comparison": {
                "absolute_best_validation_checkpoint_id": (
                    self.absolute_posthoc_best_validation_checkpoint_id
                ),
                "final_checkpoint_id": self.final_checkpoint_id,
                "posthoc_best_used_for_selection": False,
                "final_checkpoint_used_for_selection": False,
            },
        }


def make_evaluation_purpose(
    purpose: PurposeKind,
    *,
    suite_id: str,
    suite_identity_sha256: str,
    metric_names: Sequence[str],
    notes: str = "",
) -> EvaluationPurpose:
    """Construct a purpose identity with selection eligibility fixed by purpose."""

    return EvaluationPurpose(
        purpose=purpose,
        suite_id=suite_id,
        suite_identity_sha256=suite_identity_sha256,
        metric_names=tuple(metric_names),
        selection_eligible=purpose == "selection_validation",
        notes=notes,
    )


def _validate_checkpoint_registry(
    checkpoints: Sequence[CheckpointRef],
) -> tuple[CheckpointRef, ...]:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    ordered = tuple(checkpoints)
    ids = [item.checkpoint_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("checkpoint_id values must be unique")
    ordinals = [item.ordinal for item in ordered]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise ValueError("checkpoints must have unique increasing ordinals")
    steps = [item.optimizer_step for item in ordered]
    tokens = [item.optimized_tokens for item in ordered]
    if steps != sorted(steps) or tokens != sorted(tokens):
        raise ValueError("checkpoint step/token order must be monotone")
    return ordered


def select_checkpoint(
    checkpoints: Sequence[CheckpointRef],
    observations: Sequence[SelectionValidationObservation],
    *,
    selection_purpose: EvaluationPurpose,
    rule: SelectionRule | None = None,
) -> SelectionDecision:
    """Freeze one checkpoint using only preregistered selection-validation evidence."""

    rule = rule or SelectionRule()
    if (
        selection_purpose.purpose != "selection_validation"
        or not selection_purpose.selection_eligible
    ):
        raise ValueError("selector requires a selection_validation purpose")
    if rule.metric_name not in selection_purpose.metric_names:
        raise ValueError("selection metric is not preregistered in the purpose identity")

    ordered = _validate_checkpoint_registry(checkpoints)
    checkpoint_by_id = {item.checkpoint_id: item for item in ordered}
    observation_by_id: dict[str, SelectionValidationObservation] = {}
    for obs in observations:
        if not isinstance(obs, SelectionValidationObservation):
            raise TypeError("selector accepts SelectionValidationObservation only")
        if obs.purpose_identity_sha256 != selection_purpose.identity_sha256:
            raise ValueError("observation purpose identity is not selection-validation identity")
        if obs.metric_name != rule.metric_name:
            raise ValueError("observation metric differs from preregistered selection metric")
        if obs.checkpoint_id not in checkpoint_by_id:
            raise ValueError("observation references an unknown checkpoint")
        if obs.checkpoint_id in observation_by_id:
            raise ValueError("duplicate selection observation for checkpoint")
        observation_by_id[obs.checkpoint_id] = obs
    if len(observation_by_id) != len(ordered):
        raise ValueError("every checkpoint must have exactly one selection observation")

    raw_values = [float(observation_by_id[item.checkpoint_id].value) for item in ordered]
    trace: list[dict[str, Any]] = []
    incumbent_id: str | None = None
    incumbent_score: float | None = None
    window = rule.smoothing_window
    for index, checkpoint in enumerate(ordered):
        raw = raw_values[index]
        score: float | None = None
        accepted = False
        reason = "smoothing_warmup"
        if index + 1 >= window:
            score = float(statistics.median(raw_values[index + 1 - window : index + 1]))
            if incumbent_id is None:
                accepted = True
                reason = "first_eligible_smoothed_checkpoint"
            elif score <= float(incumbent_score) - rule.minimum_improvement:
                accepted = True
                reason = "minimum_improvement_met"
            else:
                reason = "minimum_improvement_not_met"
            if accepted:
                incumbent_id = checkpoint.checkpoint_id
                incumbent_score = score
        trace.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "ordinal": checkpoint.ordinal,
                "optimizer_step": checkpoint.optimizer_step,
                "optimized_tokens": checkpoint.optimized_tokens,
                "selection_metric_name": rule.metric_name,
                "raw_selection_metric": raw,
                "smoothed_selection_metric": score,
                "accepted_as_incumbent": accepted,
                "reason": reason,
            }
        )
    if incumbent_id is None:
        raise ValueError("not enough checkpoints to fill the preregistered smoothing window")

    posthoc_index = min(range(len(raw_values)), key=lambda idx: (raw_values[idx], idx))
    final_id = ordered[-1].checkpoint_id
    ordered_observations = tuple(observation_by_id[item.checkpoint_id] for item in ordered)
    decision_core = {
        "schema": SCHEMA,
        "selection_purpose_identity_sha256": selection_purpose.identity_sha256,
        "selection_rule_identity_sha256": rule.identity_sha256,
        "checkpoint_registry": [asdict(item) for item in ordered],
        "selection_observations": [asdict(item) for item in ordered_observations],
        "selected_checkpoint_id": incumbent_id,
        "trace": trace,
    }
    return SelectionDecision(
        selected_checkpoint_id=incumbent_id,
        checkpoint_registry=ordered,
        selection_observations=ordered_observations,
        trace=tuple(trace),
        selection_purpose=selection_purpose,
        rule=rule,
        absolute_posthoc_best_validation_checkpoint_id=ordered[posthoc_index].checkpoint_id,
        final_checkpoint_id=final_id,
        decision_identity_sha256=hash_json(decision_core),
    )


def build_experiment_selection_report(
    *,
    experiment_id: str,
    decision: SelectionDecision,
    evaluation_purposes: Sequence[EvaluationPurpose],
    nonselection_observations: Sequence[MetricObservation] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach train/final-test/diagnostic evidence after selection is frozen."""

    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    purposes = tuple(evaluation_purposes)
    purpose_by_hash = {item.identity_sha256: item for item in purposes}
    if len(purpose_by_hash) != len(purposes):
        raise ValueError("evaluation purpose identities must be unique")
    kind_counts = {kind: 0 for kind in _ALLOWED_PURPOSES}
    for purpose in purposes:
        kind_counts[purpose.purpose] += 1
    for required_singleton in ("training_metrics", "selection_validation", "final_test"):
        if kind_counts[required_singleton] != 1:
            raise ValueError(f"exactly one {required_singleton} purpose is required")
    if kind_counts["diagnostic_only"] < 1:
        raise ValueError("at least one diagnostic_only purpose is required")
    suite_hashes = [item.suite_identity_sha256 for item in purposes]
    if len(set(suite_hashes)) != len(suite_hashes):
        raise ValueError("training, selection, final-test, and diagnostic suites must be distinct")
    if decision.selection_purpose.identity_sha256 not in purpose_by_hash:
        raise ValueError("decision selection purpose not retained in purpose registry")

    checkpoint_ids = {item.checkpoint_id for item in decision.checkpoint_registry}
    retained_nonselection: list[dict[str, Any]] = []
    for obs in nonselection_observations:
        if not isinstance(obs, MetricObservation):
            raise TypeError("nonselection observations must be MetricObservation")
        purpose = purpose_by_hash.get(obs.purpose_identity_sha256)
        if purpose is None:
            raise ValueError("observation references unknown evaluation purpose")
        if purpose.purpose == "selection_validation":
            raise ValueError(
                "selection evidence must not enter the post-selection attachment channel"
            )
        if obs.metric_name not in purpose.metric_names:
            raise ValueError("observation metric not registered for its evaluation purpose")
        if obs.checkpoint_id not in checkpoint_ids:
            raise ValueError("nonselection observation references unknown checkpoint")
        if (
            purpose.purpose == "final_test"
            and obs.checkpoint_id != decision.selected_checkpoint_id
        ):
            raise ValueError("final_test may be evaluated only on the frozen selected checkpoint")
        retained_nonselection.append(asdict(obs))

    report = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "selection": decision.to_dict(),
        "evaluation_purposes": [
            {**item.identity_payload(), "purpose_identity_sha256": item.identity_sha256}
            for item in purposes
        ],
        "nonselection_observations": retained_nonselection,
        "provenance": dict(provenance or {}),
        "isolation_guarantees": {
            "selector_input_purpose": "selection_validation_only",
            "final_test_can_change_selection": False,
            "diagnostic_only_can_change_selection": False,
            "training_metrics_can_change_selection": False,
            "later_checkpoints_preserved_when_earlier_checkpoint_wins": True,
            "final_test_only_on_frozen_selected_checkpoint": True,
        },
        "retention_policy": "preserve_all_registered_checkpoints",
    }
    report["report_sha256"] = hash_json(report)
    return report
