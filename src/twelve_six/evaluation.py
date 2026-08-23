"""Evaluation, benchmark-provenance, and stage-gate primitives for 12-6 AI.

This module deliberately has no model-framework dependency. D06 consumes
machine-readable evidence emitted by model/training/data/checkpoint/inference
lanes and converts it into conservative stage-gate results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_MISSING = object()
_FORBIDDEN_HELDOUT_USES = frozenset(
    {
        "train",
        "training",
        "pretrain",
        "pretraining",
        "finetune",
        "fine-tune",
        "sft",
        "dpo",
        "rl",
        "posttrain",
        "post-training",
    }
)


class GateStatus(str, Enum):
    """Tri-state gate result. Missing evidence is never silently treated as PASS."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_TESTED = "NOT_TESTED"


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    title: str
    status: GateStatus
    reason: str
    required: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["evidence"] = dict(self.evidence)
        return result


@dataclass(frozen=True)
class S0GatePolicy:
    """Current D06 S0 gate policy.

    ``policy_status`` makes it explicit that numeric tolerances are D06's current
    proposal until the integration/audit process freezes them for promotion.
    """

    schema_version: str = "12-6.s0-gate-policy.v1"
    stage: str = "S0"
    policy_status: str = "PROPOSED_NOT_FROZEN"
    target_parameters: int = 10_000
    min_parameters: int = 8_000
    max_parameters: int = 12_000
    min_distinct_train_batches: int = 2
    max_train_validation_overlap: int = 0
    require_trained_validation_better_than_random: bool = True

    def __post_init__(self) -> None:
        if self.stage != "S0":
            raise ValueError("S0GatePolicy.stage must be S0")
        if self.target_parameters <= 0:
            raise ValueError("target_parameters must be positive")
        if self.min_parameters <= 0 or self.max_parameters < self.min_parameters:
            raise ValueError("invalid parameter range")
        if not (self.min_parameters <= self.target_parameters <= self.max_parameters):
            raise ValueError("target_parameters must lie inside parameter range")
        if self.min_distinct_train_batches < 2:
            raise ValueError("min_distinct_train_batches must be >= 2")
        if self.max_train_validation_overlap < 0:
            raise ValueError("max_train_validation_overlap must be >= 0")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "S0GatePolicy":
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark/dataset reserved for evaluation or other non-training use."""

    benchmark_id: str
    version: str
    source_id: str
    held_out: bool = True
    allowed_uses: tuple[str, ...] = ("evaluation",)
    license_id: str | None = None
    source_url: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id is required")
        if not self.version.strip():
            raise ValueError("version is required")
        if not self.source_id.strip():
            raise ValueError("source_id is required")
        normalized = {item.strip().lower() for item in self.allowed_uses}
        if not normalized:
            raise ValueError("allowed_uses cannot be empty")
        if self.held_out and normalized & _FORBIDDEN_HELDOUT_USES:
            bad = sorted(normalized & _FORBIDDEN_HELDOUT_USES)
            raise ValueError(f"held-out benchmark cannot allow training uses: {bad}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["allowed_uses"] = list(self.allowed_uses)
        return result


class BenchmarkRegistry:
    """Contamination-aware registry with stable machine-readable manifests."""

    def __init__(self, specs: Iterable[BenchmarkSpec] = ()) -> None:
        self._specs: dict[str, BenchmarkSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: BenchmarkSpec) -> None:
        key = f"{spec.benchmark_id}@{spec.version}"
        if key in self._specs:
            raise ValueError(f"duplicate benchmark key: {key}")
        self._specs[key] = spec

    def training_collisions(self, training_source_ids: Iterable[str]) -> list[dict[str, str]]:
        source_ids = {item for item in training_source_ids}
        collisions: list[dict[str, str]] = []
        for key, spec in sorted(self._specs.items()):
            if spec.held_out and spec.source_id in source_ids:
                collisions.append(
                    {
                        "benchmark_key": key,
                        "source_id": spec.source_id,
                    }
                )
        return collisions

    def manifest(self) -> dict[str, Any]:
        entries = [self._specs[key].to_dict() for key in sorted(self._specs)]
        payload = {
            "schema_version": "12-6.benchmark-registry.v1",
            "benchmarks": entries,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return payload


def perplexity_from_nll(loss: float) -> float:
    """Convert mean natural-log token NLL to perplexity.

    Perplexity is only meaningful here for a finite, non-negative mean token NLL.
    """

    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError("loss must be a real number")
    value = float(loss)
    if not math.isfinite(value):
        raise ValueError("loss must be finite")
    if value < 0:
        raise ValueError("mean token NLL cannot be negative")
    try:
        result = math.exp(value)
    except OverflowError:
        result = math.inf
    return result


def relative_loss_improvement(before: float, after: float) -> float:
    """Return fractional loss reduction, e.g. 0.25 for a 25% decrease."""

    before_value = _finite_number(before, "before")
    after_value = _finite_number(after, "after")
    if before_value <= 0:
        raise ValueError("before must be > 0")
    return (before_value - after_value) / before_value


def stable_text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return sha256(text.encode("utf-8")).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _get(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _missing(paths: Sequence[str], evidence: Mapping[str, Any]) -> list[str]:
    return [path for path in paths if _get(evidence, path) is _MISSING]


def _gate_missing(gate_id: str, title: str, missing_paths: Sequence[str]) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        title=title,
        status=GateStatus.NOT_TESTED,
        reason=f"missing evidence: {', '.join(missing_paths)}",
        evidence={"missing": list(missing_paths)},
    )


def _exact_bool_gate(
    evidence: Mapping[str, Any],
    *,
    gate_id: str,
    title: str,
    path: str,
    expected: bool = True,
) -> GateResult:
    value = _get(evidence, path)
    if value is _MISSING:
        return _gate_missing(gate_id, title, [path])
    if type(value) is not bool:
        return GateResult(
            gate_id,
            title,
            GateStatus.FAIL,
            f"{path} must be exact boolean",
            evidence={path: value},
        )
    status = GateStatus.PASS if value is expected else GateStatus.FAIL
    relation = "matches" if status is GateStatus.PASS else "does not match"
    return GateResult(
        gate_id,
        title,
        status,
        f"{path} {relation} required value {expected}",
        evidence={path: value},
    )


def _identity_gate(evidence: Mapping[str, Any]) -> GateResult:
    required = ["candidate.sha", "candidate.id", "eval_config.id", "dataset.identity"]
    missing = _missing(required, evidence)
    if missing:
        return _gate_missing("s0.identity", "Candidate/eval/dataset identity", missing)
    values = {path: _get(evidence, path) for path in required}
    invalid = [
        path
        for path, value in values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        return GateResult(
            "s0.identity",
            "Candidate/eval/dataset identity",
            GateStatus.FAIL,
            f"identity fields must be non-empty strings: {', '.join(invalid)}",
            evidence=values,
        )
    return GateResult(
        "s0.identity",
        "Candidate/eval/dataset identity",
        GateStatus.PASS,
        "candidate, eval config, and dataset identities are present",
        evidence=values,
    )


def _parameter_gate(evidence: Mapping[str, Any], policy: S0GatePolicy) -> GateResult:
    path = "candidate.parameter_count"
    value = _get(evidence, path)
    if value is _MISSING:
        return _gate_missing("s0.parameter_range", "Expected parameter range", [path])
    if isinstance(value, bool) or not isinstance(value, int):
        return GateResult(
            "s0.parameter_range",
            "Expected parameter range",
            GateStatus.FAIL,
            "parameter_count must be an integer",
            evidence={path: value},
        )
    passed = policy.min_parameters <= value <= policy.max_parameters
    return GateResult(
        "s0.parameter_range",
        "Expected parameter range",
        GateStatus.PASS if passed else GateStatus.FAIL,
        (
            f"parameter_count={value} is inside [{policy.min_parameters}, {policy.max_parameters}]"
            if passed
            else (
                f"parameter_count={value} is outside "
                f"[{policy.min_parameters}, {policy.max_parameters}]"
            )
        ),
        evidence={
            path: value,
            "target_parameters": policy.target_parameters,
            "min_parameters": policy.min_parameters,
            "max_parameters": policy.max_parameters,
        },
    )


def _loss_decrease_gate(evidence: Mapping[str, Any]) -> GateResult:
    paths = ["metrics.train_loss_before", "metrics.train_loss_after"]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing("s0.training_learns", "Training loss decreases", missing)
    try:
        before = _finite_number(_get(evidence, paths[0]), paths[0])
        after = _finite_number(_get(evidence, paths[1]), paths[1])
    except (TypeError, ValueError) as exc:
        return GateResult(
            "s0.training_learns",
            "Training loss decreases",
            GateStatus.FAIL,
            str(exc),
            evidence={path: _get(evidence, path) for path in paths},
        )
    if before < 0 or after < 0:
        return GateResult(
            "s0.training_learns",
            "Training loss decreases",
            GateStatus.FAIL,
            "loss values must be non-negative token NLL/cross-entropy",
            evidence={"before": before, "after": after},
        )
    passed = after < before
    return GateResult(
        "s0.training_learns",
        "Training loss decreases",
        GateStatus.PASS if passed else GateStatus.FAIL,
        (
            f"training loss decreased from {before:.8g} to {after:.8g}"
            if passed
            else f"training loss did not decrease: {before:.8g} -> {after:.8g}"
        ),
        evidence={
            "train_loss_before": before,
            "train_loss_after": after,
            "relative_improvement": (
                relative_loss_improvement(before, after) if before > 0 else None
            ),
        },
    )


def _validation_measurement_gate(evidence: Mapping[str, Any]) -> GateResult:
    paths = ["metrics.validation_loss_before", "metrics.validation_loss_after"]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing("s0.validation_measured", "Validation behavior measured", missing)
    try:
        before = _finite_number(_get(evidence, paths[0]), paths[0])
        after = _finite_number(_get(evidence, paths[1]), paths[1])
    except (TypeError, ValueError) as exc:
        return GateResult(
            "s0.validation_measured",
            "Validation behavior measured",
            GateStatus.FAIL,
            str(exc),
            evidence={path: _get(evidence, path) for path in paths},
        )
    if before < 0 or after < 0:
        return GateResult(
            "s0.validation_measured",
            "Validation behavior measured",
            GateStatus.FAIL,
            "validation losses must be non-negative token NLL/cross-entropy",
            evidence={"validation_loss_before": before, "validation_loss_after": after},
        )
    return GateResult(
        "s0.validation_measured",
        "Validation behavior measured",
        GateStatus.PASS,
        "finite held-out validation losses are recorded before and after training",
        evidence={
            "validation_loss_before": before,
            "validation_loss_after": after,
            "validation_loss_delta": after - before,
        },
    )


def _baseline_gate(evidence: Mapping[str, Any], policy: S0GatePolicy) -> GateResult:
    paths = ["metrics.random_validation_loss", "metrics.trained_validation_loss"]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing(
            "s0.random_vs_trained",
            "Random baseline differs from trained",
            missing,
        )
    try:
        random_loss = _finite_number(_get(evidence, paths[0]), paths[0])
        trained_loss = _finite_number(_get(evidence, paths[1]), paths[1])
    except (TypeError, ValueError) as exc:
        return GateResult(
            "s0.random_vs_trained",
            "Random baseline differs from trained",
            GateStatus.FAIL,
            str(exc),
            evidence={path: _get(evidence, path) for path in paths},
        )
    if random_loss < 0 or trained_loss < 0:
        return GateResult(
            "s0.random_vs_trained",
            "Random baseline differs from trained",
            GateStatus.FAIL,
            "baseline losses must be non-negative",
            evidence={
                "random_validation_loss": random_loss,
                "trained_validation_loss": trained_loss,
            },
        )
    if policy.require_trained_validation_better_than_random:
        passed = trained_loss < random_loss
        reason = (
            "trained held-out loss is lower than random-init held-out loss"
            if passed
            else "trained held-out loss is not lower than random-init held-out loss"
        )
    else:
        passed = trained_loss != random_loss
        reason = (
            "trained held-out loss differs from random-init held-out loss"
            if passed
            else "trained and random-init held-out losses are identical"
        )
    return GateResult(
        "s0.random_vs_trained",
        "Random baseline differs from trained",
        GateStatus.PASS if passed else GateStatus.FAIL,
        reason,
        evidence={
            "random_validation_loss": random_loss,
            "trained_validation_loss": trained_loss,
            "delta": trained_loss - random_loss,
        },
    )


def _generation_gate(evidence: Mapping[str, Any]) -> GateResult:
    probes = _get(evidence, "generation_probes")
    if probes is _MISSING:
        return _gate_missing("s0.generation", "Generation probe", ["generation_probes"])
    if not isinstance(probes, Sequence) or isinstance(probes, (str, bytes)):
        return GateResult(
            "s0.generation",
            "Generation probe",
            GateStatus.FAIL,
            "generation_probes must be an array",
            evidence={"generation_probes": probes},
        )
    if not probes:
        return GateResult(
            "s0.generation",
            "Generation probe",
            GateStatus.FAIL,
            "at least one generation probe is required",
            evidence={"probe_count": 0},
        )
    normalized: list[dict[str, Any]] = []
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping):
            return GateResult(
                "s0.generation",
                "Generation probe",
                GateStatus.FAIL,
                f"generation probe {index} must be an object",
                evidence={"index": index},
            )
        token_count = probe.get("token_count", _MISSING)
        output_sha = probe.get("output_sha256", _MISSING)
        probe_id = probe.get("id", _MISSING)
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or not isinstance(output_sha, str)
            or len(output_sha) != 64
            or not isinstance(probe_id, str)
            or not probe_id.strip()
        ):
            return GateResult(
                "s0.generation",
                "Generation probe",
                GateStatus.FAIL,
                f"generation probe {index} lacks valid id/token_count/output_sha256",
                evidence={"index": index, "probe": dict(probe)},
            )
        normalized.append(
            {
                "id": probe_id,
                "token_count": token_count,
                "output_sha256": output_sha,
                "seed": probe.get("seed"),
                "sampler": probe.get("sampler"),
            }
        )
    return GateResult(
        "s0.generation",
        "Generation probe",
        GateStatus.PASS,
        f"{len(normalized)} generation probe(s) produced non-empty output",
        evidence={"probes": normalized},
    )


def _split_integrity_gate(evidence: Mapping[str, Any], policy: S0GatePolicy) -> GateResult:
    paths = [
        "dataset.heldout_used_for_training",
        "dataset.train_validation_overlap",
        "dataset.validation_examples",
    ]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing("s0.heldout_integrity", "Held-out split integrity", missing)
    used = _get(evidence, paths[0])
    overlap = _get(evidence, paths[1])
    validation_examples = _get(evidence, paths[2])
    if type(used) is not bool:
        return GateResult(
            "s0.heldout_integrity",
            "Held-out split integrity",
            GateStatus.FAIL,
            "heldout_used_for_training must be exact boolean",
            evidence={"heldout_used_for_training": used},
        )
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        return GateResult(
            "s0.heldout_integrity",
            "Held-out split integrity",
            GateStatus.FAIL,
            "train_validation_overlap must be a non-negative integer",
            evidence={"train_validation_overlap": overlap},
        )
    if (
        isinstance(validation_examples, bool)
        or not isinstance(validation_examples, int)
        or validation_examples <= 0
    ):
        return GateResult(
            "s0.heldout_integrity",
            "Held-out split integrity",
            GateStatus.FAIL,
            "validation_examples must be a positive integer",
            evidence={"validation_examples": validation_examples},
        )
    passed = (not used) and overlap <= policy.max_train_validation_overlap
    return GateResult(
        "s0.heldout_integrity",
        "Held-out split integrity",
        GateStatus.PASS if passed else GateStatus.FAIL,
        (
            "held-out data is not used for training and split overlap is within policy"
            if passed
            else "held-out data/training overlap violates policy"
        ),
        evidence={
            "heldout_used_for_training": used,
            "train_validation_overlap": overlap,
            "max_train_validation_overlap": policy.max_train_validation_overlap,
            "validation_examples": validation_examples,
        },
    )


def _anti_fixed_batch_gate(evidence: Mapping[str, Any], policy: S0GatePolicy) -> GateResult:
    path = "dataset.distinct_train_batches"
    value = _get(evidence, path)
    if value is _MISSING:
        return _gate_missing(
            "s0.not_single_fixed_batch",
            "Not a single fixed-batch memorization demo",
            [path],
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return GateResult(
            "s0.not_single_fixed_batch",
            "Not a single fixed-batch memorization demo",
            GateStatus.FAIL,
            "distinct_train_batches must be a non-negative integer",
            evidence={path: value},
        )
    passed = value >= policy.min_distinct_train_batches
    return GateResult(
        "s0.not_single_fixed_batch",
        "Not a single fixed-batch memorization demo",
        GateStatus.PASS if passed else GateStatus.FAIL,
        (
            f"training used {value} distinct batches"
            if passed
            else (
                f"training used {value} distinct batches; "
                f"requires >= {policy.min_distinct_train_batches}"
            )
        ),
        evidence={
            "distinct_train_batches": value,
            "minimum_required": policy.min_distinct_train_batches,
        },
    )


def _contamination_gate(evidence: Mapping[str, Any]) -> GateResult:
    paths = [
        "contamination.checked",
        "contamination.benchmark_overlap_count",
        "contamination.heldout_overlap_count",
    ]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing("s0.contamination", "Benchmark/held-out contamination check", missing)
    checked = _get(evidence, paths[0])
    benchmark_overlap = _get(evidence, paths[1])
    heldout_overlap = _get(evidence, paths[2])
    if type(checked) is not bool:
        return GateResult(
            "s0.contamination",
            "Benchmark/held-out contamination check",
            GateStatus.FAIL,
            "contamination.checked must be exact boolean",
            evidence={"checked": checked},
        )
    for name, value in (
        ("benchmark_overlap_count", benchmark_overlap),
        ("heldout_overlap_count", heldout_overlap),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return GateResult(
                "s0.contamination",
                "Benchmark/held-out contamination check",
                GateStatus.FAIL,
                f"{name} must be a non-negative integer",
                evidence={name: value},
            )
    passed = checked and benchmark_overlap == 0 and heldout_overlap == 0
    return GateResult(
        "s0.contamination",
        "Benchmark/held-out contamination check",
        GateStatus.PASS if passed else GateStatus.FAIL,
        (
            "contamination check ran and found zero benchmark/held-out overlap"
            if passed
            else "contamination check is incomplete or found overlap"
        ),
        evidence={
            "checked": checked,
            "benchmark_overlap_count": benchmark_overlap,
            "heldout_overlap_count": heldout_overlap,
        },
    )


def _regression_gate(evidence: Mapping[str, Any]) -> GateResult:
    paths = ["regressions.executed", "regressions.failures"]
    missing = _missing(paths, evidence)
    if missing:
        return _gate_missing("s0.regressions", "Regression suite", missing)
    executed = _get(evidence, paths[0])
    failures = _get(evidence, paths[1])
    if type(executed) is not bool:
        return GateResult(
            "s0.regressions",
            "Regression suite",
            GateStatus.FAIL,
            "regressions.executed must be exact boolean",
            evidence={"executed": executed},
        )
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        return GateResult(
            "s0.regressions",
            "Regression suite",
            GateStatus.FAIL,
            "regressions.failures must be a non-negative integer",
            evidence={"failures": failures},
        )
    passed = executed and failures == 0
    return GateResult(
        "s0.regressions",
        "Regression suite",
        GateStatus.PASS if passed else GateStatus.FAIL,
        "regression suite executed with zero failures" if passed else "regression suite not clean",
        evidence={"executed": executed, "failures": failures},
    )


def _derived_metrics(evidence: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in (
        "train_loss_before",
        "train_loss_after",
        "validation_loss_before",
        "validation_loss_after",
        "random_validation_loss",
        "trained_validation_loss",
    ):
        value = _get(evidence, f"metrics.{key}")
        if value is _MISSING:
            continue
        try:
            number = _finite_number(value, key)
            if number >= 0:
                result[f"{key}_perplexity"] = perplexity_from_nll(number)
        except (TypeError, ValueError):
            continue
    return result


def evaluate_s0(
    evidence: Mapping[str, Any],
    policy: S0GatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate S0 evidence without silently converting missing evidence into PASS."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    policy = policy or S0GatePolicy()
    gates = [
        _identity_gate(evidence),
        _exact_bool_gate(
            evidence,
            gate_id="s0.random_init",
            title="Random initialization lineage",
            path="candidate.random_init",
        ),
        _exact_bool_gate(
            evidence,
            gate_id="s0.model_constructs",
            title="Model construction",
            path="candidate.model_constructed",
        ),
        _parameter_gate(evidence, policy),
        _loss_decrease_gate(evidence),
        _validation_measurement_gate(evidence),
        _baseline_gate(evidence, policy),
        _generation_gate(evidence),
        _exact_bool_gate(
            evidence,
            gate_id="s0.save_load",
            title="Checkpoint save/load",
            path="checkpoint.save_load_verified",
        ),
        _exact_bool_gate(
            evidence,
            gate_id="s0.resume",
            title="Interrupted-run resume",
            path="checkpoint.resume_verified",
        ),
        _split_integrity_gate(evidence, policy),
        _anti_fixed_batch_gate(evidence, policy),
        _contamination_gate(evidence),
        _regression_gate(evidence),
    ]

    counts = {status.value: 0 for status in GateStatus}
    for gate in gates:
        counts[gate.status.value] += 1
    required = [gate for gate in gates if gate.required]
    promotion_eligible = all(gate.status is GateStatus.PASS for gate in required)
    if promotion_eligible:
        overall_status = GateStatus.PASS
    elif any(gate.status is GateStatus.FAIL for gate in required):
        overall_status = GateStatus.FAIL
    else:
        overall_status = GateStatus.NOT_TESTED

    return {
        "schema_version": "12-6.stage-gate-result.v1",
        "stage": "S0",
        "policy": policy.to_dict(),
        "candidate": dict(evidence.get("candidate", {}))
        if isinstance(evidence.get("candidate", {}), Mapping)
        else evidence.get("candidate"),
        "eval_config": dict(evidence.get("eval_config", {}))
        if isinstance(evidence.get("eval_config", {}), Mapping)
        else evidence.get("eval_config"),
        "dataset": dict(evidence.get("dataset", {}))
        if isinstance(evidence.get("dataset", {}), Mapping)
        else evidence.get("dataset"),
        "metrics": dict(evidence.get("metrics", {}))
        if isinstance(evidence.get("metrics", {}), Mapping)
        else evidence.get("metrics"),
        "derived_metrics": _derived_metrics(evidence),
        "generation_probes": list(evidence.get("generation_probes", []))
        if isinstance(evidence.get("generation_probes", []), Sequence)
        and not isinstance(evidence.get("generation_probes", []), (str, bytes))
        else evidence.get("generation_probes"),
        "gates": [gate.to_dict() for gate in gates],
        "summary": {
            "overall_status": overall_status.value,
            "promotion_eligible": promotion_eligible,
            "counts": counts,
            "required_gate_count": len(required),
        },
    }


def dump_stage_gate_result(result: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 12-6 AI stage-gate evidence")
    parser.add_argument("evidence", type=Path, help="machine-readable evidence JSON")
    parser.add_argument("--policy", type=Path, help="optional S0 policy JSON")
    parser.add_argument("--output", type=Path, required=True, help="output result JSON")
    parser.add_argument(
        "--fail-on-ineligible",
        action="store_true",
        help="exit non-zero unless every required gate passes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evidence = load_json_object(args.evidence)
    policy = (
        S0GatePolicy.from_mapping(load_json_object(args.policy))
        if args.policy is not None
        else S0GatePolicy()
    )
    result = evaluate_s0(evidence, policy)
    dump_stage_gate_result(result, args.output)
    if args.fail_on_ineligible and not result["summary"]["promotion_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
