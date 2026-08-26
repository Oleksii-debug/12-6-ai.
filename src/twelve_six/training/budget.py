"""Fail-closed data/exposure budgeting for scratch Base pretraining.

The budget unit is post-pack non-ignored causal loss positions. Raw bytes, source
bytes, documents, and pre-dedup token estimates are intentionally not accepted as
training capacity because they can overstate the amount of unique supervised
signal that will actually reach the loss.

This module does not authorize compute and does not encode a claim that one
specific tokens-per-parameter ratio is universally optimal. A contract may carry
one or more preregistered planning ratios that downstream experiments can compare.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEX = frozenset("0123456789abcdef")


class TrainingBudgetError(ValueError):
    """Raised when a training-budget or corpus-capacity contract is invalid."""


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """One preregistered exposure point expressed in causal loss positions."""

    name: str
    tokens_per_parameter: float
    loss_positions: int


@dataclass(frozen=True, slots=True)
class TrainingBudgetContract:
    """Validated model-bound exposure plan."""

    schema: str
    candidate: str
    parameter_count: int
    model_identity_sha256: str
    unit: str
    budget_points: tuple[BudgetPoint, ...]

    def point(self, name: str) -> BudgetPoint:
        for point in self.budget_points:
            if point.name == name:
                return point
        raise TrainingBudgetError(f"unknown budget point {name!r}")


@dataclass(frozen=True, slots=True)
class CorpusCapacity:
    """Terminal post-pack corpus capacity accepted for training planning."""

    postpack_loss_positions: int
    dataset_manifest_sha256: str
    rights_status: str
    dedup_status: str
    contamination_status: str
    split_status: str
    replay_policy: str


@dataclass(frozen=True, slots=True)
class BudgetReadiness:
    """Capacity comparison only; never a compute/training authorization."""

    budget_name: str
    required_loss_positions: int
    available_loss_positions: int
    capacity_satisfied: bool
    training_authorized: bool = False


def loss_positions_for_ratio(parameter_count: int, tokens_per_parameter: float) -> int:
    """Return a deterministic ceil(parameter_count * planning ratio)."""

    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
        raise TrainingBudgetError("parameter_count must be an integer")
    if parameter_count <= 0:
        raise TrainingBudgetError("parameter_count must be positive")
    if not isinstance(tokens_per_parameter, (int, float)) or isinstance(
        tokens_per_parameter, bool
    ):
        raise TrainingBudgetError("tokens_per_parameter must be numeric")
    ratio = float(tokens_per_parameter)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise TrainingBudgetError("tokens_per_parameter must be finite and positive")
    return math.ceil(parameter_count * ratio)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingBudgetError(f"{field} must be a mapping")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingBudgetError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_nonempty_string(value, field)
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise TrainingBudgetError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def validate_training_budget_contract(payload: Mapping[str, Any]) -> TrainingBudgetContract:
    """Validate a model-bound training-budget contract and its exact arithmetic."""

    schema = _require_nonempty_string(payload.get("schema"), "schema")
    if schema != "12-6.training-budget.v1":
        raise TrainingBudgetError("unsupported training-budget schema")

    candidate_payload = _require_mapping(payload.get("candidate"), "candidate")
    candidate = _require_nonempty_string(candidate_payload.get("name"), "candidate.name")
    parameter_count = candidate_payload.get("parameter_count")
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
        raise TrainingBudgetError("candidate.parameter_count must be an integer")
    if parameter_count <= 0:
        raise TrainingBudgetError("candidate.parameter_count must be positive")
    model_identity = _require_sha256(
        candidate_payload.get("model_identity_sha256"),
        "candidate.model_identity_sha256",
    )

    unit = _require_nonempty_string(payload.get("unit"), "unit")
    if unit != "postpack_nonignored_causal_loss_positions":
        raise TrainingBudgetError("training budget must use post-pack causal loss positions")

    raw_points = payload.get("budget_points")
    if not isinstance(raw_points, list) or not raw_points:
        raise TrainingBudgetError("budget_points must be a non-empty list")

    names: set[str] = set()
    points: list[BudgetPoint] = []
    previous_ratio = 0.0
    for index, raw_point in enumerate(raw_points):
        point = _require_mapping(raw_point, f"budget_points[{index}]")
        name = _require_nonempty_string(point.get("name"), f"budget_points[{index}].name")
        if name in names:
            raise TrainingBudgetError(f"duplicate budget point name {name!r}")
        names.add(name)

        ratio_raw = point.get("tokens_per_parameter")
        if not isinstance(ratio_raw, (int, float)) or isinstance(ratio_raw, bool):
            raise TrainingBudgetError(
                f"budget_points[{index}].tokens_per_parameter must be numeric"
            )
        ratio = float(ratio_raw)
        expected = loss_positions_for_ratio(parameter_count, ratio)
        declared = point.get("loss_positions")
        if not isinstance(declared, int) or isinstance(declared, bool):
            raise TrainingBudgetError(
                f"budget_points[{index}].loss_positions must be an integer"
            )
        if declared != expected:
            raise TrainingBudgetError(
                f"budget point {name!r} arithmetic mismatch: expected {expected}, got {declared}"
            )
        if ratio <= previous_ratio:
            raise TrainingBudgetError("budget_points ratios must be strictly increasing")
        previous_ratio = ratio
        points.append(BudgetPoint(name, ratio, declared))

    return TrainingBudgetContract(
        schema=schema,
        candidate=candidate,
        parameter_count=parameter_count,
        model_identity_sha256=model_identity,
        unit=unit,
        budget_points=tuple(points),
    )


def load_training_budget_contract(path: str | Path) -> TrainingBudgetContract:
    """Load and validate one UTF-8 JSON budget contract."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TrainingBudgetError("training-budget JSON root must be a mapping")
    return validate_training_budget_contract(payload)


def validate_corpus_capacity(payload: Mapping[str, Any]) -> CorpusCapacity:
    """Accept only terminal, post-pack, no-replay capacity evidence."""

    positions = payload.get("postpack_loss_positions")
    if not isinstance(positions, int) or isinstance(positions, bool) or positions < 0:
        raise TrainingBudgetError("postpack_loss_positions must be a non-negative integer")

    manifest = _require_sha256(
        payload.get("dataset_manifest_sha256"),
        "dataset_manifest_sha256",
    )
    statuses: dict[str, str] = {}
    for field in ("rights_status", "dedup_status", "contamination_status", "split_status"):
        value = _require_nonempty_string(payload.get(field), field)
        if value != "PASS":
            raise TrainingBudgetError(f"{field} must be terminal PASS")
        statuses[field] = value

    replay_policy = _require_nonempty_string(payload.get("replay_policy"), "replay_policy")
    if replay_policy != "NO_REPLAY":
        raise TrainingBudgetError("replay_policy must be NO_REPLAY")

    return CorpusCapacity(
        postpack_loss_positions=positions,
        dataset_manifest_sha256=manifest,
        rights_status=statuses["rights_status"],
        dedup_status=statuses["dedup_status"],
        contamination_status=statuses["contamination_status"],
        split_status=statuses["split_status"],
        replay_policy=replay_policy,
    )


def evaluate_budget_readiness(
    contract: TrainingBudgetContract,
    capacity: CorpusCapacity,
    *,
    budget_name: str,
) -> BudgetReadiness:
    """Compare terminal capacity with one preregistered point.

    The returned ``training_authorized`` flag is intentionally always false.
    User/project compute authorization is a separate gate and must never be inferred
    from data capacity.
    """

    point = contract.point(budget_name)
    return BudgetReadiness(
        budget_name=point.name,
        required_loss_positions=point.loss_positions,
        available_loss_positions=capacity.postpack_loss_positions,
        capacity_satisfied=capacity.postpack_loss_positions >= point.loss_positions,
    )
