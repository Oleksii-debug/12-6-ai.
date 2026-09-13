"""Backend-neutral logical rank layout for DP/TP/PP/CP and EP subgroups."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import ParallelPlan

Axis = Literal["dp", "pp", "cp", "tp"]
_LAYOUT_SCHEMA = "12-6.logical-rank-layout.v1"
_AXIS_ORDER: tuple[Axis, ...] = ("dp", "pp", "cp", "tp")


@dataclass(frozen=True, slots=True)
class RankCoordinate:
    dp: int
    pp: int
    cp: int
    tp: int


@dataclass(frozen=True, slots=True)
class ExpertCoordinate:
    expert_parallel_rank: int
    expert_data_parallel_rank: int


@dataclass(frozen=True, slots=True)
class RankLayout:
    """Stable project-local rank identity, independent of any backend's rank ordering."""

    plan: ParallelPlan

    def __post_init__(self) -> None:
        self.plan.validate()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.plan.data_parallel,
            self.plan.pipeline_parallel,
            self.plan.context_parallel,
            self.plan.tensor_parallel,
        )

    @property
    def axis_order(self) -> tuple[Axis, ...]:
        return _AXIS_ORDER

    @property
    def identity_payload(self) -> dict[str, object]:
        return {
            "schema": _LAYOUT_SCHEMA,
            "axis_order": list(_AXIS_ORDER),
            "parallel_plan": asdict(self.plan),
            "shape": list(self.shape),
            "world_size": self.plan.world_size,
            "expert_semantics": "ep-is-subgroup-of-dp",
        }

    @property
    def identity_sha256(self) -> str:
        raw = json.dumps(
            self.identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def coordinate(self, rank: int) -> RankCoordinate:
        self._validate_rank(rank)
        quotient, tp = divmod(rank, self.plan.tensor_parallel)
        quotient, cp = divmod(quotient, self.plan.context_parallel)
        dp, pp = divmod(quotient, self.plan.pipeline_parallel)
        return RankCoordinate(dp=dp, pp=pp, cp=cp, tp=tp)

    def rank(self, coordinate: RankCoordinate) -> int:
        self._validate_coordinate(coordinate)
        return (
            (
                (coordinate.dp * self.plan.pipeline_parallel + coordinate.pp)
                * self.plan.context_parallel
                + coordinate.cp
            )
            * self.plan.tensor_parallel
            + coordinate.tp
        )

    def axis_group(self, rank: int, axis: Axis) -> tuple[int, ...]:
        coordinate = self.coordinate(rank)
        if axis not in _AXIS_ORDER:
            raise ValueError(f"unsupported axis: {axis!r}")
        size = {
            "dp": self.plan.data_parallel,
            "pp": self.plan.pipeline_parallel,
            "cp": self.plan.context_parallel,
            "tp": self.plan.tensor_parallel,
        }[axis]
        members = []
        for value in range(size):
            values = asdict(coordinate)
            values[axis] = value
            members.append(self.rank(RankCoordinate(**values)))
        return tuple(members)

    def expert_coordinate(self, rank: int) -> ExpertCoordinate:
        coordinate = self.coordinate(rank)
        ep = self.plan.expert_parallel
        return ExpertCoordinate(
            expert_parallel_rank=coordinate.dp % ep,
            expert_data_parallel_rank=coordinate.dp // ep,
        )

    def expert_parallel_group(self, rank: int) -> tuple[int, ...]:
        """Ranks that split experts while holding the expert-DP coordinate fixed."""

        coordinate = self.coordinate(rank)
        expert = self.expert_coordinate(rank)
        base_dp = expert.expert_data_parallel_rank * self.plan.expert_parallel
        return tuple(
            self.rank(
                RankCoordinate(
                    dp=base_dp + ep_rank,
                    pp=coordinate.pp,
                    cp=coordinate.cp,
                    tp=coordinate.tp,
                )
            )
            for ep_rank in range(self.plan.expert_parallel)
        )

    def expert_data_parallel_group(self, rank: int) -> tuple[int, ...]:
        """Ranks replicating the same expert slice across the DP domain."""

        coordinate = self.coordinate(rank)
        expert = self.expert_coordinate(rank)
        return tuple(
            self.rank(
                RankCoordinate(
                    dp=edp_rank * self.plan.expert_parallel + expert.expert_parallel_rank,
                    pp=coordinate.pp,
                    cp=coordinate.cp,
                    tp=coordinate.tp,
                )
            )
            for edp_rank in range(self.plan.expert_data_parallel)
        )

    def dense_gradient_sync_group(self, rank: int) -> tuple[int, ...]:
        """Shared-weight gradient domain across DP and CP replicas.

        TP and PP coordinates are held fixed because those dimensions own different model shards.
        CP duplicates shared weights while partitioning activations, so CP joins DP sync.
        """

        coordinate = self.coordinate(rank)
        return tuple(
            self.rank(RankCoordinate(dp=dp, pp=coordinate.pp, cp=cp, tp=coordinate.tp))
            for dp in range(self.plan.data_parallel)
            for cp in range(self.plan.context_parallel)
        )

    def _validate_rank(self, rank: int) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("rank must be an integer")
        if rank < 0 or rank >= self.plan.world_size:
            raise ValueError(f"rank must be in [0, {self.plan.world_size})")

    def _validate_coordinate(self, coordinate: RankCoordinate) -> None:
        if not isinstance(coordinate, RankCoordinate):
            raise TypeError("coordinate must be RankCoordinate")
        limits = {
            "dp": self.plan.data_parallel,
            "pp": self.plan.pipeline_parallel,
            "cp": self.plan.context_parallel,
            "tp": self.plan.tensor_parallel,
        }
        for axis, limit in limits.items():
            value = getattr(coordinate, axis)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
                raise ValueError(f"{axis} coordinate must be in [0, {limit})")
