"""Deterministic fake-rank topology for local scale-contract validation.

The project contract models EP as a subgroup of DP. EP therefore does not add a
physical mesh dimension here. Backend adapters that use a different topology
must translate explicitly instead of silently reusing this rank mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ParallelPlan


@dataclass(frozen=True, slots=True)
class RankCoordinate:
    global_rank: int
    data_parallel_rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    context_parallel_rank: int
    expert_parallel_rank: int
    expert_data_parallel_rank: int


@dataclass(frozen=True, slots=True)
class FakeProcessGroups:
    data_parallel: tuple[tuple[int, ...], ...]
    tensor_parallel: tuple[tuple[int, ...], ...]
    pipeline_parallel: tuple[tuple[int, ...], ...]
    context_parallel: tuple[tuple[int, ...], ...]
    expert_parallel: tuple[tuple[int, ...], ...]
    expert_data_parallel: tuple[tuple[int, ...], ...]


def rank_from_axes(
    plan: ParallelPlan,
    *,
    data_parallel_rank: int,
    tensor_parallel_rank: int,
    pipeline_parallel_rank: int,
    context_parallel_rank: int,
) -> int:
    """Linearize physical TP/CP/DP/PP axes with TP as the smallest stride."""

    plan.validate()
    ranks = (
        ("data_parallel_rank", data_parallel_rank, plan.data_parallel),
        ("tensor_parallel_rank", tensor_parallel_rank, plan.tensor_parallel),
        ("pipeline_parallel_rank", pipeline_parallel_rank, plan.pipeline_parallel),
        ("context_parallel_rank", context_parallel_rank, plan.context_parallel),
    )
    for name, rank, degree in ranks:
        if not 0 <= rank < degree:
            raise ValueError(f"{name} must be inside [0, {degree})")

    return tensor_parallel_rank + plan.tensor_parallel * (
        context_parallel_rank
        + plan.context_parallel
        * (data_parallel_rank + plan.data_parallel * pipeline_parallel_rank)
    )


def coordinate_for_rank(global_rank: int, plan: ParallelPlan) -> RankCoordinate:
    """Decode a physical rank and derive project EP/EDP subgroup coordinates."""

    plan.validate()
    if not 0 <= global_rank < plan.world_size:
        raise ValueError(f"global_rank must be inside [0, {plan.world_size})")

    remainder = global_rank
    tp_rank = remainder % plan.tensor_parallel
    remainder //= plan.tensor_parallel
    cp_rank = remainder % plan.context_parallel
    remainder //= plan.context_parallel
    dp_rank = remainder % plan.data_parallel
    pp_rank = remainder // plan.data_parallel

    ep_rank = dp_rank % plan.expert_parallel
    expert_dp_rank = dp_rank // plan.expert_parallel
    return RankCoordinate(
        global_rank=global_rank,
        data_parallel_rank=dp_rank,
        tensor_parallel_rank=tp_rank,
        pipeline_parallel_rank=pp_rank,
        context_parallel_rank=cp_rank,
        expert_parallel_rank=ep_rank,
        expert_data_parallel_rank=expert_dp_rank,
    )


def _group(
    plan: ParallelPlan,
    *,
    dp_values: range | tuple[int, ...],
    tp_values: range | tuple[int, ...],
    pp_values: range | tuple[int, ...],
    cp_values: range | tuple[int, ...],
) -> tuple[int, ...]:
    ranks = [
        rank_from_axes(
            plan,
            data_parallel_rank=dp,
            tensor_parallel_rank=tp,
            pipeline_parallel_rank=pp,
            context_parallel_rank=cp,
        )
        for pp in pp_values
        for dp in dp_values
        for cp in cp_values
        for tp in tp_values
    ]
    return tuple(sorted(ranks))


def fake_process_groups(plan: ParallelPlan) -> FakeProcessGroups:
    """Enumerate deterministic logical process groups without initializing torch.distributed."""

    plan.validate()
    dp_groups: list[tuple[int, ...]] = []
    tp_groups: list[tuple[int, ...]] = []
    pp_groups: list[tuple[int, ...]] = []
    cp_groups: list[tuple[int, ...]] = []
    ep_groups: list[tuple[int, ...]] = []
    edp_groups: list[tuple[int, ...]] = []

    for pp in range(plan.pipeline_parallel):
        for cp in range(plan.context_parallel):
            for tp in range(plan.tensor_parallel):
                dp_groups.append(
                    _group(
                        plan,
                        dp_values=range(plan.data_parallel),
                        tp_values=(tp,),
                        pp_values=(pp,),
                        cp_values=(cp,),
                    )
                )

    for pp in range(plan.pipeline_parallel):
        for dp in range(plan.data_parallel):
            for cp in range(plan.context_parallel):
                tp_groups.append(
                    _group(
                        plan,
                        dp_values=(dp,),
                        tp_values=range(plan.tensor_parallel),
                        pp_values=(pp,),
                        cp_values=(cp,),
                    )
                )

    for dp in range(plan.data_parallel):
        for cp in range(plan.context_parallel):
            for tp in range(plan.tensor_parallel):
                pp_groups.append(
                    _group(
                        plan,
                        dp_values=(dp,),
                        tp_values=(tp,),
                        pp_values=range(plan.pipeline_parallel),
                        cp_values=(cp,),
                    )
                )

    for pp in range(plan.pipeline_parallel):
        for dp in range(plan.data_parallel):
            for tp in range(plan.tensor_parallel):
                cp_groups.append(
                    _group(
                        plan,
                        dp_values=(dp,),
                        tp_values=(tp,),
                        pp_values=(pp,),
                        cp_values=range(plan.context_parallel),
                    )
                )

    for pp in range(plan.pipeline_parallel):
        for cp in range(plan.context_parallel):
            for tp in range(plan.tensor_parallel):
                for expert_dp_rank in range(plan.expert_data_parallel):
                    dp_values = tuple(
                        expert_dp_rank * plan.expert_parallel + ep_rank
                        for ep_rank in range(plan.expert_parallel)
                    )
                    ep_groups.append(
                        _group(
                            plan,
                            dp_values=dp_values,
                            tp_values=(tp,),
                            pp_values=(pp,),
                            cp_values=(cp,),
                        )
                    )
                for ep_rank in range(plan.expert_parallel):
                    dp_values = tuple(
                        expert_dp_rank * plan.expert_parallel + ep_rank
                        for expert_dp_rank in range(plan.expert_data_parallel)
                    )
                    edp_groups.append(
                        _group(
                            plan,
                            dp_values=dp_values,
                            tp_values=(tp,),
                            pp_values=(pp,),
                            cp_values=(cp,),
                        )
                    )

    return FakeProcessGroups(
        data_parallel=tuple(dp_groups),
        tensor_parallel=tuple(tp_groups),
        pipeline_parallel=tuple(pp_groups),
        context_parallel=tuple(cp_groups),
        expert_parallel=tuple(ep_groups),
        expert_data_parallel=tuple(edp_groups),
    )
