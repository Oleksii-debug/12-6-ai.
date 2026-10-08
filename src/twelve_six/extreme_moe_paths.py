"""Plan 7 S9: bounded sparse-MoE engineering contracts, never a training launcher.

S8's sparse alternative explicitly requires this versioned expert authority.
All billion/trillion figures are arithmetic; tokens and shards are tiny fixtures.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import (
    _digest, _finite, _positive, _sha, inspect_shards,
)
from twelve_six.very_large_scale_paths import very_large_recipe


class MoEDenied(ValueError):
    """Malformed or unqualified sparse scale input."""


# shared (nonexpert), experts, parameters/expert, top-k. No weights allocated.
_PROFILES = {"300B": (20_000_000_000, 16, 18_000_000_000, 2),
             "1T": (40_000_000_000, 32, 30_000_000_000, 2)}


def moe_recipe(tier: str) -> dict[str, Any]:
    if type(tier) is not str or tier not in _PROFILES:
        raise MoEDenied("unsupported extreme scale tier")
    shared, experts, per_expert, top_k = _PROFILES[tier]
    total = shared + experts * per_expert
    active = shared + top_k * per_expert
    target = 300_000_000_000 if tier == "300B" else 1_000_000_000_000
    assert abs(total - target) <= target // 10
    record = {
        "schema_version": 1, "tier": tier,
        "routing": "TOP_K_DETERMINISTIC_FAIL_CLOSED_NO_TOKEN_DROP_V1",
        "expert_parallelism": "RANKED_REPLACEABLE_ADAPTER_V1",
        "shared_parameters": shared, "num_experts": experts,
        "parameters_per_expert": per_expert, "top_k": top_k,
        "total_parameters": total, "active_parameters_per_token": active,
        "dense_baseline_parameters": target,
        "active_fraction": active / total,
        "training_state_floor_bytes": 24 * total,
        "checkpoint_dual_copy_floor_bytes": 4 * total,
        "activation_parameters_are_not_peak_memory": True,
        "s8_sparse_contract": "PLAN7_S9_VERSIONED_MOE_CONTRACT_REQUIRED",
        "s8_dense_100b_reference_sha256": very_large_recipe("100B")["recipe_sha256"],
        "data_contract": "PLAN2_VERSIONED_ORDER_REQUIRED",
        "evaluation_contract": "PLAN4_EXPERT_UTILIZATION_AND_DENSE_BASELINE_REQUIRED",
        "serving_contract": "PLAN4_EXACT_EXPERT_PLACEMENT_AND_ROUTING_REQUIRED",
        "backend": "PROXY_ONLY_NO_DISTRIBUTED_EXECUTION",
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False, "checkpoint_promoted": False,
    }
    record["recipe_sha256"] = _digest(record)
    return record


@dataclass(frozen=True, slots=True)
class ExpertTopology:
    schema_version: int
    node_ids: tuple[str, ...]
    # For each expert, distinct primary and recovery node; exactly all experts.
    expert_replicas: tuple[tuple[str, str], ...]
    expert_parallel: int
    run_sha256: str
    data_order_sha256: str
    recipe_sha256: str
    resource_budget_bytes: int
    free_state_bytes: int
    free_checkpoint_bytes: int
    estimated_cost_usd: float
    max_cost_usd: float
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise MoEDenied("topology version")
        for field in ("run_sha256", "data_order_sha256", "recipe_sha256"):
            _sha(field, getattr(self, field))
        if (type(self.node_ids) is not tuple or not 2 <= len(self.node_ids) <= 128
                or len(set(self.node_ids)) != len(self.node_ids)
                or any(type(n) is not str or not n or len(n) > 64
                       for n in self.node_ids)):
            raise MoEDenied("invalid distinct node ids")
        if (type(self.expert_replicas) is not tuple or not self.expert_replicas
                or len(self.expert_replicas) > 256):
            raise MoEDenied("expert placement shape")
        for replica in self.expert_replicas:
            if (type(replica) is not tuple or len(replica) != 2
                    or replica[0] == replica[1]
                    or any(type(n) is not str or n not in self.node_ids for n in replica)):
                raise MoEDenied("expert failover placement invalid")
        _positive("expert parallel", self.expert_parallel)
        if (self.expert_parallel > len(self.node_ids)
                or len(self.expert_replicas) % self.expert_parallel):
            raise MoEDenied("expert parallel divisibility")
        for field in ("resource_budget_bytes", "free_state_bytes",
                      "free_checkpoint_bytes"):
            _positive(field, getattr(self, field))
        for field in ("estimated_cost_usd", "max_cost_usd"):
            _finite(field, getattr(self, field), allow_zero=True)
        if type(self.paid_compute_requested) is not bool:
            raise MoEDenied("paid authority type")


def route_fixture(scores: tuple[tuple[float, ...], ...], *,
                  num_experts: int, top_k: int,
                  capacity_per_expert: int) -> dict[str, Any]:
    """Deterministic top-k; over-capacity fails rather than drops or reroutes."""
    _positive("num experts", num_experts)
    _positive("top k", top_k)
    _positive("capacity", capacity_per_expert)
    if not 1 <= top_k <= num_experts <= 256:
        raise MoEDenied("router geometry")
    if type(scores) is not tuple or not 1 <= len(scores) <= 256:
        raise MoEDenied("bounded token batch required")
    loads = [0] * num_experts
    assignments = []
    for row in scores:
        if type(row) is not tuple or len(row) != num_experts:
            raise MoEDenied("invalid token expert logits")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise MoEDenied("nonfinite/untrusted expert logits")
        chosen = tuple(sorted(range(num_experts), key=lambda j: (-row[j], j))[:top_k])
        for expert in chosen:
            loads[expert] += 1
            if loads[expert] > capacity_per_expert:
                raise MoEDenied("expert capacity exceeded; no silent token drop")
        assignments.append(chosen)
    result = {
        "schema_version": 1, "assignments": assignments, "loads": loads,
        "tokens": len(scores), "routes": len(scores) * top_k,
        "max_load": max(loads), "min_load": min(loads),
        "load_imbalance": max(loads) - min(loads),
        "tokens_dropped": 0, "overflow_rerouted": False,
    }
    result["routing_sha256"] = _digest(result)
    return result


def assess_moe(tier: str, topology: ExpertTopology,
               routing: dict[str, Any],
               shards: tuple[tuple[str, bytes, str], ...],
               manifest_sha256: str, *,
               lost_node: str | None = None,
               resumed_run_sha256: str | None = None,
               resumed_data_order_sha256: str | None = None) -> dict[str, Any]:
    """Read-only resource, failover and checkpoint admission; never execute."""
    recipe = moe_recipe(tier)
    if not isinstance(topology, ExpertTopology):
        raise MoEDenied("typed topology required")
    ExpertTopology(**asdict(topology))  # Revalidate even if frozen object was forged.
    if topology.recipe_sha256 != recipe["recipe_sha256"]:
        raise MoEDenied("recipe mismatch")
    if len(topology.expert_replicas) != recipe["num_experts"]:
        raise MoEDenied("expert count mismatch")
    for name, candidate, current in (
        ("run", resumed_run_sha256, topology.run_sha256),
        ("data order", resumed_data_order_sha256, topology.data_order_sha256),
    ):
        if candidate is not None:
            _sha(name, candidate)
            if candidate != current:
                raise MoEDenied(name + " changed after restart")
    if lost_node is not None and lost_node not in topology.node_ids:
        raise MoEDenied("unknown lost node")
    if type(routing) is not dict or _digest({
        k: v for k, v in routing.items() if k != "routing_sha256"
    }) != routing.get("routing_sha256"):
        raise MoEDenied("routing receipt integrity")
    assignments = routing.get("assignments")
    loads = routing.get("loads")
    if (type(assignments) is not list or type(loads) is not list
            or len(loads) != recipe["num_experts"]
            or type(routing.get("tokens")) is not int
            or not 1 <= routing["tokens"] <= 256
            or len(assignments) != routing["tokens"]
            or any(type(x) is not int or x < 0 for x in loads)
            or routing.get("routes") != routing["tokens"] * recipe["top_k"]
            or routing.get("tokens_dropped") != 0
            or routing.get("overflow_rerouted") is not False):
        raise MoEDenied("routing accounting invalid")
    recalculated = [0] * len(loads)
    for assignment in assignments:
        if (not isinstance(assignment, (tuple, list))
                or len(assignment) != recipe["top_k"]
                or len(set(assignment)) != len(assignment)
                or any(type(x) is not int or not 0 <= x < len(loads) for x in assignment)):
            raise MoEDenied("routing assignment invalid")
        for expert in assignment:
            recalculated[expert] += 1
    if recalculated != loads or routing.get("max_load") != max(loads):
        raise MoEDenied("routing load forgery")
    shard_receipt = inspect_shards(shards, manifest_sha256)
    owners = [backup if primary == lost_node else primary
              for primary, backup in topology.expert_replicas]
    reasons = []
    if topology.paid_compute_requested:
        reasons.append("paid_compute_denied")
    if (topology.free_state_bytes < recipe["training_state_floor_bytes"]
            or topology.resource_budget_bytes < recipe["training_state_floor_bytes"]):
        reasons.append("state_capacity_denied")
    if topology.free_checkpoint_bytes < recipe["checkpoint_dual_copy_floor_bytes"]:
        reasons.append("checkpoint_capacity_denied")
    if topology.estimated_cost_usd > topology.max_cost_usd:
        reasons.append("cost_limit_denied")
    if lost_node is not None and lost_node in owners:
        reasons.append("failover_unavailable")
    if len(set(owners)) > topology.expert_parallel:
        reasons.append("expert_parallel_rank_capacity_denied")
    result = {
        "schema_version": 1, "tier": tier,
        "recipe_sha256": recipe["recipe_sha256"],
        "topology_sha256": _digest(asdict(topology)),
        "run_sha256": topology.run_sha256,
        "data_order_sha256": topology.data_order_sha256,
        "routing_sha256": routing["routing_sha256"],
        "checkpoint_fixture_sha256": _digest(shard_receipt),
        "active_parameters_per_token": recipe["active_parameters_per_token"],
        "total_parameters": recipe["total_parameters"],
        "effective_expert_nodes": owners,
        "lost_node": lost_node,
        "status": "NO_GO" if reasons else "PROXY_DESIGN_VALID_NOT_AUTHORIZED",
        "reasons": sorted(set(reasons)),
        "evidence": "LOCAL_FREE_SIMULATED_EXPERTS_AND_CHECKPOINT_SHARDS",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False, "checkpoint_promoted": False,
        "serving_authorized": False,
    }
    result["packet_sha256"] = _digest(result)
    return result
