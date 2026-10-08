"""Plan 7 Section 9: 300B/1T dense and sparse MoE LOCAL_FREE contracts.

All paths are arithmetic/synthetic. No weights, remote nodes, paid training,
real backend execution, checkpoint publication or promotion is performed.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import _digest, _finite, _positive, _sha
from twelve_six.large_scale_paths import LargeAdapter, LargeCapacity, LargeLimits
from twelve_six.model import ModelSpec
from twelve_six.very_large_paths import NodePlacement, inspect_transport

# Width/layers/FFN/KV heads, head_dim=128. Arithmetic only.
PROFILES = {"300B": (12288, 144, 49152, 8),
            "1T": (20480, 144, 92160, 8)}


class ExtremeDenied(ValueError):
    """Invalid or insufficient evidence cannot promote an extreme-scale plan."""


@dataclass(frozen=True, slots=True)
class MoEPolicy:
    schema_version: int
    expert_count: int
    top_k: int
    capacity_factor: float
    expert_parallel_groups: int
    router_seed_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ExtremeDenied("policy schema version")
        for field in ("expert_count", "top_k", "expert_parallel_groups"):
            _positive(field, getattr(self, field))
        if (not 2 <= self.expert_count <= 128
                or not 1 <= self.top_k < self.expert_count
                or self.top_k > 4
                or self.expert_count % self.expert_parallel_groups != 0):
            raise ExtremeDenied("unsupported expert topology")
        _finite("capacity factor", self.capacity_factor)
        if self.capacity_factor > 8:
            raise ExtremeDenied("unbounded expert capacity")
        _sha("router seed", self.router_seed_sha256)


def extreme_spec(tier: str) -> ModelSpec:
    if type(tier) is not str or tier not in PROFILES:
        raise ExtremeDenied("unknown extreme tier")
    width, layers, ff, kv = PROFILES[tier]
    return ModelSpec(schema_version=1, vocab_size=8192, max_seq_len=1024,
                     d_model=width, n_layers=layers, n_heads=width // 128,
                     n_kv_heads=kv, head_dim=128, d_ff=ff, rope_rotary_dim=128)


def extreme_recipe(tier: str, policy: MoEPolicy | None = None) -> dict[str, Any]:
    spec = extreme_spec(tier)
    baseline = spec.parameter_breakdown()
    target = 300_000_000_000 if tier == "300B" else 1_000_000_000_000
    dense_total = baseline["total"]
    if abs(dense_total - target) > target // 10:
        raise ExtremeDenied("model parameter tolerance")
    if policy is not None:
        if not isinstance(policy, MoEPolicy):
            raise ExtremeDenied("typed MoE policy required")
        MoEPolicy(**asdict(policy))
    # Each expert uses half the dense FFN width; top-2 is comparable to
    # the dense MLP active-parameter count, but all experts need storage.
    shared = dense_total - baseline["mlp_per_layer"] * spec.n_layers
    expert_mlp = 3 * spec.d_model * (spec.d_ff // 2) * spec.n_layers
    if policy is None:
        total, active, alternative = dense_total, dense_total, "DENSE"
    else:
        total = shared + policy.expert_count * expert_mlp
        active = shared + policy.top_k * expert_mlp
        alternative = "SPARSE_MOE"
    record = {
        "schema_version": 1, "tier": tier, "alternative": alternative,
        "model_spec": spec.to_dict(), "model_sha256": spec.identity_sha256(),
        "dense_reference_parameters": dense_total, "target_dense_parameters": target,
        "total_parameters": total, "active_parameters": active,
        "active_fraction": active / total,
        "dense_active_ratio": active / dense_total,
        "training_state_floor_bytes": 24 * total,
        "checkpoint_dual_copy_floor_bytes": 4 * total,
        "policy_sha256": _digest(asdict(policy)) if policy is not None else None,
        "router_contract": "HASH_SEEDED_TOPK_NO_SILENT_DROP_V1"
        if policy is not None else "NOT_APPLICABLE",
        "expert_parallel_contract": "DETERMINISTIC_PRIMARY_REPLICA_FIXTURE"
        if policy is not None else "NOT_APPLICABLE",
        "evaluation_contract": "PLAN4_PRODUCER_REQUIRED",
        "serving_contract": "PLAN4_SERVING_ADAPTER_REQUIRED",
        "data_contract": "PLAN2_VERSIONED_PRODUCER_REQUIRED",
        "compute_contract": "PLAN9_EXPLICIT_APPROVAL_REQUIRED",
        "measurement_class": "ARITHMETIC_ONLY_NOT_MEASURED",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
    }
    record["recipe_sha256"] = _digest(record)
    return record


def route_tiny(policy: MoEPolicy, token_ids: tuple[int, ...]) -> dict[str, Any]:
    """Bounded deterministic CPU-free synthetic expert routing.

    Over-capacity results are NO_GO, never silently dropped, truncated,
    renumbered or represented as valid exposure.
    """
    if not isinstance(policy, MoEPolicy):
        raise ExtremeDenied("typed routing policy required")
    MoEPolicy(**asdict(policy))
    if (not isinstance(token_ids, tuple) or not 1 <= len(token_ids) <= 4096
            or any(type(t) is not int or not 0 <= t < 2**63 for t in token_ids)):
        raise ExtremeDenied("invalid bounded tokens")
    loads = [0] * policy.expert_count
    routes: list[tuple[int, ...]] = []
    for token in token_ids:
        digest = hashlib.sha256(
            f"{policy.router_seed_sha256}:{token}".encode("ascii")
        ).digest()
        start = int.from_bytes(digest[:8], "big") % policy.expert_count
        step = 1 + int.from_bytes(digest[8:16], "big") % (policy.expert_count - 1)
        # Wraparound step is not necessarily coprime with expert_count.
        # Sequential collision resolution guarantees distinct destinations.
        choices = tuple((start + i) % policy.expert_count for i in range(policy.top_k))
        routes.append(choices)
        for expert in choices:
            loads[expert] += 1
    capacity = math.ceil(
        policy.capacity_factor * len(token_ids) * policy.top_k / policy.expert_count
    )
    overload = [i for i, n in enumerate(loads) if n > capacity]
    return {
        "schema_version": 1, "policy_sha256": _digest(asdict(policy)),
        "input_sha256": _digest(token_ids), "routing_sha256": _digest(routes),
        "token_count": len(token_ids), "assignments_count": sum(loads),
        "capacity_per_expert": capacity, "expert_loads": loads,
        "max_to_mean_load": max(loads) * policy.expert_count / sum(loads),
        "overloaded_experts": overload, "dropped_assignments": 0,
        "status": "NO_GO" if overload else "SYNTHETIC_ROUTING_PLAUSIBLE",
        "real_expert_execution": False, "training_authorized": False,
    }


def expert_placement(
    policy: MoEPolicy, placement: NodePlacement,
    transport: dict[str, Any], lost_nodes: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(policy, MoEPolicy) or not isinstance(placement, NodePlacement):
        raise ExtremeDenied("typed policy and placement required")
    MoEPolicy(**asdict(policy))
    NodePlacement(**asdict(placement))
    if not isinstance(transport, dict) or (
        transport.get("run_sha256") != placement.run_sha256
        or transport.get("physical_shards_hashed") is not True
        or transport.get("remote_transport_executed") is not False
        or transport.get("canonical_checkpoint_published") is not False
    ):
        raise ExtremeDenied("untrusted fixture transport receipt")
    _sha("transport", transport.get("transport_sha256"))
    if (not isinstance(lost_nodes, tuple) or len(lost_nodes) != len(set(lost_nodes))
            or any(type(n) is not str or n not in placement.node_ids for n in lost_nodes)):
        raise ExtremeDenied("invalid node-loss event")
    owners = []
    live = set(placement.node_ids) - set(lost_nodes)
    for expert in range(policy.expert_count):
        primary = placement.node_ids[expert % len(placement.node_ids)]
        backup = placement.node_ids[(expert + 1) % len(placement.node_ids)]
        if primary not in live and backup not in live:
            raise ExtremeDenied("expert unavailable after failure")
        owners.append({"expert": expert, "primary": primary, "backup": backup,
                       "survivor": primary if primary in live else backup})
    record = {
        "schema_version": 1, "run_sha256": placement.run_sha256,
        "policy_sha256": _digest(asdict(policy)),
        "transport_sha256": transport["transport_sha256"],
        "lost_nodes": sorted(lost_nodes), "expert_owners": owners,
        "real_failover_executed": False, "checkpoint_published": False,
    }
    record["placement_sha256"] = _digest(record)
    return record


def assess_extreme(
    tier: str, policy: MoEPolicy | None, tokens: tuple[int, ...] | None,
    capacity: LargeCapacity, limits: LargeLimits, adapter: LargeAdapter,
    placement: NodePlacement, shards: tuple[tuple[str, bytes, str], ...],
    manifest_sha256: str, *, evaluation_protocol_sha256: str,
    serving_protocol_sha256: str, cost_usd_per_hour: float, cost_budget_usd: float,
    lost_nodes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fail-closed observer-only resource and expert acceptance packet."""
    if (not isinstance(capacity, LargeCapacity) or not isinstance(limits, LargeLimits)
            or not isinstance(adapter, LargeAdapter) or not isinstance(placement, NodePlacement)):
        raise ExtremeDenied("typed capacity/topology evidence required")
    LargeCapacity(**asdict(capacity))
    LargeLimits(**asdict(limits))
    LargeAdapter(**asdict(adapter))
    NodePlacement(**asdict(placement))
    _sha("evaluation", evaluation_protocol_sha256)
    _sha("serving", serving_protocol_sha256)
    _finite("hourly compute cost", cost_usd_per_hour, allow_zero=True)
    _finite("cost ceiling", cost_budget_usd, allow_zero=True)
    recipe = extreme_recipe(tier, policy)
    transport = inspect_transport(placement, shards, manifest_sha256,
                                  lost_nodes=lost_nodes,
                                  recovered_run_sha256=capacity.run_sha256)
    reasons: list[str] = []
    router = None
    experts = None
    if policy is None:
        if tokens is not None:
            raise ExtremeDenied("dense path has no expert tokens")
    else:
        if tokens is None:
            raise ExtremeDenied("MoE path requires tiny routing fixture")
        router = route_tiny(policy, tokens)
        experts = expert_placement(policy, placement, transport, lost_nodes)
        if router["status"] == "NO_GO":
            reasons.append("expert_capacity_denied")
        if capacity.workers < policy.expert_count:
            reasons.append("expert_parallelism_capacity_denied")
    if capacity.run_sha256 != placement.run_sha256:
        reasons.append("run_identity_mismatch")
    if capacity.nodes != len(placement.node_ids) or (
        capacity.workers != capacity.nodes * placement.workers_per_node
    ):
        reasons.append("topology_mismatch")
    if capacity.workers > limits.max_workers or capacity.nodes > limits.max_nodes:
        reasons.append("topology_limit_denied")
    if (adapter.model_sha256 != recipe["model_sha256"]
            or not all((adapter.checkpoint_roundtrip_tested,
                        adapter.same_run_resume_tested, adapter.node_loss_tested,
                        adapter.serving_adapter_tested, adapter.optimizer_recovery_tested))):
        reasons.append("adapter_contract_denied")
    if not adapter.real_backend_executed:
        reasons.append("physical_backend_unverified")
    if capacity.paid_compute_requested:
        reasons.append("paid_compute_denied")
    train_bytes = recipe["training_state_floor_bytes"]
    free_memory = capacity.workers * capacity.free_bytes_per_worker
    if free_memory < train_bytes or free_memory > limits.max_total_memory_bytes:
        reasons.append("memory_denied")
    checkpoint_bytes = recipe["checkpoint_dual_copy_floor_bytes"]
    if (checkpoint_bytes > capacity.free_checkpoint_bytes
            or checkpoint_bytes > limits.max_checkpoint_bytes):
        reasons.append("checkpoint_denied")
    if (capacity.tokens_per_second <= 0
            or capacity.tokens_per_second < limits.min_tokens_per_second):
        reasons.append("throughput_denied")
    elif capacity.target_unique_tokens / capacity.tokens_per_second > limits.max_wallclock_seconds:
        reasons.append("wallclock_denied")
    if capacity.interconnect_bytes_per_second < limits.min_interconnect_bytes_per_second:
        reasons.append("network_denied")
    if (capacity.checkpoint_bytes_per_second <= 0
            or checkpoint_bytes / capacity.checkpoint_bytes_per_second
            > limits.max_checkpoint_seconds):
        reasons.append("transport_denied")
    if capacity.recovery_seconds > limits.max_recovery_seconds:
        reasons.append("recovery_denied")
    hourly = cost_usd_per_hour
    projected = hourly * capacity.target_unique_tokens / max(
        capacity.tokens_per_second, 1e-12
    ) / 3600
    if projected > cost_budget_usd:
        reasons.append("economic_budget_denied")
    record = {
        "schema_version": 1, "tier": tier,
        "alternative": recipe["alternative"], "run_sha256": capacity.run_sha256,
        "recipe_sha256": recipe["recipe_sha256"],
        "capacity_sha256": _digest(asdict(capacity)),
        "limits_sha256": _digest(asdict(limits)),
        "adapter_sha256": _digest(asdict(adapter)),
        "transport_sha256": transport["transport_sha256"],
        "routing_sha256": router["routing_sha256"] if router else None,
        "expert_placement_sha256": experts["placement_sha256"] if experts else None,
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "serving_protocol_sha256": serving_protocol_sha256,
        "projected_cost_usd": projected,
        "status": "NO_GO" if reasons else "PROXY_PLAUSIBLE_NOT_AUTHORIZED",
        "reasons": sorted(set(reasons)),
        "evidence_class": "LOCAL_FREE_SYNTHETIC_EXPERT_TINY_SHARD_ONLY",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
        "real_checkpoint_published": False, "real_expert_execution": False,
    }
    record["packet_sha256"] = _digest(record)
    return record
