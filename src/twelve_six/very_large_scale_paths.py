"""Plan 7 Section 8: 30B/70B/100B topology and economic admission.

No full-size tensors, remote actions, training, backend launches or paid compute.
All physical shards are small local fixtures checked by accepted S6 verifier.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import (
    _digest, _finite, _positive, _sha, inspect_shards,
)
from twelve_six.large_scale_paths import (
    LargeAdapter, LargeCapacity, LargeLimits, LargePathDenied,
)
from twelve_six.model import ModelSpec

# Dense baseline only. Sparse/MoE alternatives have separate future authority.
PROFILES = {
    "30B": (6656, 66, 17920, 4),
    "70B": (8192, 80, 28672, 8),
    "100B": (9216, 112, 24576, 8),
}
MODES = frozenset({"DENSE", "SPARSE_CONTRACT_ONLY"})


def very_large_spec(tier: str) -> ModelSpec:
    if type(tier) is not str or tier not in PROFILES:
        raise LargePathDenied("unsupported very-large tier")
    width, layers, ff, kv = PROFILES[tier]
    return ModelSpec(
        schema_version=1, vocab_size=8192, max_seq_len=1024,
        d_model=width, n_layers=layers, n_heads=width // 128,
        n_kv_heads=kv, head_dim=128, d_ff=ff, rope_rotary_dim=128,
    )


def very_large_recipe(tier: str, mode: str = "DENSE") -> dict[str, Any]:
    if type(mode) is not str or mode not in MODES:
        raise LargePathDenied("unsupported architecture mode")
    spec = very_large_spec(tier)
    n = spec.parameter_count()
    target = int(tier[:-1]) * 1_000_000_000
    if abs(n - target) > target // 10:
        raise LargePathDenied("parameter tolerance exceeded")
    record = {
        "schema_version": 1, "tier": tier, "architecture_mode": mode,
        "modelspec_sha256": spec.identity_sha256(),
        "dense_baseline_modelspec": spec.to_dict(),
        "dense_baseline_parameters": n, "target_parameters": target,
        "dense_train_state_floor_bytes": 24 * n,
        "dense_dual_checkpoint_floor_bytes": 4 * n,
        "dense_bf16_weights_bytes": 2 * n,
        "sparse_alternative": "PLAN7_S9_VERSIONED_MOE_CONTRACT_REQUIRED",
        "sparse_parameter_accounting_valid": mode == "DENSE",
        "data_contract": "PLAN2_VERSIONED_DATA_AND_ORDER_REQUIRED",
        "checkpoint_contract": "S6_ORDERED_SHA256_MANIFEST_AND_ATOMIC_PUBLISH",
        "artifact_transport_contract": "EXACT_SHARD_OWNER_HASH_AND_COMPLETE_SET",
        "distributed_resume_contract": "SAME_RUN_RECIPE_AND_DATA_ORDER_SHA256",
        "serving_contract": "PLAN4_TOPOLOGY_AWARE_EXPORT_REQUIRED",
        "evaluation_contract": "PLAN4_ISOLATED_HOLDOUT_REQUIRED",
        "compute_contract": "PLAN9_EXPLICIT_TRAINING_COMPUTE_AUTHORIZATION",
        "measurement_class": "ARITHMETIC_ONLY_NOT_ACTUAL_PEAK_OR_MOE",
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False,
    }
    record["recipe_sha256"] = _digest(record)
    return record


@dataclass(frozen=True, slots=True)
class MultiNodeEvidence:
    schema_version: int
    node_ids: tuple[str, ...]
    worker_nodes: tuple[str, ...]
    topology_sha256: str
    scientific_recipe_sha256: str
    data_order_sha256: str
    dollars_per_node_hour: float
    max_estimated_dollars: float
    transport_roundtrip_tested: bool
    resume_same_run_tested: bool
    failover_tested: bool
    physical_multinode_executed: bool = False
    sparse_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise LargePathDenied("topology schema")
        for name in ("topology_sha256", "scientific_recipe_sha256",
                     "data_order_sha256"):
            _sha(name, getattr(self, name))
        if self.sparse_contract_sha256 is not None:
            _sha("sparse_contract", self.sparse_contract_sha256)
        if (not isinstance(self.node_ids, tuple) or len(self.node_ids) < 2
                or len(self.node_ids) > 128 or len(set(self.node_ids)) != len(self.node_ids)):
            raise LargePathDenied("invalid distinct multi-node placement")
        if any(type(v) is not str or not v or len(v) > 64 for v in self.node_ids):
            raise LargePathDenied("invalid node id")
        if (not isinstance(self.worker_nodes, tuple)
                or not 2 <= len(self.worker_nodes) <= 1024
                or any(type(v) is not str or v not in self.node_ids
                       for v in self.worker_nodes)
                or tuple(sorted(set(self.worker_nodes))) != tuple(sorted(self.node_ids))):
            raise LargePathDenied("workers missing/unknown nodes")
        if tuple(self.worker_nodes) != tuple(sorted(self.worker_nodes)):
            raise LargePathDenied("unstable rank-to-node ordering")
        identity = {"node_ids": self.node_ids, "worker_nodes": self.worker_nodes}
        if self.topology_sha256 != _digest(identity):
            raise LargePathDenied("topology placement digest mismatch")
        for name in ("dollars_per_node_hour", "max_estimated_dollars"):
            _finite(name, getattr(self, name), allow_zero=True)
        for name in ("transport_roundtrip_tested", "resume_same_run_tested",
                     "failover_tested", "physical_multinode_executed"):
            if type(getattr(self, name)) is not bool:
                raise LargePathDenied("untrusted boolean evidence")


def assess_very_large(
    tier: str, mode: str, capacity: LargeCapacity, limits: LargeLimits,
    adapter: LargeAdapter, topology: MultiNodeEvidence,
    shards: tuple[tuple[str, bytes, str], ...], manifest_sha256: str,
    *, recovered_run_sha256: str | None = None,
    recovered_data_order_sha256: str | None = None,
    failed_node: str | None = None,
) -> dict[str, Any]:
    """Simulated admission with topology/failure, transport and cost fencing."""
    if not isinstance(topology, MultiNodeEvidence):
        raise LargePathDenied("topology evidence required")
    MultiNodeEvidence(**asdict(topology))
    if not isinstance(capacity, LargeCapacity) or not isinstance(limits, LargeLimits):
        raise LargePathDenied("capacity and limits required")
    if not isinstance(adapter, LargeAdapter):
        raise LargePathDenied("backend required")
    LargeCapacity(**asdict(capacity))
    LargeLimits(**asdict(limits))
    LargeAdapter(**asdict(adapter))
    if recovered_run_sha256 is not None:
        _sha("recovered_run", recovered_run_sha256)
        if recovered_run_sha256 != capacity.run_sha256:
            raise LargePathDenied("scientific run identity changed")
    if recovered_data_order_sha256 is not None:
        _sha("recovered_data_order_sha256", recovered_data_order_sha256)
        if recovered_data_order_sha256 != topology.data_order_sha256:
            raise LargePathDenied("data order changed on resume")
    if failed_node is not None and failed_node not in topology.node_ids:
        raise LargePathDenied("unknown lost node")
    recipe = very_large_recipe(tier, mode)
    if topology.scientific_recipe_sha256 != recipe["recipe_sha256"]:
        raise LargePathDenied("scientific recipe changed during recovery")
    receipt = inspect_shards(shards, manifest_sha256)
    reasons: list[str] = []
    if capacity.nodes != len(topology.node_ids) or capacity.workers != len(topology.worker_nodes):
        reasons.append("topology_capacity_mismatch")
    if topology.physical_multinode_executed is False:
        reasons.append("physical_multinode_unverified")
    if not (topology.transport_roundtrip_tested and topology.resume_same_run_tested
            and topology.failover_tested):
        reasons.append("topology_recovery_unqualified")
    if not (adapter.checkpoint_roundtrip_tested and adapter.same_run_resume_tested
            and adapter.node_loss_tested and adapter.optimizer_recovery_tested):
        reasons.append("backend_recovery_unqualified")
    if adapter.model_sha256 != recipe["modelspec_sha256"]:
        reasons.append("backend_model_mismatch")
    if not adapter.real_backend_executed:
        reasons.append("backend_not_executed")
    if capacity.paid_compute_requested:
        reasons.append("paid_compute_denied")
    if mode == "SPARSE_CONTRACT_ONLY":
        reasons.append("sparse_expert_backend_not_qualified")
        if topology.sparse_contract_sha256 is None:
            reasons.append("sparse_contract_missing")
    if capacity.nodes > limits.max_nodes or capacity.workers > limits.max_workers:
        reasons.append("resource_topology_denied")
    active_workers = sum(node != failed_node for node in topology.worker_nodes)
    active_nodes = len(topology.node_ids) - int(failed_node is not None)
    if active_nodes < 2:
        reasons.append("node_loss_no_quorum")
    available = active_workers * capacity.free_bytes_per_worker
    if (available < recipe["dense_train_state_floor_bytes"]
            or available > limits.max_total_memory_bytes):
        reasons.append("memory_denied")
    storage = recipe["dense_dual_checkpoint_floor_bytes"]
    if storage > capacity.free_checkpoint_bytes or storage > limits.max_checkpoint_bytes:
        reasons.append("checkpoint_storage_denied")
    if capacity.interconnect_bytes_per_second < limits.min_interconnect_bytes_per_second:
        reasons.append("interconnect_denied")
    if capacity.tokens_per_second < limits.min_tokens_per_second:
        reasons.append("throughput_denied")
    estimated_seconds = (capacity.target_unique_tokens / capacity.tokens_per_second
                         if capacity.tokens_per_second > 0 else math.inf)
    if estimated_seconds > limits.max_wallclock_seconds:
        reasons.append("wallclock_denied")
    estimated_dollars = (estimated_seconds / 3600.0
                         * len(topology.node_ids) * topology.dollars_per_node_hour)
    if not math.isfinite(estimated_dollars):
        reasons.append("economic_estimate_invalid")
    elif estimated_dollars > topology.max_estimated_dollars:
        reasons.append("economic_budget_denied")
    if (capacity.checkpoint_bytes_per_second <= 0
            or storage / capacity.checkpoint_bytes_per_second
            > limits.max_checkpoint_seconds):
        reasons.append("checkpoint_transport_denied")
    if capacity.recovery_seconds > limits.max_recovery_seconds:
        reasons.append("recovery_slo_denied")
    packet = {
        "schema_version": 1, "tier": tier, "mode": mode,
        "recipe_sha256": recipe["recipe_sha256"],
        "run_sha256": capacity.run_sha256,
        "capacity_sha256": _digest(asdict(capacity)),
        "limits_sha256": _digest(asdict(limits)),
        "adapter_sha256": _digest(asdict(adapter)),
        "topology_evidence_sha256": _digest(asdict(topology)),
        "physical_shard_receipt_sha256": _digest(receipt),
        "failed_node": failed_node, "active_workers": active_workers,
        "estimated_cost_usd": estimated_dollars if math.isfinite(estimated_dollars) else None,
        "status": "NO_GO" if reasons else "PROXY_PATH_PLAUSIBLE_NOT_AUTHORIZED",
        "reasons": sorted(set(reasons)),
        "evidence_class": "BOUNDED_MULTINODE_SIMULATION_NOT_PHYSICAL",
        "launch_authorized": False, "training_authorized": False,
        "paid_compute_authorized": False, "checkpoint_promoted": False,
        "production_artifact_transported": False,
    }
    packet["packet_sha256"] = _digest(packet)
    return packet
