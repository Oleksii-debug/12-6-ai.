"""Plan 7 Section 8: 30B/70B/100B bounded multi-node planning.

The model sizes are arithmetic specifications. No tensors, nodes, paid
compute, training, checkpoint publication or serving are launched here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import _digest, _finite, _positive, _sha, inspect_shards
from twelve_six.large_scale_paths import LargeAdapter, LargeCapacity, LargeLimits
from twelve_six.model import ModelSpec

# Width, layers, FFN dimension, KV heads. All use 128-wide heads and GQA.
PROFILES = {"30B": (6144, 70, 18432, 8),
            "70B": (8192, 80, 28672, 8),
            "100B": (9216, 100, 28672, 8)}


class VeryLargeDenied(ValueError):
    """Malformed input or an invalid checkpoint/recovery claim."""


def very_large_spec(tier: str) -> ModelSpec:
    if type(tier) is not str or tier not in PROFILES:
        raise VeryLargeDenied("unknown tier")
    width, layers, ff, kv = PROFILES[tier]
    return ModelSpec(schema_version=1, vocab_size=8192, max_seq_len=1024,
                     d_model=width, n_layers=layers, n_heads=width // 128,
                     n_kv_heads=kv, head_dim=128, d_ff=ff, rope_rotary_dim=128)


def very_large_recipe(tier: str, *, alternative: str = "DENSE") -> dict[str, Any]:
    if type(alternative) is not str or alternative not in ("DENSE", "SPARSE_MOE"):
        raise VeryLargeDenied("unknown architecture alternative")
    spec = very_large_spec(tier)
    parameters = spec.parameter_count()
    target = int(tier[:-1]) * 1_000_000_000
    if abs(parameters - target) > target // 10:
        raise VeryLargeDenied("parameter tolerance")
    record = {
        "schema_version": 1, "tier": tier, "alternative": alternative,
        "model_spec": spec.to_dict(), "model_sha256": spec.identity_sha256(),
        "dense_reference_parameters": parameters, "target_parameters": target,
        "training_state_floor_bytes": 24 * parameters,
        "checkpoint_dual_copy_floor_bytes": 4 * parameters,
        "bf16_full_weights_floor_bytes": 2 * parameters,
        "sparse_alternative_contract": "PLAN7_S9_EXPERT_LAYOUT_REQUIRED"
        if alternative == "SPARSE_MOE" else "NOT_APPLICABLE",
        "capacity_class": "ARITHMETIC_ONLY_NO_PEAK_MEASUREMENT",
        "data_contract": "PLAN2_PRODUCER_REQUIRED",
        "evaluation_contract": "PLAN4_HOLDOUT_PRODUCER_REQUIRED",
        "authority_contract": "PLAN9_EXPLICIT_COMPUTE_AND_TRAINING_APPROVAL",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
    }
    # A hypothetical sparse path receives NO discount on memory admission.
    record["recipe_sha256"] = _digest(record)
    return record


@dataclass(frozen=True, slots=True)
class NodePlacement:
    schema_version: int
    run_sha256: str
    node_ids: tuple[str, ...]
    workers_per_node: int
    artifact_protocol_sha256: str
    network_failure_domain: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise VeryLargeDenied("placement version")
        _sha("run", self.run_sha256)
        _sha("artifact protocol", self.artifact_protocol_sha256)
        if (not isinstance(self.node_ids, tuple) or not 2 <= len(self.node_ids) <= 128
                or any(type(x) is not str or not x or len(x) > 64
                       or not x.isascii() or not all(c.isalnum() or c in "-_" for c in x)
                       for x in self.node_ids)
                or len(self.node_ids) != len(set(self.node_ids))):
            raise VeryLargeDenied("node ids must be distinct bounded identities")
        _positive("workers per node", self.workers_per_node)
        if type(self.network_failure_domain) is not str or not self.network_failure_domain:
            raise VeryLargeDenied("missing failure domain")


def inspect_transport(
    placement: NodePlacement, shards: tuple[tuple[str, bytes, str], ...],
    manifest_sha256: str, *, lost_nodes: tuple[str, ...] = (),
    recovered_run_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify physical tiny shards and deterministic replicated-node recovery.

    This is a fixture-only *transport plan*, not an actual remote copy.
    """
    if not isinstance(placement, NodePlacement):
        raise VeryLargeDenied("typed placement required")
    NodePlacement(**asdict(placement))
    if (not isinstance(lost_nodes, tuple) or any(type(n) is not str for n in lost_nodes)
            or len(lost_nodes) != len(set(lost_nodes))
            or any(n not in placement.node_ids for n in lost_nodes)):
        raise VeryLargeDenied("untrusted lost-node report")
    if recovered_run_sha256 is not None:
        _sha("recovered run", recovered_run_sha256)
        if recovered_run_sha256 != placement.run_sha256:
            raise VeryLargeDenied("scientific run identity changed")
    receipt = inspect_shards(shards, manifest_sha256)
    placements = []
    available = set(placement.node_ids) - set(lost_nodes)
    for i, (name, _, claimed_sha) in enumerate(shards):
        primary = placement.node_ids[i % len(placement.node_ids)]
        replica = placement.node_ids[(i + 1) % len(placement.node_ids)]
        owners = (primary, replica)
        placements.append({"shard": name, "sha256": claimed_sha, "owners": owners})
        if not any(n in available for n in owners):
            raise VeryLargeDenied("incomplete checkpoint after node loss")
    record = {"schema_version": 1, "run_sha256": placement.run_sha256,
              "artifact_protocol_sha256": placement.artifact_protocol_sha256,
              "placement_sha256": _digest(asdict(placement)),
              "manifest_sha256": receipt["manifest_sha256"],
              "placements": placements, "lost_nodes": sorted(lost_nodes),
              "physical_shards_hashed": True, "remote_transport_executed": False,
              "canonical_checkpoint_published": False,
              "same_run_recovery_verified": True}
    record["transport_sha256"] = _digest(record)
    return record


def assess_very_large(
    tier: str, alternative: str, capacity: LargeCapacity, limits: LargeLimits,
    adapter: LargeAdapter, placement: NodePlacement,
    shards: tuple[tuple[str, bytes, str], ...], manifest_sha256: str,
    *, hourly_usd: float, budget_usd: float,
    lost_nodes: tuple[str, ...] = (), serving_tensor_parallel: int = 1,
) -> dict[str, Any]:
    """Conservative, fail-closed economic/topology/admission receipt."""
    if (not isinstance(capacity, LargeCapacity) or not isinstance(limits, LargeLimits)
            or not isinstance(adapter, LargeAdapter)):
        raise VeryLargeDenied("typed evidence required")
    LargeCapacity(**asdict(capacity))
    LargeLimits(**asdict(limits))
    LargeAdapter(**asdict(adapter))
    _finite("hourly cost", hourly_usd, allow_zero=True)
    _finite("budget", budget_usd, allow_zero=True)
    _positive("tensor parallel", serving_tensor_parallel)
    recipe = very_large_recipe(tier, alternative=alternative)
    transport = inspect_transport(placement, shards, manifest_sha256,
                                  lost_nodes=lost_nodes,
                                  recovered_run_sha256=capacity.run_sha256)
    reasons = []
    if placement.run_sha256 != capacity.run_sha256:
        reasons.append("run_identity_mismatch")
    if (len(placement.node_ids) != capacity.nodes
            or capacity.workers != capacity.nodes * placement.workers_per_node):
        reasons.append("topology_mismatch")
    if capacity.nodes < 2 or capacity.nodes > limits.max_nodes:
        reasons.append("node_admission_denied")
    if capacity.workers > limits.max_workers:
        reasons.append("worker_admission_denied")
    if (adapter.model_sha256 != recipe["model_sha256"]
            or not all((adapter.checkpoint_roundtrip_tested,
                        adapter.same_run_resume_tested, adapter.node_loss_tested,
                        adapter.serving_adapter_tested, adapter.optimizer_recovery_tested))):
        reasons.append("backend_contract_unqualified")
    if not adapter.real_backend_executed:
        reasons.append("physical_backend_unverified")
    if capacity.paid_compute_requested:
        reasons.append("paid_compute_denied")
    free_memory = capacity.workers * capacity.free_bytes_per_worker
    if (free_memory < recipe["training_state_floor_bytes"]
            or free_memory > limits.max_total_memory_bytes):
        reasons.append("memory_denied")
    storage = recipe["checkpoint_dual_copy_floor_bytes"]
    if (storage > capacity.free_checkpoint_bytes or storage > limits.max_checkpoint_bytes):
        reasons.append("checkpoint_denied")
    if (capacity.tokens_per_second < limits.min_tokens_per_second
            or capacity.tokens_per_second <= 0):
        reasons.append("throughput_denied")
    elif capacity.target_unique_tokens / capacity.tokens_per_second > limits.max_wallclock_seconds:
        reasons.append("wallclock_denied")
    if capacity.interconnect_bytes_per_second < limits.min_interconnect_bytes_per_second:
        reasons.append("interconnect_denied")
    if (capacity.checkpoint_bytes_per_second <= 0
            or storage / capacity.checkpoint_bytes_per_second > limits.max_checkpoint_seconds):
        reasons.append("artifact_transport_denied")
    if capacity.recovery_seconds > limits.max_recovery_seconds:
        reasons.append("recovery_slo_denied")
    if serving_tensor_parallel > capacity.workers:
        reasons.append("serving_parallelism_denied")
    if alternative == "SPARSE_MOE":
        reasons.append("moe_expert_contract_pending_section9")
    projected_usd = hourly_usd * capacity.target_unique_tokens / max(
        capacity.tokens_per_second, 1e-12) / 3600
    if projected_usd > budget_usd:
        reasons.append("economic_budget_denied")
    record = {"schema_version": 1, "tier": tier, "alternative": alternative,
              "run_sha256": capacity.run_sha256, "recipe_sha256": recipe["recipe_sha256"],
              "capacity_sha256": _digest(asdict(capacity)),
              "limits_sha256": _digest(asdict(limits)),
              "adapter_sha256": _digest(asdict(adapter)),
              "transport_sha256": transport["transport_sha256"],
              "projected_cost_usd": projected_usd,
              "status": "NO_GO" if reasons else "PROXY_PLAUSIBLE_NOT_AUTHORIZED",
              "reasons": sorted(set(reasons)),
              "evidence_class": "LOCAL_FREE_TINY_SHARD_AND_ARITHMETIC_ONLY",
              "training_authorized": False, "launch_authorized": False,
              "paid_compute_authorized": False, "canonical_checkpoint_published": False}
    record["packet_sha256"] = _digest(record)
    return record
