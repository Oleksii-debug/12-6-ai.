"""Plan 7 Section 7: 3B/7B/13B bounded distributed path, no launch authority.

Reuse the accepted 1B shard verifier and versioned ModelSpec. All full-size
models are arithmetic descriptions; neither tensors nor paid workers launch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.billion_systems_gate import (
    ADAPTERS,
    BillionGateDenied,
    _digest,
    _finite,
    inspect_shards,
    _positive,
    _sha,
)
from twelve_six.model import ModelSpec

# width, layers, SwiGLU intermediate width, KV heads; attention head_dim=128.
PROFILES = {
    "3B": (3072, 30, 8192, 8),
    "7B": (4096, 40, 11008, 8),
    "13B": (5120, 48, 13824, 8),
}
SERVING = {
    "3B": {"min_tensor_parallel": 1, "min_context": 1024},
    "7B": {"min_tensor_parallel": 2, "min_context": 1024},
    "13B": {"min_tensor_parallel": 4, "min_context": 1024},
}


class LargePathDenied(BillionGateDenied):
    """Malformed or untrusted distributed planning evidence."""


def large_spec(tier: str) -> ModelSpec:
    if type(tier) is not str or tier not in PROFILES:
        raise LargePathDenied("unknown large-scale tier")
    width, layers, ff, kv = PROFILES[tier]
    return ModelSpec(
        schema_version=1, vocab_size=8192, max_seq_len=1024,
        d_model=width, n_layers=layers, n_heads=width // 128,
        n_kv_heads=kv, head_dim=128, d_ff=ff, rope_rotary_dim=128,
    )


def large_recipe(tier: str) -> dict[str, Any]:
    spec = large_spec(tier)
    parameters = spec.parameter_count()
    target = int(tier[:-1]) * 1_000_000_000
    if abs(parameters - target) > target // 10:
        raise LargePathDenied("scale parameter tolerance exceeded")
    record = {
        "schema_version": 1, "tier": tier, "model_spec": spec.to_dict(),
        "model_sha256": spec.identity_sha256(),
        "parameters": parameters, "target_parameters": target,
        "train_state_floor_bytes": 24 * parameters,
        "checkpoint_dual_copy_floor_bytes": 4 * parameters,
        "bf16_weights_bytes": 2 * parameters,
        "architecture": "DENSE_GQA_SWIGLU_ROPE_MODELSPEC_V1",
        "data_contract": "PLAN2_VERSIONED_DATA_TOKENIZER_PACKING_PRODUCER_REQUIRED",
        "evaluation_contract": "PLAN4_VERSIONED_HOLDOUT_PRODUCER_REQUIRED",
        "compute_contract": "PLAN9_EXPLICIT_TRAINING_AND_COMPUTE_AUTHORITY_REQUIRED",
        "parent_contract": "S6_1B_CHECKPOINT_OR_NEW_SCIENTIFIC_RUN",
        "sharded_checkpoint_contract": "S6_ORDERED_PHYSICAL_SHARDS_AND_MANIFEST",
        "serving": dict(SERVING[tier]),
        "backend_contracts": sorted(ADAPTERS),
        "measurement_class": "ARITHMETIC_ONLY_NOT_MEASURED_PEAK",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
    }
    record["recipe_sha256"] = _digest(record)
    return record


@dataclass(frozen=True, slots=True)
class LargeCapacity:
    schema_version: int
    run_sha256: str
    workers: int
    nodes: int
    free_bytes_per_worker: int
    free_checkpoint_bytes: int
    tokens_per_second: float
    interconnect_bytes_per_second: float
    checkpoint_bytes_per_second: float
    recovery_seconds: float
    target_unique_tokens: int
    data_contract_sha256: str
    evaluation_contract_sha256: str
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise LargePathDenied("capacity version")
        for name in ("run_sha256", "data_contract_sha256",
                     "evaluation_contract_sha256"):
            _sha(name, getattr(self, name))
        for name in ("workers", "nodes", "free_bytes_per_worker",
                     "free_checkpoint_bytes", "target_unique_tokens"):
            _positive(name, getattr(self, name))
        if self.nodes > self.workers:
            raise LargePathDenied("nodes exceed workers")
        for name in ("tokens_per_second", "interconnect_bytes_per_second",
                     "checkpoint_bytes_per_second", "recovery_seconds"):
            _finite(name, getattr(self, name), allow_zero=True)
        if type(self.paid_compute_requested) is not bool:
            raise LargePathDenied("paid compute flag")


@dataclass(frozen=True, slots=True)
class LargeLimits:
    schema_version: int
    max_workers: int
    max_nodes: int
    max_total_memory_bytes: int
    max_checkpoint_bytes: int
    max_wallclock_seconds: float
    max_checkpoint_seconds: float
    max_recovery_seconds: float
    min_tokens_per_second: float
    min_interconnect_bytes_per_second: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise LargePathDenied("limits version")
        for name in ("max_workers", "max_nodes", "max_total_memory_bytes",
                     "max_checkpoint_bytes"):
            _positive(name, getattr(self, name))
        for name in ("max_wallclock_seconds", "max_checkpoint_seconds",
                     "max_recovery_seconds", "min_tokens_per_second",
                     "min_interconnect_bytes_per_second"):
            _finite(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class LargeAdapter:
    schema_version: int
    name: str
    implementation_sha256: str
    model_sha256: str
    protocol_sha256: str
    checkpoint_roundtrip_tested: bool
    same_run_resume_tested: bool
    node_loss_tested: bool
    serving_adapter_tested: bool
    optimizer_recovery_tested: bool
    real_backend_executed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise LargePathDenied("adapter version")
        if type(self.name) is not str or self.name not in ADAPTERS:
            raise LargePathDenied("unsupported backend")
        for name in ("implementation_sha256", "model_sha256",
                     "protocol_sha256"):
            _sha(name, getattr(self, name))
        for name in ("checkpoint_roundtrip_tested", "same_run_resume_tested",
                     "node_loss_tested", "serving_adapter_tested",
                     "optimizer_recovery_tested", "real_backend_executed"):
            if type(getattr(self, name)) is not bool:
                raise LargePathDenied("nonboolean adapter evidence")


def assess_large(
    tier: str, capacity: LargeCapacity, limits: LargeLimits,
    adapter: LargeAdapter, physical_shards: tuple[tuple[str, bytes, str], ...],
    manifest_sha256: str, *, recovered_run_sha256: str | None = None,
    serving_tensor_parallel: int = 1,
) -> dict[str, Any]:
    """Bounded deterministic capacity/recovery matrix, strictly advisory."""
    if not isinstance(capacity, LargeCapacity) or not isinstance(limits, LargeLimits):
        raise LargePathDenied("typed capacity and limits required")
    if not isinstance(adapter, LargeAdapter):
        raise LargePathDenied("typed backend required")
    LargeCapacity(**asdict(capacity))
    LargeLimits(**asdict(limits))
    LargeAdapter(**asdict(adapter))
    _positive("serving_tensor_parallel", serving_tensor_parallel)
    if recovered_run_sha256 is not None:
        _sha("recovered_run_sha256", recovered_run_sha256)
        if recovered_run_sha256 != capacity.run_sha256:
            raise LargePathDenied("scientific run identity changed")
    recipe = large_recipe(tier)
    shard_receipt = inspect_shards(physical_shards, manifest_sha256)
    reasons = []
    if adapter.model_sha256 != recipe["model_sha256"]:
        reasons.append("model_binding_mismatch")
    if not all((adapter.checkpoint_roundtrip_tested, adapter.same_run_resume_tested,
                adapter.node_loss_tested, adapter.serving_adapter_tested,
                adapter.optimizer_recovery_tested)):
        reasons.append("adapter_unqualified")
    if not adapter.real_backend_executed:
        reasons.append("physical_backend_unverified")
    if capacity.paid_compute_requested:
        reasons.append("paid_compute_denied")
    if capacity.workers > limits.max_workers or capacity.nodes > limits.max_nodes:
        reasons.append("topology_limit_denied")
    available = capacity.workers * capacity.free_bytes_per_worker
    if (available < recipe["train_state_floor_bytes"]
            or available > limits.max_total_memory_bytes):
        reasons.append("memory_denied")
    storage = recipe["checkpoint_dual_copy_floor_bytes"]
    if storage > capacity.free_checkpoint_bytes or storage > limits.max_checkpoint_bytes:
        reasons.append("checkpoint_storage_denied")
    if capacity.interconnect_bytes_per_second < limits.min_interconnect_bytes_per_second:
        reasons.append("interconnect_denied")
    if capacity.tokens_per_second < limits.min_tokens_per_second:
        reasons.append("throughput_denied")
    elif capacity.target_unique_tokens / capacity.tokens_per_second > limits.max_wallclock_seconds:
        reasons.append("time_budget_denied")
    if (capacity.checkpoint_bytes_per_second <= 0
            or storage / capacity.checkpoint_bytes_per_second
            > limits.max_checkpoint_seconds):
        reasons.append("checkpoint_transport_denied")
    if capacity.recovery_seconds > limits.max_recovery_seconds:
        reasons.append("recovery_denied")
    if serving_tensor_parallel < recipe["serving"]["min_tensor_parallel"]:
        reasons.append("serving_topology_denied")
    record = {
        "schema_version": 1, "tier": tier, "run_sha256": capacity.run_sha256,
        "recipe_sha256": recipe["recipe_sha256"],
        "capacity_sha256": _digest(asdict(capacity)),
        "limits_sha256": _digest(asdict(limits)),
        "adapter_sha256": _digest(asdict(adapter)),
        "shard_receipt_sha256": _digest(shard_receipt),
        "status": "NO_GO" if reasons else "PROXY_SYSTEMS_PLAUSIBLE_NOT_AUTHORIZED",
        "reasons": sorted(set(reasons)), "serving_tensor_parallel": serving_tensor_parallel,
        "evidence_class": "LOCAL_FREE_PHYSICAL_TINY_SHARD_FIXTURE_ONLY",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False, "canonical_checkpoint_published": False,
    }
    record["packet_sha256"] = _digest(record)
    return record
