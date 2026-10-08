"""Plan 7 Section 6: 1B distributed systems admission, LOCAL_FREE only.

No adapter execution, paid compute, training, launch, or checkpoint promotion.
This is an independent capacity and failure/recovery *contract* gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from twelve_six.model import ModelSpec
from twelve_six.product_scale_recipes import scale_recipe

ADAPTERS = frozenset({"FSDP", "DeepSpeed", "TorchTitan", "Megatron"})


class BillionGateDenied(ValueError):
    """Malformed input or unqualified evidence must never reach admission."""


def _digest(value: Any) -> str:
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BillionGateDenied("noncanonical evidence") from exc
    return hashlib.sha256(data).hexdigest()


def _sha(name: str, value: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise BillionGateDenied(f"{name}: lowercase SHA256 required")


def _positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise BillionGateDenied(f"{name}: positive int required")


def _finite(name: str, value: float, *, allow_zero: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise BillionGateDenied(f"{name}: finite positive measurement required")


def billion_spec() -> ModelSpec:
    """Arithmetic spec only; no 1B model weights allocated."""
    return ModelSpec(
        schema_version=1, vocab_size=8192, max_seq_len=1024,
        d_model=2048, n_layers=21, n_heads=32, n_kv_heads=8,
        head_dim=64, d_ff=6144, rope_rotary_dim=64,
    )


def billion_recipe() -> dict[str, Any]:
    spec = billion_spec()
    parameters = spec.parameter_count()
    assert abs(parameters - 1_000_000_000) < 50_000_000
    recipe = {
        "schema_version": 1, "target": "1B",
        "modelspec_sha256": spec.identity_sha256(),
        "modelspec": spec.to_dict(), "parameters": parameters,
        "train_state_floor_bytes": 24 * parameters,
        "checkpoint_dual_copy_floor_bytes": 4 * parameters,
        "source_200m_recipe_sha256": scale_recipe("200M")["recipe_sha256"],
        "required_adapter_contracts": sorted(ADAPTERS),
        "evidence_class": "ARITHMETIC_ONLY_NO_PEAK_MEMORY_MEASUREMENT",
        "training_authorized": False, "launch_authorized": False,
        "paid_compute_authorized": False,
    }
    recipe["recipe_sha256"] = _digest(recipe)
    return recipe


@dataclass(frozen=True, slots=True)
class AdapterEvidence:
    schema_version: int
    name: str
    version: str
    implementation_sha256: str
    model_sha256: str
    protocol_sha256: str
    licence_reviewed: bool
    optimizer_semantics_tested: bool
    sharded_save_load_tested: bool
    restart_same_run_tested: bool
    worker_loss_tested: bool
    actual_backend_executed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise BillionGateDenied("unsupported adapter schema")
        if self.name not in ADAPTERS or not isinstance(self.version, str):
            raise BillionGateDenied("unsupported adapter/version")
        if not self.version.strip() or len(self.version) > 128:
            raise BillionGateDenied("missing adapter version")
        for field in ("implementation_sha256", "model_sha256", "protocol_sha256"):
            _sha(field, getattr(self, field))
        for field in ("licence_reviewed", "optimizer_semantics_tested",
                      "sharded_save_load_tested", "restart_same_run_tested",
                      "worker_loss_tested", "actual_backend_executed"):
            if type(getattr(self, field)) is not bool:
                raise BillionGateDenied(f"{field}: bool required")


@dataclass(frozen=True, slots=True)
class CapacityEvidence:
    schema_version: int
    run_sha256: str
    workers: int
    nodes: int
    free_bytes_per_worker: int
    free_checkpoint_bytes: int
    measured_tokens_per_second: float
    measured_interconnect_bytes_per_second: float
    measured_checkpoint_bytes_per_second: float
    measured_recovery_seconds: float
    target_unique_tokens: int
    parent_200m_terminal_proof_sha256: str
    paid_compute_requested: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise BillionGateDenied("unsupported capacity schema")
        for field in ("run_sha256", "parent_200m_terminal_proof_sha256"):
            _sha(field, getattr(self, field))
        for field in ("workers", "nodes", "free_bytes_per_worker",
                      "free_checkpoint_bytes", "target_unique_tokens"):
            _positive(field, getattr(self, field))
        if self.nodes > self.workers:
            raise BillionGateDenied("nodes exceed workers")
        for field in ("measured_tokens_per_second",
                      "measured_interconnect_bytes_per_second",
                      "measured_checkpoint_bytes_per_second"):
            _finite(field, getattr(self, field), allow_zero=True)
        _finite("measured_recovery_seconds", self.measured_recovery_seconds,
                allow_zero=True)
        if type(self.paid_compute_requested) is not bool:
            raise BillionGateDenied("paid compute marker is not bool")


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
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
            raise BillionGateDenied("unsupported limits schema")
        for field in ("max_workers", "max_nodes", "max_total_memory_bytes",
                      "max_checkpoint_bytes"):
            _positive(field, getattr(self, field))
        for field in ("max_wallclock_seconds", "max_checkpoint_seconds",
                      "max_recovery_seconds", "min_tokens_per_second",
                      "min_interconnect_bytes_per_second"):
            _finite(field, getattr(self, field))


def inspect_shards(shards: tuple[tuple[str, bytes, str], ...],
                   expected_manifest_sha256: str) -> dict[str, Any]:
    """Physically hash bounded, ordered shards; partial/corrupt sets are refused."""
    _sha("manifest", expected_manifest_sha256)
    if not isinstance(shards, tuple) or not 1 <= len(shards) <= 16:
        raise BillionGateDenied("invalid shard count")
    manifest = []
    seen = set()
    for i, shard in enumerate(shards):
        if not isinstance(shard, tuple) or len(shard) != 3:
            raise BillionGateDenied("malformed shard tuple")
        name, payload, claimed = shard
        if (name != f"part-{i:02d}" or name in seen
                or not isinstance(payload, bytes) or not 0 < len(payload) <= 65536):
            raise BillionGateDenied("missing, reordered, or oversized shard")
        seen.add(name)
        _sha("shard", claimed)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != claimed:
            raise BillionGateDenied("corrupt shard payload")
        manifest.append({"name": name, "sha256": actual, "bytes": len(payload)})
    computed = _digest(manifest)
    if computed != expected_manifest_sha256:
        raise BillionGateDenied("shard manifest mismatch")
    return {"schema_version": 1, "manifest_sha256": computed,
            "shard_count": len(shards), "partial": False,
            "training_authorized": False, "launch_authorized": False}


def shard_manifest(pieces: tuple[bytes, ...]) -> str:
    """Fixture-only manifest producer; never installs canonical checkpoints."""
    if not isinstance(pieces, tuple) or not 1 <= len(pieces) <= 16:
        raise BillionGateDenied("invalid fixture pieces")
    for payload in pieces:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 65536:
            raise BillionGateDenied("invalid fixture payload")
    return _digest([{"name": f"part-{i:02d}",
                     "sha256": hashlib.sha256(payload).hexdigest(),
                     "bytes": len(payload)} for i, payload in enumerate(pieces)])


def assess_1b(capacity: CapacityEvidence, limits: AdmissionLimits,
              adapter: AdapterEvidence, shard_receipt: dict[str, Any],
              *, recovered_run_sha256: str | None = None) -> dict[str, Any]:
    """Observer-only go/no-go capacity packet; never actual GO/launch authority."""
    if (not isinstance(capacity, CapacityEvidence)
            or not isinstance(limits, AdmissionLimits)
            or not isinstance(adapter, AdapterEvidence)):
        raise BillionGateDenied("typed evidence required")
    # Dataclass frozen fields can be forged with object.__setattr__.
    CapacityEvidence(**asdict(capacity))
    AdmissionLimits(**asdict(limits))
    AdapterEvidence(**asdict(adapter))
    if recovered_run_sha256 is not None:
        _sha("recovered_run", recovered_run_sha256)
        if recovered_run_sha256 != capacity.run_sha256:
            raise BillionGateDenied("recovery changed scientific run identity")
    recipe = billion_recipe()
    reasons = []
    if adapter.model_sha256 != recipe["modelspec_sha256"]:
        reasons.append("modelspec_mismatch")
    if not all((adapter.licence_reviewed, adapter.optimizer_semantics_tested,
                adapter.sharded_save_load_tested, adapter.restart_same_run_tested,
                adapter.worker_loss_tested)):
        reasons.append("adapter_contract_unqualified")
    if not adapter.actual_backend_executed:
        reasons.append("real_backend_unverified")
    if not isinstance(shard_receipt, dict) or (
        shard_receipt.get("schema_version") != 1
        or shard_receipt.get("partial") is not False
        or shard_receipt.get("training_authorized") is not False
        or shard_receipt.get("launch_authorized") is not False
        or not isinstance(shard_receipt.get("shard_count"), int)
        or type(shard_receipt.get("shard_count")) is bool
        or shard_receipt.get("shard_count") <= 0
    ):
        reasons.append("checkpoint_receipt_unverified")
    else:
        _sha("shard_manifest", shard_receipt.get("manifest_sha256"))
    if capacity.paid_compute_requested:
        reasons.append("paid_compute_denied")
    if capacity.workers > limits.max_workers or capacity.nodes > limits.max_nodes:
        reasons.append("topology_limit_denied")
    available = capacity.workers * capacity.free_bytes_per_worker
    if (available < recipe["train_state_floor_bytes"]
            or available > limits.max_total_memory_bytes):
        reasons.append("memory_budget_denied")
    storage_floor = recipe["checkpoint_dual_copy_floor_bytes"]
    if (capacity.free_checkpoint_bytes < storage_floor
            or storage_floor > limits.max_checkpoint_bytes):
        reasons.append("checkpoint_space_denied")
    if (capacity.measured_interconnect_bytes_per_second
            < limits.min_interconnect_bytes_per_second):
        reasons.append("interconnect_unqualified")
    if capacity.measured_tokens_per_second < limits.min_tokens_per_second:
        reasons.append("throughput_unqualified")
    elif (capacity.target_unique_tokens / capacity.measured_tokens_per_second
          > limits.max_wallclock_seconds):
        reasons.append("wallclock_denied")
    if (capacity.measured_checkpoint_bytes_per_second <= 0 or
            storage_floor / capacity.measured_checkpoint_bytes_per_second
            > limits.max_checkpoint_seconds):
        reasons.append("checkpoint_transport_denied")
    if capacity.measured_recovery_seconds > limits.max_recovery_seconds:
        reasons.append("recovery_slo_denied")
    packet = {"schema_version": 1, "tier": "1B",
              "recipe_sha256": recipe["recipe_sha256"],
              "capacity_sha256": _digest(asdict(capacity)),
              "limits_sha256": _digest(asdict(limits)),
              "adapter_sha256": _digest(asdict(adapter)),
              "run_sha256": capacity.run_sha256,
              "status": "NO_GO" if reasons else "PROXY_SYSTEMS_PLAUSIBLE_NOT_AUTHORIZED",
              "reasons": sorted(set(reasons)),
              "evidence_class": "LOCAL_FREE_CAPACITY_AND_SHARD_FIXTURE",
              "launch_authorized": False, "training_authorized": False,
              "paid_compute_authorized": False,
              "real_backend_promotion_authorized": False}
    packet["packet_sha256"] = _digest(packet)
    return packet
