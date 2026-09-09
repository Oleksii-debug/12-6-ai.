"""Project-owned qualification contract for replaceable training backends.

The project-native PyTorch trainer is the semantic reference. This module does not
adopt third-party backends and does not grant training or compute authorization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .config import TrainerConfig
from .trainer import Trainer

SCHEMA = "12-6.training-backend-qualification.v1"
REFERENCE_BACKEND = "PROJECT_NATIVE_PYTORCH"
REFERENCE_ROLE = "SEMANTIC_REFERENCE_BASELINE"
REPOSITORY = "Oleksii-debug/12-6-ai."
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_PARITY = (
    "modelspec_parity",
    "optimizer_semantics_parity",
    "causal_objective_parity",
    "valid_target_accounting_parity",
    "gradient_accumulation_parity",
    "precision_semantics_parity",
    "checkpoint_save_load_resume",
    "fresh_instance_recovery",
    "determinism_same_seed",
    "rollback_path",
)

_REQUIRED_BOUNDARY_FALSE = (
    "foreign_pretrained_weights_used",
    "teacher_logits_used",
    "evaluation_or_final_test_used_for_training",
    "paid_compute_authorized",
    "material_training_authorized",
    "backend_promotion_authorized",
    "stage_promotion_authorized",
)


@dataclass(frozen=True, slots=True)
class BackendQualificationAssessment:
    contract_valid: bool
    mechanics_reference_proven: bool
    fresh_process_recovery_proven: bool
    benchmark_complete: bool
    ready_for_candidate_parity_comparison: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_report_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical identity for a qualification report body."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _valid_terminal_authority(value: Any, *, expected_claim: str) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("claim") == expected_claim
        and value.get("repository") == REPOSITORY
        and isinstance(value.get("git_sha"), str)
        and _SHA40.fullmatch(value["git_sha"]) is not None
        and isinstance(value.get("evidence_sha256"), str)
        and _SHA256.fullmatch(value["evidence_sha256"]) is not None
        and isinstance(value.get("workflow_run_id"), int)
        and not isinstance(value.get("workflow_run_id"), bool)
        and value["workflow_run_id"] > 0
        and value.get("workflow_conclusion") == "success"
        and value.get("terminal") is True
    )


def assess_backend_qualification(report: Mapping[str, Any]) -> BackendQualificationAssessment:
    """Fail closed unless a report preserves the project-owned training semantics."""
    blockers: list[str] = []
    authority_git_shas: set[str] = set()
    if report.get("schema") != SCHEMA:
        blockers.append("schema_mismatch")
    if report.get("backend_id") != REFERENCE_BACKEND:
        blockers.append("reference_backend_id_mismatch")
    if report.get("role") != REFERENCE_ROLE:
        blockers.append("reference_backend_role_mismatch")

    version = report.get("exact_version")
    if not isinstance(version, str) or not version.strip():
        blockers.append("exact_version_missing")
    if report.get("license") != "BSD-3-Clause":
        blockers.append("license_mismatch")

    parity = report.get("parity")
    if not isinstance(parity, Mapping):
        blockers.append("parity_missing")
        parity = {}
    parity_authorities = report.get("parity_authorities")
    if not isinstance(parity_authorities, Mapping):
        blockers.append("parity_authorities_missing")
        parity_authorities = {}
    for key in _REQUIRED_PARITY:
        if parity.get(key) is not True:
            blockers.append(f"{key}_not_proven")
        authority = parity_authorities.get(key)
        if not _valid_terminal_authority(authority, expected_claim=key):
            blockers.append(f"{key}_authority_missing")
        else:
            authority_git_shas.add(authority["git_sha"])

    runtime = report.get("runtime_probe")
    if not isinstance(runtime, Mapping):
        blockers.append("runtime_probe_missing")
        runtime = {}
    for key in (
        "optimizer_steps",
        "optimized_targets",
        "checkpoint_state_bytes",
        "elapsed_seconds",
    ):
        if not _positive_finite(runtime.get(key)):
            blockers.append(f"runtime_{key}_missing")
    if runtime.get("resume_equivalent") is not True:
        blockers.append("runtime_resume_equivalence_not_proven")
    if runtime.get("finite_loss") is not True:
        blockers.append("runtime_finite_loss_not_proven")
    runtime_scope_valid = (
        runtime.get("backend_id") == REFERENCE_BACKEND
        and runtime.get("exact_version") == version
        and runtime.get("device") == "cpu"
        and runtime.get("fresh_instance_only") is True
        and runtime.get("model_is_test_fixture_only") is True
    )
    if not runtime_scope_valid:
        blockers.append("runtime_probe_scope_mismatch")

    benchmark = report.get("bounded_benchmark")
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    method = benchmark.get("measurement_method")
    benchmark_authority = benchmark.get("authority")
    benchmark_authority_valid = _valid_terminal_authority(
        benchmark_authority, expected_claim="bounded_benchmark"
    )
    if benchmark_authority_valid:
        authority_git_shas.add(benchmark_authority["git_sha"])
    benchmark_scope_valid = (
        benchmark.get("backend_id") == REFERENCE_BACKEND
        and benchmark.get("exact_version") == version
        and benchmark.get("device") == "cpu"
        and benchmark.get("model_is_test_fixture_only") is True
        and benchmark.get("model_scale_throughput_claimed") is False
    )
    benchmark_complete = (
        isinstance(method, str)
        and bool(method.strip())
        and all(
            _positive_finite(benchmark.get(key))
            for key in ("tokens_per_second", "step_time_seconds", "peak_ram_bytes")
        )
        and benchmark_authority_valid
        and benchmark_scope_valid
    )
    if not benchmark_complete:
        blockers.append("bounded_throughput_memory_benchmark_missing")
    if not benchmark_scope_valid:
        blockers.append("bounded_benchmark_scope_mismatch")

    fresh_process = report.get("fresh_process_recovery")
    fresh_process_authority = (
        fresh_process.get("authority") if isinstance(fresh_process, Mapping) else None
    )
    fresh_process_authority_valid = (
        isinstance(fresh_process, Mapping)
        and fresh_process.get("proven") is True
        and _valid_terminal_authority(
            fresh_process_authority, expected_claim="fresh_process_recovery"
        )
    )
    if fresh_process_authority_valid:
        authority_git_shas.add(fresh_process_authority["git_sha"])
    fresh_process_scope_valid = (
        isinstance(fresh_process, Mapping)
        and fresh_process.get("backend_id") == REFERENCE_BACKEND
        and fresh_process.get("exact_version") == version
        and fresh_process.get("device") == "cpu"
        and fresh_process.get("fresh_process_only") is True
        and fresh_process.get("model_is_test_fixture_only") is True
    )
    fresh_process_proven = fresh_process_authority_valid and fresh_process_scope_valid
    if not fresh_process_authority_valid:
        blockers.append("fresh_process_recovery_authority_missing")
    if not fresh_process_scope_valid:
        blockers.append("fresh_process_recovery_scope_mismatch")

    if len(authority_git_shas) > 1:
        blockers.append("authority_git_sha_cohort_mismatch")

    boundaries = report.get("truth_boundary")
    if not isinstance(boundaries, Mapping):
        blockers.append("truth_boundary_missing")
        boundaries = {}
    for key in _REQUIRED_BOUNDARY_FALSE:
        if boundaries.get(key) is not False:
            blockers.append(f"truth_boundary_{key}_must_be_false")
    if boundaries.get("canonical_base_random_init_only") is not True:
        blockers.append("truth_boundary_random_init_must_be_true")

    claimed_identity = report.get("report_sha256")
    body = dict(report)
    body.pop("report_sha256", None)
    if claimed_identity != stable_report_sha256(body):
        blockers.append("report_identity_mismatch")

    mechanics_blockers = {
        item
        for item in blockers
        if item
        not in {
            "bounded_throughput_memory_benchmark_missing",
            "bounded_benchmark_scope_mismatch",
            "fresh_process_recovery_authority_missing",
            "fresh_process_recovery_scope_mismatch",
        }
    }
    mechanics_reference_proven = not mechanics_blockers
    ready = mechanics_reference_proven and fresh_process_proven and benchmark_complete
    return BackendQualificationAssessment(
        contract_valid=not blockers,
        mechanics_reference_proven=mechanics_reference_proven,
        fresh_process_recovery_proven=fresh_process_proven,
        benchmark_complete=benchmark_complete,
        ready_for_candidate_parity_comparison=ready,
        blockers=tuple(sorted(set(blockers))),
    )


class _TinyCausalModel(nn.Module):
    """Test-only mechanics model; never a canonical Base or capability artifact."""

    def __init__(self, vocab_size: int = 16, width: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.head(self.embedding(input_ids))


def _tiny_config(*, seed: int, max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        precision="fp32",
        seed=seed,
    )


def _tiny_batch() -> dict[str, Tensor]:
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)}


def run_project_native_resume_probe(*, seed: int = 126) -> dict[str, Any]:
    """Execute a bounded CPU fresh-instance resume equivalence probe.

    This deliberately proves only in-process fresh-instance mechanics. A separate D05
    authority is still required before the backend contract may claim fresh-process
    recovery.
    """
    started = time.perf_counter()
    torch.manual_seed(seed)
    initial = _TinyCausalModel()
    initial_state = copy.deepcopy(initial.state_dict())
    config = _tiny_config(seed=seed, max_steps=2)
    batch = _tiny_batch()

    uninterrupted = _TinyCausalModel()
    uninterrupted.load_state_dict(initial_state)
    uninterrupted_trainer = Trainer(uninterrupted, config, device="cpu")
    metric_a = uninterrupted_trainer.train_microbatch(batch)
    metric_b = uninterrupted_trainer.train_microbatch(batch)

    resumed = _TinyCausalModel()
    resumed.load_state_dict(initial_state)
    resumed_trainer = Trainer(resumed, config, device="cpu")
    first_metric = resumed_trainer.train_microbatch(batch)
    model_checkpoint = copy.deepcopy(resumed.state_dict())
    trainer_checkpoint = resumed_trainer.state_dict()

    restored = _TinyCausalModel()
    restored.load_state_dict(model_checkpoint)
    fresh_trainer = Trainer(restored, config, device="cpu")
    fresh_trainer.load_state_dict(trainer_checkpoint)
    second_metric = fresh_trainer.train_microbatch(batch)

    equivalent = all(
        torch.equal(left, right)
        for left, right in zip(uninterrupted.parameters(), restored.parameters(), strict=True)
    )
    checkpoint_payload = json.dumps(
        {
            "micro_step": trainer_checkpoint.micro_step,
            "optimizer_step": trainer_checkpoint.optimizer_step,
            "tokens_seen": trainer_checkpoint.tokens_seen,
            "config": trainer_checkpoint.config,
        },
        sort_keys=True,
    ).encode("utf-8")
    losses = (metric_a.loss, metric_b.loss, first_metric.loss, second_metric.loss)
    return {
        "backend_id": REFERENCE_BACKEND,
        "exact_version": torch.__version__,
        "device": "cpu",
        "optimizer_steps": fresh_trainer.optimizer_step,
        "optimized_targets": fresh_trainer.tokens_seen,
        "checkpoint_state_bytes": len(checkpoint_payload),
        "elapsed_seconds": time.perf_counter() - started,
        "finite_loss": all(math.isfinite(value) for value in losses),
        "resume_equivalent": equivalent,
        "fresh_instance_only": True,
        "model_is_test_fixture_only": True,
    }


def run_project_native_benchmark(*, steps: int = 8, seed: int = 126) -> dict[str, Any]:
    """Measure a bounded CPU Trainer workload without claiming model-scale throughput."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    try:
        import resource
    except ImportError:
        return {
            "backend_id": REFERENCE_BACKEND,
            "exact_version": torch.__version__,
            "device": "cpu",
            "steps": steps,
            "measurement_method": "PROCESS_RSS_UNAVAILABLE",
            "tokens_per_second": None,
            "step_time_seconds": None,
            "peak_ram_bytes": None,
            "model_is_test_fixture_only": True,
            "model_scale_throughput_claimed": False,
        }

    torch.manual_seed(seed)
    model = _TinyCausalModel()
    trainer = Trainer(model, _tiny_config(seed=seed, max_steps=steps), device="cpu")
    batch = _tiny_batch()
    before_targets = trainer.tokens_seen
    started = time.perf_counter()
    for _ in range(steps):
        trainer.train_microbatch(batch)
    elapsed = time.perf_counter() - started
    optimized_targets = trainer.tokens_seen - before_targets
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_ram_bytes = int(max_rss if sys.platform == "darwin" else max_rss * 1024)

    return {
        "backend_id": REFERENCE_BACKEND,
        "exact_version": torch.__version__,
        "device": "cpu",
        "steps": steps,
        "optimized_targets": optimized_targets,
        "measurement_method": "PROCESS_MAX_RSS_RUSAGE",
        "tokens_per_second": optimized_targets / elapsed,
        "step_time_seconds": elapsed / steps,
        "peak_ram_bytes": peak_ram_bytes,
        "model_is_test_fixture_only": True,
        "model_scale_throughput_claimed": False,
    }
