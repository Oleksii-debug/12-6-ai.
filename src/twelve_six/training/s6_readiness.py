"""S6 ~1B launch-readiness contract built on the SCALE-05 runtime seam."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import AdamW

from twelve_six.model import InitSpec, ModelSpec, StageConfig, load_stage_config

from .config import TrainerConfig
from .scale_runtime import (
    ActivationCheckpointedDecoder,
    ExternallyPlacedTrainer,
    build_meta_decoder,
    estimate_scale_resources,
)

SCHEMA_VERSION = "12-6.s6-scale06-readiness.v1"
AUTHORITY = "ENGINEERING_LAUNCH_READINESS_ONLY_NOT_STAGE_EVIDENCE"
CANDIDATE_PATH = "configs/stages/s6_1b.scale06_candidate.json"
LAUNCH_PATH = "configs/runs/s6_1b.scale06_launch.json"
MODEL_SHA256 = "cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261"
INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
PARAMETERS = 999_106_560
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class S6ReadinessError(ValueError):
    """Raised when S6 launch-readiness evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S6ReadinessError(message)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _small_analogue_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=32,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=160,
        rope_rotary_dim=16,
    )


def load_s6_candidate(root: str | Path) -> StageConfig:
    root = Path(root).resolve()
    path = root / CANDIDATE_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(raw.get("status") == "engineering_candidate_not_frozen", "candidate status drift")
    _require(raw.get("promotion_allowed") is False, "S6 candidate cannot promote")
    _require(
        raw.get("requires_preceding_stage_pass") is True,
        "preceding-stage gate missing",
    )
    scale06 = raw.get("scale06_readiness")
    _require(isinstance(scale06, dict), "SCALE-06 candidate metadata missing")
    _require(scale06.get("compute_authorized") is False, "candidate compute overclaim")
    _require(
        scale06.get("tokenizer_vocab_is_geometry_budget_only") is True,
        "candidate tokenizer authority overclaim",
    )
    _require(
        scale06.get("real_scale_data_tokenizer_frozen") is False,
        "candidate data/tokenizer freeze overclaim",
    )
    _require(
        scale06.get("dcp_sharded_checkpoint_integration_required") is True,
        "candidate DCP gate missing",
    )

    candidate = load_stage_config(path)
    _require(candidate.stage == "S6", "candidate stage must be S6")
    _require(candidate.canonical_base == "random_init", "canonical Base drift")
    _require(candidate.expected_parameters == PARAMETERS, "S6 parameter count drift")
    _require(candidate.model.parameter_count() == PARAMETERS, "ModelSpec count drift")
    _require(candidate.model.identity_sha256() == MODEL_SHA256, "S6 ModelSpec drift")
    _require(candidate.init.identity_sha256() == INIT_SHA256, "S6 InitSpec drift")
    return candidate


def load_s6_launch_profile(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    payload = json.loads((root / LAUNCH_PATH).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "launch profile must be an object")
    _require(
        payload.get("schema_version") == "12-6.s6-scale06-launch.v1",
        "launch schema drift",
    )
    _require(
        payload.get("status") == "PREPARED_NOT_LAUNCHED",
        "launch status overclaim",
    )
    _require(payload.get("canonical_base") == "random_init", "launch Base drift")
    _require(payload.get("candidate_path") == CANDIDATE_PATH, "candidate path drift")
    _require(
        payload.get("candidate_model_identity_sha256") == MODEL_SHA256,
        "launch ModelSpec drift",
    )
    _require(payload.get("expected_parameters") == PARAMETERS, "launch parameter drift")

    topology = payload.get("topology")
    _require(isinstance(topology, dict), "topology missing")
    _require(topology.get("nodes") == 1, "S6 v1 launch must be single-node")
    _require(topology.get("gpus_per_node") == 8, "S6 v1 launch requires 8 GPUs")
    _require(topology.get("world_size") == 8, "S6 v1 world size drift")
    _require(
        topology.get("minimum_cuda_memory_gib_per_gpu") == 24,
        "minimum GPU memory drift",
    )
    _require(
        topology.get("distributed_strategy") == "fsdp2_full_shard",
        "distributed strategy drift",
    )
    _require(
        topology.get("tensor_parallel_degree") == 1,
        "TP is not composed in SCALE-06 v1",
    )

    training = payload.get("training")
    _require(isinstance(training, dict), "training profile missing")
    _require(
        training.get("precision") == "bf16_autocast_fp32_persistent_state",
        "precision drift",
    )
    _require(training.get("sequence_length") == 4096, "sequence length drift")
    _require(training.get("microbatch_size") == 1, "microbatch drift")
    _require(
        training.get("gradient_accumulation_steps") == 16,
        "accumulation drift",
    )
    _require(training.get("optimizer_steps_pilot") == 64, "pilot step count drift")
    _require(
        training.get("optimized_tokens_per_step") == 65_536,
        "token/update drift",
    )
    _require(
        training.get("optimized_tokens_pilot") == 4_194_304,
        "pilot token count drift",
    )
    _require(
        training.get("activation_checkpointing") == "blockwise_non_reentrant",
        "checkpointing drift",
    )
    _require(
        training.get("attention_policy") == "flash_required",
        "attention policy drift",
    )

    checkpoint = payload.get("checkpoint")
    _require(isinstance(checkpoint, dict), "checkpoint launch gate missing")
    _require(
        checkpoint.get("strategy")
        == "torch_distributed_checkpoint_sharded_required",
        "S6 requires sharded DCP",
    )
    _require(
        checkpoint.get("single_process_checkpoint_v1_allowed_for_paid_fsdp2_launch")
        is False,
        "single-process checkpoint must remain blocked for paid FSDP2",
    )

    data = payload.get("data_tokenizer")
    _require(isinstance(data, dict), "data/tokenizer gate missing")
    _require(
        data.get("status") == "BLOCKED_REAL_SCALE_ARTIFACTS_NOT_FROZEN",
        "data gate overclaim",
    )
    _require(data.get("vocab_size_geometry_budget") == 32_768, "vocab budget drift")
    _require(
        data.get("s0_byte_fixture_allowed_for_capability_training") is False,
        "S0 fixture overclaim",
    )
    _require(
        data.get("versioned_tokenizer_artifact_required") is True,
        "tokenizer artifact gate missing",
    )
    _require(
        data.get("versioned_corpus_manifest_required") is True,
        "corpus manifest gate missing",
    )
    _require(
        data.get("held_out_zero_optimization_split_required") is True,
        "held-out gate missing",
    )

    gates = payload.get("launch_gates")
    _require(isinstance(gates, dict), "launch gates missing")
    required_false = {
        "preceding_s5_stage_pass",
        "real_scale_data_tokenizer_ready",
        "dcp_sharded_checkpoint_validated",
        "cuda_nccl_fsdp2_smoke_validated",
        "compute_authorized",
    }
    _require(set(gates) == required_false, "launch gate set drift")
    _require(
        all(gates[name] is False for name in required_false),
        "S6 launch must remain blocked",
    )
    return payload


def run_local_scale_analogue(seed: int = 20260825) -> dict[str, Any]:
    """Execute inherited placement/checkpointing trainer semantics at bounded scale."""
    torch.manual_seed(seed)
    spec = _small_analogue_spec()
    model = ActivationCheckpointedDecoder(spec, InitSpec())
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=1,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas)
    trainer = ExternallyPlacedTrainer(
        model,
        config,
        device="cpu",
        optimizer=optimizer,
        attention_policy="math",
    )
    before = model.token_embedding.weight.detach().clone()
    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=torch.long,
        )
    }
    first = trainer.train_microbatch(batch)
    second = trainer.train_microbatch(batch)
    trainer.assert_checkpoint_safe()
    delta = model.token_embedding.weight.detach() - before
    grad_norm = second.grad_norm
    _require(first.optimizer_stepped is False, "analogue accumulation boundary failed")
    _require(second.optimizer_stepped is True, "analogue optimizer step missing")
    _require(trainer.optimizer_step == 1, "analogue optimizer step count drift")
    _require(
        grad_norm is not None and math.isfinite(float(grad_norm)),
        "analogue gradient non-finite",
    )
    _require(math.isfinite(float(second.loss)), "analogue loss non-finite")
    _require(int(delta.ne(0).sum().item()) > 0, "analogue weights did not change")
    return {
        "status": "PASS_LOCAL_FREE_ANALOGUE_ONLY",
        "parameters": spec.parameter_count(),
        "optimizer_steps": trainer.optimizer_step,
        "micro_steps": trainer.micro_step,
        "tokens_seen": trainer.tokens_seen,
        "loss": float(second.loss),
        "grad_norm": float(grad_norm),
        "changed_embedding_elements": int(delta.ne(0).sum().item()),
        "activation_checkpointing": True,
        "externally_placed_trainer": True,
        "device": "cpu",
        "precision": "fp32",
    }


def build_s6_readiness_report(
    root: str | Path,
    *,
    source_sha: str,
    run_analogue: bool = True,
) -> dict[str, Any]:
    """Build allocation-safe S6 evidence while preserving every launch blocker."""
    _require(
        _GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase Git SHA",
    )
    root = Path(root).resolve()
    candidate = load_s6_candidate(root)
    launch = load_s6_launch_profile(root)

    meta = build_meta_decoder(
        candidate.model,
        candidate.init,
        activation_checkpointing=True,
    )
    meta_count = sum(parameter.numel() for parameter in meta.parameters())
    _require(meta_count == PARAMETERS, "meta-instantiated S6 parameter count drift")
    _require(
        all(parameter.device.type == "meta" for parameter in meta.parameters()),
        "S6 meta allocation escaped",
    )

    estimates: dict[str, dict[str, Any]] = {}
    for world_size in (1, 4, 8):
        estimate = estimate_scale_resources(
            candidate.model,
            sequence_length=4096,
            microbatch_size=1,
            activation_checkpointing=True,
            world_size=world_size,
            fsdp2_sharded=world_size > 1,
        )
        estimates[str(world_size)] = asdict(estimate)

    gates = launch["launch_gates"]
    blockers = [name for name, passed in gates.items() if passed is False]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "candidate": {
            "stage": candidate.stage,
            "canonical_base": candidate.canonical_base,
            "parameters": candidate.model.parameter_count(),
            "meta_instantiated_parameters": meta_count,
            "all_parameters_meta": True,
            "model_identity_sha256": candidate.model.identity_sha256(),
            "init_identity_sha256": candidate.init.identity_sha256(),
            "vocab_size_geometry_budget": candidate.model.vocab_size,
            "max_seq_len": candidate.model.max_seq_len,
            "promotion_allowed": False,
            "architecture_frozen": False,
            "tokenizer_frozen": False,
        },
        "resource_estimates": estimates,
        "pilot": {
            "optimized_tokens": launch["training"]["optimized_tokens_pilot"],
            "estimated_training_flops": (
                estimates["8"]["estimated_training_flops_per_token"]
                * launch["training"]["optimized_tokens_pilot"]
            ),
            "topology": launch["topology"],
            "precision": launch["training"]["precision"],
            "status": "PREPARED_NOT_LAUNCHED",
        },
        "launch": {
            "ready": False,
            "blockers": sorted(blockers),
            "checkpoint_strategy": launch["checkpoint"]["strategy"],
            "data_tokenizer_status": launch["data_tokenizer"]["status"],
            "compute_authorized": False,
            "paid_compute_used": False,
            "cuda_execution_claimed": False,
            "distributed_execution_claimed": False,
        },
        "local_execution_analogue": (
            run_local_scale_analogue() if run_analogue else {"status": "NOT_RUN"}
        ),
        "dependencies": {
            "scale_runtime": "PR_164_SCALE05",
            "tensor_parallel_optional_follow_on": "PR_151_DIST17",
            "sharded_checkpoint_owner": "D05_D12_EXISTING_OWNERSHIP",
            "compute_plan_owner": "PR_70_D13",
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_s6_readiness_report(report)
    return report


def validate_s6_readiness_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA_VERSION, "wrong readiness schema")
    _require(report.get("authority") == AUTHORITY, "wrong readiness authority")
    source_sha = report.get("source_sha")
    _require(
        isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None,
        "invalid source SHA",
    )

    candidate = report.get("candidate")
    _require(isinstance(candidate, Mapping), "candidate evidence missing")
    _require(candidate.get("canonical_base") == "random_init", "foreign Base detected")
    _require(candidate.get("parameters") == PARAMETERS, "S6 parameter evidence drift")
    _require(
        candidate.get("meta_instantiated_parameters") == PARAMETERS,
        "S6 meta count drift",
    )
    _require(
        candidate.get("all_parameters_meta") is True,
        "S6 full allocation unexpectedly materialized",
    )
    _require(
        candidate.get("model_identity_sha256") == MODEL_SHA256,
        "S6 model identity drift",
    )
    _require(
        candidate.get("init_identity_sha256") == INIT_SHA256,
        "S6 init identity drift",
    )
    for key in ("promotion_allowed", "architecture_frozen", "tokenizer_frozen"):
        _require(
            candidate.get(key) is False,
            f"prohibited candidate claim enabled: {key}",
        )

    estimates = report.get("resource_estimates")
    _require(isinstance(estimates, Mapping), "resource estimates missing")
    expected_totals = {
        "1": 15_985_704_960,
        "4": 3_996_426_240,
        "8": 1_998_213_120,
    }
    for world_size, persistent_total in expected_totals.items():
        estimate = estimates.get(world_size)
        _require(
            isinstance(estimate, Mapping),
            f"world-size {world_size} estimate missing",
        )
        _require(estimate.get("parameters") == PARAMETERS, "resource parameter drift")
        _require(
            estimate.get("persistent_total_bytes_per_rank") == persistent_total,
            f"world-size {world_size} persistent-state drift",
        )
        _require(
            estimate.get("full_training_checkpoint_bytes") == 11_989_278_720,
            "checkpoint byte drift",
        )
        _require(
            estimate.get("weight_only_checkpoint_bytes") == 3_996_426_240,
            "weight byte drift",
        )
        _require(
            estimate.get("kv_cache_bytes_per_token_per_sequence") == 36_864,
            "KV-cache byte drift",
        )
        _require(
            estimate.get("estimated_activation_bytes_per_microbatch") == 797_966_336,
            "activation estimate drift",
        )
        _require(
            estimate.get("estimated_training_flops_per_token") == 10_408_771_584,
            "FLOP estimate drift",
        )

    launch = report.get("launch")
    _require(isinstance(launch, Mapping), "launch evidence missing")
    _require(launch.get("ready") is False, "S6 launch cannot be ready")
    blockers = launch.get("blockers")
    _require(
        isinstance(blockers, list) and len(blockers) == 5,
        "S6 blocker set incomplete",
    )
    _require(
        launch.get("compute_authorized") is False,
        "compute authorization fabricated",
    )
    _require(launch.get("paid_compute_used") is False, "paid compute fabricated")
    _require(
        launch.get("cuda_execution_claimed") is False,
        "CUDA execution fabricated",
    )
    _require(
        launch.get("distributed_execution_claimed") is False,
        "distributed execution fabricated",
    )

    analogue = report.get("local_execution_analogue")
    _require(isinstance(analogue, Mapping), "local analogue missing")
    _require(
        analogue.get("status") in {"PASS_LOCAL_FREE_ANALOGUE_ONLY", "NOT_RUN"},
        "local analogue status invalid",
    )
    if analogue.get("status") == "PASS_LOCAL_FREE_ANALOGUE_ONLY":
        _require(
            analogue.get("optimizer_steps") == 1,
            "analogue optimizer step drift",
        )
        _require(
            int(analogue.get("changed_embedding_elements", 0)) > 0,
            "analogue weight delta missing",
        )
        _require(
            math.isfinite(float(analogue.get("loss"))),
            "analogue loss non-finite",
        )
        _require(
            math.isfinite(float(analogue.get("grad_norm"))),
            "analogue grad norm non-finite",
        )

    claimed_hash = report.get("report_sha256")
    _require(
        isinstance(claimed_hash, str) and len(claimed_hash) == 64,
        "report hash missing",
    )
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    _require(_canonical_hash(unhashed) == claimed_hash, "report self-hash mismatch")
