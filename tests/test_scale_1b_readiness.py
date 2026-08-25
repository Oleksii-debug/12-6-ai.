from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.training.s6_readiness import (
    MODEL_SHA256,
    PARAMETERS,
    S6ReadinessError,
    build_s6_readiness_report,
    load_s6_candidate,
    load_s6_launch_profile,
    validate_s6_readiness_report,
)
from twelve_six.training.scale_runtime import estimate_scale_resources

ROOT = Path(__file__).resolve().parents[1]


def test_s6_candidate_preserves_d01_identity_and_exact_resource_algebra() -> None:
    candidate = load_s6_candidate(ROOT)
    assert candidate.model.parameter_count() == PARAMETERS
    assert candidate.model.identity_sha256() == MODEL_SHA256

    estimate = estimate_scale_resources(
        candidate.model,
        sequence_length=4096,
        microbatch_size=1,
        activation_checkpointing=True,
        world_size=8,
        fsdp2_sharded=True,
    )
    assert estimate.persistent_total_bytes_per_rank == 1_998_213_120
    assert estimate.full_training_checkpoint_bytes == 11_989_278_720
    assert estimate.weight_only_checkpoint_bytes == 3_996_426_240
    assert estimate.kv_cache_bytes_per_token_per_sequence == 36_864
    assert estimate.estimated_activation_bytes_per_microbatch == 797_966_336
    assert estimate.estimated_training_flops_per_token == 10_408_771_584


def test_s6_launch_profile_is_fail_closed_before_paid_compute() -> None:
    launch = load_s6_launch_profile(ROOT)
    assert launch["status"] == "PREPARED_NOT_LAUNCHED"
    assert launch["topology"]["world_size"] == 8
    assert launch["topology"]["distributed_strategy"] == "fsdp2_full_shard"
    assert launch["checkpoint"]["strategy"] == (
        "torch_distributed_checkpoint_sharded_required"
    )
    assert launch["data_tokenizer"]["status"] == (
        "BLOCKED_REAL_SCALE_ARTIFACTS_NOT_FROZEN"
    )
    assert all(value is False for value in launch["launch_gates"].values())


def test_s6_readiness_executes_real_meta_model_and_local_training_analogue() -> None:
    report = build_s6_readiness_report(
        ROOT,
        source_sha="a" * 40,
        run_analogue=True,
    )
    assert report["candidate"]["parameters"] == PARAMETERS
    assert report["candidate"]["meta_instantiated_parameters"] == PARAMETERS
    assert report["candidate"]["all_parameters_meta"] is True
    assert report["resource_estimates"]["8"][
        "persistent_total_bytes_per_rank"
    ] == 1_998_213_120
    assert report["pilot"]["estimated_training_flops"] == 43_657_552_289_857_536
    assert report["launch"]["ready"] is False
    assert len(report["launch"]["blockers"]) == 5
    assert report["local_execution_analogue"]["status"] == (
        "PASS_LOCAL_FREE_ANALOGUE_ONLY"
    )
    assert report["local_execution_analogue"]["optimizer_steps"] == 1
    assert report["local_execution_analogue"]["changed_embedding_elements"] > 0


def test_s6_readiness_rejects_rehashed_launch_overclaim() -> None:
    report = build_s6_readiness_report(
        ROOT,
        source_sha="b" * 40,
        run_analogue=False,
    )
    tampered = copy.deepcopy(report)
    tampered["launch"]["ready"] = True
    with pytest.raises(S6ReadinessError, match="cannot be ready"):
        validate_s6_readiness_report(tampered)
