from pathlib import Path

import pytest

from twelve_six.training.s6_readiness import (
    S6_CURRENT_TOKENIZER_EXPECTED_PARAMETERS,
    S6_CURRENT_TOKENIZER_MODEL_SHA256,
    build_s6_readiness_report,
    validate_s6_candidate,
)

CONFIG = Path("configs/stages/s6_1b.scale06_current_tokenizer.candidate.json")


def test_s6_candidate_exact_identity_and_meta_construction() -> None:
    config = validate_s6_candidate(CONFIG)
    assert config.expected_parameters == S6_CURRENT_TOKENIZER_EXPECTED_PARAMETERS
    assert config.model.parameter_count() == 999_761_920
    assert config.model.identity_sha256() == S6_CURRENT_TOKENIZER_MODEL_SHA256
    assert config.model.vocab_size == 256
    assert config.model.max_seq_len == 4096
    assert (config.model.n_heads, config.model.n_kv_heads) == (32, 8)

    report = build_s6_readiness_report(CONFIG)
    assert report.meta_parameter_count == 999_761_920
    assert report.exact_parameters == 999_761_920
    assert report.relative_target_error == pytest.approx(-0.00023808)
    assert report.world_size == 4
    assert report.persistent_total_bytes_per_rank > 0
    assert report.full_training_checkpoint_bytes == 999_761_920 * 12
    assert report.weight_only_checkpoint_bytes == 999_761_920 * 4
    assert report.kv_cache_bytes_per_token_per_sequence == 36_864
    assert report.estimated_activation_bytes_per_microbatch > 0
    assert report.estimated_training_flops_per_token > 0


def test_s6_launch_gate_stays_fail_closed_without_authorization() -> None:
    report = build_s6_readiness_report(CONFIG, world_size=1, sequence_length=1024)
    assert "COMPUTE_AUTHORIZED_ABSENT" in report.launch_blockers
    assert "PRODUCTION_TOKENIZER_NOT_FROZEN" in report.launch_blockers
    assert "REPRESENTATIVE_CORPUS_NOT_FROZEN" in report.launch_blockers
    assert "DCP_FSDP2_CHECKPOINT_RESUME_NOT_COMPOSED_ON_S6" in report.launch_blockers
    assert report.authority == "ENGINEERING_READINESS_ONLY_NOT_COMPUTE_AUTHORIZATION"


def test_s6_paid_flag_does_not_erase_independent_technical_gates() -> None:
    report = build_s6_readiness_report(CONFIG, compute_authorized=True)
    assert "COMPUTE_AUTHORIZED_ABSENT" not in report.launch_blockers
    assert "TARGET_GPU_NCCL_NOT_MEASURED" in report.launch_blockers
    assert "NATIVE_GQA_TARGET_GPU_PARITY_NOT_MEASURED" in report.launch_blockers
    assert "HELD_OUT_EVALUATION_NOT_BOUND_TO_S6_RUN" in report.launch_blockers
