from __future__ import annotations

import json
from pathlib import Path

from twelve_six.s4_readiness import (
    S4RunProfile,
    accelerator_preflight,
    estimate_s4_resources,
    meta_parameter_probe,
    run_scaled_analogue,
    validate_s4_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
S4_CONFIG = ROOT / "configs" / "stages" / "s4_100m_accelerator.candidate.json"
PILOT_RUN = ROOT / "configs" / "runs" / "s4_100m_pilot.json"
SERIOUS_RUN = ROOT / "configs" / "runs" / "s4_100m_serious.json"


def test_s4_candidate_binds_current_tokenizer_and_exact_count() -> None:
    config = validate_s4_candidate(S4_CONFIG)
    assert config.expected_parameters == 99_897_600
    assert config.model.parameter_count() == 99_897_600
    assert config.model.identity_sha256() == (
        "6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170"
    )
    assert config.model.vocab_size == 256
    assert config.model.max_seq_len == 4096
    assert config.model.n_heads == config.model.n_kv_heads == 12
    assert config.model.d_ff / config.model.d_model == 3.0


def test_full_s4_constructs_on_meta_without_weight_allocation() -> None:
    assert meta_parameter_probe(S4_CONFIG) == 99_897_600


def test_serious_profile_fits_single_gpu_first_order_budget() -> None:
    evidence = estimate_s4_resources(
        S4_CONFIG,
        S4RunProfile(
            name="serious",
            sequence_length=4096,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            max_steps=15_259,
        ),
    )
    assert 3.0 < evidence.total_training_gib_estimate < 5.0
    assert 1.0 < evidence.checkpoint_payload_gib_estimate < 1.2
    assert 2.0 < evidence.checkpoint_load_transient_host_gib_estimate < 2.4
    assert evidence.parameter_gib_fp32 > evidence.inference_weight_gib_bf16_if_cast
    assert evidence.tokens_per_optimizer_step == 131_072
    assert evidence.scheduled_tokens == 2_000_027_648


def test_run_profiles_fail_closed_on_spend_and_serious_data_readiness() -> None:
    pilot = json.loads(PILOT_RUN.read_text(encoding="utf-8"))
    serious = json.loads(SERIOUS_RUN.read_text(encoding="utf-8"))
    assert pilot["compute_authorized"] is False
    assert serious["compute_authorized"] is False
    assert pilot["data_quality_claim_allowed"] is False
    assert serious["data_quality_claim_allowed"] is False
    assert serious["launch_state"] == "blocked_pending_d03_scaled_corpus"
    assert serious["required_training_byte_tokens"] == serious["scheduled_tokens"]
    assert len(serious["launch_blockers"]) >= 3


def test_scaled_analogue_executes_real_forward_backward_update() -> None:
    result = run_scaled_analogue(sequence_length=16, batch_size=1)
    assert result.parameters > 100_000
    assert result.optimizer_step == 1
    assert result.tokens == 15
    assert result.loss > 0.0
    assert result.wall_seconds > 0.0


def test_accelerator_preflight_is_nonallocating_and_typed() -> None:
    result = accelerator_preflight()
    assert isinstance(result["cuda_available"], bool)
    assert isinstance(result["bf16_supported"], bool)
    assert isinstance(result["torch_version"], str)
