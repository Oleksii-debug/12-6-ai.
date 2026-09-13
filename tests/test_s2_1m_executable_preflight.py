from pathlib import Path

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.s2_preflight import (
    CANDIDATE_PATH,
    FIXTURE_SCOPE,
    MODEL_SPEC_SHA256,
    PARAMETER_COUNT,
    collect_s2_1m_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def test_s2_byte_gqa_geometry_is_exact_and_instantiable() -> None:
    stage = load_stage_config(ROOT / CANDIDATE_PATH)
    model = TwelveSixDecoder(stage.model, stage.init)

    assert stage.stage == "S2"
    assert stage.model.identity_sha256() == MODEL_SPEC_SHA256
    assert stage.expected_parameters == PARAMETER_COUNT == 992_896
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert stage.model.vocab_size == 256
    assert stage.model.n_heads == 4
    assert stage.model.n_kv_heads == 2
    assert stage.model.d_ff == 288
    assert stage.model.parameter_breakdown()["token_embedding"] == 32_768
    assert stage.model.parameter_breakdown()["blocks_total"] == 960_000


def test_s2_real_trainer_checkpoint_resume_and_first_party_inference(tmp_path: Path) -> None:
    evidence = collect_s2_1m_preflight(
        ROOT,
        "a" * 40,
        tmp_path / "s2-preflight",
        total_steps=2,
        split_step=1,
        sequence_length=32,
        verify_checkout=False,
    )

    assert evidence["model"]["parameter_count"] == PARAMETER_COUNT
    assert evidence["fixture"]["scope"] == FIXTURE_SCOPE
    assert evidence["training"]["optimizer_steps"] == 2
    assert evidence["training"]["gradient_norm_min"] > 0.0
    assert evidence["training"]["weight_delta"]["changed_parameter_elements"] > 0
    assert evidence["context_probe"] == {
        "sequence_length": 512,
        "logits_shape": [1, 512, 256],
        "finite": True,
    }
    assert evidence["checkpoint"]["resume_model_state_exact"] is True
    assert evidence["checkpoint"]["resume_trainer_state_exact"] is True
    assert evidence["checkpoint"]["directory_bytes"] > 0
    assert evidence["tensor_state_footprint"]["model_parameter_bytes"] == PARAMETER_COUNT * 4
    assert evidence["first_party_inference"]["backend"] == "first_party_torch"
    assert evidence["first_party_inference"]["parameter_count"] == PARAMETER_COUNT
    assert len(evidence["first_party_inference"]["generated_token_ids"]) == 2
    assert evidence["blockers"]["meaningful_s2_training_experiment_ready"] is False
    assert evidence["claims"]["s2_corpus_or_tokenizer_frozen"] is False
