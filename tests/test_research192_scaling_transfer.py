from __future__ import annotations

import math

from twelve_six import research192_scaling_transfer as r192
from twelve_six import research212_contract_recovery as r212
from twelve_six.model import ModelSpec


def test_exact_scale_geometries_and_fixed_family() -> None:
    expected = {
        "1m": (1_037_696, "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07", 352),
        "3m": (3_213_120, "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc", 528),
        "10m": (10_000_640, "f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53", 736),
    }
    for scale, (count, identity, d_ff) in expected.items():
        spec = ModelSpec.from_dict(dict(r192.SCALE_SPECS[scale]["model"]))
        assert spec.parameter_count() == count
        assert spec.identity_sha256() == identity
        assert spec.d_ff == d_ff
        assert spec.vocab_size == 256
        assert spec.max_seq_len == 256
        assert spec.n_heads == spec.n_kv_heads
        assert spec.head_dim == 16
        assert spec.rope_rotary_dim == 16
        assert spec.attention_dropout == 0.0
        assert spec.tie_word_embeddings is True


def test_frozen_non_size_recipe_and_exact_token_budgets() -> None:
    assert r192.EXPECTED_CORPUS_ID == (
        "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
    )
    assert r192.EXPECTED_EVALUATION_ID == (
        "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
    )
    assert r192.CHECKPOINT_STEPS == (18, 70, 139)
    assert r192.EXPECTED_TOKEN_BUDGETS == {18: 17_125, 70: 66_417, 139: 131_938}
    assert r192.MIDPOINT_STEP == 70
    assert r192.FINAL_STEP == 139
    assert r192.PAIRED_SEEDS == (1337, 1338)
    assert r192.ARM_MATRIX == (
        ("1m", 1337),
        ("1m", 1338),
        ("3m", 1337),
        ("3m", 1338),
        ("10m", 1337),
    )


def test_compute_and_bpb_definitions_are_dimensionally_explicit() -> None:
    n = r192.SCALE_SPECS["3m"]["expected_parameters"]
    t = r192.EXPECTED_TOKEN_BUDGETS[r192.FINAL_STEP]
    assert 6 * n * t == 2_543_595_759_360
    loss_nats = math.log(2.0)
    assert loss_nats / math.log(2.0) == 1.0
    assert 4 * r192.SCALE_SPECS["10m"]["expected_parameters"] == 40_002_560


def test_incumbent_is_bound_to_terminal_m150_artifact() -> None:
    assert r192.M150_PRODUCER == {
        "source_sha": "5838cd16869dcfcf762368d8673eddf52d51b7e3",
        "workflow_run_id": 32937411703,
        "artifact_id": 9595677772,
        "artifact_name": "milestone150-learned-base-ladder-v1",
        "artifact_sha256": "c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71",
        "ladder_report_sha256": "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a",
        "one_m_report_identity_sha256": "1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738",
    }
    assert r212.FROZEN_CONTRACT["m150_producer"] == r192.M150_PRODUCER


def test_learn191_bridge_authority_is_exact() -> None:
    assert r192.LEARN191_GEOMETRY == {
        "pr": 348,
        "source_sha": "a75920cef8bde37a8c590e34095be83c97b75f1d",
        "model_spec_sha256": "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc",
        "parameters": 3_213_120,
        "nominal_targets": [16_632, 65_772, 131_292],
        "role": "geometry/budget preregistration authority; checkpoint reuse requires terminal artifact",
    }


def test_current_frozen_contract_resolves_without_drift() -> None:
    result = r212.diagnose_contract(r212.FROZEN_CONTRACT)
    assert result["status"] == "PASS"
    assert result["reason_count"] == 0
    assert result["reasons"] == []
    r212.require_frozen_contract()
    r192.validate_static_contract()


def test_historical_run_32941405721_mismatch_is_human_readable_and_machine_coded() -> None:
    result = r212.diagnose_contract(r212.HISTORICAL_RUN_32941405721_CONTRACT)
    assert result["status"] == "FAIL"
    codes = {reason["code"] for reason in result["reasons"]}
    assert "SCALE_PARAMETER_COUNT_MISMATCH" in codes
    assert "SCALE_MODEL_SPEC_IDENTITY_MISMATCH" in codes
    assert "SCALE_GEOMETRY_MISMATCH" in codes
    assert "CHECKPOINT_STEPS_MISMATCH" in codes
    assert "OPTIMIZED_TOKEN_BUDGET_MISMATCH" in codes
    messages = "\n".join(reason["message"] for reason in result["reasons"])
    assert "scales.3m.parameters" in messages
    assert "3221184" in messages
    assert "3213120" in messages
    assert "checkpoint_steps" in messages
    assert "optimized_token_budgets" in messages
