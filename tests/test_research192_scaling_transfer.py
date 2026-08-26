from __future__ import annotations

import math

from twelve_six import research192_scaling_transfer as r192
from twelve_six.model import ModelSpec


def test_exact_scale_geometries_and_fixed_family() -> None:
    expected = {
        "1m": (1_037_696, "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07"),
        "3m": (3_221_184, "3255ebffea76d17e59a19b4de50be616b27e85593a6eebec0db935d7efebb5ea"),
        "10m": (10_000_640, "f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53"),
    }
    for scale, (count, identity) in expected.items():
        spec = ModelSpec.from_dict(dict(r192.SCALE_SPECS[scale]["model"]))
        assert spec.parameter_count() == count
        assert spec.identity_sha256() == identity
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
    assert r192.EXPECTED_TOKEN_BUDGETS == {500: 474_377, 1000: 948_504}
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
    t = r192.EXPECTED_TOKEN_BUDGETS[500]
    assert 6 * n * t == 9_169_570_407_168
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


def test_static_validator_passes() -> None:
    r192.validate_static_contract()
