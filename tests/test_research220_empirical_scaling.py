from __future__ import annotations

import math

from twelve_six import research192_scaling_transfer as r192
from twelve_six import research220_empirical_scaling as r220


def _curve() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(1, r192.FINAL_STEP + 1):
        tokens = step
        if step in r192.EXPECTED_TOKEN_BUDGETS:
            tokens = r192.EXPECTED_TOKEN_BUDGETS[step]
        rows.append({
            "optimizer_step": step,
            "optimized_tokens": tokens,
            "gradient_norm_pre_clip": 2.0 if step % 2 == 0 else 0.5,
        })
    return rows


def test_research220_keeps_exact_preregistered_five_arm_matrix() -> None:
    assert r192.ARM_MATRIX == (
        ("1m", 1337),
        ("1m", 1338),
        ("3m", 1337),
        ("3m", 1338),
        ("10m", 1337),
    )
    assert r192.CHECKPOINT_STEPS == (18, 70, 139)
    assert r192.EXPECTED_TOKEN_BUDGETS == {18: 17_125, 70: 66_417, 139: 131_938}


def test_curve_telemetry_reports_boundary_grad_and_clip_rate_without_new_updates() -> None:
    result = r220.curve_telemetry(_curve())
    assert set(result) == {"18", "70", "139"}
    assert result["18"]["optimized_tokens"] == 17_125
    assert result["70"]["optimized_tokens"] == 66_417
    assert result["139"]["optimized_tokens"] == 131_938
    assert result["18"]["checkpoint_gradient_norm_pre_clip"] == 2.0
    assert result["18"]["cumulative"]["samples"] == 18
    assert math.isclose(result["18"]["cumulative"]["clip_rate"], 0.5)
    assert result["70"]["since_prior_boundary"]["samples"] == 52
    assert result["139"]["since_prior_boundary"]["samples"] == 69


def test_research220_contract_and_m150_control_are_exact() -> None:
    assert r220.RESEARCH212_CONTRACT_IDENTITY == (
        "458cbb22e43bb405029fc256f4d9f29f3ab6b81bcab0db69c9b8cde5d6d5798a"
    )
    assert r220.M150_ONE_M_RANDOM_INIT == (
        "630671c032f4a000a98bc3bf74422e04ed2d6badba32e31b049349d6be9b99f2"
    )
    assert r220.CLIP_THRESHOLD == 1.0
    assert r192.SCALE_SPECS["10m"]["model"]["n_heads"] == r192.SCALE_SPECS["10m"]["model"]["n_kv_heads"]
    assert r192.SCALE_SPECS["10m"]["model"]["max_seq_len"] == 256
    assert r192.trainer_config(1337).weight_decay == 0.0
