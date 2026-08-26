from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "r01_flops_vocab_accounting.py"
CONFIG_PATH = ROOT / "configs" / "research" / "r01_flops_vocab_accounting_v1.json"
SPEC = importlib.util.spec_from_file_location("r01_flops_vocab_accounting", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _baseline():
    return MODULE.DecoderGeometry.from_mapping(_config()["baseline_geometry"])


def test_model341_exact_parameter_arithmetic() -> None:
    result = MODULE.parameter_breakdown(_baseline())
    assert result["attention_projection_per_layer"] == 245_760
    assert result["swiglu_mlp_per_layer"] == 1_036_800
    assert result["rmsnorm_per_layer"] == 640
    assert result["block_per_layer"] == 1_283_200
    assert result["all_blocks"] == 20_531_200
    assert result["token_embedding"] == 81_920
    assert result["final_rmsnorm"] == 320
    assert result["total_parameters"] == 20_613_440


def test_gqa_divisibility_fails_closed() -> None:
    config = _config()["baseline_geometry"] | {"n_kv_heads": 3}
    with pytest.raises(MODULE.AccountingError, match="divisible"):
        MODULE.DecoderGeometry.from_mapping(config)


def test_head_geometry_fails_closed() -> None:
    config = _config()["baseline_geometry"] | {"d_model": 321}
    with pytest.raises(MODULE.AccountingError, match=r"n_heads \* head_dim"):
        MODULE.DecoderGeometry.from_mapping(config)


def test_untied_embeddings_add_one_full_vocab_matrix() -> None:
    tied = _baseline()
    untied = MODULE.replace(tied, tie_word_embeddings=False)
    tied_params = MODULE.parameter_breakdown(tied)["total_parameters"]
    untied_params = MODULE.parameter_breakdown(untied)["total_parameters"]
    assert untied_params - tied_params == tied.vocab_size * tied.d_model


def test_vocab_growth_changes_params_and_output_compute_at_fixed_geometry() -> None:
    rows = MODULE.fixed_geometry_vocab_sweep(_baseline(), [256, 512])
    base, larger = rows
    assert larger["total_parameters"] - base["total_parameters"] == 256 * 320
    assert (
        larger["vocabulary_projection_forward_flops_per_token"]
        - base["vocabulary_projection_forward_flops_per_token"]
        == 2 * 320 * 256
    )


def test_fixed_total_vocab_comparison_moves_dff_instead_of_hiding_confound() -> None:
    result = MODULE.nearest_dff_for_fixed_total(
        _baseline(), target_total_parameters=20_613_440, vocab_size=512
    )
    assert result["chosen_d_ff"] < 1080
    assert abs(result["parameter_delta_from_target"]) < 16 * 3 * 320


def test_six_n_is_not_presented_as_exact_flops() -> None:
    f = MODULE.flop_breakdown(_baseline())
    assert f["attention_context_forward_flops_per_token"] > 0
    assert f["vocabulary_projection_forward_flops_per_token"] > 0
    assert (
        f["training_dominant_matmul_estimate_flops_per_token"]
        != f["six_n_training_proxy_flops_per_token"]
    )


def test_candidate_planning_geometries_land_in_declared_bands() -> None:
    report = MODULE.build_report(_config())
    candidates = report["candidate_geometries"]
    assert candidates[0]["parameters"]["total_parameters"] == 50_596_992
    assert candidates[1]["parameters"]["total_parameters"] == 99_117_568
    assert all(item["status"] == "PLANNING_ONLY_NOT_MODELSPEC" for item in candidates)


def test_claim_self_promotion_fails_closed() -> None:
    config = _config()
    config["claims"]["training_authorized"] = True
    with pytest.raises(MODULE.AccountingError, match="claims drifted"):
        MODULE.build_report(config)


def test_report_is_deterministic() -> None:
    first = MODULE.build_report(_config())
    second = MODULE.build_report(_config())
    assert first == second
    assert first["baseline_parameters"]["total_parameters"] == 20_613_440
    assert first["truth_boundary"]["model100m_frozen"] is False
