from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "verify_model341_20m_candidate_a_independent.py"
CONFIG_PATH = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"

_spec = importlib.util.spec_from_file_location("model341_independent_verify", TOOL_PATH)
assert _spec is not None and _spec.loader is not None
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)


def _payload() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_static_identity_and_independent_parameter_count() -> None:
    got = verifier.validate_candidate_payload(_payload())
    assert got["model_sha256"] == verifier.EXPECTED_MODEL_SHA256
    assert got["init_sha256"] == verifier.EXPECTED_INIT_SHA256
    assert got["breakdown"]["total"] == 20_613_440
    assert got["breakdown"]["q_dim"] == 320
    assert got["breakdown"]["kv_dim"] == 64
    assert got["breakdown"]["block_per_layer"] == 1_283_200


def test_adversarial_self_checks_all_fail_closed() -> None:
    checks = verifier.adversarial_self_checks(_payload())
    assert checks
    assert all(checks.values())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tie_word_embeddings", False),
        ("n_kv_heads", 4),
        ("max_seq_len", 2048),
        ("rope_rotary_dim", 30),
    ],
)
def test_material_model_drift_rejected(key: str, value: object) -> None:
    bad = copy.deepcopy(_payload())
    bad["model"][key] = value
    with pytest.raises(ValueError):
        verifier.validate_candidate_payload(bad)


def test_exact_repository_decoder_runtime_identity() -> None:
    runtime = verifier.runtime_checks(_payload(), seed=341)
    assert runtime["parameter_count"] == 20_613_440
    assert runtime["tied_embedding_object_identity"] is True
    assert runtime["tied_embedding_storage_identity"] is True
    assert runtime["q_projection_shape"] == [320, 320]
    assert runtime["k_projection_shape"] == [64, 320]
    assert runtime["v_projection_shape"] == [64, 320]
    assert runtime["out_projection_shape"] == [320, 320]
    assert runtime["rope_inv_freq_shape"] == [16]
    assert runtime["same_seed_reproduced"] is True
    assert runtime["different_seed_changed"] is True
    assert runtime["oversized_context_rejected"] is True
    assert runtime["checkpoint_loaded"] is False
