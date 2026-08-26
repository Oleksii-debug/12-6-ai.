from __future__ import annotations

import json
from pathlib import Path

from twelve_six.learn191_contract import (
    FINAL_TARGET,
    MIDPOINT_TARGET,
    MODEL_SPEC_SHA256,
    PARAMETERS,
    SELECTION_EXAMPLES,
    SOURCE_FAMILY,
    TARGETS,
    _identity,
    model_spec,
)
from twelve_six.tokenization import ByteTokenizer


def test_exact_3m_fixed_control_geometry() -> None:
    spec = model_spec()
    assert spec.parameter_count() == PARAMETERS == 3_213_120
    assert spec.identity_sha256() == MODEL_SPEC_SHA256
    assert spec.max_seq_len == 256
    assert spec.vocab_size == 256
    assert spec.n_heads == spec.n_kv_heads == 12
    assert spec.head_dim == 16
    assert spec.d_ff / spec.d_model == 2.75
    assert abs(PARAMETERS - 3_221_432) / 3_221_432 < 0.003


def test_budget_is_preregistered_and_sub_one_percent_exposure() -> None:
    assert TARGETS == (16_632, 65_772, 131_292)
    assert MIDPOINT_TARGET == TARGETS[1]
    assert FINAL_TARGET == TARGETS[-1]
    assert FINAL_TARGET / 20_000_775 < 0.01
    assert 0.04 < FINAL_TARGET / PARAMETERS < 0.05


def test_source_family_contract_is_one_to_one_for_data25() -> None:
    assert SOURCE_FAMILY == {
        "uk": "project-authored:uk:corpus-v01",
        "en": "project-authored:en:corpus-v01",
        "code": "project-authored:code:corpus-v01",
    }
    assert set(SELECTION_EXAMPLES) == {"uk", "en", "code"}


def test_selection_identity_is_deterministic() -> None:
    manifest = {"corpus_identity_sha256": "a" * 64}
    tok = ByteTokenizer()
    first = _identity(
        manifest,
        tok,
        "validation",
        SELECTION_EXAMPLES,
        "test-selection",
        "checkpoint selection only; not final test",
    )
    second = _identity(
        manifest,
        tok,
        "validation",
        SELECTION_EXAMPLES,
        "test-selection",
        "checkpoint selection only; not final test",
    )
    assert first == second
    assert first["split"] == "validation"
    assert len(first["identity_sha256"]) == 64


def test_launch_request_matches_geometry_and_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    request = json.loads(
        (root / "configs/launch/learn191-3m.json").read_text(encoding="utf-8")
    )
    assert request["binding"] == {"workflow": "learn191-real-3m", "scale": "3m"}
    assert request["model_spec_sha256"] == MODEL_SPEC_SHA256
    assert request["parameter_count"] == PARAMETERS
    assert request["budget"]["target_optimized_tokens"] == FINAL_TARGET
    assert request["requires_gpu"] is False
