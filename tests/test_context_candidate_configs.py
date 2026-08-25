from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.model import load_stage_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_context", "expected_parameters"),
    [
        ("configs/context/s1_100k_context_512.experimental.json", 512, 107_856),
        ("configs/context/s2_1m_context_1024.experimental.json", 1024, 1_066_112),
        ("configs/context/s3_10m_context_2048.experimental.json", 2048, 10_059_840),
        ("configs/context/s3_10m_context_4096.research.json", 4096, 10_059_840),
    ],
)
def test_context_candidate_stage_configs_are_executable_and_identity_valid(
    relative_path: str,
    expected_context: int,
    expected_parameters: int,
) -> None:
    config = load_stage_config(ROOT / relative_path)
    assert config.model.max_seq_len == expected_context
    assert config.model.parameter_count() == expected_parameters
    assert config.expected_parameters == expected_parameters


def test_context_candidates_do_not_change_canonical_stage_contexts() -> None:
    canonical = {
        "configs/stages/s0_10k.json": 128,
        "configs/stages/s1_100k.json": 256,
        "configs/stages/s2_1m.json": 512,
        "configs/stages/s3_10m.json": 1024,
    }
    for relative_path, expected_context in canonical.items():
        config = load_stage_config(ROOT / relative_path)
        assert config.model.max_seq_len == expected_context
