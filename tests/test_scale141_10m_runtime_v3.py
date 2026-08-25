from __future__ import annotations

from twelve_six.scale141_10m_runtime_v3 import _json_normalize


def test_json_normalization_makes_tuple_config_cross_process_stable() -> None:
    value = {
        "trainer_config": {
            "betas": (0.9, 0.95),
            "nested": ("constant", 20000),
        }
    }
    normalized = _json_normalize(value)
    assert normalized == {
        "trainer_config": {
            "betas": [0.9, 0.95],
            "nested": ["constant", 20000],
        }
    }
    assert _json_normalize(normalized) == normalized
