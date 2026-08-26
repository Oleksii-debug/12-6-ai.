from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

CONFIG = Path("configs/data/next100_065c_live_source_reconciliation_v1.json")
VALIDATOR = Path("tools/validate_next100_065c_live_source_reconciliation.py")


def _module():
    spec = importlib.util.spec_from_file_location("reconcile", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_canonical_reconciliation_passes():
    result = _module().validate(_data())
    assert result["status"] == "PASS"
    assert result["required_pre_global_dedup_bytes"] == 2_045_180
    assert result["required_independent_families"] == 14
    assert result["authorized_unique_causal_loss_positions"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("registry_v3", "identity_sha256"), "0" * 64),
        (("accepted_only_cpython", "eligible_capacity_bytes"), 17_901),
        (("gutenberg_terminal", "normalized_utf8_bytes"), 1),
        (
            (
                "minimum_required_successor_vector_before_global_dedup",
                "numeric_capacity_bytes",
                "total",
            ),
            2_045_181,
        ),
        (("promotion_gate", "authorized_unique_causal_loss_positions"), 1),
        (("promotion_gate", "long_training_authorized"), True),
    ],
)
def test_reconciliation_fails_closed(path, value):
    module = _module()
    data = copy.deepcopy(_data())
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(module.ReconciliationError):
        module.validate(data)
