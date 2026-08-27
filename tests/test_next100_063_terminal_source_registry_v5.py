from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_next100_063_terminal_source_registry_v5.py"
V4 = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
V5 = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"

spec = importlib.util.spec_from_file_location("next100_063_v5", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _inputs() -> tuple[dict, dict, str]:
    v4 = json.loads(V4.read_text(encoding="utf-8"))
    raw = V5.read_bytes()
    v5 = json.loads(raw.decode("utf-8"))
    return v4, v5, module.git_blob_sha1(raw)


def test_v5_consumes_terminal_attrs_without_promoting_training() -> None:
    v4, v5, blob = _inputs()
    module.validate(v4, v5, v5_blob_sha1=blob)
    inv = v5["derived_pre_successor_global_dedup_inventory"]
    assert inv["candidate_numeric_training_capacity_bytes"] == 2_215_615
    assert inv["candidate_source_normalized_envelope_bytes"] == 2_217_976
    assert inv["candidate_independent_family_count"] == 15
    assert inv["by_stratum"]["code"] == {
        "family_count": 6,
        "numeric_training_capacity_bytes": 276_466,
        "source_normalized_envelope_bytes": 276_466,
    }
    assert inv["target_gap_numeric_training_capacity_bytes"] == 17_784_385
    assert v5["downstream_gate_vector"]["authorized_balanced_no_replay_loss_positions"] == 0
    assert v5["downstream_gate_vector"]["long_training"] == "BLOCKED"
    assert v5["downstream_gate_vector"]["paid_compute"] == "NOT_AUTHORIZED"


def test_v5_rejects_nonterminal_attrs_rebinding() -> None:
    v4, v5, _ = _inputs()
    bad = copy.deepcopy(v5)
    bad["terminal_addition"]["dedicated_workflow_conclusion"] = "queued"
    with pytest.raises(module.RegistryV5Error, match="terminal success"):
        module.validate(v4, bad, v5_blob_sha1=module.EXPECTED_V5_BLOB_SHA1)


def test_v5_rejects_attrs_capacity_inflation() -> None:
    v4, v5, _ = _inputs()
    bad = copy.deepcopy(v5)
    bad["terminal_addition"]["numeric_training_capacity_bytes"] += 1
    with pytest.raises(module.RegistryV5Error, match="attrs capacity drift"):
        module.validate(v4, bad, v5_blob_sha1=module.EXPECTED_V5_BLOB_SHA1)


def test_v5_rejects_base_v4_identity_drift() -> None:
    v4, v5, _ = _inputs()
    bad_v4 = copy.deepcopy(v4)
    bad_v4["registry_identity_sha256"] = "0" * 64
    with pytest.raises(module.RegistryV5Error, match="V4 declared identity drift"):
        module.validate(bad_v4, v5, v5_blob_sha1=module.EXPECTED_V5_BLOB_SHA1)


def test_v5_rejects_config_blob_drift() -> None:
    v4, v5, _ = _inputs()
    with pytest.raises(module.RegistryV5Error, match="V5 config blob drift"):
        module.validate(v4, v5, v5_blob_sha1="0" * 40)
