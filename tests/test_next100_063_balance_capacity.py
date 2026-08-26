from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/diagnose_next100_063_balance_capacity.py"
REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v2.json"

spec = importlib.util.spec_from_file_location("next100_063_balance_capacity", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_live_v2_registry_family_caps_make_uk_the_real_bottleneck() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    report = module.build_report(registry)

    assert report["raw_pre_global_dedup_bytes"] == 303_374
    assert report["diagnostic_exact_mixture_family_capped_source_bytes"] == 61_440
    assert report["next_20_byte_increment_limiting_strata"] == ["uk"]

    assert report["strata"]["uk"]["20m_raw_capacity_gap_bytes"] == 8_899_144
    assert report["strata"]["en"]["20m_raw_capacity_gap_bytes"] == 6_849_357
    assert report["strata"]["code"]["20m_raw_capacity_gap_bytes"] == 3_948_125

    assert report["truth_boundary"]["post_pack_unique_loss_positions"] == 0
    assert report["truth_boundary"]["training_authorized"] is False


def test_registry_identity_drift_fails_closed() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["registry_identity_sha256"] = "0" * 64

    with pytest.raises(module.CapacityDiagnosticError, match="registry identity drifted"):
        module.build_report(registry)
