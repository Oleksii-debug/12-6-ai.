from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/diagnose_next100_063_balance_capacity.py"
REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"

spec = importlib.util.spec_from_file_location("next100_063_balance_capacity", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_live_v4_registry_family_caps_keep_uk_as_real_bottleneck() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    report = module.build_report(registry)

    assert (
        report["raw_pre_successor_global_dedup_numeric_training_capacity_bytes"]
        == 2_045_180
    )
    assert report["source_normalized_envelope_bytes"] == 2_047_541
    assert report["uncredited_source_normalized_bytes"] == 2_361
    assert report["diagnostic_exact_mixture_family_capped_source_bytes"] == 61_440
    assert report["next_20_byte_increment_limiting_strata"] == ["uk"]

    assert report["strata"]["uk"]["family_count"] == 4
    assert report["strata"]["en"]["family_count"] == 5
    assert report["strata"]["code"]["family_count"] == 5
    assert report["strata"]["uk"]["20m_raw_capacity_gap_bytes"] == 8_899_144
    assert report["strata"]["en"]["20m_raw_capacity_gap_bytes"] == 5_161_707
    assert report["strata"]["code"]["20m_raw_capacity_gap_bytes"] == 3_893_969

    # The large Gutenberg addition increases raw EN capacity but cannot increase
    # the balanced no-replay envelope while UK remains family-cap constrained.
    assert report["strata"]["en"]["family_capacity_bytes"][
        "en.project-gutenberg.public-domain-books"
    ] == 1_672_110
    assert report["strata"]["uk"]["feasible_exact_mixture_required_bytes"] == 27_648

    assert report["truth_boundary"]["post_pack_unique_loss_positions"] == 0
    assert report["truth_boundary"]["training_authorized"] is False


def test_registry_identity_drift_fails_closed() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["registry_identity_sha256"] = "0" * 64

    with pytest.raises(module.CapacityDiagnosticError, match="registry identity drifted"):
        module.build_report(registry)
