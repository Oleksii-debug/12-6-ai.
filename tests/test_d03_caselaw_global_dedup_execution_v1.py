from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "run_d03_caselaw_global_dedup_execution_v1.py"
)
spec = importlib.util.spec_from_file_location("caselaw_global_dedup_execution", MODULE)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _row(source_id: str, *, family: str = "base", size: int = 5) -> dict:
    return {
        "source_id": source_id,
        "source_family": family,
        "declared_capacity_bytes": size,
    }


def test_compose_graph_is_add_only_and_preserves_inputs() -> None:
    base_inventory = {
        "schema_version": "fixture",
        "sources": [_row("base:a")],
        "lineage_edges": [{"left_source_id": "base:a", "right_source_id": "base:a"}],
        "final_refresh_required": True,
    }
    base_payloads = {"base:a": b"alpha"}
    extension_sources = [_row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY, size=4)]
    extension_payloads = {"caselaw:a": b"beta"}
    inventory_before = deepcopy(base_inventory)

    inventory, payloads = mod._compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )

    assert base_inventory == inventory_before
    assert inventory["sources"] == [base_inventory["sources"][0], extension_sources[0]]
    assert inventory["lineage_edges"] == base_inventory["lineage_edges"]
    assert inventory["final_refresh_required"] is False
    assert set(payloads) == {"base:a", "caselaw:a"}
    assert payloads["base:a"] == b"alpha"
    assert payloads["caselaw:a"] == b"beta"


def test_compose_graph_rejects_source_id_collision() -> None:
    with pytest.raises(mod.CaselawGlobalDedupError, match="collision"):
        mod._compose_graph(
            {"sources": [_row("same")]},
            {"same": b"a"},
            [_row("same", family=mod.caselaw.SOURCE_FAMILY)],
            {"same": b"b"},
        )


def test_compose_graph_requires_exact_payload_coverage() -> None:
    with pytest.raises(mod.CaselawGlobalDedupError, match="base inventory/payload"):
        mod._compose_graph(
            {"sources": [_row("base:a")]},
            {"other": b"a"},
            [_row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY)],
            {"caselaw:a": b"b"},
        )
    with pytest.raises(mod.CaselawGlobalDedupError, match="extension inventory/payload"):
        mod._compose_graph(
            {"sources": [_row("base:a")]},
            {"base:a": b"a"},
            [_row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY)],
            {"other": b"b"},
        )


def test_survivor_wrapper_accounts_caselaw_after_cross_family_selection() -> None:
    dedup = {
        "report_sha256": "1" * 64,
        "sources": [
            _row("base:a", size=10),
            _row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY, size=7),
            _row("caselaw:b", family=mod.caselaw.SOURCE_FAMILY, size=11),
        ],
    }
    projection = {
        "schema_version": "fixture-survivors",
        "survivor_authority_sha256": "2" * 64,
        "pre_dedup_source_object_count": 3,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 28,
        "post_dedup_declared_capacity_bytes": 21,
        "duplicate_discount_bytes": 7,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [
            {
                "member_source_ids": ["base:a", "caselaw:a"],
                "selected_source_id": "base:a",
                "selected_declared_capacity_bytes": 10,
            }
        ],
        "survivor_source_ids": ["base:a", "caselaw:b"],
    }

    authority = mod._outer_survivor_authority(dedup, projection)

    assert authority["matcher_report_sha256"] == "1" * 64
    assert authority["caselaw"]["post_dedup_survivor_source_object_count"] == 1
    assert authority["caselaw"]["post_dedup_survivor_declared_capacity_bytes"] == 11
    assert authority["truth_boundary"]["canonical_capacity_credited"] == 0
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert authority["truth_boundary"]["tokenizer_fit_authorized"] is False
    assert len(authority["survivor_authority_sha256"]) == 64


def test_survivor_wrapper_rejects_unknown_selected_source() -> None:
    with pytest.raises(mod.CaselawGlobalDedupError, match="unknown survivor"):
        mod._outer_survivor_authority(
            {"report_sha256": "1" * 64, "sources": [_row("base:a")]},
            {
                "schema_version": "fixture",
                "survivor_authority_sha256": "2" * 64,
                "survivor_source_ids": ["missing"],
            },
        )


def test_production_arithmetic_is_exact() -> None:
    assert mod.EXPECTED_COMBINED_OBJECTS == 5_922
    assert mod.EXPECTED_COMBINED_BYTES == 12_057_771
    assert mod.EXPECTED_MAIN == "bd2d445dfd8fbd7ec6759c1398bb913f4e0c0093"
    assert mod.CASELAW_FINAL_HEAD == "deaf0730fe04a12e9abb8f3cecb14d6ad2cc7a4d"


def test_authority_surface_includes_matcher_adapter_and_materializer() -> None:
    expected = {
        "src/twelve_six/data/caselaw_source_admitted_dedup_intake.py",
        "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
        "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
        "tools/run_d03_expanded_global_dedup_v9.py",
        "tools/run_next100_065f_global_dedup_v8.py",
        "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
    }
    assert expected <= set(mod.AUTHORITY_PATHS)
