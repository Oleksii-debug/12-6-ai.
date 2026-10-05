from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "run_d03_rada_current_global_dedup_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_d03_rada_current_global_dedup_test",
        MODULE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(
    source_id: str,
    *,
    family: str = "base.family",
    payload: bytes = b"alpha",
) -> tuple[dict[str, object], bytes]:
    payload_sha = hashlib.sha256(payload).hexdigest()
    return (
        {
            "source_id": source_id,
            "source_family": family,
            "stable_origin_id": f"origin:{source_id}",
            "stable_object_id": f"sha256:{payload_sha}",
            "modality": "uk",
            "evidence_status": "TEST",
            "declared_capacity_bytes": len(payload),
            "expected_raw_bytes": len(payload),
            "expected_raw_sha256": payload_sha,
            "acquisition_url": "https://example.invalid/source",
            "origin_key": f"unit:{source_id}",
        },
        payload,
    )


def _report_row(mod, row: dict[str, object], payload: bytes) -> dict[str, object]:
    stable_origin_id = str(row["stable_origin_id"])
    stable_object_id = str(row["stable_object_id"])
    payload_sha = mod._sha256(payload)
    return {
        "source_id": row["source_id"],
        "source_family": row["source_family"],
        "modality": row["modality"],
        "evidence_status": row["evidence_status"],
        "declared_capacity_bytes": row["declared_capacity_bytes"],
        "stable_origin_id_sha256": mod._sha256(stable_origin_id.encode("utf-8")),
        "stable_object_id_sha256": mod._sha256(stable_object_id.encode("utf-8")),
        "verified_raw_bytes": len(payload),
        "verified_raw_sha256": payload_sha,
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": payload_sha,
    }


def test_carrier_constants_bind_current_rada_projection() -> None:
    mod = _load()
    assert mod.EXPECTED_PRODUCT_PARENT == (
        "fb49b7e212444547219df2bd2aa466db955d51e5"
    )
    assert mod.EXPECTED_BASE_OBJECTS == 263
    assert mod.EXPECTED_BASE_BYTES == 6_093_965
    assert mod.EXPECTED_RADA_OBJECTS == 101_733
    assert mod.EXPECTED_RADA_BYTES == 192_393_157
    assert mod.EXPECTED_COMBINED_OBJECTS == 101_996
    assert mod.EXPECTED_COMBINED_BYTES == 198_487_122
    assert mod.EXPECTED_MAX_CANDIDATE_PAIRS == 5_000_000
    assert mod.EXPECTED_MAX_INDEX_POSTINGS == 50_000_000
    assert mod.EXPECTED_MAX_PAIR_EXPANSIONS == 50_000_000


def test_work_budgets_accept_only_independently_selected_rada_scale_bounds() -> None:
    mod = _load()
    mod._validate_work_budgets(5_000_000, 50_000_000, 50_000_000)

    for values in (
        (4_999_999, 50_000_000, 50_000_000),
        (5_000_001, 50_000_000, 50_000_000),
        (5_000_000, 49_999_999, 50_000_000),
        (5_000_000, 50_000_001, 50_000_000),
        (5_000_000, 50_000_000, 49_999_999),
        (5_000_000, 50_000_000, 50_000_001),
        (True, 50_000_000, 50_000_000),
        (5_000_000, 50_000_000.0, 50_000_000),
    ):
        with pytest.raises(
            mod.RadaCurrentGlobalDedupError,
            match="independently selected Rada-scale bound",
        ):
            mod._validate_work_budgets(*values)


def test_declared_capacity_rejects_bool_negative_and_coverage_drift() -> None:
    mod = _load()
    row, payload = _source("base:a")
    assert (
        mod._declared_capacity_bytes(
            {"sources": [row]},
            {"base:a": payload},
            label="fixture",
        )
        == len(payload)
    )

    bad_bool = deepcopy(row)
    bad_bool["declared_capacity_bytes"] = True
    with pytest.raises(mod.RadaCurrentGlobalDedupError, match="capacity invalid"):
        mod._declared_capacity_bytes(
            {"sources": [bad_bool]},
            {"base:a": payload},
            label="fixture",
        )

    bad_negative = deepcopy(row)
    bad_negative["declared_capacity_bytes"] = -1
    with pytest.raises(mod.RadaCurrentGlobalDedupError, match="capacity invalid"):
        mod._declared_capacity_bytes(
            {"sources": [bad_negative]},
            {"base:a": payload},
            label="fixture",
        )

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="inventory/payload coverage mismatch",
    ):
        mod._declared_capacity_bytes(
            {"sources": [row]},
            {"base:other": payload},
            label="fixture",
        )


def test_compose_graph_rejects_any_preexisting_rada_family() -> None:
    mod = _load()
    base_row, base_payload = _source(
        "historical-rada",
        family=mod.rada.SOURCE_FAMILY,
    )
    rada_row, rada_payload = _source(
        "rada-laws-qp:d100.htm.q00000",
        family=mod.rada.SOURCE_FAMILY,
        payload=b"current",
    )

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="append is forbidden",
    ):
        mod._compose_graph(
            {"sources": [base_row]},
            {"historical-rada": base_payload},
            [rada_row],
            {str(rada_row["source_id"]): rada_payload},
        )


def test_compose_graph_preserves_base_and_current_rada_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    base_row, base_payload = _source("base:a", payload=b"base")
    rada_row, rada_payload = _source(
        "rada-laws-qp:d100.htm.q00000",
        family=mod.rada.SOURCE_FAMILY,
        payload="Рада".encode("utf-8"),
    )
    base_inventory = {"sources": [base_row], "lineage_edges": []}
    base_payloads = {"base:a": base_payload}
    base_before = deepcopy(base_inventory)
    payloads_before = dict(base_payloads)

    monkeypatch.setattr(
        mod,
        "_verify_clean_payload_graph",
        lambda inventory, payloads, authority: None,
    )
    monkeypatch.setattr(
        mod.json,
        "loads",
        lambda payload: {"schema_version": "test"},
    )
    inventory, payloads = mod._compose_graph(
        base_inventory,
        base_payloads,
        [rada_row],
        {str(rada_row["source_id"]): rada_payload},
    )

    assert base_inventory == base_before
    assert base_payloads == payloads_before
    assert [row["source_id"] for row in inventory["sources"]] == [
        "base:a",
        "rada-laws-qp:d100.htm.q00000",
    ]
    assert payloads == {
        "base:a": base_payload,
        "rada-laws-qp:d100.htm.q00000": rada_payload,
    }
    assert inventory["final_refresh_required"] is False
    assert "not additive" in inventory["terminal_refresh_rule"]


def test_compose_graph_rejects_source_id_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    base_row, base_payload = _source("same")
    rada_row, rada_payload = _source(
        "same",
        family=mod.rada.SOURCE_FAMILY,
        payload=b"current",
    )
    monkeypatch.setattr(
        mod,
        "_verify_clean_payload_graph",
        lambda inventory, payloads, authority: None,
    )

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="source-id collision",
    ):
        mod._compose_graph(
            {"sources": [base_row]},
            {"same": base_payload},
            [rada_row],
            {"same": rada_payload},
        )


def test_report_binding_authenticates_rada_payload_and_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    row, payload = _source(
        "rada-laws-qp:d100.htm.q00000",
        family=mod.rada.SOURCE_FAMILY,
        payload="Український закон".encode("utf-8"),
    )
    report_row = _report_row(mod, row, payload)
    monkeypatch.setattr(mod, "EXPECTED_RADA_OBJECTS", 1)

    mod._validate_rada_report_binding(
        {"sources": [report_row]},
        [row],
        {str(row["source_id"]): payload},
    )

    for field, replacement, pattern in (
        ("stable_origin_id_sha256", "0" * 64, "stable origin drift"),
        ("stable_object_id_sha256", "0" * 64, "stable object drift"),
        ("verified_raw_sha256", "0" * 64, "raw payload verification drift"),
        ("comparison_payload_sha256", "0" * 64, "comparison payload drift"),
        ("comparison_policy", "OTHER", "comparison policy drift"),
    ):
        changed = deepcopy(report_row)
        changed[field] = replacement
        with pytest.raises(mod.RadaCurrentGlobalDedupError, match=pattern):
            mod._validate_rada_report_binding(
                {"sources": [changed]},
                [row],
                {str(row["source_id"]): payload},
            )


def test_report_binding_rejects_bool_alias_for_declared_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    row, payload = _source(
        "rada-laws-qp:d100.htm.q00000",
        family=mod.rada.SOURCE_FAMILY,
    )
    report_row = _report_row(mod, row, payload)
    report_row["declared_capacity_bytes"] = True
    monkeypatch.setattr(mod, "EXPECTED_RADA_OBJECTS", 1)

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="declared_capacity_bytes",
    ):
        mod._validate_rada_report_binding(
            {"sources": [report_row]},
            [row],
            {str(row["source_id"]): payload},
        )


def test_terminal_summary_rejects_bool_alias_and_bad_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_OBJECTS", 2)
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_BYTES", 10)
    report = {
        "source_count": 2,
        "terminal_candidates": {
            "declared_capacity_bytes_before": 10,
            "conservative_unique_capacity_bytes_after": 8,
            "duplicate_discount_bytes": 2,
            "duplicate_cluster_count": 1,
        },
    }
    assert mod._validated_terminal_summary(report)["duplicate_discount_bytes"] == 2

    for field, bad in (
        ("declared_capacity_bytes_before", True),
        ("conservative_unique_capacity_bytes_after", True),
        ("duplicate_discount_bytes", True),
        ("duplicate_cluster_count", False),
    ):
        changed = deepcopy(report)
        changed["terminal_candidates"][field] = bad
        with pytest.raises(mod.RadaCurrentGlobalDedupError):
            mod._validated_terminal_summary(changed)

    changed = deepcopy(report)
    changed["terminal_candidates"]["duplicate_discount_bytes"] = 3
    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="arithmetic drift",
    ):
        mod._validated_terminal_summary(changed)


def test_survivor_projection_rejects_count_and_cluster_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_OBJECTS", 2)
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_BYTES", 15)
    report = {
        "report_sha256": "1" * 64,
        "source_count": 2,
        "terminal_candidates": {
            "declared_capacity_bytes_before": 15,
            "conservative_unique_capacity_bytes_after": 15,
            "duplicate_discount_bytes": 0,
            "duplicate_cluster_count": 0,
        },
    }
    core = {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": "1" * 64,
        "pre_dedup_source_object_count": 2,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 15,
        "post_dedup_declared_capacity_bytes": 15,
        "duplicate_discount_bytes": 0,
        "duplicate_cluster_count": 0,
        "duplicate_clusters": [],
        "survivor_source_ids": ["base:a", "rada:a"],
    }

    for field, bad, message in (
        ("post_dedup_survivor_source_object_count", True, "object count drift"),
        ("duplicate_cluster_count", False, "cluster count drift"),
    ):
        changed = deepcopy(core)
        changed[field] = bad
        changed["survivor_authority_sha256"] = mod._sha256(
            mod._canonical(changed)
        )
        unhashed = dict(changed)
        unhashed.pop("survivor_authority_sha256")
        changed["survivor_authority_sha256"] = mod._sha256(
            mod._canonical(unhashed)
        )
        with pytest.raises(mod.RadaCurrentGlobalDedupError, match=message):
            mod._validate_survivor_projection(report, changed)


def test_outer_survivor_authority_keeps_rights_and_training_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_OBJECTS", 2)
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_BYTES", 15)
    report = {
        "report_sha256": "1" * 64,
        "source_count": 2,
        "terminal_candidates": {
            "declared_capacity_bytes_before": 15,
            "conservative_unique_capacity_bytes_after": 15,
            "duplicate_discount_bytes": 0,
            "duplicate_cluster_count": 0,
        },
        "sources": [
            {
                "source_id": "base:a",
                "source_family": "base.family",
                "declared_capacity_bytes": 5,
            },
            {
                "source_id": "rada:a",
                "source_family": mod.rada.SOURCE_FAMILY,
                "declared_capacity_bytes": 10,
            },
        ],
    }
    core = {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": "1" * 64,
        "pre_dedup_source_object_count": 2,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 15,
        "post_dedup_declared_capacity_bytes": 15,
        "duplicate_discount_bytes": 0,
        "duplicate_cluster_count": 0,
        "duplicate_clusters": [],
        "survivor_source_ids": ["base:a", "rada:a"],
    }
    projection = {
        **core,
        "survivor_authority_sha256": mod._sha256(mod._canonical(core)),
    }
    mod._validate_survivor_projection(report, projection)
    authority = mod._outer_survivor_authority(report, projection)

    assert authority["rada_survivor_source_object_count"] == 1
    assert authority["rada_survivor_declared_capacity_bytes"] == 10
    assert authority["rights_scope"] == (
        "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY"
    )
    assert authority["rights_recheck_for_training_required"] is True
    assert authority["canonical_capacity_credited"] == 0
    assert authority["training_authorized_bytes"] == 0
    claimed = authority["survivor_authority_sha256"]
    unhashed = dict(authority)
    unhashed.pop("survivor_authority_sha256")
    assert claimed == mod._sha256(mod._canonical(unhashed))


def test_publish_output_directory_is_create_only_and_all_or_nothing(
    tmp_path: Path,
) -> None:
    mod = _load()
    output = tmp_path / "evidence"
    values = {
        "a.json": {"a": 1},
        "b.json": {"b": 2},
    }
    mod._publish_output_directory(output, values)
    assert json.loads((output / "a.json").read_text(encoding="utf-8")) == {"a": 1}
    assert json.loads((output / "b.json").read_text(encoding="utf-8")) == {"b": 2}

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="refusing to overwrite",
    ):
        mod._publish_output_directory(output, values)

    failed = tmp_path / "failed"
    with pytest.raises(ValueError):
        mod._publish_output_directory(
            failed,
            {
                "first.json": {"ok": 1},
                "second.json": {"bad": float("nan")},
            },
        )
    assert not failed.exists()


def test_publish_output_directory_rejects_dangling_symlink(tmp_path: Path) -> None:
    mod = _load()
    output = tmp_path / "evidence"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="refusing to overwrite",
    ):
        mod._publish_output_directory(output, {"a.json": {"a": 1}})


@pytest.mark.parametrize("filename", ["../escape.json", "nested/a.json", "", "."])
def test_publish_output_directory_rejects_unsafe_filenames(
    tmp_path: Path,
    filename: str,
) -> None:
    mod = _load()
    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="filename is unsafe",
    ):
        mod._publish_output_directory(
            tmp_path / "evidence",
            {filename: {"a": 1}},
        )


def test_execute_rejects_budget_drift_before_base_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load()
    events: list[str] = []
    monkeypatch.setattr(
        mod,
        "_bind_execution_head",
        lambda head: events.append("head") or head,
    )
    monkeypatch.setattr(
        mod,
        "verify_repository_authority",
        lambda: events.append("authority") or {},
    )
    monkeypatch.setattr(
        mod,
        "_reconstruct_v8_with_historical_namespace",
        lambda **kwargs: pytest.fail("base reconstruction ran after budget rejection"),
    )

    with pytest.raises(
        mod.RadaCurrentGlobalDedupError,
        match="independently selected Rada-scale bound",
    ):
        mod.execute(
            v7_root=tmp_path,
            bulk_workspace=tmp_path / "bulk",
            candidate_jsonl=tmp_path / "candidate.jsonl",
            output_dir=tmp_path / "out",
            expected_execution_head="a" * 40,
            max_candidate_pairs=5_000_001,
            max_index_postings=50_000_000,
            max_pair_expansions=50_000_000,
        )
    assert events == ["head", "authority"]
