from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "run_d03_franko1901_global_dedup_execution_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("franko1901_global_dedup_execution", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(source_id: str, *, family: str = "base", size: int = 5) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": family,
        "declared_capacity_bytes": size,
    }


def test_compose_graph_is_add_only_and_preserves_inputs() -> None:
    mod = _load()
    base_inventory = {
        "schema_version": "fixture",
        "sources": [_row("base:a")],
        "lineage_edges": [{"left_source_id": "base:a", "right_source_id": "base:a"}],
        "final_refresh_required": True,
    }
    base_payloads = {"base:a": b"alpha"}
    extension_sources = [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=4)]
    extension_payloads = {"franko:a": b"beta"}
    before = deepcopy(base_inventory)

    inventory, payloads = mod._compose_graph(
        base_inventory, base_payloads, extension_sources, extension_payloads
    )

    assert base_inventory == before
    assert inventory["sources"] == [base_inventory["sources"][0], extension_sources[0]]
    assert inventory["lineage_edges"] == base_inventory["lineage_edges"]
    assert inventory["final_refresh_required"] is False
    assert payloads == {"base:a": b"alpha", "franko:a": b"beta"}


@pytest.mark.parametrize(
    ("base_payloads", "extension_payloads", "message"),
    [
        ({"other": b"a"}, {"franko:a": b"b"}, "base inventory/payload"),
        ({"base:a": b"a"}, {"other": b"b"}, "extension inventory/payload"),
    ],
)
def test_compose_graph_rejects_incomplete_payload_coverage(
    base_payloads: dict[str, bytes],
    extension_payloads: dict[str, bytes],
    message: str,
) -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match=message):
        mod._compose_graph(
            {"sources": [_row("base:a")]},
            base_payloads,
            [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY)],
            extension_payloads,
        )


def test_compose_graph_rejects_source_id_collision() -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match="collision"):
        mod._compose_graph(
            {"sources": [_row("same")]},
            {"same": b"a"},
            [_row("same", family=mod.franko1901.SOURCE_FAMILY)],
            {"same": b"b"},
        )


def test_survivor_authority_accounts_only_franko_family() -> None:
    mod = _load()
    dedup = {
        "report_sha256": "1" * 64,
        "sources": [
            _row("base:a", size=10),
            _row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=7),
            _row("franko:b", family=mod.franko1901.SOURCE_FAMILY, size=11),
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
        "duplicate_clusters": [],
        "survivor_source_ids": ["base:a", "franko:b"],
    }

    authority = mod._outer_survivor_authority(dedup, projection)

    assert authority["franko1901"]["post_dedup_survivor_source_object_count"] == 1
    assert authority["franko1901"]["post_dedup_survivor_declared_capacity_bytes"] == 11
    assert authority["truth_boundary"]["canonical_capacity_credited"] == 0
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert authority["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert len(authority["survivor_authority_sha256"]) == 64


@pytest.mark.parametrize("bad", [True, 11.0, "11", -1])
def test_survivor_authority_rejects_nonexact_capacity(bad: object) -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match="exact nonnegative int"):
        mod._outer_survivor_authority(
            {
                "report_sha256": "1" * 64,
                "sources": [
                    _row(
                        "franko:a",
                        family=mod.franko1901.SOURCE_FAMILY,
                        size=bad,  # type: ignore[arg-type]
                    )
                ],
            },
            {
                "schema_version": "fixture",
                "survivor_authority_sha256": "2" * 64,
                "survivor_source_ids": ["franko:a"],
            },
        )


def test_production_arithmetic_binds_exact_franko_candidate() -> None:
    mod = _load()
    assert mod.EXPECTED_BASE_OBJECTS == 264
    assert mod.EXPECTED_BASE_BYTES == 6_095_624
    assert mod.EXPECTED_FRANKO1901_OBJECTS == 30_660
    assert mod.EXPECTED_FRANKO1901_BYTES == 1_762_005
    assert mod.EXPECTED_COMBINED_OBJECTS == 30_924
    assert mod.EXPECTED_COMBINED_BYTES == 7_857_629
    assert mod.franko1901.UPSTREAM_PRODUCT_PR == 1025
    assert mod.FRANKO1901_FINAL_HEAD == "5816f0ff4ca2f53053123063cd2471442a07d974"
    assert mod.EXPECTED_MAIN == "ba9e49cedba4a110e1c4f7d83702e8fcf8a42461"


def test_composed_inventory_names_exact_franko_product_authority() -> None:
    mod = _load()
    inventory, _ = mod._compose_graph(
        {"sources": [_row("base:a")]},
        {"base:a": b"alpha"},
        [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=4)],
        {"franko:a": b"beta"},
    )
    rule = inventory["terminal_refresh_rule"]
    assert "PR #1025 source-admitted Franko1901" in rule
    assert "PR #1347" not in rule


def test_authority_surface_binds_franko_and_incumbent_execution() -> None:
    mod = _load()
    expected = {
        "src/twelve_six/data/franko1901_dedup_intake.py",
        "configs/data/d03_franko1901_fresh_execution_authority_v1.json",
        "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
        "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
        "tools/run_d03_expanded_global_dedup_v9.py",
        "tools/run_next100_065f_global_dedup_v8.py",
    }
    assert expected <= set(mod.AUTHORITY_PATHS)


def test_repository_authority_is_current_main_ancestry_bound() -> None:
    mod = _load()
    blobs = mod.verify_repository_authority()
    assert set(blobs) == set(mod.AUTHORITY_PATHS)
    assert all(len(value) == 40 for value in blobs.values())


def test_historical_matcher_namespace_includes_pipeline() -> None:
    mod = _load()
    assert "twelve_six.data.pipeline" in mod._HISTORICAL_MATCHER_MODULES


def test_runtime_attestation_precedes_indexed_matcher_path() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    attest = source.index("indexed.attest_incumbent_runtime(matcher)")
    indexed = source.index("indexed_report = indexed.audit_payloads_indexed(")
    assert attest < indexed


def test_production_execution_does_not_reintroduce_all_pairs_reference() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "matcher.audit_payloads(inventory, payloads)" not in source
    assert '"indexed_executor_performance_equivalence_authority": "MERGED_PR_1459"' in source


def test_execution_does_not_rebuild_index_only_for_telemetry() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "candidate_pair_indices_with_stats" not in source
    assert "execution_stats(" not in source


def test_execute_uses_exact_three_part_franko_authority() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "franko1901.validate_and_project_franko(" in source
    assert "candidate_jsonl," in source
    assert "historical_terminal_evidence_json," in source
    assert "fresh_execution_authority_json," in source
    assert "validate_and_project_franko1901" not in source


def test_terminal_summary_requires_exact_arithmetic() -> None:
    mod = _load()
    good = {
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": mod.EXPECTED_COMBINED_BYTES - 17,
            "duplicate_discount_bytes": 17,
            "duplicate_cluster_count": 3,
        },
    }
    assert mod._validated_terminal_summary(good) == good["terminal_candidates"]

    for field, bad in (
        ("conservative_unique_capacity_bytes_after", mod.EXPECTED_COMBINED_BYTES + 1),
        ("duplicate_discount_bytes", 18),
        ("duplicate_cluster_count", -1),
    ):
        broken = {
            "source_count": good["source_count"],
            "terminal_candidates": dict(good["terminal_candidates"]),
        }
        broken["terminal_candidates"][field] = bad
        with pytest.raises(mod.Franko1901GlobalDedupError):
            mod._validated_terminal_summary(broken)


def test_terminal_summary_rejects_bool_aliases() -> None:
    mod = _load()
    report = {
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": True,
            "duplicate_discount_bytes": mod.EXPECTED_COMBINED_BYTES - 1,
            "duplicate_cluster_count": 0,
        },
    }
    with pytest.raises(mod.Franko1901GlobalDedupError):
        mod._validated_terminal_summary(report)


def _good_survivor_projection(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 17
    report = {
        "report_sha256": "1" * 64,
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": after,
            "duplicate_discount_bytes": 17,
            "duplicate_cluster_count": 1,
        },
    }
    projection = {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": report["report_sha256"],
        "pre_dedup_source_object_count": mod.EXPECTED_COMBINED_OBJECTS,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": mod.EXPECTED_COMBINED_BYTES,
        "post_dedup_declared_capacity_bytes": after,
        "duplicate_discount_bytes": 17,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [
            {
                "member_source_ids": ["base:a", "franko:a"],
                "selected_source_id": "base:a",
                "selected_declared_capacity_bytes": after,
            }
        ],
        "survivor_source_ids": ["base:a", "franko:b"],
    }
    return report, projection


def test_survivor_projection_is_cross_bound_to_terminal_report() -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    mod._validate_survivor_projection(report, projection)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("matcher_report_sha256", "2" * 64, "matcher identity"),
        ("pre_dedup_source_object_count", 1, "pre-dedup count"),
        ("pre_dedup_declared_capacity_bytes", 1, "pre-dedup bytes"),
        ("post_dedup_declared_capacity_bytes", 1, "post-dedup bytes"),
        ("duplicate_discount_bytes", 16, "duplicate discount"),
        ("duplicate_cluster_count", 0, "cluster count"),
    ],
)
def test_survivor_projection_rejects_terminal_drift(
    field: str, bad: object, message: str
) -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection[field] = bad
    with pytest.raises(mod.Franko1901GlobalDedupError, match=message):
        mod._validate_survivor_projection(report, projection)


def test_survivor_projection_rejects_id_cardinality_drift() -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection["survivor_source_ids"] = ["base:a", "base:a"]
    with pytest.raises(mod.Franko1901GlobalDedupError, match="ids invalid"):
        mod._validate_survivor_projection(report, projection)


def test_execute_validates_survivor_projection_before_outer_publication() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    derived = source.index("selection_projection = v9_semantics._derive_survivors(indexed_report)")
    validated = source.index("_validate_survivor_projection(indexed_report, selection_projection)")
    wrapped = source.index("survivors = _outer_survivor_authority(indexed_report, selection_projection)")
    assert derived < validated < wrapped


def test_source_provenance_negative_is_explicitly_source_local() -> None:
    mod = _load()
    scope = mod._source_admission_provenance_scope(
        {"truth_boundary": {"external_llm_or_api_used_for_data_or_intelligence": False}}
    )
    assert scope == {
        "upstream_source_package_external_llm_or_api_used": False,
        "current_corpus_external_llm_free_claimed_by_this_carrier": False,
        "source_local_negative_promoted_as_corpus_global_truth": False,
    }
    with pytest.raises(mod.Franko1901GlobalDedupError):
        mod._source_admission_provenance_scope(
            {"truth_boundary": {"external_llm_or_api_used_for_data_or_intelligence": True}}
        )


def test_truth_boundary_never_promotes_source_local_external_llm_negative() -> None:
    mod = _load()
    authority = mod._outer_survivor_authority(
        {
            "report_sha256": "1" * 64,
            "sources": [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY)],
        },
        {
            "schema_version": "fixture",
            "survivor_authority_sha256": "2" * 64,
            "survivor_source_ids": ["franko:a"],
        },
    )
    truth = authority["truth_boundary"]
    assert truth["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert "external_llm_or_api_used_for_data_or_intelligence" not in truth


def test_linux_max_rss_preserves_kib_even_above_ten_million(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=12_000_000),
        ),
    )
    assert mod._max_rss_kib() == 12_000_000


def test_darwin_max_rss_converts_bytes_to_kib(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=4097),
        ),
    )
    assert mod._max_rss_kib() == 5


def test_windows_max_rss_uses_windows_backend(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(mod, "_windows_peak_working_set_kib", lambda: 777)
    assert mod._max_rss_kib() == 777


def test_unknown_platform_max_rss_fails_closed(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "freebsd")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=500),
        ),
    )
    assert mod._max_rss_kib() is None


def test_execute_requires_positive_measured_rss_before_evidence() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    rss_measure = source.index("process_max_rss_kib = _max_rss_kib()")
    rss_gate = source.index('"process max RSS unavailable; refusing terminal execution evidence"')
    evidence = source.index("evidence_core = {")
    assert rss_measure < rss_gate < evidence
    assert '"process_max_rss_kib": process_max_rss_kib' in source


def test_cli_requires_both_franko_authority_documents() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--historical-terminal-evidence-json", type=Path, required=True)'
        in raw
    )
    assert (
        'parser.add_argument("--fresh-execution-authority-json", type=Path, required=True)'
        in raw
    )


def test_write_json_is_true_create_only(tmp_path: Path) -> None:
    mod = _load()
    output = tmp_path / "evidence.json"
    mod._write_json(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'

    with pytest.raises(mod.Franko1901GlobalDedupError, match="refusing to overwrite"):
        mod._write_json(output, {"changed": True})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'


def test_write_json_nonfinite_fails_before_publication(tmp_path: Path) -> None:
    mod = _load()
    output = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        mod._write_json(output, {"rss": float("nan")})
    assert not output.exists()


def test_output_set_rolls_back_when_later_path_exists(tmp_path: Path) -> None:
    mod = _load()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b"existing-evidence\n")

    with pytest.raises(mod.Franko1901GlobalDedupError, match="refusing to overwrite"):
        mod._publish_json_outputs(
            (
                (report, {"kind": "report"}),
                (survivors, {"kind": "survivors"}),
                (evidence, {"kind": "evidence"}),
            )
        )

    assert not report.exists()
    assert not survivors.exists()
    assert evidence.read_bytes() == b"existing-evidence\n"


def test_output_set_serializes_all_values_before_publication(tmp_path: Path) -> None:
    mod = _load()
    report = tmp_path / "report.json"
    evidence = tmp_path / "evidence.json"

    with pytest.raises(ValueError):
        mod._publish_json_outputs(
            (
                (report, {"kind": "report"}),
                (evidence, {"rss": float("nan")}),
            )
        )

    assert not report.exists()
    assert not evidence.exists()


def test_no_training_or_capacity_promotion_in_execution_evidence() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert '"canonical_capacity_credited": 0' in raw
    assert '"authorized_optimized_target_exposure": 0' in raw
    assert '"tokenizer_fit_authorized": False' in raw
    assert '"training_executed": False' in raw
    assert '"learned_weights_created": False' in raw
    assert '"final_test_outcomes_read": False' in raw
    assert '"paid_compute_used": False' in raw
