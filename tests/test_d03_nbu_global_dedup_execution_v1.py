from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "run_d03_nbu_global_dedup_execution_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_d03_nbu_execution_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carrier_constants_bind_current_nbu_projection() -> None:
    mod = _load()
    assert mod.EXECUTION_CLAIM == 2398
    assert mod.EXECUTION_PR == 2454
    assert mod.EXPECTED_NBU_OBJECTS == 40
    assert mod.EXPECTED_NBU_BYTES == 794_091
    assert mod.EXPECTED_COMBINED_OBJECTS == 304
    assert mod.EXPECTED_COMBINED_BYTES == 6_889_715
    assert mod.INTAKE_PATH in mod.PRODUCT_PATHS
    assert mod.CARRIER_PATH in mod.PRODUCT_PATHS
    assert mod.INTAKE_PATH not in mod.MAIN_AUTHORITY_PATHS


def test_selected_execution_head_must_equal_observed_head(monkeypatch) -> None:
    mod = _load()
    selected = "a" * 40
    monkeypatch.setattr(
        mod,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=selected + "\n",
            stderr="",
        ),
    )
    assert mod._bind_execution_head(selected) == selected


@pytest.mark.parametrize("bad", ["", "a" * 39, "A" * 40, "g" * 40])
def test_selected_execution_head_rejects_malformed_sha(bad: str) -> None:
    mod = _load()
    with pytest.raises(mod.NbuGlobalDedupError, match="exact lowercase 40-hex"):
        mod._bind_execution_head(bad)


def test_selected_execution_head_rejects_stale_or_synthetic_checkout(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(
        mod,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="b" * 40 + "\n",
            stderr="",
        ),
    )
    with pytest.raises(mod.NbuGlobalDedupError, match="execution HEAD drift"):
        mod._bind_execution_head("a" * 40)


def test_compose_graph_is_one_to_one_and_collision_safe() -> None:
    mod = _load()
    base_inventory = {"sources": [{"source_id": "base:a"}]}
    base_payloads = {"base:a": b"a"}
    extension_sources = [{"source_id": "nbu:a"}]
    extension_payloads = {"nbu:a": b"b"}

    inventory, payloads = mod._compose_graph(
        base_inventory,
        base_payloads,
        extension_sources,
        extension_payloads,
    )
    assert [row["source_id"] for row in inventory["sources"]] == ["base:a", "nbu:a"]
    assert payloads == {"base:a": b"a", "nbu:a": b"b"}
    assert base_inventory == {"sources": [{"source_id": "base:a"}]}
    assert base_payloads == {"base:a": b"a"}


def test_compose_graph_rejects_source_id_collision() -> None:
    mod = _load()
    with pytest.raises(mod.NbuGlobalDedupError, match="source-id collision"):
        mod._compose_graph(
            {"sources": [{"source_id": "same"}]},
            {"same": b"a"},
            [{"source_id": "same"}],
            {"same": b"b"},
        )


def _report(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 10
    return {
        "report_sha256": "1" * 64,
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "sources": [
            {
                "source_id": "base:a",
                "source_family": "base",
                "declared_capacity_bytes": after,
            },
            {
                "source_id": "nbu:a",
                "source_family": mod.nbu.SOURCE_FAMILY,
                "declared_capacity_bytes": 10,
            },
        ],
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": after,
            "duplicate_discount_bytes": 10,
            "duplicate_cluster_count": 1,
        },
    }


def _projection(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 10
    return {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": "1" * 64,
        "survivor_authority_sha256": "2" * 64,
        "pre_dedup_source_object_count": mod.EXPECTED_COMBINED_OBJECTS,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": mod.EXPECTED_COMBINED_BYTES,
        "post_dedup_declared_capacity_bytes": after,
        "duplicate_discount_bytes": 10,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [{"selected_source_id": "base:a"}],
        "survivor_source_ids": ["base:a", "nbu:a"],
    }


def test_survivor_projection_cross_binds_terminal_report() -> None:
    mod = _load()
    report = _report(mod)
    projection = _projection(mod)
    mod._validate_survivor_projection(report, projection)
    outer = mod._outer_survivor_authority(report, projection)
    assert outer["nbu_survivor_source_ids"] == ["nbu:a"]
    assert outer["nbu_survivor_source_object_count"] == 1
    assert outer["nbu_survivor_declared_capacity_bytes"] == 10
    assert outer["canonical_capacity_credited"] == 0
    assert len(outer["survivor_authority_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("matcher_report_sha256", "f" * 64, "matcher identity"),
        ("pre_dedup_source_object_count", 1, "pre-dedup count"),
        ("pre_dedup_declared_capacity_bytes", 1, "pre-dedup bytes"),
        ("post_dedup_declared_capacity_bytes", 1, "post-dedup bytes"),
        ("duplicate_discount_bytes", 9, "duplicate discount"),
        ("duplicate_cluster_count", 0, "duplicate cluster count"),
    ],
)
def test_survivor_projection_rejects_terminal_drift(
    field: str,
    bad: object,
    message: str,
) -> None:
    mod = _load()
    report = _report(mod)
    projection = _projection(mod)
    projection[field] = bad
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._validate_survivor_projection(report, projection)


def test_runtime_environment_local_is_provider_neutral(monkeypatch) -> None:
    mod = _load()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    observed = mod._runtime_environment()
    assert observed["github_actions"] is False
    assert observed["runner_environment"] == "local"


def test_runtime_environment_rejects_ambiguous_actions_runner(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("RUNNER_ENVIRONMENT", raising=False)
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    with pytest.raises(mod.NbuGlobalDedupError, match="runner environment missing"):
        mod._runtime_environment()


def test_publish_is_create_only(tmp_path) -> None:
    mod = _load()
    output = tmp_path / "evidence.json"
    mod._publish_json_outputs(((output, {"ok": True}),))
    assert json.loads(output.read_text()) == {"ok": True}
    with pytest.raises(mod.NbuGlobalDedupError, match="refusing to overwrite"):
        mod._publish_json_outputs(((output, {"ok": False}),))


def test_publish_rolls_back_files_from_failed_publication(tmp_path) -> None:
    mod = _load()
    first = tmp_path / "first.json"
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(mod.NbuGlobalDedupError, match="refusing to overwrite"):
        mod._publish_json_outputs(
            (
                (first, {"one": 1}),
                (existing, {"two": 2}),
            )
        )
    assert not first.exists()
    assert existing.read_text(encoding="utf-8") == "{}\n"


def test_execute_binds_head_and_authority_before_reconstruction() -> None:
    mod = _load()
    source = MODULE.read_text(encoding="utf-8")
    head = source.index("execution_head = _bind_execution_head(expected_execution_head)")
    authority = source.index("main_blobs, product_blobs = verify_repository_authority()")
    reconstruction = source.index("matcher, base_inventory, base_payloads =")
    assert head < authority < reconstruction


def test_production_requires_reference_and_indexed_equivalence() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "matcher.audit_payloads(inventory, payloads)" in source
    assert "indexed.audit_payloads_indexed(" in source
    assert "indexed.attest_incumbent_runtime(matcher)" in source
    assert "indexed/reference report mismatch" in source


def test_reference_indexed_mismatch_is_rejected(monkeypatch) -> None:
    mod = _load()

    class Matcher:
        def audit_payloads(self, inventory, payloads):
            return {"report_sha256": "1" * 64, "sources": []}

        def verify_report(self, report):
            assert isinstance(report, dict)

    monkeypatch.setattr(mod.indexed, "attest_incumbent_runtime", lambda matcher: None)
    monkeypatch.setattr(
        mod.indexed,
        "audit_payloads_indexed",
        lambda matcher, inventory, payloads, **kwargs: {
            "report_sha256": "2" * 64,
            "sources": [],
        },
    )
    with pytest.raises(mod.NbuGlobalDedupError, match="indexed/reference report mismatch"):
        mod._execute_equivalent_matchers(
            Matcher(),
            {"sources": []},
            {},
            max_candidate_pairs=10,
            max_index_postings=10,
            max_pair_expansions=10,
        )
