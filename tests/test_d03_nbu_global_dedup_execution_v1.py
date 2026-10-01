from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
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
    core = {
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
    return {**core, "report_sha256": mod._sha256(mod._canonical(core))}


def _projection(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 10
    return {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": _report(mod)["report_sha256"],
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


def test_survivor_projection_rejects_empty_or_oversized_survivor_set() -> None:
    mod = _load()
    report = _report(mod)
    for ids in ([], [f"source:{index}" for index in range(mod.EXPECTED_COMBINED_OBJECTS + 1)]):
        projection = _projection(mod)
        projection["survivor_source_ids"] = ids
        projection["post_dedup_survivor_source_object_count"] = len(ids)
        with pytest.raises(mod.NbuGlobalDedupError, match="survivor count drift"):
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


def test_production_executes_only_independently_qualified_indexed_path() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "indexed.attest_incumbent_runtime(matcher)" in source
    assert "indexed.audit_payloads_indexed(" in source
    assert "matcher.audit_payloads(inventory, payloads)" not in source
    assert "all_pairs_reference_executed\": False" in source


def test_indexed_execution_attests_and_verifies_report(monkeypatch) -> None:
    mod = _load()
    calls: list[str] = []

    class Matcher:
        def verify_report(self, report):
            calls.append("verify")
            assert report["report_sha256"] == "2" * 64

    matcher = Matcher()
    monkeypatch.setattr(
        mod.indexed,
        "attest_incumbent_runtime",
        lambda observed: calls.append("attest") if observed is matcher else None,
    )
    monkeypatch.setattr(
        mod.indexed,
        "audit_payloads_indexed",
        lambda observed, inventory, payloads, **kwargs: (
            calls.append("indexed"),
            {"report_sha256": "2" * 64, "sources": []},
        )[1],
    )
    report, elapsed = mod._execute_indexed_matcher(
        matcher,
        {"sources": []},
        {},
        max_candidate_pairs=10,
        max_index_postings=10,
        max_pair_expansions=10,
    )
    assert report["report_sha256"] == "2" * 64
    assert elapsed >= 0
    assert calls == ["attest", "indexed", "verify"]



def _two_clean_evidence(mod, marker: str) -> dict[str, object]:
    core: dict[str, object] = {
        "execution_head_sha": "a" * 40,
        "pinned_main_sha": mod.EXPECTED_MAIN,
        "nbu": {"candidate_sha256": mod.nbu.CANDIDATE_SHA256},
        "matcher_execution": {
            "engine": "MERGED_PR_1459",
            "report_sha256": _report(mod)["report_sha256"],
            "test_marker": marker,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "authorized_optimized_target_exposure": 0,
            "training_executed": False,
        },
    }
    return {
        **core,
        "evidence_identity_sha256": mod._sha256(mod._canonical(core)),
    }


def _two_clean_survivors(mod) -> dict[str, object]:
    core: dict[str, object] = {
        "matcher_report_sha256": "1" * 64,
        "nbu_survivor_source_object_count": 1,
        "nbu_survivor_declared_capacity_bytes": 10,
    }
    return {
        **core,
        "survivor_authority_sha256": mod._sha256(mod._canonical(core)),
    }


def test_two_clean_authority_is_zero_credit_and_does_not_claim_source_replay() -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    authority = mod._build_two_clean_authority(
        report,
        deepcopy(report),
        survivors,
        deepcopy(survivors),
        _two_clean_evidence(mod, "run-a"),
        _two_clean_evidence(mod, "run-b"),
    )
    assert authority["dedup"]["fresh_process_count"] == 2
    assert authority["dedup"]["report_sha256"] == report["report_sha256"]
    assert len(authority["dedup"]["survivor_authority_sha256"]) == 64
    assert authority["materialization_authority"]["distinct_input_copies_required"] is True
    assert authority["materialization_authority"]["source_replay_executed_by_this_carrier"] is False
    assert authority["truth_boundary"]["canonical_capacity_credited"] == 0
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert authority["truth_boundary"]["training_executed"] is False
    assert len(authority["two_clean_authority_sha256"]) == 64


def test_two_clean_authority_rejects_self_consistent_pair_with_bad_report_hash() -> None:
    mod = _load()
    report = _report(mod)
    report["report_sha256"] = "f" * 64
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    with pytest.raises(mod.NbuGlobalDedupError, match="report self-hash mismatch"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )


def test_two_clean_authority_rejects_report_or_survivor_drift() -> None:
    mod = _load()
    report = _report(mod)
    changed_report = deepcopy(report)
    changed_report["source_count"] = mod.EXPECTED_COMBINED_OBJECTS - 1
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "3" * 64)
    evidence_b = _two_clean_evidence(mod, "4" * 64)
    with pytest.raises(mod.NbuGlobalDedupError, match="two-clean dedup reports differ"):
        mod._build_two_clean_authority(
            report,
            changed_report,
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )

    changed_survivors = deepcopy(survivors)
    changed_survivors["nbu_survivor_declared_capacity_bytes"] = 11
    with pytest.raises(mod.NbuGlobalDedupError, match="two-clean survivor authorities differ"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            changed_survivors,
            evidence_a,
            evidence_b,
        )


def test_two_clean_requires_non_aliasing_materialization_copies(tmp_path) -> None:
    mod = _load()
    candidate = tmp_path / "candidate.jsonl"
    evidence = tmp_path / "evidence.json"
    candidate.write_text("{}\n", encoding="utf-8")
    evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(mod.NbuGlobalDedupError, match="candidate paths must be distinct"):
        mod._require_distinct_materialization_copies(
            candidate,
            evidence,
            candidate,
            evidence,
        )


def test_two_clean_rejects_symlink_materialization_input(tmp_path) -> None:
    mod = _load()
    candidate_a = tmp_path / "candidate-a.jsonl"
    candidate_b = tmp_path / "candidate-b.jsonl"
    evidence_a = tmp_path / "evidence-a.json"
    evidence_b = tmp_path / "evidence-b.json"
    for path in (candidate_a, candidate_b, evidence_a, evidence_b):
        path.write_text("{}\n", encoding="utf-8")
    candidate_link = tmp_path / "candidate-link.jsonl"
    candidate_link.symlink_to(candidate_a)
    with pytest.raises(mod.NbuGlobalDedupError, match="must not be symlink"):
        mod._require_distinct_materialization_copies(
            candidate_link,
            evidence_a,
            candidate_b,
            evidence_b,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"x":1,"x":2}', "duplicate generated JSON key"),
        (b'{"x":NaN}', "non-finite generated JSON constant"),
        (b'{"x":1e400}', "non-finite generated JSON number"),
    ],
)
def test_two_clean_generated_json_is_strict(
    tmp_path, payload: bytes, message: str
) -> None:
    mod = _load()
    path = tmp_path / "generated.json"
    path.write_bytes(payload)
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._strict_generated_json(path)


def test_two_clean_incomplete_is_zero_authority(tmp_path) -> None:
    mod = _load()
    root = tmp_path / "run"
    root.mkdir()
    mod._write_two_clean_incomplete(root, ["clean-a"], "worker_timeout")
    value = json.loads((root / "incomplete.json").read_text(encoding="utf-8"))
    assert value["status"] == "INCOMPLETE_NO_TWO_CLEAN_AUTHORITY"
    assert value["completed_run_ids"] == ["clean-a"]
    assert value["canonical_capacity_credited"] == 0
    assert value["authorized_optimized_target_exposure"] == 0
    assert value["training_executed"] is False
    assert not (root / "two-clean-authority.json").exists()
