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


def _nbu_comparison_binding_fixture(mod):
    payload = b"nbu payload"
    expected = {
        "source_id": "nbu:test",
        "source_family": mod.nbu.SOURCE_FAMILY,
        "modality": "natural_language",
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": len(payload),
        "stable_origin_id": "https://bank.gov.ua/ua/legislation/Resolution_20260101_test",
        "stable_object_id": "sha256:" + mod._sha256(payload),
    }
    observed = {
        "source_id": expected["source_id"],
        "source_family": expected["source_family"],
        "modality": expected["modality"],
        "evidence_status": expected["evidence_status"],
        "declared_capacity_bytes": expected["declared_capacity_bytes"],
        "stable_origin_id_sha256": mod._sha256(expected["stable_origin_id"].encode("utf-8")),
        "stable_object_id_sha256": mod._sha256(expected["stable_object_id"].encode("utf-8")),
        "verified_raw_bytes": len(payload),
        "verified_raw_sha256": mod._sha256(payload),
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": mod._sha256(payload),
    }
    return expected, payload, observed


def test_nbu_report_binding_accepts_exact_generic_raw_projection() -> None:
    mod = _load()
    expected, payload, observed = _nbu_comparison_binding_fixture(mod)
    mod._validate_nbu_report_binding(
        {"sources": [observed]},
        [expected],
        {expected["source_id"]: payload},
    )


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("comparison_policy", "OTHER", "comparison policy drift"),
        ("comparison_payload_bytes", 1, "comparison byte drift"),
        ("comparison_payload_sha256", "f" * 64, "comparison hash drift"),
    ],
)
def test_nbu_report_binding_rejects_comparison_drift(
    field: str, bad: object, message: str
) -> None:
    mod = _load()
    expected, payload, observed = _nbu_comparison_binding_fixture(mod)
    observed[field] = bad
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._validate_nbu_report_binding(
            {"sources": [observed]},
            [expected],
            {expected["source_id"]: payload},
        )


def _nbu_binding_fixture(mod):
    payload = b"abc"
    stable_origin_id = "https://bank.gov.ua/example#pdf-sha256:" + "1" * 64
    stable_object_id = "sha256:" + "2" * 64
    source = {
        "source_id": "nbu-admitted:test",
        "source_family": mod.nbu.SOURCE_FAMILY,
        "stable_origin_id": stable_origin_id,
        "stable_object_id": stable_object_id,
        "modality": mod.nbu.MATCHER_MODALITY,
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": len(payload),
    }
    report_row = {
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "stable_origin_id_sha256": mod._sha256(stable_origin_id.encode("utf-8")),
        "stable_object_id_sha256": mod._sha256(stable_object_id.encode("utf-8")),
        "modality": source["modality"],
        "evidence_status": source["evidence_status"],
        "declared_capacity_bytes": len(payload),
        "verified_raw_bytes": len(payload),
        "verified_raw_sha256": mod._sha256(payload),
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": mod._sha256(payload),
    }
    return {"sources": [report_row]}, [source], {source["source_id"]: payload}


def test_nbu_report_binding_cross_binds_exact_projected_source() -> None:
    mod = _load()
    report, sources, payloads = _nbu_binding_fixture(mod)
    mod._validate_nbu_report_binding(report, sources, payloads)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("stable_origin_id_sha256", "f" * 64, "stable origin drift"),
        ("verified_raw_sha256", "f" * 64, "verified raw hash drift"),
        ("comparison_payload_sha256", "f" * 64, "comparison hash drift"),
    ],
)
def test_nbu_report_binding_rejects_report_drift(
    field: str, bad: object, message: str
) -> None:
    mod = _load()
    report, sources, payloads = _nbu_binding_fixture(mod)
    report["sources"][0][field] = bad
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._validate_nbu_report_binding(report, sources, payloads)


def test_nbu_report_binding_rejects_duplicate_projected_ids_and_orphan_payload() -> None:
    mod = _load()
    report, sources, payloads = _nbu_binding_fixture(mod)
    duplicate_sources = [sources[0], deepcopy(sources[0])]
    orphan_payloads = {**payloads, "orphan": b"x"}
    with pytest.raises(mod.NbuGlobalDedupError, match="source ids duplicate"):
        mod._validate_nbu_report_binding(report, duplicate_sources, orphan_payloads)


def test_nbu_report_binding_rejects_nonstring_stable_identity() -> None:
    mod = _load()
    report, sources, payloads = _nbu_binding_fixture(mod)
    sources[0]["stable_origin_id"] = None
    with pytest.raises(mod.NbuGlobalDedupError, match="stable origin invalid"):
        mod._validate_nbu_report_binding(report, sources, payloads)


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
    return {**core, "report_sha256": mod._incumbent_report_identity(core)}


def _projection(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 10
    core = {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": _report(mod)["report_sha256"],
        "pre_dedup_source_object_count": mod.EXPECTED_COMBINED_OBJECTS,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": mod.EXPECTED_COMBINED_BYTES,
        "post_dedup_declared_capacity_bytes": after,
        "duplicate_discount_bytes": 10,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [{"selected_source_id": "base:a"}],
        "survivor_source_ids": ["base:a", "nbu:a"],
    }
    return {
        **core,
        "survivor_authority_sha256": mod._sha256(mod._canonical(core)),
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


def test_survivor_projection_rejects_forged_self_hash() -> None:
    mod = _load()
    projection = _projection(mod)
    projection["survivor_authority_sha256"] = "f" * 64
    with pytest.raises(mod.NbuGlobalDedupError, match="projection self-hash mismatch"):
        mod._validate_survivor_projection(_report(mod), projection)


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


def test_two_clean_survivor_readback_rederives_canonical_selection(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_OBJECTS", 2)
    monkeypatch.setattr(mod, "EXPECTED_COMBINED_BYTES", 20)
    core = {
        "source_count": 2,
        "sources": [
            {
                "source_id": "base:a",
                "source_family": "base",
                "declared_capacity_bytes": 10,
            },
            {
                "source_id": "nbu:a",
                "source_family": mod.nbu.SOURCE_FAMILY,
                "declared_capacity_bytes": 10,
            },
        ],
        "terminal_candidates": {
            "declared_capacity_bytes_before": 20,
            "conservative_unique_capacity_bytes_after": 10,
            "duplicate_discount_bytes": 10,
            "duplicate_cluster_count": 1,
            "duplicate_clusters": [["base:a", "nbu:a"]],
        },
    }
    report = {**core, "report_sha256": mod._incumbent_report_identity(core)}
    selection = mod.v9_semantics._derive_survivors(report)
    mod._validate_survivor_projection(report, selection)
    survivor = mod._outer_survivor_authority(report, selection)
    mod._validate_two_clean_survivor_readback(report, survivor)

    forged = deepcopy(survivor)
    forged["nbu_survivor_source_ids"] = ["nbu:a"]
    forged["nbu_survivor_source_object_count"] = 1
    forged["nbu_survivor_declared_capacity_bytes"] = 10
    forged_core = dict(forged)
    forged_core.pop("survivor_authority_sha256")
    forged["survivor_authority_sha256"] = mod._sha256(mod._canonical(forged_core))
    with pytest.raises(mod.NbuGlobalDedupError, match="survivor semantic readback drift"):
        mod._validate_two_clean_survivor_readback(report, forged)


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
    report_sha = _report(mod)["report_sha256"]
    survivor_sha = _two_clean_survivors(mod)["survivor_authority_sha256"]
    core: dict[str, object] = {
        "schema_version": mod.SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_claim_issue": mod.EXECUTION_CLAIM,
        "execution_pr": mod.EXECUTION_PR,
        "execution_head_sha": "a" * 40,
        "pinned_main_sha": mod.EXPECTED_MAIN,
        "baseline_v8": {
            "v7_head_sha": mod.v8.EXPECTED_V7_HEAD,
            "source_object_count": mod.EXPECTED_BASE_OBJECTS,
            "payload_bytes": mod.EXPECTED_BASE_BYTES,
        },
        "nbu": {
            "materialization_head": mod.nbu.MATERIALIZATION_HEAD,
            "workflow_run_id": mod.nbu.MATERIALIZATION_RUN,
            "workflow_job_id": mod.nbu.MATERIALIZATION_JOB,
            "artifact_id": mod.nbu.MATERIALIZATION_ARTIFACT,
            "independent_audit_issue": mod.nbu.MATERIALIZATION_AUDIT,
            "candidate_sha256": mod.nbu.CANDIDATE_SHA256,
            "source_object_count": mod.EXPECTED_NBU_OBJECTS,
            "payload_bytes": mod.EXPECTED_NBU_BYTES,
            "intake_receipt_identity_sha256": "5" * 64,
        },
        "combined": {
            "source_object_count": mod.EXPECTED_COMBINED_OBJECTS,
            "payload_bytes": mod.EXPECTED_COMBINED_BYTES,
            "indexed_report_sha256": report_sha,
        },
        "matcher_execution": {
            "engine": "MERGED_PR_1459",
            "performance_equivalence_authority": "MERGED_PR_1459",
            "incumbent_runtime_attested": True,
            "report_sha256": report_sha,
            "all_pairs_reference_executed": False,
            "test_marker": marker,
        },
        "survivor_authority_sha256": survivor_sha,
        "content_boundary": {
            "raw_text_emitted": False,
            "raw_candidate_written_to_durable_evidence": False,
            "dedup_report_text_free": True,
            "survivor_authority_text_free": True,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    return {
        **core,
        "evidence_identity_sha256": mod._sha256(mod._canonical(core)),
    }


def _two_clean_survivors(mod) -> dict[str, object]:
    core: dict[str, object] = {
        "matcher_report_sha256": _report(mod)["report_sha256"],
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


def test_two_clean_authority_rejects_nonhex_report_identity() -> None:
    mod = _load()
    report = _report(mod)
    report["report_sha256"] = "g" * 64
    with pytest.raises(mod.NbuGlobalDedupError, match="report identity drift"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            _two_clean_survivors(mod),
            _two_clean_survivors(mod),
            _two_clean_evidence(mod, "run-a"),
            _two_clean_evidence(mod, "run-b"),
        )


def test_two_clean_authority_rejects_nonhex_execution_head_even_when_rehashed() -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    for evidence in (evidence_a, evidence_b):
        evidence["execution_head_sha"] = "z" * 40
        core = dict(evidence)
        core.pop("evidence_identity_sha256")
        evidence["evidence_identity_sha256"] = mod._sha256(mod._canonical(core))
    with pytest.raises(mod.NbuGlobalDedupError, match="execution head drift"):
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


def test_two_clean_authority_rejects_tampered_survivor_self_hash() -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    tampered = deepcopy(survivors)
    tampered["survivor_authority_sha256"] = "f" * 64
    with pytest.raises(mod.NbuGlobalDedupError, match="survivor authorities differ"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            tampered,
            _two_clean_evidence(mod, "run-a"),
            _two_clean_evidence(mod, "run-b"),
        )

    both_tampered = deepcopy(survivors)
    both_tampered["survivor_authority_sha256"] = "f" * 64
    with pytest.raises(mod.NbuGlobalDedupError, match="survivor self-hash mismatch"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            both_tampered,
            deepcopy(both_tampered),
            _two_clean_evidence(mod, "run-a"),
            _two_clean_evidence(mod, "run-b"),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("combined", "combined report identity drift"),
        ("survivor", "evidence/survivor identity drift"),
        ("receipt", "intake receipt identities differ"),
    ],
)
def test_two_clean_authority_cross_binds_child_outputs(field: str, message: str) -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    if field == "combined":
        evidence_b["combined"]["indexed_report_sha256"] = "f" * 64
    elif field == "survivor":
        evidence_b["survivor_authority_sha256"] = "f" * 64
    else:
        evidence_b["nbu"]["intake_receipt_identity_sha256"] = "6" * 64
    core = dict(evidence_b)
    core.pop("evidence_identity_sha256")
    evidence_b["evidence_identity_sha256"] = mod._sha256(mod._canonical(core))
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )


def test_two_clean_truth_zero_fields_reject_bool_alias() -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    evidence_b["truth_boundary"]["canonical_capacity_credited"] = False
    core = dict(evidence_b)
    core.pop("evidence_identity_sha256")
    evidence_b["evidence_identity_sha256"] = mod._sha256(mod._canonical(core))
    with pytest.raises(
        mod.NbuGlobalDedupError,
        match="truth boundary drift: canonical_capacity_credited",
    ):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )


@pytest.mark.parametrize(
    ("section", "field", "bad", "message"),
    [
        ("truth_boundary", "paid_compute_used", True, "truth boundary drift: paid_compute_used"),
        (
            "truth_boundary",
            "foreign_pretrained_weights_used",
            True,
            "truth boundary drift: foreign_pretrained_weights_used",
        ),
        ("content_boundary", "raw_text_emitted", True, "child content boundary drift"),
        (
            "matcher_execution",
            "all_pairs_reference_executed",
            True,
            "matcher authority drift",
        ),
    ],
)
def test_two_clean_authority_rejects_rehashed_child_authority_laundering(
    section: str, field: str, bad: object, message: str
) -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    evidence_b[section][field] = bad
    core = dict(evidence_b)
    core.pop("evidence_identity_sha256")
    evidence_b["evidence_identity_sha256"] = mod._sha256(mod._canonical(core))
    with pytest.raises(mod.NbuGlobalDedupError, match=message):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )


def test_two_clean_authority_rejects_tampered_execution_evidence() -> None:
    mod = _load()
    report = _report(mod)
    survivors = _two_clean_survivors(mod)
    evidence_a = _two_clean_evidence(mod, "run-a")
    evidence_b = _two_clean_evidence(mod, "run-b")
    evidence_b["truth_boundary"]["training_executed"] = True
    with pytest.raises(mod.NbuGlobalDedupError, match="run evidence self-hash mismatch"):
        mod._build_two_clean_authority(
            report,
            deepcopy(report),
            survivors,
            deepcopy(survivors),
            evidence_a,
            evidence_b,
        )


def test_parent_aggregate_binding_cross_binds_head_and_intake_receipt() -> None:
    mod = _load()
    authority = {
        "execution_head_sha": "a" * 40,
        "materialization_authority": {
            "intake_receipt_identity_sha256": "b" * 64,
        },
    }
    mod._validate_parent_aggregate_binding(
        authority,
        orchestration_head="a" * 40,
        expected_intake_receipt_sha="b" * 64,
    )

    with pytest.raises(mod.NbuGlobalDedupError, match="execution head drift"):
        mod._validate_parent_aggregate_binding(
            {**authority, "execution_head_sha": "c" * 40},
            orchestration_head="a" * 40,
            expected_intake_receipt_sha="b" * 64,
        )

    forged = deepcopy(authority)
    forged["materialization_authority"]["intake_receipt_identity_sha256"] = "d" * 64
    with pytest.raises(mod.NbuGlobalDedupError, match="parent/child intake receipt drift"):
        mod._validate_parent_aggregate_binding(
            forged,
            orchestration_head="a" * 40,
            expected_intake_receipt_sha="b" * 64,
        )


def test_two_clean_binds_parent_head_before_input_preflight(tmp_path, monkeypatch) -> None:
    mod = _load()

    def reject_head(expected: str) -> str:
        raise mod.NbuGlobalDedupError(f"parent-head-sentinel:{expected}")

    monkeypatch.setattr(mod, "_bind_execution_head", reject_head)
    with pytest.raises(mod.NbuGlobalDedupError, match="parent-head-sentinel"):
        mod.run_two_clean(
            v7_root=tmp_path / "missing-v7",
            bulk_workspace=tmp_path / "missing-bulk",
            candidate_jsonl_a=tmp_path / "missing-a.jsonl",
            materialization_evidence_json_a=tmp_path / "missing-a.json",
            candidate_jsonl_b=tmp_path / "missing-b.jsonl",
            materialization_evidence_json_b=tmp_path / "missing-b.json",
            output_root=tmp_path / "out",
            expected_execution_head="a" * 40,
            max_candidate_pairs=1,
            max_index_postings=1,
            max_pair_expansions=1,
        )


def test_two_clean_workers_use_isolated_bulk_workspaces() -> None:
    source = MODULE.read_text(encoding="utf-8")
    start = source.index("def run_two_clean(")
    end = source.index("\ndef execute(", start)
    run_two_clean_source = source[start:end]
    assert 'str(run_dir / "bulk-workspace")' in run_two_clean_source
    assert "str(bulk_workspace)" not in run_two_clean_source


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
