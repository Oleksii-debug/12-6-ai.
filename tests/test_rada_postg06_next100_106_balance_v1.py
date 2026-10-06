from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_d03_rada_postg06_next100_106_balance_v1 as target


UPSTREAM_EVIDENCE_RAW = b'''{"authorized_optimized_target_exposure":0,"authorized_unique_loss_positions":0,"candidate_filter":{"algorithm":"INCUMBENT_V1_CONSERVATIVE_RARE_PREFIX_V1","bounded_differential_selftest":{"fixture_groups":9,"incumbent_positive_pairs":9,"prefix_candidate_pairs":9},"lineage_and_report_authority":"QUALIFIED_INCUMBENT_INDEXED_EXECUTOR","pair_decision_authority":"EXACT_INCUMBENT_V1_PAIR_MATCHES","science_changed":false,"work_telemetry":{"algorithm":"INCUMBENT_V1_CONSERVATIVE_RARE_PREFIX_V1","candidate_limit":100000000,"edge_prefix_sources":93807,"exact_bucket_signatures":747,"frequency_scan_items":17440032,"index_posting_limit":250000000,"index_postings":17746017,"pair_expansion_attempts":738618,"pair_expansion_limit":250000000,"prefix_queries":2649276,"source_count":101995,"unique_candidate_pairs":382329}},"candidate_jsonl_sha256":"8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7","candidate_payload_bytes":192393157,"candidate_source_object_count":101733,"canonical_capacity_credited":0,"combined_declared_capacity_bytes":198398557,"combined_source_object_count":101995,"current_rada_survivor_declared_capacity_bytes":186855914,"current_rada_survivor_source_object_count":98601,"evidence_identity_sha256":"a3f7e396cb13a6b107aaa0eb330edcbdc61bead6fa12eb68542ff5c3a3ed9301","execution_head_sha":"a4663e87b010b190343caf1d42784f5dc7984601","final_test_outcomes_read":false,"learned_weights_created":false,"matcher_model_training_executed":false,"matcher_report_sha256":"64e687ae431804862003d5839b9a90c715838e794daad907200f0abfc73b4333","matcher_source_admission_authority":false,"nbu_helper_blob_sha1":"dc0ed88924610fc7ac19ba4fc7d960d8679c9742","nbu_helper_head_sha":"b5235cfd83852854e345dea6c2e89cfbcd1cef79","next_gate":"CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING","optimizer_updates_executed_on_real_targets":0,"paid_compute_used":false,"parent_product_head_sha":"0294dd8f04e28cd7992f2afce4cdb3af39846d98","parent_replay_authority_identity_sha256":"543f9cdd5a9aacaf2cc00b5d4057ad8b142aa685870545cff8bd518f8085055e","projection_receipt_identity_sha256":"d439cceebaf5b7a5ef365fca8c20fc9e5ad873435513380793d3020fad1c28e7","raw_text_persisted":false,"replacement_proof":{"append_forbidden_by_current_adapter":true,"matcher_only_status_projection":{"canonical_capacity_credited":0,"input_evidence_status":"CURRENT_SNAPSHOT_REPLAY_ZERO_CREDIT","matcher_evidence_status":"DEDICATED_TERMINAL","projected_source_object_count":101733,"projection_scope":"GLOBAL_DEDUP_MATCHER_INPUT_ONLY","source_admission_authority_granted":false,"tokenizer_fit_authorized":false,"training_authority_granted":false,"training_authorized_bytes":0},"replaced_declared_capacity_bytes":88565,"replaced_raw_bytes":332400,"replaced_raw_sha256":"36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4","replaced_source_ids":["ua.rada.open-data.laws-texts.d23314"],"replaced_source_object_count":1,"replacement_required_by_current_adapter":true,"source_family":"ua.rada.open-data.laws-texts"},"rights_recheck_for_training_required":true,"rights_scope":"ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY","scale_promotion_authorized":false,"schema_version":"12-6.d03-rada-current-global-dedup-execution.v1","survivor_authority_sha256":"f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb","tokenizer_fit_authorized":false,"training_authorized_bytes":0,"training_executed":false,"v7_head_sha":"d3333ec1b4a508df232a5aefccd6686adda745fb"}\n'''
UPSTREAM_TWO_CLEAN_RAW = b'''{"byte_identical_outputs":true,"canonical_capacity_credited":0,"current_rada_survivor_declared_capacity_bytes":186855914,"current_rada_survivor_source_object_count":98601,"execution_head_sha":"a4663e87b010b190343caf1d42784f5dc7984601","fresh_process_count":2,"matcher_report_sha256":"64e687ae431804862003d5839b9a90c715838e794daad907200f0abfc73b4333","next_gate":"CURRENT_RADA_RIGHTS_PROVENANCE_RECHECK_FOR_TRAINING","output_file_sha256":{"dedup-report.json":"9a38961a5dc5bc7ef93ba046fdfb4f22ccddc175b8b028c875316b3f80324c1f","execution-evidence.json":"43a9621ab11fd82232e8251ab26aae4126cf0b4854878af1e97184d587740caa","rada-survivor-authority.json":"ebdb68e689625e2f46e8a44a02b9d38ffafd9e79a04dbdc49bf6f12b5dcfb6c9"},"rights_recheck_for_training_required":true,"schema_version":"12-6.d03-rada-current-global-dedup-two-clean.v1","survivor_authority_sha256":"f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb","tokenizer_fit_authorized":false,"training_authorized_bytes":0,"two_clean_identity_sha256":"f24f4b2d23bee6cb1273a4680297d942aa59030dab50d5c5de7a4d659ff99a5e"}\n'''


def _family_vector() -> dict[str, object]:
    core: dict[str, object] = {
        "schema": target.FAMILY_VECTOR_SCHEMA,
        "status": "PASS",
        "source_git_sha": "a" * 40,
        "materialization_identity_sha256": "1" * 64,
        "materialization_execution_head_sha": "b" * 40,
        "record_payload_jsonl_sha256": "2" * 64,
        "record_inventory_digest_sha256": "3" * 64,
        "payload_inventory_digest_sha256": "4" * 64,
        "record_count": 6,
        "total_payload_bytes": 6000,
        "source_object_count": 6,
        "record_membership_sha256": "5" * 64,
        "trusted_family_authority_root_sha256": "6" * 64,
        "families": [
            {
                "stratum": "uk",
                "family": "ua.rada.open-data.laws-texts",
                "record_count": 2,
                "capacity_bytes": 2000,
            },
            {
                "stratum": "en",
                "family": "en.standardebooks.manual",
                "record_count": 2,
                "capacity_bytes": 2000,
            },
            {
                "stratum": "code",
                "family": "github:encode/httpx",
                "record_count": 2,
                "capacity_bytes": 2000,
            },
        ],
        "stratum_capacity_bytes": {"code": 2000, "en": 2000, "uk": 2000},
        "stratum_family_counts": {"code": 1, "en": 1, "uk": 1},
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "training_authorized_by_this_report": False,
    }
    return {
        **core,
        "family_vector_identity_sha256": target.sha256(target.canonical(core)),
    }


def _post_g06(vector: dict[str, object]) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": target.POST_G06_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": vector["materialization_execution_head_sha"],
        "parent": {
            "artifact_id": 123,
            "artifact_zip_sha256": "8" * 64,
        },
        "g05": {
            "execution_identity_sha256": "9" * 64,
        },
        "g06": {
            "execution_identity_sha256": "a" * 64,
            "exact_payload_collision_free": True,
            "unique_payload_count": vector["record_count"],
            "payload_set_identity_sha256": "7" * 64,
        },
        "durable_artifacts": {
            "g05_authority_file_sha256": "b" * 64,
            "g06_authority_file_sha256": "c" * 64,
            "survivor_inventory_file_sha256": "d" * 64,
        },
        "survivor_inventory": {
            "record_payload_jsonl_sha256": vector[
                "record_payload_jsonl_sha256"
            ],
            "record_count": vector["record_count"],
            "total_payload_bytes": vector["total_payload_bytes"],
            "record_inventory_digest_sha256": vector[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": vector[
                "payload_inventory_digest_sha256"
            ],
        },
        "counts": {},
        "authority_blobs": {},
        "content_boundary": {
            "raw_training_text_persisted": False,
            "raw_evaluation_text_persisted": False,
            "raw_survivor_text_persisted": False,
            "durable_output_text_free": True,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "training_executed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
        "next_gate": "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST",
    }
    evidence = {
        **core,
        "evidence_identity_sha256": target.sha256(target.canonical(core)),
    }
    vector["materialization_identity_sha256"] = evidence[
        "evidence_identity_sha256"
    ]
    vector_core = dict(vector)
    vector_core.pop("family_vector_identity_sha256")
    vector["family_vector_identity_sha256"] = target.sha256(
        target.canonical(vector_core)
    )
    return evidence


def _post_g06_two_clean(
    vector: dict[str, object],
    evidence: dict[str, object],
    *,
    evidence_file_sha256: str,
) -> dict[str, object]:
    parent = evidence["parent"]
    g05 = evidence["g05"]
    g06 = evidence["g06"]
    survivor = evidence["survivor_inventory"]
    artifacts = evidence["durable_artifacts"]
    core: dict[str, object] = {
        "schema_version": target.G05_G06_TWO_CLEAN_SCHEMA,
        "execution_head_sha": evidence["execution_head_sha"],
        "parent_execution_head_sha": target.PARENT_DATA232_HEAD,
        "parent_artifact_id": parent["artifact_id"],
        "parent_artifact_zip_sha256": parent["artifact_zip_sha256"],
        "fresh_execution_count": 2,
        "independent_runner_jobs": True,
        "byte_identical_outputs": True,
        "output_file_sha256": {
            target.G05_G06_BUNDLE_FILES["evidence"]: evidence_file_sha256,
            target.G05_G06_BUNDLE_FILES["quality"]: artifacts[
                "g05_authority_file_sha256"
            ],
            target.G05_G06_BUNDLE_FILES["privacy"]: artifacts[
                "g06_authority_file_sha256"
            ],
            target.G05_G06_BUNDLE_FILES["survivor_inventory"]: artifacts[
                "survivor_inventory_file_sha256"
            ],
        },
        "evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "g05_execution_identity_sha256": g05["execution_identity_sha256"],
        "g06_execution_identity_sha256": g06["execution_identity_sha256"],
        "record_payload_jsonl_sha256": survivor[
            "record_payload_jsonl_sha256"
        ],
        "record_inventory_digest_sha256": survivor[
            "record_inventory_digest_sha256"
        ],
        "payload_inventory_digest_sha256": survivor[
            "payload_inventory_digest_sha256"
        ],
        "payload_set_identity_sha256": g06["payload_set_identity_sha256"],
        "record_count": vector["record_count"],
        "total_payload_bytes": vector["total_payload_bytes"],
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "next_gate": target.G05_G06_NEXT_GATE,
    }
    return {
        **core,
        "proof_identity_sha256": target.sha256(target.canonical(core)),
    }


def test_known_global_dedup_terminal_artifact_is_accepted() -> None:
    evidence = target.load_json_bytes(UPSTREAM_EVIDENCE_RAW, "upstream")
    two_clean = target.load_json_bytes(UPSTREAM_TWO_CLEAN_RAW, "two-clean")
    target.verify_global_dedup(evidence, two_clean)


def test_known_global_dedup_tamper_fails_closed() -> None:
    evidence = target.load_json_bytes(UPSTREAM_EVIDENCE_RAW, "upstream")
    two_clean = target.load_json_bytes(UPSTREAM_TWO_CLEAN_RAW, "two-clean")
    evidence["training_authorized_bytes"] = 1
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="self-hash mismatch",
    ):
        target.verify_global_dedup(evidence, two_clean)


def test_known_artifact_transport_hashes_are_exact() -> None:
    assert target.sha256(UPSTREAM_EVIDENCE_RAW) == (
        target.UPSTREAM_GLOBAL_DEDUP_EVIDENCE_FILE_SHA256
    )
    assert target.sha256(UPSTREAM_TWO_CLEAN_RAW) == (
        target.UPSTREAM_TWO_CLEAN_FILE_SHA256
    )


def test_strict_json_rejects_duplicate_keys_and_nonfinite() -> None:
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="duplicate JSON key",
    ):
        target.load_json_bytes(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="non-finite",
    ):
        target.load_json_bytes(b'{"a":NaN}', "nan")


def test_post_g06_receipt_cross_binds_physical_vector() -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)
    normalized = target.verify_post_g06_receipt(
        evidence,
        vector,
        expected_evidence_identity_sha256=evidence[
            "evidence_identity_sha256"
        ],
    )
    assert normalized["unique_payload_count"] == vector["record_count"]
    assert normalized["payload_set_identity_sha256"] == "7" * 64


def test_post_g06_receipt_rejects_uniqueness_or_root_drift() -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)

    bad_unique = copy.deepcopy(evidence)
    bad_unique["g06"]["unique_payload_count"] = 5
    bad_core = dict(bad_unique)
    bad_core.pop("evidence_identity_sha256")
    bad_unique["evidence_identity_sha256"] = target.sha256(
        target.canonical(bad_core)
    )
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="unique payload count",
    ):
        target.verify_post_g06_receipt(
            bad_unique,
            vector,
            expected_evidence_identity_sha256=bad_unique[
                "evidence_identity_sha256"
            ],
        )

    bad_root = copy.deepcopy(evidence)
    bad_root["survivor_inventory"][
        "payload_inventory_digest_sha256"
    ] = "f" * 64
    bad_core = dict(bad_root)
    bad_core.pop("evidence_identity_sha256")
    bad_root["evidence_identity_sha256"] = target.sha256(
        target.canonical(bad_core)
    )
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="cross-bind drift",
    ):
        target.verify_post_g06_receipt(
            bad_root,
            vector,
            expected_evidence_identity_sha256=bad_root[
                "evidence_identity_sha256"
            ],
        )


def test_composition_proof_is_deterministic_terminal_and_zero_credit() -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)
    post = target.verify_post_g06_receipt(
        evidence,
        vector,
        expected_evidence_identity_sha256=evidence[
            "evidence_identity_sha256"
        ],
    )
    first = target.build_composition_proof(
        source_git_sha="c" * 40,
        family_vector=vector,
        post_g06=post,
    )
    second = target.build_composition_proof(
        source_git_sha="c" * 40,
        family_vector=copy.deepcopy(vector),
        post_g06=copy.deepcopy(post),
    )
    assert first == second
    assert first["terminal_verdict"] == "PASS"
    assert first["cross_transform_exact_payload_collision_free"] is True
    assert first["upstream_global_dedup"]["evidence_identity_sha256"] == (
        target.UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID
    )
    for field, expected in target.ZERO_CREDIT.items():
        assert first[field] == expected
    assert first["evidence_identity_sha256"] == target.self_hash(
        first, "evidence_identity_sha256"
    )


class _FakeGate:
    def validate_vector(self, value: dict) -> None:
        assert value["dedup_authority"]["terminal_verdict"] == "PASS"

    def evaluate(self, policy: dict, value: dict) -> dict:
        assert policy["policy_identity_sha256"] == target.POLICY_IDENTITY_SHA256
        result = {
            "schema_version": "12-6.next100-106-balance-gate-result.v1",
            "policy_identity_sha256": target.POLICY_IDENTITY_SHA256,
            "dedup_authority": value["dedup_authority"],
            "input_totals": value["totals"],
            "family_minimum": {
                "required_per_stratum": 2,
                "observed": {"ua": 1, "en": 1, "code": 1},
                "pass": False,
            },
            "maximum_feasible_total_source_bytes": 0,
            "maximum_feasible_stratum_bytes": {"ua": 0, "en": 0, "code": 0},
            "target_total_source_bytes": 20_000_000,
            "target_stratum_bytes": {
                "ua": 9_000_000,
                "en": 7_000_000,
                "code": 4_000_000,
            },
            "raw_capacity_by_stratum": {
                "ua": 2000,
                "en": 2000,
                "code": 2000,
            },
            "raw_gap_to_target_by_stratum": {
                "ua": 8_998_000,
                "en": 6_998_000,
                "code": 3_998_000,
            },
            "deterministic_maximum_allocation": [],
            "status": "BLOCKED_NO_NONZERO_POLICY_COMPLIANT_MIXTURE",
            "claim_boundary": {
                "authorized_training_exposure_loss_positions": 0,
                "corpus_identity": None,
                "shard_identity": None,
                "tokenizer_fit_authorized": False,
                "model_training_authorized": False,
                "paid_compute_authorized": False,
                "source_bytes_are_loss_positions": False,
            },
        }
        result["result_identity_sha256"] = target.sha256(
            target.canonical(result)
        )
        return result


class _FakeBridge:
    def verify_postmaterialization_family_vector(
        self, value: dict, *, expected_identity_sha256: str
    ) -> str:
        assert value["family_vector_identity_sha256"] == expected_identity_sha256
        return expected_identity_sha256

    def adapt_postmaterialization_family_vector_to_next100_106(
        self,
        family_vector: dict,
        *,
        expected_family_vector_identity_sha256: str,
        dedup_authority: dict,
        expected_dedup_worker_id: str,
        expected_dedup_head_sha: str,
        expected_dedup_evidence_identity_sha256: str,
    ) -> dict:
        assert expected_dedup_worker_id == target.DEDUP_WORKER_ID
        assert dedup_authority["head_sha"] == expected_dedup_head_sha
        assert dedup_authority["evidence_identity_sha256"] == (
            expected_dedup_evidence_identity_sha256
        )
        return {
            "schema_version": "12-6.next100-106-post-dedup-family-vector.v1",
            "terminal": True,
            "dedup_authority": dedup_authority,
            "families": [
                {
                    "family_id": row["family"],
                    "stratum": {"uk": "ua"}.get(
                        row["stratum"], row["stratum"]
                    ),
                    "unique_bytes": row["capacity_bytes"],
                }
                for row in family_vector["families"]
            ],
            "totals": {
                "total_unique_bytes": family_vector["total_payload_bytes"],
                "by_stratum": {"ua": 2000, "en": 2000, "code": 2000},
                "family_count": {"ua": 1, "en": 1, "code": 1},
            },
            "physical_authority": {},
        }

    def verify_balance_result(
        self, value: dict, *, expected_result_identity_sha256: str
    ) -> str:
        assert value["result_identity_sha256"] == expected_result_identity_sha256
        return expected_result_identity_sha256

    def build_balance_result_binding(self, **kwargs) -> dict:
        result = kwargs["balance_result"]
        return {
            "binding_identity_sha256": "d" * 64,
            "balance_status": result["status"],
        }


def _write_json(path: Path, value: dict) -> str:
    raw = target.canonical_line(value)
    path.write_bytes(raw)
    return target.sha256(raw)


def test_execute_delegates_policy_without_widening_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)
    vector_path = tmp_path / "family.json"
    evidence_path = tmp_path / "post-g06.json"
    upstream_path = tmp_path / "upstream.json"
    two_path = tmp_path / "two.json"
    vector_sha = _write_json(vector_path, vector)
    evidence_sha = _write_json(evidence_path, evidence)
    post_g06_two_clean = _post_g06_two_clean(
        vector,
        evidence,
        evidence_file_sha256=evidence_sha,
    )
    post_g06_two_clean_path = tmp_path / "post-g06-two-clean.json"
    post_g06_two_clean_sha = _write_json(
        post_g06_two_clean_path,
        post_g06_two_clean,
    )
    upstream_path.write_bytes(UPSTREAM_EVIDENCE_RAW)
    two_path.write_bytes(UPSTREAM_TWO_CLEAN_RAW)

    source_sha = "c" * 40
    monkeypatch.setattr(target, "verify_source_head", lambda value: source_sha)
    monkeypatch.setattr(
        target,
        "load_canonical_authorities",
        lambda: (
            _FakeGate(),
            _FakeBridge(),
            {"policy_identity_sha256": target.POLICY_IDENTITY_SHA256},
        ),
    )

    args = argparse.Namespace(
        source_git_sha=source_sha,
        family_vector=vector_path,
        expected_family_vector_file_sha256=vector_sha,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        post_g06_evidence=evidence_path,
        expected_post_g06_evidence_file_sha256=evidence_sha,
        expected_post_g06_evidence_identity_sha256=evidence[
            "evidence_identity_sha256"
        ],
        post_g06_two_clean_proof=post_g06_two_clean_path,
        expected_post_g06_two_clean_proof_file_sha256=post_g06_two_clean_sha,
        expected_post_g06_two_clean_proof_identity_sha256=post_g06_two_clean[
            "proof_identity_sha256"
        ],
        upstream_global_dedup_evidence=upstream_path,
        upstream_global_dedup_two_clean=two_path,
    )
    values = target.execute(args)
    receipt = values["execution-receipt"]
    assert receipt["next100_input_identity_sha256"] == target.sha256(
        target.canonical(values["next100-input"])
    )
    assert receipt["balance_status"] == (
        "BLOCKED_NO_NONZERO_POLICY_COMPLIANT_MIXTURE"
    )
    assert receipt["next_scientific_gate"] == (
        "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY"
    )
    assert values["composition-dedup-proof"]["terminal_verdict"] == "PASS"
    assert receipt["post_g06_two_clean_proof_identity_sha256"] == (
        post_g06_two_clean["proof_identity_sha256"]
    )
    assert values["composition-dedup-proof"][
        "post_g06_physical_uniqueness"
    ]["two_clean_proof_identity_sha256"] == (
        post_g06_two_clean["proof_identity_sha256"]
    )
    for field, expected in target.ZERO_CREDIT.items():
        assert receipt[field] == expected


def test_compare_outputs_requires_byte_identity_and_seals_proof(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)

    proof_path = tmp_path / "proof.json"
    proof = target.compare_outputs(a, b, proof_path)
    assert proof["fresh_process_count"] == 2
    assert proof["byte_identical_outputs"] is True
    assert proof["post_g06_two_clean_proof_identity_sha256"] == "5" * 64
    assert proof["proof_identity_sha256"] == target.self_hash(
        proof, "proof_identity_sha256"
    )
    assert target.compare_outputs(a, b, proof_path) == proof

    (b / "balance-result.json").write_bytes(b"{}\n")
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="two-clean output differs",
    ):
        target.compare_outputs(a, b, tmp_path / "second-proof.json")


def test_write_output_dir_is_immutable_and_resumable(tmp_path: Path) -> None:
    output = tmp_path / "out"
    values = {
        "composition-dedup-proof": {"value": 1},
        "next100-input": {"value": 2},
        "balance-result": {"value": 3},
        "balance-binding": {"value": 4},
        "execution-receipt": {"value": 5},
    }
    target.write_output_dir(output, values)
    assert (output / "execution-receipt.json").read_bytes() == (
        target.canonical_line({"value": 5})
    )

    target.write_output_dir(output, values)
    divergent = copy.deepcopy(values)
    divergent["balance-result"]["value"] = 99
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="refusing to overwrite divergent durable evidence",
    ):
        target.write_output_dir(output, divergent)


def test_resolve_existing_path_normalizes_existing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    assert target._resolve_existing_path(
        str(existing),
        "canonical path missing",
    ) == existing.resolve(strict=True)


def test_resolve_existing_path_maps_missing_paths_to_domain_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="canonical path missing",
    ):
        target._resolve_existing_path(
            str(tmp_path / "missing"),
            "canonical path missing",
        )


def test_verify_module_provenance_missing_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "canonical.py"
    expected.write_text("x = 1\n", encoding="utf-8")
    missing = tmp_path / "missing.py"

    class MissingModule:
        __file__ = str(missing)

    monkeypatch.setattr(target, "ROOT", tmp_path)
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="module provenance drift: canonical.py",
    ):
        target.verify_module_provenance(MissingModule(), "canonical.py")


def _write_two_clean_fixture(
    output: Path,
    *,
    zero_credit_override: dict | None = None,
) -> None:
    output.mkdir()
    execution_head = "c" * 40
    family_identity = "8" * 64
    post_g06_identity = "6" * 64
    post_g06_two_clean_identity = "5" * 64
    composition_core = {
        "schema": "synthetic-composition",
        "execution_head_sha": execution_head,
        "upstream_global_dedup": {
            "evidence_identity_sha256": target.UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID,
            "two_clean_identity_sha256": target.UPSTREAM_TWO_CLEAN_ID,
        },
        "post_g06_physical_uniqueness": {
            "evidence_identity_sha256": post_g06_identity,
            "two_clean_proof_identity_sha256": post_g06_two_clean_identity,
        },
        "family_vector_identity_sha256": family_identity,
        "x": 1,
        **target.ZERO_CREDIT,
    }
    composition = {
        **composition_core,
        "evidence_identity_sha256": target.sha256(
            target.canonical(composition_core)
        ),
    }
    next100 = {
        "schema_version": "synthetic-next100",
        "dedup_authority": {
            "worker_id": target.DEDUP_WORKER_ID,
            "head_sha": execution_head,
            "evidence_identity_sha256": composition[
                "evidence_identity_sha256"
            ],
            "terminal_verdict": "PASS",
        },
        "totals": {"total_unique_bytes": 2},
        "physical_authority": {
            "family_vector_identity_sha256": family_identity,
        },
        "x": 2,
    }
    next100_identity = target.sha256(target.canonical(next100))
    policy_identity = "9" * 64
    result_core = {
        "schema_version": "synthetic-result",
        "policy_identity_sha256": policy_identity,
        "dedup_authority": next100["dedup_authority"],
        "input_totals": next100["totals"],
        "status": "synthetic-status",
        "x": 3,
    }
    balance_result = {
        **result_core,
        "result_identity_sha256": target.sha256(target.canonical(result_core)),
    }
    binding_core = {
        "schema": "synthetic-binding",
        "family_vector_identity_sha256": family_identity,
        "next100_input_identity_sha256": next100_identity,
        "balance_policy_identity_sha256": policy_identity,
        "balance_status": balance_result["status"],
        "x": 4,
    }
    balance_binding = {
        **binding_core,
        "binding_identity_sha256": target.sha256(
            target.canonical(binding_core)
        ),
    }
    receipt_core = {
        "schema": target.RECEIPT_SCHEMA,
        "execution_head_sha": execution_head,
        "upstream_global_dedup_evidence_identity_sha256": (
            target.UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID
        ),
        "upstream_global_dedup_two_clean_identity_sha256": (
            target.UPSTREAM_TWO_CLEAN_ID
        ),
        "post_g06_evidence_identity_sha256": post_g06_identity,
        "post_g06_two_clean_proof_identity_sha256": post_g06_two_clean_identity,
        "composition_dedup_identity_sha256": composition[
            "evidence_identity_sha256"
        ],
        "next100_input_identity_sha256": next100_identity,
        "family_vector_identity_sha256": family_identity,
        "balance_policy_identity_sha256": policy_identity,
        "balance_status": balance_result["status"],
        "balance_result_identity_sha256": balance_result[
            "result_identity_sha256"
        ],
        "balance_binding_identity_sha256": balance_binding[
            "binding_identity_sha256"
        ],
        **target.ZERO_CREDIT,
    }
    if zero_credit_override:
        receipt_core.update(zero_credit_override)
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": target.sha256(target.canonical(receipt_core)),
    }
    payloads = {
        "composition-dedup-proof": composition,
        "next100-input": next100,
        "balance-result": balance_result,
        "balance-binding": balance_binding,
        "execution-receipt": receipt,
    }
    for name, value in payloads.items():
        (output / f"{name}.json").write_bytes(target.canonical_line(value))


def test_compare_outputs_rejects_same_directory(tmp_path: Path) -> None:
    output = tmp_path / "same"
    _write_two_clean_fixture(output)
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="two-clean output directories must be distinct",
    ):
        target.compare_outputs(output, output, tmp_path / "proof.json")


def test_compare_outputs_rejects_coherently_resealed_zero_credit_drift(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-zero"
    b = tmp_path / "b-zero"
    override = {"training_executed": True}
    _write_two_clean_fixture(a, zero_credit_override=override)
    _write_two_clean_fixture(b, zero_credit_override=override)
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="execution receipt zero-credit drift: training_executed",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-zero.json")


def test_write_output_dir_recovers_matching_interrupted_temp(tmp_path: Path) -> None:
    output = tmp_path / "resume"
    output.mkdir()
    values = {
        "composition-dedup-proof": {"value": 1},
        "next100-input": {"value": 2},
        "balance-result": {"value": 3},
        "balance-binding": {"value": 4},
        "execution-receipt": {"value": 5},
    }
    payload = target.canonical_line(values["composition-dedup-proof"])
    temp = output / "composition-dedup-proof.json.tmp"
    temp.write_bytes(payload)

    target.write_output_dir(output, values)

    assert (output / "composition-dedup-proof.json").read_bytes() == payload
    assert not temp.exists()


def test_write_output_dir_commits_receipt_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def capture(path: Path, payload: bytes, *, label: str) -> None:
        labels.append(label)

    monkeypatch.setattr(target, "write_immutable_bytes", capture)
    values = {
        "composition-dedup-proof": {"value": 1},
        "next100-input": {"value": 2},
        "balance-result": {"value": 3},
        "balance-binding": {"value": 4},
        "execution-receipt": {"value": 5},
    }
    target.write_output_dir(tmp_path / "ordered", values)
    assert labels == [
        "composition-dedup-proof",
        "next100-input",
        "balance-result",
        "balance-binding",
        "execution receipt",
    ]


def test_compare_outputs_rejects_coherently_tampered_result_self_hash(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-result"
    b = tmp_path / "b-result"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "balance-result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["x"] = 99
        path.write_bytes(target.canonical_line(value))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="balance result self-hash mismatch",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-result.json")


def test_compare_outputs_rejects_coherently_tampered_next100_input(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-next100"
    b = tmp_path / "b-next100"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "next100-input.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["x"] = 200
        path.write_bytes(target.canonical_line(value))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="balance binding next100-input identity mismatch",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-next100.json")


def test_compare_outputs_rejects_cross_binding_policy_drift(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-policy"
    b = tmp_path / "b-policy"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "execution-receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["balance_policy_identity_sha256"] = "7" * 64
        receipt["receipt_identity_sha256"] = target.self_hash(
            receipt,
            "receipt_identity_sha256",
        )
        path.write_bytes(target.canonical_line(receipt))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="execution receipt balance-policy identity mismatch",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-policy.json")


def test_compare_outputs_enforces_checkout_provenance_before_proof_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a-provenance"
    b = tmp_path / "b-provenance"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    calls: list[str] = []

    monkeypatch.setattr(
        target,
        "verify_dependency_blobs",
        lambda: calls.append("dependencies"),
    )

    def reject_head(value: str) -> str:
        calls.append(f"head:{value}")
        raise target.RadaPostG06BalanceError("execution HEAD drift")

    monkeypatch.setattr(target, "verify_source_head", reject_head)
    proof = tmp_path / "proof-provenance.json"
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="execution HEAD drift",
    ):
        target.compare_outputs(
            a,
            b,
            proof,
            enforce_checkout_provenance=True,
        )

    assert calls == ["dependencies", "head:" + "c" * 40]
    assert not proof.exists()


def test_compare_outputs_accepts_bound_checkout_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a-provenance-ok"
    b = tmp_path / "b-provenance-ok"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    calls: list[str] = []
    monkeypatch.setattr(
        target,
        "verify_dependency_blobs",
        lambda: calls.append("dependencies"),
    )

    def accept_head(value: str) -> str:
        calls.append(f"head:{value}")
        return value

    monkeypatch.setattr(target, "verify_source_head", accept_head)
    expected_result = json.loads(
        (a / "balance-result.json").read_text(encoding="utf-8")
    )

    class ReplayGate:
        def validate_vector(self, value: dict) -> None:
            assert value["dedup_authority"]["terminal_verdict"] == "PASS"

        def evaluate(self, policy: dict, value: dict) -> dict:
            return expected_result

    monkeypatch.setattr(
        target,
        "load_canonical_authorities",
        lambda: (ReplayGate(), object(), {}),
    )
    proof_path = tmp_path / "proof-provenance-ok.json"
    proof = target.compare_outputs(
        a,
        b,
        proof_path,
        enforce_checkout_provenance=True,
    )

    assert calls == ["dependencies", "head:" + "c" * 40]
    assert proof_path.exists()
    assert proof["execution_head_sha"] == "c" * 40


def test_write_output_dir_rejects_unbound_stale_file(tmp_path: Path) -> None:
    output = tmp_path / "stale-output"
    output.mkdir()
    (output / "stale.json").write_text("{}\n", encoding="utf-8")
    values = {
        "composition-dedup-proof": {"value": 1},
        "next100-input": {"value": 2},
        "balance-result": {"value": 3},
        "balance-binding": {"value": 4},
        "execution-receipt": {"value": 5},
    }

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="output directory contains unexpected entries",
    ):
        target.write_output_dir(output, values)


def test_compare_outputs_rejects_unbound_extra_files(tmp_path: Path) -> None:
    a = tmp_path / "a-extra"
    b = tmp_path / "b-extra"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    (a / "unbound.json").write_text("{}\n", encoding="utf-8")
    (b / "unbound.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="output directory contains unexpected entries",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-extra.json")


def test_write_output_dir_rejects_receipt_before_children(tmp_path: Path) -> None:
    output = tmp_path / "receipt-first"
    output.mkdir()
    (output / "execution-receipt.json").write_bytes(
        target.canonical_line({"value": 5})
    )
    values = {
        "composition-dedup-proof": {"value": 1},
        "next100-input": {"value": 2},
        "balance-result": {"value": 3},
        "balance-binding": {"value": 4},
        "execution-receipt": {"value": 5},
    }

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="output directory bundle is incomplete",
    ):
        target.write_output_dir(output, values)

    assert sorted(path.name for path in output.iterdir()) == [
        "execution-receipt.json"
    ]


def test_git_helper_maps_called_process_error_to_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse"])

    monkeypatch.setattr(target.subprocess, "check_output", fail)
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="git rev-parse failed",
    ):
        target._git("rev-parse", "HEAD")


def test_verify_source_head_maps_git_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "c" * 40
    monkeypatch.setattr(target, "_git", lambda *args: expected)

    def fail_run(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(target.subprocess, "run", fail_run)
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="git merge-base failed",
    ):
        target.verify_source_head(expected)


def test_compare_outputs_rejects_composition_zero_credit_reseal(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-composition-credit"
    b = tmp_path / "b-composition-credit"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        composition_path = output / "composition-dedup-proof.json"
        composition = json.loads(composition_path.read_text(encoding="utf-8"))
        composition["training_executed"] = True
        composition["evidence_identity_sha256"] = target.self_hash(
            composition,
            "evidence_identity_sha256",
        )
        composition_path.write_bytes(target.canonical_line(composition))

        receipt_path = output / "execution-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["composition_dedup_identity_sha256"] = composition[
            "evidence_identity_sha256"
        ]
        receipt["receipt_identity_sha256"] = target.self_hash(
            receipt,
            "receipt_identity_sha256",
        )
        receipt_path.write_bytes(target.canonical_line(receipt))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="composition zero-credit drift: training_executed",
    ):
        target.compare_outputs(
            a,
            b,
            tmp_path / "proof-composition-credit.json",
        )


def test_compare_outputs_rejects_dedup_worker_drift(tmp_path: Path) -> None:
    a = tmp_path / "a-worker"
    b = tmp_path / "b-worker"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "next100-input.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["dedup_authority"]["worker_id"] = "forged-worker"
        path.write_bytes(target.canonical_line(value))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="NEXT100 dedup worker drift",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-worker.json")


def test_compare_outputs_rejects_post_g06_chain_drift(tmp_path: Path) -> None:
    a = tmp_path / "a-post-g06"
    b = tmp_path / "b-post-g06"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "execution-receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["post_g06_evidence_identity_sha256"] = "7" * 64
        receipt["receipt_identity_sha256"] = target.self_hash(
            receipt,
            "receipt_identity_sha256",
        )
        path.write_bytes(target.canonical_line(receipt))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="post-G06 identity differs across evidence chain",
    ):
        target.compare_outputs(a, b, tmp_path / "proof-post-g06.json")


def test_compare_outputs_rejects_nonreproducible_canonical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a-replay-drift"
    b = tmp_path / "b-replay-drift"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    monkeypatch.setattr(target, "verify_dependency_blobs", lambda: None)
    monkeypatch.setattr(target, "verify_source_head", lambda value: value)

    class DriftGate:
        def validate_vector(self, value: dict) -> None:
            assert value["dedup_authority"]["terminal_verdict"] == "PASS"

        def evaluate(self, policy: dict, value: dict) -> dict:
            return {"replayed": False}

    monkeypatch.setattr(
        target,
        "load_canonical_authorities",
        lambda: (DriftGate(), object(), {}),
    )
    proof = tmp_path / "proof-replay-drift.json"
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="balance result is not a deterministic replay of canonical gate",
    ):
        target.compare_outputs(
            a,
            b,
            proof,
            enforce_checkout_provenance=True,
        )

    assert not proof.exists()



def test_matrix_adapter_stack_root_is_pinned() -> None:
    assert target.STACK_BASE_HEAD == "039ae67cfd0a1393936c49a7bf463f5f68927a03"


def test_post_g06_two_clean_proof_rejects_nonindependent_reseal() -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)
    evidence_raw = target.canonical_line(evidence)
    proof = _post_g06_two_clean(
        vector,
        evidence,
        evidence_file_sha256=target.sha256(evidence_raw),
    )
    proof["independent_runner_jobs"] = False
    proof["proof_identity_sha256"] = target.self_hash(
        proof,
        "proof_identity_sha256",
    )
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="independent-runner proof missing",
    ):
        target.verify_post_g06_two_clean_proof(
            proof,
            evidence,
            vector,
            expected_proof_identity_sha256=proof[
                "proof_identity_sha256"
            ],
            expected_evidence_file_sha256=target.sha256(evidence_raw),
        )


def test_post_g06_two_clean_proof_rejects_output_root_reseal() -> None:
    vector = _family_vector()
    evidence = _post_g06(vector)
    evidence_raw = target.canonical_line(evidence)
    proof = _post_g06_two_clean(
        vector,
        evidence,
        evidence_file_sha256=target.sha256(evidence_raw),
    )
    proof["output_file_sha256"][
        target.G05_G06_BUNDLE_FILES["evidence"]
    ] = "0" * 64
    proof["proof_identity_sha256"] = target.self_hash(
        proof,
        "proof_identity_sha256",
    )
    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="output-file roots drift",
    ):
        target.verify_post_g06_two_clean_proof(
            proof,
            evidence,
            vector,
            expected_proof_identity_sha256=proof[
                "proof_identity_sha256"
            ],
            expected_evidence_file_sha256=target.sha256(evidence_raw),
        )


def test_compare_outputs_rejects_post_g06_two_clean_chain_drift(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-two-clean-chain"
    b = tmp_path / "b-two-clean-chain"
    _write_two_clean_fixture(a)
    _write_two_clean_fixture(b)
    for output in (a, b):
        path = output / "execution-receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["post_g06_two_clean_proof_identity_sha256"] = "7" * 64
        receipt["receipt_identity_sha256"] = target.self_hash(
            receipt,
            "receipt_identity_sha256",
        )
        path.write_bytes(target.canonical_line(receipt))

    with pytest.raises(
        target.RadaPostG06BalanceError,
        match="post-G06 two-clean identity differs across evidence chain",
    ):
        target.compare_outputs(
            a,
            b,
            tmp_path / "proof-two-clean-chain.json",
        )
