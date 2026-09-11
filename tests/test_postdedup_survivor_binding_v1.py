from __future__ import annotations

import copy
import unittest

from twelve_six.data import postdedup_inventory_v1 as postdedup
from twelve_six.data import postdedup_survivor_binding_v1 as binding


def _sha(label: str) -> str:
    return postdedup._sha256_bytes(label.encode("utf-8"))


def _source(source_id: str, family: str, modality: str, capacity: int) -> dict:
    return {
        "source_id": source_id,
        "source_family": family,
        "stable_origin_id_sha256": _sha(f"origin:{source_id}"),
        "stable_object_id_sha256": _sha(f"object:{source_id}"),
        "modality": modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": capacity,
        "verified_raw_bytes": capacity,
        "verified_raw_sha256": _sha(f"raw:{source_id}"),
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": capacity,
        "comparison_payload_sha256": _sha(f"comparison:{source_id}"),
        "normalized_sha256": _sha(f"normalized:{source_id}"),
    }


def _match(left: str, right: str) -> dict:
    return {
        "left_source_id": left,
        "right_source_id": right,
        "left_source_family": f"family.{left}",
        "right_source_family": f"family.{right}",
        "match_type": "raw_exact",
        "score": 1.0,
        "cross_source_family": True,
        "capacity_collapsing": True,
        "independence_collapsing": True,
        "evidence_class": "synthetic-test",
    }


def _v8() -> dict:
    sources = [
        _source("a", "family.a", "uk", 10),
        _source("b", "family.b", "uk", 20),
        _source("c", "family.c", "en", 5),
        _source("d", "family.d", "en", 5),
        _source("e", "family.e", "code", 7),
    ]
    terminal = {
        "source_count": 5,
        "declared_source_family_count": 5,
        "stable_origin_count": 5,
        "effective_independent_origin_count": 3,
        "raw_bytes_before": 47,
        "declared_capacity_bytes_before": 47,
        "conservative_unique_capacity_bytes_after": 32,
        "duplicate_discount_bytes": 15,
        "duplicate_discount_fraction": round(15 / 47, 12),
        "duplicate_cluster_count": 2,
        "duplicate_clusters": [["a", "b"], ["c", "d"]],
        "origin_clusters": [
            ["origin:a", "origin:b"],
            ["origin:c", "origin:d"],
            ["origin:e"],
        ],
        "by_modality": {},
    }
    v3_core = {
        "schema_version": postdedup.V3_SCHEMA,
        "algorithm": "synthetic-v3",
        "local_free_only": True,
        "model_training_executed": False,
        "source_admission_authority": False,
        "source_count": 5,
        "matching_authority": "synthetic",
        "thresholds": {},
        "sources": sources,
        "matches": [_match("a", "b"), _match("c", "d")],
        "match_counts": {"raw_exact": 2},
        "capacity_collapsing_match_counts": {"raw_exact": 2},
        "terminal_candidates": terminal,
        "capacity_policy": {},
        "raw_text_emitted": False,
    }
    v3 = {**v3_core, "report_sha256": postdedup._v3_report_sha256(v3_core)}
    v8_core = {
        "schema_version": postdedup.V8_SCHEMA,
        "worker_id": "NEXT100-065F-CURRENT-MAIN-GLOBAL-DEDUP-V8",
        "execution_profile": "LOCAL_FREE",
        "base_main_sha": "0" * 40,
        "baseline_v7": {},
        "data_bulk_code1": {},
        "source_vector": {
            "source_object_count": 5,
            "source_family_counts": {"uk": 2, "en": 2, "code": 1},
            "source_capacity_bytes_before_global_dedup": 47,
            "conservative_unique_capacity_bytes_after_global_dedup": 32,
            "duplicate_discount_bytes": 15,
            "duplicate_cluster_count": 2,
            "effective_independent_origin_count": 3,
            "by_modality": {},
        },
        "dedup_v3": v3,
        "raw_text_emitted": False,
        "claim_boundary": {
            "authorized_training_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "final_test_payload_read": False,
            "paid_compute_used": False,
        },
        "remaining_blockers": [],
    }
    return {**v8_core, "report_sha256": postdedup._sha256_obj(v8_core)}


def _survivor_authority(report: dict) -> dict:
    by_id = {row["source_id"]: row for row in report["dedup_v3"]["sources"]}
    survivors = [
        {
            field: by_id[source_id][field]
            for field in postdedup.SURVIVOR_SHARED_SOURCE_FIELDS
        }
        for source_id in ("b", "c", "e")
    ]
    core = {
        "schema_version": postdedup.SURVIVOR_SCHEMA,
        "selection_rule": postdedup.SURVIVOR_SELECTION_RULE,
        "v8_report_sha256": report["report_sha256"],
        "nested_v3_report_sha256": report["dedup_v3"]["report_sha256"],
        "pre_dedup_source_object_count": 5,
        "post_dedup_survivor_source_object_count": 3,
        "pre_dedup_declared_capacity_bytes": 47,
        "post_dedup_declared_capacity_bytes": 32,
        "duplicate_discount_bytes": 15,
        "duplicate_cluster_count": 2,
        "duplicate_clusters": [
            {
                "member_source_ids": ["a", "b"],
                "selected_source_id": "b",
                "selected_declared_capacity_bytes": 20,
            },
            {
                "member_source_ids": ["c", "d"],
                "selected_source_id": "c",
                "selected_declared_capacity_bytes": 5,
            },
        ],
        "survivors": sorted(survivors, key=lambda row: row["source_id"]),
        "by_modality": {},
        "truth_boundary": {
            "source_object_authority_only": True,
            "training_record_inventory_materialized": False,
            "evaluation_decontamination_passed": False,
            "tokenizer_fit_authorized": False,
            "authorized_training_exposure": 0,
            "model_training_executed": False,
            "final_test_payload_read": False,
            "paid_compute_used": False,
        },
    }
    core["survivor_authority_sha256"] = postdedup._sha256_bytes(
        postdedup._survivor_canonical_bytes(core)
    )
    return core


def _rehash_survivor(authority: dict) -> None:
    core = copy.deepcopy(authority)
    core.pop("survivor_authority_sha256", None)
    authority["survivor_authority_sha256"] = postdedup._sha256_bytes(
        postdedup._survivor_canonical_bytes(core)
    )


class PostDedupSurvivorBindingV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = _v8()
        self.authority = _survivor_authority(self.report)
        self.inventory = postdedup.materialize_postdedup_inventory(
            self.report,
            self.authority,
            expected_v8_report_sha256=self.report["report_sha256"],
            expected_survivor_authority_sha256=self.authority[
                "survivor_authority_sha256"
            ],
        )

    def test_builds_text_free_terminal_survivor_binding(self) -> None:
        evidence = binding.build_survivor_inventory_binding(
            self.report,
            self.authority,
            self.inventory,
            expected_v8_report_sha256=self.report["report_sha256"],
            expected_survivor_authority_sha256=self.authority[
                "survivor_authority_sha256"
            ],
        )
        self.assertEqual(evidence["retained_source_count"], 3)
        self.assertEqual(evidence["retained_unique_capacity_bytes"], 32)
        self.assertEqual(evidence["authorized_training_exposure"], 0)
        self.assertEqual(
            evidence["survivor_authority_sha256"],
            self.inventory["input_survivor_authority_sha256"],
        )
        binding.verify_survivor_inventory_binding(
            self.report,
            self.authority,
            self.inventory,
            evidence,
            expected_v8_report_sha256=self.report["report_sha256"],
            expected_survivor_authority_sha256=self.authority[
                "survivor_authority_sha256"
            ],
        )

    def test_rejects_self_consistent_survivor_substitution(self) -> None:
        expected = self.authority["survivor_authority_sha256"]
        substituted = copy.deepcopy(self.authority)
        substituted["by_modality"] = {"forged": {"source_object_count": 3}}
        _rehash_survivor(substituted)
        with self.assertRaisesRegex(
            binding.PostDedupSurvivorBindingError,
            "does not match expected terminal identity",
        ):
            binding.build_survivor_inventory_binding(
                self.report,
                substituted,
                self.inventory,
                expected_v8_report_sha256=self.report["report_sha256"],
                expected_survivor_authority_sha256=expected,
            )

    def test_rejects_self_consistent_inventory_substitution(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["retained_sources"][0]["source_family"] = "forged.family"
        core = copy.deepcopy(tampered)
        core.pop("inventory_identity_sha256", None)
        tampered["inventory_identity_sha256"] = postdedup._sha256_obj(core)
        with self.assertRaisesRegex(
            binding.PostDedupSurvivorBindingError,
            "does not match terminal-survivor-bound rebuild",
        ):
            binding.build_survivor_inventory_binding(
                self.report,
                self.authority,
                tampered,
                expected_v8_report_sha256=self.report["report_sha256"],
                expected_survivor_authority_sha256=self.authority[
                    "survivor_authority_sha256"
                ],
            )


if __name__ == "__main__":
    unittest.main()