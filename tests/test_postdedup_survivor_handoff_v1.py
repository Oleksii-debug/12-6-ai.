from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from twelve_six.data import postdedup_survivor_handoff_v1 as handoff


def _sha(label: str) -> str:
    return handoff._sha256_bytes(label.encode())


def _source(source_id: str, family: str, modality: str, capacity: int, origin: str) -> dict:
    return {
        "source_id": source_id,
        "source_family": family,
        "stable_origin_id_sha256": _sha(origin),
        "stable_object_id_sha256": _sha(f"obj:{source_id}"),
        "modality": modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": capacity,
        "verified_raw_bytes": capacity,
        "verified_raw_sha256": _sha(f"raw:{source_id}"),
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": capacity,
        "comparison_payload_sha256": _sha(f"cmp:{source_id}"),
        "normalized_sha256": _sha(f"norm:{source_id}"),
    }


def _fixture() -> tuple[dict, dict]:
    rows = [
        _source("a", "fa", "uk", 10, "origin-a"),
        _source("b", "fb", "uk", 20, "origin-b"),
        _source("c", "fc", "code", 7, "origin-c"),
    ]
    nested_core = {
        "schema_version": handoff.V3_SCHEMA,
        "algorithm": "fixture",
        "local_free_only": True,
        "model_training_executed": False,
        "source_admission_authority": False,
        "source_count": 3,
        "matching_authority": "fixture",
        "thresholds": {},
        "sources": rows,
        "matches": [],
        "match_counts": {},
        "capacity_collapsing_match_counts": {},
        "terminal_candidates": {
            "source_count": 3,
            "declared_source_family_count": 3,
            "stable_origin_count": 3,
            "effective_independent_origin_count": 2,
            "raw_bytes_before": 37,
            "declared_capacity_bytes_before": 37,
            "conservative_unique_capacity_bytes_after": 27,
            "duplicate_discount_bytes": 10,
            "duplicate_discount_fraction": round(10 / 37, 12),
            "duplicate_cluster_count": 1,
            "duplicate_clusters": [["a", "b"]],
            "origin_clusters": [["origin-a", "origin-b"], ["origin-c"]],
            "by_modality": {},
        },
        "capacity_policy": {},
        "raw_text_emitted": False,
    }
    nested = dict(nested_core)
    nested["report_sha256"] = handoff._sha256_bytes(
        handoff._canonical_bytes(nested_core, ensure_ascii=True) + b"\n"
    )
    v8_core = {
        "schema_version": handoff.V8_SCHEMA,
        "worker_id": "fixture",
        "execution_profile": "LOCAL_FREE",
        "base_main_sha": "0" * 40,
        "baseline_v7": {},
        "data_bulk_code1": {},
        "source_vector": {
            "source_object_count": 3,
            "source_family_counts": {"uk": 2, "en": 0, "code": 1},
            "source_capacity_bytes_before_global_dedup": 37,
            "conservative_unique_capacity_bytes_after_global_dedup": 27,
            "duplicate_discount_bytes": 10,
            "duplicate_cluster_count": 1,
            "effective_independent_origin_count": 2,
            "by_modality": {},
        },
        "dedup_v3": nested,
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
    v8 = dict(v8_core)
    v8["report_sha256"] = handoff._sha256_bytes(handoff._canonical_bytes(v8_core))
    survivor_rows = [
        {
            key: rows[index][key]
            for key in (
                "source_id",
                "source_family",
                "modality",
                "declared_capacity_bytes",
                "verified_raw_sha256",
                "normalized_sha256",
                "stable_origin_id_sha256",
                "stable_object_id_sha256",
            )
        }
        for index in (1, 2)
    ]
    survivor_core = {
        "schema_version": handoff.SURVIVOR_SCHEMA,
        "selection_rule": "largest_declared_capacity_then_lexicographically_smallest_source_id",
        "v8_report_sha256": v8["report_sha256"],
        "nested_v3_report_sha256": nested["report_sha256"],
        "pre_dedup_source_object_count": 3,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": 37,
        "post_dedup_declared_capacity_bytes": 27,
        "duplicate_discount_bytes": 10,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [{
            "member_source_ids": ["a", "b"],
            "selected_source_id": "b",
            "selected_declared_capacity_bytes": 20,
        }],
        "survivors": survivor_rows,
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
    survivor = dict(survivor_core)
    survivor["survivor_authority_sha256"] = handoff._sha256_bytes(
        handoff._canonical_bytes(survivor_core, ensure_ascii=True)
    )
    return v8, survivor


class SurvivorHandoffV1Tests(unittest.TestCase):
    def test_consumes_canonical_survivors_without_reselection(self) -> None:
        v8, survivor = _fixture()
        result = handoff.materialize_postdedup_survivor_handoff(
            v8,
            survivor,
            expected_v8_report_sha256=v8["report_sha256"],
            expected_survivor_authority_sha256=survivor["survivor_authority_sha256"],
        )
        self.assertEqual([row["source_id"] for row in result["retained_sources"]], ["b", "c"])
        self.assertEqual(result["retained_declared_capacity_bytes"], 27)
        self.assertEqual(result["independence_cluster_count"], 2)
        self.assertEqual(result["authorized_training_exposure"], 0)

    def test_rejects_terminal_survivor_substitution(self) -> None:
        v8, survivor = _fixture()
        expected = survivor["survivor_authority_sha256"]
        substitute = copy.deepcopy(survivor)
        substitute["survivors"].reverse()
        core = dict(substitute)
        core.pop("survivor_authority_sha256")
        substitute["survivor_authority_sha256"] = handoff._sha256_bytes(
            handoff._canonical_bytes(core, ensure_ascii=True)
        )
        with self.assertRaisesRegex(handoff.PostDedupSurvivorHandoffError, "terminal binding"):
            handoff.materialize_postdedup_survivor_handoff(
                v8,
                substitute,
                expected_v8_report_sha256=v8["report_sha256"],
                expected_survivor_authority_sha256=expected,
            )

    def test_rejects_comparison_identity_tamper(self) -> None:
        v8, survivor = _fixture()
        v8["dedup_v3"]["sources"][0]["comparison_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(handoff.PostDedupSurvivorHandoffError, "self-hash mismatch"):
            handoff.materialize_postdedup_survivor_handoff(
                v8,
                survivor,
                expected_v8_report_sha256=v8["report_sha256"],
                expected_survivor_authority_sha256=survivor["survivor_authority_sha256"],
            )

    def test_rejects_origin_cluster_coverage_drift(self) -> None:
        v8, survivor = _fixture()
        v8["dedup_v3"]["terminal_candidates"]["origin_clusters"][0][0] = "wrong-origin"
        nested = v8["dedup_v3"]
        nested_core = dict(nested)
        nested_core.pop("report_sha256")
        nested["report_sha256"] = handoff._sha256_bytes(
            handoff._canonical_bytes(nested_core, ensure_ascii=True) + b"\n"
        )
        v8_core = dict(v8)
        v8_core.pop("report_sha256")
        v8["report_sha256"] = handoff._sha256_bytes(handoff._canonical_bytes(v8_core))
        survivor["v8_report_sha256"] = v8["report_sha256"]
        survivor["nested_v3_report_sha256"] = nested["report_sha256"]
        survivor_core = dict(survivor)
        survivor_core.pop("survivor_authority_sha256")
        survivor["survivor_authority_sha256"] = handoff._sha256_bytes(
            handoff._canonical_bytes(survivor_core, ensure_ascii=True)
        )
        with self.assertRaisesRegex(handoff.PostDedupSurvivorHandoffError, "origin clusters"):
            handoff.materialize_postdedup_survivor_handoff(
                v8,
                survivor,
                expected_v8_report_sha256=v8["report_sha256"],
                expected_survivor_authority_sha256=survivor["survivor_authority_sha256"],
            )

    def test_committed_terminal_handoff_identity_if_present(self) -> None:
        path = Path("evidence/data/next100_065f_postdedup_survivor_handoff_v1.json")
        if not path.exists():
            self.skipTest("terminal handoff evidence not present in this checkout")
        value = json.loads(path.read_text())
        self.assertEqual(value["schema_version"], handoff.OUTPUT_SCHEMA)
        self.assertEqual(
            value["input_v8_report_sha256"],
            "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a",
        )
        self.assertEqual(
            value["input_survivor_authority_sha256"],
            "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf",
        )
        self.assertEqual(
            value["handoff_identity_sha256"],
            "5a14a9d5abc3917768adcbc64128758a053b1fea5062eb577bf990ca155342be",
        )
        core = dict(value)
        identity = core.pop("handoff_identity_sha256")
        self.assertEqual(identity, handoff._sha256_bytes(handoff._canonical_bytes(core)))
        self.assertEqual(value["retained_source_count"], 262)
        self.assertEqual(value["retained_declared_capacity_bytes"], 6_095_321)
        self.assertEqual(value["independence_cluster_count"], 21)
        self.assertEqual(value["authorized_training_exposure"], 0)
        self.assertFalse(value["reserved_evaluation_decontamination_complete"])


if __name__ == "__main__":
    unittest.main()
