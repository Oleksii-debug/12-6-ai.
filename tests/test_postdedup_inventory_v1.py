from __future__ import annotations

import copy
import unittest

from twelve_six.data import postdedup_inventory_v1 as postdedup


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


def _match(left: str, right: str, *, collapsing: bool) -> dict:
    return {
        "left_source_id": left,
        "right_source_id": right,
        "left_source_family": f"family.{left}",
        "right_source_family": f"family.{right}",
        "match_type": "raw_exact" if collapsing else "natural_near_copy",
        "score": 1.0 if collapsing else 0.81,
        "cross_source_family": True,
        "capacity_collapsing": collapsing,
        "independence_collapsing": collapsing,
        "evidence_class": "synthetic-test",
    }


def _v3() -> dict:
    sources = [
        _source("a", "family.a", "uk", 10),
        _source("b", "family.b", "uk", 20),
        _source("c", "family.c", "en", 5),
        _source("d", "family.d", "en", 5),
        _source("e", "family.e", "code", 7),
    ]
    matches = [
        _match("a", "b", collapsing=True),
        _match("c", "d", collapsing=True),
        _match("b", "e", collapsing=False),
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
        "origin_clusters": [["a", "b"], ["c", "d"], ["e"]],
        "by_modality": {},
    }
    core = {
        "schema_version": postdedup.V3_SCHEMA,
        "algorithm": "synthetic-v3",
        "local_free_only": True,
        "model_training_executed": False,
        "source_admission_authority": False,
        "source_count": 5,
        "matching_authority": "synthetic",
        "thresholds": {},
        "sources": sources,
        "matches": matches,
        "match_counts": {"natural_near_copy": 1, "raw_exact": 2},
        "capacity_collapsing_match_counts": {"raw_exact": 2},
        "terminal_candidates": terminal,
        "capacity_policy": {},
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": postdedup._v3_report_sha256(core)}


def _v8() -> dict:
    dedup = _v3()
    core = {
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
        "dedup_v3": dedup,
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
    return {**core, "report_sha256": postdedup._sha256_obj(core)}


class PostDedupInventoryV1Tests(unittest.TestCase):
    def test_incumbent_v3_hash_serialization_is_distinct_from_v8(self) -> None:
        core = {"label": "Україна", "value": 1}
        self.assertNotEqual(
            postdedup._sha256_obj(core),
            postdedup._sha256_bytes(postdedup._v3_canonical_bytes(core)),
        )
        report = {**core, "report_sha256": postdedup._v3_report_sha256(core)}
        self.assertEqual(
            postdedup._v3_self_hash_matches(report, "synthetic V3"),
            report["report_sha256"],
        )

    def test_selects_largest_then_lexical_representative(self) -> None:
        report = _v8()
        inventory = postdedup.materialize_postdedup_inventory(
            report,
            expected_v8_report_sha256=report["report_sha256"],
        )
        self.assertEqual(
            [row["source_id"] for row in inventory["retained_sources"]],
            ["b", "c", "e"],
        )
        self.assertEqual(inventory["retained_unique_capacity_bytes"], 32)
        self.assertEqual(inventory["retained_source_count"], 3)
        self.assertEqual(inventory["excluded_duplicate_source_count"], 2)
        self.assertFalse(inventory["reserved_evaluation_decontamination_complete"])
        self.assertEqual(inventory["authorized_training_exposure"], 0)

    def test_rebuild_is_byte_deterministic(self) -> None:
        report = _v8()
        first = postdedup.materialize_postdedup_inventory(
            report,
            expected_v8_report_sha256=report["report_sha256"],
        )
        second = postdedup.materialize_postdedup_inventory(
            copy.deepcopy(report),
            expected_v8_report_sha256=report["report_sha256"],
        )
        self.assertEqual(postdedup._canonical_bytes(first), postdedup._canonical_bytes(second))
        postdedup.verify_postdedup_inventory(
            report,
            first,
            expected_v8_report_sha256=report["report_sha256"],
        )

    def test_rejects_self_hashed_v8_substitution_against_expected_handoff(self) -> None:
        report = _v8()
        expected = report["report_sha256"]
        substitute = copy.deepcopy(report)
        substitute["remaining_blockers"] = ["different"]
        core = dict(substitute)
        core.pop("report_sha256")
        substitute["report_sha256"] = postdedup._sha256_obj(core)
        with self.assertRaisesRegex(
            postdedup.PostDedupInventoryError,
            "does not match expected terminal handoff",
        ):
            postdedup.materialize_postdedup_inventory(
                substitute,
                expected_v8_report_sha256=expected,
            )

    def test_rejects_nested_report_tamper(self) -> None:
        report = _v8()
        report["dedup_v3"]["sources"][0]["declared_capacity_bytes"] = 11
        core = dict(report)
        core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(core)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "nested V3 report self-hash"):
            postdedup.materialize_postdedup_inventory(
                report,
                expected_v8_report_sha256=report["report_sha256"],
            )

    def test_rejects_duplicate_cluster_summary_drift(self) -> None:
        report = _v8()
        dedup = report["dedup_v3"]
        dedup["terminal_candidates"]["duplicate_clusters"] = [["a", "b"]]
        dedup["report_sha256"] = postdedup._v3_report_sha256(dedup)
        report_core = dict(report)
        report_core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(report_core)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "duplicate clusters"):
            postdedup.materialize_postdedup_inventory(
                report,
                expected_v8_report_sha256=report["report_sha256"],
            )

    def test_rejects_capacity_arithmetic_drift(self) -> None:
        report = _v8()
        dedup = report["dedup_v3"]
        dedup["terminal_candidates"]["conservative_unique_capacity_bytes_after"] = 31
        dedup["report_sha256"] = postdedup._v3_report_sha256(dedup)
        report["source_vector"]["conservative_unique_capacity_bytes_after_global_dedup"] = 31
        report_core = dict(report)
        report_core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(report_core)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "capacity arithmetic"):
            postdedup.materialize_postdedup_inventory(
                report,
                expected_v8_report_sha256=report["report_sha256"],
            )

    def test_rejects_training_authority_promotion(self) -> None:
        report = _v8()
        report["claim_boundary"]["authorized_training_exposure"] = 1
        core = dict(report)
        core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(core)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "grants training exposure"):
            postdedup.materialize_postdedup_inventory(
                report,
                expected_v8_report_sha256=report["report_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
