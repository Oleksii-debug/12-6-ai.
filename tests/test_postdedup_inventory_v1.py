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


def _survivor_authority(report: dict, *, choose_d: bool = False) -> dict:
    by_id = {row["source_id"]: row for row in report["dedup_v3"]["sources"]}
    selected_ids = ["b", "d" if choose_d else "c", "e"]
    survivor_rows = [
        {
            field: by_id[source_id][field]
            for field in postdedup.SURVIVOR_SHARED_SOURCE_FIELDS
        }
        for source_id in selected_ids
    ]
    survivor_rows.sort(key=lambda row: row["source_id"])
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
                "selected_source_id": "d" if choose_d else "c",
                "selected_declared_capacity_bytes": 5,
            },
        ],
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


def _materialize(report: dict, authority: dict | None = None) -> dict:
    authority = authority or _survivor_authority(report)
    return postdedup.materialize_postdedup_inventory(
        report,
        authority,
        expected_v8_report_sha256=report["report_sha256"],
        expected_survivor_authority_sha256=authority["survivor_authority_sha256"],
    )


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

    def test_uses_terminal_survivor_authority_as_physical_retained_set(self) -> None:
        report = _v8()
        authority = _survivor_authority(report)
        inventory = _materialize(report, authority)
        self.assertEqual(
            [row["source_id"] for row in inventory["retained_sources"]],
            ["b", "c", "e"],
        )
        self.assertEqual(
            inventory["input_survivor_authority_sha256"],
            authority["survivor_authority_sha256"],
        )
        self.assertEqual(inventory["retained_unique_capacity_bytes"], 32)
        self.assertEqual(inventory["retained_source_count"], 3)
        self.assertEqual(inventory["excluded_duplicate_source_count"], 2)
        self.assertFalse(inventory["reserved_evaluation_decontamination_complete"])
        self.assertEqual(inventory["authorized_training_exposure"], 0)

    def test_external_tie_survivor_wins_over_local_lexical_crosscheck(self) -> None:
        report = _v8()
        authority = _survivor_authority(report, choose_d=True)
        inventory = _materialize(report, authority)
        self.assertEqual(
            [row["source_id"] for row in inventory["retained_sources"]],
            ["b", "d", "e"],
        )
        excluded = {
            row["source_id"]: row["retained_source_id"]
            for row in inventory["excluded_duplicate_sources"]
        }
        self.assertEqual(excluded["c"], "d")

    def test_rebuild_is_byte_deterministic(self) -> None:
        report = _v8()
        authority = _survivor_authority(report)
        first = _materialize(report, authority)
        second = _materialize(copy.deepcopy(report), copy.deepcopy(authority))
        self.assertEqual(postdedup._canonical_bytes(first), postdedup._canonical_bytes(second))
        postdedup.verify_postdedup_inventory(
            report,
            authority,
            first,
            expected_v8_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=authority[
                "survivor_authority_sha256"
            ],
        )

    def test_rejects_self_hashed_v8_substitution_against_expected_handoff(self) -> None:
        report = _v8()
        authority = _survivor_authority(report)
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
                authority,
                expected_v8_report_sha256=expected,
                expected_survivor_authority_sha256=authority[
                    "survivor_authority_sha256"
                ],
            )

    def test_rejects_self_consistent_survivor_substitution_against_expected_identity(self) -> None:
        report = _v8()
        authority = _survivor_authority(report)
        expected = authority["survivor_authority_sha256"]
        substitute = copy.deepcopy(authority)
        substitute["by_modality"] = {"forged": {"source_object_count": 3}}
        _rehash_survivor(substitute)
        with self.assertRaisesRegex(
            postdedup.PostDedupInventoryError,
            "survivor authority does not match expected terminal identity",
        ):
            postdedup.materialize_postdedup_inventory(
                report,
                substitute,
                expected_v8_report_sha256=report["report_sha256"],
                expected_survivor_authority_sha256=expected,
            )

    def test_locally_different_inventory_fails_against_external_survivor(self) -> None:
        report = _v8()
        canonical = _survivor_authority(report)
        alternate = _survivor_authority(report, choose_d=True)
        alternate_inventory = _materialize(report, alternate)
        with self.assertRaisesRegex(
            postdedup.PostDedupInventoryError,
            "does not match terminal-survivor-bound rebuild",
        ):
            postdedup.verify_postdedup_inventory(
                report,
                canonical,
                alternate_inventory,
                expected_v8_report_sha256=report["report_sha256"],
                expected_survivor_authority_sha256=canonical[
                    "survivor_authority_sha256"
                ],
            )

    def test_rejects_nested_report_tamper(self) -> None:
        report = _v8()
        authority = _survivor_authority(report)
        report["dedup_v3"]["sources"][0]["declared_capacity_bytes"] = 11
        core = dict(report)
        core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(core)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "nested V3 report self-hash"):
            postdedup.materialize_postdedup_inventory(
                report,
                authority,
                expected_v8_report_sha256=report["report_sha256"],
                expected_survivor_authority_sha256=authority[
                    "survivor_authority_sha256"
                ],
            )

    def test_rejects_unknown_capacity_collapsing_match_type(self) -> None:
        report = _v8()
        dedup = report["dedup_v3"]
        forged = _match("b", "e", collapsing=True)
        forged["match_type"] = "future_unknown_copy_semantics"
        dedup["matches"].append(forged)
        dedup["report_sha256"] = postdedup._v3_report_sha256(dedup)
        report_core = dict(report)
        report_core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(report_core)
        authority = _survivor_authority(report)
        with self.assertRaisesRegex(
            postdedup.PostDedupInventoryError,
            "unsupported capacity-collapsing match type",
        ):
            _materialize(report, authority)

    def test_rejects_duplicate_cluster_summary_drift(self) -> None:
        report = _v8()
        dedup = report["dedup_v3"]
        dedup["terminal_candidates"]["duplicate_clusters"] = [["a", "b"]]
        dedup["report_sha256"] = postdedup._v3_report_sha256(dedup)
        report_core = dict(report)
        report_core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(report_core)
        authority = _survivor_authority(report)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "duplicate clusters"):
            _materialize(report, authority)

    def test_rejects_capacity_arithmetic_drift(self) -> None:
        report = _v8()
        dedup = report["dedup_v3"]
        dedup["terminal_candidates"]["conservative_unique_capacity_bytes_after"] = 31
        dedup["report_sha256"] = postdedup._v3_report_sha256(dedup)
        report["source_vector"]["conservative_unique_capacity_bytes_after_global_dedup"] = 31
        report_core = dict(report)
        report_core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(report_core)
        authority = _survivor_authority(report)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "capacity arithmetic"):
            _materialize(report, authority)

    def test_rejects_training_authority_promotion(self) -> None:
        report = _v8()
        report["claim_boundary"]["authorized_training_exposure"] = 1
        core = dict(report)
        core.pop("report_sha256")
        report["report_sha256"] = postdedup._sha256_obj(core)
        authority = _survivor_authority(report)
        with self.assertRaisesRegex(postdedup.PostDedupInventoryError, "grants training exposure"):
            _materialize(report, authority)


if __name__ == "__main__":
    unittest.main()
