from __future__ import annotations

import copy
import hashlib
import json
import unittest

from twelve_six.data import postdedup_decontam_handoff_v1 as handoff
from twelve_six.data import postdedup_inventory_v1 as postdedup


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inventory(payloads: dict[str, bytes]) -> dict:
    retained = []
    for index, (source_id, payload) in enumerate(sorted(payloads.items())):
        retained.append(
            {
                "source_id": source_id,
                "source_family": f"family.{source_id}",
                "modality": "code" if source_id == "b" else "uk",
                "declared_capacity_bytes": len(payload),
                "stable_origin_id_sha256": _sha(f"origin:{source_id}".encode()),
                "stable_object_id_sha256": _sha(f"object:{source_id}".encode()),
                "verified_raw_bytes": len(payload),
                "verified_raw_sha256": _sha(payload),
                "comparison_policy": "DATA232_GENERIC_FROM_RAW",
                "comparison_payload_bytes": len(payload),
                "comparison_payload_sha256": _sha(payload),
                "normalized_sha256": _sha(payload.lower()),
                "component_index": index,
                "component_size": 1,
                "component_identity_sha256": _sha(f"component:{source_id}".encode()),
            }
        )
    core = {
        "schema_version": postdedup.OUTPUT_SCHEMA,
        "selection_policy": postdedup.SELECTION_POLICY,
        "upstream_survivor_selection_rule": postdedup.SURVIVOR_SELECTION_RULE,
        "input_v8_report_sha256": "1" * 64,
        "input_v3_dedup_report_sha256": "2" * 64,
        "input_survivor_authority_sha256": "3" * 64,
        "input_source_count": len(retained),
        "capacity_component_count": len(retained),
        "duplicate_component_count": 0,
        "independence_cluster_count": len(retained),
        "retained_source_count": len(retained),
        "excluded_duplicate_source_count": 0,
        "retained_unique_capacity_bytes": sum(len(payload) for payload in payloads.values()),
        "retained_by_modality": {},
        "retained_by_source_family": {},
        "retained_sources": retained,
        "excluded_duplicate_sources": [],
        "raw_text_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_packing_complete": False,
        "postpack_unique_loss_ledger_complete": False,
        "tokenizer_fit_authorized": False,
        "authorized_training_exposure": 0,
        "model_training_executed": False,
        "final_test_payload_read": False,
        "paid_compute_used": False,
    }
    return {**core, "inventory_identity_sha256": postdedup._sha256_obj(core)}


def _rehash(inventory: dict) -> None:
    core = copy.deepcopy(inventory)
    core.pop("inventory_identity_sha256", None)
    inventory["inventory_identity_sha256"] = postdedup._sha256_obj(core)


def _prepare(inventory: dict, payloads: dict[str, bytes]):
    return handoff.prepare_ephemeral_data232_rows(
        inventory,
        payloads,
        expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
    )


class PostDedupDecontamHandoffV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.payloads = {
            "a": "Україна\n".encode(),
            "b": b"def f():\n    return 1\n",
        }
        self.inventory = _inventory(self.payloads)

    def test_prepares_exact_ephemeral_rows_and_text_free_evidence(self) -> None:
        rows, evidence = _prepare(self.inventory, self.payloads)
        self.assertEqual([row["record_id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["text"], "Україна\n")
        self.assertEqual(rows[1]["source_family"], "family.b")
        self.assertEqual(evidence["retained_source_count"], 2)
        self.assertEqual(
            evidence["postdedup_inventory_identity_sha256"],
            self.inventory["inventory_identity_sha256"],
        )
        self.assertEqual(
            evidence["input_survivor_authority_sha256"],
            self.inventory["input_survivor_authority_sha256"],
        )
        self.assertEqual(evidence["authorized_training_exposure"], 0)
        self.assertFalse(evidence["final_test_payload_accessed"])
        durable = json.dumps(evidence, ensure_ascii=False)
        self.assertNotIn("Україна", durable)
        self.assertNotIn("def f", durable)

    def test_rejects_missing_or_extra_payload_coverage(self) -> None:
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "coverage must equal",
        ):
            _prepare(self.inventory, {"a": self.payloads["a"]})
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "coverage must equal",
        ):
            _prepare(self.inventory, {**self.payloads, "extra": b"x"})

    def test_rejects_payload_identity_drift(self) -> None:
        tampered_payloads = dict(self.payloads)
        tampered_payloads["b"] = b"def f():\n    return 2\n"
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "payload identity drift",
        ):
            _prepare(self.inventory, tampered_payloads)

    def test_rejects_inventory_self_hash_tamper(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["retained_sources"][0]["source_family"] = "forged.family"
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "inventory self-hash mismatch",
        ):
            handoff.prepare_ephemeral_data232_rows(
                tampered,
                self.payloads,
                expected_inventory_identity_sha256=self.inventory[
                    "inventory_identity_sha256"
                ],
            )

    def test_rejects_self_consistent_substituted_inventory(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["retained_sources"][0]["source_family"] = "forged.family"
        _rehash(tampered)
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "does not match expected terminal inventory identity",
        ):
            handoff.prepare_ephemeral_data232_rows(
                tampered,
                self.payloads,
                expected_inventory_identity_sha256=self.inventory[
                    "inventory_identity_sha256"
                ],
            )

    def test_rejects_inventory_without_survivor_authority_binding(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered.pop("input_survivor_authority_sha256")
        _rehash(tampered)
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "input_survivor_authority_sha256 must be",
        ):
            _prepare(tampered, self.payloads)

    def test_rejects_rehashed_final_test_boundary_weakening(self) -> None:
        tampered = copy.deepcopy(self.inventory)
        tampered["final_test_payload_read"] = True
        _rehash(tampered)
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "boundary weakened: final_test_payload_read",
        ):
            _prepare(tampered, self.payloads)

    def test_rejects_non_utf8_comparison_payload(self) -> None:
        payloads = {"a": b"\xff"}
        inventory = _inventory(payloads)
        with self.assertRaisesRegex(
            handoff.PostDedupDecontamHandoffError,
            "not strict UTF-8",
        ):
            _prepare(inventory, payloads)


if __name__ == "__main__":
    unittest.main()
