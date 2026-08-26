from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools import validate_data526_research_corpus_predecontam_v2 as validator

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v2.json")
BASE = json.loads(MANIFEST.read_text(encoding="utf-8"))


def rehash(doc):
    doc["evidence_identity_sha256"] = validator.canonical_sha256(doc)
    return doc


class Data526PredecontamBlockerV2Tests(unittest.TestCase):
    def test_baseline(self):
        validator.validate_doc(copy.deepcopy(BASE))

    def test_source_queue_cannot_be_promoted_to_pass(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["exact_head_ci_state"] = "SUCCESS"
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_dedup_pending_cannot_be_promoted_to_pass(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["exact_head_ci_state"] = "SUCCESS"
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_nonterminal_source_authority_cannot_be_consumed(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["terminal_authority_consumed"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_nonterminal_dedup_authority_cannot_be_consumed(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["terminal_authority_consumed"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_post_dedup_capacity_cannot_be_fabricated(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["post_dedup_unique_capacity_bytes"] = 2_045_180
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_post_dedup_report_cannot_be_fabricated(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["post_dedup_report_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_candidate_digest_cannot_be_fabricated(self):
        doc = copy.deepcopy(BASE)
        doc["candidate_freeze"]["candidate_set_digest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_bytes_are_not_loss_capacity(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["authorized_unique_optimized_targets"] = 2_045_180
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_bytes_cannot_be_declared_tokens(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["source_bytes_are_training_tokens"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_training_cannot_be_authorized(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["long_training_authorized"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_head_drift_fails_closed(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["head_sha"] = "f" * 40
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_dedup_head_drift_fails_closed(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["head_sha"] = "f" * 40
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_registry_identity_drift_fails_closed(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["registry_identity_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_dedup_capacity_vector_mismatch_fails(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["expected_pre_dedup_numeric_capacity_bytes"] -= 1
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_dedup_family_vector_mismatch_fails(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["expected_pre_dedup_family_counts"]["en"] = 4
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_record_freeze_cannot_precede_global_dedup(self):
        doc = copy.deepcopy(BASE)
        doc["downstream_gates"]["record_inventory_freeze"] = "PERMITTED"
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_global_dedup_cannot_be_skipped_in_required_order(self):
        doc = copy.deepcopy(BASE)
        doc["unblock_rule"]["required_order"][1] = "Freeze records immediately after source registry."
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_self_hash_detects_unrehash_tamper(self):
        doc = copy.deepcopy(BASE)
        doc["status"] = "READY"
        with self.assertRaises(ValueError):
            validator.validate_doc(doc)


if __name__ == "__main__":
    unittest.main()
