from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools import validate_data526_research_corpus_predecontam as validator

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v1.json")
BASE = json.loads(MANIFEST.read_text(encoding="utf-8"))


def rehash(doc):
    doc["evidence_identity_sha256"] = validator.canonical_sha256(doc)
    return doc


class Data526PredecontamBlockerTests(unittest.TestCase):
    def test_baseline(self):
        validator.validate_doc(copy.deepcopy(BASE))

    def test_queue_cannot_be_promoted_to_pass(self):
        doc = copy.deepcopy(BASE)
        doc["required_source_convergence"]["exact_head_ci_state"] = "SUCCESS"
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_nonterminal_authority_cannot_be_consumed(self):
        doc = copy.deepcopy(BASE)
        doc["required_source_convergence"]["terminal_authority_consumed"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_candidate_digest_cannot_be_fabricated(self):
        doc = copy.deepcopy(BASE)
        doc["candidate_freeze"]["candidate_set_digest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_pending_source_bytes_are_not_loss_capacity(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["authorized_unique_optimized_targets"] = 565743
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_training_cannot_be_authorized(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["long_training_authorized"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_dependency_head_drift_fails_closed(self):
        doc = copy.deepcopy(BASE)
        doc["required_source_convergence"]["observed_head_sha"] = "f" * 40
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_superseded_issue_cannot_be_rebound_as_current(self):
        doc = copy.deepcopy(BASE)
        doc["required_source_convergence"]["issue"] = 521
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_registry_identity_drift_fails_closed(self):
        doc = copy.deepcopy(BASE)
        doc["required_source_convergence"]["reported_registry_identity_sha256_non_authoritative_until_terminal"] = "0" * 64
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_stratum_redistribution_preserving_totals_fails(self):
        doc = copy.deepcopy(BASE)
        vector = doc["required_source_convergence"]["reported_pre_successor_global_dedup_vector_non_authoritative_until_terminal"]["by_stratum"]
        vector["uk"]["bytes"] += 1
        vector["en"]["bytes"] -= 1
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_family_redistribution_preserving_total_fails(self):
        doc = copy.deepcopy(BASE)
        vector = doc["required_source_convergence"]["reported_pre_successor_global_dedup_vector_non_authoritative_until_terminal"]["by_stratum"]
        vector["uk"]["families"] += 1
        vector["code"]["families"] -= 1
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_self_hash_detects_unrehash_tamper(self):
        doc = copy.deepcopy(BASE)
        doc["status"] = "READY"
        with self.assertRaises(ValueError):
            validator.validate_doc(doc)


if __name__ == "__main__":
    unittest.main()
