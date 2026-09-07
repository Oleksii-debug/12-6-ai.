from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools import validate_data526_research_corpus_predecontam_v3 as validator

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_blocker_v3.json")
BASE = json.loads(MANIFEST.read_text(encoding="utf-8"))


def rehash(doc):
    doc["evidence_identity_sha256"] = validator.canonical_sha256(doc)
    return doc


class Data526PredecontamBlockerV3Tests(unittest.TestCase):
    def test_baseline(self):
        validator.validate_doc(copy.deepcopy(BASE))

    def test_source_and_dedup_must_bind_same_35_object_vector(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["expected_pre_dedup_source_object_count"] = 31
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_attrs_code_capacity_cannot_be_dropped(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["pre_global_dedup_vector"]["by_stratum"]["code"]["numeric_training_capacity_bytes"] = 106_031
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_queued_source_cannot_be_consumed(self):
        doc = copy.deepcopy(BASE)
        doc["source_convergence_candidate"]["terminal_authority_consumed"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_queued_v7_cannot_publish_post_dedup_capacity(self):
        doc = copy.deepcopy(BASE)
        doc["global_dedup_candidate"]["post_dedup_unique_capacity_bytes"] = 2_215_615
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_candidate_cannot_freeze_before_terminal_v7(self):
        doc = copy.deepcopy(BASE)
        doc["candidate_freeze"]["frozen"] = True
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_source_bytes_cannot_be_promoted_to_loss_positions(self):
        doc = copy.deepcopy(BASE)
        doc["claim_boundary"]["authorized_unique_optimized_targets"] = 2_215_615
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_tokenizer_fit_remains_blocked(self):
        doc = copy.deepcopy(BASE)
        doc["downstream_gates"]["tokenizer_fit"] = "PERMITTED"
        with self.assertRaises(ValueError):
            validator.validate_doc(rehash(doc))

    def test_self_hash_detects_unrehash_tamper(self):
        doc = copy.deepcopy(BASE)
        doc["status"] = "READY"
        with self.assertRaises(ValueError):
            validator.validate_doc(doc)


if __name__ == "__main__":
    unittest.main()
