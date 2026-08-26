from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import validate_d03_rada_trees_acquisition_probe as validator


class RadaTreesAcquisitionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(validator.CONFIG.read_text(encoding="utf-8"))

    def test_committed_probe_is_fail_closed_valid(self) -> None:
        validator.validate(copy.deepcopy(self.value))

    def test_short_or_mutable_head_is_rejected(self) -> None:
        value = copy.deepcopy(self.value)
        value["source"]["observed_head_sha"] = "main"
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_discovery_metadata_cannot_create_training_credit(self) -> None:
        value = copy.deepcopy(self.value)
        value["claim_boundary"]["training_authorized_bytes"] = 88_000_000
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_annotation_layer_cannot_create_family_credit(self) -> None:
        value = copy.deepcopy(self.value)
        value["lineage"]["annotation_layers_create_new_family_credit"] = True
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_archive_without_exact_identity_cannot_be_promoted(self) -> None:
        value = copy.deepcopy(self.value)
        value["claim_boundary"]["bulk_source_admitted"] = True
        with self.assertRaises(AssertionError):
            validator.validate(value)

    def test_evaluation_permission_is_not_inferred_from_training_rights(self) -> None:
        value = copy.deepcopy(self.value)
        value["rights"]["evaluation_decision"] = "ALLOWED"
        with self.assertRaises(AssertionError):
            validator.validate(value)


if __name__ == "__main__":
    unittest.main()
