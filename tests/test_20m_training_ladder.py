from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.validate_20m_training_ladder import LadderValidationError, validate


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "control" / "20m_training_ladder_v1.json"


class Test20MTrainingLadder(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_canonical_config_passes(self) -> None:
        validate(self.data)

    def test_reference_target_is_exactly_bound_to_parameter_count(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["scientific_reference"]["reference_unique_optimized_targets"] += 1
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_engineering_pilot_cannot_claim_general_base_quality(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["engineering_pilot"]["claim_ceiling"] = "GENERAL_BASE_READY"
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_zero_corpus_authority_cannot_enable_long_training(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["current_data_gate"]["long_training_runnable"] = True
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_paid_compute_cannot_be_silently_authorized(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["compute_boundary"]["material_paid_compute_authorized"] = True
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_future_scale_reference_preserves_planning_ratio(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["future_scale_reference"][0]["reference_tokens_at_20_per_parameter"] = 1_999_999_999
        with self.assertRaises(LadderValidationError):
            validate(mutated)


if __name__ == "__main__":
    unittest.main()
