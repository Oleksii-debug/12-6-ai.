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

    def test_engineering_pilot_cannot_claim_general_base_quality(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["engineering_pilot"]["claim_ceiling"] = "GENERAL_BASE_READY"
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_external_reference_cannot_become_direct_byte_budget(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["external_chinchilla_style_anchor"]["direct_conversion_to_byte_positions"] = True
        mutated["training_ladder"]["science_complete_20m_budget"]["value"] = 412_268_800
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_science_budget_remains_undefined_before_calibration(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["science_complete_20m_budget"]["value"] = 100_000_000
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

    def test_100m_future_stage_cannot_receive_fabricated_byte_budget(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["future_scale_reference"][0]["direct_reference_byte_positions"] = 2_000_000_000
        with self.assertRaises(LadderValidationError):
            validate(mutated)

    def test_flop_normalized_byte_vs_subword_ablation_is_mandatory(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["training_ladder"]["decision_sequence"].remove("L2_FLOP_NORMALIZED_BYTE_VS_SUBWORD_ABLATION")
        with self.assertRaises(LadderValidationError):
            validate(mutated)


if __name__ == "__main__":
    unittest.main()
