from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_scale_ladder_20m_to_1b.py"
MANIFEST = ROOT / "configs/control/scale_ladder_20m_to_1b_v1.json"


def _load_validator():
    spec = importlib.util.spec_from_file_location("scale_ladder_validator", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load scale ladder validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScaleLadder20MTo1BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = _load_validator()

    def test_exact_parameter_algebra_is_locked(self) -> None:
        report = self.validator.validate()
        exact = [stage["exact_parameters"] for stage in report["stages"]]
        self.assertEqual(
            exact,
            [20_613_440, 99_897_600, 400_421_888, 999_761_920],
        )
        self.assertEqual(report["verdict"], "PASS_CONTRACT_VALID")

    def test_every_scale_remains_fail_closed(self) -> None:
        with MANIFEST.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for stage in manifest["stages"]:
            self.assertFalse(stage["training_authorized"])
            self.assertFalse(stage["promotion_allowed"])
            self.assertTrue(stage["requires_preceding_stage_pass"])
            self.assertIn("MATERIAL_COMPUTE_AUTHORIZATION", stage["required_training_gates"])
        self.assertEqual(
            manifest["current_decision"],
            "BLOCK_MATERIAL_SCALE_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING_AND_DATA_READINESS",
        )

    def test_gqa_scale_transition_is_explicitly_gated(self) -> None:
        with MANIFEST.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        by_target = {stage["target_parameters"]: stage for stage in manifest["stages"]}
        self.assertEqual(by_target[100_000_000]["attention"]["kind"], "MHA")
        for target in (400_000_000, 1_000_000_000):
            stage = by_target[target]
            self.assertEqual(stage["attention"]["kind"], "GQA")
            self.assertTrue(
                any(gate.startswith("NATIVE_GQA") for gate in stage["required_training_gates"])
            )


if __name__ == "__main__":
    unittest.main()
