from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_liger_kernel_bootstrap_stress_v1 import validate_evidence, validate_manifest


class LigerBootstrapContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "configs" / "research" / "liger_kernel_bootstrap_stress_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.evidence = json.loads(
            (ROOT / "evidence" / "research" / "liger_kernel_bootstrap_stress_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_canonical_positive_contract(self) -> None:
        self.assertEqual(validate_manifest(self.manifest), [])
        self.assertEqual(validate_evidence(self.evidence, self.manifest), [])

    def test_wrong_commit_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["upstream"]["commit"] = "0" * 40
        self.assertIn("upstream commit drift", validate_manifest(mutated))

    def test_floating_tag_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["upstream"]["tag"] = "main"
        self.assertIn("upstream tag drift", validate_manifest(mutated))

    def test_paid_compute_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["forbidden"]["paid_compute"] = True
        self.assertIn("paid compute boundary violated", validate_manifest(mutated))

    def test_foreign_weight_boundary_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["forbidden"]["foreign_pretrained_weights"] = True
        self.assertIn("foreign weights boundary violated", validate_manifest(mutated))

    def test_runtime_unproven_cannot_be_promoted(self) -> None:
        mutated = copy.deepcopy(self.evidence)
        mutated["promotion_state"] = "PARITY_PROVEN"
        self.assertIn(
            "runtime-unproven evidence must remain RETEST_RUNTIME_REQUIRED",
            validate_evidence(mutated, self.manifest),
        )

    def test_fabricated_benchmark_fails(self) -> None:
        mutated = copy.deepcopy(self.evidence)
        mutated["benchmark"]["executed"] = True
        self.assertIn(
            "benchmark cannot be marked executed without real runtime",
            validate_evidence(mutated, self.manifest),
        )

    def test_fabricated_parity_fails(self) -> None:
        mutated = copy.deepcopy(self.evidence)
        mutated["parity"]["proven"] = True
        self.assertIn(
            "parity cannot be proven without real runtime",
            validate_evidence(mutated, self.manifest),
        )

    def test_identity_tampering_fails(self) -> None:
        mutated = copy.deepcopy(self.evidence)
        mutated["environment"]["python_version"] = "0.0.0"
        self.assertIn("environment identity mismatch", validate_evidence(mutated, self.manifest))

    def test_canonical_lineage_is_strict(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["project"]["canonical_base_random_init_only"] = False
        self.assertIn("canonical Base boundary not enforced", validate_manifest(mutated))


if __name__ == "__main__":
    unittest.main()
