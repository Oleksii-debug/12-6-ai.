from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/twelve_six/data/research_corpus_acquisition.py"
PLAN_PATH = ROOT / "configs/data/research_corpus_v1_acquisition_plan.json"
BASE_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"

spec = importlib.util.spec_from_file_location("research_corpus_acquisition", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResearchCorpusV1AcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    def test_current_plan_passes_without_promoting_corpus_or_training(self) -> None:
        report = module.load_and_validate(PLAN_PATH, ROOT)
        self.assertEqual(report["status"], "PASS_PLANNING_CONTRACT_ONLY")
        self.assertEqual(
            report["remaining_gap_bytes"],
            {"uk": 8_899_144, "en": 6_849_357, "code": 3_930_867, "total": 19_679_368},
        )
        self.assertEqual(
            report["buffered_gross_required_bytes"],
            {"uk": 14_831_907, "en": 11_415_595, "code": 6_551_445, "total": 32_798_947},
        )
        self.assertEqual(
            report["planned_gross_bytes"],
            {"uk": 15_000_000, "en": 11_500_000, "code": 6_700_000, "total": 33_200_000},
        )
        self.assertEqual(
            report["planning_headroom_bytes"],
            {"uk": 168_093, "en": 84_405, "code": 148_555, "total": 401_053},
        )
        self.assertEqual(report["terminal_package_credit_bytes"]["total"], 0)
        self.assertEqual(report["base_authority_binding"]["status"], "PASS_BASE_AUTHORITY_BINDING")
        self.assertEqual(report["base_authority_binding"]["base_config_blob_sha1"], "d5b640b386219290f69d02a7f2e30a338c883009")
        self.assertFalse(report["research_corpus_v1_released"])
        self.assertFalse(report["training_authorized_by_this_contract"])
        self.assertEqual(report["next_gate"], "SOURCE_ACQUISITION_AND_TERMINAL_AUTHORITY")

    def test_plan_identity_is_deterministic(self) -> None:
        first = module.compute_plan_identity(copy.deepcopy(self.plan))
        second = module.compute_plan_identity(copy.deepcopy(self.plan))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)

    def test_arithmetic_drift_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["remaining_gap_bytes"]["uk"] += 1
        broken["remaining_gap_bytes"]["total"] += 1
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_false_capacity_credit_on_prospect_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["work_packages"][0]["authority_credit_bytes"] = 1
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_nonterminal_terminal_authority_object_fails_with_contract_error(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["work_packages"][0]["terminal_authority"] = {"status": "ADMIT"}
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_public_availability_cannot_be_promoted_to_authority(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["work_packages"][0]["public_availability_is_training_authority"] = True
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_evaluation_leakage_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["work_packages"][0]["evaluation_authorized"] = True
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_paid_compute_or_training_claim_fails_closed(self) -> None:
        for field in ("paid_compute_used", "model_training_executed", "optimizer_update_executed", "tokenizer_fit_executed"):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.plan)
                broken[field] = True
                with self.assertRaises(module.AcquisitionContractError):
                    module.validate_plan(broken)

    def test_underbuffered_stratum_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        for package in broken["work_packages"]:
            if package["package_id"] == "en-open-technical-docs-pool":
                package["planned_gross_bytes"] = 1
                break
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_package_concentration_drift_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["work_packages"][0]["planned_gross_bytes"] = 10_000_000
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_fake_terminal_authority_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        package = broken["work_packages"][0]
        package["stage"] = "TERMINAL_ADMIT"
        package["authority_credit_bytes"] = 100
        package["rights_state"] = "PASS"
        package["provenance_state"] = "PASS"
        package["terminal_authority"] = {
            "status": "ADMIT",
            "training_authorized": True,
            "evaluation_authorized": False,
            "exact_head_sha": "0" * 40,
            "authority_identity_sha256": "1" * 64,
            "capacity_ledger_sha256": "2" * 64,
            "execution": {"run_id": 1, "conclusion": "queued"},
            "rights": {"training_decision": "ALLOW", "provenance_review": "PASS", "evidence_reference": "fixture"},
            "stratum": "uk",
            "family_id": "fixture.family",
            "capacity_bytes": 100,
            "objects": [{"object_id": "x", "content_sha256": "3" * 64, "eligible_bytes": 100}],
        }
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_release_claim_fails_closed(self) -> None:
        broken = copy.deepcopy(self.plan)
        broken["claim_boundary"]["research_corpus_v1_released"] = True
        with self.assertRaises(module.AcquisitionContractError):
            module.validate_plan(broken)

    def test_base_authority_content_drift_fails_closed(self) -> None:
        base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
        base["converged_pre_successor_dedup_vector"]["numeric_capacity_bytes"]["en"] += 1
        base["converged_pre_successor_dedup_vector"]["numeric_capacity_bytes"]["total"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.json"
            path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(module.AcquisitionContractError):
                module.validate_plan_against_base_authority(copy.deepcopy(self.plan), path)

    def test_base_authority_blob_binding_fails_closed_on_format_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.json"
            path.write_text(BASE_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(module.AcquisitionContractError):
                module.validate_plan_against_base_authority(copy.deepcopy(self.plan), path)


if __name__ == "__main__":
    unittest.main()
