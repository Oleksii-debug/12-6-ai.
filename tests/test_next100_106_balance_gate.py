from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/next100_106_balance_gate.py"
POLICY_PATH = ROOT / "configs/data/next100_106_balance_gate_policy_v1.json"

SPEC = importlib.util.spec_from_file_location("next100_106_balance_gate", TOOL_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def family(family_id: str, stratum: str, unique_bytes: int) -> dict:
    return {
        "family_id": family_id,
        "stratum": stratum,
        "unique_bytes": unique_bytes,
    }


def vector(families: list[dict]) -> dict:
    by_stratum = {"ua": 0, "en": 0, "code": 0}
    counts = {"ua": 0, "en": 0, "code": 0}
    for item in families:
        by_stratum[item["stratum"]] += item["unique_bytes"]
        counts[item["stratum"]] += 1
    return {
        "schema_version": gate.INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "NEXT100-105-CROSSSOURCE-DEDUP-V4",
            "head_sha": "a" * 40,
            "evidence_identity_sha256": "b" * 64,
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": sum(by_stratum.values()),
            "by_stratum": by_stratum,
            "family_count": counts,
        },
    }


class Next100106BalanceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        gate.validate_policy(cls.policy)

    def test_exact_20m_policy_mix_is_feasible(self) -> None:
        families = [
            family("ua.a", "ua", 4_500_000),
            family("ua.b", "ua", 4_500_000),
            family("en.a", "en", 3_500_000),
            family("en.b", "en", 3_500_000),
            family("code.a", "code", 2_000_000),
            family("code.b", "code", 2_000_000),
        ]
        result = gate.evaluate(self.policy, vector(families))
        self.assertEqual(result["maximum_feasible_total_source_bytes"], 20_000_000)
        self.assertEqual(result["status"], "TARGET_20M_SOURCE_MIX_FEASIBLE")
        self.assertEqual(
            result["maximum_feasible_stratum_bytes"],
            {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000},
        )
        self.assertEqual(
            sum(row["allocated_bytes"] for row in result["deterministic_maximum_allocation"]),
            20_000_000,
        )
        self.assertEqual(
            result["claim_boundary"]["authorized_training_exposure_loss_positions"], 0
        )
        self.assertFalse(result["claim_boundary"]["model_training_authorized"])

    def test_insufficient_family_count_fails_closed(self) -> None:
        families = [
            family("ua.a", "ua", 9_000_000),
            family("ua.b", "ua", 1),
            family("en.only", "en", 7_000_000),
            family("code.a", "code", 2_000_000),
            family("code.b", "code", 2_000_000),
        ]
        result = gate.evaluate(self.policy, vector(families))
        self.assertEqual(result["maximum_feasible_total_source_bytes"], 0)
        self.assertFalse(result["family_minimum"]["pass"])
        self.assertEqual(result["status"], "BLOCKED_NO_NONZERO_POLICY_COMPLIANT_MIXTURE")

    def test_family_caps_can_reduce_feasible_budget_below_raw_capacity(self) -> None:
        families = [
            family("ua.dominant", "ua", 8_900_000),
            family("ua.small", "ua", 100_000),
            family("en.a", "en", 3_500_000),
            family("en.b", "en", 3_500_000),
            family("code.a", "code", 2_000_000),
            family("code.b", "code", 2_000_000),
        ]
        result = gate.evaluate(self.policy, vector(families))
        self.assertEqual(result["maximum_feasible_total_source_bytes"], 500_000)
        self.assertEqual(result["status"], "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA")

    def test_nonterminal_dedup_is_rejected(self) -> None:
        data = vector(
            [
                family("ua.a", "ua", 1),
                family("ua.b", "ua", 1),
                family("en.a", "en", 1),
                family("en.b", "en", 1),
                family("code.a", "code", 1),
                family("code.b", "code", 1),
            ]
        )
        data["terminal"] = False
        with self.assertRaisesRegex(gate.GateError, "nonterminal"):
            gate.evaluate(self.policy, data)

    def test_duplicate_family_id_is_rejected_as_replay_like_credit(self) -> None:
        data = vector(
            [
                family("same", "ua", 100),
                family("same", "en", 100),
                family("ua.b", "ua", 100),
                family("en.b", "en", 100),
                family("code.a", "code", 100),
                family("code.b", "code", 100),
            ]
        )
        with self.assertRaisesRegex(gate.GateError, "duplicate family_id"):
            gate.evaluate(self.policy, data)

    def test_nonpositive_capacity_is_rejected(self) -> None:
        data = vector(
            [
                family("ua.a", "ua", 100),
                family("ua.b", "ua", 100),
                family("en.a", "en", 100),
                family("en.b", "en", 100),
                family("code.a", "code", 100),
                family("code.b", "code", 0),
            ]
        )
        with self.assertRaisesRegex(gate.GateError, "positive integer"):
            gate.evaluate(self.policy, data)

    def test_declared_totals_must_match_recomputed_vector(self) -> None:
        data = vector(
            [
                family("ua.a", "ua", 100),
                family("ua.b", "ua", 100),
                family("en.a", "en", 100),
                family("en.b", "en", 100),
                family("code.a", "code", 100),
                family("code.b", "code", 100),
            ]
        )
        data["totals"]["total_unique_bytes"] += 1
        with self.assertRaisesRegex(gate.GateError, "total_unique_bytes mismatch"):
            gate.evaluate(self.policy, data)

    def test_result_identity_is_deterministic(self) -> None:
        families = [
            family("ua.a", "ua", 4_500_000),
            family("ua.b", "ua", 4_500_000),
            family("en.a", "en", 3_500_000),
            family("en.b", "en", 3_500_000),
            family("code.a", "code", 2_000_000),
            family("code.b", "code", 2_000_000),
        ]
        first = gate.evaluate(self.policy, vector(families))
        second = gate.evaluate(self.policy, vector(list(reversed(families))))
        self.assertEqual(first, second)
        self.assertEqual(
            first["result_identity_sha256"],
            gate.canonical_sha(first, "result_identity_sha256"),
        )

    def test_policy_mutation_fails_identity_check(self) -> None:
        mutated = copy.deepcopy(self.policy)
        mutated["policy"]["target_total_source_bytes"] = 19_000_000
        with self.assertRaisesRegex(gate.GateError, "policy identity mismatch"):
            gate.validate_policy(mutated)


if __name__ == "__main__":
    unittest.main()
