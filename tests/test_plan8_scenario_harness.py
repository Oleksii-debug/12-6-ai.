import json
import unittest
from pathlib import Path

from twelve_six.scenario_harness import (
    Scenario,
    ScenarioStep,
    _digest,
    coverage_map,
    fixture_journey_adapters,
    parse_scenario_bytes,
    run_scenario,
)


class ScenarioHarnessTests(unittest.TestCase):
    def scenario(self):
        return Scenario(
            "fixture-journey", ("capability-a",), ("fixture-ready",),
            (ScenarioStep("readback", "fixture-adapter", _digest({"value": 1})),), "Q2",
        )

    def test_reproducible_receipt_and_physical_gap(self):
        scenario = self.scenario()
        kwargs = {"source_sha": "a" * 40, "available_prerequisites": frozenset({"fixture-ready"}),
                  "adapters": {"fixture-adapter": lambda: {"value": 1}}}
        first = run_scenario(scenario, **kwargs)
        self.assertEqual(first, run_scenario(scenario, **kwargs))
        coverage = coverage_map((scenario,), (first,))
        self.assertEqual(coverage["capability-a"]["q0_scenarios"], ["fixture-journey"])
        self.assertEqual(coverage["capability-a"]["missing_level"], "Q2")

    def test_missing_prerequisite_and_adapter_fail_closed(self):
        scenario = self.scenario()
        with self.assertRaisesRegex(ValueError, "prerequisites missing"):
            run_scenario(scenario, source_sha="a" * 40,
                         available_prerequisites=frozenset(), adapters={})
        with self.assertRaisesRegex(ValueError, "adapter unavailable"):
            run_scenario(scenario, source_sha="a" * 40,
                         available_prerequisites=frozenset({"fixture-ready"}), adapters={})
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            run_scenario(scenario, source_sha="a" * 40,
                         available_prerequisites=frozenset({"fixture-ready"}),
                         adapters={"fixture-adapter": lambda: {"value": 2}})

    def test_receipt_tampering_and_invalid_scenario_rejected(self):
        scenario = self.scenario()
        receipt = run_scenario(scenario, source_sha="a" * 40,
                               available_prerequisites=frozenset({"fixture-ready"}),
                               adapters={"fixture-adapter": lambda: {"value": 1}})
        altered = dict(receipt, source_sha="b" * 40)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            coverage_map((scenario,), (altered,))
        forged = dict(receipt, observations=[])
        forged["identity_sha256"] = _digest({k: v for k, v in forged.items() if k != "identity_sha256"})
        with self.assertRaisesRegex(ValueError, "observations differ"):
            coverage_map((scenario,), (forged,))
        with self.assertRaisesRegex(ValueError, "duplicate scenario receipt"):
            coverage_map((scenario,), (receipt, receipt))
        with self.assertRaisesRegex(ValueError, "duplicate step"):
            Scenario("bad", ("cap",), ("ready",),
                     (scenario.steps[0], scenario.steps[0]), "Q0")

    def test_manifest_roundtrip_and_fail_closed_json(self):
        scenario = self.scenario()
        raw = json.dumps(scenario.to_dict()).encode("utf-8")
        self.assertEqual(parse_scenario_bytes(raw), scenario)
        with self.assertRaisesRegex(ValueError, "duplicate scenario JSON key"):
            parse_scenario_bytes(raw.replace(b'"schema_version":',
                                             b'"schema_version": "duplicate", "schema_version":'))
        with self.assertRaisesRegex(ValueError, "bounded bytes"):
            parse_scenario_bytes(raw + b" " * 65_536)

    def test_versioned_product_fixture_journeys_and_coverage_gaps(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "configs/plan8/scenarios/v1").glob("*.json"))
        self.assertEqual(len(paths), 3)
        scenarios = tuple(parse_scenario_bytes(path.read_bytes()) for path in paths)
        adapters = fixture_journey_adapters(root)
        receipts = tuple(
            run_scenario(scenario, source_sha="a" * 40,
                         available_prerequisites=frozenset(scenario.prerequisites),
                         adapters=adapters)
            for scenario in scenarios
        )
        coverage = coverage_map(scenarios, receipts)
        self.assertEqual(len(coverage), 3)
        for item in coverage.values():
            self.assertEqual(item["missing_level"], "Q2")
            self.assertEqual(len(item["q0_scenarios"]), 1)
        self.assertEqual(receipts, tuple(
            run_scenario(s, source_sha="a" * 40,
                         available_prerequisites=frozenset(s.prerequisites),
                         adapters=adapters) for s in scenarios
        ))

    def test_fixture_journey_rejects_changed_source_and_lost_firewall(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "configs/data/plan2_reserved_eval_fixture_v1.json"
        scenario_path = root / "configs/plan8/scenarios/v1/plan2-reserved-eval-firewall.json"
        scenario = parse_scenario_bytes(scenario_path.read_bytes())
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            local = Path(directory) / "configs/data/plan2_reserved_eval_fixture_v1.json"
            local.parent.mkdir(parents=True)
            local.write_bytes(source.read_bytes())
            args = {"source_sha": "a" * 40,
                    "available_prerequisites": frozenset(scenario.prerequisites),
                    "adapters": fixture_journey_adapters(directory)}
            run_scenario(scenario, **args)
            local.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "observation mismatch"):
                run_scenario(scenario, **args)
            value = json.loads(source.read_text(encoding="utf-8"))
            value["training_allowed"] = True
            local.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "adapter failed"):
                run_scenario(scenario, **args)


if __name__ == "__main__":
    unittest.main()
