import unittest

from twelve_six.scenario_harness import Scenario, ScenarioStep, _digest, coverage_map, run_scenario


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
        with self.assertRaisesRegex(ValueError, "duplicate step"):
            Scenario("bad", ("cap",), ("ready",),
                     (scenario.steps[0], scenario.steps[0]), "Q0")


if __name__ == "__main__":
    unittest.main()
