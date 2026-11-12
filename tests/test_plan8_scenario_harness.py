import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from twelve_six.scenario_harness import (
    Scenario,
    ScenarioStep,
    _digest,
    coverage_map,
    fixture_journey_adapters,
    parse_scenario_bytes,
    run_scenario,
)


def _commit(root: Path) -> str:
    subprocess.run(("git", "add", "--all"), cwd=root, check=True)
    subprocess.run(
        ("git", "-c", "user.name=Scenario test", "-c", "user.email=scenario@example.invalid",
         "commit", "--quiet", "--allow-empty", "-m", "Test scenario source"),
        cwd=root,
        check=True,
    )
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()


class ScenarioHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name).resolve()
        subprocess.run(("git", "init", "--quiet"), cwd=self.repo, check=True)
        (self.repo / "fixture.txt").write_text("1", encoding="utf-8")
        self.source_sha = _commit(self.repo)
        self.checkout = {"repo_root": self.repo, "source_sha": self.source_sha}
        self.available = frozenset({"fixture-ready"})
        self.adapters = {"fixture-adapter": lambda root: {"value": int(
            (root / "fixture.txt").read_text(encoding="utf-8")
        )}}

    def scenario(self):
        return Scenario(
            "fixture-journey", ("capability-a",), ("fixture-ready",),
            (ScenarioStep("readback", "fixture-adapter", _digest({"value": 1})),), "Q2",
        )

    def test_reproducible_receipt_and_physical_gap(self):
        scenario = self.scenario()
        kwargs = {**self.checkout, "available_prerequisites": self.available, "adapters": self.adapters}
        first = run_scenario(scenario, **kwargs)
        self.assertEqual(first, run_scenario(scenario, **kwargs))
        coverage = coverage_map((scenario,), (first,), expected_source_sha=self.source_sha)
        self.assertEqual(coverage["capability-a"]["q0_scenarios"], ["fixture-journey"])
        self.assertEqual(coverage["capability-a"]["missing_scenarios"], ["fixture-journey"])
        self.assertEqual(coverage["capability-a"]["missing_level"], "Q2")

    def test_missing_prerequisite_and_adapter_fail_closed(self):
        scenario = self.scenario()
        with self.assertRaisesRegex(ValueError, "prerequisites missing"):
            run_scenario(scenario, **self.checkout, available_prerequisites=frozenset(), adapters={})
        with self.assertRaisesRegex(ValueError, "adapter unavailable"):
            run_scenario(scenario, **self.checkout, available_prerequisites=self.available, adapters={})
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            run_scenario(scenario, **self.checkout, available_prerequisites=self.available,
                         adapters={"fixture-adapter": lambda _: {"value": 2}})

    def test_rejects_wrong_head_and_checkout_drift(self):
        scenario = self.scenario()
        with self.assertRaisesRegex(ValueError, "exact-head mismatch"):
            run_scenario(scenario, repo_root=self.repo, source_sha="a" * 40,
                         available_prerequisites=self.available, adapters=self.adapters)

        tracked = self.repo / "fixture.txt"
        tracked.write_text("2", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty|clean"):
            run_scenario(scenario, **self.checkout, available_prerequisites=self.available,
                         adapters=self.adapters)
        tracked.write_text("1", encoding="utf-8")

        def mutate_checkout(root: Path):
            (root / "fixture.txt").write_text("2", encoding="utf-8")
            return {"value": 1}

        with self.assertRaisesRegex(ValueError, "dirty|clean"):
            run_scenario(scenario, **self.checkout, available_prerequisites=self.available,
                         adapters={"fixture-adapter": mutate_checkout})

    def test_receipt_tampering_and_invalid_scenario_rejected(self):
        scenario = self.scenario()
        receipt = run_scenario(scenario, **self.checkout, available_prerequisites=self.available,
                               adapters=self.adapters)
        altered = dict(receipt, source_sha="b" * 40)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            coverage_map((scenario,), (altered,), expected_source_sha=self.source_sha)
        forged = dict(receipt, observations=[])
        forged["identity_sha256"] = _digest({k: v for k, v in forged.items() if k != "identity_sha256"})
        with self.assertRaisesRegex(ValueError, "observations differ"):
            coverage_map((scenario,), (forged,), expected_source_sha=self.source_sha)
        with self.assertRaisesRegex(ValueError, "duplicate scenario receipt"):
            coverage_map((scenario,), (receipt, receipt), expected_source_sha=self.source_sha)
        with self.assertRaisesRegex(ValueError, "duplicate step"):
            Scenario("bad", ("cap",), ("ready",),
                     (scenario.steps[0], scenario.steps[0]), "Q0")

    def test_coverage_rejects_mixed_source_and_exposes_partial_q0_gap(self):
        first = replace(self.scenario(), required_level="Q0")
        second = replace(first, scenario_id="second-journey")
        kwargs = {"available_prerequisites": self.available, "adapters": self.adapters}
        receipt = run_scenario(first, **self.checkout, **kwargs)
        incomplete = coverage_map((first, second), (receipt,), expected_source_sha=self.source_sha)
        self.assertEqual(incomplete["capability-a"]["missing_level"], "Q0")
        self.assertEqual(incomplete["capability-a"]["missing_scenarios"], ["second-journey"])
        covered = run_scenario(second, **self.checkout, **kwargs)
        complete = coverage_map((first, second), (receipt, covered), expected_source_sha=self.source_sha)
        self.assertIsNone(complete["capability-a"]["missing_level"])
        self.assertEqual(complete["capability-a"]["missing_scenarios"], [])

        newer = _commit(self.repo)
        foreign = run_scenario(second, repo_root=self.repo, source_sha=newer, **kwargs)
        with self.assertRaisesRegex(ValueError, "source SHA differs"):
            coverage_map((first, second), (receipt, foreign), expected_source_sha=self.source_sha)

    def test_manifest_roundtrip_and_fail_closed_json(self):
        scenario = self.scenario()
        raw = json.dumps(scenario.to_dict()).encode("utf-8")
        self.assertEqual(parse_scenario_bytes(raw), scenario)
        with self.assertRaisesRegex(ValueError, "duplicate scenario JSON key"):
            parse_scenario_bytes(raw.replace(b'"schema_version":',
                                             b'"schema_version": "duplicate", "schema_version":'))
        with self.assertRaisesRegex(ValueError, "bounded bytes"):
            parse_scenario_bytes(raw + b" " * 65_536)
        malformed = dict(scenario.to_dict(), capability_ids=[["not", "an", "identifier"]])
        with self.assertRaisesRegex(ValueError, "identifier is invalid"):
            parse_scenario_bytes(json.dumps(malformed).encode("utf-8"))

    def test_versioned_product_fixture_journeys_and_coverage_gaps(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "configs/plan8/scenarios/v1").glob("*.json"))
        self.assertEqual(len(paths), 3)
        scenarios = tuple(parse_scenario_bytes(path.read_bytes()) for path in paths)
        sha = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()
        receipts = tuple(
            run_scenario(scenario, repo_root=root, source_sha=sha,
                         available_prerequisites=frozenset(scenario.prerequisites))
            for scenario in scenarios
        )
        coverage = coverage_map(scenarios, receipts, expected_source_sha=sha)
        self.assertEqual(len(coverage), 3)
        for item in coverage.values():
            self.assertEqual(item["missing_level"], "Q2")
            self.assertEqual(len(item["q0_scenarios"]), 1)
            self.assertEqual(item["q0_scenarios"], item["missing_scenarios"])
        self.assertEqual(receipts, tuple(
            run_scenario(s, repo_root=root, source_sha=sha,
                         available_prerequisites=frozenset(s.prerequisites)) for s in scenarios
        ))
        agent = fixture_journey_adapters()["agent-goal"](root)["runtime"]
        self.assertEqual(agent["terminal_step_id"], "fixture-done")
        self.assertEqual(agent["effect_status"], "resolved")
        self.assertEqual(agent["receipt_id"], agent["memory_evidence_id"])
        self.assertEqual(agent["host_dispatches"], 1)
        self.assertEqual(agent["gateway_proposals"], 1)
        self.assertTrue(agent["restart_same_snapshot"])

    def test_fixture_journey_rejects_changed_source_and_lost_firewall(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "configs/data/plan2_reserved_eval_fixture_v1.json"
        scenario_path = root / "configs/plan8/scenarios/v1/plan2-reserved-eval-firewall.json"
        scenario = parse_scenario_bytes(scenario_path.read_bytes())
        local = self.repo / "configs/data/plan2_reserved_eval_fixture_v1.json"
        local.parent.mkdir(parents=True)
        local.write_bytes(source.read_bytes())
        args = {"repo_root": self.repo,
                "available_prerequisites": frozenset(scenario.prerequisites)}
        run_scenario(scenario, source_sha=_commit(self.repo), **args)
        local.write_bytes(source.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            run_scenario(scenario, source_sha=_commit(self.repo), **args)
        value = json.loads(source.read_text(encoding="utf-8"))
        value["training_allowed"] = True
        local.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            run_scenario(scenario, source_sha=_commit(self.repo), **args)

    def test_sil_fixture_bool_schema_cannot_alias_version_one(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "configs/control/sil_scenario_v1.json"
        target = self.repo / "configs/control/sil_scenario_v1.json"
        target.parent.mkdir(parents=True)
        fixture = json.loads(source.read_text(encoding="utf-8"))
        fixture["schema_version"] = True
        target.write_text(json.dumps(fixture), encoding="utf-8")
        adapter = fixture_journey_adapters()["sil-journey"]
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            adapter(self.repo)
        fixture.update(schema_version=1, journey_selector="UNAPPROVED")
        target.write_text(json.dumps(fixture), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ALL_AVAILABLE"):
            adapter(self.repo)


if __name__ == "__main__":
    unittest.main()
