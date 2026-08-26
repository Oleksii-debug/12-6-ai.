from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_research_corpus_v1_source_intake import DEFAULT_INTAKE, DEFAULT_PARENT, validate


class ResearchCorpusV1SourceIntakeTests(unittest.TestCase):
    def _mutated(self, mutate):
        payload = json.loads(DEFAULT_INTAKE.read_text(encoding="utf-8"))
        mutate(payload)
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "intake.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return directory, path

    def test_committed_intake_passes(self) -> None:
        validated = validate(DEFAULT_INTAKE, DEFAULT_PARENT)
        self.assertEqual(validated["readiness"]["authorized_unique_loss_positions"], 0)
        self.assertFalse(validated["readiness"]["long_training_authorized"])

    def test_training_ready_lie_fails_closed(self) -> None:
        directory, path = self._mutated(
            lambda payload: payload["readiness"].__setitem__("long_training_authorized", True)
        )
        with directory, self.assertRaises(AssertionError):
            validate(path, DEFAULT_PARENT)

    def test_authority_run_drift_fails_closed(self) -> None:
        def mutate(payload):
            payload["additive_terminal_authorities"][0]["dedicated_workflow_run_id"] += 1
        directory, path = self._mutated(mutate)
        with directory, self.assertRaises(AssertionError):
            validate(path, DEFAULT_PARENT)

    def test_duplicate_family_credit_fails_closed(self) -> None:
        def mutate(payload):
            duplicate = copy.deepcopy(payload["additive_terminal_authorities"][0])
            duplicate["worker"] = "NEXT100-038-DATA-EN-MDN"
            duplicate["head_sha"] = "902eccc0b3efff09a38dc89cda789180b6c6e754"
            duplicate["dedicated_workflow_run_id"] = 32998544359
            payload["additive_terminal_authorities"][4] = duplicate
        directory, path = self._mutated(mutate)
        with directory, self.assertRaises(AssertionError):
            validate(path, DEFAULT_PARENT)


if __name__ == "__main__":
    unittest.main()
