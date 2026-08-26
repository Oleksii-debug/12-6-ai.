from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from twelve_six.local_assistant.authority import AUTHORITIES, CapabilityGate, CapabilityUnavailableError
from twelve_six.local_assistant.orchestrator import LocalAssistantOrchestrator, RunOptions, write_trace


class LocalAssistantOrchestratorTests(unittest.TestCase):
    def test_nonterminal_mock_tools_fail_closed_before_execution(self) -> None:
        orchestrator = LocalAssistantOrchestrator()
        with self.assertRaises(CapabilityUnavailableError):
            orchestrator.run(
                "calc:2+2",
                RunOptions(mock_model=True, use_mock_tools=True),
            )

    def test_terminal_hypothesis_and_memory_surfaces_are_composable(self) -> None:
        result = LocalAssistantOrchestrator().run(
            "fixture",
            RunOptions(
                mock_model=True,
                use_hypothesis_search=True,
                memory_db=":memory:",
            ),
        )
        self.assertEqual(result.text, "fixture")
        self.assertIn("hypothesis_before", result.trace)
        self.assertIn("hypothesis_after", result.trace)
        self.assertEqual(result.trace["memory"]["evidence"], [])
        self.assertTrue(result.trace["authorities"]["hypothesis_search"]["terminal"])
        self.assertTrue(result.trace["authorities"]["memory_rag"]["terminal"])

    def test_probe_is_plain_text_and_machine_trace_is_safety_bounded(self) -> None:
        result = LocalAssistantOrchestrator().run(
            "LOCAL_FREE_PROBE",
            RunOptions(mock_model=True, expected_answer_fixture="LOCAL_FREE_PROBE"),
        )
        self.assertEqual(result.text, "LOCAL_FREE_PROBE")
        self.assertEqual(result.trace["execution_profile"], "LOCAL_FREE")
        self.assertFalse(result.trace["safety"]["base_weights_modified"])
        self.assertFalse(result.trace["safety"]["training_executed"])
        self.assertFalse(result.trace["safety"]["external_llm_used"])
        self.assertFalse(result.trace["safety"]["chat_personality_claim"])
        self.assertEqual(result.trace["verifier"]["records"][0]["status"], "PASS")
        self.assertTrue(result.trace["verifier"]["records"][0]["fixture_only"])
        self.assertNotIn("private_scratch", json.dumps(result.trace))

    def test_trace_write_is_deterministic_structure(self) -> None:
        result = LocalAssistantOrchestrator().run(
            "LOCAL_FREE_PROBE",
            RunOptions(mock_model=True, expected_answer_fixture="LOCAL_FREE_PROBE"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_trace(path, result.trace)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema"], "12-6.local-assistant-orchestration.v1")
        self.assertEqual(loaded["output_sha256"], result.trace["output_sha256"])

    def test_mock_tool_bridge_works_only_when_authority_is_explicitly_terminal(self) -> None:
        authorities = dict(AUTHORITIES)
        authorities["mock_tools"] = replace(
            authorities["mock_tools"],
            terminal=True,
            source_status="TEST_ONLY_TERMINAL_OVERRIDE",
            reason="project-owned test override",
        )
        result = LocalAssistantOrchestrator(CapabilityGate(authorities)).run(
            "calc:2+2",
            RunOptions(
                mock_model=True,
                use_mock_tools=True,
                expected_answer_fixture="4",
                max_model_calls=4,
            ),
        )
        self.assertEqual(result.text, "4")
        self.assertEqual(len(result.trace["mock_tools"]), 1)
        self.assertFalse(result.trace["mock_tools"][0]["training_eligible"])
        self.assertFalse(result.trace["mock_tools"][0]["weight_update_eligible"])


if __name__ == "__main__":
    unittest.main()
