from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_authority_supersession.py"
GRAPH_PATH = ROOT / "evidence" / "ci286" / "authority-supersession.v1.json"

spec = importlib.util.spec_from_file_location("ci286_validator", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class AuthoritySupersessionGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    def test_committed_graph_is_valid(self) -> None:
        validator.validate_graph(copy.deepcopy(self.graph))

    def test_circular_supersession_fails_closed(self) -> None:
        graph = copy.deepcopy(self.graph)
        graph["edges"].extend(
            [
                {
                    "type": "supersedes",
                    "from": "CHECKPOINT-211",
                    "to": "PERF-250",
                    "reason": "synthetic cycle test",
                },
                {
                    "type": "supersedes",
                    "from": "PERF-250",
                    "to": "CHECKPOINT-211",
                    "reason": "synthetic cycle test",
                },
            ]
        )
        with self.assertRaisesRegex(
            validator.GraphValidationError, "circular supersession"
        ):
            validator.validate_graph(graph)

    def test_duplicate_incumbent_fails_closed(self) -> None:
        graph = copy.deepcopy(self.graph)
        duplicate = copy.deepcopy(
            next(node for node in graph["nodes"] if node["id"] == "TRAIN-243")
        )
        duplicate["id"] = "TRAIN-243-SYNTHETIC-INCUMBENT"
        duplicate["disposition"] = "incumbent"
        duplicate["evidence_state"] = "terminal_green"
        duplicate["source"]["head_sha"] = "1" * 40
        duplicate.pop("non_authority_reason", None)
        graph["nodes"].append(duplicate)
        with self.assertRaisesRegex(
            validator.GraphValidationError, "expected exactly one incumbent"
        ):
            validator.validate_graph(graph)


if __name__ == "__main__":
    unittest.main()
