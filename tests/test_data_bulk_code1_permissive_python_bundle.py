from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "data_bulk_code1",
    ROOT / "tools" / "materialize_data_bulk_code1_permissive_python_bundle.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
CONFIG_PATH = ROOT / "configs/data/data_bulk_code1_permissive_python_bundle_v1.json"


class DataBulkCode1Tests(unittest.TestCase):
    def test_contract_is_exact_and_fail_closed(self) -> None:
        config = MOD.load_contract(CONFIG_PATH)
        self.assertEqual(config["contract_identity_sha256"], MOD.EXPECTED_CONTRACT_IDENTITY)
        self.assertEqual(len(config["sources"]), 6)
        self.assertEqual(len({source["family_id"] for source in config["sources"]}), 6)
        self.assertEqual(config["truth_boundary"]["authorized_training_exposure"], 0)
        self.assertIs(config["rights_boundary"]["automatic_canonical_capacity_credit"], False)

    def test_contract_rejects_training_or_capacity_promotion(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["truth_boundary"]["authorized_training_exposure"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(MOD.MaterializationError):
                MOD.load_contract(path)

    def test_eligible_python_file_records_exact_bytes_and_ast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "module.py"
            raw = b"def add(a, b):\n    return a + b\n"
            path.write_bytes(raw)
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
            self.assertIsNone(exclusion)
            self.assertEqual(
                record,
                {
                    "path": "src/pkg/module.py",
                    "sha256": MOD._sha256(raw),
                    "utf8_bytes": len(raw),
                },
            )

    def test_credential_pattern_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "secret.py"
            path.write_bytes(b"TOKEN = 'AKIAABCDEFGHIJKLMNOP'\n")
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
            self.assertIsNone(record)
            self.assertIsNotNone(exclusion)
            assert exclusion is not None
            self.assertEqual(exclusion["reason"], "credential_pattern")
            self.assertIn("aws_access_key", exclusion["patterns"])

    def test_invalid_python_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "broken.py"
            path.write_text("def broken(:\n", encoding="utf-8")
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
            self.assertIsNone(record)
            self.assertIsNotNone(exclusion)
            assert exclusion is not None
            self.assertEqual(exclusion["reason"], "ast_parse_failure")

    def test_excluded_directory_receives_no_credit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            path = root / "tests" / "test_example.py"
            path.parent.mkdir(parents=True)
            path.write_text("assert True\n", encoding="utf-8")
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
            self.assertIsNone(record)
            self.assertEqual(
                exclusion,
                {"path": "src/pkg/tests/test_example.py", "reason": "excluded_directory_component"},
            )


if __name__ == "__main__":
    unittest.main()
