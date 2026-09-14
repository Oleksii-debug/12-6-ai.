from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_next100_065f_global_dedup_v8.py"
SPEC = importlib.util.spec_from_file_location("next100_065f_v8", TOOL)
assert SPEC and SPEC.loader
v8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v8)


class Next100065FGlobalDedupV8Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs/data/next100_065f_global_dedup_v8.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_config_and_terminal_bulk_evidence_are_bound(self) -> None:
        v8._validate_config(self.config)
        v8._validate_bulk_terminal_evidence(ROOT, self.config)

    def test_rejects_main_authority_drift(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["main_authority"]["sha"] = "0" * 40
        with self.assertRaisesRegex(v8.V8Error, "main authority drift"):
            v8._validate_config(bad)

    def test_rejects_v7_identity_drift(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["baseline_v7"]["report_sha256"] = "f" * 64
        with self.assertRaisesRegex(v8.V8Error, "V7 report identity drift"):
            v8._validate_config(bad)

    def test_rejects_bulk_capacity_inflation(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["data_bulk_code1"]["eligible_utf8_bytes"] += 1
        with self.assertRaisesRegex(v8.V8Error, "bulk byte count drift"):
            v8._validate_config(bad)

    def test_rejects_composed_capacity_alias(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["expected_composed_input"]["source_capacity_bytes_before_global_dedup"] = 6_095_625
        with self.assertRaisesRegex(v8.V8Error, "composed capacity drift"):
            v8._validate_config(bad)

    def test_rejects_training_authority_promotion(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["claim_boundary"]["tokenizer_fit_authorized"] = True
        with self.assertRaisesRegex(v8.V8Error, "claim boundary weakened"):
            v8._validate_config(bad)

    def test_source_id_is_path_and_repository_bound(self) -> None:
        a = v8._source_id("pallets/flask", "src/flask/app.py")
        b = v8._source_id("pallets/flask", "src/flask/cli.py")
        c = v8._source_id("pallets/click", "src/flask/app.py")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_report_self_hash_is_stable_and_mutation_sensitive(self) -> None:
        report = {"schema_version": "x", "value": 7}
        first = v8._self_hash(report)
        self.assertEqual(first, v8._self_hash({"value": 7, "schema_version": "x"}))
        report["value"] = 8
        self.assertNotEqual(first, v8._self_hash(report))

    def test_terminal_evidence_mutation_fails_closed(self) -> None:
        evidence_path = ROOT / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "evidence/data_bulk_code1"
            target.mkdir(parents=True)
            evidence["aggregate"]["eligible_utf8_bytes"] += 1
            (target / "permissive_python_bundle_v1_terminal.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )
            with self.assertRaisesRegex(v8.V8Error, "terminal bulk byte count drift"):
                v8._validate_bulk_terminal_evidence(root, self.config)


if __name__ == "__main__":
    unittest.main()
