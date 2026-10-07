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
        self.assertEqual(
            config["contract_identity_sha256"],
            MOD.EXPECTED_CONTRACT_IDENTITY,
        )
        self.assertEqual(len(config["sources"]), 6)
        self.assertEqual(len({source["family_id"] for source in config["sources"]}), 6)
        self.assertEqual(config["truth_boundary"]["authorized_training_exposure"], 0)
        self.assertIs(
            config["rights_boundary"]["automatic_canonical_capacity_credit"],
            False,
        )

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
            record, exclusion = MOD._eligible_file_record(
                repo,
                root,
                path,
                config["selection_policy"],
            )
            self.assertIsNone(exclusion)
            self.assertEqual(
                record,
                {
                    "path": "src/pkg/module.py",
                    "sha256": MOD._sha256(raw),
                    "utf8_bytes": len(raw),
                },
            )

    def test_credential_pattern_hard_fails_in_core_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "secret.py"
            path.write_bytes(b"TOKEN = 'AKIAABCDEFGHIJKLMNOP'\n")
            config = MOD.load_contract(CONFIG_PATH)
            with self.assertRaisesRegex(
                MOD.MaterializationError,
                "credential/secret pattern",
            ):
                MOD._eligible_file_record(
                    repo,
                    root,
                    path,
                    config["selection_policy"],
                )

    def test_generated_marker_receives_zero_credit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "generated.py"
            path.write_text(
                "# This file is auto-generated; do not edit.\nVALUE = 1\n",
                encoding="utf-8",
            )
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(
                repo,
                root,
                path,
                config["selection_policy"],
            )
            self.assertIsNone(record)
            self.assertIsNotNone(exclusion)
            assert exclusion is not None
            self.assertEqual(exclusion["reason"], "generated_marker")

    def test_invalid_python_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            root = repo / "src" / "pkg"
            root.mkdir(parents=True)
            path = root / "broken.py"
            path.write_text("def broken(:\n", encoding="utf-8")
            config = MOD.load_contract(CONFIG_PATH)
            record, exclusion = MOD._eligible_file_record(
                repo,
                root,
                path,
                config["selection_policy"],
            )
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
            record, exclusion = MOD._eligible_file_record(
                repo,
                root,
                path,
                config["selection_policy"],
            )
            self.assertIsNone(record)
            self.assertEqual(
                exclusion,
                {
                    "path": "src/pkg/tests/test_example.py",
                    "reason": "excluded_directory_component",
                },
            )

    def test_report_validator_rejects_credential_exclusion(self) -> None:
        config = MOD.load_contract(CONFIG_PATH)
        report = self._synthetic_report(config)
        report["sources"][0]["exclusions"] = [
            {
                "path": "src/flask/secret.py",
                "reason": "credential_pattern",
                "patterns": ["aws_access_key"],
            }
        ]
        report["report_identity_sha256"] = self._report_identity(report)
        with self.assertRaisesRegex(
            MOD.MaterializationError,
            "harmless exclusion",
        ):
            MOD.validate_report(config, report)

    def test_report_validator_rejects_source_commit_drift(self) -> None:
        config = MOD.load_contract(CONFIG_PATH)
        report = self._synthetic_report(config)
        report["sources"][0]["commit"] = "0" * 40
        report["report_identity_sha256"] = self._report_identity(report)
        with self.assertRaisesRegex(MOD.MaterializationError, "source commit mismatch"):
            MOD.validate_report(config, report)

    def test_report_validator_accepts_exact_synthetic_shape(self) -> None:
        config = MOD.load_contract(CONFIG_PATH)
        report = self._synthetic_report(config)
        MOD.validate_report(config, report)

    @staticmethod
    def _report_identity(report: dict[str, object]) -> str:
        core = dict(report)
        core.pop("report_identity_sha256", None)
        return MOD._sha256(MOD._canonical_bytes(core))

    def _synthetic_report(self, config: dict[str, object]) -> dict[str, object]:
        sources = []
        for index, source in enumerate(config["sources"]):
            assert isinstance(source, dict)
            raw = f"source-{index}".encode()
            file_record = {
                "path": f"{source['source_root']}/module_{index}.py",
                "sha256": MOD._sha256(raw),
                "utf8_bytes": len(raw),
            }
            license_record = {
                "license_id": source["license_id"],
                "license_path": source["license_path"],
                "license_git_blob_sha1": source["license_git_blob_sha1"],
                "license_sha256": MOD._sha256(f"license-{index}".encode()),
            }
            source_core = {
                "repository": source["repository"],
                "family_id": source["family_id"],
                "commit": source["commit"],
                "license": license_record,
                "files": [file_record],
            }
            sources.append(
                {
                    "repository": source["repository"],
                    "family_id": source["family_id"],
                    "commit": source["commit"],
                    "source_root": source["source_root"],
                    "license": license_record,
                    "eligible_file_count": 1,
                    "eligible_utf8_bytes": len(raw),
                    "excluded_file_count": 0,
                    "exclusions": [],
                    "files": [file_record],
                    "family_materialization_identity_sha256": MOD._sha256(
                        MOD._canonical_bytes(source_core)
                    ),
                }
            )
        report = {
            "schema_version": MOD.REPORT_SCHEMA,
            "worker_id": config["worker_id"],
            "contract_identity_sha256": config["contract_identity_sha256"],
            "execution_profile": "LOCAL_FREE",
            "source_family_count": 6,
            "eligible_file_count": 6,
            "eligible_utf8_bytes": sum(
                source["eligible_utf8_bytes"] for source in sources
            ),
            "sources": sources,
            "security": {
                "credential_scan": "PASS_NO_HIGH_CONFIDENCE_HITS",
                "credential_patterns": sorted(MOD.CREDENTIAL_PATTERNS),
            },
            "purpose_firewall": {
                "model_training_source_use": config["rights_boundary"][
                    "model_training_source_use"
                ],
                "evaluation": "NOT_SEPARATELY_ADMITTED",
                "final_test": "PROHIBITED",
                "automatic_canonical_capacity_credit": False,
            },
            "downstream_required": config["downstream_required"],
            "claim_boundary": {
                "corpus_identity": None,
                "shard_identity": None,
                "post_dedup_capacity_claimed": False,
                "authorized_training_exposure": 0,
                "tokenizer_fit_authorized": False,
                "long_training_authorized": False,
                "paid_compute_authorized": False,
            },
        }
        report["report_identity_sha256"] = self._report_identity(report)
        return report


if __name__ == "__main__":
    unittest.main()
