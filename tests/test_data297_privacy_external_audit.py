from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _load_audit_module_for_test():
    repository_root = Path(__file__).resolve().parents[1]
    package_root = repository_root / "src" / "twelve_six"
    data_root = package_root / "data"

    package = types.ModuleType("twelve_six")
    package.__path__ = [str(package_root)]
    package.__package__ = "twelve_six"
    sys.modules.setdefault("twelve_six", package)

    data_package = types.ModuleType("twelve_six.data")
    data_package.__path__ = [str(data_root)]
    data_package.__package__ = "twelve_six.data"
    sys.modules.setdefault("twelve_six.data", data_package)

    module_name = "twelve_six.data297_privacy_external_audit"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name, package_root / "data297_privacy_external_audit.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load DATA-297 audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_audit_module_for_test()
_fixture_specs = _audit._fixture_specs
_resolve_data227_code = _audit._resolve_data227_code
_resolve_data229_text = _audit._resolve_data229_text
assert_incumbent_privacy_authority = _audit.assert_incumbent_privacy_authority
audit_labeled_fixtures = _audit.audit_labeled_fixtures
load_config = _audit.load_config


class Data297PrivacyExternalAuditTests(unittest.TestCase):
    def test_exact_data33_authority_is_reused(self) -> None:
        config = load_config()
        self.assertEqual(
            assert_incumbent_privacy_authority(config),
            "8c905e3b8f81391c3f928f375bca8fe6d1b5d38b41dec5c081577e2c5ce58526",
        )
        self.assertEqual(config["privacy_authority"]["worker"], "DATA-33-PII-SECRETS")

    def test_complete_terminal_inventory_boundary_is_fixed(self) -> None:
        config = load_config()
        expected = config["expected_training_inventory"]
        self.assertEqual(expected["source_count"], 5)
        self.assertEqual(expected["family_count"], 4)
        self.assertEqual(expected["input_utf8_bytes"], 183061)
        self.assertEqual(
            expected["by_family_input_utf8_bytes"],
            {
                "en.standardebooks.manual": 84793,
                "github:encode/httpx": 8161,
                "github:psf/requests": 1542,
                "ua.rada.open-data.laws-texts": 88565,
            },
        )
        self.assertEqual(
            config["inventory_authorities"]["DATA-228"]["status"],
            "NONTERMINAL_EXCLUDED_FROM_TRAINING_INVENTORY",
        )
        hashes = {
            item["source_id"]: item["expected_input_sha256"]
            for item in config["admitted_inventory"]
        }
        self.assertEqual(
            hashes["code.encode.httpx._content"],
            "2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28",
        )
        self.assertEqual(
            hashes["code.psf.requests._internal_utils"],
            "4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff",
        )

    def test_fixture_confusion_matrix_measures_current_gaps(self) -> None:
        metrics = audit_labeled_fixtures()
        self.assertEqual(metrics["fixture_count"], 22)
        self.assertEqual(metrics["positive_fixture_count"], 15)
        self.assertEqual(metrics["negative_fixture_count"], 7)
        self.assertEqual(
            metrics["confusion"],
            {
                "false_negative": 7,
                "false_positive": 0,
                "true_negative": 7,
                "true_positive": 8,
            },
        )
        self.assertAlmostEqual(metrics["false_negative_rate"], 7 / 15)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertEqual(
            set(metrics["false_negative_fixture_ids"]),
            {
                "positive-basic-auth-header",
                "positive-gitlab-token-shape",
                "positive-huggingface-token-shape",
                "positive-npm-auth-token-shape",
                "positive-openai-style-key-shape",
                "positive-private-unix-ssh-path",
                "positive-private-windows-cloud-path",
            },
        )

    def test_fixture_metrics_never_retain_fixture_payloads(self) -> None:
        serialized = json.dumps(audit_labeled_fixtures(), ensure_ascii=False, sort_keys=True)
        for fixture in _fixture_specs():
            self.assertNotIn(fixture.build(), serialized)
        self.assertEqual(audit_labeled_fixtures()["payload_retention"], "NONE")

    def test_data229_resolver_accepts_exact_normalized_bytes_with_artifact_newline(self) -> None:
        payload = "Synthetic normalized document."
        encoded = payload.encode("utf-8")
        expected = {
            "expected_input_utf8_bytes": len(encoded),
            "expected_input_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "normalized").mkdir()
            (root / "normalized" / "fixture.txt").write_bytes(encoded + b"\n")
            self.assertEqual(_resolve_data229_text(root, expected), payload)

    def test_data227_resolver_binds_report_registry_and_off_git_payload(self) -> None:
        payload = b"print('fixture')\n"
        digest = hashlib.sha256(payload).hexdigest()
        expected = {
            "source_id": "code.fixture.repo.file",
            "expected_input_utf8_bytes": len(payload),
            "expected_input_sha256": digest,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_dir = root / "data227-evidence"
            report_dir.mkdir(parents=True)
            (report_dir / "data227-real-code-source-admission.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "source_id": expected["source_id"],
                                "raw_sha256": digest,
                                "normalization_sha256": digest,
                            }
                        ],
                        "registry": {
                            "sources": [
                                {
                                    "source_id": expected["source_id"],
                                    "snapshot": {
                                        "sha256": digest,
                                        "size_bytes": len(payload),
                                    },
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            snapshot = root / "data" / "external" / "snapshots" / "sha256" / digest
            snapshot.mkdir(parents=True)
            (snapshot / "payload").write_bytes(payload)
            text, actual_digest = _resolve_data227_code(root, expected)
            self.assertEqual(text.encode("utf-8"), payload)
            self.assertEqual(actual_digest, digest)


if __name__ == "__main__":
    unittest.main()
