from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.materialize_data526_records_from_v7 import (
    _load_data213_payloads,
    _normalize_kmu,
    _split_cpython,
    materialize_records,
    verify_config,
)

CONFIG = Path("configs/data/data526_record_materialization_v5.json")


class Data526RecordMaterializationV5Tests(unittest.TestCase):
    def test_committed_config_identity_and_truth_boundary(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        verify_config(config)
        self.assertEqual(config["expected_materialization"]["record_count"], 48)
        self.assertEqual(config["expected_materialization"]["total_payload_bytes"], 2_215_615)
        self.assertEqual(config["claim_boundary"]["authorized_unique_optimized_targets"], 0)

    def test_data213_wrapper_drops_exactly_one_bound_trailing_lf(self) -> None:
        artifact_payload = b"abc\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "data213.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as package:
                package.writestr("normalized/record.txt", artifact_payload)
            archive_bytes = archive.read_bytes()
            config = {
                "data213_normalized_artifact": {
                    "artifact_zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "sources": {
                        "source": {
                            "artifact_file_bytes": len(artifact_payload),
                            "artifact_file_sha256": hashlib.sha256(artifact_payload).hexdigest(),
                            "path": "normalized/record.txt",
                            "payload_bytes": 3,
                            "payload_sha256": hashlib.sha256(b"abc").hexdigest(),
                            "payload_transform": "DROP_EXACT_ONE_TRAILING_LF",
                        }
                    },
                }
            }
            self.assertEqual(_load_data213_payloads(archive, config), {"source": b"abc"})

            weakened = copy.deepcopy(config)
            weakened["data213_normalized_artifact"]["sources"]["source"]["payload_transform"] = "STRIP"
            with self.assertRaisesRegex(ValueError, "unsupported DATA-213 payload transform"):
                _load_data213_payloads(archive, weakened)

    def test_kmu_normalization_is_exact_and_trailing_lf_bounded(self) -> None:
        raw = "  A\u0308  Б  \n\n  В\tГ \n".encode("utf-8")
        normalized = _normalize_kmu(raw)
        self.assertEqual(normalized.decode("utf-8"), "Ä Б\nВ Г\n")
        self.assertTrue(normalized.endswith(b"\n"))
        self.assertFalse(normalized.endswith(b"\n\n"))

    def test_cpython_separator_cannot_silently_change_record_identity(self) -> None:
        chunks = ["alpha\nrecord".encode(), "beta".encode()]
        spec = {
            "comparison_separator": "DOUBLE_LF",
            "accepted_normalized_sha256": [hashlib.sha256(chunk).hexdigest() for chunk in chunks],
            "accepted_chunk_count": 2,
            "accepted_capacity_bytes": sum(len(chunk) for chunk in chunks),
        }
        self.assertEqual(_split_cpython(b"\n\n".join(chunks), spec), chunks)
        with self.assertRaisesRegex(ValueError, "identity/order drift"):
            _split_cpython(b"beta\n\nalpha\nrecord", spec)

    def test_direct_v7_source_requires_exact_capacity_and_comparison_hash(self) -> None:
        payload = b"hello\n"
        source = {"source_id": "s1", "source_family": "family", "modality": "en"}
        report_source = {
            **source,
            "declared_capacity_bytes": len(payload),
            "comparison_payload_bytes": len(payload),
            "comparison_payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        config = {
            "data213_normalized_artifact": {"sources": {}},
            "kmu_authority": {"sources": {}},
            "cpython_authority": {
                "source_id": "never",
                "comparison_separator": "DOUBLE_LF",
                "accepted_normalized_sha256": [],
                "accepted_chunk_count": 0,
                "accepted_capacity_bytes": 0,
            },
            "expected_materialization": {
                "source_object_count": 1,
                "direct_v7_source_count": 1,
                "direct_v7_payload_bytes": len(payload),
                "record_count": 1,
                "total_payload_bytes": len(payload),
                "by_modality_payload_bytes": {"uk": 0, "en": len(payload), "code": 0},
            },
        }
        records, stats = materialize_records(
            config=config,
            inventory={"sources": [source]},
            payloads={"s1": payload},
            v7_report={"dedup_v3": {"sources": [report_source]}},
            data213_payloads={},
        )
        self.assertEqual(records[0]["normalized_payload"], "hello\n")
        self.assertEqual(stats["direct_v7_source_count"], 1)

        bad_report = copy.deepcopy(report_source)
        bad_report["comparison_payload_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "explicit authority adapter"):
            materialize_records(
                config=config,
                inventory={"sources": [source]},
                payloads={"s1": payload},
                v7_report={"dedup_v3": {"sources": [bad_report]}},
                data213_payloads={},
            )


if __name__ == "__main__":
    unittest.main()
