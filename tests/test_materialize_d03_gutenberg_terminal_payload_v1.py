from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_d03_gutenberg_terminal_payload_v1.py"
CONFIG = ROOT / "configs" / "data" / "next100_107_gutenberg_terminal_seal_v1.json"

spec = importlib.util.spec_from_file_location("gutenberg_payload_materializer", TOOL)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def git_blob_sha1(data: bytes) -> str:
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


class GutenbergTerminalPayloadMaterializerTests(unittest.TestCase):
    def test_current_terminal_seal_is_exactly_bound(self):
        seal = module.load_terminal_seal(CONFIG)
        self.assertEqual(
            seal["authority_identity_sha256"],
            module.EXPECTED_AUTHORITY,
        )
        self.assertEqual(
            seal["source_family"]["normalized_utf8_bytes"],
            1_672_110,
        )

    def test_seal_byte_substitution_fails_before_transport(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["records"][0]["raw_bytes"] += 1
        with tempfile.TemporaryDirectory() as td:
            candidate = Path(td) / "seal.json"
            candidate.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                module.MaterializationError,
                "terminal seal Git blob identity drift",
            ):
                module.load_terminal_seal(candidate)

    def test_historical_normalizer_rules_are_preserved(self):
        raw = (
            b"outside\r\n"
            b"*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
            b"\r\n"
            b"Cafe\xcc\x81\r\n"
            b"line two\r\n"
            b"\r\n"
            b"*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n"
            b"outside"
        )
        normalized = module.normalize_pg_body(raw, "utf-8")
        self.assertEqual(normalized, "Caf\u00e9\nline two\n".encode())

    def test_duplicate_marker_fails_closed(self):
        raw = (
            b"*** START OF THE PROJECT GUTENBERG EBOOK A ***\n"
            b"*** START OF THE PROJECT GUTENBERG EBOOK B ***\n"
            b"body\n"
            b"*** END OF THE PROJECT GUTENBERG EBOOK A ***\n"
        )
        with self.assertRaisesRegex(module.MaterializationError, "exactly one Gutenberg START"):
            module.normalize_pg_body(raw, "ascii")

    def test_record_transport_and_normalized_identity_are_both_required(self):
        raw = (
            b"prefix\n"
            b"*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            b"\nhello world\n\n"
            b"*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            b"suffix\n"
        )
        normalized = b"hello world\n"
        record = {
            "source_id": "synthetic",
            "encoding": "ascii",
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "transport_git_blob_sha1": git_blob_sha1(raw),
            "normalized_utf8_bytes": len(normalized),
            "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        }
        self.assertEqual(module.verify_and_normalize_record(record, raw), normalized)

        wrong = dict(record)
        wrong["raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(module.MaterializationError, "raw SHA-256 drift"):
            module.verify_and_normalize_record(wrong, raw)

    def test_receipt_preserves_zero_credit_boundary_and_contains_no_payload_text(self):
        seal = json.loads(CONFIG.read_text(encoding="utf-8"))
        rows = []
        for record in seal["records"]:
            rows.append(
                {
                    "source_id": record["source_id"],
                    "source_family": module.EXPECTED_FAMILY,
                    "stable_origin_id": module.EXPECTED_FAMILY,
                    "stable_object_id": f"sha256:{record['normalized_sha256']}",
                    "modality": "en",
                    "evidence_status": "DEDICATED_TERMINAL",
                    "authority_ref": "terminal-source-test",
                    "transport_repo": record["transport_repo"],
                    "transport_commit": record["transport_commit"],
                    "transport_path": record["transport_path"],
                    "transport_git_blob_sha1": record["transport_git_blob_sha1"],
                    "raw_bytes": record["raw_bytes"],
                    "raw_sha256": record["raw_sha256"],
                    "payload_file": f"payload/{record['source_id']}.txt",
                    "payload_utf8_bytes": record["normalized_utf8_bytes"],
                    "payload_sha256": record["normalized_sha256"],
                }
            )
        receipt = module._build_receipt(seal, rows)
        boundary = receipt["claim_boundary"]
        self.assertEqual(boundary["canonical_corpus_capacity_credited_bytes"], 0)
        self.assertEqual(boundary["training_authorized_bytes"], 0)
        self.assertEqual(boundary["authorized_unique_loss_positions"], 0)
        self.assertEqual(boundary["authorized_optimized_target_exposure"], 0)
        self.assertFalse(boundary["tokenizer_fit_authorized"])
        self.assertFalse(boundary["training_executed"])
        serialized = json.dumps(receipt)
        self.assertNotIn("Ludvig Holberg, The Founder", serialized)
        self.assertNotIn("A Literary History of the Arabs", serialized)


if __name__ == "__main__":
    unittest.main()
