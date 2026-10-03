from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def synthetic_source_fixture():
    """Three synthetic records; no source download or corpus credit."""
    records = []
    raw_by_url = {}
    expected_payloads = {}
    for index in range(3):
        source_id = f"en.project-gutenberg.synthetic-{index}"
        raw = (
            b"outside\n"
            b"*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
            + f"synthetic chapter {index}\n".encode("ascii")
            + b"\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        )
        normalized = module.normalize_pg_body(raw, "ascii")
        record = {
            "source_id": source_id,
            "ebook_id": index + 1,
            "encoding": "ascii",
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "transport_git_blob_sha1": git_blob_sha1(raw),
            "normalized_utf8_bytes": len(normalized),
            "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
            "transport_repo": f"GITenberg/Synthetic_{index}",
            "transport_commit": f"{index + 1:040x}",
            "transport_path": f"{index}.txt",
        }
        records.append(record)
        raw_by_url[module._transport_url(record)] = raw
        expected_payloads[source_id.replace(".", "_") + ".txt"] = normalized
    seal = {
        "records": records,
        "parent_authority": {"head_sha": "a" * 40},
        "exact_head_execution": {
            "workflow_run_id": 1,
            "artifact_id": 2,
            "artifact_digest": "sha256:" + "0" * 64,
        },
    }
    return seal, records, raw_by_url, expected_payloads


def run_synthetic(seal, records, out_root, fetcher):
    expected_bytes = sum(record["normalized_utf8_bytes"] for record in records)
    expected_records = {record["source_id"]: record for record in records}
    with (
        mock.patch.object(module, "load_terminal_seal", return_value=seal),
        mock.patch.object(module, "EXPECTED_RECORDS", expected_records),
        mock.patch.object(module, "EXPECTED_NORMALIZED_TOTAL", expected_bytes),
    ):
        return module.materialize(
            Path("synthetic-seal"), out_root, fetcher=fetcher
        )


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
        self.assertNotIn(
            "external_llm_or_api_used_for_data_or_intelligence",
            boundary,
        )
        self.assertFalse(
            boundary["upstream_source_evidence_external_llm_or_api_used"]
        )
        self.assertFalse(
            boundary["current_corpus_external_llm_free_claimed_by_this_rematerializer"]
        )
        serialized = json.dumps(receipt)
        self.assertNotIn("Ludvig Holberg, The Founder", serialized)
        self.assertNotIn("A Literary History of the Arabs", serialized)


    def test_transport_failure_leaves_no_output_and_same_out_retry_succeeds(self):
        for fail_index in (0, 1):
            with self.subTest(fail_index=fail_index):
                seal, records, raw_by_url, expected = synthetic_source_fixture()
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / "output"
                    count = 0

                    def failing_fetch(url, *, fail_index=fail_index, raw_by_url=raw_by_url):
                        nonlocal count
                        count += 1
                        if count == fail_index + 1:
                            raise module.MaterializationError(
                                "injected transport error"
                            )
                        return raw_by_url[url]

                    with self.assertRaisesRegex(
                        module.MaterializationError, "injected transport error"
                    ):
                        run_synthetic(seal, records, out, failing_fetch)
                    self.assertFalse(out.exists())
                    self.assertFalse(
                        out.with_name(".output.gutenberg-stage-v1").exists()
                    )
                    receipt = run_synthetic(
                        seal, records, out, raw_by_url.__getitem__
                    )
                    self.assertEqual(receipt["payload_inventory"]["record_count"], 3)
                    for name, data in expected.items():
                        self.assertEqual((out / "payload" / name).read_bytes(), data)
                    self.assertEqual(
                        (out / "receipt.json").read_bytes(),
                        module._canonical(receipt),
                    )

    def test_transport_bound_rejects_oversize_before_staging(self):
        # A hostile server must never force an unlimited response.read().
        cap = module.MAX_TRANSPORT_BYTES
        self.assertEqual(cap, 1_203_657)
        requests = []

        class BoundedResponse:
            def __init__(self, data):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, size=-1):
                requests.append(size)
                if size < 0:
                    raise AssertionError("unbounded transport read")
                return self.data[:size]

        with mock.patch.object(
            module.urllib.request, "urlopen",
            return_value=BoundedResponse(b"valid-short"),
        ):
            self.assertEqual(
                module.fetch_bytes("https://example.invalid/short", attempts=1),
                b"valid-short",
            )
        self.assertEqual(requests, [cap + 1])

        seal, records, _, _ = synthetic_source_fixture()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"
            with (
                mock.patch.object(
                    module.urllib.request, "urlopen",
                    return_value=BoundedResponse(b"x" * (cap + 2)),
                ),
                self.assertRaisesRegex(
                    module.MaterializationError,
                    "transport response exceeds pinned maximum",
                ),
            ):
                run_synthetic(seal, records, out, module.fetch_bytes)
            self.assertEqual(requests, [cap + 1, cap + 1])
            self.assertFalse(out.exists())
            self.assertFalse(
                out.with_name(".output.gutenberg-stage-v1").exists()
            )

    def test_second_record_bad_checksum_does_not_publish_partial_output(self):
        seal, records, raw_by_url, _ = synthetic_source_fixture()
        bad_url = module._transport_url(records[1])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"

            def corrupt_second(url):
                data = raw_by_url[url]
                return data + b"x" if url == bad_url else data

            with self.assertRaises(module.MaterializationError):
                run_synthetic(seal, records, out, corrupt_second)
            self.assertFalse(out.exists())
            self.assertFalse(
                out.with_name(".output.gutenberg-stage-v1").exists()
            )

    def test_receipt_write_error_retains_single_stage_and_refuses_retry(self):
        seal, records, raw_by_url, _ = synthetic_source_fixture()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"
            stage = out.with_name(".output.gutenberg-stage-v1")
            real_open = Path.open

            def fail_receipt_open(path, mode="r", *args, **kwargs):
                if path.name == "receipt.json" and mode == "xb":
                    raise OSError("injected receipt write failure")
                return real_open(path, mode, *args, **kwargs)

            with (
                mock.patch.object(Path, "open", fail_receipt_open),
                self.assertRaisesRegex(
                    module.MaterializationError, "injected receipt write failure"
                ),
            ):
                run_synthetic(seal, records, out, raw_by_url.__getitem__)
            self.assertFalse(out.exists())
            self.assertTrue((stage / "payload").is_dir())
            calls = []

            def count_fetch(url):
                calls.append(url)
                return raw_by_url[url]

            with self.assertRaisesRegex(
                module.MaterializationError, "inspect/remove the retained"
            ):
                run_synthetic(seal, records, out, count_fetch)
            self.assertEqual(calls, [])

    def test_existing_empty_destination_at_publish_is_not_replaced(self):
        seal, records, raw_by_url, _ = synthetic_source_fixture()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"
            stage = out.with_name(".output.gutenberg-stage-v1")
            original = module._publish_directory_noreplace

            def create_destination_before_publish(candidate, destination):
                destination.mkdir()
                original(candidate, destination)

            with (
                mock.patch.object(
                    module, "_publish_directory_noreplace",
                    create_destination_before_publish,
                ),
                self.assertRaises(module.MaterializationError),
            ):
                run_synthetic(seal, records, out, raw_by_url.__getitem__)
            self.assertTrue(out.is_dir())
            self.assertEqual(list(out.iterdir()), [])
            self.assertTrue((stage / "receipt.json").is_file())
            self.assertEqual(len(list((stage / "payload").iterdir())), 3)

    def test_private_stage_tampering_rejects_publication(self):
        seal, records, raw_by_url, _ = synthetic_source_fixture()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "output"
            stage = out.with_name(".output.gutenberg-stage-v1")
            original = module._verify_private_stage

            def replace_stage_payload(candidate, payloads, receipt_bytes):
                name = payloads[0][0]
                (candidate / "payload" / name).write_bytes(b"tampered")
                original(candidate, payloads, receipt_bytes)

            with (
                mock.patch.object(
                    module, "_verify_private_stage", replace_stage_payload
                ),
                self.assertRaisesRegex(
                    module.MaterializationError, "staged payload identity drift"
                ),
            ):
                run_synthetic(seal, records, out, raw_by_url.__getitem__)
            self.assertFalse(out.exists())
            self.assertTrue((stage / "receipt.json").exists())

    def test_preexisting_destination_or_private_stage_refuses_before_fetch(self):
        seal, records, raw_by_url, _ = synthetic_source_fixture()
        for occupant in ("destination", "stage"):
            with self.subTest(occupant=occupant), tempfile.TemporaryDirectory() as td:
                    out = Path(td) / "output"
                    stage = out.with_name(".output.gutenberg-stage-v1")
                    occupied = out if occupant == "destination" else stage
                    occupied.mkdir()
                    (occupied / "user-marker").write_text("untouched")
                    calls = []

                    def count_fetch(url, *, calls=calls):
                        calls.append(url)
                        return raw_by_url[url]

                    with self.assertRaises(module.MaterializationError):
                        run_synthetic(seal, records, out, count_fetch)
                    self.assertEqual(calls, [])
                    self.assertEqual(
                        (occupied / "user-marker").read_text(), "untouched"
                    )


if __name__ == "__main__":
    unittest.main()
