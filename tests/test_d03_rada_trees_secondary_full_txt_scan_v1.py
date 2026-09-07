from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import scan_d03_rada_trees_secondary_full_txt_v1 as scan


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _evidence() -> dict:
    evidence = {
        "schema_version": scan.EVIDENCE_SCHEMA,
        "source": {
            "dataset": scan.DATASET,
            "dataset_revision": scan.REVISION,
            "archive_path": scan.ARCHIVE,
            "expected_size_bytes": scan.ARCHIVE_BYTES,
            "git_blob_oid": scan.GIT_BLOB_OID,
            "xet_hash": scan.XET_HASH,
            "content_sha256": scan.ARCHIVE_SHA256,
        },
        "inventory": {
            "member_count": scan.EXPECTED_MEMBER_COUNT,
            "uncompressed_bytes": scan.EXPECTED_UNCOMPRESSED_BYTES,
            "listing_identity_sha256": scan.EXPECTED_LISTING_IDENTITY,
        },
        "classification": {
            "plain_text_suffix_member_count": scan.EXPECTED_TXT_MEMBER_COUNT,
        },
        "execution": {
            "conclusion": "success",
            "head_sha": scan.PARENT_SOURCE_HEAD,
            "workflow_run": scan.PARENT_RUN,
            "workflow_job": scan.PARENT_JOB,
            "artifact_id": scan.PARENT_ARTIFACT,
            "artifact_digest": scan.PARENT_ARTIFACT_DIGEST,
        },
        "claim_boundary": {
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    evidence["evidence_identity_sha256"] = scan.sha256_obj(evidence)
    return evidence


def _validate_synthetic_evidence(evidence: dict) -> None:
    with mock.patch.object(
        scan,
        "EXPECTED_PARENT_EVIDENCE_ID",
        evidence["evidence_identity_sha256"],
    ):
        scan.validate_terminal_evidence(evidence)


def _small_listing() -> tuple[list[dict], dict[str, bytes]]:
    payloads = {
        "session/2001/a.txt": "Україна\n".encode("utf-8"),
        "session/2001/b.txt": "Україна\n".encode("utf-8"),
        "session/2002/c.txt": "Київ\n".encode("utf-8"),
    }
    listing = [
        {"path": path, "size_bytes": len(payload)}
        for path, payload in payloads.items()
    ]
    listing.extend(
        [
            {"path": "session/2001/a.xtag", "size_bytes": 2},
            {"path": "session/2002/c.xtag", "size_bytes": 3},
        ]
    )
    return listing, payloads


def _listing_patches(listing: list[dict]):
    total = sum(row["size_bytes"] for row in listing)
    txt_count = sum(row["path"].endswith(".txt") for row in listing)
    expected_identity = scan.listing_identity(listing)
    return (
        mock.patch.object(scan, "EXPECTED_MEMBER_COUNT", len(listing)),
        mock.patch.object(scan, "EXPECTED_TXT_MEMBER_COUNT", txt_count),
        mock.patch.object(scan, "EXPECTED_UNCOMPRESSED_BYTES", total),
        mock.patch.object(scan, "EXPECTED_LISTING_IDENTITY", expected_identity),
    )


class FullTextScanTests(unittest.TestCase):
    def test_terminal_evidence_exact_binding(self) -> None:
        _validate_synthetic_evidence(_evidence())
        mutated = _evidence()
        mutated["source"]["content_sha256"] = "0" * 64
        mutated.pop("evidence_identity_sha256")
        mutated["evidence_identity_sha256"] = scan.sha256_obj(mutated)
        with self.assertRaisesRegex(scan.FullTextScanError, "content_sha256"):
            _validate_synthetic_evidence(mutated)

    def test_terminal_evidence_self_hash_tamper_rejected(self) -> None:
        evidence = _evidence()
        evidence["runtime_extra"] = "forged"
        with mock.patch.object(
            scan,
            "EXPECTED_PARENT_EVIDENCE_ID",
            evidence["evidence_identity_sha256"],
        ):
            with self.assertRaisesRegex(scan.FullTextScanError, "self-hash"):
                scan.validate_terminal_evidence(evidence)

    def test_terminal_evidence_rejects_training_credit(self) -> None:
        mutated = _evidence()
        mutated["claim_boundary"]["training_authorized_bytes"] = 1
        mutated.pop("evidence_identity_sha256")
        mutated["evidence_identity_sha256"] = scan.sha256_obj(mutated)
        with self.assertRaisesRegex(scan.FullTextScanError, "training credit"):
            _validate_synthetic_evidence(mutated)

    def test_validate_listing_rejects_count_and_path_drift(self) -> None:
        listing, _ = _small_listing()
        patches = _listing_patches(listing)
        with patches[0], patches[1], patches[2], patches[3]:
            rows = scan.validate_listing(listing)
            self.assertEqual(
                [row["path"] for row in rows],
                sorted(
                    path
                    for path in (item["path"] for item in listing)
                    if path.endswith(".txt")
                ),
            )
            bad = copy.deepcopy(listing)
            bad[0]["path"] = "../escape.txt"
            with self.assertRaisesRegex(scan.FullTextScanError, "noncanonical"):
                scan.validate_listing(bad)

    def test_listing_identity_substitution_rejected(self) -> None:
        listing, _ = _small_listing()
        patches = _listing_patches(listing)
        altered = copy.deepcopy(listing)
        altered[0]["path"] = "session/2001/renamed.txt"
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(scan.FullTextScanError, "listing identity"):
                scan.validate_listing(altered)

    def test_scans_all_txt_and_exact_duplicate_collapses(self) -> None:
        listing, payloads = _small_listing()

        def stream(item: dict) -> bytes:
            return payloads[item["path"]]

        policy = {"unused": True}
        patches = _listing_patches(listing)
        with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
            scan.classify,
            "classify_content",
            side_effect=lambda path, data, policy: {
                "class": "PLAIN_TEXT_CANDIDATE",
                "encoding": "utf-8-sig",
                "metrics": {"characters": len(data.decode("utf-8"))},
            },
        ), mock.patch.object(
            scan.classify,
            "year_hints",
            side_effect=lambda path: [2001] if "2001" in path else [2002],
        ):
            result = scan.scan_plain_text_members(
                listing,
                stream_member=stream,
                classifier_policy=policy,
            )
        self.assertEqual(result["txt_member_count_scanned"], 3)
        self.assertEqual(result["plain_text_candidate_member_count"], 3)
        self.assertEqual(result["plain_text_candidate_unique_content_count"], 2)
        self.assertEqual(result["exact_duplicate_member_count"], 1)
        self.assertEqual(len(result["exact_duplicate_groups"]), 1)
        self.assertGreater(result["exact_duplicate_discount_bytes"], 0)

    def test_non_candidate_is_not_counted_as_capacity(self) -> None:
        listing, payloads = _small_listing()

        def classify_content(path: str, data: bytes, policy: dict) -> dict:
            del data, policy
            label = (
                "PLAIN_TEXT_CANDIDATE"
                if not path.endswith("c.txt")
                else "DERIVED_UD_HOLD"
            )
            return {"class": label, "encoding": "utf-8-sig", "metrics": {}}

        patches = _listing_patches(listing)
        with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
            scan.classify,
            "classify_content",
            side_effect=classify_content,
        ), mock.patch.object(scan.classify, "year_hints", return_value=[]):
            result = scan.scan_plain_text_members(
                listing,
                stream_member=lambda item: payloads[item["path"]],
                classifier_policy={},
            )
        self.assertEqual(result["classification_counts"]["DERIVED_UD_HOLD"], 1)
        self.assertEqual(result["plain_text_candidate_member_count"], 2)
        self.assertEqual(result["plain_text_candidate_unique_content_count"], 1)

    def test_streamed_member_size_drift_fails_closed(self) -> None:
        listing, _ = _small_listing()
        patches = _listing_patches(listing)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(scan.FullTextScanError, "streamed size drift"):
                scan.scan_plain_text_members(
                    listing,
                    stream_member=lambda item: b"x",
                    classifier_policy={},
                )

    def test_report_stays_zero_credit_and_text_free(self) -> None:
        listing, _ = _small_listing()
        total = sum(row["size_bytes"] for row in listing)
        fake_scan = {
            "txt_member_count_scanned": 3,
            "plain_text_candidate_member_count": 2,
        }
        patches = _listing_patches(listing)
        with patches[0], patches[1], patches[2], patches[3]:
            evidence = _evidence()
            evidence["inventory"]["member_count"] = len(listing)
            evidence["inventory"]["uncompressed_bytes"] = total
            evidence["inventory"]["listing_identity_sha256"] = scan.EXPECTED_LISTING_IDENTITY
            evidence["classification"]["plain_text_suffix_member_count"] = 3
            evidence.pop("evidence_identity_sha256")
            evidence["evidence_identity_sha256"] = scan.sha256_obj(evidence)
            with mock.patch.object(
                scan,
                "EXPECTED_PARENT_EVIDENCE_ID",
                evidence["evidence_identity_sha256"],
            ):
                report = scan.build_report(
                    evidence,
                    archive_sha256=scan.ARCHIVE_SHA256,
                    listing=listing,
                    scan=fake_scan,
                )
        self.assertEqual(report["claim_boundary"]["training_authorized_bytes"], 0)
        self.assertFalse(report["claim_boundary"]["tokenizer_fit_authorized"])
        self.assertFalse(report["raw_member_text_emitted"])
        rendered = str(report)
        self.assertNotIn("Україна", rendered)

    def test_report_self_hash_tamper_rejected(self) -> None:
        report = {
            "schema_version": scan.SCHEMA,
            "execution_profile": "LOCAL_FREE",
            "source": {
                "archive_sha256": scan.ARCHIVE_SHA256,
                "plain_text_suffix_member_count": scan.EXPECTED_TXT_MEMBER_COUNT,
            },
            "scan": {"txt_member_count_scanned": scan.EXPECTED_TXT_MEMBER_COUNT},
            "claim_boundary": {
                "training_authorized_bytes": 0,
                "unique_causal_loss_positions_authorized": 0,
                "tokenizer_fit_authorized": False,
                "model_training_executed": False,
                "final_test_payload_accessed": False,
                "paid_compute_used": False,
                "research_corpus_v1_released": False,
            },
            "raw_member_text_emitted": False,
        }
        report["report_identity_sha256"] = scan.sha256_obj(report)
        scan.verify_report(report)
        report["claim_boundary"]["training_authorized_bytes"] = 1
        with self.assertRaisesRegex(scan.FullTextScanError, "self-hash"):
            scan.verify_report(report)

    def test_report_rehashed_training_overclaim_rejected(self) -> None:
        report = {
            "schema_version": scan.SCHEMA,
            "execution_profile": "LOCAL_FREE",
            "source": {
                "archive_sha256": scan.ARCHIVE_SHA256,
                "plain_text_suffix_member_count": scan.EXPECTED_TXT_MEMBER_COUNT,
            },
            "scan": {"txt_member_count_scanned": scan.EXPECTED_TXT_MEMBER_COUNT},
            "claim_boundary": {
                "training_authorized_bytes": 1,
                "unique_causal_loss_positions_authorized": 0,
                "tokenizer_fit_authorized": False,
                "model_training_executed": False,
                "final_test_payload_accessed": False,
                "paid_compute_used": False,
                "research_corpus_v1_released": False,
            },
            "raw_member_text_emitted": False,
        }
        report["report_identity_sha256"] = scan.sha256_obj(report)
        with self.assertRaisesRegex(scan.FullTextScanError, "training credit"):
            scan.verify_report(report)

    def test_candidate_projection_identity_is_order_deterministic(self) -> None:
        rows = [
            {
                "path": "b.txt",
                "size_bytes": 1,
                "sha256": _sha(b"b"),
                "decoded_encoding": "utf-8",
                "year_hints": [],
            },
            {
                "path": "a.txt",
                "size_bytes": 1,
                "sha256": _sha(b"a"),
                "decoded_encoding": "utf-8",
                "year_hints": [],
            },
        ]
        first = scan.sha256_obj(sorted(rows, key=lambda row: row["path"]))
        second = scan.sha256_obj(sorted(reversed(rows), key=lambda row: row["path"]))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
