from __future__ import annotations

import copy
import json
import sys
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_rada_trees_xet_identity as probe


CONFIG = ROOT / "configs" / "data" / "d03_rada_trees_xet_identity_probe_v1.json"


class _RedirectingOpener:
    def __init__(self) -> None:
        self.request = None

    def open(self, request, timeout=30):  # type: ignore[no-untyped-def]
        self.request = request
        headers = Message()
        headers["X-Xet-Hash"] = "a" * 64
        headers["X-Repo-Commit"] = "1" * 40
        headers["X-Linked-Size"] = "536000000"
        raise urllib.error.HTTPError(request.full_url, 302, "Found", headers, None)


class _UnexpectedBodyResponse:
    headers = Message()

    def close(self) -> None:
        return None


class _DirectBodyOpener:
    def open(self, request, timeout=30):  # type: ignore[no-untyped-def]
        return _UnexpectedBodyResponse()


class RadaTreesXetIdentityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_committed_config_keeps_immutable_hugging_face_endpoints(self) -> None:
        probe._validate_config(copy.deepcopy(self.config), ROOT)

    def test_resolve_handshake_uses_documented_get_without_following_redirect(self) -> None:
        opener = _RedirectingOpener()
        with mock.patch.object(probe.urllib.request, "build_opener", return_value=opener):
            headers = probe._resolve_headers(
                "https://huggingface.co/datasets/uacorpus/Rada_Trees/resolve/"
                + "1" * 40
                + "/Rada_Trees.7z"
            )
        self.assertIsNotNone(opener.request)
        self.assertEqual(opener.request.get_method(), "GET")
        self.assertEqual(headers["x-xet-hash"], "a" * 64)

    def test_direct_body_response_is_rejected_fail_closed(self) -> None:
        with mock.patch.object(
            probe.urllib.request, "build_opener", return_value=_DirectBodyOpener()
        ):
            with self.assertRaisesRegex(probe.ProbeError, "possible archive body"):
                probe._resolve_headers("https://huggingface.co/example")

    def test_tree_api_origin_cannot_be_rebound(self) -> None:
        value = copy.deepcopy(self.config)
        value["upstream"]["tree_api_template"] = (
            "https://example.invalid/api/datasets/{repo_id}/tree/{revision}"
        )
        with self.assertRaisesRegex(probe.ProbeError, "tree API template drift"):
            probe._validate_config(value, ROOT)

    def test_resolve_origin_cannot_be_rebound(self) -> None:
        value = copy.deepcopy(self.config)
        value["upstream"]["resolve_template"] = (
            "https://example.invalid/datasets/{repo_id}/resolve/{revision}/{path}"
        )
        with self.assertRaisesRegex(probe.ProbeError, "resolve template drift"):
            probe._validate_config(value, ROOT)

    def test_identity_requirements_cannot_be_weakened(self) -> None:
        value = copy.deepcopy(self.config)
        value["identity_requirements"]["resolve_header_xet_crosscheck_required"] = False
        with self.assertRaisesRegex(probe.ProbeError, "identity requirement weakened"):
            probe._validate_config(value, ROOT)

    def test_report_requires_get_no_redirect_evidence(self) -> None:
        core = {
            "schema_version": probe.REPORT_SCHEMA,
            "worker_id": "test",
            "execution_profile": "LOCAL_FREE",
            "dataset": "uacorpus/Rada_Trees",
            "exact_revision": "1" * 40,
            "files": [
                {
                    "git_blob_oid": "2" * 40,
                    "xet_hash": "a" * 64,
                    "resolve_xet_hash": "a" * 64,
                    "resolve_http_contract": "HEAD",
                    "archive_body_downloaded": False,
                    "training_capacity_credit_bytes": 0,
                },
                {
                    "git_blob_oid": "3" * 40,
                    "xet_hash": "b" * 64,
                    "resolve_xet_hash": "b" * 64,
                    "resolve_http_contract": "GET_NO_REDIRECT",
                    "archive_body_downloaded": False,
                    "training_capacity_credit_bytes": 0,
                },
            ],
            "archive_count": 2,
            "total_archive_bytes": 1,
            "claim_boundary": {
                "archive_bodies_downloaded": False,
                "training_authorized_bytes": 0,
            },
        }
        report = {**core, "report_sha256": probe._sha256(probe._canonical_bytes(core))}
        with self.assertRaisesRegex(probe.ProbeError, "HTTP contract drift"):
            probe.verify_report(report)


if __name__ == "__main__":
    unittest.main()
