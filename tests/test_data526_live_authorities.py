from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path
from urllib.error import HTTPError

from tools.validate_data526_live_authorities import (
    AUTHORITIES,
    DEFAULT_REPOSITORY,
    ProvenanceError,
    validate_live_authorities,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "configs/data/data526_predecontam_source_records_v1.json"


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _content_payload(authority: dict) -> dict:
    body = {authority["identity_field"]: authority["identity"]}
    encoded = base64.b64encode(json.dumps(body).encode("utf-8")).decode("ascii")
    return {
        "encoding": "base64",
        "content": encoded,
        "sha": "1" * 40,
    }


def _run_payload(authority: dict) -> dict:
    return {
        "id": authority["run_id"],
        "name": authority["workflow_name"],
        "head_sha": authority["head_sha"],
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": DEFAULT_REPOSITORY},
    }


class _FakeGitHub:
    def __init__(self) -> None:
        self.payloads: dict[str, dict] = {}
        for authority in AUTHORITIES.values():
            run_suffix = f"/actions/runs/{authority['run_id']}"
            content_suffix = f"/contents/{authority['path']}?ref={authority['head_sha']}"
            self.payloads[run_suffix] = _run_payload(authority)
            self.payloads[content_suffix] = _content_payload(authority)

    def __call__(self, request, timeout: int = 20) -> _Response:
        del timeout
        for suffix, payload in self.payloads.items():
            if request.full_url.endswith(suffix):
                return _Response(payload)
        raise HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


class Data526LiveAuthorityTests(unittest.TestCase):
    def test_exact_live_provenance_passes(self) -> None:
        report = validate_live_authorities(
            _inventory(),
            DEFAULT_REPOSITORY,
            opener=_FakeGitHub(),
        )
        self.assertEqual(report["status"], "PASS_LIVE_AUTHORITY_PROVENANCE_ONLY")
        self.assertEqual(report["authority_count"], 3)
        self.assertEqual(report["record_count"], 9)
        self.assertFalse(report["final_training_authorized"])

    def test_live_workflow_head_drift_fails_closed(self) -> None:
        fake = _FakeGitHub()
        authority = AUTHORITIES["next100_022_ua_wikisource"]
        suffix = f"/actions/runs/{authority['run_id']}"
        fake.payloads[suffix]["head_sha"] = "0" * 40
        with self.assertRaisesRegex(ProvenanceError, "live workflow head drift"):
            validate_live_authorities(_inventory(), DEFAULT_REPOSITORY, opener=fake)

    def test_live_workflow_failure_fails_closed(self) -> None:
        fake = _FakeGitHub()
        authority = AUTHORITIES["next100_034_nist_terminal"]
        suffix = f"/actions/runs/{authority['run_id']}"
        fake.payloads[suffix]["conclusion"] = "failure"
        with self.assertRaisesRegex(ProvenanceError, "live workflow is not success"):
            validate_live_authorities(_inventory(), DEFAULT_REPOSITORY, opener=fake)

    def test_live_authority_identity_drift_fails_closed(self) -> None:
        fake = _FakeGitHub()
        authority = AUTHORITIES["data287_incumbent_registry"]
        suffix = f"/contents/{authority['path']}?ref={authority['head_sha']}"
        payload = {authority["identity_field"]: "0" * 64}
        fake.payloads[suffix]["content"] = base64.b64encode(
            json.dumps(payload).encode("utf-8")
        ).decode("ascii")
        with self.assertRaisesRegex(ProvenanceError, "live authority identity drift"):
            validate_live_authorities(_inventory(), DEFAULT_REPOSITORY, opener=fake)

    def test_unverified_record_authority_fails_closed(self) -> None:
        inventory = _inventory()
        inventory["records"][0]["authority_head_sha"] = "0" * 40
        with self.assertRaisesRegex(ProvenanceError, "not bound to a verified live authority"):
            validate_live_authorities(
                inventory,
                DEFAULT_REPOSITORY,
                opener=_FakeGitHub(),
            )


if __name__ == "__main__":
    unittest.main()
