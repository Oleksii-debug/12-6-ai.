from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "materialize_d03_ecfr_point_in_time.py"
CONFIG_PATH = ROOT / "configs" / "data" / "d03_ecfr_point_in_time_request_v1.json"

SPEC = importlib.util.spec_from_file_location("materialize_d03_ecfr_point_in_time", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rehash(candidate: dict) -> dict:
    candidate["request_identity_sha256"] = MODULE._canonical_sha256(
        candidate, "request_identity_sha256"
    )
    return candidate


def _fetch_result(body: bytes = b"<ECFR><DIV1><P>Hello world.</P></DIV1></ECFR>"):
    request = _load()
    return MODULE.FetchResult(
        status=200,
        final_url=request["selection"]["url"],
        content_type="application/xml",
        body=body,
    )


def test_canonical_request_validates() -> None:
    request = MODULE.validate_request(_load())
    assert request["selection"]["date"] == "2026-09-03"
    assert request["selection"]["title"] == 1
    assert request["claims"]["canonical_capacity_credit_bytes"] == 0
    assert request["claims"]["authorized_unique_loss_positions"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["authority"].__setitem__("import_in_progress", True), "import"),
        (lambda c: c["authority"].__setitem__("reserved_titles", []), "reserved title"),
        (lambda c: c["selection"].__setitem__("title", 35), "reserved title"),
        (lambda c: c["selection"].__setitem__("date", "2026-09-04"), "URL date"),
        (
            lambda c: c["selection"].__setitem__(
                "url", "https://www.ecfr.gov/current/title-1"
            ),
            "historical full-title",
        ),
        (
            lambda c: c["selection"].__setitem__(
                "url", "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-1.xml?part=1"
            ),
            "without query",
        ),
        (lambda c: c["bounds"].__setitem__("acquisitions_required", 1), "two acquisitions"),
        (lambda c: c["bounds"].__setitem__("allow_redirects", True), "redirects"),
        (
            lambda c: c["claims"].__setitem__("canonical_capacity_credit_bytes", 1),
            "cannot grant",
        ),
        (lambda c: c["claims"].__setitem__("training_authorized_bytes", 1), "cannot grant"),
    ],
)
def test_request_mutations_fail_closed(mutation, message: str) -> None:
    candidate = copy.deepcopy(_load())
    mutation(candidate)
    _rehash(candidate)
    with pytest.raises(MODULE.MaterializationError, match=message):
        MODULE.validate_request(candidate)


def test_date_beyond_frozen_metadata_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["selection"]["date"] = "2026-09-04"
    candidate["selection"]["url"] = (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-09-04/title-1.xml"
    )
    _rehash(candidate)
    with pytest.raises(MODULE.MaterializationError, match="exceeds frozen metadata"):
        MODULE.validate_request(candidate)


def test_identity_tamper_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["selection"]["title"] = 2
    with pytest.raises(MODULE.MaterializationError, match="identity mismatch"):
        MODULE.validate_request(candidate)


def test_identical_acquisitions_emit_zero_credit_evidence() -> None:
    result = _fetch_result()
    calls = 0

    def fetcher(url: str, max_bytes: int, timeout_seconds: int):
        nonlocal calls
        calls += 1
        assert url == _load()["selection"]["url"]
        assert max_bytes == _load()["bounds"]["max_response_bytes"]
        assert timeout_seconds == _load()["bounds"]["timeout_seconds"]
        return result

    raw, evidence = MODULE.materialize(_load(), fetcher=fetcher)
    assert calls == 2
    assert raw == result.body
    assert evidence["byte_identical"] is True
    assert evidence["raw_sha256"] == evidence["acquisitions"][0]["raw_sha256"]
    assert evidence["acquisitions"][0]["raw_sha256"] == evidence["acquisitions"][1]["raw_sha256"]
    assert evidence["xml"]["element_count"] == 3
    assert evidence["xml"]["normalized_text_fragment_count"] == 1
    assert evidence["rights_and_provenance_status"] == "NOT_RUN"
    assert evidence["global_dedup_status"] == "NOT_RUN"
    assert evidence["claims"] == MODULE.ZERO_CLAIMS
    assert len(evidence["evidence_identity_sha256"]) == 64


def test_repeat_byte_drift_fails_closed() -> None:
    results = [_fetch_result(b"<ECFR><P>A</P></ECFR>"), _fetch_result(b"<ECFR><P>B</P></ECFR>")]

    def fetcher(url: str, max_bytes: int, timeout_seconds: int):
        return results.pop(0)

    with pytest.raises(MODULE.MaterializationError, match="not byte-identical"):
        MODULE.materialize(_load(), fetcher=fetcher)


def test_redirect_or_final_url_drift_fails_closed() -> None:
    result = _fetch_result()
    drifted = MODULE.FetchResult(
        status=result.status,
        final_url="https://example.invalid/title-1.xml",
        content_type=result.content_type,
        body=result.body,
    )
    with pytest.raises(MODULE.MaterializationError, match="redirected"):
        MODULE.materialize(_load(), fetcher=lambda *_: drifted)


def test_wrong_content_type_fails_closed() -> None:
    result = _fetch_result()
    wrong = MODULE.FetchResult(
        status=result.status,
        final_url=result.final_url,
        content_type="text/html",
        body=result.body,
    )
    with pytest.raises(MODULE.MaterializationError, match="content type"):
        MODULE.materialize(_load(), fetcher=lambda *_: wrong)


def test_malformed_xml_fails_closed() -> None:
    malformed = _fetch_result(b"<ECFR><P>broken</ECFR>")
    with pytest.raises(MODULE.MaterializationError, match="malformed XML"):
        MODULE.materialize(_load(), fetcher=lambda *_: malformed)


def test_bounded_reader_rejects_oversize_body() -> None:
    with pytest.raises(MODULE.MaterializationError, match="exceeds max_response_bytes"):
        MODULE._read_bounded(io.BytesIO(b"12345"), 4)


def test_non_200_status_fails_closed() -> None:
    result = _fetch_result()
    failed = MODULE.FetchResult(
        status=503,
        final_url=result.final_url,
        content_type=result.content_type,
        body=result.body,
    )
    with pytest.raises(MODULE.MaterializationError, match="HTTP 503"):
        MODULE.materialize(_load(), fetcher=lambda *_: failed)
