from __future__ import annotations

import copy
import importlib.util
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
        candidate,
        "request_identity_sha256",
    )
    return candidate


def _result(body: bytes) -> object:
    request = _load()
    return MODULE.FetchResult(
        status=200,
        final_url=request["selection"]["url"],
        content_type="application/xml",
        body=body,
    )


def test_checked_in_request_is_bound_to_exact_pinned_identity() -> None:
    request = _load()
    assert request["request_identity_sha256"] == MODULE.EXPECTED_REQUEST_IDENTITY_SHA256
    assert MODULE.validate_request(request) is request


def test_rehashed_semantically_valid_title_substitution_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["selection"]["title"] = 2
    candidate["selection"]["title_name"] = "Grants and Agreements"
    candidate["selection"]["url"] = (
        "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-2.xml"
    )
    _rehash(candidate)

    with pytest.raises(MODULE.MaterializationError, match="pinned v1 authority"):
        MODULE.validate_request(candidate)


def test_rehashed_base_authority_substitution_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["base_authority"]["live_main_observed_sha"] = "0" * 40
    _rehash(candidate)

    with pytest.raises(MODULE.MaterializationError, match="pinned v1 authority"):
        MODULE.validate_request(candidate)


def test_rehashed_downstream_gate_removal_is_rejected() -> None:
    candidate = copy.deepcopy(_load())
    candidate["next_required_gates"] = candidate["next_required_gates"][:-1]
    _rehash(candidate)

    with pytest.raises(MODULE.MaterializationError, match="pinned v1 authority"):
        MODULE.validate_request(candidate)


@pytest.mark.parametrize(
    "body",
    [
        b'<!DOCTYPE ECFR [<!ENTITY x "boom">]><ECFR><P>&x;</P></ECFR>',
        b'<!doctype ECFR><ECFR><P>text</P></ECFR>',
        b'<!ENTITY x "boom"><ECFR><P>text</P></ECFR>',
    ],
)
def test_dtd_or_entity_declarations_are_rejected_before_parse(body: bytes) -> None:
    result = _result(body)
    with pytest.raises(MODULE.MaterializationError, match="DTD/entity declarations"):
        MODULE.materialize(_load(), fetcher=lambda *_: result)


def test_normal_xml_declaration_remains_allowed() -> None:
    body = b'<?xml version="1.0" encoding="UTF-8"?><ECFR><P>Hello.</P></ECFR>'
    result = _result(body)

    raw, evidence = MODULE.materialize(_load(), fetcher=lambda *_: result)

    assert raw == body
    assert evidence["status"] == "TRANSPORT_AND_XML_PARSE_PASS_ZERO_CREDIT"
    assert evidence["claims"] == MODULE.ZERO_CLAIMS
