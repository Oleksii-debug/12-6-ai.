from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.cross_source_capacity_audit_v6 import (
    CrossSourceV6Error,
    GUTENBERG_TOTAL_BYTES,
    normalize_gutenberg_body,
    validate_config,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/data/next100_065d_gutenberg_global_dedup_v6.json"
)


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_v6_config_is_fail_closed_and_valid() -> None:
    config = _config()
    validate_config(config)
    authority = config["gutenberg_authority"]
    assert isinstance(authority, dict)
    records = authority["records"]
    assert isinstance(records, list)
    assert sum(int(row["normalized_utf8_bytes"]) for row in records) == GUTENBERG_TOTAL_BYTES
    assert len({row["source_id"] for row in records}) == 3


def test_gutenberg_normalizer_reproduces_body_boundary_and_nfc() -> None:
    raw = (
        "transport header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "\r\n"
        "Cafe\u0301\r\n"
        "inside body\r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "transport footer\r\n"
    ).encode("utf-8")
    assert normalize_gutenberg_body(raw, "utf-8") == "Café\ninside body\n".encode("utf-8")


def test_gutenberg_normalizer_fails_on_ambiguous_markers() -> None:
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "one\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "two\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    ).encode("utf-8")
    with pytest.raises(CrossSourceV6Error, match="START marker"):
        normalize_gutenberg_body(raw, "utf-8")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("independent_family_credit", 3, "family-credit"),
        ("evaluation", "ALLOWED", "evaluation boundary"),
        ("workflow_conclusion", "queued", "not green"),
        ("worldwide_public_domain_claim", True, "worldwide public-domain"),
    ],
)
def test_authority_drift_fails_closed(field: str, value: object, match: str) -> None:
    config = copy.deepcopy(_config())
    authority = config["gutenberg_authority"]
    assert isinstance(authority, dict)
    authority[field] = value
    with pytest.raises(CrossSourceV6Error, match=match):
        validate_config(config)


def test_record_capacity_drift_fails_closed() -> None:
    config = copy.deepcopy(_config())
    authority = config["gutenberg_authority"]
    assert isinstance(authority, dict)
    records = authority["records"]
    assert isinstance(records, list)
    records[0]["normalized_utf8_bytes"] = int(records[0]["normalized_utf8_bytes"]) + 1
    with pytest.raises(CrossSourceV6Error, match="normalized-byte total"):
        validate_config(config)


def test_training_authorization_cannot_be_smuggled_into_v6() -> None:
    config = copy.deepcopy(_config())
    boundary = config["claim_boundary"]
    assert isinstance(boundary, dict)
    boundary["training_authorized"] = True
    with pytest.raises(CrossSourceV6Error, match="claim boundary"):
        validate_config(config)
