from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.cross_source_capacity_audit_v6 import (
    CrossSourceV6Error,
    _git_blob_sha1,
    _normalize_pg_body,
    _validate_config,
)

CONFIG = Path("configs/data/next100_065d_cross_source_dedup_v6.json")


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_v6_config_is_fail_closed() -> None:
    _validate_config(_config())


def test_v6_rejects_training_authorization_mutation() -> None:
    config = copy.deepcopy(_config())
    config["claim_boundary"]["training_authorized"] = True
    with pytest.raises(CrossSourceV6Error, match="claim boundary weakened"):
        _validate_config(config)


def test_v6_rejects_gutenberg_capacity_mutation() -> None:
    config = copy.deepcopy(_config())
    config["gutenberg"]["records"][1]["normalized_bytes"] += 1
    with pytest.raises(CrossSourceV6Error, match="Gutenberg normalized capacity sum drift"):
        _validate_config(config)


def test_v6_rejects_numpy_authority_head_mutation() -> None:
    config = copy.deepcopy(_config())
    config["numpy"]["head_sha"] = "0" * 40
    with pytest.raises(CrossSourceV6Error, match="NumPy head drift"):
        _validate_config(config)


def test_v6_rejects_full_cpython_byte_credit_boundary_mutation() -> None:
    config = copy.deepcopy(_config())
    config["expected_vector"]["full_cpython_normalized_bytes_must_not_be_credited"] = 15540
    with pytest.raises(CrossSourceV6Error, match="CPython full-byte prohibition drift"):
        _validate_config(config)


def test_gutenberg_normalizer_reproduces_nfc_and_boundary_contract() -> None:
    raw = (
        "preface\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "\r\n"
        "Cafe\u0301\r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "license\r\n"
    ).encode("utf-8")
    assert _normalize_pg_body(raw, "utf-8") == "Caf\u00e9\n".encode("utf-8")


def test_gutenberg_normalizer_requires_exact_marker_pair() -> None:
    raw = b"no project gutenberg markers here\n"
    with pytest.raises(CrossSourceV6Error, match="expected one Gutenberg START marker"):
        _normalize_pg_body(raw, "utf-8")


def test_git_blob_identity_uses_canonical_git_framing() -> None:
    assert _git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"
