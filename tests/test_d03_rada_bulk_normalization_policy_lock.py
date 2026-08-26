from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.normalize_d03_rada_bulk_html import NormalizationError, normalize_html_bytes
from tools.validate_d03_rada_bulk_normalization_policy_lock import (
    PolicyLockError,
    validate_policy_lock,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_rada_bulk_normalization_v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_config_matches_executable_strict_utf8_boundary() -> None:
    assert CONFIG["normalization"]["decode"] == "STRICT_UTF8_OPTIONAL_BOM"
    assert "source_encoding" not in CONFIG["output_contract"]["jsonl_record_fields"]

    cp1251 = "<p>Закон України</p>".encode("windows-1251")
    with pytest.raises(NormalizationError, match="not strict UTF-8"):
        normalize_html_bytes(cp1251, CONFIG)


def test_canonical_config_has_stable_semantic_identity() -> None:
    first = validate_policy_lock(CONFIG)
    second = validate_policy_lock(copy.deepcopy(CONFIG))
    assert first == second
    assert len(first) == 64
    int(first, 16)


def test_removing_script_from_hidden_tags_fails_closed() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["normalization"]["hidden_tags"].remove("script")
    with pytest.raises(PolicyLockError, match="hidden tags drift"):
        validate_policy_lock(mutated)


def test_changing_block_tag_semantics_fails_closed() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["normalization"]["block_tags"].remove("p")
    with pytest.raises(PolicyLockError, match="block tags drift"):
        validate_policy_lock(mutated)


def test_decode_or_unicode_policy_drift_fails_closed() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["normalization"]["decode"] = "STRICT_UTF8_THEN_WINDOWS_1251_FALLBACK"
    with pytest.raises(PolicyLockError, match="decode drift"):
        validate_policy_lock(mutated)

    mutated = copy.deepcopy(CONFIG)
    mutated["normalization"]["unicode_normalization"] = "NFC"
    with pytest.raises(PolicyLockError, match="unicode normalization drift"):
        validate_policy_lock(mutated)


def test_record_identity_semantics_cannot_drift() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["normalization"]["record_id_prefix"] = "ua.rada.laws."
    with pytest.raises(PolicyLockError, match="record id prefix drift"):
        validate_policy_lock(mutated)


def test_output_contract_cannot_claim_unemitted_source_encoding() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["output_contract"]["jsonl_record_fields"].insert(2, "source_encoding")
    with pytest.raises(PolicyLockError, match="record fields drift"):
        validate_policy_lock(mutated)


def test_downstream_gate_order_cannot_be_weakened() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["downstream_required"].remove("EVALUATION_DECONTAMINATION")
    with pytest.raises(PolicyLockError, match="downstream gate order drift"):
        validate_policy_lock(mutated)


def test_claim_boundary_and_safe_result_remain_fail_closed() -> None:
    mutated = copy.deepcopy(CONFIG)
    mutated["claim_boundary"]["training_authorized_bytes"] = 1
    with pytest.raises(PolicyLockError, match="training authorization drift"):
        validate_policy_lock(mutated)

    mutated = copy.deepcopy(CONFIG)
    mutated["claim_boundary"]["safe_result"] = "PASS_TRAINING_READY"
    with pytest.raises(PolicyLockError, match="safe result drift"):
        validate_policy_lock(mutated)
