from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_foundation import (
    CorpusFoundationError,
    PolicyHookEvidence,
    RecordPolicyMetadata,
)
from twelve_six.data.privacy_filter import (
    COVERAGE_CLAIM,
    PrivacyFilterError,
    assert_no_secret_values_in_manifest,
    build_scan_manifest,
    detect_privacy_findings,
    privacy_policy_manifest,
    scan_record,
)

H = "a" * 64
POLICY_SHA256 = "8c905e3b8f81391c3f928f375bca8fe6d1b5d38b41dec5c081577e2c5ce58526"


def _pass_hook(hook_id: str) -> PolicyHookEvidence:
    return PolicyHookEvidence(
        hook_id=hook_id,
        status="PASS",
        policy_version="v1",
        tool_ref="tool@1",
        executed_at="2026-09-07T00:00:00Z",
        evidence_sha256=H,
    )


def test_retained_policy_matches_incumbent_runtime_identity() -> None:
    retained = json.loads(
        Path("configs/data/pii_secrets_policy_v1.json").read_text(encoding="utf-8")
    )
    runtime = privacy_policy_manifest()
    assert retained == runtime
    assert runtime["policy_sha256"] == POLICY_SHA256
    assert runtime["coverage_claim"] == COVERAGE_CLAIM


def test_contact_pii_redaction_composes_with_minimal_policy_hook_contract() -> None:
    result = scan_record(
        record_id="n1",
        source_id="fixture",
        source_version="v1",
        modality="natural",
        text="Contact Alice at alice@example.com or +421 905 123 456.",
    )
    assert result.action == "REDACT"
    assert result.status == "PASS"
    assert result.detector_counts == {"email": 1, "phone": 1}
    assert result.sanitized_text is not None
    assert "alice@example.com" not in result.sanitized_text
    assert "+421 905 123 456" not in result.sanitized_text

    metadata = RecordPolicyMetadata(
        quality=_pass_hook("quality"),
        language=_pass_hook("language"),
        pii=result.policy_hook_evidence(executed_at="2026-09-07T00:01:00Z"),
        copyright=_pass_hook("copyright"),
    )
    metadata.assert_passed()
    assert len(metadata.manifest()["metadata_sha256"]) == 64


def test_private_key_and_vendor_token_are_fail_closed_without_text_output() -> None:
    key_doc = scan_record(
        record_id="key",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text="-----BEGIN " + "PRIVATE KEY-----\nFIXTUREONLY\n-----END PRIVATE KEY-----",
    )
    assert key_doc.action == "EXCLUDE"
    assert key_doc.status == "REJECT"
    assert key_doc.sanitized_text is None
    assert key_doc.output_sha256 is None
    assert key_doc.detector_counts["private_key"] == 1

    token = "ghp_" + "A" * 36
    token_doc = scan_record(
        record_id="token",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text=f'auth = "{token}"',
    )
    assert token_doc.action == "EXCLUDE"
    assert token_doc.status == "REJECT"
    assert token_doc.sanitized_text is None
    assert token_doc.detector_counts == {"github_token": 1}


def test_generic_password_quarantines_but_placeholder_remains_allowed() -> None:
    suspicious = scan_record(
        record_id="secret",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text='pass' + 'word = "' + 'M0re' + 'ThanTen!"',
    )
    assert suspicious.action == "QUARANTINE"
    assert suspicious.status == "REVIEW_REQUIRED"
    assert suspicious.sanitized_text is None

    placeholder = scan_record(
        record_id="placeholder",
        source_id="fixture",
        source_version="v1",
        modality="code",
        text='password = "changeme"',
    )
    assert placeholder.action == "ALLOW"
    assert placeholder.status == "PASS"


def test_structured_identifier_validators_redact_only_valid_shapes() -> None:
    valid = scan_record(
        record_id="structured",
        source_id="fixture",
        source_version="v1",
        modality="natural",
        text="Fixtures: 4111 1111 1111 1111 and GB82 WEST 1234 5698 7654 32.",
    )
    assert valid.action == "REDACT"
    assert valid.detector_counts["payment_card_luhn"] == 1
    assert valid.detector_counts["iban"] == 1
    assert valid.sanitized_text is not None
    assert "4111 1111 1111 1111" not in valid.sanitized_text
    assert "GB82 WEST 1234 5698 7654 32" not in valid.sanitized_text

    invalid = detect_privacy_findings("Build 1234-56-78 and card-like 1234 5678 9012 3456.")
    assert not invalid


def test_manifest_is_compact_and_never_persists_detected_values() -> None:
    token = "ghp_" + "B" * 36
    rows = [
        scan_record(
            record_id="email",
            source_id="fixture",
            source_version="v1",
            modality="natural",
            text="Write to bob@example.org.",
        ),
        scan_record(
            record_id="token",
            source_id="fixture",
            source_version="v1",
            modality="code",
            text=f'token = "{token}"',
        ),
    ]
    manifest = build_scan_manifest(
        rows,
        input_content_sha256=hashlib.sha256(b"fixture").hexdigest(),
        source_registry_sha256=hashlib.sha256(b"registry").hexdigest(),
    )
    serialized = json.dumps(manifest)
    assert manifest["records_total"] == 2
    assert manifest["detector_counts"]["email"] == 1
    assert manifest["detector_counts"]["github_token"] == 1
    assert "bob@example.org" not in serialized
    assert token not in serialized
    assert COVERAGE_CLAIM in serialized
    assert_no_secret_values_in_manifest(manifest)


def test_manifest_structure_guard_rejects_value_bearing_fields() -> None:
    with pytest.raises(PrivacyFilterError, match="forbidden"):
        assert_no_secret_values_in_manifest({"records": [{"match_value": "do-not-log"}]})


def test_non_pass_privacy_hook_keeps_record_non_train_eligible() -> None:
    metadata = RecordPolicyMetadata(
        quality=_pass_hook("quality"),
        language=_pass_hook("language"),
        pii=PolicyHookEvidence(
            hook_id="pii",
            status="REVIEW_REQUIRED",
            policy_version="12-6.pii-secrets-policy.v1",
            tool_ref="privacy@1",
            executed_at="2026-09-07T00:02:00Z",
            evidence_sha256=H,
        ),
        copyright=_pass_hook("copyright"),
    )
    with pytest.raises(CorpusFoundationError, match="pii"):
        metadata.assert_passed()
