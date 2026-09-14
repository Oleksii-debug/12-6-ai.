from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import ubuntu_irc_v9_intake as intake

ROOT = Path(__file__).parents[1]
CROSSBIND = ROOT / "configs/data/d03_ubuntu_irc_repaired_execution_rights_crossbind_v1.json"
RIGHTS = ROOT / "configs/data/d03_common_pile_ubuntu_irc_source_rights_v1.json"
PARENT = ROOT / "configs/data/common_pile_source_rights_v1.json"
EVIDENCE = ROOT / "evidence/d03_common_pile_ubuntu_irc_real_execution_v1.json"
V9 = ROOT / "src/twelve_six/data/expanded_global_dedup_v9.py"


def canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for row in rows
    )


def row(text: str = "This is a bounded public Ubuntu IRC candidate payload." * 8) -> dict:
    raw = text.encode()
    return {
        "record_id": "2024-01-01-#ubuntu",
        "source_key": "ubuntu_irc",
        "source_label": "ubuntu-chat",
        "normalized_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_bytes": len(raw),
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def prepare(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> bytes:
    candidate = canonical_jsonl(rows)
    monkeypatch.setattr(intake, "CANDIDATE_SHA256", hashlib.sha256(candidate).hexdigest())
    monkeypatch.setattr(intake, "CANDIDATE_RECORDS", len(rows))
    monkeypatch.setattr(
        intake,
        "CANDIDATE_NORMALIZED_BYTES",
        sum(item["normalized_bytes"] for item in rows if type(item.get("normalized_bytes")) is int),
    )
    return candidate


def test_real_repository_authority_blobs_are_pinned() -> None:
    assert intake.git_blob_sha1(CROSSBIND.read_bytes()) == intake.CROSSBIND_BLOB_SHA1
    assert intake.git_blob_sha1(EVIDENCE.read_bytes()) == intake.EVIDENCE_BLOB_SHA1
    assert intake.git_blob_sha1(V9.read_bytes()) == intake.INCUMBENT_V9_FACADE_BLOB_SHA1


def test_crossbind_revalidates_real_merged_authorities() -> None:
    crossbind = intake._strict_object(CROSSBIND.read_bytes(), "crossbind")
    rights = intake._strict_object(RIGHTS.read_bytes(), "rights")
    parent = intake._strict_object(PARENT.read_bytes(), "parent")
    assert (
        intake.validate_crossbind(crossbind, rights, parent, RIGHTS.read_bytes())
        == "REPAIRED_EXECUTION_RIGHTS_CROSSBOUND_ZERO_CREDIT"
    )
    evidence = intake._strict_object(EVIDENCE.read_bytes(), "evidence")
    intake._validate_execution_evidence(evidence, EVIDENCE.read_bytes(), crossbind)


def test_candidate_projection_is_zero_credit_and_incumbent_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [row()]
    candidate = prepare(monkeypatch, rows)
    parsed = intake._parse_candidate(candidate)
    inventory, payloads = intake._build_matcher_inputs(parsed)
    assert len(inventory) == 1
    source = inventory[0]
    assert set(source) == {
        "source_id",
        "source_family",
        "stable_origin_id",
        "stable_object_id",
        "modality",
        "evidence_status",
        "authority_ref",
        "declared_capacity_bytes",
        "expected_raw_bytes",
        "expected_raw_sha256",
        "acquisition_url",
        "origin_key",
    }
    assert source["source_family"] == "common-pile/ubuntu_irc"
    assert source["modality"] == "en"
    assert source["stable_object_id"] == f"sha256:{rows[0]['normalized_sha256']}"
    assert payloads[source["source_id"]] == rows[0]["text"].encode()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.__setitem__("source_key", "other"), "source_key"),
        (lambda value: value.__setitem__("source_label", "other"), "source_label"),
        (lambda value: value.__setitem__("training_eligible", True), "training flag"),
        (lambda value: value.__setitem__("evaluation_eligible", True), "evaluation flag"),
        (lambda value: value.__setitem__("normalized_bytes", True), "normalized bytes"),
        (lambda value: value.__setitem__("extra", 1), "keys drift"),
    ],
)
def test_candidate_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    match: str,
) -> None:
    value = row()
    mutator(value)
    candidate = canonical_jsonl([value])
    monkeypatch.setattr(intake, "CANDIDATE_SHA256", hashlib.sha256(candidate).hexdigest())
    monkeypatch.setattr(intake, "CANDIDATE_RECORDS", 1)
    monkeypatch.setattr(
        intake,
        "CANDIDATE_NORMALIZED_BYTES",
        value["normalized_bytes"] if type(value.get("normalized_bytes")) is int else 1,
    )
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match=match):
        intake._parse_candidate(candidate)


def test_row_hash_recomputed_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    value = row()
    value["normalized_sha256"] = "0" * 64
    candidate = prepare(monkeypatch, [value])
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="normalized SHA-256"):
        intake._parse_candidate(candidate)


def test_duplicate_record_id_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    one = row("alpha payload " * 100)
    two = row("beta payload " * 100)
    candidate = prepare(monkeypatch, [one, two])
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="duplicate candidate record id"):
        intake._parse_candidate(candidate)


def test_duplicate_json_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b'{"record_id":"x","record_id":"y"}\n'
    monkeypatch.setattr(intake, "CANDIDATE_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="duplicate candidate JSON key"):
        intake._parse_candidate(payload)


def test_incumbent_v9_substitution_fails_before_candidate_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = prepare(monkeypatch, [row()])
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="product head"):
        intake.prepare_ubuntu_v9_intake(
            incumbent_v9_product_head="0" * 40,
            incumbent_v9_facade_bytes=V9.read_bytes(),
            crossbind_bytes=CROSSBIND.read_bytes(),
            rights_authority_bytes=RIGHTS.read_bytes(),
            parent_registry_bytes=PARENT.read_bytes(),
            execution_evidence_bytes=EVIDENCE.read_bytes(),
            candidate_bytes=candidate,
        )


def test_incumbent_v9_bytes_substitution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = prepare(monkeypatch, [row()])
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="facade bytes"):
        intake.prepare_ubuntu_v9_intake(
            incumbent_v9_product_head=intake.INCUMBENT_V9_PRODUCT_HEAD,
            incumbent_v9_facade_bytes=V9.read_bytes() + b"\n",
            crossbind_bytes=CROSSBIND.read_bytes(),
            rights_authority_bytes=RIGHTS.read_bytes(),
            parent_registry_bytes=PARENT.read_bytes(),
            execution_evidence_bytes=EVIDENCE.read_bytes(),
            candidate_bytes=candidate,
        )


def test_authority_byte_substitution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = prepare(monkeypatch, [row()])
    with pytest.raises(intake.UbuntuIrcV9IntakeError, match="crossbind bytes"):
        intake.prepare_ubuntu_v9_intake(
            incumbent_v9_product_head=intake.INCUMBENT_V9_PRODUCT_HEAD,
            incumbent_v9_facade_bytes=V9.read_bytes(),
            crossbind_bytes=CROSSBIND.read_bytes() + b"\n",
            rights_authority_bytes=RIGHTS.read_bytes(),
            parent_registry_bytes=PARENT.read_bytes(),
            execution_evidence_bytes=EVIDENCE.read_bytes(),
            candidate_bytes=candidate,
        )
