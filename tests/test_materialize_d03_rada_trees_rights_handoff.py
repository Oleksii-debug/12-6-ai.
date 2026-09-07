from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import materialize_d03_rada_trees_rights_handoff as handoff


def test_exact_rights_authority_replays_and_selects_expected_survivors() -> None:
    report, accepted = handoff.verify_authority()
    assert report["report_sha256"] == handoff.EXPECTED_RIGHTS_REPORT_SHA256
    assert len(accepted) == handoff.EXPECTED_ACCEPTED_MEMBERS
    assert sum(int(row["size_bytes"]) for row in accepted) == handoff.EXPECTED_ACCEPTED_SOURCE_BYTES
    assert all(row["classification"] == "PLAIN_TEXT_CANDIDATE" for row in accepted)
    assert all(row["decoded_encoding"] in handoff.ALLOWED_ENCODINGS for row in accepted)
    assert all(
        row["path"] != "texts/stenogramy-zasidannya-rnbo-vid-28-lyutogo-2014-roku.txt"
        for row in accepted
    )
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["decision"]["attribution_required"] is True


def test_decode_normalized_is_strict_and_utf8_canonical() -> None:
    source = "Український текст\r\n".encode("windows-1251")
    text, normalized = handoff.decode_normalized(source, "windows-1251")
    assert text == "Український текст\r\n"
    assert normalized == text.encode("utf-8")


def test_build_record_is_zero_credit_and_deterministic() -> None:
    text = "Пленарне засідання\n"
    normalized = text.encode("utf-8")
    payload = normalized
    digest = hashlib.sha256(payload).hexdigest()
    row = {
        "path": "texts/2024-01-16__example.txt",
        "size_bytes": len(payload),
        "sha256": digest,
        "classification": "PLAIN_TEXT_CANDIDATE",
        "decoded_encoding": "utf-8-sig",
        "text_emitted": False,
    }
    record1 = handoff.build_record(row, payload, text, normalized)
    record2 = handoff.build_record(row, payload, text, normalized)
    assert record1 == record2
    assert record1["record_id"] == hashlib.sha256(
        (handoff.SOURCE_ID + "\0" + row["path"] + "\0" + digest).encode("utf-8")
    ).hexdigest()
    assert record1["session_date"] == "2024-01-16"
    assert record1["normalized_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert record1["license"] == "CC-BY-4.0"
    assert record1["attribution_required"] is True
    assert record1["current_corpus_eligible"] is False
    assert record1["training_eligible"] is False
    assert record1["evaluation_eligible"] is False
