"""W2 exact pinned UA-GEC metadata contract; deliberately not a rights grant."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.plan2_s16_ua_gec_candidate_contract_v1 import (
    CandidateEvidenceDenied,
    DECISION,
    EXTERNAL_EVIDENCE,
    FALSE_FLAGS,
    SOURCE,
    inspect_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/plan2_s16_ua_gec_candidate_evidence_v1.json"


def load() -> dict:
    return json.loads(PATH.read_bytes())


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def test_original_fixture_exact_non_authorizing_readback() -> None:
    raw = PATH.read_bytes()
    receipt = inspect_bytes(raw)
    assert receipt["decision"] == DECISION
    assert receipt["raw_input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["source_upstream_commit"] == SOURCE["upstream_commit"]
    assert receipt["missing_external_evidence"] == EXTERNAL_EVIDENCE
    assert all(receipt[k] is False for k in (
        "physical_corpus_admitted", "training_authorized",
        "production_release_authorized", "plan9_binding_authorized", "terminal_done"
    ))


def test_clean_two_process_deterministic_readback() -> None:
    script = ROOT / "tools/plan2_s16_ua_gec_candidate_contract_v1.py"
    cmd = [sys.executable, str(script), "--candidate", str(PATH)]
    a = subprocess.run(cmd, capture_output=True, check=True)
    b = subprocess.run(cmd, capture_output=True, check=True)
    assert a.stdout == b.stdout
    assert json.loads(a.stdout)["terminal_done"] is False


def test_semantic_and_raw_byte_custody_are_distinct() -> None:
    candidate = load()
    first = inspect_bytes(encode(candidate))
    second = inspect_bytes(json.dumps(candidate, ensure_ascii=False, indent=2).encode())
    assert first["canonical_candidate_sha256"] == second["canonical_candidate_sha256"]
    assert first["raw_input_sha256"] != second["raw_input_sha256"]


@pytest.mark.parametrize("field", list(FALSE_FLAGS))
def test_authority_promotion_denied(field: str) -> None:
    candidate = load()
    candidate[field] = True
    with pytest.raises(CandidateEvidenceDenied, match="unsafe candidate promotion"):
        inspect_bytes(encode(candidate))


@pytest.mark.parametrize("field", list(SOURCE))
def test_source_revision_and_license_bytes_cannot_be_substituted(field: str) -> None:
    candidate = load()
    candidate["source"][field] = "0" * 40 if field.endswith("sha1") or field == "upstream_commit" else "other"
    with pytest.raises(CandidateEvidenceDenied, match="source"):
        inspect_bytes(encode(candidate))


@pytest.mark.parametrize("field,value", [
    ("family_count_credit", 2),
    ("family_count_credit", True),
    ("test_split_for_training", True),
    ("train_test_separation_required", False),
    ("metadata_sensitive_fields", ["author_id"]),
    ("language", "en"),
    ("annotated_source_target_are_independent_documents", True),
    ("gec_only_and_gec_fluency_are_independent_families", True),
])
def test_scope_drift_denied(field: str, value: object) -> None:
    candidate = load()
    candidate["scope"][field] = value
    with pytest.raises(CandidateEvidenceDenied, match="scope"):
        inspect_bytes(encode(candidate))


@pytest.mark.parametrize("mutation", ["drop", "replace", "append", "duplicate"])
def test_external_evidence_list_cannot_be_weakened(mutation: str) -> None:
    candidate = load()
    evidence = candidate["required_external_evidence"]
    if mutation == "drop":
        evidence.pop()
    elif mutation == "replace":
        evidence[0] = "UNVERIFIED_FAKE_GRANT"
    elif mutation == "append":
        evidence.append("SELF_ISSUED_RELEASE")
    else:
        evidence.append(evidence[0])
    with pytest.raises(CandidateEvidenceDenied, match="external rights/privacy evidence"):
        inspect_bytes(encode(candidate))


@pytest.mark.parametrize("mutation", ["extra", "missing", "nested_extra", "nested_missing"])
def test_authority_shape_drift_denied(mutation: str) -> None:
    candidate = load()
    if mutation == "extra":
        candidate["training_ready"] = True
    elif mutation == "missing":
        del candidate["decision"]
    elif mutation == "nested_extra":
        candidate["source"]["rights_approved"] = True
    else:
        del candidate["source"]["license_name"]
    with pytest.raises(CandidateEvidenceDenied):
        inspect_bytes(encode(candidate))


@pytest.mark.parametrize("raw", [
    b'{"decision":"BLOCKED_CANDIDATE_ONLY","decision":"AUTHORIZED"}',
    b'{"schema_version":NaN}',
    b'\xff',
    b'[]',
    b'',
    b'{}' * 40000,
])
def test_invalid_json_or_unbounded_authority_denied(raw: bytes) -> None:
    with pytest.raises(CandidateEvidenceDenied):
        inspect_bytes(raw)


def test_cli_denies_symlinked_candidate(tmp_path: Path) -> None:
    link = tmp_path / "candidate.json"
    try:
        link.symlink_to(PATH)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    script = ROOT / "tools/plan2_s16_ua_gec_candidate_contract_v1.py"
    result = subprocess.run([sys.executable, str(script), "--candidate", str(link)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["decision"] == "DENIED"
