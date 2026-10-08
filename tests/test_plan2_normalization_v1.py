"""Plan-2 S4 normalization and provenance regression tests (LOCAL_FREE)."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from tools.plan2_normalization_v1 import (
    Plan2NormalizationError,
    normalize_verified_candidate,
    stage_normalized_candidate,
    verify_normalized_candidate,
)
from tools.plan2_physical_materialization_v1 import _canonical

ROOT = Path(__file__).resolve().parents[1]


def _predecessor(data: bytes) -> dict:
    sha = hashlib.sha256(data).hexdigest()
    core = {
        "schema_version": "12-6.plan2-physical-cohort.v1",
        "source_id": "fixture.uk.source",
        "source_member_count": 1,
        "normalized": {"sha256": sha, "bytes": len(data)},
        "source_level_candidate_only": True,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    core["manifest_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def test_utf8_unicode_newlines_boundaries_and_identity_are_deterministic():
    raw = "\ufeffКиїв і мова\r\n\r\nУкраїнські люди\r\n".encode()
    first, manifest = normalize_verified_candidate(raw, _predecessor(raw))
    second, reread = normalize_verified_candidate(raw, _predecessor(raw))
    assert first == second and reread == manifest
    assert first.decode() == "Київ і мова\n\nУкраїнські люди"
    assert manifest["record_count"] == 2
    assert [r["language"] for r in manifest["records"]] == ["uk", "uk"]
    assert all(r["modality"] == "text" for r in manifest["records"])
    assert not manifest["training_corpus_authorized"]
    assert not manifest["tokenizer_fit_authorized"]


def test_canonical_equivalent_unicode_is_equal_only_under_same_policy():
    a = "Cafe\u0301 world".encode("utf-8")
    b = "Caf\u00e9 world".encode("utf-8")
    norm_a, ma = normalize_verified_candidate(a, _predecessor(a))
    norm_b, mb = normalize_verified_candidate(b, _predecessor(b))
    assert norm_a == norm_b
    assert ma["predecessor_manifest_sha256"] != mb["predecessor_manifest_sha256"]
    assert ma["normalized_sha256"] == mb["normalized_sha256"]
    assert ma["manifest_sha256"] != mb["manifest_sha256"]


@pytest.mark.parametrize("data", [
    b"\xff", b"\x00hello world", b"", b"\xed\xa0\x80",
])
def test_malformed_encoding_empty_or_nul_never_admitted(data):
    with pytest.raises(Plan2NormalizationError):
        normalize_verified_candidate(data, _predecessor(data))


def test_unrecognized_mixed_and_short_languages_never_promoted():
    data = "xy\n\nПривет мир\n\nHola світе hello".encode()
    _, manifest = normalize_verified_candidate(data, _predecessor(data))
    assert manifest["record_count"] == 3
    assert all(r["language"] == "unknown" for r in manifest["records"])
    assert all(r["supported"] is False for r in manifest["records"])


@pytest.mark.parametrize("field,value", [
    ("training_corpus_authorized", True),
    ("tokenizer_fit_authorized", True),
    ("evaluation_authorized", True),
    ("source_member_count", 2),
])
def test_predecessor_forgery_and_promotion_fail_closed(field, value):
    payload = b"hello world"
    predecessor = _predecessor(payload)
    predecessor[field] = value
    with pytest.raises(Plan2NormalizationError):
        normalize_verified_candidate(payload, predecessor)


def test_predecessor_checksum_and_payload_drift_fail_closed():
    payload = b"hello world"
    p = _predecessor(payload)
    p["normalized"]["bytes"] += 1
    with pytest.raises(Plan2NormalizationError):
        normalize_verified_candidate(payload, p)
    p = _predecessor(payload)
    p["manifest_sha256"] = "0" * 64
    with pytest.raises(Plan2NormalizationError):
        normalize_verified_candidate(payload, p)


def test_real_incumbent_physical_cohort_integrates_and_restarts(tmp_path):
    directory = tmp_path / "normalized"
    first = stage_normalized_candidate(ROOT, directory)
    second = stage_normalized_candidate(ROOT, directory)
    payload, readback = verify_normalized_candidate(directory)
    assert first == second == readback
    assert payload
    assert first["record_count"] > 0
    assert set(p.name for p in directory.iterdir()) == {
        "normalized.utf8", "manifest.json"
    }
    assert first["source_level_candidate_only"]
    assert first["training_corpus_authorized"] is False


@pytest.mark.parametrize("member", ["normalized.utf8", "manifest.json"])
def test_immutable_postpublication_corruption_denied(tmp_path, member):
    directory = tmp_path / "normalized"
    stage_normalized_candidate(ROOT, directory)
    path = directory / member
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((Plan2NormalizationError, ValueError)):
        stage_normalized_candidate(ROOT, directory)


def test_missing_after_publication_denied_instead_of_repaired(tmp_path):
    directory = tmp_path / "normalized"
    stage_normalized_candidate(ROOT, directory)
    (directory / "normalized.utf8").unlink()
    with pytest.raises(Plan2NormalizationError, match="membership"):
        stage_normalized_candidate(ROOT, directory)


def test_symlink_and_unknown_extra_members_are_denied(tmp_path):
    directory = tmp_path / "normalized"
    stage_normalized_candidate(ROOT, directory)
    (directory / "unauthorized").write_text("no")
    with pytest.raises(Plan2NormalizationError, match="unknown"):
        stage_normalized_candidate(ROOT, directory)
    (directory / "unauthorized").unlink()
    link = tmp_path / "other"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(Plan2NormalizationError, match="symlink"):
        stage_normalized_candidate(ROOT, link)


def test_restart_verifier_rejects_valid_json_with_resealed_but_false_records(tmp_path):
    directory = tmp_path / "normalized"
    stage_normalized_candidate(ROOT, directory)
    manifest_path = directory / "manifest.json"
    import json

    obj = json.loads(manifest_path.read_text())
    obj["records"][0]["language"] = "invalid-not-a-language"
    core = copy.deepcopy(obj)
    core.pop("manifest_sha256")
    obj["manifest_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(obj))
    with pytest.raises(Plan2NormalizationError, match="provenance"):
        verify_normalized_candidate(directory)


def test_two_identical_inputs_and_same_policy_have_same_manifest_without_disk():
    data = "Hello world\n\nAnother paragraph".encode()
    hashes = [normalize_verified_candidate(data, _predecessor(data))[1][
        "manifest_sha256"] for _ in range(3)]
    assert len(set(hashes)) == 1
