from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from tools import plan2_normalization_evidence_v1 as norm

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_data324_is_the_same_normalized_snapshot(tmp_path):
    receipt = norm.stage_normalization(ROOT, tmp_path / "run")
    material = json.loads((tmp_path / "run/cohort/manifest.json").read_text())
    assert receipt["source_id"] == material["source_id"]
    assert receipt["upstream_materialization_sha256"] == material["manifest_sha256"]
    assert receipt["normalized_sha256"] == material["normalized"]["sha256"]
    assert receipt["raw_sha256"] == material["raw"]["sha256"]
    assert receipt["policy_version"] == "data324-uk-markdown-nfkc-lf.v1"
    assert receipt["classification"]["language"] == "uk"
    assert receipt["classification"]["modality"] == "text"
    assert receipt["classification"]["language_confidence_heuristic"] < 1
    assert receipt["record_count"] == len(receipt["records"]) > 0
    assert not receipt["training_corpus_authorized"]
    assert not receipt["tokenizer_fit_authorized"]
    assert not receipt["evaluation_authorized"]
    content = (tmp_path / "run/cohort/normalized.utf8").read_bytes()
    offset = 0
    for index, record in enumerate(receipt["records"]):
        assert record["index"] == index
        assert record["start_byte"] == offset
        payload = content[offset:record["end_byte"]]
        assert payload
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        offset = record["end_byte"]
    assert offset == len(content)


def test_exact_restart_and_unrelated_destination_yield_same_receipt(tmp_path):
    first = norm.stage_normalization(ROOT, tmp_path / "a")
    payload = (tmp_path / "a/normalization-manifest.json").read_bytes()
    assert norm.stage_normalization(ROOT, tmp_path / "a") == first
    assert (tmp_path / "a/normalization-manifest.json").read_bytes() == payload
    assert norm.stage_normalization(ROOT, tmp_path / "b") == first
    assert len(list((tmp_path / "a/cohort").iterdir())) == 3


@pytest.mark.parametrize("payload", [b"", b"\x00A", b"\xff", b"\xc3(", b"\x01Hello"])
def test_invalid_binary_encoding_and_controls_rejected(payload):
    with pytest.raises(norm.NormalizationEvidenceError):
        norm.inspect_raw(payload)


@pytest.mark.parametrize("payload", [
    b"Only English Latin letters here.\r\n",
    "Короткий рядок кирилицею".encode(),
    b"1234567890",
    b"English words and few text\n1234\n",
])
def test_unknown_language_is_not_mislabeled(payload):
    normalized, evidence, rows = norm.inspect_raw(payload)
    assert normalized and rows
    assert evidence["language"] == "und"
    assert evidence["language_confidence_heuristic"] == 0
    assert evidence["language_evidence"]["decision"] == "UNKNOWN"
    assert evidence["modality"] == "text"


def test_unicode_and_line_endings_reuse_incumbent(tmp_path):
    raw = "ІЇЄҐ " * 24 + "те\u0301ст\r\nЛінія\rрядок"
    actual, evidence, records = norm.inspect_raw(raw.encode("utf-8"))
    assert b"\r" not in actual
    assert unicodedata.is_normalized("NFKC", actual.decode("utf-8"))
    assert actual == norm.normalize_markdown_uk(raw.encode("utf-8")).encode("utf-8")
    assert len(records) == actual.count(b"\n")
    assert evidence["language"] == "uk"


def test_policy_and_input_digest_change_invalidates_immutable_artifact(tmp_path, monkeypatch):
    dest = tmp_path / "cohort-with-manifest"
    baseline = norm.stage_normalization(ROOT, dest)
    original_policy = norm._policy
    monkeypatch.setattr(norm, "_policy",
                        lambda root: (original_policy(root)[0], "0" * 64))
    with pytest.raises(norm.NormalizationEvidenceError, match="policy/input drift"):
        norm.stage_normalization(ROOT, dest)
    assert json.loads((dest / "normalization-manifest.json").read_text()) == baseline


def test_incumbent_derivative_mismatch_cannot_be_silently_accepted(tmp_path, monkeypatch):
    original_preflight = norm._preflight

    def altered(*args):
        receipt, raw, normalized = original_preflight(*args)
        return receipt, raw, normalized + b"forged"

    monkeypatch.setattr(norm, "_preflight", altered)
    with pytest.raises(norm.NormalizationEvidenceError, match="drift"):
        norm.stage_normalization(ROOT, tmp_path / "bad")


def test_postpublication_corruption_stays_failed(tmp_path):
    destination = tmp_path / "run"
    norm.stage_normalization(ROOT, destination)
    artifact = destination / "normalization-manifest.json"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(norm.NormalizationEvidenceError, match="policy/input drift"):
        norm.stage_normalization(ROOT, destination)


def test_downstream_binding_rejects_mutation_and_unknown_keys(tmp_path):
    receipt = norm.stage_normalization(ROOT, tmp_path / "run")
    binding = {key: receipt[key] for key in (
        "policy_sha256", "raw_sha256", "normalized_sha256", "manifest_sha256")}
    norm.assert_downstream_binding(receipt, binding)
    for key in binding:
        altered = copy.copy(binding)
        altered[key] = "0" * 64
        with pytest.raises(norm.NormalizationEvidenceError, match="stale"):
            norm.assert_downstream_binding(receipt, altered)
    with pytest.raises(norm.NormalizationEvidenceError, match="stale"):
        norm.assert_downstream_binding(receipt, {**binding, "extra": "forged"})


def test_corrupt_s3_member_fails_even_with_valid_s4_manifest(tmp_path):
    destination = tmp_path / "run"
    norm.stage_normalization(ROOT, destination)
    (destination / "cohort/raw.snapshot").write_bytes(b"evil")
    with pytest.raises(norm.NormalizationEvidenceError, match="upstream"):
        norm.stage_normalization(ROOT, destination)


def test_symlink_destination_and_parent_rejected(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (tmp_path / "link").symlink_to(good, target_is_directory=True)
    with pytest.raises(norm.NormalizationEvidenceError, match="symlink"):
        norm.stage_normalization(ROOT, tmp_path / "link" / "child")
    assert not (good / "child").exists()


def test_concurrent_publication_refuses_overwrite(tmp_path, monkeypatch):
    from tools import plan2_physical_materialization_v1 as physical

    destination = tmp_path / "run"
    original_link = physical.os.link
    racing = b"other publisher's receipt"

    def race(source, target):
        if Path(target).name == "normalization-manifest.json":
            Path(target).write_bytes(racing)
        return original_link(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(physical.os, "link", race)
        with pytest.raises(physical.Plan2MaterializationError, match="concurrent"):
            norm.stage_normalization(ROOT, destination)
    assert (destination / "normalization-manifest.json").read_bytes() == racing
    with pytest.raises(norm.NormalizationEvidenceError, match="drift"):
        norm.stage_normalization(ROOT, destination)


@pytest.mark.parametrize("field,replacement", [
    ("unicode", "NFC"),
    ("encoding", "UTF-8 replace"),
    ("policy_id", "unversioned"),
    ("language", "unknown-is-English"),
    ("record_boundary", "arbitrary"),
])
def test_policy_semantics_cannot_drift_without_versioned_code_change(
    tmp_path, field, replacement,
):
    seed = json.loads((ROOT / norm.POLICY_FILE).read_text(encoding="utf-8"))
    seed[field] = replacement
    fixture = tmp_path / norm.POLICY_FILE
    fixture.parent.mkdir(parents=True)
    fixture.write_text(json.dumps(seed), encoding="utf-8")
    with pytest.raises(norm.NormalizationEvidenceError, match="unsupported"):
        norm._policy(tmp_path)
