"""Plan2 S11 independent fixture comparisons, negative and restart contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_tokenizer_candidates_v1 as cand

ROOT = Path(__file__).resolve().parents[1]
POLICY = (ROOT / cand.POLICY_PATH).read_bytes()
FIXTURE = (ROOT / cand.FIXTURE_PATH).read_bytes()


def test_pinned_candidate_policy_and_source_identity():
    assert cand._git_blob(POLICY) == cand.POLICY_BLOB
    assert cand._git_blob(FIXTURE) == cand.FIXTURE_BLOB
    assert cand._git_blob(Path(cand.incumbent.__file__).read_bytes()) == cand.BYTE_BLOB


def test_deterministic_multilingual_compression_comparison_and_recommendation():
    result = cand.inspect(POLICY, FIXTURE)
    assert result == cand.inspect(POLICY, FIXTURE)
    core = {k: v for k, v in result.items() if k != "manifest_sha256"}
    assert result["manifest_sha256"] == cand._sha(cand._canonical(core))
    byte, bpe = result["candidate_comparisons"]
    assert byte["total_fixture_proxy_units"] == byte["total_utf8_bytes"]
    assert bpe["total_fixture_proxy_units"] < byte["total_fixture_proxy_units"]
    assert bpe["vocab_target"] == 32768
    assert bpe["prospective_special_names_without_token_ids"] == [
        "<pad>", "<bos>", "<eos>", "<unk>"
    ]
    assert result["recommended_architecture_for_future_fit"] == cand.IDS[1]
    assert {row["stratum"] for row in bpe["sample_metrics"]} == cand.STRATA
    assert all(row["roundtrip_exact"] for row in bpe["sample_metrics"])


def test_no_claim_widening_or_payload_leakage():
    result = cand.inspect(POLICY, FIXTURE)
    for key in ("fitted_tokenizer_runtime_approved", "tokenizer_fit_authorized",
                "training_corpus_authorized", "checkpoint_reuse_authorized",
                "real_final_test_material_accessed", "paid_compute_used"):
        assert result[key] is False
    assert all(row["fitted_vocab_sha256"] is None
               for row in result["candidate_comparisons"])
    encoded = json.dumps(result, ensure_ascii=False)
    assert "Українська інженерна" not in encoded
    assert "import json" not in encoded


def test_lossless_all_strata_including_crlf_unicode_null_and_literal_specials():
    policy = cand._policy(POLICY)
    samples = cand._fixture(FIXTURE)
    for row in samples:
        raw = row["text"].encode("utf-8", "strict")
        for candidate in policy:
            assert b"".join(cand._chunks(raw, candidate)) == raw
            assert b"".join(cand._chunks(raw, candidate)).decode("utf-8") == row["text"]
    boundary = next(row["text"] for row in samples if row["stratum"] == "boundary")
    assert "e\u0301" in boundary and "é" in boundary


@pytest.mark.parametrize("needle", [
    b"schema_version", b"vocab_target", b"normalization", b"special_token_names",
    b"probe_pieces", b"revision",
])
def test_forged_policy_revision_or_fields_fail_closed(needle):
    assert needle in POLICY
    with pytest.raises(cand.CandidateError):
        cand.inspect(POLICY.replace(needle, b"forged", 1), FIXTURE)


@pytest.mark.parametrize("needle", [
    b"uk_01", b"code_01", b"LOCAL_FREE_HANDWRITTEN_TEXTS_NO_CORPUS",
])
def test_forged_fixture_or_replayed_reserve_fails_closed(needle):
    assert needle in FIXTURE
    with pytest.raises(cand.CandidateError):
        cand.inspect(POLICY, FIXTURE.replace(needle, b"forged", 1))


def test_strict_json_rejects_duplicates_nonfinite_and_invalid_utf8():
    for bad in (b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff"):
        with pytest.raises(cand.CandidateError):
            cand._pinned_json(bad, cand._git_blob(bad))


def test_forged_incumbent_byte_source_rejected(monkeypatch, tmp_path):
    fake = tmp_path / "byte.py"
    fake.write_text("# swapped tokenizer", encoding="utf-8")
    monkeypatch.setattr(cand.incumbent, "__file__", str(fake))
    with pytest.raises(cand.CandidateError, match="identity drift"):
        cand.inspect(POLICY, FIXTURE)


def test_restart_clean_rebuild_and_tampered_manifest_fail_closed(tmp_path):
    first = cand.stage(ROOT, tmp_path / "one")
    restart = cand.stage(ROOT, tmp_path / "one")
    fresh = cand.stage(ROOT, tmp_path / "two")
    assert first == restart == fresh
    p = tmp_path / "one" / "tokenizer-candidate-manifest.json"
    q = tmp_path / "two" / "tokenizer-candidate-manifest.json"
    assert p.read_bytes() == q.read_bytes()
    p.write_text('{"forged":true}', encoding="utf-8")
    with pytest.raises(cand.CandidateError, match="stage denied"):
        cand.stage(ROOT, tmp_path / "one")


def test_symlink_destination_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(cand.CandidateError, match="symlink"):
        cand.stage(ROOT, tmp_path / "alias" / "child")
