"""Section-15 qualification must report real candidate limits, never synthetic DONE."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_terminal_qualification_v1 as terminal

ROOT = Path(__file__).resolve().parents[1]


def test_real_candidate_and_fixture_clean_rebuild_are_not_release(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    report = terminal.stage(ROOT, out)
    assert report["reproducible_clean_builds"] == 2
    assert report["decision"] == "COMPONENT_AUDIT_ONLY_NOT_TERMINAL"
    assert report["terminal_done"] is False
    assert report["production_release_authorized"] is False
    assert report["physical_corpus_training_authorized"] is False
    assert report["evidence"]["real_books_distinct_document_families"] == 3
    assert report["evidence"]["real_books_snapshot_bytes"] == 1_265_481
    assert len(report["evidence"]["real_books_manifest_sha256"]) == 64
    assert len(report["evidence"]["real_books_s10_mechanics_probe_sha256"]) == 64
    assert report["evidence"]["real_books_s10_probe_decision"] == (
        "S10_MECHANICS_ONLY_NOT_PHYSICAL_S9_ADMISSION"
    )
    assert report["evidence"]["real_books_s10_probe_document_clusters"] == 3
    assert report["evidence"]["real_three_family_document_count"] == 15
    assert len(report["evidence"]["real_eval233_final_custody_audit_sha256"]) == 64
    assert report["evidence"]["real_eval233_final_record_count"] == 16
    assert report["evidence"]["real_eval233_final_outcomes_read"] is False
    assert type(report["evidence"]["real_eval233_decontamination_clean"]) is bool
    assert len(report["evidence"]["real_three_family_source_split_probe_sha256"]) == 64
    assert report["evidence"]["real_three_family_source_cluster_count"] == 5
    assert len(report["evidence"]["real_three_family_train_partition_manifest_sha256"]) == 64
    assert report["evidence"]["real_three_family_train_only_document_count"] > 0
    assert len(report["evidence"]["real_three_family_train_only_bpe_manifest_sha256"]) == 64
    assert report["evidence"]["real_three_family_bpe_candidate_vocab_size"] <= 388
    assert report["evidence"]["real_three_family_bpe_target_vocab_size"] == 32768
    assert report["evidence"]["real_three_family_train_only_physical_bytes"] > 0
    assert report["evidence"]["real_three_family_heldout_document_count"] > 0
    assert report["evidence"]["real_three_family_source_split_leakage_count"] == 0
    assert report["evidence"]["real_three_family_total_bytes"] == 1_314_156
    assert len(report["evidence"]["real_three_family_candidate_sha256"]) == 64
    assert type(report["evidence"]["real_three_family_fixture_eval_clean"]) is bool
    assert len(report["evidence"]["d03_real_ua_normalized_physical_sha256"]) == 64
    assert report["evidence"]["d03_real_ua_source_candidate_families"] == 2
    assert report["evidence"]["d03_real_ua_source_candidate_members"] == 12
    assert report["evidence"]["d03_real_ua_source_candidate_bytes"] == 48675
    assert report["evidence"]["real_multifamily_source_candidates_not_s3_s9_admitted"] == 3
    assert len(report["evidence"]["d03_real_ua_source_candidate_sha256"]) == 64
    assert len(report["evidence"]["real_books_physical_heldout_manifest_sha256"]) == 64
    assert type(report["evidence"]["real_books_physical_heldout_clean"]) is bool
    assert len(report["evidence"]["physical_real_text_tokenizer_candidate_sha256"]) == 64
    assert report["evidence"]["physical_real_text_tokenizer_training_bytes"] > 100_000
    assert len(report["evidence"]["physical_real_text_shard_manifest_sha256"]) == 64
    assert report["evidence"]["physical_real_text_packed_target_count"] > 0
    assert len(report["evidence"]["physical_real_text_exposure_manifest_sha256"]) == 64
    assert len(report["evidence"]["physical_real_text_exposure_chain_head_sha256"]) == 64
    assert len(report["evidence"]["physical_s13_s14_independent_readback_sha256"]) == 64
    assert report["evidence"]["physical_split"] == "DENIED_SINGLE_SOURCE_FAMILY"
    assert report["evidence"]["physical_source_family_count"] == 1
    assert report["evidence"]["synthetic_target_count"] > 0
    assert len(report["evidence"]["member_sha256"]) >= 7
    assert len(report["blocking_gates"]) == 5
    published = out / terminal.OUTPUT
    assert json.loads(published.read_bytes()) == report
    assert terminal._digest(terminal._canonical({
        k: v for k, v in report.items() if k != "audit_sha256"
    })) == report["audit_sha256"]


def test_immutable_readback_refuses_tampering(tmp_path: Path, monkeypatch) -> None:
    report = {
        "schema_version": terminal.SCHEMA,
        "terminal_done": False,
        "production_release_authorized": False,
    }
    monkeypatch.setattr(terminal, "audit", lambda _: report)
    directory = tmp_path / "audit"
    assert terminal.stage(ROOT, directory) == report
    target = directory / terminal.OUTPUT
    target.write_bytes(b'{"terminal_done":true}\n')
    with pytest.raises(terminal.QualificationDenied, match="immutable"):
        terminal.stage(ROOT, directory)
    assert target.read_bytes() == b'{"terminal_done":true}\n'


def test_symlink_publication_refused_before_processing(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    monkeypatch.setattr(terminal, "audit", lambda _: pytest.fail("unsafe audit"))
    with pytest.raises(terminal.QualificationDenied, match="symlink"):
        terminal.stage(ROOT, link)


def test_missing_physical_source_does_not_fallback_to_fixture(monkeypatch) -> None:
    def no_source(*_args, **_kwargs):
        raise terminal.physical.Plan2MaterializationError("source missing")

    monkeypatch.setattr(terminal.physical, "stage_candidate_cohort", no_source)
    with pytest.raises(terminal.physical.Plan2MaterializationError, match="missing"):
        terminal.audit(ROOT)


def test_unexpected_split_error_is_not_misclassified(tmp_path: Path, monkeypatch) -> None:
    def broken_split(*_args, **_kwargs):
        raise terminal.split.Plan2SplitError("unexpected integrity failure")

    monkeypatch.setattr(terminal.split, "stage_candidate", broken_split)
    with pytest.raises(terminal.QualificationDenied, match="unexpected physical split"):
        terminal.audit(ROOT)


def test_changed_physical_family_count_refuses_stale_audit(monkeypatch) -> None:
    def multiple_families(*_args, **_kwargs):
        return {
            "training_corpus_authorized": False,
            "contributions": {"family": {"document-a": {}, "document-b": {}}},
        }

    monkeypatch.setattr(terminal.mixture, "stage_mixture", multiple_families)
    with pytest.raises(terminal.QualificationDenied, match="source-family count changed"):
        terminal.audit(ROOT)


def _signed_gate(path: Path, **fields):
    if path.name == "privacy-manifest.json":
        fields = {
            "source_id": "fixture.source",
            "normalization_manifest_sha256": "a" * 64,
            **fields,
        }
    core = {"training_corpus_authorized": False, **fields}
    result = {**core, "manifest_sha256": terminal._digest(terminal._canonical(core))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(terminal._canonical(result))
    return result


def test_physical_gate_self_hash_and_authority_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "privacy-manifest.json"
    proof = _signed_gate(path, candidate_record_count=2)
    assert terminal._read_gate(path) == proof
    forged = {**proof, "training_corpus_authorized": True}
    core = {k: v for k, v in forged.items() if k != "manifest_sha256"}
    forged["manifest_sha256"] = terminal._digest(terminal._canonical(core))
    path.write_bytes(terminal._canonical(forged))
    with pytest.raises(terminal.QualificationDenied, match="authorizes training"):
        terminal._read_gate(path)
    forged["training_corpus_authorized"] = False
    path.write_bytes(terminal._canonical(forged))
    with pytest.raises(terminal.QualificationDenied, match="hash mismatch"):
        terminal._read_gate(path)


def test_partial_s5_s6_source_privacy_coverage_is_refused(tmp_path: Path) -> None:
    """One good S5 receipt must not cover a forged second S6 source."""
    candidate = tmp_path / "physical-candidate"
    firewall = candidate / "firewall"
    near = firewall / "near"
    exact = near / "exact"
    privacy = _signed_gate(exact / "privacy" / "privacy-manifest.json")
    dedup = _signed_gate(
        exact / "exact-dedup-manifest.json",
        sources=[
            {"source_id": privacy["source_id"],
             "normalization_manifest_sha256":
                 privacy["normalization_manifest_sha256"],
             "privacy_manifest_sha256": privacy["manifest_sha256"]},
            {"privacy_manifest_sha256": "0" * 64},
        ],
    )
    near_receipt = _signed_gate(
        near / "near-dedup-manifest.json",
        upstream_exact_manifest_sha256=dedup["manifest_sha256"],
    )
    reserved = _signed_gate(
        firewall / "reserved-eval-firewall-manifest.json",
        physical_s7_manifest_sha256=near_receipt["manifest_sha256"],
    )
    core = {
        "training_corpus_authorized": False,
        "physical_s8_manifest_sha256": reserved["manifest_sha256"],
    }
    mixture_hash = terminal._digest(terminal._canonical(core))
    (candidate / "corpus-mixture-manifest.json").write_bytes(
        terminal._canonical({**core, "dataset_candidate_sha256": mixture_hash})
    )
    with pytest.raises(terminal.QualificationDenied, match="S5-S6 privacy"):
        terminal._audit_physical_gates(candidate, mixture_hash)


def test_physical_chain_rejects_disconnected_signed_receipts(tmp_path: Path) -> None:
    candidate = tmp_path / "physical-candidate"
    fw = candidate / "firewall"
    near = fw / "near"
    exact = near / "exact"
    privacy = _signed_gate(exact / "privacy" / "privacy-manifest.json")
    exact_receipt = _signed_gate(
        exact / "exact-dedup-manifest.json",
        sources=[{"source_id": privacy["source_id"],
             "normalization_manifest_sha256":
                 privacy["normalization_manifest_sha256"],
             "privacy_manifest_sha256": privacy["manifest_sha256"]}],
    )
    near_receipt = _signed_gate(
        near / "near-dedup-manifest.json",
        upstream_exact_manifest_sha256=exact_receipt["manifest_sha256"],
    )
    reserved = _signed_gate(
        fw / "reserved-eval-firewall-manifest.json",
        physical_s7_manifest_sha256=near_receipt["manifest_sha256"],
        real_final_test_material_accessed=False,
    )
    mixture_core = {
        "training_corpus_authorized": False,
        "physical_s8_manifest_sha256": reserved["manifest_sha256"],
    }
    mixture_hash = terminal._digest(terminal._canonical(mixture_core))
    mixture = {**mixture_core, "dataset_candidate_sha256": mixture_hash}
    (candidate / "corpus-mixture-manifest.json").write_bytes(
        terminal._canonical(mixture)
    )
    observed = terminal._audit_physical_gates(candidate, mixture_hash)
    assert observed["reserved_eval"] == reserved["manifest_sha256"]
    _signed_gate(
        near / "near-dedup-manifest.json",
        upstream_exact_manifest_sha256="0" * 64,
    )
    with pytest.raises(terminal.QualificationDenied, match="lineage disconnected"):
        terminal._audit_physical_gates(candidate, mixture_hash)


@pytest.mark.parametrize("unsafe_flag", [
    "training_corpus_authorized",
    "production_release_authorized",
    "physical_tokenizer_fit_authorized",
    "tokenizer_fit_authorized",
    "evaluation_authorized",
    "generated_auto_reentry_authorized",
    "raw_text_emitted",
    "real_final_test_material_accessed",
])
def test_signed_mixture_cannot_grant_training(
    tmp_path: Path, unsafe_flag: str,
) -> None:
    candidate = tmp_path / "physical-candidate"
    fw = candidate / "firewall"
    near = fw / "near"
    exact = near / "exact"
    privacy = _signed_gate(exact / "privacy" / "privacy-manifest.json")
    exact_receipt = _signed_gate(
        exact / "exact-dedup-manifest.json",
        sources=[{"source_id": privacy["source_id"],
             "normalization_manifest_sha256":
                 privacy["normalization_manifest_sha256"],
             "privacy_manifest_sha256": privacy["manifest_sha256"]}],
    )
    near_receipt = _signed_gate(
        near / "near-dedup-manifest.json",
        upstream_exact_manifest_sha256=exact_receipt["manifest_sha256"],
    )
    reserved = _signed_gate(
        fw / "reserved-eval-firewall-manifest.json",
        physical_s7_manifest_sha256=near_receipt["manifest_sha256"],
    )
    altered = {
        "training_corpus_authorized": False,
        "physical_s8_manifest_sha256": reserved["manifest_sha256"],
    }
    altered[unsafe_flag] = True
    forged_hash = terminal._digest(terminal._canonical(altered))
    (candidate / "corpus-mixture-manifest.json").write_bytes(
        terminal._canonical({**altered, "dataset_candidate_sha256": forged_hash})
    )
    expected = ("lineage disconnected" if unsafe_flag == "training_corpus_authorized"
                else "S9 mixture incorrectly authorizes")
    with pytest.raises(terminal.QualificationDenied, match=expected):
        terminal._audit_physical_gates(candidate, forged_hash)


@pytest.mark.parametrize("unsafe_flag", [
    "tokenizer_fit_authorized",
    "evaluation_authorized",
    "generated_auto_reentry_authorized",
    "raw_text_emitted",
    "real_final_test_material_accessed",
    "production_release_authorized",
    "physical_tokenizer_fit_authorized",
])
def test_signed_gate_rejects_authority_and_payload_leaks(
    tmp_path: Path, unsafe_flag: str,
) -> None:
    target = tmp_path / "gate.json"
    _signed_gate(target, **{unsafe_flag: True})
    with pytest.raises(terminal.QualificationDenied, match="incorrectly authorizes"):
        terminal._read_gate(target)


@pytest.mark.parametrize("defect", [
    "source_id",
    "normalization_manifest_sha256",
    "duplicate_source",
])
def test_s5_s6_source_identity_fails_closed_even_with_valid_self_hashes(
    tmp_path: Path, defect: str,
) -> None:
    """A validly re-signed S6 cannot borrow one unrelated S5 privacy proof."""
    candidate = tmp_path / "candidate"
    fw = candidate / "firewall"
    near = fw / "near"
    exact = near / "exact"
    privacy = _signed_gate(exact / "privacy" / "privacy-manifest.json")
    legitimate = {
        "source_id": privacy["source_id"],
        "normalization_manifest_sha256":
            privacy["normalization_manifest_sha256"],
        "privacy_manifest_sha256": privacy["manifest_sha256"],
    }
    altered = dict(legitimate)
    if defect == "source_id":
        altered["source_id"] = "forged.source"
    elif defect == "normalization_manifest_sha256":
        altered["normalization_manifest_sha256"] = "b" * 64
    sources = ([legitimate, altered] if defect == "duplicate_source"
               else [altered])
    dedup = _signed_gate(exact / "exact-dedup-manifest.json",
                         sources=sources)
    near_receipt = _signed_gate(
        near / "near-dedup-manifest.json",
        upstream_exact_manifest_sha256=dedup["manifest_sha256"],
    )
    reserved = _signed_gate(
        fw / "reserved-eval-firewall-manifest.json",
        physical_s7_manifest_sha256=near_receipt["manifest_sha256"],
    )
    core = {
        "training_corpus_authorized": False,
        "physical_s8_manifest_sha256": reserved["manifest_sha256"],
    }
    mixture_hash = terminal._digest(terminal._canonical(core))
    (candidate / "corpus-mixture-manifest.json").write_bytes(
        terminal._canonical({**core, "dataset_candidate_sha256": mixture_hash})
    )
    with pytest.raises(terminal.QualificationDenied, match="S5-S6 privacy"):
        terminal._audit_physical_gates(candidate, mixture_hash)
