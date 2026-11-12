"""Real S16 EVAL303/EVAL233 source-family reuse without corpus promotion."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s16_independent_eval_binding_v1 as binding
from tools.plan2_s16_independent_eval_binding_v1 import (
    OUTPUT,
    SELECTION_CONFIG,
    SELECTION_MEMBERS,
    IndependentEvalDenied,
    _scan_selection,
    inspect,
    inspect_selection_rows,
    stage,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence() -> dict:
    return inspect(ROOT)


def selection_inputs():
    config = json.loads(books.read_checked(ROOT, SELECTION_CONFIG))
    raw = books.read_checked(ROOT, SELECTION_MEMBERS)
    return config, raw


def mutate_selection(mutator):
    config, raw = selection_inputs()
    rows = [json.loads(line) for line in raw.splitlines()]
    mutator(config, rows)
    result = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    return config, result


@pytest.mark.parametrize("copied", [False, True])
def test_selection_scan_reuses_matcher_and_retains_only_evidence(monkeypatch, copied):
    source_text = "one two three four five six seven eight nine ten eleven twelve"
    heldout_text = source_text if copied else "red orange blue yellow violet crimson gold green"
    training = {"record_id": "train", "source_id": "train-src", "text": source_text}
    evaluation = {
        "record_id": "val", "source_id": "val-src", "source_family": "val-family",
        "modality": "en", "text": heldout_text,
    }
    sources = {
        "source_members": [{
            "record_id": "train", "source_family": "train-family", "document_family": "train-doc",
            "modality": "en",
        }],
        "source_members_sha256": "1" * 64,
        "g06_rejected_record_ids": [],
    }
    authority = {
        "authority_id": "eval303-selection-validation", "role": "selection_validation",
        "source_sha": binding.eval303.EVAL303_TERMINAL_SOURCE_SHA,
        "identity_sha256": binding.SELECTION_IDENTITY,
    }
    monkeypatch.setattr(binding.split_probe, "physical_rows", lambda *_: [training])
    monkeypatch.setattr(
        binding.eval303, "resolve_eval303_selection_payloads_from_reconstructed",
        lambda **_: ([evaluation], authority, {"resolver_identity_sha256": "2" * 64}),
    )
    result = _scan_selection(ROOT, sources, (Path("unused"),) * 4)
    assert result["clean"] is not copied
    assert result["excluded_training_documents"] == int(copied)
    assert result["training_documents_compared"] == result["selection_documents_compared"] == 1
    assert "text" not in result
    assert source_text not in json.dumps(result)


def test_selection_scan_refuses_missing_resolver_inputs(tmp_path):
    with pytest.raises(IndependentEvalDenied, match="EVAL-303 payload decontamination denied"):
        _scan_selection(ROOT, {}, (tmp_path / "missing",) * 4)


def test_cpython_extension_requires_selection_evidence():
    with pytest.raises(IndependentEvalDenied, match="requires the frozen selection inputs"):
        inspect(ROOT, include_cpython=True)


def test_cpython_extension_denies_changed_license_before_loading_text(monkeypatch):
    monkeypatch.setattr(binding.cpython, "fetch_exact", lambda _url: b"changed license")

    def should_not_load(_authority):
        pytest.fail("source text loaded before the license was verified")

    monkeypatch.setattr(binding.cpython, "verified_accepted_texts", should_not_load)
    with pytest.raises(IndependentEvalDenied, match="license bytes differ"):
        binding._load_cpython_rows(ROOT)


def test_franko_extension_requires_verified_cpython(tmp_path):
    candidate = tmp_path / "unread-candidate.jsonl"
    out = tmp_path / "no-receipt"
    with pytest.raises(IndependentEvalDenied, match="requires.*CPython"):
        stage(ROOT, out, franko_candidate=candidate)
    assert not out.exists()


def test_arxiv_extension_denies_unbound_payload(tmp_path, monkeypatch):
    candidate = tmp_path / "arxiv-candidate.jsonl"
    candidate.write_bytes(b'{"text":"changed candidate"}\n')
    monkeypatch.setattr(binding.arxiv_rights, "validate_repository_bindings", lambda *_: None)

    def should_not_screen(*_):
        pytest.fail("candidate accepted for privacy scan without matching its rights-bound digest")

    monkeypatch.setattr(binding, "_privacy_screen", should_not_screen)
    with pytest.raises(IndependentEvalDenied, match="arXiv source or rights verification denied") as error:
        binding._load_arxiv_rows(ROOT, candidate)
    assert "candidate SHA drift" in str(error.value.__cause__)


@pytest.mark.parametrize("sidecar", ["LICENSE.cc0.txt", "ATTRIBUTION.txt"])
def test_franko_denies_invalid_rights_sidecar_before_candidate_read(tmp_path, monkeypatch, sidecar):
    (tmp_path / "LICENSE.cc0.txt").write_bytes(b"invalid license")
    (tmp_path / "ATTRIBUTION.txt").write_bytes(b"invalid attribution")
    if sidecar == "ATTRIBUTION.txt":
        monkeypatch.setattr(binding.franko_materializer, "validate_license_bytes", lambda *_: {})

    def should_not_project(*_):
        pytest.fail("candidate loaded before rights sidecars were verified")

    monkeypatch.setattr(binding.franko, "validate_and_project_franko", should_not_project)
    with pytest.raises(IndependentEvalDenied, match="source or rights verification denied"):
        binding._load_franko_rows(ROOT, tmp_path / "unread-candidate.jsonl")


def test_candidate_projection_removes_transitive_duplicates_and_reserved_exclusions():
    rows = [
        {"record_id": label, "source_id": label, "source_family": family,
         "modality": "en", "text": "x" * size}
        for label, family, size in [
            ("a", "short", 11), ("b", "long", 40), ("c", "medium", 20), ("d", "reserved", 80),
        ]
    ]
    matches = [
        {"train_record_id_sha256": binding._sha_text(a),
         "peer_record_id_sha256": binding._sha_text(b)}
        for a, b in [("a", "c"), ("c", "b")]
    ]
    excluded = [{"record_id_sha256": binding._sha_text("d")}]
    report = {"match_evidence": matches, "excluded_records": excluded}
    result = binding._candidate_projection(rows, report)
    assert result["surviving_utf8_bytes_by_family_and_modality"] == {"long": {"en": 40}}
    assert result["surviving_records"] == result["excluded_for_reserved_sets"] == 1
    assert result["removed_as_training_duplicates"] == 2
    assert not result["production_admission"]
    assert result == binding._candidate_projection(rows[::-1], {**report, "match_evidence": matches[::-1]})


def test_extended_training_corpus_checks_final_reserve(monkeypatch):
    text = "one two three four five six seven eight nine ten eleven twelve"
    copied = {
        "record_id": "extra", "source_id": "extra-source", "source_family": "extra-family",
        "modality": "en", "text": text,
    }
    validation = dict(copied, record_id="val", source_id="val-source", source_family="val-family",
                      text="red orange blue yellow violet crimson gold green")
    final = dict(copied, record_id="final", source_id="final-source", source_family="final-family")
    authority = {
        "authority_id": "eval303-selection-validation", "role": "selection_validation",
        "source_sha": binding.eval303.EVAL303_TERMINAL_SOURCE_SHA,
        "identity_sha256": binding.SELECTION_IDENTITY,
    }
    monkeypatch.setattr(binding, "_training_rows", lambda *_: [])
    monkeypatch.setattr(
        binding.eval303, "resolve_eval303_selection_payloads_from_reconstructed",
        lambda **_: ([validation], authority, {"resolver_identity_sha256": "2" * 64}),
    )
    result = _scan_selection(
        ROOT, {"source_members_sha256": "1" * 64}, (Path("unused"),) * 4,
        extra_rows=[copied], final_rows=[final],
    )
    assert result["clean"] is False
    assert result["excluded_training_documents"] == 1
    assert result["quarantined_training_families"] == 1
    assert text not in json.dumps(result)


def test_exact_real_ua_en_independent_heldout_family_authorities(evidence: dict):
    assert evidence["role_language_records"] == {
        "train": {"uk": 12, "en": 3},
        "validation": {"uk": 8, "en": 2},
        "test": {"uk": 8, "en": 8},
    }
    assert evidence["role_family_counts"] == {
        "train": 3, "validation": 4, "test": 2,
    }
    assert evidence["all_three_roles_physical_ua_en"] is True
    assert evidence["all_source_families_disjoint_across_roles"] is True
    assert evidence["role_exact_content_overlap_count"] == 0
    assert evidence["evaluation_use_rights_separately_admitted"] is True
    assert evidence["real_final_test_outcomes_read"] is False
    for key in (
        "heldout_fitted_to_tokenizer",
        "current_s15_train_sources_are_corpus_admitted",
        "production_balance_approved",
        "train_selection_cross_source_near_decontamination_verified",
        "production_s3_s9_dataset_authorized", "training_authorized",
        "plan2_evaluation_release_authorized", "tokenizer_fit_authorized",
        "production_release_authorized", "plan9_binding_authorized",
        "terminal_done",
    ):
        assert evidence[key] is False


def test_selection_membership_real_metadata_only():
    config, raw = selection_inputs()
    records = inspect_selection_rows(raw, config)
    assert len(records) == 10
    assert sum(x["language"] == "uk" for x in records) == 8
    assert sum(x["language"] == "en" for x in records) == 2
    assert not any("text" in x for x in records)


@pytest.mark.parametrize("row_index", [0, 2])  # Frozen EN and UA shapes.
@pytest.mark.parametrize("field,value", [
    ("text", "private heldout payload"),
    ("model_score", 1.0),
    ("training_authorized", True),
    ("optimizer_target", [1, 2, 3]),
])
def test_unrecognized_heldout_or_authorization_field_denied(
    row_index: int, field: str, value: object,
):
    config, raw = mutate_selection(
        lambda _c, rows: rows[row_index].__setitem__(field, value)
    )
    with pytest.raises(IndependentEvalDenied, match="unknown or missing metadata"):
        inspect_selection_rows(raw, config)


@pytest.mark.parametrize("row_index", [0, 2])
def test_missing_heldout_authority_field_denied(row_index: int):
    config, raw = mutate_selection(
        lambda _c, rows: rows[row_index].pop("rights_authority_ref")
    )
    with pytest.raises(IndependentEvalDenied, match="unknown or missing metadata"):
        inspect_selection_rows(raw, config)


@pytest.mark.parametrize("field,value", [
    ("training_eligible", True),
    ("future_training_prohibited", False),
    ("tokenizer_fit_eligible", True),
    ("final_test_eligible", True),
    ("selection_eligible", False),
    ("rights_authority_ref", ""),
    ("reservation_authority_ref", ""),
    ("language", "de"),
    ("purpose", "train"),
    ("source_family", "forged.replaced"),
])
def test_forged_selection_member_never_promotes(field: str, value: object):
    config, raw = mutate_selection(lambda _c, rows: rows[0].__setitem__(field, value))
    with pytest.raises(IndependentEvalDenied):
        inspect_selection_rows(raw, config)


def test_duplicate_selection_record_refused():
    config, raw = mutate_selection(
        lambda _c, rows: rows[0].__setitem__("record_id", rows[1]["record_id"])
    )
    with pytest.raises(IndependentEvalDenied, match="duplicate"):
        inspect_selection_rows(raw, config)


def test_duplicate_selected_text_refused():
    config, raw = mutate_selection(
        lambda _c, rows: rows[0].__setitem__("content_sha256", rows[1]["content_sha256"])
    )
    with pytest.raises(IndependentEvalDenied, match="duplicate"):
        inspect_selection_rows(raw, config)


def test_forged_selection_global_release_refused():
    config, raw = selection_inputs()
    forged = copy.deepcopy(config)
    forged["rights_and_reservation"]["all_included_records_training_prohibited"] = False
    with pytest.raises(IndependentEvalDenied, match="scope/rights"):
        inspect_selection_rows(raw, forged)


def test_missing_physical_selection_member_refused():
    config, raw = selection_inputs()
    shortened = b"".join(raw.splitlines(keepends=True)[:-1])
    with pytest.raises(IndependentEvalDenied, match="count"):
        inspect_selection_rows(shortened, config)


def test_duplicate_json_member_keys_refused():
    config, raw = selection_inputs()
    entry = raw.splitlines(keepends=True)[0]
    bad = entry.replace(b'"selection_eligible":true', (
        b'"selection_eligible":true,"selection_eligible":true'
    ))
    assert bad != entry
    forged = bad + b"".join(raw.splitlines(keepends=True)[1:])
    with pytest.raises(IndependentEvalDenied, match="duplicate"):
        inspect_selection_rows(forged, config)


def test_missing_or_mutated_real_selection_source_refused(monkeypatch: pytest.MonkeyPatch):
    original = books.read_checked

    def changed(root: Path, path: str):
        data = original(root, path)
        if path == SELECTION_MEMBERS:
            return data + b"tamper\n"
        return data

    monkeypatch.setattr(books, "read_checked", changed)
    with pytest.raises(IndependentEvalDenied, match="membership bytes drift"):
        inspect(ROOT)


def test_immutable_exact_rebuild_without_heldout_text(tmp_path: Path, evidence: dict):
    first = stage(ROOT, tmp_path / "first")
    restarted = stage(ROOT, tmp_path / "first")
    fresh = stage(ROOT, tmp_path / "fresh")
    assert first == restarted == fresh == evidence
    assert (tmp_path / "first" / OUTPUT).read_bytes() == (
        tmp_path / "fresh" / OUTPUT
    ).read_bytes()
    raw = (tmp_path / "first" / OUTPUT).read_text()
    assert '"text":' not in raw
    assert '"real_final_test_outcomes_read":false' in raw


def test_immutable_output_tampering_fails(tmp_path: Path):
    out = tmp_path / "published"
    stage(ROOT, out)
    (out / OUTPUT).write_bytes(b'{"training_authorized":true}\n')
    with pytest.raises(IndependentEvalDenied, match="immutable"):
        stage(ROOT, out)


def test_symlink_publication_refused_without_source_read(tmp_path: Path):
    folder = tmp_path / "location"
    folder.mkdir()
    symlink = tmp_path / "alias"
    try:
        symlink.symlink_to(folder, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlink unavailable")
    with pytest.raises(IndependentEvalDenied, match="unsafe"):
        stage(ROOT, symlink)
