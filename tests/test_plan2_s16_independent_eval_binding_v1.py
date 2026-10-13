"""Real S16 EVAL303/EVAL233 source-family reuse without corpus promotion."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_public_domain_books_v1 as books
from tools.plan2_s16_independent_eval_binding_v1 import (
    OUTPUT,
    SELECTION_CONFIG,
    SELECTION_MEMBERS,
    IndependentEvalDenied,
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
