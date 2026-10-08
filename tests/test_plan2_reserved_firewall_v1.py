"""Plan 2 S8: source-level reservation, DATA-232 and immutable restart gates."""
from __future__ import annotations

import copy
import json

import pytest

from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy
from tools import plan2_reserved_firewall_v1 as firewall

TRAIN_TEXT = (
    "Незалежний відкритий навчальний опис фізики розглядає дослідження "
    "атмосферних явищ із детальними прикладами та формулами."
)
EVAL_TEXT = (
    "Захищений оцінювальний текст пояснює принципи програмного тестування "
    "та перевірки алгоритмів у різних незалежних середовищах."
)
FINAL_TEXT = (
    "Захищений підсумковий текст аналізує законодавчі документи різних "
    "відкритих установ та вимоги до перевірки походження."
)


def fixture(source: str, *lines: str):
    raw = "".join(x + "\n" for x in lines).encode("utf-8")
    rows = []
    off = 0
    for index, part in enumerate(raw.splitlines(keepends=True)):
        rows.append({"index": index, "start_byte": off, "end_byte": off + len(part),
                     "sha256": firewall._sha(part)})
        off += len(part)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(rows),
        "records": rows,
        "normalized_sha256": firewall._sha(raw),
        "normalized_bytes": len(raw),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    manifest = {**core, "manifest_sha256": firewall._sha(firewall._canonical(core))}
    proof = privacy.inspect_normalized(manifest, raw, policy_sha256="f" * 64)
    return manifest, raw, proof


def sources(*, contaminated=False):
    train1 = fixture("train-main", TRAIN_TEXT)
    train2 = fixture("train-copy", EVAL_TEXT if contaminated else TRAIN_TEXT + " Додатково.")
    selected = fixture("eval-selection", EVAL_TEXT)
    final = fixture("eval-final", FINAL_TEXT)
    cohorts = (train1, train2, selected, final)
    roles = {
        "train-main": "train_candidate",
        "train-copy": "train_candidate",
        "eval-selection": "selection_validation",
        "eval-final": "final_test",
    }
    declarations = {
        c[0]["source_id"]: {
            "role": roles[c[0]["source_id"]],
            "origin": "lawful_source",
            "normalization_manifest_sha256": c[0]["manifest_sha256"],
            "privacy_manifest_sha256": c[2]["manifest_sha256"],
        } for c in cohorts
    }
    return cohorts, declarations


def inspect(cohorts, declarations):
    return firewall.inspect_firewall(
        cohorts, declarations,
        expected_reservation_sha256=firewall.reservation_identity(declarations),
    )


def test_clean_firewall_order_independent_text_free_and_no_promotion():
    cohorts, declarations = sources()
    a = inspect(cohorts, declarations)
    b = inspect(tuple(reversed(cohorts)), declarations)
    assert a == b
    assert a["retained_record_count"] >= 1
    assert a["reserved_source_ids"] == ["eval-final", "eval-selection"]
    assert a["excluded_data232_record_count"] == 0
    assert a["data232_report"]["status"] == "PASS_CLEAN"
    assert a["evaluation_sources_excluded_before_packing"] is True
    assert a["training_corpus_authorized"] is False
    assert a["tokenizer_fit_authorized"] is False
    assert a["evaluation_authorized"] is False
    durable = json.dumps(a, ensure_ascii=False)
    assert TRAIN_TEXT not in durable and EVAL_TEXT not in durable
    assert all(x.startswith("train-") for x in a["retained_record_ids"])


def test_eval_answer_exact_overlap_quarantines_training_source():
    cohorts, declarations = sources(contaminated=True)
    result = inspect(cohorts, declarations)
    assert result["data232_report"]["status"] == "PASS_WITH_EXCLUSIONS"
    assert result["excluded_data232_record_count"] >= 1
    assert result["retained_record_ids"] == ["train-main:r00000000"]
    assert result["training_corpus_authorized"] is False


def test_unreserved_evaluation_source_fails_closed():
    cohorts, declarations = sources()
    declarations.pop("eval-final")
    with pytest.raises(firewall.FirewallError, match="roles required"):
        inspect(cohorts, declarations)


def test_reservation_source_substitution_denied_by_independent_pin():
    cohorts, declarations = sources()
    expected = firewall.reservation_identity(declarations)
    forged = copy.deepcopy(declarations)
    forged["eval-final"]["role"] = "train_candidate"
    with pytest.raises(firewall.FirewallError, match="reservation"):
        firewall.inspect_firewall(cohorts, forged, expected_reservation_sha256=expected)


def test_manifest_tamper_and_removed_source_are_rejected():
    cohorts, declarations = sources()
    altered = copy.deepcopy(declarations)
    altered["train-main"]["privacy_manifest_sha256"] = "0" * 64
    with pytest.raises(firewall.FirewallError, match="pinned|manifest"):
        inspect(cohorts, altered)
    with pytest.raises(firewall.FirewallError, match="coverage"):
        inspect(cohorts[:-1], declarations)


def test_teacher_generation_cannot_automatically_reenter_training():
    cohorts, declarations = sources()
    declarations["train-main"]["origin"] = "teacher_generated"
    with pytest.raises(firewall.FirewallError, match="teacher"):
        firewall.reservation_identity(declarations)


def test_upstream_tampering_not_hidden_by_rehashed_reservation():
    cohorts, declarations = sources()
    changed = list(cohorts)
    manifest, raw, proof = changed[0]
    changed[0] = (manifest, raw + b"forged\n", proof)
    with pytest.raises(firewall.FirewallError, match="upstream"):
        inspect(tuple(changed), declarations)


def test_immutable_restart_and_clean_rebuild_parity(tmp_path):
    cohorts, declarations = sources()
    result = inspect(cohorts, declarations)
    out = tmp_path / "run"
    firewall.stage_firewall(out, result)
    original = (out / "reserved-firewall-manifest.json").read_bytes()
    firewall.stage_firewall(out, result)
    assert original == (out / "reserved-firewall-manifest.json").read_bytes()
    firewall.stage_firewall(tmp_path / "new", result)
    assert original == (tmp_path / "new" / "reserved-firewall-manifest.json").read_bytes()
    (out / "reserved-firewall-manifest.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(firewall.FirewallError, match="immutable"):
        firewall.stage_firewall(out, result)


def test_symlink_publication_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    cohorts, declarations = sources()
    with pytest.raises(firewall.FirewallError, match="symlink"):
        firewall.stage_firewall(alias / "candidate", inspect(cohorts, declarations))
