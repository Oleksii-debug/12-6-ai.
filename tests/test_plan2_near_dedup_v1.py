"""Plan 2 S7 near/global dedup: deterministic families, audit, failure/recovery."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_near_dedup_v1 as near
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    "Документація докладно пояснює правила налаштування відкритого "
    "програмного забезпечення для українських дослідників та інженерів системи."
)


def _fixture(source: str, *lines: str):
    raw = "".join(line + "\n" for line in lines).encode("utf-8")
    offsets = []
    cursor = 0
    for index, chunk in enumerate(raw.splitlines(keepends=True)):
        offsets.append({"index": index, "start_byte": cursor,
                        "end_byte": cursor + len(chunk), "sha256": near._sha(chunk)})
        cursor += len(chunk)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(offsets),
        "records": offsets,
        "normalized_sha256": near._sha(raw),
        "normalized_bytes": len(raw),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    manifest = {**core, "manifest_sha256": near._sha(near._canonical(core))}
    receipt = privacy.inspect_normalized(
        manifest, raw, policy_sha256="f" * 64)
    return manifest, raw, receipt


def test_exact_survivors_only_and_near_family_cap_are_order_independent():
    a = _fixture("source.a", BASE, BASE.upper())
    b = _fixture("source.b", BASE + " Також.", BASE)
    first = near.inspect_near((a, b))
    second = near.inspect_near((b, a))
    assert first == second
    # S6 removes the identical BASE record; S7 groups the remaining
    # casefold mirror and derivative without admitting raw text.
    assert first["input_exact_survivor_count"] == 3
    assert first["retained_record_count"] == 1
    assert first["suppressed_near_record_count"] == 2
    assert first["retained_record_ids"] == ["source.a:r00000000"]
    assert len(first["near_families"]) == 1
    assert first["near_families"][0]["retained_family_cap"] == 1
    assert first["training_corpus_authorized"] is False
    assert BASE not in json.dumps(first, ensure_ascii=False)


def test_no_transitive_single_link_or_order_based_representative():
    a = _fixture("z", BASE)
    b = _fixture("a", "Система застосовує довільний алгоритм до текстових файлів і незалежних архівів.")
    c = _fixture("b", BASE.lower())
    x = near.inspect_near((a, b, c))
    y = near.inspect_near((c, a, b))
    assert x == y
    assert x["retained_record_count"] == 2
    assert x["retained_record_ids"] == ["a:r00000000", "b:r00000000"]


@pytest.mark.parametrize("left,right,expected", [
    (
        "Публічний опис версії 2026 надає документацію для українських "
        "дослідників відкритої системи даних.",
        "Публічний опис версії 2027 надає документацію для українських "
        "дослідників відкритої системи даних.",
        "TEMPLATE_NUMERIC_VARIANT",
    ),
    (
        "Технічний документ докладно пояснює процедуру реєстрації файлів "
        "та законного використання українських відкритих архівів.",
        "ТЕХНІЧНИЙ ДОКУМЕНТ ДОКЛАДНО ПОЯСНЮЄ ПРОЦЕДУРУ РЕЄСТРАЦІЇ ФАЙЛІВ "
        "ТА ЗАКОННОГО ВИКОРИСТАННЯ УКРАЇНСЬКИХ ВІДКРИТИХ АРХІВІВ.",
        "MIRROR_CASEFOLD",
    ),
    (
        "Короткий опис відкритого навчального матеріалу.",
        "Короткий опис закритого навчального матеріалу.",
        None,
    ),
])
def test_matcher_policy_and_short_text_fail_closed(left, right, expected):
    assert near.match_kind(left, right) == expected


def test_numeric_template_relation_requires_changed_digits():
    base = (
        "Публічний архів надає офіційний опис версії 2026 "
        "програмного компонента з відкритою документацією."
    )
    # A punctuation-only mirror must never be labeled as a changed-number template.
    punctuation = base.replace("документацією.", "документацією .")
    assert near.match_kind(base, punctuation) != "TEMPLATE_NUMERIC_VARIANT"
    assert near.match_kind(base, base.replace("2026", "2027")) == (
        "TEMPLATE_NUMERIC_VARIANT"
    )
    without_digits = base.replace("версії 2026", "версії системи")
    no_digit_mirror = without_digits.replace("документацією.", "документацією .")
    assert near.match_kind(without_digits, no_digit_mirror) != (
        "TEMPLATE_NUMERIC_VARIANT"
    )


def test_negative_forged_s6_privacy_payload_or_promotion_is_denied():
    manifest, payload, receipt = _fixture("a", BASE)
    for mutation in ("payload", "receipt", "promote"):
        changed = copy.deepcopy(receipt)
        bad_payload = payload
        if mutation == "payload":
            bad_payload += b"x"
        elif mutation == "receipt":
            changed["manifest_sha256"] = "0" * 64
        else:
            changed["training_corpus_authorized"] = True
        with pytest.raises(near.NearDedupError, match="S6"):
            near.inspect_near(((manifest, bad_payload, changed),))


def test_privacy_tombstone_never_reintroduced():
    manifest, payload, _ = _fixture("a", BASE, BASE.lower())
    removed = privacy.inspect_normalized(
        manifest, payload, policy_sha256="f" * 64,
        tombstones=("a:r00000001",))
    result = near.inspect_near(((manifest, payload, removed),))
    assert result["input_exact_survivor_count"] == 1
    assert result["retained_record_ids"] == ["a:r00000000"]


def test_versioned_false_positive_false_negative_audit():
    audit = near.audit_samples(ROOT)
    assert audit["false_positive"] == 0
    assert audit["true_positive"] >= 3
    assert audit["false_negative"] == 1
    assert audit["true_negative"] >= 2


def test_versioned_audit_duplicate_json_key_is_denied(tmp_path):
    fixture = tmp_path / near.AUDIT
    fixture.parent.mkdir(parents=True)
    original = (ROOT / near.AUDIT).read_text(encoding="utf-8")
    changed = original.replace(
        '"schema": "12-6.plan2-near-audit.v1"',
        '"schema": "forged", "schema": "12-6.plan2-near-audit.v1"',
        1,
    )
    assert original != changed
    fixture.write_text(changed, encoding="utf-8")
    with pytest.raises(near.NearDedupError, match="audit fixture invalid"):
        near.audit_samples(tmp_path)


def test_candidate_physical_restart_and_clean_rebuild(tmp_path):
    first = near.stage_near(ROOT, tmp_path / "one")
    again = near.stage_near(ROOT, tmp_path / "one")
    rebuilt = near.stage_near(ROOT, tmp_path / "clean")
    assert first == again == rebuilt
    assert first["input_exact_survivor_count"] == 50
    assert (first["retained_record_count"] +
            first["suppressed_near_record_count"] == 50)
    assert first["versioned_audit"]["false_positive"] == 0
    path = tmp_path / "one" / "near-dedup-manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == first
    path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(near.NearDedupError, match="immutable"):
        near.stage_near(ROOT, tmp_path / "one")


def test_symlink_publication_denied(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "alias").symlink_to(
        tmp_path / "actual", target_is_directory=True)
    with pytest.raises(near.NearDedupError, match="symlink"):
        near.stage_near(ROOT, tmp_path / "alias" / "candidate")
