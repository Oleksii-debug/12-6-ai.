"""Plan 2 S7 acceptance: verified lineage, order, caps, audit, recovery and negatives."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_near_dedup_v1 as near
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy

ROOT = Path(__file__).resolve().parents[1]
AUDIT = json.loads((ROOT / near.AUDIT_FILE).read_text(encoding="utf-8"))
BASE = (
    "Це довгий опис відкритої документації програмного забезпечення для "
    "українських користувачів з описом загальних правил перевірки системи "
    "даних тестування безпеки відтворюваності та незалежного відновлення "
    "після завершення роботи локального процесу з контрольованими "
    "джерелами документів правилами оновлення і стандартами перевірки "
    "готового технічного результату."
)
NEAR = BASE.replace("технічного результату", "практичного результату")
DIFFERENT = (
    "Водяні рослини на берегах річок отримують сонячне світло й вологу "
    "під час весняного сезону біля гір та природних луків."
)


def _fixture(source: str, *lines: str):
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    offsets = []
    start = 0
    for index, chunk in enumerate(payload.splitlines(keepends=True)):
        offsets.append({
            "index": index, "start_byte": start,
            "end_byte": start + len(chunk), "sha256": exact._sha(chunk),
        })
        start += len(chunk)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(offsets),
        "records": offsets,
        "normalized_bytes": len(payload),
        "normalized_sha256": exact._sha(payload),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    normalized = {**core, "manifest_sha256": exact._sha(exact._canonical(core))}
    receipt = privacy.inspect_normalized(
        normalized, payload, policy_sha256="f" * 64)
    return normalized, payload, receipt


def test_policy_is_exact_and_audit_reports_fp_and_fn():
    near._policy(json.loads((ROOT / near.POLICY_FILE).read_text()))
    result = near.audit_samples(AUDIT)
    assert result["sample_count"] == 6
    assert result["confusion"] == {"tp": 3, "tn": 3, "fp": 0, "fn": 0}
    assert len(result["audit_sha256"]) == 64
    with pytest.raises(near.NearDedupError, match="policy"):
        near._policy(near.POLICY | {"family_cap": 2})
    forged = copy.deepcopy(AUDIT)
    forged["pairs"][0]["related"] = False
    with pytest.raises(near.NearDedupError, match="audit"):
        near.audit_samples(forged)


def test_near_family_and_caps_are_order_independent_without_raw_text():
    a = _fixture("source.a", BASE, DIFFERENT)
    b = _fixture("source.b", NEAR)
    forward = near.inspect_near((a, b), audit=AUDIT)
    reverse = near.inspect_near((b, a), audit=AUDIT)
    assert forward == reverse
    assert forward["input_exact_retained_count"] == 3
    assert forward["near_retained_count"] == 2
    assert forward["near_excluded_count"] == 1
    assert forward["retained_record_ids"] == [
        "source.a:r00000000", "source.a:r00000001"]
    assert forward["excluded_near_records"][0]["record_id"] == "source.b:r00000000"
    assert forward["near_families"][0]["family_cap"] == 1
    assert forward["parent_exact_manifest_sha256"] == exact.inspect_exact(
        (a, b))["manifest_sha256"]
    assert BASE not in json.dumps(forward, ensure_ascii=False)
    assert NEAR not in json.dumps(forward, ensure_ascii=False)
    assert forward["training_corpus_authorized"] is False
    assert forward["tokenizer_fit_authorized"] is False
    assert forward["evaluation_authorized"] is False


def test_exact_duplicates_are_already_removed_and_do_not_reenter():
    a = _fixture("source.a", BASE, BASE)
    result = near.inspect_near((a,), audit=AUDIT)
    assert result["input_exact_retained_count"] == 1
    assert result["near_excluded_count"] == 0
    assert not result["near_families"]


def test_unrelated_lexical_family_remains_separate():
    result = near.inspect_near((_fixture("source.a", BASE, DIFFERENT),), audit=AUDIT)
    assert result["near_retained_count"] == 2
    assert not result["excluded_near_records"]


@pytest.mark.parametrize("change", ["payload", "receipt", "promotion", "duplicate"])
def test_forged_upstream_authority_fails_closed(change):
    a = _fixture("source.a", BASE)
    n, payload, receipt = a
    if change == "payload":
        payload += b"altered"
    elif change == "receipt":
        receipt = receipt | {"manifest_sha256": "0" * 64}
    elif change == "promotion":
        receipt = receipt | {"evaluation_authorized": True}
    else:
        with pytest.raises(exact.ExactDedupError, match="duplicate"):
            near.inspect_near((a, a), audit=AUDIT)
        return
    with pytest.raises((exact.ExactDedupError, near.NearDedupError)):
        near.inspect_near(((n, payload, receipt),), audit=AUDIT)


def test_current_cohort_restart_rebuild_and_immutable_drift(tmp_path):
    one = near.stage_near(ROOT, tmp_path / "first")
    repeated = near.stage_near(ROOT, tmp_path / "first")
    rebuild = near.stage_near(ROOT, tmp_path / "fresh")
    assert one == repeated == rebuild
    assert one["input_exact_retained_count"] == 50
    assert one["near_retained_count"] + one["near_excluded_count"] == 50
    manifest = tmp_path / "first" / "near-dedup-manifest.json"
    expected = manifest.read_bytes()
    assert expected == (tmp_path / "fresh" / "near-dedup-manifest.json").read_bytes()
    assert json.loads(expected) == one
    manifest.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(near.NearDedupError, match="immutable"):
        near.stage_near(ROOT, tmp_path / "first")


def test_policy_and_audit_are_read_only_authorities(tmp_path):
    (tmp_path / "policy.json").write_text("{}", encoding="utf-8")
    with pytest.raises(near.NearDedupError, match="policy"):
        near._policy({})
    with pytest.raises(near.NearDedupError, match="missing"):
        near._load_json(tmp_path, "absent.json")
    (tmp_path / "bad.json").write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(near.NearDedupError, match="duplicate"):
        near._load_json(tmp_path, "bad.json")


def test_symlink_destination_denied(tmp_path):
    target = tmp_path / "actual"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(near.NearDedupError, match="symlink"):
        near.stage_near(ROOT, link / "candidate")
