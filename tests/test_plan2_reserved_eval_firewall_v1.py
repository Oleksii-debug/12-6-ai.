"""Plan 2 Section 8: reserved holdout firewall, adversarial and restart tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy
from tools import plan2_reserved_eval_firewall_v1 as firewall

ROOT = Path(__file__).resolve().parents[1]
RESERVED = (ROOT / firewall.RESERVE_PATH).read_bytes()
BASE = (
    "Документ докладно пояснює правила налаштування відкритої системи "
    "для українських дослідників та інженерів доступного програмного забезпечення."
)
OTHER = (
    "Незалежний культурний архів зберігає рукописи та історичні джерела "
    "для вивчення громадських ініціатив і освітніх матеріалів."
)


def _fixture(source: str, *lines: str):
    payload = "".join(x + "\n" for x in lines).encode("utf-8")
    offsets = []
    position = 0
    for index, chunk in enumerate(payload.splitlines(keepends=True)):
        offsets.append({
            "index": index,
            "start_byte": position,
            "end_byte": position + len(chunk),
            "sha256": exact._sha(chunk),
        })
        position += len(chunk)
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


def test_reservation_identity_and_separate_roles_are_pinned():
    manifest, rows, authority = firewall._reserve(RESERVED)
    assert manifest["real_final_test_material_accessed"] is False
    assert {r["role"] for r in authority["authorities"]} == {
        "final_test", "selection_validation",
    }
    assert len(rows) == 2
    assert all(r["text"] for r in rows)
    assert all(r["source_sha"] == firewall.RESERVE_GIT_BLOB
               for r in authority["authorities"])


def test_clean_source_is_order_independent_and_candidate_only():
    cohorts = (_fixture("source.b", OTHER), _fixture("source.a", BASE))
    first = firewall.inspect_firewall(cohorts, RESERVED)
    second = firewall.inspect_firewall(tuple(reversed(cohorts)), RESERVED)
    assert first == second
    assert first["decontaminated_record_count"] == 2
    assert first["excluded_record_ids"] == []
    assert first["data232_report"]["status"] == "PASS_CLEAN"
    assert first["data232_report"]["hash_only_evidence"] is True
    assert first["training_corpus_authorized"] is False
    assert first["tokenizer_fit_authorized"] is False
    assert first["evaluation_authorized"] is False
    assert first["generated_auto_reentry_authorized"] is False
    assert first["real_final_test_material_accessed"] is False
    serialized = json.dumps(first, ensure_ascii=False)
    assert BASE not in serialized and OTHER not in serialized
    assert "Незалежна контрольна добірка" not in serialized


def test_exact_eval_overlap_is_excluded_without_exposing_answers():
    _config, eval_rows, _authority = firewall._reserve(RESERVED)
    leaked_text = eval_rows[0]["text"]
    result = firewall.inspect_firewall(
        (_fixture("source.a", leaked_text, BASE),), RESERVED)
    assert result["input_training_candidate_count"] == 2
    assert result["decontaminated_record_count"] == 1
    assert result["excluded_record_ids"] == ["source.a:r00000000"]
    assert result["data232_report"]["status"] == "PASS_WITH_EXCLUSIONS"
    assert leaked_text not in json.dumps(result, ensure_ascii=False)


def test_eval_contamination_transitively_excludes_related_training_family():
    _config, eval_rows, _authority = firewall._reserve(RESERVED)
    base = eval_rows[0]["text"]
    altered = base.replace("процедури", "практики")
    result = firewall.inspect_firewall((
        _fixture("source.a", altered), _fixture("source.b", BASE)
    ), RESERVED)
    assert result["input_training_candidate_count"] in {1, 2}
    assert result["decontaminated_record_count"] == 0
    assert len(result["excluded_record_ids"]) == result["input_training_candidate_count"]


def test_tombstone_or_exact_duplicate_never_reappears():
    normalized, payload, _receipt = _fixture("source.a", BASE, OTHER)
    tombstone = privacy.inspect_normalized(
        normalized, payload, policy_sha256="f" * 64,
        tombstones=("source.a:r00000001",))
    result = firewall.inspect_firewall(
        ((normalized, payload, tombstone),), RESERVED)
    assert result["input_training_candidate_count"] == 1
    assert result["decontaminated_record_ids"] == ["source.a:r00000000"]


@pytest.mark.parametrize("mutation", ["payload", "receipt", "promote"])
def test_forged_source_or_privacy_receipt_fail_closed(mutation):
    normalized, payload, receipt = _fixture("source.a", BASE)
    modified = copy.deepcopy(receipt)
    if mutation == "payload":
        payload += b"forged"
    elif mutation == "receipt":
        modified["manifest_sha256"] = "0" * 64
    else:
        modified["training_corpus_authorized"] = True
    with pytest.raises(firewall.Plan2EvalFirewallError, match="S7/S6"):
        firewall.inspect_firewall(((normalized, payload, modified),), RESERVED)


@pytest.mark.parametrize("mutation", ["content", "origin", "roles", "truncated"])
def test_pinned_holdout_cannot_be_replaced_by_self_issued_samples(mutation):
    changed = RESERVED
    if mutation == "content":
        changed = RESERVED.replace("Незалежна".encode(), "Підроблена".encode(), 1)
    elif mutation == "origin":
        changed = RESERVED.replace(b"LOCAL_FREE_SYNTHETIC", b"REAL_EVALUATION", 1)
    elif mutation == "roles":
        changed = RESERVED.replace(b"selection_validation", b"final_test", 1)
    else:
        changed = RESERVED[:30]
    assert changed != RESERVED
    with pytest.raises(firewall.Plan2EvalFirewallError,
                       match="independent reserved fixture"):
        firewall.inspect_firewall((_fixture("source.a", BASE),), changed)


def test_generated_eval_answer_leakage_is_rejected_and_no_auto_admission():
    _config, eval_rows, _authority = firewall._reserve(RESERVED)
    candidate = {
        "record_id": "teacher.001", "source_id": "teacher.source",
        "source_family": "teacher.family",
        "origin_type": "teacher",
        "text": eval_rows[0]["text"],
    }
    with pytest.raises(firewall.Plan2EvalFirewallError, match="leakage"):
        firewall.inspect_firewall(
            (_fixture("source.a", BASE),), RESERVED,
            generated_candidates=[candidate])
    safe = candidate | {"text": OTHER, "origin_type": "self_generated"}
    receipt = firewall.inspect_firewall(
        (_fixture("source.a", BASE),), RESERVED,
        generated_candidates=[safe])
    assert receipt["generated_candidates_audited"] == 1
    assert receipt["generated_auto_reentry_authorized"] is False
    assert receipt["training_corpus_authorized"] is False


def test_generated_candidate_requires_typed_external_origin():
    bad = {"record_id": "x", "source_id": "s", "source_family": "f",
           "text": OTHER, "origin_type": "eval"}
    with pytest.raises(firewall.Plan2EvalFirewallError,
                       match="untrusted generated"):
        firewall.inspect_firewall(
            (_fixture("source.a", BASE),), RESERVED, generated_candidates=[bad])


def test_current_physical_cohort_restart_rebuild_and_no_clobber(tmp_path):
    first = firewall.stage_firewall(ROOT, tmp_path / "first")
    repeated = firewall.stage_firewall(ROOT, tmp_path / "first")
    rebuilt = firewall.stage_firewall(ROOT, tmp_path / "fresh")
    assert first == repeated == rebuilt
    assert first["input_training_candidate_count"] == 50
    assert first["decontaminated_record_count"] + len(
        first["excluded_record_ids"]) == 50
    assert first["reserved_record_count"] == 2
    path = tmp_path / "first" / "reserved-eval-firewall-manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert path.read_bytes() == (
        tmp_path / "fresh" / "reserved-eval-firewall-manifest.json").read_bytes()
    path.write_text('{"forged": true}', encoding="utf-8")
    with pytest.raises(firewall.Plan2EvalFirewallError, match="staging denied"):
        firewall.stage_firewall(ROOT, tmp_path / "first")


def test_symlink_destination_fail_closed(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "link").symlink_to(
        tmp_path / "actual", target_is_directory=True)
    with pytest.raises(firewall.Plan2EvalFirewallError, match="symlink"):
        firewall.stage_firewall(ROOT, tmp_path / "link" / "candidate")
