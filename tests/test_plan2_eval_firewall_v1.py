"""Plan 2 S8 reserved holdout and DATA-232 firewall regression/evidence tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_eval_firewall_v1 as firewall
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy

ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT = (
    "Дослідники документують незалежні правила запуску відкритих програм "
    "та аналізують мережеві протоколи для майбутньої перевірки."
)


def _cohort(source: str, *values: str):
    payload = "".join(value + "\n" for value in values).encode("utf-8")
    offsets = []
    cursor = 0
    for index, raw in enumerate(payload.splitlines(keepends=True)):
        offsets.append({
            "index": index, "start_byte": cursor,
            "end_byte": cursor + len(raw), "sha256": firewall.sha(raw),
        })
        cursor += len(raw)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(offsets),
        "records": offsets,
        "normalized_sha256": firewall.sha(payload),
        "normalized_bytes": len(payload),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    manifest = {
        **core, "manifest_sha256": firewall.sha(
            norm._canonical(core)
        ),
    }
    receipt = privacy.inspect_normalized(
        manifest, payload, policy_sha256="f" * 64
    )
    return manifest, payload, receipt


def _reserved_text(role: str) -> str:
    data = json.loads((ROOT / firewall.RESERVED).read_text(encoding="utf-8"))
    return next(x["members"][0]["text"] for x in data["sets"]
                if x["role"] == role)


def test_pinned_separate_reserved_roles_and_no_outcome_authority():
    rows, meta = firewall.reservation(ROOT)
    assert len(rows) == 2
    assert len({row["record_id"] for row in rows}) == 2
    assert {row["source_family"] for row in rows} == {
        "project-authored-reserved-selection",
        "project-authored-reserved-final",
    }
    assert {x["role"] for x in meta["authorities"]} == {
        "selection_validation", "final_test",
    }


def test_real_incumbent_exact_eval_overlap_is_excluded():
    leaked = _cohort("train-leaked", _reserved_text("final_test"))
    clean = _cohort("train-independent", INDEPENDENT)
    result = firewall.inspect((leaked, clean), ROOT)
    assert result["excluded_record_ids"] == ["train-leaked:r00000000"]
    assert result["retained_record_ids"] == ["train-independent:r00000000"]
    assert result["reserved_before_packing"] is True
    assert result["final_test_outcomes_read"] is False
    assert result["training_corpus_authorized"] is False
    assert _reserved_text("final_test") not in json.dumps(result, ensure_ascii=False)


def test_near_or_casefold_selection_contamination_is_excluded():
    variant = _reserved_text("selection_validation").upper()
    output = firewall.inspect((
        _cohort("train-near", variant),
        _cohort("train-independent", INDEPENDENT),
    ), ROOT)
    assert "train-near:r00000000" in output["excluded_record_ids"]


def test_order_independent_cross_source_contamination():
    overlap = _cohort("z", _reserved_text("final_test"))
    clean = _cohort("a", INDEPENDENT)
    assert firewall.inspect((overlap, clean), ROOT) == (
        firewall.inspect((clean, overlap), ROOT)
    )


def test_generated_teacher_self_generated_quarantined_before_matching():
    keep = _cohort("a", INDEPENDENT)
    generated = _cohort(
        "b",
        ("Новий генерований документ описує спеціальні технічні "
         "правила обробки незалежних наборів українських даних."),
    )
    for origin in ("teacher", "self_generated"):
        result = firewall.inspect(
            (keep, generated), ROOT, origins={"b:r00000000": origin}
        )
        assert "b:r00000000" in result["excluded_record_ids"]
        assert result["retained_record_ids"] == ["a:r00000000"]


def test_unknown_or_forged_origin_is_refused():
    with pytest.raises(firewall.EvalFirewallError, match="generation"):
        firewall.inspect(
            (_cohort("a", INDEPENDENT),), ROOT,
            origins={"a:r00000000": "unknown_provider"}
        )
    with pytest.raises(firewall.EvalFirewallError, match="generation"):
        firewall.inspect(
            (_cohort("a", INDEPENDENT),), ROOT,
            origins={"forged-record": "teacher"}
        )


def test_privacy_receipt_forgery_and_payload_drift_refused():
    manifest, payload, receipt = _cohort("a", INDEPENDENT)
    forged = copy.deepcopy(receipt)
    forged["training_corpus_authorized"] = True
    with pytest.raises(firewall.EvalFirewallError, match="S7"):
        firewall.inspect(((manifest, payload, forged),), ROOT)
    with pytest.raises(firewall.EvalFirewallError, match="S7"):
        firewall.inspect(((manifest, payload + b"x", receipt),), ROOT)


def test_reserved_fixture_replacement_and_duplicate_key_denied(tmp_path):
    data = (ROOT / firewall.RESERVED).read_text(encoding="utf-8")
    fixture = tmp_path / firewall.RESERVED
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        data.replace('"outcome_fields_included": false',
                     '"outcome_fields_included": true'),
        encoding="utf-8",
    )
    with pytest.raises(firewall.EvalFirewallError, match="changed"):
        firewall.reservation(tmp_path)
    fixture.write_text(
        data.replace('"schema_version":',
                     '"schema_version": "forged", "schema_version":', 1),
        encoding="utf-8",
    )
    with pytest.raises(firewall.EvalFirewallError, match="changed"):
        firewall.reservation(tmp_path)


def test_physical_stage_restart_clean_rebuild_and_immutable_tamper(tmp_path):
    first = firewall.stage(ROOT, tmp_path / "first")
    second = firewall.stage(ROOT, tmp_path / "first")
    third = firewall.stage(ROOT, tmp_path / "clean")
    assert first == second == third
    assert first["input_record_count"] == (
        first["retained_count"] + first["excluded_count"]
    )
    assert first["paid_compute_used"] is False
    assert first["raw_text_emitted"] is False
    path = tmp_path / "first" / "eval-firewall-manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == first
    path.write_text('{"forged":true}', encoding="utf-8")
    with pytest.raises(firewall.EvalFirewallError, match="immutable"):
        firewall.stage(ROOT, tmp_path / "first")


def test_symlink_destination_and_reserved_source_denied(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "alias").symlink_to(
        tmp_path / "actual", target_is_directory=True
    )
    with pytest.raises(firewall.EvalFirewallError, match="symlink"):
        firewall.stage(ROOT, tmp_path / "alias" / "candidate")
    fixture = tmp_path / firewall.RESERVED
    fixture.parent.mkdir(parents=True)
    fixture.symlink_to(ROOT / firewall.RESERVED)
    with pytest.raises(firewall.EvalFirewallError, match="unavailable|missing"):
        firewall.reservation(tmp_path)
