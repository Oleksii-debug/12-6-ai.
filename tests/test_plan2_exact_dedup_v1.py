from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_exact_dedup_v1 as dedup
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy
from tools.plan2_physical_materialization_v1 import stage_candidate_cohort

ROOT = Path(__file__).resolve().parents[1]


def _cohort(tmp_path):
    base = tmp_path / "p"
    p = privacy.stage_privacy(ROOT, base)
    n = norm.stage_normalization(ROOT, base / "normalization")
    ph = stage_candidate_cohort(ROOT, base / "normalization" / "cohort")
    return {"physical": ph, "normalization": n, "privacy": p}


def _reseal(doc):
    doc["manifest_sha256"] = dedup._hash({k: v for k, v in doc.items()
                                          if k != "manifest_sha256"})


def _another(source, name):
    alt = copy.deepcopy(source)
    orig = alt["physical"]["source_id"]
    for kind in ("physical", "normalization", "privacy"):
        alt[kind]["source_id"] = name
    _reseal(alt["physical"])
    alt["normalization"]["upstream_materialization_sha256"] = alt["physical"]["manifest_sha256"]
    _reseal(alt["normalization"])
    alt["privacy"]["normalization_manifest_sha256"] = alt["normalization"]["manifest_sha256"]
    alt["privacy"]["clean_record_hashes"] = [
        {**row, "record_id": row["record_id"].replace(orig, name, 1)}
        for row in alt["privacy"]["clean_record_hashes"]]
    alt["privacy"]["excluded_records"] = [
        {**row, "record_id": row["record_id"].replace(orig, name, 1)}
        for row in alt["privacy"]["excluded_records"]]
    _reseal(alt["privacy"])
    return alt


def test_real_candidate_immutability_restart_policy_and_no_training(tmp_path):
    out = tmp_path / "candidate"
    a = dedup.stage_exact_dedup(ROOT, out)
    b = dedup.stage_exact_dedup(ROOT, out)
    c = dedup.stage_exact_dedup(ROOT, tmp_path / "fresh")
    assert a == b == c
    assert a["counts"]["record"] == {"input": 50, "surviving": 50,
                                      "duplicate_removed": 0}
    assert a["counts"]["content"]["input"] == 1
    assert a["counts"]["member"]["input"] == 2
    assert a["training_corpus_authorized"] is False
    assert a["tokenizer_fit_authorized"] is False
    assert json.loads((out / "exact-dedup-manifest.json").read_text()) == a
    assert "what-is-kubernetes" in a["input_cohorts"][0]["source_id"]
    assert "Kubernetes is" not in json.dumps(a, ensure_ascii=False)


def test_global_content_member_record_families_sorted_and_deterministic(tmp_path):
    first = _cohort(tmp_path)
    second = _another(first, "fixture.same-normalized")
    a = dedup.build_exact_dedup([second, first])
    b = dedup.build_exact_dedup([first, second])
    assert a == b
    assert a["counts"]["content"] == {"input": 2, "surviving": 1,
                                       "duplicate_removed": 1}
    assert a["counts"]["member"] == {"input": 4, "surviving": 2,
                                      "duplicate_removed": 2}
    assert a["counts"]["record"] == {"input": 100, "surviving": 50,
                                      "duplicate_removed": 50}
    assert len(a["duplicate_families"]) == 53
    assert all(len(f["member_ids"]) == 2 for f in a["duplicate_families"])
    assert all(f["member_ids"] == sorted(f["member_ids"])
               for f in a["duplicate_families"])
    assert a["manifest_sha256"] == dedup._hash({k: v for k, v in a.items()
                                                 if k != "manifest_sha256"})


def test_duplicate_source_lineage_fail_closed(tmp_path):
    first = _cohort(tmp_path)
    with pytest.raises(dedup.ExactDedupError, match="duplicate source"):
        dedup.build_exact_dedup([first, copy.deepcopy(first)])


@pytest.mark.parametrize("component", ["physical", "normalization", "privacy"])
def test_corrupt_any_manifest_self_hash_rejected(tmp_path, component):
    source = _cohort(tmp_path)
    source[component]["manifest_sha256"] = "0" * 64
    with pytest.raises(dedup.ExactDedupError, match="manifest hash drift"):
        dedup.build_exact_dedup([source])


def test_clean_record_hash_forgery_rejected_even_if_resealed(tmp_path):
    source = _cohort(tmp_path)
    source["privacy"]["clean_record_hashes"][0]["sha256"] = "0" * 64
    _reseal(source["privacy"])
    with pytest.raises(dedup.ExactDedupError, match="not bound"):
        dedup.build_exact_dedup([source])


def test_conflicting_normalization_policy_cannot_coalesce(tmp_path):
    source = _cohort(tmp_path)
    alt = _another(source, "fixture.different-policy")
    alt["normalization"]["policy_sha256"] = "a" * 64
    _reseal(alt["normalization"])
    alt["privacy"]["normalization_manifest_sha256"] = alt["normalization"]["manifest_sha256"]
    _reseal(alt["privacy"])
    with pytest.raises(dedup.ExactDedupError, match="incompatible normalization"):
        dedup.build_exact_dedup([source, alt])


def test_members_and_normalized_policy_must_bind(tmp_path):
    source = _cohort(tmp_path)
    source["physical"]["raw"]["sha256"] = "0" * 64
    _reseal(source["physical"])
    source["normalization"]["upstream_materialization_sha256"] = (
        source["physical"]["manifest_sha256"])
    _reseal(source["normalization"])
    source["privacy"]["normalization_manifest_sha256"] = source["normalization"]["manifest_sha256"]
    _reseal(source["privacy"])
    with pytest.raises(dedup.ExactDedupError, match="physical member"):
        dedup.build_exact_dedup([source])


def test_no_clobber_corrupted_manifest_or_symlink(tmp_path):
    out = tmp_path / "old"
    dedup.stage_exact_dedup(ROOT, out)
    (out / "exact-dedup-manifest.json").write_text('{"tampered":true}')
    with pytest.raises(dedup.ExactDedupError, match="immutable"):
        dedup.stage_exact_dedup(ROOT, out)
    (tmp_path / "link").symlink_to(out, target_is_directory=True)
    with pytest.raises(dedup.ExactDedupError, match="symlink"):
        dedup.stage_exact_dedup(ROOT, tmp_path / "link" / "child")


def test_privacy_cannot_be_promoted_to_training(tmp_path):
    source = _cohort(tmp_path)
    source["privacy"]["training_corpus_authorized"] = True
    _reseal(source["privacy"])
    with pytest.raises(dedup.ExactDedupError, match="improperly promoted"):
        dedup.build_exact_dedup([source])
