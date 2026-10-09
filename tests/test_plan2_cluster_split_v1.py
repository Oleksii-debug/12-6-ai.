"""Plan2 S10: cluster exclusivity, source/permission, restart and refusal."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_cluster_split_v1 as split
from tools import plan2_corpus_mixture_v1 as mixture

ROOT = Path(__file__).resolve().parents[1]
POLICY_RAW = (ROOT / split.POLICY_PATH).read_bytes()


def _sample():
    return split._synthetic_rows()


def _policy():
    return split._policy(POLICY_RAW)


def test_pinned_incumbent_policy_and_exact_three_way_family_exclusivity():
    assert split._git_blob(POLICY_RAW) == split.POLICY_GIT_BLOB
    s9, rows = _sample()
    first = split.build_cluster_split(s9, rows, _policy(), fixture=True)
    second = split.build_cluster_split(s9, rows[::-1], _policy(), fixture=True)
    assert first == second
    all_ids = set(s9["selected_record_ids"])
    groups = [set(first[f"{part}_record_ids"]) for part in
              ("train", "validation", "test")]
    assert all(groups)
    assert groups[0] | groups[1] | groups[2] == all_ids
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    for i in range(16):
        r1 = f"fixture.document.{i:03d}:r00000000"
        r2 = f"fixture.document.{i:03d}:r00000001"
        assert first["record_assignments"][r1] == first["record_assignments"][r2]
    assert first["document_cluster_count"] == 16
    assert first["cluster_leakage_count"] == 0
    assert first["training_corpus_authorized"] is False
    assert first["evaluation_release_authorized"] is False
    assert first["tokenizer_fit_authorized"] is False
    assert first["paid_compute_used"] is False
    assert first["actual_model_tokens"] is None
    assert "Independent public synthetic" not in json.dumps(first)
    assert first["split_manifest_sha256"] == split._sha(
        split._canonical({k: v for k, v in first.items() if k != "split_manifest_sha256"}))


@pytest.mark.parametrize("attack", [
    "self_hash", "training", "tokenizer", "final_test", "ids", "count",
    "missing", "duplicate", "extra", "wrong_source", "blank_payload",
])
def test_upstream_and_record_attacks_are_denied(attack):
    s9, rows = _sample()
    if attack == "self_hash":
        s9["dataset_candidate_sha256"] = "a" * 64
    elif attack in ("training", "tokenizer", "final_test"):
        s9[{"training": "training_corpus_authorized",
            "tokenizer": "tokenizer_fit_authorized",
            "final_test": "real_final_test_material_accessed"}[attack]] = True
        s9["dataset_candidate_sha256"] = split._sha(split._canonical({
            k: v for k, v in s9.items() if k != "dataset_candidate_sha256"}))
    elif attack == "ids":
        s9["selected_record_ids"][-1] = s9["selected_record_ids"][0]
    elif attack == "count":
        s9["selected_record_count"] += 1
    elif attack == "missing":
        rows = rows[:-1]
    elif attack == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif attack == "extra":
        rows.append({"record_id": "external:r00000000", "source_id": "external",
                     "text": "Foreign record"})
    elif attack == "wrong_source":
        rows[0]["source_id"] = "forged"
    elif attack == "blank_payload":
        rows[0]["text"] = "  "
    with pytest.raises(split.Plan2SplitError):
        split.build_cluster_split(s9, rows, _policy(), fixture=True)


@pytest.mark.parametrize("attack", ["seed", "fractions", "boundary", "rights"])
def test_policy_mutation_never_silently_changes_split(attack):
    s9, rows = _sample()
    policy = _policy()
    if attack == "seed":
        policy["seed"] = "0" * 64
    elif attack == "fractions":
        policy["test_fraction"] = 0.3
    elif attack == "boundary":
        policy["cluster_boundary"] = "INDIVIDUAL_RECORD"
    else:
        policy["purpose"] = "TRAINING_AUTHORIZED"
    with pytest.raises(split.Plan2SplitError, match="unapproved split policy"):
        split.build_cluster_split(s9, rows, policy, fixture=True)


def test_single_real_document_family_fails_closed(tmp_path):
    # Actual accepted S9 yields 50 lines of one document; it is not 50 safe clusters.
    receipt = mixture.stage_mixture(ROOT, tmp_path / "accepted-s9")
    rows = [{"record_id": rid, "source_id": rid.split(":r")[0],
             "text": f"Validated placeholder line {ix}"}
            for ix, rid in enumerate(receipt["selected_record_ids"])]
    with pytest.raises(split.Plan2SplitError,
                       match="insufficient independent document families"):
        split.build_cluster_split(receipt, rows, _policy())


def test_forged_incumbent_mechanics_disallowed(monkeypatch, tmp_path):
    fake = tmp_path / "split_robustness.py"
    fake.write_text("# unauthorized replacement", encoding="utf-8")
    monkeypatch.setattr(split.canonical, "__file__", str(fake))
    s9, rows = _sample()
    with pytest.raises(split.Plan2SplitError, match="canonical split mechanics"):
        split.build_cluster_split(s9, rows, _policy(), fixture=True)


def test_restart_fresh_rebuild_and_immutable_corruption(tmp_path):
    a = split.stage_fixture(ROOT, tmp_path / "a")
    b = split.stage_fixture(ROOT, tmp_path / "a")
    c = split.stage_fixture(ROOT, tmp_path / "clean")
    assert a == b == c
    pa = tmp_path / "a" / "cluster-split-manifest.json"
    pb = tmp_path / "clean" / "cluster-split-manifest.json"
    assert pa.read_bytes() == pb.read_bytes()
    pa.write_text('{"forged":true}', encoding="utf-8")
    with pytest.raises(split.Plan2SplitError, match="split manifest publication"):
        split.stage_fixture(ROOT, tmp_path / "a")


def test_symlink_destination_denied(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(split.Plan2SplitError, match="symlink split"):
        split.stage_fixture(ROOT, tmp_path / "alias" / "output")


def test_verified_physical_s9_one_document_candidate_denied(tmp_path):
    # Must use the actual normalized S3/S7/S8/S9 source bytes, not a fake row set.
    with pytest.raises(split.Plan2SplitError,
                       match="S9 physical split admission denied"):
        split.stage_candidate(ROOT, tmp_path / "actual-s9")
    assert not (tmp_path / "actual-s9" / "cluster-split-manifest.json").exists()
