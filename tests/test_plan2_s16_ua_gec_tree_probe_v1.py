"""Candidate-only UA-GEC source-tree admission controls; no training rights grant."""
from __future__ import annotations

import copy
import csv
import hashlib
import io

import pytest

from tools.plan2_s16_ua_gec_tree_probe_v1 import (
    HEADER,
    SourceTreeDenied,
    inspect_metadata_tree,
)


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def candidate_fixture():
    rows = [
        ["0001", "author1", "1", "kyiv", "", "", "essay", "", "ann1", "train", "0"],
        ["0002", "author2", "1", "lviv", "", "", "translation", "en", "ann1", "train", "0"],
        ["0003", "author3", "1", "odesa", "", "", "text_donation", "", "ann2", "test", "1"],
    ]
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(HEADER)
    writer.writerows(rows)
    metadata = buf.getvalue().encode("utf-8")
    tree = {"sha": "a" * 40, "truncated": False, "tree": []}
    for variant in ("gec-only", "gec-fluency"):
        for document in rows:
            number = document[0]
            split = document[9]
            payload = f"Ukrainian source {number}".encode()
            tree["tree"].append({
                "path": f"data/{variant}/{split}/source/{number}.src.txt",
                "mode": "100644",
                "type": "blob",
                "sha": _git_blob(payload),
                "size": len(payload),
            })
    return rows, metadata, tree


def _metadata(rows):
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(HEADER)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _inspect(metadata, tree):
    return inspect_metadata_tree(
        metadata, tree, expected_metadata_blob=_git_blob(metadata),
        expected_tree=tree["sha"],
    )


def test_positive_synthetic_metadata_remains_unadmitted():
    _, metadata, tree = candidate_fixture()
    result = _inspect(metadata, tree)
    assert result["metadata_records"] == 3
    assert result["split_document_counts"] == {"train": 2, "test": 1}
    assert result["translation_origin_language_counts_not_corpus_language"] == {"en": 1}
    assert result["cross_split_author_overlap"] == 0
    assert result["cross_split_exact_blob_overlap"] == 0
    assert result["variant_source_blob_memberships_identical"] is True
    for flag in (
        "corpus_language_coverage_independently_verified",
        "independent_ua_en_validation_and_test_verified",
        "independent_validation_present", "source_rights_approved",
        "record_level_privacy_cleared", "near_duplicate_decontamination_verified",
        "training_corpus_authorized", "tokenizer_fit_authorized",
        "evaluation_release_authorized", "production_release_authorized",
    ):
        assert result[flag] is False


def test_candidate_receipt_is_order_independent():
    _, metadata, tree = candidate_fixture()
    a = _inspect(metadata, tree)
    tree["tree"].reverse()
    assert a == _inspect(metadata, tree)


@pytest.mark.parametrize("mutation", [
    "duplicate_metadata_id", "author_leakage", "unsafe_sensitive",
    "nontranslation_english", "unknown_submission", "unknown_partition",
])
def test_malformed_metadata_denied(mutation):
    rows, _, tree = candidate_fixture()
    if mutation == "duplicate_metadata_id":
        rows[2][0] = "0002"
    elif mutation == "author_leakage":
        rows[2][1] = rows[0][1]
    elif mutation == "unsafe_sensitive":
        rows[0][10] = "true"
    elif mutation == "nontranslation_english":
        rows[0][7] = "en"
    elif mutation == "unknown_submission":
        rows[0][6] = "machine_generated"
    else:
        rows[0][9] = "validation"
    with pytest.raises(SourceTreeDenied):
        _inspect(_metadata(rows), tree)


@pytest.mark.parametrize("mutation", [
    "missing_document", "wrong_variant", "cross_split_blob_overlap",
    "unknown_document", "truncated_tree", "wrong_tree_sha",
])
def test_malformed_source_tree_denied(mutation):
    _, metadata, tree = candidate_fixture()
    if mutation == "missing_document":
        tree["tree"].pop()
    elif mutation == "wrong_variant":
        tree["tree"][-1]["sha"] = "c" * 40
    elif mutation == "cross_split_blob_overlap":
        tree["tree"][2]["sha"] = tree["tree"][0]["sha"]
        tree["tree"][5]["sha"] = tree["tree"][3]["sha"]
    elif mutation == "unknown_document":
        extra = copy.deepcopy(tree["tree"][0])
        extra["path"] = "data/gec-only/train/source/9999.src.txt"
        tree["tree"].append(extra)
    elif mutation == "truncated_tree":
        tree["truncated"] = True
    else:
        tree["sha"] = "bad"
        with pytest.raises(SourceTreeDenied):
            inspect_metadata_tree(
                metadata, tree, expected_metadata_blob=_git_blob(metadata),
            )
        return
    with pytest.raises(SourceTreeDenied):
        _inspect(metadata, tree)


def test_within_split_exact_duplicate_is_recorded_not_cleared():
    _, metadata, tree = candidate_fixture()
    for index in (1, 4):
        tree["tree"][index]["sha"] = tree["tree"][0]["sha"]
        tree["tree"][index]["size"] = tree["tree"][0]["size"]
    result = _inspect(metadata, tree)
    assert result["within_split_duplicate_extra_documents"]["train"] == 1
    assert result["near_duplicate_decontamination_verified"] is False
    assert result["training_corpus_authorized"] is False
