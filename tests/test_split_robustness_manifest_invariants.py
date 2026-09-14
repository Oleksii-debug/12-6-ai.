from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    SplitRobustnessError,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    verify_split_family_manifest,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _records() -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for cluster_index in range(60):
        for member in range(2):
            text = f"cluster={cluster_index:03d};member={member};fixture=split-manifest-invariants"
            records.append(
                SplitRecord(
                    id=f"doc-{cluster_index:03d}-{member}",
                    text=text,
                    source_id=f"source-{cluster_index % 4}",
                    modality=("uk", "en", "code")[cluster_index % 3],
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    near_duplicate_cluster_id=f"cluster-{cluster_index:03d}",
                )
            )
    return records


def _family() -> tuple[list[SplitRecord], dict[str, Any]]:
    records = _records()
    spec = SplitFamilySpec(
        eligible_corpus_sha256=eligible_corpus_identity(records),
        dedup_relations_sha256=dedup_relations_identity(records),
        variant_seeds=("split-a", "split-b", "split-c", "split-d"),
        validation_fraction=0.10,
    )
    return records, build_split_family(records, spec)


def _rehash_variant_and_family(family: dict[str, Any], variant_index: int) -> None:
    variant = family["variants"][variant_index]
    variant_core = dict(variant)
    variant_core.pop("split_identity_sha256", None)
    variant["split_identity_sha256"] = _sha256(variant_core)
    family["variant_split_identities"][variant_index] = variant[
        "split_identity_sha256"
    ]
    family_core = dict(family)
    family_core.pop("split_family_identity_sha256", None)
    family["split_family_identity_sha256"] = _sha256(family_core)


def _rehash_all(family: dict[str, Any]) -> None:
    for index, variant in enumerate(family["variants"]):
        variant_core = dict(variant)
        variant_core.pop("split_identity_sha256", None)
        variant["split_identity_sha256"] = _sha256(variant_core)
        family["variant_split_identities"][index] = variant[
            "split_identity_sha256"
        ]
    family_core = dict(family)
    family_core.pop("split_family_identity_sha256", None)
    family["split_family_identity_sha256"] = _sha256(family_core)


def test_valid_manifest_still_verifies_after_semantic_reconstruction() -> None:
    records, family = _family()
    verify_split_family_manifest(records, family)


def test_rehashed_train_validation_overlap_fails_closed() -> None:
    records, family = _family()
    tampered = deepcopy(family)
    variant = tampered["variants"][0]
    overlapping = variant["validation_record_ids"][0]
    variant["train_record_ids"].append(overlapping)
    variant["train_record_ids"].sort()
    variant["train_documents"] += 1
    _rehash_variant_and_family(tampered, 0)

    with pytest.raises(SplitRobustnessError, match="semantic content mismatch"):
        verify_split_family_manifest(records, tampered)


def test_rehashed_duplicate_partition_id_fails_closed() -> None:
    records, family = _family()
    tampered = deepcopy(family)
    variant = tampered["variants"][0]
    duplicate = variant["train_record_ids"][0]
    variant["train_record_ids"].append(duplicate)
    variant["train_record_ids"].sort()
    variant["train_documents"] += 1
    _rehash_variant_and_family(tampered, 0)

    with pytest.raises(SplitRobustnessError, match="semantic content mismatch"):
        verify_split_family_manifest(records, tampered)


def test_rehashed_validation_cluster_description_drift_fails_closed() -> None:
    records, family = _family()
    tampered = deepcopy(family)
    variant = tampered["variants"][0]
    variant["validation_clusters"] = ["cluster-not-selected"]
    _rehash_variant_and_family(tampered, 0)

    with pytest.raises(SplitRobustnessError, match="semantic content mismatch"):
        verify_split_family_manifest(records, tampered)


def test_rehashed_document_count_drift_fails_closed() -> None:
    records, family = _family()
    tampered = deepcopy(family)
    tampered["variants"][0]["train_documents"] += 1
    _rehash_variant_and_family(tampered, 0)

    with pytest.raises(SplitRobustnessError, match="semantic content mismatch"):
        verify_split_family_manifest(records, tampered)


def test_rehashed_unsupported_algorithm_fails_closed() -> None:
    records, family = _family()
    tampered = deepcopy(family)
    tampered["algorithm"] = "cluster-hash-ranked-greedy-v2"
    for variant in tampered["variants"]:
        variant["algorithm"] = "cluster-hash-ranked-greedy-v2"
    _rehash_all(tampered)

    with pytest.raises(SplitRobustnessError, match="unsupported split algorithm"):
        verify_split_family_manifest(records, tampered)
