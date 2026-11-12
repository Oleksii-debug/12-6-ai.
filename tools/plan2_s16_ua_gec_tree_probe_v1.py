"""Hash-only UA-GEC source-tree triage for Plan 2 / Section 16.

Candidate reconnaissance only. Does not read source/test plaintext, authorize
training, certify contributor rights, or independently clear privacy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2-s16-ua-gec-source-tree-probe.v1"
UPSTREAM_COMMIT = "4757f72f192c4a41e4c8fb1d9690a948f87cf6d6"
UPSTREAM_TREE = "450bb08698d91fdfc19e856ac7a372676e70926a"
METADATA_GIT_BLOB = "1d6f69c8f26c8db90a4f0502bba241363242fc18"
HEADER = (
    "id", "author_id", "is_native", "region", "gender", "occupation",
    "submission_type", "source_language", "annotator_id", "partition",
    "is_sensitive",
)
PATH = re.compile(r"^data/(gec-only|gec-fluency)/(train|test)/source/(\d{4})\.src\.txt$")
SHA1 = re.compile(r"^[0-9a-f]{40}$")


class SourceTreeDenied(ValueError):
    """Unverified source identity or split metadata."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise SourceTreeDenied(reason)


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def inspect_metadata_tree(
    metadata_bytes: bytes,
    tree: dict[str, Any],
    *,
    expected_metadata_blob: str = METADATA_GIT_BLOB,
    expected_tree: str = UPSTREAM_TREE,
) -> dict[str, Any]:
    """Inspect pinned Git tree membership without reading corpus text.

    Git's recursive tree response is a remote inventory assertion, not an
    independent checkout/byte audit. Source language on translation metadata
    is the *origin language*, NOT evidence of EN training text.
    """
    _need(type(metadata_bytes) is bytes and _git_blob(metadata_bytes) == expected_metadata_blob,
          "metadata Git blob differs from pinned upstream")
    _need(type(tree) is dict and tree.get("sha") == expected_tree
          and tree.get("truncated") is False and type(tree.get("tree")) is list,
          "upstream tree identity/truncation invalid")
    try:
        text = metadata_bytes.decode("utf-8-sig", "strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        _need(tuple(reader.fieldnames or ()) == HEADER, "metadata schema drift")
        rows = list(reader)
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise SourceTreeDenied("malformed metadata CSV") from exc
    _need(bool(rows), "empty metadata")
    by_id: dict[str, dict[str, str]] = {}
    authors: dict[str, set[str]] = {"train": set(), "test": set()}
    sensitive: Counter[str] = Counter()
    translation_origins: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        _need(type(row) is dict and set(row) == set(HEADER)
              and all(type(v) is str for v in row.values()), "metadata row malformed")
        doc_id, author, partition = row["id"], row["author_id"], row["partition"]
        _need(re.fullmatch(r"\d{4}", doc_id) is not None and doc_id not in by_id,
              "duplicate/unsafe metadata document identity")
        _need(bool(author.strip()) and author == author.strip(),
              "missing or ambiguous author identity")
        _need(partition in authors, "unknown partition")
        _need(row["is_sensitive"] in ("0", "1"), "sensitive flag not explicit")
        _need(row["submission_type"] in ("essay", "translation", "text_donation"),
              "unknown submission type")
        origin = row["source_language"]
        if row["submission_type"] == "translation":
            _need(origin in ("de", "en", "fr", "ru", "pl"),
                  "unknown translation origin language")
            translation_origins[origin] += 1
        else:
            _need(origin == "", "translation origin supplied for nontranslation")
        by_id[doc_id] = row
        authors[partition].add(author)
        counts[partition] += 1
        sensitive[partition] += int(row["is_sensitive"])
    _need(not authors["train"] & authors["test"], "author leakage across train/test")

    source_blobs: dict[tuple[str, str], dict[str, tuple[str, int]]] = defaultdict(dict)
    for item in tree["tree"]:
        _need(type(item) is dict and type(item.get("path")) is str,
              "malformed recursive tree entry")
        match = PATH.fullmatch(item["path"])
        if match is None:
            continue
        variant, partition, doc_id = match.groups()
        _need(item.get("type") == "blob" and type(item.get("sha")) is str
              and SHA1.fullmatch(item["sha"]) is not None
              and type(item.get("size")) is int and item["size"] > 0,
              "invalid physical source blob entry")
        row = by_id.get(doc_id)
        _need(row is not None and row["partition"] == partition,
              "source file absent from metadata or wrong split")
        bucket = source_blobs[(variant, partition)]
        _need(doc_id not in bucket, "duplicate physical source path")
        bucket[doc_id] = (item["sha"], item["size"])

    for partition in ("train", "test"):
        ids = {doc_id for doc_id, row in by_id.items() if row["partition"] == partition}
        left = source_blobs[("gec-only", partition)]
        right = source_blobs[("gec-fluency", partition)]
        _need(set(left) == ids and set(right) == ids,
              "missing physical source document in either variant")
        _need(left == right, "variants have different raw source blobs")
    train = source_blobs[("gec-only", "train")]
    test = source_blobs[("gec-only", "test")]
    train_hashes = Counter(sha for sha, _ in train.values())
    test_hashes = Counter(sha for sha, _ in test.values())
    _need(not set(train_hashes) & set(test_hashes),
          "exact source-content leakage across train/test")
    core = {
        "schema_version": SCHEMA,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": expected_tree,
        "metadata_git_blob": expected_metadata_blob,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "metadata_records": len(rows),
        "split_document_counts": dict(sorted(counts.items())),
        "split_distinct_author_counts": {k: len(v) for k, v in sorted(authors.items())},
        "cross_split_author_overlap": 0,
        "cross_split_exact_blob_overlap": 0,
        "sensitive_flag_counts": dict(sorted(sensitive.items())),
        "translation_origin_language_counts_not_corpus_language": dict(sorted(translation_origins.items())),
        "corpus_language_coverage_independently_verified": False,
        "independent_ua_en_validation_and_test_verified": False,
        "within_split_duplicate_blob_groups": {
            "train": sum(n > 1 for n in train_hashes.values()),
            "test": sum(n > 1 for n in test_hashes.values()),
        },
        "within_split_duplicate_extra_documents": {
            "train": sum(n - 1 for n in train_hashes.values() if n > 1),
            "test": sum(n - 1 for n in test_hashes.values() if n > 1),
        },
        "raw_source_bytes_one_variant": sum(size for _, size in (*train.values(), *test.values())),
        "variant_source_blob_memberships_identical": True,
        "independent_validation_present": False,
        "source_rights_approved": False,
        "record_level_privacy_cleared": False,
        "near_duplicate_decontamination_verified": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_release_authorized": False,
        "production_release_authorized": False,
        "decision": "CANDIDATE_SOURCE_METADATA_ONLY_NOT_ADMITTED",
    }
    return {**core, "receipt_sha256": hashlib.sha256(_canonical(core)).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = args.metadata.read_bytes()
        tree = json.loads(args.tree.read_text(encoding="utf-8"))
        receipt = inspect_metadata_tree(metadata, tree)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(_canonical(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
