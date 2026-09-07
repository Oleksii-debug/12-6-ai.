"""Immutable reserved-evaluation material registry for future corpus decontamination."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

INDEX_SCHEMA = "12-6.reserved-evaluation-index.v1"
RESERVATION_SCHEMA = "12-6.reserved-evaluation-material.v1"


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_eval_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_eval_sha256(text: str) -> str:
    return hashlib.sha256(normalized_eval_text(text).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _load_suite_items(repo_root: Path, reservation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    relative = reservation.get("suite_path")
    if not isinstance(relative, str) or not relative.startswith("data/evaluation/"):
        raise ValueError("reservation has invalid suite_path")
    path = repo_root / relative
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        items.append(item)
    if not items:
        raise ValueError(f"{path} contains no evaluation items")
    return tuple(items)


def load_reserved_index(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "data/evaluation/reserved/index.json"
    index = _load_json(path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise ValueError("reserved evaluation index schema mismatch")
    expected = index.get("registry_sha256")
    unsigned = {key: value for key, value in index.items() if key != "registry_sha256"}
    if expected != canonical_json_sha256(unsigned):
        raise ValueError("reserved evaluation index identity mismatch")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reserved evaluation index must contain entries")
    return index


def load_reservation(repo_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative.startswith("data/evaluation/reserved/"):
        raise ValueError("invalid reserved evaluation path")
    path = repo_root / relative
    reservation = _load_json(path)
    if reservation.get("schema_version") != RESERVATION_SCHEMA:
        raise ValueError("reserved evaluation material schema mismatch")
    expected = entry.get("reservation_sha256")
    unsigned = {key: value for key, value in reservation.items() if key != "reservation_sha256"}
    if expected != canonical_json_sha256(unsigned):
        raise ValueError("reservation identity does not match index")
    if reservation.get("reservation_sha256") != expected:
        raise ValueError("reservation self identity mismatch")
    if reservation.get("suite_sha256") != entry.get("suite_sha256"):
        raise ValueError("suite identity differs between reservation and index")
    forbidden = {str(value) for value in reservation.get("forbidden_uses", [])}
    required = {"training", "pretraining", "finetuning", "tokenizer_training", "data_selection"}
    if not required.issubset(forbidden):
        raise ValueError("reservation does not forbid all training/data-selection uses")
    if reservation.get("held_out") is not True:
        raise ValueError("reserved evaluation material must be held out")

    suite_items = _load_suite_items(repo_root, reservation)
    fingerprints = reservation.get("items")
    if not isinstance(fingerprints, list) or len(fingerprints) != len(suite_items):
        raise ValueError("reservation item fingerprints do not cover the suite")
    expected_by_id = {
        str(item["id"]): canonical_json_sha256(item)
        for item in suite_items
        if isinstance(item.get("id"), str)
    }
    observed_by_id = {
        str(item.get("id")): item.get("item_sha256")
        for item in fingerprints
        if isinstance(item, Mapping)
    }
    if observed_by_id != expected_by_id:
        raise ValueError("reservation item fingerprint mismatch")
    return reservation


def load_all_reservations(repo_root: Path) -> tuple[dict[str, Any], ...]:
    index = load_reserved_index(repo_root)
    return tuple(load_reservation(repo_root, entry) for entry in index["entries"])


def _reserved_fragments(repo_root: Path) -> tuple[str, ...]:
    fragments: list[str] = []
    seen: set[str] = set()
    for reservation in load_all_reservations(repo_root):
        for item in _load_suite_items(repo_root, reservation):
            context = item.get("context")
            preferred = item.get("preferred")
            dispreferred = item.get("dispreferred")
            if not all(
                isinstance(value, str) and value
                for value in (context, preferred, dispreferred)
            ):
                raise ValueError("reserved cloze item has invalid text fields")
            for text in (context, context + preferred, context + dispreferred):
                normalized = normalized_eval_text(text)
                if normalized not in seen:
                    seen.add(normalized)
                    fragments.append(normalized)
    return tuple(fragments)


def reserved_full_normalized_hashes(repo_root: Path) -> frozenset[str]:
    hashes: set[str] = set()
    for reservation in load_all_reservations(repo_root):
        for item in _load_suite_items(repo_root, reservation):
            context = str(item["context"])
            for key in ("preferred", "dispreferred"):
                hashes.add(normalized_eval_sha256(context + str(item[key])))
    return frozenset(hashes)


def training_text_collisions(
    repo_root: Path,
    texts: Iterable[str],
) -> list[dict[str, str]]:
    """Find exact or substring leakage of reserved cloze contexts/full alternatives."""

    fragments = _reserved_fragments(repo_root)
    full_hashes = reserved_full_normalized_hashes(repo_root)
    collisions: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        normalized = normalized_eval_text(text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in full_hashes:
            collisions.append(
                {
                    "record_index": str(index),
                    "kind": "exact_full_alternative",
                    "normalized_sha256": digest,
                }
            )
            continue
        for fragment in fragments:
            if fragment in normalized:
                collisions.append(
                    {
                        "record_index": str(index),
                        "kind": "reserved_substring",
                        "normalized_sha256": hashlib.sha256(
                            fragment.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                break
    return collisions


def assert_training_text_not_reserved(repo_root: Path, texts: Iterable[str]) -> None:
    collisions = training_text_collisions(repo_root, texts)
    if collisions:
        raise RuntimeError(f"training material collides with reserved evaluation: {collisions}")
