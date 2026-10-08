"""Plan-2 source-inventory boundary over existing D03/source-rights authorities.

This is an index and validation boundary, not a second acquisition, deduplication,
or rights adjudication framework. In particular, an inventory 'accepted' source
is NOT itself authorization for tokenizer fitting or training.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "12-6.plan2-source-inventory.v1"
SOURCE_KEYS = {
    "source_id", "source_family", "location", "acquisition_method",
    "language", "modality", "update_cadence", "status", "provenance",
}
PROVENANCE_KEYS = {"authority_path", "upstream_revision", "content_sha256"}
SOURCE_ID = re.compile(r"^[a-z][a-z0-9._:-]{2,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATUSES = frozenset({"accepted", "candidate", "rejected"})
METHODS = frozenset({"pinned_repository", "immutable_snapshot", "local_fixture", "authenticated_import"})
CADENCES = frozenset({"pinned", "one_off", "daily", "weekly", "monthly"})
MODALITIES = frozenset({"text", "code", "multimodal"})


class SourceInventoryError(ValueError):
    """Unsafe, incomplete, or drifted source identity."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceInventoryError(f"{name} must be nonempty and canonical")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise SourceInventoryError(f"{name} contains control characters")
    return value


def validate_source(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != SOURCE_KEYS:
        raise SourceInventoryError("source fields missing or unexpected")
    source = dict(row)
    if not SOURCE_ID.fullmatch(_text(source["source_id"], "source_id")):
        raise SourceInventoryError("source_id syntax invalid")
    if not SOURCE_ID.fullmatch(_text(source["source_family"], "source_family")):
        raise SourceInventoryError("source_family syntax invalid")
    location = _text(source["location"], "location")
    if not (location.startswith("https://") or location.startswith("repo://")):
        raise SourceInventoryError("location must be pinned HTTPS or repo path")
    if any(x in location for x in ("?", "#", "..", "@", "\\")):
        raise SourceInventoryError("location is not a stable unambiguous locator")
    if source["acquisition_method"] not in METHODS:
        raise SourceInventoryError("unsupported acquisition method")
    if source["modality"] not in MODALITIES:
        raise SourceInventoryError("unsupported modality")
    if source["update_cadence"] not in CADENCES:
        raise SourceInventoryError("unsupported update cadence")
    lang = _text(source["language"], "language")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,8})?", lang):
        raise SourceInventoryError("invalid explicit language")
    if source["status"] not in STATUSES:
        raise SourceInventoryError("status must be accepted/candidate/rejected")
    provenance = source["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != PROVENANCE_KEYS:
        raise SourceInventoryError("provenance fields missing or unexpected")
    for key in ("authority_path", "upstream_revision"):
        _text(provenance[key], f"provenance.{key}")
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", provenance["authority_path"]):
        raise SourceInventoryError("authority path is not repository relative")
    if not SHA256.fullmatch(_text(provenance["content_sha256"], "content_sha256")):
        raise SourceInventoryError("content_sha256 invalid")
    return source


def build_inventory(revision: str, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    revision = _text(revision, "revision")
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,128}", revision):
        raise SourceInventoryError("revision syntax invalid")
    if not isinstance(sources, (list, tuple)):
        raise SourceInventoryError("sources must be a finite ordered collection")
    rows = [validate_source(x) for x in sources]
    ids = [row["source_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SourceInventoryError("duplicate source identity")
    rows.sort(key=lambda x: x["source_id"])
    doc = {"schema_version": SCHEMA, "revision": revision, "sources": rows,
           "status_counts": {s: sum(row["status"] == s for row in rows)
                             for s in sorted(STATUSES)},
           "training_authorized": False}
    doc["inventory_sha256"] = _digest(doc)
    return doc


def verify_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "schema_version", "revision", "sources", "status_counts",
        "training_authorized", "inventory_sha256",
    }:
        raise SourceInventoryError("inventory envelope invalid")
    if inventory["schema_version"] != SCHEMA or inventory["training_authorized"] is not False:
        raise SourceInventoryError("schema or training authority changed")
    candidate = build_inventory(inventory["revision"], inventory["sources"])
    if candidate != dict(inventory):
        raise SourceInventoryError("inventory identity/order/counts drift")
    return candidate


def diff_inventories(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    old = verify_inventory(before)
    new = verify_inventory(after)
    a = {r["source_id"]: r for r in old["sources"]}
    b = {r["source_id"]: r for r in new["sources"]}
    if old["revision"] == new["revision"] and old["inventory_sha256"] != new["inventory_sha256"]:
        raise SourceInventoryError("same corpus revision has conflicting sources")
    return {
        "before_sha256": old["inventory_sha256"],
        "after_sha256": new["inventory_sha256"],
        "added": sorted(b.keys() - a.keys()),
        "removed": sorted(a.keys() - b.keys()),
        "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
    }


def require_known_inputs(inventory: Mapping[str, Any],
                         members: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Bind physical inputs to accepted source IDs; rights remain a separate gate."""
    index = {s["source_id"]: s for s in verify_inventory(inventory)["sources"]}
    if not isinstance(members, (list, tuple)):
        raise SourceInventoryError("members must be a sequence")
    seen: set[str] = set()
    bindings = []
    for member in members:
        if not isinstance(member, Mapping) or set(member) != {"source_id", "content_sha256"}:
            raise SourceInventoryError("member fields invalid")
        sid = member["source_id"]
        if sid in seen:
            raise SourceInventoryError("duplicate source input")
        seen.add(sid)
        source = index.get(sid)
        if source is None or source["status"] != "accepted":
            raise SourceInventoryError("unknown or non-accepted corpus input")
        if member["content_sha256"] != source["provenance"]["content_sha256"]:
            raise SourceInventoryError("input content fingerprint drift")
        bindings.append({"source_id": sid, "content_sha256": member["content_sha256"]})
    return sorted(bindings, key=lambda row: row["source_id"])
