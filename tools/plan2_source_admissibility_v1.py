"""Plan 2 source rights and record/member provenance: non-authorizing adapter."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from tools.plan2_source_inventory_v1 import verify_inventory

CATALOG_SCHEMA = "12-6.plan2-rights-catalog.v1"
RECEIPT_SCHEMA = "12-6.plan2-provenance-receipt.v1"
GRANT_KEYS = {"source_id", "rights_class", "license_id", "terms_ref",
              "permission_ref", "legal_basis", "allowed_uses",
              "evidence_path", "evidence_sha256", "revoked"}
RECORD_KEYS = {"source_id", "record_id", "member_id", "origin_member_ref",
               "source_snapshot_sha256", "text"}
RIGHTS_CLASSES = {"public_licensed", "research_only", "private", "unknown"}
USES = {"source_candidate", "research", "training", "release"}
BASES = {"license_grant", "explicit_permission", "public_domain", "research_terms",
         "private_owner", "unknown"}
ID = re.compile(r"^[A-Za-z0-9._:/-]{1,256}$")
HEX = re.compile(r"^[a-f0-9]{64}$")


class SourceRightsError(ValueError):
    """No legal basis, incomplete rights, or broken record lineage."""


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _id(value: object, label: str) -> str:
    if (not isinstance(value, str) or not ID.fullmatch(value)
            or ".." in value or value != value.strip()):
        raise SourceRightsError(label + " is invalid or unsafe")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise SourceRightsError(label + " must be lowercase SHA-256")
    return value


def _validate_grant(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != GRANT_KEYS:
        raise SourceRightsError("incomplete rights or unexpected fields")
    g = dict(raw)
    _id(g["source_id"], "source_id")
    if g["rights_class"] not in RIGHTS_CLASSES or g["legal_basis"] not in BASES:
        raise SourceRightsError("unrecognized rights classification or legal basis")
    if type(g["revoked"]) is not bool:
        raise SourceRightsError("revocation must be an explicit boolean")
    lic = g["license_id"]
    if not isinstance(lic, str) or not re.fullmatch(r"[A-Za-z0-9.+-]{2,64}", lic):
        raise SourceRightsError("missing/invalid license ID")
    for key in ("terms_ref", "permission_ref", "evidence_path"):
        _id(g[key], key)
    if not g["terms_ref"].startswith(("https://", "repo://")):
        raise SourceRightsError("unverifiable terms reference")
    if not g["permission_ref"].startswith("repo://"):
        raise SourceRightsError("permission authority not repository-pinned")
    if not g["evidence_path"].startswith(("data/", "tests/fixtures/")):
        raise SourceRightsError("rights evidence outside controlled source paths")
    _sha(g["evidence_sha256"], "evidence")
    uses = g["allowed_uses"]
    if (not isinstance(uses, list) or not uses or len(uses) != len(set(uses))
            or not all(isinstance(x, str) and x in USES for x in uses)):
        raise SourceRightsError("invalid use permission vector")
    if g["rights_class"] == "public_licensed":
        if lic == "UNKNOWN" or g["legal_basis"] not in {
            "license_grant", "explicit_permission", "public_domain"
        }:
            raise SourceRightsError("public grant lacks a defensible license basis")
    elif "training" in uses or "release" in uses:
        raise SourceRightsError("unknown/private/research-only cannot authorize training/release")
    if g["rights_class"] == "unknown" and g["legal_basis"] != "unknown":
        raise SourceRightsError("unknown rights falsely assert a permission")
    g["allowed_uses"] = sorted(uses)
    return g


def build_catalog(inventory: Mapping[str, Any],
                  grants: Sequence[Mapping[str, Any]],
                  rights_bytes: Mapping[str, bytes]) -> dict[str, Any]:
    inv = verify_inventory(inventory)
    if not isinstance(grants, (list, tuple)) or not isinstance(rights_bytes, Mapping):
        raise SourceRightsError("rights evidence not explicitly supplied")
    rows = [_validate_grant(g) for g in grants]
    ids = [x["source_id"] for x in rows]
    expected = {x["source_id"] for x in inv["sources"]}
    if set(ids) != expected or len(ids) != len(set(ids)):
        raise SourceRightsError("missing, duplicate or unregistered rights authority")
    for row in rows:
        raw = rights_bytes.get(row["evidence_path"])
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != row["evidence_sha256"]:
            raise SourceRightsError("rights evidence missing or SHA-256 drifted")
    rows.sort(key=lambda x: x["source_id"])
    doc = {"schema_version": CATALOG_SCHEMA,
           "source_inventory_sha256": inv["inventory_sha256"],
           "grants": rows, "training_corpus_authorized": False}
    doc["catalog_sha256"] = fingerprint(doc)
    return doc


def verify_catalog(inventory: Mapping[str, Any],
                   catalog: Mapping[str, Any],
                   rights_bytes: Mapping[str, bytes]) -> dict[str, Any]:
    if not isinstance(catalog, Mapping) or set(catalog) != {
        "schema_version", "source_inventory_sha256", "grants",
        "training_corpus_authorized", "catalog_sha256"
    }:
        raise SourceRightsError("rights catalog envelope incomplete")
    if catalog["schema_version"] != CATALOG_SCHEMA or catalog["training_corpus_authorized"] is not False:
        raise SourceRightsError("rights catalog claims unauthorized corpus permission")
    expected = build_catalog(inventory, catalog["grants"], rights_bytes)
    if expected != dict(catalog):
        raise SourceRightsError("rights catalog identity or ordering drift")
    return expected


def materialization_receipt(inventory: Mapping[str, Any],
                            catalog: Mapping[str, Any],
                            rights_bytes: Mapping[str, bytes],
                            records: Sequence[Mapping[str, Any]],
                            purpose: str) -> dict[str, Any]:
    """Fail closed before returning a text-free, non-materializing audit receipt."""
    inv = verify_inventory(inventory)
    rights = verify_catalog(inv, catalog, rights_bytes)
    if purpose not in {"research", "training", "release"}:
        raise SourceRightsError("unsupported use")
    if not isinstance(records, (list, tuple)):
        raise SourceRightsError("records are not an explicit ordered collection")
    sources = {x["source_id"]: x for x in inv["sources"]}
    grants = {x["source_id"]: x for x in rights["grants"]}
    seen: set[str] = set()
    members: list[dict[str, str]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != RECORD_KEYS:
            raise SourceRightsError("missing record/member provenance")
        sid = _id(raw["source_id"], "source_id")
        source = sources.get(sid)
        grant = grants.get(sid)
        if source is None or grant is None or source["status"] != "accepted":
            raise SourceRightsError("unknown/candidate/rejected source in corpus")
        if grant["revoked"] or purpose not in grant["allowed_uses"]:
            raise SourceRightsError("source use not permitted or permission revoked")
        if purpose in {"training", "release"} and grant["rights_class"] != "public_licensed":
            raise SourceRightsError("private/research-only source forbidden")
        if _sha(raw["source_snapshot_sha256"], "source fingerprint") != source["provenance"]["content_sha256"]:
            raise SourceRightsError("record not bound to immutable source snapshot")
        record_id = _id(raw["record_id"], "record_id")
        member_id = _id(raw["member_id"], "member_id")
        origin = _id(raw["origin_member_ref"], "origin")
        if record_id in seen:
            raise SourceRightsError("duplicate record id")
        seen.add(record_id)
        text = raw["text"]
        if not isinstance(text, str) or not text or "\x00" in text:
            raise SourceRightsError("empty, missing or binary member content")
        try:
            raw_bytes = text.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise SourceRightsError("invalid UTF-8 payload") from exc
        members.append({
            "source_id": sid, "source_snapshot_sha256": source["provenance"]["content_sha256"],
            "record_id": record_id, "member_id": member_id,
            "origin_member_ref": origin,
            "payload_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "license_id": grant["license_id"], "rights_evidence_sha256": grant["evidence_sha256"]
        })
    members.sort(key=lambda x: (x["source_id"], x["member_id"], x["record_id"]))
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_inventory_sha256": inv["inventory_sha256"],
        "rights_catalog_sha256": rights["catalog_sha256"],
        "purpose": purpose, "record_count": len(members),
        "source_ids": sorted({x["source_id"] for x in members}),
        "members": members, "raw_payload_persisted": False,
        "training_corpus_authorized": False
    }
    receipt["receipt_sha256"] = fingerprint(receipt)
    return receipt


def verify_receipt(inventory: Mapping[str, Any], catalog: Mapping[str, Any],
                   rights_bytes: Mapping[str, bytes], records: Sequence[Mapping[str, Any]],
                   purpose: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    expected = materialization_receipt(inventory, catalog, rights_bytes, records, purpose)
    if not isinstance(receipt, Mapping) or dict(receipt) != expected:
        raise SourceRightsError("record/member provenance or receipt checksum drift")
    return expected


def removal_plan(receipt: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    sid = _id(source_id, "removal source")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version", "source_inventory_sha256", "rights_catalog_sha256", "purpose",
        "record_count", "source_ids", "members", "raw_payload_persisted",
        "training_corpus_authorized", "receipt_sha256",
    } or receipt["schema_version"] != RECEIPT_SCHEMA:
        raise SourceRightsError("malformed removal receipt")
    d = dict(receipt)
    supplied_hash = d.pop("receipt_sha256")
    if fingerprint(d) != supplied_hash:
        raise SourceRightsError("removal source receipt tampered")
    if sid not in receipt["source_ids"]:
        raise SourceRightsError("requested source not in member manifest")
    affected = [{"record_id": x["record_id"], "member_id": x["member_id"]}
                for x in receipt["members"] if x["source_id"] == sid]
    result = {
        "source_id": sid, "receipt_sha256": supplied_hash,
        "affected": affected, "affected_records": len(affected),
        "downstream_rebuild_required": True
    }
    result["removal_plan_sha256"] = fingerprint(result)
    return result
