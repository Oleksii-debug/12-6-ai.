"""Immutable DATA-229 registry for admitted real snapshots."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "12-6.real-snapshot-registry.v1"
INPUT_SCHEMA = "12-6.data229-real-snapshot-registry-inputs.v1"
PURPOSES = ("evaluation", "model_training", "redistribution")
ALLOWED = "ALLOWED"
NOT_ADMITTED = "NOT_SEPARATELY_ADMITTED"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class RealSnapshotRegistryError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise RealSnapshotRegistryError(f"{field} must be lowercase SHA-256")
    return value


def _identity(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RealSnapshotRegistryError(f"{path}: JSON object required")
    return value


def _self_identity(value: Mapping[str, Any], field: str) -> str:
    core = copy.deepcopy(dict(value))
    core.pop(field, None)
    return _identity(core)


def _rights(source: Mapping[str, Any], eval_policy: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rights = source.get("rights")
    if not isinstance(rights, Mapping) or not isinstance(rights.get("uses"), Mapping):
        raise RealSnapshotRegistryError("rights/uses missing")
    refs = rights.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise RealSnapshotRegistryError("rights evidence missing")
    exact_refs = sorted(
        (
            {
                "captured_at": ref.get("captured_at"),
                "evidence_id": ref.get("evidence_id"),
                "evidence_kind": ref.get("evidence_kind"),
                "sha256": _digest(ref.get("sha256"), "rights evidence"),
                "source_id": ref.get("source_id"),
                "source_version": ref.get("source_version"),
                "uri": ref.get("uri"),
            }
            for ref in refs
        ),
        key=lambda ref: str(ref["evidence_id"]),
    )
    evidence_set = _identity({"evidence_refs": exact_refs})
    profile = {
        "license_id": rights.get("license_id"),
        "terms_url": rights.get("terms_url"),
        "evidence_ids": [ref["evidence_id"] for ref in exact_refs],
        "evidence_set_identity_sha256": evidence_set,
    }
    decisions: dict[str, Any] = {}
    for purpose in PURPOSES:
        status = NOT_ADMITTED if purpose == "evaluation" else str(rights["uses"].get(purpose, "UNKNOWN"))
        authority = eval_policy if purpose == "evaluation" else rights.get("policy_ref")
        purpose_evidence = _identity(
            {"evidence_set_identity_sha256": evidence_set, "purpose": purpose, "status": status}
        )
        core = {
            "authority": authority,
            "evidence_set_identity_sha256": evidence_set,
            "license_id": profile["license_id"],
            "purpose": purpose,
            "status": status,
            "terms_url": profile["terms_url"],
        }
        decisions[purpose] = {
            "status": status,
            "authority": authority,
            "purpose_evidence_identity_sha256": purpose_evidence,
            "decision_identity_sha256": _identity(core),
        }
    return profile, decisions


def build_real_snapshot_registry(
    *,
    inputs_path: str | Path,
    data213_plan_path: str | Path,
    data24_registry_path: str | Path,
    data213_report_path: str | Path,
    data213_artifact_manifest_path: str | Path,
) -> dict[str, Any]:
    inputs, plan, ext = _load(inputs_path), _load(data213_plan_path), _load(data24_registry_path)
    report, artifact = _load(data213_report_path), _load(data213_artifact_manifest_path)
    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("local_free_only") is not True:
        raise RealSnapshotRegistryError("invalid DATA-229 authority manifest")
    if plan.get("schema_version") != "12-6.data181-real-snapshot-promotion.v1" or ext.get("schema_version") != "12-6.external-source-registry.v2":
        raise RealSnapshotRegistryError("unsupported DATA-213/DATA-24 authority")
    if report.get("schema_version") != "12-6.data181-real-snapshot-promotion-report.v1" or report.get("status") != "PASS":
        raise RealSnapshotRegistryError("DATA-213 terminal PASS required")
    authorities = inputs.get("authorities")
    authority = authorities.get("DATA-213") if isinstance(authorities, Mapping) else None
    if not isinstance(authority, Mapping) or authority.get("status") != "TERMINAL_SUCCESS":
        raise RealSnapshotRegistryError("DATA-213 terminal-success authority required")
    if _self_identity(report, "report_sha256") != report.get("report_sha256") or report.get("report_sha256") != authority.get("promotion_report_identity_sha256"):
        raise RealSnapshotRegistryError("DATA-213 report identity drift")
    if artifact.get("schema_version") != "12-6.data181-artifact-manifest.v1" or _self_identity(artifact, "manifest_sha256") != artifact.get("manifest_sha256"):
        raise RealSnapshotRegistryError("DATA-213 artifact manifest identity drift")
    if artifact.get("source_sha") != authority.get("source_sha") or artifact.get("manifest_sha256") != authority.get("artifact_manifest_identity_sha256"):
        raise RealSnapshotRegistryError("DATA-213 artifact authority drift")
    reg_id = ext.get("registry_identity_sha256")
    if reg_id != plan.get("canonical_registry_identity_sha256") or reg_id != report.get("canonical_registry_identity_sha256"):
        raise RealSnapshotRegistryError("DATA-24 registry identity drift")
    small = report.get("small_corpus")
    if not isinstance(small, Mapping):
        raise RealSnapshotRegistryError("D03 evidence missing")
    for left, right in (
        ("dataset_identity_sha256", "d03_dataset_identity_sha256"),
        ("source_registry_sha256", "d03_source_registry_sha256"),
        ("contamination_registry_sha256", "d03_contamination_registry_sha256"),
    ):
        if small.get(left) != authority.get(right):
            raise RealSnapshotRegistryError(f"D03 {left} drift")
    files = {
        str(row.get("path")): row
        for row in artifact.get("files", [])
        if isinstance(row, Mapping)
    }
    for apath, local in (
        ("configs/data/data181_real_snapshot_promotion_v1.json", Path(data213_plan_path)),
        ("data/external/external_sources.json", Path(data24_registry_path)),
    ):
        if files.get(apath, {}).get("sha256") != sha256_bytes(local.read_bytes()):
            raise RealSnapshotRegistryError(f"DATA-213 consumed input drift: {apath}")
    p = {(str(x["promoted_source_id"]), str(x["source_version"])): x for x in plan.get("objects", [])}
    e = {(str(x["source_id"]), str(x["source_version"])): x for x in ext.get("sources", [])}
    r = {(str(x["promoted_source_id"]), str(x["source_version"])): x for x in report.get("objects", [])}
    if set(p) != set(e) or set(p) != set(r):
        raise RealSnapshotRegistryError("admitted source sets disagree")
    namespaces = inputs.get("origin_namespaces")
    if not isinstance(namespaces, Mapping) or not namespaces.get("EXTERNAL_REAL") or not namespaces.get("PROJECT_AUTHORED"):
        raise RealSnapshotRegistryError("origin namespace contract missing")
    eval_policy = str(inputs.get("evaluation_rights_policy"))
    sources = []
    for key in sorted(p):
        pp, ee, rr = p[key], e[key], r[key]
        sid, version = key
        raw = _digest(pp.get("raw_sha256"), f"{sid} raw")
        norm = _digest(pp.get("normalized_sha256"), f"{sid} normalized")
        snap = ee.get("snapshot")
        if not isinstance(snap, Mapping) or snap.get("sha256") != raw or rr.get("raw_sha256") != raw or rr.get("normalized_sha256") != norm:
            raise RealSnapshotRegistryError(f"{sid}: source identity drift")
        if rr.get("repeat_acquisition_same_raw_identity") is not True or rr.get("repeat_extraction_same_normalized_identity") is not True:
            raise RealSnapshotRegistryError(f"{sid}: repeat identity proof missing")
        raw_art = files.get(str(rr.get("snapshot_path")))
        norm_art = files.get("data181-evidence/" + str(rr.get("normalized_path")))
        if raw_art is None or raw_art.get("sha256") != raw or norm_art is None:
            raise RealSnapshotRegistryError(f"{sid}: artifact identity missing")
        profile, purpose_rights = _rights(ee, eval_policy)
        sources.append(
            {
                "registry_source_id": str(namespaces["EXTERNAL_REAL"]) + sid,
                "origin_class": "EXTERNAL_REAL",
                "source_family": {
                    "family_id": pp.get("parent_source_id"),
                    "family_identity_sha256": _digest(pp.get("parent_source_identity_sha256"), f"{sid} family"),
                },
                "modality": "text",
                "language": pp.get("language"),
                "raw_identity": {
                    "source_id": sid,
                    "source_version": version,
                    "raw_sha256": raw,
                    "raw_size_bytes": pp.get("raw_bytes"),
                    "source_manifest_sha256": _digest(pp.get("source_manifest_sha256"), f"{sid} manifest"),
                    "source_url": ee.get("source_url"),
                    "record_id": pp.get("record_id"),
                },
                "retrieval_identity": {
                    "acquisition_url": pp.get("acquisition_url"),
                    "snapshot_uri": snap.get("uri"),
                    "upstream_version": snap.get("upstream_version"),
                    "retrieval_method": snap.get("retrieval_method"),
                    "retrieved_at": snap.get("retrieved_at"),
                },
                "normalization": {
                    "policy_id": "DATA181_EXTRACT_FIRST_50000_CHARS_STRICT_NORMALIZE_UTF8_V1",
                    "adapter": pp.get("adapter"),
                    "decoded_encoding": rr.get("decoded_encoding"),
                    "max_extracted_characters": 50000,
                    "pre_normalization_extracted_sha256": None,
                    "pre_normalization_extracted_hash_status": "NOT_RETAINED_BY_DATA213",
                    "extracted_normalized_sha256": norm,
                    "extracted_normalized_utf8_bytes": pp.get("normalized_utf8_bytes"),
                    "normalized_artifact_file_sha256": _digest(norm_art.get("sha256"), f"{sid} normalized artifact"),
                    "normalized_artifact_size_bytes": norm_art.get("size_bytes"),
                },
                "rights_profile": profile,
                "rights": purpose_rights,
                "d03": {
                    "dataset_identity_sha256": authority["d03_dataset_identity_sha256"],
                    "admitted_chunk_count": rr.get("admitted_chunk_count"),
                    "admitted_chunk_identity_sha256": _digest(rr.get("admitted_chunk_identity_sha256"), f"{sid} chunk identity"),
                },
                "decontamination": {
                    "status": "D03_PURPOSE_AND_DUPLICATE_GATES_PASS_NO_UNIVERSAL_BENCHMARK_CLEAN_CLAIM",
                    "universal_benchmark_clean": False,
                },
            }
        )
    out = {
        "schema_version": REGISTRY_SCHEMA,
        "local_free_only": True,
        "cutoff_utc": inputs.get("cutoff_utc"),
        "authority_status": copy.deepcopy(authorities),
        "origin_namespaces": copy.deepcopy(namespaces),
        "canonical_data24_registry_identity_sha256": reg_id,
        "d03_authority": {
            "producer_worker": "DATA-213",
            "producer_source_sha": authority["source_sha"],
            "promotion_report_identity_sha256": authority["promotion_report_identity_sha256"],
            "dataset_identity_sha256": authority["d03_dataset_identity_sha256"],
            "source_registry_sha256": authority["d03_source_registry_sha256"],
            "contamination_registry_sha256": authority["d03_contamination_registry_sha256"],
        },
        "source_count": len(sources),
        "sources": sources,
        "claim_boundary": {
            "code_source_count": sum(x["modality"] == "code" for x in sources),
            "evaluation_authorized_source_count": sum(x["rights"]["evaluation"]["status"] == ALLOWED for x in sources),
            "external_real_source_count": sum(x["origin_class"] == "EXTERNAL_REAL" for x in sources),
            "project_authored_source_count": sum(x["origin_class"] == "PROJECT_AUTHORED" for x in sources),
            "missing_terminal_workers": [
                n
                for n in ("DATA-227", "DATA-228")
                if authorities.get(n, {}).get("status") != "TERMINAL_SUCCESS"
            ],
            "universal_benchmark_clean": False,
        },
    }
    out["registry_identity_sha256"] = registry_identity(out)
    validate_real_snapshot_registry(out)
    return out


def registry_identity(registry: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(registry))
    core.pop("registry_identity_sha256", None)
    return _identity(core)


def validate_real_snapshot_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != REGISTRY_SCHEMA or registry.get("registry_identity_sha256") != registry_identity(registry):
        raise RealSnapshotRegistryError("registry schema/self-identity mismatch")
    sources, namespaces = registry.get("sources"), registry.get("origin_namespaces")
    if not isinstance(sources, list) or not isinstance(namespaces, Mapping) or registry.get("source_count") != len(sources):
        raise RealSnapshotRegistryError("registry inventory malformed")
    ids: set[str] = set()
    raw_keys: set[tuple[str, str]] = set()
    if [str(x.get("registry_source_id")) for x in sources] != sorted(str(x.get("registry_source_id")) for x in sources):
        raise RealSnapshotRegistryError("sources must be sorted")
    for source in sources:
        rid = str(source.get("registry_source_id", ""))
        origin = str(source.get("origin_class", ""))
        prefix = namespaces.get(origin)
        if not isinstance(prefix, str) or not rid.startswith(prefix):
            raise RealSnapshotRegistryError(f"{rid}: origin class does not match namespace")
        raw, profile, rights = source.get("raw_identity"), source.get("rights_profile"), source.get("rights")
        if rid in ids or not isinstance(raw, Mapping):
            raise RealSnapshotRegistryError(f"{rid}: duplicate/missing raw identity")
        ids.add(rid)
        raw_key = (str(raw.get("source_id")), str(raw.get("source_version")))
        if raw_key in raw_keys:
            raise RealSnapshotRegistryError(f"{rid}: raw source identity appears under multiple origins")
        raw_keys.add(raw_key)
        _digest(raw.get("raw_sha256"), f"{rid} raw")
        if not isinstance(profile, Mapping) or not isinstance(rights, Mapping) or set(rights) != set(PURPOSES):
            raise RealSnapshotRegistryError(f"{rid}: purpose rights incomplete")
        evidence = profile.get("evidence_set_identity_sha256")
        for purpose in PURPOSES:
            decision = rights[purpose]
            status, authority = decision.get("status"), decision.get("authority")
            pe = _identity({"evidence_set_identity_sha256": evidence, "purpose": purpose, "status": status})
            core = {
                "authority": authority,
                "evidence_set_identity_sha256": evidence,
                "license_id": profile.get("license_id"),
                "purpose": purpose,
                "status": status,
                "terms_url": profile.get("terms_url"),
            }
            if decision.get("purpose_evidence_identity_sha256") != pe or decision.get("decision_identity_sha256") != _identity(core):
                raise RealSnapshotRegistryError(f"{rid}: {purpose} rights identity mismatch")


def verify_source_payload(source: Mapping[str, Any], payload: bytes) -> None:
    raw = source.get("raw_identity")
    if not isinstance(raw, Mapping):
        raise RealSnapshotRegistryError("source raw identity missing")
    size = raw.get("raw_size_bytes")
    digest = _digest(raw.get("raw_sha256"), "raw payload")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RealSnapshotRegistryError("raw_size_bytes invalid")
    if len(payload) != size or sha256_bytes(payload) != digest:
        raise RealSnapshotRegistryError("materialized source bytes do not match registry identity")


def serialize_registry(registry: Mapping[str, Any]) -> bytes:
    validate_real_snapshot_registry(registry)
    return canonical_json_bytes(registry)


def load_real_snapshot_registry(path: str | Path) -> dict[str, Any]:
    value = _load(path)
    validate_real_snapshot_registry(value)
    return value


def select_sources(
    registry: Mapping[str, Any],
    *,
    purpose: str,
    languages: Iterable[str] | None = None,
    modalities: Iterable[str] | None = None,
    origins: Iterable[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if purpose not in PURPOSES:
        raise RealSnapshotRegistryError(f"unsupported purpose: {purpose}")
    validate_real_snapshot_registry(registry)
    langs = None if languages is None else frozenset(languages)
    mods = None if modalities is None else frozenset(modalities)
    orgs = None if origins is None else frozenset(origins)
    return tuple(
        copy.deepcopy(x)
        for x in registry["sources"]
        if x["rights"][purpose]["status"] == ALLOWED
        and (langs is None or x["language"] in langs)
        and (mods is None or x["modality"] in mods)
        and (orgs is None or x["origin_class"] in orgs)
    )


def sources_for_corpus(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="model_training", **filters)


def sources_for_holdout(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="evaluation", **filters)


def sources_for_redistribution(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="redistribution", **filters)
