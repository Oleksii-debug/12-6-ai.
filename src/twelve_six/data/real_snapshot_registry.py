"""DATA-229 immutable registry for admitted real source snapshots."""
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
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RealSnapshotRegistryError(ValueError):
    """Fail-closed registry contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise RealSnapshotRegistryError(f"{field} must be lowercase SHA-256")
    return value


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RealSnapshotRegistryError(f"{path}: JSON object required")
    return value


def _identity(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _self_identity(value: Mapping[str, Any], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return _identity(payload)


def _rights(rights: Mapping[str, Any], purpose: str, eval_policy: str) -> dict[str, Any]:
    raw_refs = rights.get("evidence_refs")
    uses = rights.get("uses")
    if not isinstance(raw_refs, list) or not raw_refs or not isinstance(uses, Mapping):
        raise RealSnapshotRegistryError("source rights evidence/uses missing")
    refs = sorted(
        (
            {
                "captured_at": ref.get("captured_at"),
                "evidence_id": ref.get("evidence_id"),
                "evidence_kind": ref.get("evidence_kind"),
                "sha256": _sha(ref.get("sha256"), "rights evidence sha256"),
                "source_id": ref.get("source_id"),
                "source_version": ref.get("source_version"),
                "uri": ref.get("uri"),
            }
            for ref in raw_refs
        ),
        key=lambda ref: str(ref["evidence_id"]),
    )
    evidence_id = _identity({"evidence_refs": refs})
    status = NOT_ADMITTED if purpose == "evaluation" else str(uses.get(purpose, "UNKNOWN"))
    authority = eval_policy if purpose == "evaluation" else rights.get("policy_ref")
    core = {
        "authority": authority,
        "evidence_identity_sha256": evidence_id,
        "license_id": rights.get("license_id"),
        "purpose": purpose,
        "status": status,
        "terms_url": rights.get("terms_url"),
    }
    return {**core, "decision_identity_sha256": _identity(core), "evidence_refs": refs}


def _artifact_files(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RealSnapshotRegistryError("DATA-213 artifact files missing")
    return {str(row.get("path")): row for row in rows if isinstance(row, Mapping)}


def _one_source(
    plan: Mapping[str, Any],
    ext: Mapping[str, Any],
    report: Mapping[str, Any],
    authority: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    eval_policy: str,
    prefix: str,
) -> dict[str, Any]:
    sid, version = str(plan["promoted_source_id"]), str(plan["source_version"])
    if (ext.get("source_id"), ext.get("source_version")) != (sid, version):
        raise RealSnapshotRegistryError(f"{sid}: DATA-24 identity mismatch")
    if (report.get("promoted_source_id"), report.get("source_version")) != (sid, version):
        raise RealSnapshotRegistryError(f"{sid}: DATA-213 identity mismatch")
    raw_sha = _sha(plan.get("raw_sha256"), f"{sid} raw")
    norm_sha = _sha(plan.get("normalized_sha256"), f"{sid} normalized")
    snapshot, rights = ext.get("snapshot"), ext.get("rights")
    if not isinstance(snapshot, Mapping) or not isinstance(rights, Mapping):
        raise RealSnapshotRegistryError(f"{sid}: snapshot/rights missing")
    checks = (
        snapshot.get("sha256") == raw_sha,
        report.get("raw_sha256") == raw_sha,
        report.get("normalized_sha256") == norm_sha,
        report.get("source_manifest_sha256") == plan.get("source_manifest_sha256"),
        report.get("repeat_acquisition_same_raw_identity") is True,
        report.get("repeat_extraction_same_normalized_identity") is True,
    )
    if not all(checks):
        raise RealSnapshotRegistryError(f"{sid}: admitted identity drift")
    raw_path = str(report.get("snapshot_path"))
    norm_path = "data181-evidence/" + str(report.get("normalized_path"))
    raw_art, norm_art = artifacts.get(raw_path), artifacts.get(norm_path)
    if not isinstance(raw_art, Mapping) or raw_art.get("sha256") != raw_sha:
        raise RealSnapshotRegistryError(f"{sid}: raw artifact identity missing")
    if not isinstance(norm_art, Mapping):
        raise RealSnapshotRegistryError(f"{sid}: normalized artifact identity missing")
    return {
        "registry_source_id": prefix + sid,
        "origin_class": "EXTERNAL_REAL",
        "source_family": {
            "family_id": plan.get("parent_source_id"),
            "family_identity_sha256": _sha(
                plan.get("parent_source_identity_sha256"), f"{sid} family"
            ),
        },
        "modality": "text",
        "language": plan.get("language"),
        "raw_identity": {
            "provider": ext.get("provider"),
            "record_id": plan.get("record_id"),
            "source_id": sid,
            "source_kind": ext.get("source_kind"),
            "source_manifest_sha256": _sha(plan.get("source_manifest_sha256"), f"{sid} manifest"),
            "source_url": ext.get("source_url"),
            "source_version": version,
            "raw_sha256": raw_sha,
            "raw_size_bytes": plan.get("raw_bytes"),
        },
        "retrieval_identity": {
            "acquisition_url": plan.get("acquisition_url"),
            "retrieval_method": snapshot.get("retrieval_method"),
            "retrieved_at": snapshot.get("retrieved_at"),
            "snapshot_uri": snapshot.get("uri"),
            "upstream_version": snapshot.get("upstream_version"),
        },
        "normalization": {
            "policy_id": "DATA181_EXTRACT_FIRST_50000_CHARS_STRICT_NORMALIZE_UTF8_V1",
            "adapter": plan.get("adapter"),
            "decoded_encoding": report.get("decoded_encoding"),
            "max_extracted_characters": 50000,
            "pre_normalization_extracted_sha256": None,
            "pre_normalization_extracted_hash_status": "NOT_RETAINED_BY_DATA213",
            "extracted_normalized_sha256": norm_sha,
            "extracted_normalized_utf8_bytes": plan.get("normalized_utf8_bytes"),
            "normalized_artifact_file_sha256": _sha(norm_art.get("sha256"), f"{sid} norm artifact"),
            "normalized_artifact_size_bytes": norm_art.get("size_bytes"),
        },
        "rights": {purpose: _rights(rights, purpose, eval_policy) for purpose in PURPOSES},
        "d03": {
            "producer_worker": "DATA-213",
            "producer_source_sha": authority["source_sha"],
            "promotion_report_identity_sha256": authority["promotion_report_identity_sha256"],
            "dataset_identity_sha256": authority["d03_dataset_identity_sha256"],
            "source_registry_sha256": authority["d03_source_registry_sha256"],
            "contamination_registry_sha256": authority["d03_contamination_registry_sha256"],
            "admitted_chunk_count": report.get("admitted_chunk_count"),
            "admitted_chunk_identity_sha256": _sha(
                report.get("admitted_chunk_identity_sha256"), f"{sid} chunk identity"
            ),
        },
        "decontamination": {
            "status": "D03_PURPOSE_AND_DUPLICATE_GATES_PASS_NO_UNIVERSAL_BENCHMARK_CLEAN_CLAIM",
            "universal_benchmark_clean": False,
        },
    }


def build_real_snapshot_registry(
    *,
    inputs_path: str | Path,
    data213_plan_path: str | Path,
    data24_registry_path: str | Path,
    data213_report_path: str | Path,
    data213_artifact_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build deterministically from terminal source evidence, never live clock/order."""
    inputs = _load(inputs_path)
    plan = _load(data213_plan_path)
    ext_reg = _load(data24_registry_path)
    report = _load(data213_report_path)
    manifest = _load(data213_artifact_manifest_path)
    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("local_free_only") is not True:
        raise RealSnapshotRegistryError("invalid DATA-229 authority manifest")
    if plan.get("schema_version") != "12-6.data181-real-snapshot-promotion.v1":
        raise RealSnapshotRegistryError("unsupported DATA-213 plan")
    if ext_reg.get("schema_version") != "12-6.external-source-registry.v2":
        raise RealSnapshotRegistryError("unsupported DATA-24 registry")
    if report.get("schema_version") != "12-6.data181-real-snapshot-promotion-report.v1" or report.get("status") != "PASS":
        raise RealSnapshotRegistryError("DATA-213 terminal PASS required")
    if manifest.get("schema_version") != "12-6.data181-artifact-manifest.v1":
        raise RealSnapshotRegistryError("unsupported DATA-213 artifact manifest")
    authorities = inputs.get("authorities")
    if not isinstance(authorities, Mapping):
        raise RealSnapshotRegistryError("authority map missing")
    authority = authorities.get("DATA-213")
    if not isinstance(authority, Mapping) or authority.get("status") != "TERMINAL_SUCCESS":
        raise RealSnapshotRegistryError("DATA-213 terminal-success authority required")
    if _self_identity(report, "report_sha256") != report.get("report_sha256"):
        raise RealSnapshotRegistryError("DATA-213 report self-identity drift")
    if _self_identity(manifest, "manifest_sha256") != manifest.get("manifest_sha256"):
        raise RealSnapshotRegistryError("DATA-213 artifact manifest self-identity drift")
    if manifest.get("source_sha") != authority.get("source_sha"):
        raise RealSnapshotRegistryError("DATA-213 source SHA drift")
    if manifest.get("manifest_sha256") != authority.get("artifact_manifest_identity_sha256"):
        raise RealSnapshotRegistryError("DATA-213 artifact manifest authority drift")
    if report.get("report_sha256") != authority.get("promotion_report_identity_sha256"):
        raise RealSnapshotRegistryError("DATA-213 report authority drift")
    reg_id = ext_reg.get("registry_identity_sha256")
    if reg_id != plan.get("canonical_registry_identity_sha256") or reg_id != report.get("canonical_registry_identity_sha256"):
        raise RealSnapshotRegistryError("canonical DATA-24 registry identity drift")
    small = report.get("small_corpus")
    if not isinstance(small, Mapping):
        raise RealSnapshotRegistryError("DATA-213 D03 evidence missing")
    for left, right in (
        ("dataset_identity_sha256", "d03_dataset_identity_sha256"),
        ("source_registry_sha256", "d03_source_registry_sha256"),
        ("contamination_registry_sha256", "d03_contamination_registry_sha256"),
    ):
        if small.get(left) != authority.get(right):
            raise RealSnapshotRegistryError(f"DATA-213 {left} drift")
    artifacts = _artifact_files(manifest)
    for artifact_path, local_path in {
        "configs/data/data181_real_snapshot_promotion_v1.json": Path(data213_plan_path),
        "data/external/external_sources.json": Path(data24_registry_path),
    }.items():
        row = artifacts.get(artifact_path)
        if not isinstance(row, Mapping) or sha256_bytes(local_path.read_bytes()) != row.get("sha256"):
            raise RealSnapshotRegistryError(f"DATA-213 consumed input drift: {artifact_path}")
    p_list, e_list, r_list = plan.get("objects"), ext_reg.get("sources"), report.get("objects")
    if not all(isinstance(items, list) for items in (p_list, e_list, r_list)):
        raise RealSnapshotRegistryError("admitted source collections malformed")
    p = {(str(x["promoted_source_id"]), str(x["source_version"])): x for x in p_list}
    e = {(str(x["source_id"]), str(x["source_version"])): x for x in e_list}
    r = {(str(x["promoted_source_id"]), str(x["source_version"])): x for x in r_list}
    if set(p) != set(e) or set(p) != set(r):
        raise RealSnapshotRegistryError("DATA-213 admitted source sets disagree")
    namespaces = inputs.get("origin_namespaces")
    if not isinstance(namespaces, Mapping) or not namespaces.get("EXTERNAL_REAL"):
        raise RealSnapshotRegistryError("origin namespace contract missing")
    sources = [
        _one_source(p[k], e[k], r[k], authority, artifacts, str(inputs["evaluation_rights_policy"]), str(namespaces["EXTERNAL_REAL"]))
        for k in sorted(p)
    ]
    out = {
        "schema_version": REGISTRY_SCHEMA,
        "local_free_only": True,
        "cutoff_utc": inputs.get("cutoff_utc"),
        "authority_status": copy.deepcopy(authorities),
        "origin_namespaces": copy.deepcopy(namespaces),
        "canonical_data24_registry_identity_sha256": reg_id,
        "source_count": len(sources),
        "sources": sources,
        "claim_boundary": {
            "code_source_count": sum(x["modality"] == "code" for x in sources),
            "evaluation_authorized_source_count": sum(x["rights"]["evaluation"]["status"] == ALLOWED for x in sources),
            "external_real_source_count": sum(x["origin_class"] == "EXTERNAL_REAL" for x in sources),
            "project_authored_source_count": sum(x["origin_class"] == "PROJECT_AUTHORED" for x in sources),
            "missing_terminal_workers": [n for n in ("DATA-227", "DATA-228") if authorities.get(n, {}).get("status") != "TERMINAL_SUCCESS"],
            "universal_benchmark_clean": False,
        },
    }
    out["registry_identity_sha256"] = registry_identity(out)
    validate_real_snapshot_registry(out)
    return out


def registry_identity(registry: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(registry))
    payload.pop("registry_identity_sha256", None)
    return _identity(payload)


def validate_real_snapshot_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != REGISTRY_SCHEMA or registry.get("registry_identity_sha256") != registry_identity(registry):
        raise RealSnapshotRegistryError("registry schema/self-identity mismatch")
    sources, namespaces = registry.get("sources"), registry.get("origin_namespaces")
    if not isinstance(sources, list) or not isinstance(namespaces, Mapping) or registry.get("source_count") != len(sources):
        raise RealSnapshotRegistryError("registry inventory malformed")
    ids, raw_keys = set(), set()
    if [str(x.get("registry_source_id")) for x in sources] != sorted(str(x.get("registry_source_id")) for x in sources):
        raise RealSnapshotRegistryError("sources must be sorted")
    for source in sources:
        rid, origin = str(source.get("registry_source_id", "")), str(source.get("origin_class", ""))
        prefix = namespaces.get(origin)
        if not isinstance(prefix, str) or not prefix or not rid.startswith(prefix):
            raise RealSnapshotRegistryError(f"{rid}: origin class does not match namespace")
        raw = source.get("raw_identity")
        rights = source.get("rights")
        if rid in ids or not isinstance(raw, Mapping):
            raise RealSnapshotRegistryError(f"{rid}: duplicate/missing raw identity")
        ids.add(rid)
        raw_key = (str(raw.get("source_id")), str(raw.get("source_version")))
        if raw_key in raw_keys:
            raise RealSnapshotRegistryError(f"{rid}: raw source identity appears under multiple origins")
        raw_keys.add(raw_key)
        _sha(raw.get("raw_sha256"), f"{rid} raw")
        if not isinstance(rights, Mapping) or set(rights) != set(PURPOSES):
            raise RealSnapshotRegistryError(f"{rid}: purpose rights incomplete")
        for purpose in PURPOSES:
            decision = rights[purpose]
            core = {k: decision.get(k) for k in ("authority", "evidence_identity_sha256", "license_id", "purpose", "status", "terms_url")}
            if decision.get("decision_identity_sha256") != _identity(core):
                raise RealSnapshotRegistryError(f"{rid}: {purpose} decision identity mismatch")


def serialize_registry(registry: Mapping[str, Any]) -> bytes:
    validate_real_snapshot_registry(registry)
    return canonical_json_bytes(registry)


def load_real_snapshot_registry(path: str | Path) -> dict[str, Any]:
    registry = _load(path)
    validate_real_snapshot_registry(registry)
    return registry


def select_sources(
    registry: Mapping[str, Any], *, purpose: str, languages: Iterable[str] | None = None,
    modalities: Iterable[str] | None = None, origins: Iterable[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if purpose not in PURPOSES:
        raise RealSnapshotRegistryError(f"unsupported purpose: {purpose}")
    validate_real_snapshot_registry(registry)
    langs = None if languages is None else frozenset(languages)
    mods = None if modalities is None else frozenset(modalities)
    orgs = None if origins is None else frozenset(origins)
    return tuple(
        copy.deepcopy(source) for source in registry["sources"]
        if source["rights"][purpose]["status"] == ALLOWED
        and (langs is None or source["language"] in langs)
        and (mods is None or source["modality"] in mods)
        and (orgs is None or source["origin_class"] in orgs)
    )


def sources_for_corpus(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="model_training", **filters)


def sources_for_holdout(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="evaluation", **filters)


def sources_for_redistribution(registry: Mapping[str, Any], **filters: Any) -> tuple[Mapping[str, Any], ...]:
    return select_sources(registry, purpose="redistribution", **filters)
