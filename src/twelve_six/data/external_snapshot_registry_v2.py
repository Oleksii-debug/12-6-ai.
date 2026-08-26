"""DATA-287 deterministic successor registry over DATA-229 and terminal Wave-1 admissions."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "12-6.external-snapshot-registry.v2"
INPUT_SCHEMA = "12-6.data287-external-snapshot-registry-inputs.v2"
PURPOSES = ("model_training", "evaluation", "redistribution")
ALLOWED = "ALLOWED"
NOT_ADMITTED = "NOT_SEPARATELY_ADMITTED"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class ExternalSnapshotRegistryV2Error(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExternalSnapshotRegistryV2Error(f"{path}: JSON object required")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ExternalSnapshotRegistryV2Error(f"{field}: lowercase SHA-256 required")
    return value


def _assert_authorities(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    authorities = inputs.get("authorities")
    if not isinstance(authorities, Mapping):
        raise ExternalSnapshotRegistryV2Error("authority map missing")
    for worker in ("DATA-213", "DATA-227"):
        if authorities.get(worker, {}).get("status") != "TERMINAL_SUCCESS":
            raise ExternalSnapshotRegistryV2Error(f"{worker}: terminal success required")
    failed = authorities.get("DATA-228")
    if not isinstance(failed, Mapping) or failed.get("status") != "TERMINAL_FAILURE":
        raise ExternalSnapshotRegistryV2Error("DATA-228 terminal failure must remain explicit")
    if failed.get("consumption") != "EXCLUDED":
        raise ExternalSnapshotRegistryV2Error("failed terminal source producer cannot be consumed")
    data227 = authorities["DATA-227"]
    if data227.get("blocked_prior_sources_reinterpreted") is not False:
        raise ExternalSnapshotRegistryV2Error("prior blocked code sources cannot be reinterpreted")
    if data227.get("source_family_count") != 2 or data227.get("admitted_object_count") != 2:
        raise ExternalSnapshotRegistryV2Error("DATA-227 terminal inventory drift")
    if data227.get("admitted_raw_bytes") != 9703:
        raise ExternalSnapshotRegistryV2Error("DATA-227 terminal byte count drift")
    for worker in ("DATA-213", "DATA-227"):
        _sha(authorities[worker].get("artifact_zip_sha256"), f"{worker} artifact")
    return authorities


def _base_text_index(base: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if base.get("schema_version") != "12-6.real-snapshot-registry.v1":
        raise ExternalSnapshotRegistryV2Error("unsupported DATA-229 base registry schema")
    sources = base.get("sources")
    if not isinstance(sources, list):
        raise ExternalSnapshotRegistryV2Error("DATA-229 base source list missing")
    index: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ExternalSnapshotRegistryV2Error("DATA-229 base source malformed")
        raw = source.get("raw_identity")
        if not isinstance(raw, Mapping):
            raise ExternalSnapshotRegistryV2Error("DATA-229 base raw identity missing")
        sid = str(raw.get("source_id"))
        if sid in index:
            raise ExternalSnapshotRegistryV2Error(f"DATA-229 duplicate source id: {sid}")
        index[sid] = source
    return index


def _verify_text_source(spec: Mapping[str, Any], base_source: Mapping[str, Any]) -> None:
    raw = base_source.get("raw_identity")
    norm = base_source.get("normalization")
    family = base_source.get("source_family")
    rights = base_source.get("rights")
    if not all(isinstance(value, Mapping) for value in (raw, norm, family, rights)):
        raise ExternalSnapshotRegistryV2Error(f"{spec['source_id']}: DATA-229 fields missing")
    profile = base_source.get("rights_profile")
    if not isinstance(profile, Mapping):
        raise ExternalSnapshotRegistryV2Error(
            f"{spec['source_id']}: DATA-229 rights profile missing"
        )
    expected = {
        "source_version": raw.get("source_version"),
        "raw_sha256": raw.get("raw_sha256"),
        "raw_bytes": raw.get("raw_size_bytes"),
        "normalized_sha256": norm.get("extracted_normalized_sha256"),
        "normalized_bytes": norm.get("extracted_normalized_utf8_bytes"),
        "normalization_policy": norm.get("policy_id"),
        "independent_family_id": family.get("family_id"),
        "family_identity_sha256": family.get("family_identity_sha256"),
        "language": base_source.get("language"),
        "modality": base_source.get("modality"),
        "license_id": profile.get("license_id"),
        "source_manifest_sha256": raw.get("source_manifest_sha256"),
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise ExternalSnapshotRegistryV2Error(
                f"{spec['source_id']}: DATA-229 {field} drift"
            )
    for purpose in PURPOSES:
        decision = rights.get(purpose)
        if not isinstance(decision, Mapping) or decision.get("status") != spec["rights"][purpose]:
            raise ExternalSnapshotRegistryV2Error(
                f"{spec['source_id']}: DATA-229 {purpose} rights drift"
            )
        if purpose != "evaluation" and decision.get("authority") != spec.get("rights_authority"):
            raise ExternalSnapshotRegistryV2Error(
                f"{spec['source_id']}: DATA-229 {purpose} authority drift"
            )


def _purpose_rights(spec: Mapping[str, Any], eval_policy: str) -> dict[str, Any]:
    rights = spec.get("rights")
    if not isinstance(rights, Mapping) or set(rights) != set(PURPOSES):
        raise ExternalSnapshotRegistryV2Error(f"{spec['source_id']}: purpose rights incomplete")
    if rights.get("model_training") != ALLOWED or rights.get("redistribution") != ALLOWED:
        raise ExternalSnapshotRegistryV2Error(
            f"{spec['source_id']}: training and redistribution admission required"
        )
    if rights.get("evaluation") != NOT_ADMITTED:
        raise ExternalSnapshotRegistryV2Error(
            f"{spec['source_id']}: evaluation must fail closed without separate admission"
        )
    result: dict[str, Any] = {}
    for purpose in PURPOSES:
        status = str(rights[purpose])
        authority = (
            eval_policy
            if purpose == "evaluation"
            else str(spec.get("rights_authority"))
        )
        result[purpose] = {
            "status": status,
            "authority": authority,
            "decision_identity_sha256": _identity(
                {
                    "authority": authority,
                    "purpose": purpose,
                    "source_id": spec["source_id"],
                    "source_version": spec["source_version"],
                    "status": status,
                }
            ),
        }
    return result


def _source_entry(spec: Mapping[str, Any], eval_policy: str) -> dict[str, Any]:
    sid = str(spec.get("source_id"))
    raw_sha = _sha(spec.get("raw_sha256"), f"{sid} raw")
    norm_sha = _sha(spec.get("normalized_sha256"), f"{sid} normalized")
    raw_bytes = spec.get("raw_bytes")
    norm_bytes = spec.get("normalized_bytes")
    if not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ExternalSnapshotRegistryV2Error(f"{sid}: invalid raw byte count")
    if not isinstance(norm_bytes, int) or norm_bytes <= 0:
        raise ExternalSnapshotRegistryV2Error(f"{sid}: invalid normalized byte count")
    if spec.get("mirror") is not False or spec.get("fork") is not False:
        raise ExternalSnapshotRegistryV2Error(f"{sid}: mirror/fork source is not independent")
    if spec.get("modality") == "code":
        if raw_sha != norm_sha or raw_bytes != norm_bytes:
            raise ExternalSnapshotRegistryV2Error(f"{sid}: DATA-227 code must preserve bytes")
        if not _SHA1.fullmatch(str(spec.get("git_blob_sha1", ""))):
            raise ExternalSnapshotRegistryV2Error(f"{sid}: exact Git blob SHA-1 required")
    family = str(spec.get("independent_family_id"))
    if not family:
        raise ExternalSnapshotRegistryV2Error(f"{sid}: independent family id required")
    if not isinstance(spec.get("source_version"), str) or not spec["source_version"]:
        raise ExternalSnapshotRegistryV2Error(f"{sid}: exact source version required")
    for field in ("family_identity_sha256", "source_manifest_sha256", "license_sha256"):
        value = spec.get(field)
        if value is not None:
            _sha(value, f"{sid} {field}")
    family_core = {
        "family_id": family,
        "canonical_repository": spec.get("canonical_repository"),
    }
    return {
        "registry_source_id": "external-real:" + sid,
        "origin_class": "EXTERNAL_REAL",
        "producer_worker": spec["producer"],
        "source_id": sid,
        "source_version": spec["source_version"],
        "independent_source_family": {
            "family_id": family,
            "family_identity_sha256": spec.get("family_identity_sha256")
            or _identity(family_core),
            "mirror": False,
            "fork": False,
        },
        "language": spec["language"],
        "language_reporting_alias": spec.get("language_reporting_alias"),
        "modality": spec["modality"],
        "snapshot": {
            "raw_sha256": raw_sha,
            "raw_bytes": raw_bytes,
            "normalized_sha256": norm_sha,
            "normalized_bytes": norm_bytes,
            "normalization_policy": spec["normalization_policy"],
            "snapshot_uri": f"file:data/external/snapshots/sha256/{raw_sha}/payload",
        },
        "exact_upstream_identity": {
            "canonical_repository": spec.get("canonical_repository"),
            "commit": spec.get("commit"),
            "path": spec.get("path"),
            "git_blob_sha1": spec.get("git_blob_sha1"),
            "source_manifest_sha256": spec.get("source_manifest_sha256"),
        },
        "producer_evidence_binding": {
            "registry_identity_sha256": spec.get("producer_registry_identity_sha256"),
            "report_identity_sha256": spec.get("producer_report_identity_sha256"),
            "rights_policy_sha256": spec.get("producer_rights_policy_sha256"),
        },
        "license": {
            "license_id": spec["license_id"],
            "license_sha256": spec.get("license_sha256"),
            "redistribution_conditions": spec.get("redistribution_conditions"),
        },
        "rights": _purpose_rights(spec, eval_policy),
    }


def _aggregate(sources: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        if key == "family":
            value = source["independent_source_family"]["family_id"]
        else:
            value = str(source[key])
        groups[value].append(source)
    rows: list[dict[str, Any]] = []
    for value in sorted(groups):
        members = groups[value]
        raw: dict[str, int] = {}
        normalized: dict[str, int] = {}
        for source in members:
            snapshot = source["snapshot"]
            raw.setdefault(snapshot["raw_sha256"], snapshot["raw_bytes"])
            normalized.setdefault(snapshot["normalized_sha256"], snapshot["normalized_bytes"])
        rows.append(
            {
                "key": value,
                "snapshot_count": len(members),
                "unique_raw_object_count": len(raw),
                "unique_raw_bytes": sum(raw.values()),
                "unique_normalized_object_count": len(normalized),
                "unique_normalized_bytes": sum(normalized.values()),
            }
        )
    return rows


def registry_identity(registry: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(registry))
    core.pop("registry_identity_sha256", None)
    return _identity(core)


def validate_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ExternalSnapshotRegistryV2Error("registry schema mismatch")
    if registry.get("registry_identity_sha256") != registry_identity(registry):
        raise ExternalSnapshotRegistryV2Error("registry self-identity mismatch")
    sources = registry.get("sources")
    if not isinstance(sources, list) or registry.get("source_count") != len(sources):
        raise ExternalSnapshotRegistryV2Error("registry inventory malformed")
    ids = [str(source.get("registry_source_id")) for source in sources]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ExternalSnapshotRegistryV2Error("registry source ids must be unique and sorted")
    families = {
        source["independent_source_family"]["family_id"]
        for source in sources
    }
    if registry.get("independent_source_family_count") != len(families):
        raise ExternalSnapshotRegistryV2Error("independent source family count drift")
    if any(source["independent_source_family"]["mirror"] for source in sources):
        raise ExternalSnapshotRegistryV2Error("mirror admitted")
    if any(source["independent_source_family"]["fork"] for source in sources):
        raise ExternalSnapshotRegistryV2Error("fork admitted")
    for source in sources:
        if set(source["rights"]) != set(PURPOSES):
            raise ExternalSnapshotRegistryV2Error("purpose rights missing")
        if source["rights"]["evaluation"]["status"] != NOT_ADMITTED:
            raise ExternalSnapshotRegistryV2Error("evaluation rights widened")
    expected_total_raw = sum(row["unique_raw_bytes"] for row in _aggregate(sources, "language"))
    expected_total_norm = sum(
        row["unique_normalized_bytes"] for row in _aggregate(sources, "language")
    )
    report = registry.get("byte_report")
    if not isinstance(report, Mapping):
        raise ExternalSnapshotRegistryV2Error("byte report missing")
    if report.get("unique_raw_bytes") != expected_total_raw:
        raise ExternalSnapshotRegistryV2Error("raw byte total drift")
    if report.get("unique_normalized_bytes") != expected_total_norm:
        raise ExternalSnapshotRegistryV2Error("normalized byte total drift")


def build_external_snapshot_registry_v2(
    *, inputs_path: str | Path, base_registry_path: str | Path
) -> dict[str, Any]:
    inputs = _load(inputs_path)
    base = _load(base_registry_path)
    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("local_free_only") is not True:
        raise ExternalSnapshotRegistryV2Error("invalid DATA-287 input manifest")
    if inputs.get("origin_namespace") != "external-real:":
        raise ExternalSnapshotRegistryV2Error("external-real origin namespace drift")
    if inputs.get("evaluation_rights_policy") != (
        "FAIL_CLOSED_UNLESS_SEPARATELY_ADMITTED_FOR_EVALUATION"
    ):
        raise ExternalSnapshotRegistryV2Error("evaluation rights policy drift")
    base_spec = inputs.get("base_registry")
    if not isinstance(base_spec, Mapping):
        raise ExternalSnapshotRegistryV2Error("DATA-229 base binding missing")
    if base.get("registry_identity_sha256") != base_spec.get("registry_identity_sha256"):
        raise ExternalSnapshotRegistryV2Error("DATA-229 base registry identity drift")
    if base.get("schema_version") != base_spec.get("schema_version"):
        raise ExternalSnapshotRegistryV2Error("DATA-229 base registry schema drift")
    authorities = _assert_authorities(inputs)
    base_index = _base_text_index(base)
    specs = inputs.get("sources")
    if not isinstance(specs, list) or not specs:
        raise ExternalSnapshotRegistryV2Error("source inventory missing")
    source_ids = [str(spec.get("source_id")) for spec in specs if isinstance(spec, Mapping)]
    if len(source_ids) != len(specs) or len(source_ids) != len(set(source_ids)):
        raise ExternalSnapshotRegistryV2Error("source inventory duplicate/malformed")
    eval_policy = str(inputs.get("evaluation_rights_policy"))
    sources: list[dict[str, Any]] = []
    for raw_spec in sorted(specs, key=lambda item: str(item["source_id"])):
        spec = dict(raw_spec)
        producer = str(spec.get("producer"))
        if authorities.get(producer, {}).get("status") != "TERMINAL_SUCCESS":
            raise ExternalSnapshotRegistryV2Error(
                f"{spec['source_id']}: non-terminal-success producer cannot be consumed"
            )
        if producer == "DATA-213":
            base_source = base_index.get(str(spec["source_id"]))
            if base_source is None:
                raise ExternalSnapshotRegistryV2Error(
                    f"{spec['source_id']}: missing from DATA-229 base registry"
                )
            _verify_text_source(spec, base_source)
            spec["producer_registry_identity_sha256"] = base_spec["registry_identity_sha256"]
        elif producer == "DATA-227":
            if spec.get("modality") != "code":
                raise ExternalSnapshotRegistryV2Error("DATA-227 may contribute code only")
            if spec.get("rights_authority") != "policy://12-6/data/explicit-model-training-evidence-v1":
                raise ExternalSnapshotRegistryV2Error("DATA-227 rights authority drift")
            spec["producer_registry_identity_sha256"] = authorities["DATA-227"][
                "producer_registry_identity_sha256"
            ]
            spec["producer_report_identity_sha256"] = authorities["DATA-227"][
                "report_identity_sha256"
            ]
            spec["producer_rights_policy_sha256"] = authorities["DATA-227"][
                "policy_sha256"
            ]
        else:
            raise ExternalSnapshotRegistryV2Error(f"unsupported producer: {producer}")
        sources.append(_source_entry(spec, eval_policy))
    sources.sort(key=lambda source: source["registry_source_id"])
    raw_unique = {
        source["snapshot"]["raw_sha256"]: source["snapshot"]["raw_bytes"]
        for source in sources
    }
    norm_unique = {
        source["snapshot"]["normalized_sha256"]: source["snapshot"]["normalized_bytes"]
        for source in sources
    }
    family_rows = _aggregate(sources, "family")
    out = {
        "schema_version": REGISTRY_SCHEMA,
        "local_free_only": True,
        "cutoff_utc": inputs.get("cutoff_utc"),
        "base_registry": copy.deepcopy(base_spec),
        "terminal_authorities": copy.deepcopy(authorities),
        "source_count": len(sources),
        "independent_source_family_count": len(family_rows),
        "sources": sources,
        "family_deduplication": {
            "policy": "CANONICAL_INDEPENDENT_FAMILY_ID_NO_MIRROR_OR_FORK_DOUBLE_COUNTING",
            "family_alias_count": 0,
            "excluded_mirror_or_fork_count": 0,
            "family_rows": family_rows,
        },
        "byte_report": {
            "unique_raw_object_count": len(raw_unique),
            "unique_raw_bytes": sum(raw_unique.values()),
            "unique_normalized_object_count": len(norm_unique),
            "unique_normalized_bytes": sum(norm_unique.values()),
            "by_language": _aggregate(sources, "language"),
            "by_modality": _aggregate(sources, "modality"),
            "by_independent_source_family": family_rows,
        },
        "explicit_exclusions": copy.deepcopy(inputs.get("explicit_exclusions", [])),
        "claim_boundary": {
            "evaluation_authorized_source_count": 0,
            "training_authorized_source_count": sum(
                source["rights"]["model_training"]["status"] == ALLOWED
                for source in sources
            ),
            "redistribution_authorized_source_count": sum(
                source["rights"]["redistribution"]["status"] == ALLOWED
                for source in sources
            ),
            "failed_terminal_candidates_consumed": False,
            "universal_benchmark_clean": False,
            "representative_corpus_claimed": False,
        },
    }
    out["registry_identity_sha256"] = registry_identity(out)
    validate_registry(out)
    return out


def serialize_registry(registry: Mapping[str, Any]) -> bytes:
    validate_registry(registry)
    return canonical_json_bytes(registry)
