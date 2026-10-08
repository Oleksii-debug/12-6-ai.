"""Fail-closed Plan-1 third-party code reuse and Base ancestry checks.

No downloader, model loader, scheduler, registry, or evaluator is created here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

HEX = re.compile(r"^[0-9a-f]{64}$")
AUTHORITIES = {
    "model": "twelve_six.model",
    "checkpoint": "twelve_six.checkpoint",
    "scheduler": "twelve_six.training",
    "evaluation": "twelve_six.capability_map",
}
ASSET_KEYS = {
    "name", "role", "status", "upstream_url", "version", "source_sha256",
    "license_spdx", "license_evidence_sha256", "security_posture",
    "data_rights", "model_weights", "replacement_boundary",
}
LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC"}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value: str) -> Any:
    raise ValueError("nonfinite JSON")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON")
    return result


def _strict(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ValueError("bounded UTF-8 bytes required")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=_bad_constant, parse_float=_finite_float)
    except ValueError as exc:
        if str(exc) == "duplicate JSON key":
            raise
        raise ValueError("invalid strict JSON") from exc
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid strict JSON") from exc
    if type(value) is not dict:
        raise ValueError("object root required")
    return value


def _hash(value: object) -> bool:
    return type(value) is str and HEX.fullmatch(value) is not None


def validate_reuse_catalog(raw: bytes) -> dict[str, Any]:
    payload = _strict(raw)
    if set(payload) != {"schema_version", "canonical_authorities", "assets"}:
        raise ValueError("reuse schema drift")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("reuse schema version drift")
    if payload["canonical_authorities"] != AUTHORITIES:
        raise ValueError("second canonical authority forbidden")
    assets = payload["assets"]
    if type(assets) is not list or not assets:
        raise ValueError("assets must be a nonempty list")
    observed: set[str] = set()
    for asset in assets:
        if type(asset) is not dict or set(asset) != ASSET_KEYS:
            raise ValueError("incomplete asset evidence")
        name = asset["name"]
        if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", name):
            raise ValueError("invalid asset identifier")
        if name in observed:
            raise ValueError("duplicate asset")
        observed.add(name)
        if type(asset["role"]) is not str or asset["role"] not in {
            "code_adapter", "dataset", "model_weights"
        }:
            raise ValueError("invalid role")
        boundary = asset["replacement_boundary"]
        if type(boundary) is not str or not boundary.startswith("adapter:"):
            raise ValueError("replaceable adapter boundary required")
        if any(token in boundary.lower().split(":")[1].split(".")
               for token in ("model", "checkpoint", "scheduler", "evaluation", "registry")):
            raise ValueError("adapter cannot replace canonical authority")
        status = asset["status"]
        if type(status) is not str or status not in {
            "CANDIDATE_UNQUALIFIED", "REVIEWED_CODE_ONLY"
        }:
            raise ValueError("invalid admission status")
        if status == "CANDIDATE_UNQUALIFIED":
            if any(asset[field] is not None for field in (
                "version", "source_sha256", "license_evidence_sha256"
            )) or asset["security_posture"] != "UNKNOWN":
                raise ValueError("candidate cannot claim partial admission")
            continue
        if asset["role"] != "code_adapter" or asset["model_weights"] != "NONE":
            raise ValueError("no foreign weights in canonical Base")
        if asset["data_rights"] != "NOT_APPLICABLE_CODE":
            raise ValueError("code permission is not dataset permission")
        if asset["security_posture"] != "REVIEWED":
            raise ValueError("unknown security posture")
        if type(asset["license_spdx"]) is not str or asset["license_spdx"] not in LICENSES:
            raise ValueError("unknown/incompatible license")
        if type(asset["upstream_url"]) is not str or not asset["upstream_url"].startswith(
            "https://"
        ):
            raise ValueError("missing HTTPS source")
        if type(asset["version"]) is not str or not asset["version"].strip():
            raise ValueError("exact version required")
        if not _hash(asset["source_sha256"]) or not _hash(
            asset["license_evidence_sha256"]
        ):
            raise ValueError("source/license hash required")
    return payload


def require_reviewed_code(raw: bytes, name: str, exact_source_sha256: str) -> dict[str, Any]:
    payload = validate_reuse_catalog(raw)
    for asset in payload["assets"]:
        if asset["name"] == name:
            if asset["status"] != "REVIEWED_CODE_ONLY":
                raise ValueError("unqualified code asset")
            if not _hash(exact_source_sha256) or asset["source_sha256"] != exact_source_sha256:
                raise ValueError("source hash drift")
            return dict(asset)
    raise ValueError("unknown external asset")


def validate_base_lineage(payload: dict[str, Any], trusted_genesis: dict[str, Any]) -> str:
    """Require closed first-party ancestry rooted in separately verified scratch init.

    This does not authenticate the caller's trusted_genesis or checkpoint bytes.
    Those must be authenticated by the canonical checkpoint authority.
    """
    expected = {
        "schema_version", "genesis_id", "model_spec_sha256",
        "init_spec_sha256", "head_id", "checkpoints"
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("Base lineage schema drift")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported Base ancestry version")
    if type(trusted_genesis) is not dict or set(trusted_genesis) != {
        "genesis_id", "model_spec_sha256", "init_spec_sha256", "origin"
    } or trusted_genesis["origin"] != "local_random_init":
        raise ValueError("independently trusted scratch genesis required")
    for key in ("genesis_id", "model_spec_sha256", "init_spec_sha256"):
        if not _hash(payload[key]) or payload[key] != trusted_genesis[key]:
            raise ValueError("foreign lineage/model identity drift")
    checkpoints = payload["checkpoints"]
    if type(checkpoints) is not list or not checkpoints:
        raise ValueError("complete checkpoint graph required")
    parents: dict[str, list[str]] = {}
    for item in checkpoints:
        if type(item) is not dict or set(item) != {"id", "parents", "origin"}:
            raise ValueError("invalid checkpoint")
        node, refs = item["id"], item["parents"]
        if not _hash(node) or node in parents:
            raise ValueError("invalid or duplicate checkpoint identity")
        if item["origin"] != "canonical_base":
            raise ValueError("foreign/pretrained/posttrained checkpoint")
        if type(refs) is not list or len(refs) > 1 or any(not _hash(x) for x in refs):
            raise ValueError("ambiguous parents")
        parents[node] = refs
    root = payload["genesis_id"]
    if root not in parents or parents[root] != [] or payload["head_id"] not in parents:
        raise ValueError("missing scratch genesis/head")
    for node, refs in parents.items():
        if node != root and len(refs) != 1:
            raise ValueError("unexpected second Base root")
        if any(p not in parents for p in refs):
            raise ValueError("foreign checkpoint parent")
    for node in parents:
        cursor = node
        seen: set[str] = set()
        while cursor != root:
            if cursor in seen:
                raise ValueError("ancestry cycle")
            seen.add(cursor)
            cursor = parents[cursor][0]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
