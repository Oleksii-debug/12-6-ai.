#!/usr/bin/env python3
"""Materialize DATA-299's hash-only exact exclusion union using local files only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "12-6.data299-eval-exclusion-registry.v1"
OUTPUT_SCHEMA = "12-6.data299-materialized-eval-exclusions.v1"
DEFAULT_REGISTRY = Path("data/evaluation/data299_eval_exclusion_registry_v1.json")
PROHIBITED_PUBLIC_KEYS = ("score", "result", "metric", "loss", "bpb", "accuracy", "perplexity", "margin", "outcome")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value, raw


def _verify_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != SCHEMA:
        raise ValueError("unexpected DATA-299 registry schema")
    observed = registry.get("registry_identity_sha256")
    without_identity = dict(registry)
    without_identity.pop("registry_identity_sha256", None)
    expected = _canonical_sha256(without_identity)
    if observed != expected:
        raise ValueError(f"registry identity mismatch: expected {expected}, got {observed}")
    for key in _walk_keys(registry):
        lowered = key.lower()
        if any(token in lowered for token in PROHIBITED_PUBLIC_KEYS):
            raise ValueError(f"prohibited evaluation field in public registry: {key}")


def _extract_shard(kind: str, data: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    if kind == "eval132_normalized_sha256_v1":
        if data.get("registry_identity_sha256") != spec.get("registry_identity_sha256"):
            raise ValueError("EVAL-132 shard registry identity mismatch")
        sets = data.get("sets")
        if not isinstance(sets, list) or len(sets) != 1:
            raise ValueError("EVAL-132 shard must contain exactly one set")
        values = sets[0].get("normalized_sha256")
    elif kind == "eval133_item_sha256_v1":
        if data.get("reservation_sha256") != spec.get("reservation_sha256"):
            raise ValueError("EVAL-133 reservation identity mismatch")
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("EVAL-133 items missing")
        values = [item.get("item_sha256") for item in items if isinstance(item, dict)]
    elif kind == "eval134_forbidden_normalized_sha256_v1":
        suites = data.get("reserved_evaluation_suites")
        if not isinstance(suites, list) or len(suites) != 1:
            raise ValueError("EVAL-134 reserved suite binding missing")
        values = data.get("forbidden_normalized_sha256")
    else:
        raise ValueError(f"unknown hash shard extractor: {kind}")

    if not isinstance(values, list) or not all(_is_sha256(value) for value in values):
        raise ValueError(f"{kind}: invalid SHA-256 list")
    if len(values) != int(spec["expected_count"]):
        raise ValueError(f"{kind}: expected {spec['expected_count']} hashes, got {len(values)}")
    return list(values)


def _inline_hashes(registry: dict[str, Any]) -> list[str]:
    values: list[str] = []
    selection = registry["selection_validation"]
    for language in ("ua", "en", "code"):
        values.extend(selection[language].get("exact_sha256", []))
    for authority in registry["reserved_authorities"]:
        values.extend(authority.get("exact_sha256", []))
        values.extend(authority.get("exact_source_sha256", []))
    if not all(_is_sha256(value) for value in values):
        raise ValueError("invalid inline SHA-256")
    return values


def materialize(repo_root: Path, registry_path: Path) -> dict[str, Any]:
    registry, _ = _load_json(registry_path)
    _verify_registry(registry)

    values = _inline_hashes(registry)
    for authority in registry["reserved_authorities"]:
        spec = authority.get("hash_shard")
        if spec is None:
            continue
        shard_path = repo_root / spec["path"]
        data, raw = _load_json(shard_path)
        observed_blob = _git_blob_sha1(raw)
        if observed_blob != spec["git_blob_sha1"]:
            raise ValueError(
                f"{spec['path']}: Git blob mismatch: expected {spec['git_blob_sha1']}, got {observed_blob}"
            )
        values.extend(_extract_shard(spec["extractor"], data, spec))

    exact = sorted(set(values))
    exact_set_sha256 = hashlib.sha256("\n".join(exact).encode()).hexdigest()
    near = registry["near_match_authority"]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "source_registry_identity_sha256": registry["registry_identity_sha256"],
        "exact_sha256": exact,
        "exact_count": len(exact),
        "exact_set_sha256": exact_set_sha256,
        "near_match_required": bool(near["required"]),
        "near_match_method_id": near["method_id"],
        "near_match_source_commit": near["source_commit"],
        "public_evidence_hash_only": True,
        "execution_class": "LOCAL_FREE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    output = materialize(repo_root, registry_path)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema_version": output["schema_version"],
                    "source_registry_identity_sha256": output["source_registry_identity_sha256"],
                    "exact_count": output["exact_count"],
                    "exact_set_sha256": output["exact_set_sha256"],
                    "near_match_method_id": output["near_match_method_id"],
                    "execution_class": output["execution_class"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.output is None:
        raise SystemExit("--output is required unless --validate-only is used")
    out_path = args.output
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
