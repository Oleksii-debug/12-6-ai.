#!/usr/bin/env python3
"""Live exact-evidence verifier for NEXT100-063 source registry V5.

This layer does not create a second source registry. It validates the committed
V4+V5 composition with the incumbent static validator, then independently checks
the immutable attrs Actions run and retained artifact that V5 consumes. On
success it may emit a self-hashed authority envelope bound to the exact current
NEXT100-063 source head. The envelope grants no corpus, tokenizer, training, or
paid-compute authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
STATIC_VALIDATOR = ROOT / "tools/validate_next100_063_terminal_source_registry_v5.py"
V4_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
V5_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"
REPO = "Oleksii-debug/12-6-ai."
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_ATTRS_PR = 474
EXPECTED_ATTRS_HEAD = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
EXPECTED_ATTRS_RUN = 33006080831
EXPECTED_ATTRS_WORKFLOW = "NEXT100-053 attrs Code Source Admission"
EXPECTED_ATTRS_WORKFLOW_PATH = ".github/workflows/next100-053-attrs-code-source-admission.yml"
EXPECTED_ATTRS_ARTIFACT_ID = 9621650719
EXPECTED_ATTRS_ARTIFACT_NAME = (
    "next100-053-attrs-admission-cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
)
EXPECTED_ATTRS_ARTIFACT_DIGEST = (
    "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2"
)
EXPECTED_ATTRS_AUTHORITY = "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
EXPECTED_V4_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
EXPECTED_V5_BLOB = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_VECTOR = {
    "source_object_count": 35,
    "independent_family_count": 15,
    "numeric_training_capacity_bytes": 2_215_615,
    "source_normalized_envelope_bytes": 2_217_976,
    "uncredited_source_normalized_bytes": 2_361,
    "by_stratum": {
        "uk": {"family_count": 4, "numeric_training_capacity_bytes": 100_856},
        "en": {"family_count": 5, "numeric_training_capacity_bytes": 1_838_293},
        "code": {"family_count": 6, "numeric_training_capacity_bytes": 276_466},
    },
}


class LiveAuthorityError(RuntimeError):
    """Raised when immutable upstream execution evidence is not exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveAuthorityError(message)


def _load_static() -> ModuleType:
    spec = importlib.util.spec_from_file_location("next100_063_v5_static", STATIC_VALIDATOR)
    if spec is None or spec.loader is None:
        raise LiveAuthorityError("cannot import V5 static validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def github_get(path: str) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GITHUB_TOKEN is required for live authority verification")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-ai-next100-063-v5-live-authority",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response must be object: {path}")
    return value


def validate_attrs_live(run: Mapping[str, Any], artifacts: Mapping[str, Any]) -> None:
    require(run.get("id") == EXPECTED_ATTRS_RUN, "attrs run id drift")
    require(run.get("name") == EXPECTED_ATTRS_WORKFLOW, "attrs workflow name drift")
    require(run.get("path") == EXPECTED_ATTRS_WORKFLOW_PATH, "attrs workflow path drift")
    require(run.get("event") == "pull_request", "attrs authority must be PR-scoped")
    require(run.get("head_sha") == EXPECTED_ATTRS_HEAD, "attrs run head drift")
    require(run.get("status") == "completed", "attrs run is nonterminal")
    require(run.get("conclusion") == "success", "attrs run did not succeed")
    pull_requests = run.get("pull_requests")
    require(isinstance(pull_requests, list), "attrs run PR binding missing")
    matching_prs = [
        item
        for item in pull_requests
        if isinstance(item, Mapping)
        and item.get("number") == EXPECTED_ATTRS_PR
        and item.get("head", {}).get("sha") == EXPECTED_ATTRS_HEAD
    ]
    require(len(matching_prs) == 1, "attrs run is not bound to exact PR/head")

    rows = artifacts.get("artifacts")
    require(isinstance(rows, list), "attrs artifact listing missing")
    matching = [
        item
        for item in rows
        if isinstance(item, Mapping) and item.get("id") == EXPECTED_ATTRS_ARTIFACT_ID
    ]
    require(len(matching) == 1, "expected attrs artifact missing or duplicated")
    artifact = matching[0]
    require(artifact.get("name") == EXPECTED_ATTRS_ARTIFACT_NAME, "attrs artifact name drift")
    require(artifact.get("digest") == EXPECTED_ATTRS_ARTIFACT_DIGEST, "attrs artifact digest drift")
    require(artifact.get("expired") is False, "attrs artifact expired")
    workflow_run = artifact.get("workflow_run")
    require(isinstance(workflow_run, Mapping), "attrs artifact run binding missing")
    require(workflow_run.get("id") == EXPECTED_ATTRS_RUN, "attrs artifact run id drift")
    require(workflow_run.get("head_sha") == EXPECTED_ATTRS_HEAD, "attrs artifact head drift")


def build_authority(source_head: str) -> dict[str, Any]:
    require(HEAD_RE.fullmatch(source_head) is not None, "source head must be exact 40-hex SHA")
    static = _load_static()
    v4 = json.loads(V4_PATH.read_text(encoding="utf-8"))
    raw_v5 = V5_PATH.read_bytes()
    v5 = json.loads(raw_v5.decode("utf-8"))
    static.validate(v4, v5, v5_blob_sha1=static.git_blob_sha1(raw_v5))

    run = github_get(f"actions/runs/{EXPECTED_ATTRS_RUN}")
    artifacts = github_get(f"actions/runs/{EXPECTED_ATTRS_RUN}/artifacts")
    validate_attrs_live(run, artifacts)

    authority: dict[str, Any] = {
        "schema_version": "12-6.next100-063-terminal-source-registry-v5-authority.v1",
        "worker_id": "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V5",
        "status": "TERMINAL_SOURCE_REGISTRY_V5_EXACT_EVIDENCE",
        "source_head_sha": source_head,
        "v4_registry_identity_sha256": EXPECTED_V4_IDENTITY,
        "v5_config_git_blob_sha1": EXPECTED_V5_BLOB,
        "attrs_terminal_authority": {
            "pr": EXPECTED_ATTRS_PR,
            "head_sha": EXPECTED_ATTRS_HEAD,
            "workflow_run": EXPECTED_ATTRS_RUN,
            "workflow_name": EXPECTED_ATTRS_WORKFLOW,
            "artifact_id": EXPECTED_ATTRS_ARTIFACT_ID,
            "artifact_name": EXPECTED_ATTRS_ARTIFACT_NAME,
            "artifact_digest": EXPECTED_ATTRS_ARTIFACT_DIGEST,
            "authority_identity_sha256": EXPECTED_ATTRS_AUTHORITY,
        },
        "source_vector": EXPECTED_VECTOR,
        "downstream_boundary": {
            "global_cross_source_dedup_required": True,
            "candidate_corpus_frozen": False,
            "authorized_unique_causal_loss_positions": 0,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    authority["authority_identity_sha256"] = _identity(authority)
    return authority


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    authority = build_authority(args.source_head)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(authority, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print("NEXT100-063 V5 LIVE AUTHORITY PASS")
    print("AUTHORITY_IDENTITY_SHA256=" + authority["authority_identity_sha256"])
    print("SOURCE_HEAD_SHA=" + authority["source_head_sha"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
