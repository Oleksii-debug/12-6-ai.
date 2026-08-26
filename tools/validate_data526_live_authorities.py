#!/usr/bin/env python3
"""Live, read-only provenance verification for the DATA-526 candidate authorities."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "configs/data/data526_predecontam_source_records_v1.json"
DEFAULT_REPOSITORY = "Oleksii-debug/12-6-ai."
API_ROOT = "https://api.github.com"

AUTHORITIES: dict[str, dict[str, Any]] = {
    "data287_incumbent_registry": {
        "head_sha": "b0523ccbc4b957615aac849d476cfa851be87578",
        "run_id": 32968622282,
        "workflow_name": "DATA-287 External Snapshot Registry V2",
        "path": "data/registry/external_snapshots.v2.json",
        "identity_field": "registry_identity_sha256",
        "identity": "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c",
    },
    "next100_022_ua_wikisource": {
        "head_sha": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
        "run_id": 32998002424,
        "workflow_name": "NEXT100-022 Ukrainian Wikisource Qualification",
        "path": "configs/data/next100022_ua_wikisource_candidate_v1.json",
        "identity_field": "authority_identity_sha256",
        "identity": "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
    },
    "next100_034_nist_terminal": {
        "head_sha": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "run_id": 32998703545,
        "workflow_name": "NEXT100-034 NIST authority",
        "path": "configs/data/next100_034_nist_terminal_authority_v2.json",
        "identity_field": "terminal_payload_sha256",
        "identity": "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
    },
}


class ProvenanceError(RuntimeError):
    """Raised when live authority provenance disagrees with the frozen candidate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def _request_json(
    url: str,
    token: str | None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "12-6-ai-data526-provenance-validator",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with opener(request, timeout=20) as response:
        payload = response.read()
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"GitHub API returned non-object JSON for {url}")
    return value


def _authority_content(
    repository: str,
    authority: dict[str, Any],
    token: str | None,
    opener: Callable[..., Any],
) -> tuple[dict[str, Any], str]:
    encoded_path = quote(authority["path"], safe="/")
    url = (
        f"{API_ROOT}/repos/{repository}/contents/{encoded_path}"
        f"?ref={authority['head_sha']}"
    )
    response = _request_json(url, token, opener)
    _require(response.get("encoding") == "base64", f"authority content encoding drift: {authority['path']}")
    _require(isinstance(response.get("content"), str), f"authority content missing: {authority['path']}")
    raw = base64.b64decode(response["content"], validate=False)
    parsed = json.loads(raw.decode("utf-8"))
    _require(isinstance(parsed, dict), f"authority JSON is not an object: {authority['path']}")
    blob_sha = response.get("sha")
    _require(isinstance(blob_sha, str) and len(blob_sha) == 40, f"authority blob SHA missing: {authority['path']}")
    return parsed, blob_sha


def validate_live_authorities(
    inventory: dict[str, Any],
    repository: str,
    token: str | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    _require(repository == inventory.get("repository") == DEFAULT_REPOSITORY, "repository identity drift")
    frozen_evidence = inventory.get("authority_evidence")
    _require(isinstance(frozen_evidence, dict), "authority_evidence must be an object")
    _require(set(frozen_evidence) == set(AUTHORITIES), "authority set drift")

    verified: dict[str, dict[str, Any]] = {}
    allowed_record_bindings: set[tuple[str, str]] = set()

    for name, authority in AUTHORITIES.items():
        frozen = frozen_evidence[name]
        _require(frozen.get("head_sha") == authority["head_sha"], f"frozen head drift: {name}")
        _require(
            frozen.get("authority_identity_sha256") == authority["identity"],
            f"frozen authority identity drift: {name}",
        )
        _require(frozen.get("dedicated_workflow_run") == authority["run_id"], f"frozen workflow run drift: {name}")
        _require(
            frozen.get("dedicated_workflow_conclusion") == "success",
            f"frozen workflow conclusion is not success: {name}",
        )

        run_url = f"{API_ROOT}/repos/{repository}/actions/runs/{authority['run_id']}"
        run = _request_json(run_url, token, opener)
        _require(run.get("id") == authority["run_id"], f"live workflow run id drift: {name}")
        _require(run.get("name") == authority["workflow_name"], f"live workflow name drift: {name}")
        _require(run.get("head_sha") == authority["head_sha"], f"live workflow head drift: {name}")
        _require(run.get("status") == "completed", f"live workflow is not completed: {name}")
        _require(run.get("conclusion") == "success", f"live workflow is not success: {name}")
        run_repo = run.get("repository")
        _require(isinstance(run_repo, dict), f"live workflow repository missing: {name}")
        _require(run_repo.get("full_name") == repository, f"live workflow repository drift: {name}")

        authority_json, blob_sha = _authority_content(repository, authority, token, opener)
        observed_identity = authority_json.get(authority["identity_field"])
        _require(observed_identity == authority["identity"], f"live authority identity drift: {name}")

        verified[name] = {
            "head_sha": authority["head_sha"],
            "workflow_run_id": authority["run_id"],
            "workflow_name": authority["workflow_name"],
            "authority_path": authority["path"],
            "authority_blob_sha1": blob_sha,
            "authority_identity_sha256": authority["identity"],
        }
        allowed_record_bindings.add((authority["head_sha"], authority["identity"]))

    records = inventory.get("records")
    _require(isinstance(records, list) and records, "records must be a non-empty list")
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"record {index} is not an object")
        binding = (record.get("authority_head_sha"), record.get("authority_identity_sha256"))
        _require(binding in allowed_record_bindings, f"record {index} is not bound to a verified live authority")
        _require(record.get("pre_decontamination_candidate") is True, f"record {index} candidate flag drift")
        _require(record.get("final_training_eligible") is False, f"record {index} prematurely training eligible")

    return {
        "status": "PASS_LIVE_AUTHORITY_PROVENANCE_ONLY",
        "repository": repository,
        "authority_count": len(verified),
        "record_count": len(records),
        "authorities": verified,
        "decontamination_executed": False,
        "final_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY))
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    report = validate_live_authorities(
        inventory,
        repository=args.repository,
        token=os.environ.get("GITHUB_TOKEN"),
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
