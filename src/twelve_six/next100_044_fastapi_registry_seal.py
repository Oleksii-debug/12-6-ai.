"""NEXT100-044 live DATA-287 registry/concurrency seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "12-6.next100-044-fastapi-registry-seal-report.v1"
WORKER_ID = "NEXT100-044-CODE-FASTAPI"
SEAL_PATH = Path("configs/data/next100_044_fastapi_registry_seal_v1.json")
POLICY_PATH = Path("configs/data/next100_044_fastapi_code_rights_policy_v1.json")
REPORT_NAME = "next100-044-fastapi-registry-seal.json"


class RegistrySealError(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, *, max_bytes: int = 2_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-NEXT100-044-registry-seal/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RegistrySealError("registry download exceeded bounded size")
    return data


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(repo: Path, admission_report: Path, output: Path, source_sha: str) -> dict[str, Any]:
    seal = _load(repo / SEAL_PATH)
    policy = _load(repo / POLICY_PATH)
    admission = _load(admission_report)

    if seal.get("worker_id") != WORKER_ID or admission.get("worker_id") != WORKER_ID:
        raise RegistrySealError("worker identity mismatch")
    if admission.get("source_head_sha") != source_sha:
        raise RegistrySealError("admission report is not bound to exact current head")
    if admission.get("terminal_verdict") != "ADMIT":
        raise RegistrySealError("source admission report is not ADMIT")

    producer_head = seal["producer_head"]
    registry_path = seal["registry_path"]
    url = f"https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./{producer_head}/{registry_path}"
    raw = _download(url)
    registry = json.loads(raw.decode("utf-8"))

    if registry.get("registry_identity_sha256") != seal["registry_identity_sha256"]:
        raise RegistrySealError("canonical registry identity drift")
    if registry.get("source_count") != seal["source_count"]:
        raise RegistrySealError("canonical registry source count drift")
    if registry.get("independent_source_family_count") != seal["independent_source_family_count"]:
        raise RegistrySealError("canonical registry family count drift")

    live_families = sorted(
        {row["independent_source_family"]["family_id"] for row in registry.get("sources", [])}
    )
    expected_families = sorted(seal["existing_families"])
    if live_families != expected_families:
        raise RegistrySealError(f"canonical registry family set drift: {live_families}")
    if seal["candidate_family"] in live_families:
        raise RegistrySealError("FastAPI family is already present in canonical registry")

    live_hashes = {row["snapshot"]["raw_sha256"] for row in registry.get("sources", [])}
    candidate_hashes = {row["raw_sha256"] for row in admission.get("objects", [])}
    exact_overlap = sorted(live_hashes & candidate_hashes)
    if exact_overlap:
        raise RegistrySealError(f"FastAPI object overlaps canonical registry: {exact_overlap}")

    static_identity = policy["current_registry_binding"]["registry_identity_sha256"]
    if static_identity != seal["supersedes_static_policy_registry_identity_sha256"]:
        raise RegistrySealError("static-policy identity changed; correction overlay must be re-reviewed")

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "source_head_sha": source_sha,
        "terminal_verdict": "PASS",
        "execution_profile": "LOCAL_FREE",
        "canonical_registry": {
            "authority": seal["authority"],
            "producer_head": producer_head,
            "path": registry_path,
            "registry_identity_sha256": registry["registry_identity_sha256"],
            "raw_file_sha256": _sha256(raw),
            "source_count": registry["source_count"],
            "independent_source_family_count": registry["independent_source_family_count"],
            "families": live_families,
        },
        "candidate": {
            "family": seal["candidate_family"],
            "family_present_before_admission": False,
            "raw_sha256": sorted(candidate_hashes),
            "exact_registry_overlap": exact_overlap,
        },
        "correction": {
            "superseded_static_policy_identity": static_identity,
            "file_backed_identity": seal["registry_identity_sha256"],
            "reason": seal["basis"],
        },
        "promotion_boundary": "This seal checks the current canonical DATA-287 registry only; successor registry convergence must recompute global dedup across all concurrent terminal code-source authorities.",
    }
    report["authority_identity_sha256"] = _sha256(_cjson(report))
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_NAME).write_bytes(_cjson(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    report = run(args.repo_root.resolve(), args.admission_report, args.output_dir, args.source_sha)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
