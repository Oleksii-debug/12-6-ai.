from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.code_sources import (
    CODE_CANDIDATE_SCHEMA,
    CODE_PILOT_REPORT_SCHEMA,
    CodeSourceCandidate,
    LicenseObservation,
    build_sample_manifest,
    code_mixture_stratum,
    evaluate_d03_rights,
    exact_duplicate_analysis,
    ingest_code_file,
    near_duplicate_analysis,
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-registry", default="configs/data/code_source_candidates.v1.json"
    )
    parser.add_argument("--d03-registry", default="data/external/external_sources.json")
    parser.add_argument("--output", default="reports/d03/code_source_pilot_20260825.json")
    parser.add_argument("--sample-manifest", default="data/samples/data23/manifest.json")
    args = parser.parse_args()

    candidate_path = Path(args.candidate_registry)
    candidate_registry = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate_registry.get("schema_version") != CODE_CANDIDATE_SCHEMA:
        raise SystemExit("unsupported code candidate registry")
    d03_path = Path(args.d03_registry)
    d03_registry = json.loads(d03_path.read_text(encoding="utf-8"))

    records = []
    source_rights = []
    for raw in candidate_registry["sources"]:
        candidate = CodeSourceCandidate(
            source_id=raw["source_id"],
            repository=raw["repository"],
            revision=raw["revision"],
            source_url=raw["source_url"],
            d03_source_version=raw["d03_source_version"],
            d03_rights_status=raw["d03_rights_status"],
            license=LicenseObservation(**raw["license"]),
        )
        rights_status, eligible = evaluate_d03_rights(candidate, d03_registry)
        source_rights.append(
            {
                "source_id": candidate.source_id,
                "revision": candidate.revision,
                "observed_license": candidate.license.spdx_id,
                "d03_status": rights_status,
                "training_eligible": eligible,
            }
        )
        license_bytes = Path(
            "data/samples/data23",
            "itsdangerous" if candidate.repository == "pallets/itsdangerous" else "pluggy",
            candidate.license.path,
        ).read_bytes()
        if sha256(license_bytes) != candidate.license.text_sha256:
            raise SystemExit(f"license bytes drift: {candidate.repository}")
        for item in raw["sample_files"]:
            payload = Path(item["sample_path"]).read_bytes()
            if len(payload) != item["size_bytes"] or sha256(payload) != item["source_sha256"]:
                raise SystemExit(f"source bytes drift: {candidate.repository}:{item['path']}")
            records.append(
                ingest_code_file(
                    candidate,
                    path=item["path"],
                    git_blob_sha1=item["git_blob_sha1"],
                    payload=payload,
                    rights_status=rights_status,
                    training_eligible=eligible,
                )
            )

    sample_manifest = build_sample_manifest(records)
    sample_manifest_path = Path(args.sample_manifest)
    sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest_path.write_bytes(canonical_json_bytes(sample_manifest))

    exact = exact_duplicate_analysis(records)
    near = near_duplicate_analysis(records, threshold=0.85)
    mechanically_accepted = [record for record in records if record.mechanically_accepted]
    training_eligible = [record for record in records if record.training_eligible]
    rejected = [record for record in records if not record.mechanically_accepted]
    blocked = [record for record in mechanically_accepted if not record.training_eligible]

    mixture = {
        "incumbent_contract": "DATA-10 uk/en/code post-tokenization loss-token mixture",
        "code_weight_units": 20,
        "packing": "incumbent D04 packing; no code-specific model behavior",
        "eligible_code_manifest_ready": bool(training_eligible),
        "blocked_sample_not_scheduled": not bool(training_eligible),
    }
    if training_eligible:
        mixture["stratum"] = code_mixture_stratum(sample_manifest["manifest_sha256"]).__dict__

    core = {
        "schema_version": CODE_PILOT_REPORT_SCHEMA,
        "authority": "LOCAL_FREE_REAL_SOURCE_MECHANICAL_PILOT_NOT_TRAINING_APPROVAL",
        "base_data10_sha": candidate_registry["base_data10_sha"],
        "candidate_registry_sha256": sha256(candidate_path.read_bytes()),
        "d03_registry_identity_sha256": d03_registry["registry_identity_sha256"],
        "sample_manifest_sha256": sample_manifest["manifest_sha256"],
        "counts": {
            "candidate_sources": len(candidate_registry["sources"]),
            "candidate_files": len(records),
            "mechanically_accepted_files": len(mechanically_accepted),
            "rejected_files": len(rejected),
            "blocked_by_rights_files": len(blocked),
            "training_eligible_files": len(training_eligible),
            "mechanically_accepted_bytes": sum(
                item.source_size_bytes for item in mechanically_accepted
            ),
            "training_eligible_bytes": sum(item.source_size_bytes for item in training_eligible),
        },
        "source_rights": source_rights,
        "filter_rejections": [
            {"record_id": item.record_id, "path": item.path, "reason": item.rejection_reason}
            for item in rejected
        ],
        "exact_duplicate_analysis": exact,
        "near_duplicate_analysis": near,
        "format_preservation": sample_manifest["format_preservation"],
        "mixture_integration": mixture,
        "truth_boundary": {
            "public_github_implies_training_permission": False,
            "observed_license_text_is_d03_approval": False,
            "comments_or_formatting_removed": False,
            "semantic_duplicate_cleanliness_claimed": False,
            "paid_compute": False,
        },
    }
    report = {**core, "report_sha256": sha256(canonical_json_bytes(core))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
