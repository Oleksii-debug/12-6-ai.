from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

MANIFEST = Path("configs/data/research_corpus_v1_predecontam_freeze_v4.json")
EXPECTED_SCHEMA = "12-6.research-corpus-v1-predecontam-freeze.v4"
EXPECTED_STATUS = "FROZEN_POST_GLOBAL_DEDUP_PREDECONTAM_CANDIDATE"
EXPECTED_V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
EXPECTED_V7_RUN = 33045763964
EXPECTED_V7_ARTIFACT = 9635595510
EXPECTED_V7_ARTIFACT_NAME = "next100-065e-cross-source-dedup-d3333ec1b4a508df232a5aefccd6686adda745fb"
EXPECTED_V7_ARTIFACT_DIGEST = "sha256:cca6921a2093d4e033976b23b0af180e9dc1945b624b82e218780f8d20bafd18"
EXPECTED_REPORT_PATH = "evidence/next100_065e/a/report.json"
EXPECTED_REPORT_FILE_SHA = "a917a5f240e2eda0015fd09564b29430ca8c95a9916a03f734a4463fe458c08f"
EXPECTED_V7_REPORT = "80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267"
EXPECTED_V7_DEDUP_REPORT = "c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84"
EXPECTED_INVENTORY = "5e54c3587d6d2c0a7d540a1e8aefae77522c97a41c0033a05e78ce22a8e7c617"
EXPECTED_CANDIDATE = "a33baaa3a7ff7a9622dc420c1d61b0d113ab0f11588a39674996bd198845efb6"
EXPECTED_EVIDENCE = "dca1d00f45f53bd98c38e35cdc08dc7d9cdb25b99db611b3c7a940df41b5fdf9"
EXPECTED_TOTAL = 2_215_615
EXPECTED_BY_MODALITY = {"uk": 100_856, "en": 1_838_293, "code": 276_466}
EXPECTED_FAMILIES = {"uk": 4, "en": 5, "code": 6}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_sha256(doc: dict[str, Any]) -> str:
    payload = dict(doc)
    payload.pop("evidence_identity_sha256", None)
    return _sha256(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _candidate_identity(doc: dict[str, Any]) -> str:
    authority = doc["terminal_global_dedup_authority"]
    freeze = doc["candidate_freeze"]
    return _sha256(
        {
            "v7_head_sha": authority["head_sha"],
            "v7_workflow_run": authority["workflow_run"],
            "v7_artifact_id": authority["artifact_id"],
            "v7_artifact_zip_sha256": authority["artifact_zip_sha256"].removeprefix("sha256:"),
            "v7_report_sha256": authority["report_sha256"],
            "v7_dedup_report_sha256": authority["dedup_report_sha256"],
            "record_inventory_digest_sha256": freeze["record_inventory_digest_sha256"],
            "record_count": freeze["record_count"],
            "post_dedup_unique_capacity_bytes": doc["dedup_result"]["post_dedup_conservative_unique_capacity_bytes"],
        }
    )


def validate_doc(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == EXPECTED_SCHEMA, "schema drift")
    _require(doc.get("worker_id") == "DATA-526-RESEARCH-CORPUS-V1-PREDECONTAM-V4", "worker drift")
    _require(doc.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")
    _require(doc.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require(doc.get("status") == EXPECTED_STATUS, "freeze status drift")
    _require(doc.get("evidence_identity_sha256") == canonical_sha256(doc), "evidence self-hash mismatch")

    registry = doc["source_registry_context"]
    _require(registry.get("pull_request") == 538, "registry PR drift")
    _require(registry.get("head_sha") == "10342d590d91b6999c42515cdf87fe31e2355844", "registry observed head drift")
    _require(registry.get("v5_registry_git_blob_sha1") == "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf", "registry V5 blob drift")
    _require(registry.get("exact_head_ci_run") == 33010079393, "registry run drift")
    _require(registry.get("exact_head_ci_conclusion") == "failure", "registry failure truth drift")
    _require(registry.get("role") == "NONAUTHORITATIVE_COORDINATION_CONTEXT_ONLY", "failed registry tip promoted to authority")
    _require(registry.get("capacity_credit_granted_by_registry_tip") is False, "failed registry tip granted capacity")

    authority = doc["terminal_global_dedup_authority"]
    expected_authority = {
        "pull_request": 632,
        "head_sha": EXPECTED_V7_HEAD,
        "worker_id": "NEXT100-065E-CROSSSOURCE-DEDUP-V7",
        "workflow_name": "NEXT100-065E Cross-Source Dedup V7",
        "workflow_run": EXPECTED_V7_RUN,
        "workflow_conclusion": "success",
        "artifact_id": EXPECTED_V7_ARTIFACT,
        "artifact_name": EXPECTED_V7_ARTIFACT_NAME,
        "artifact_zip_sha256": EXPECTED_V7_ARTIFACT_DIGEST,
        "artifact_report_path": EXPECTED_REPORT_PATH,
        "artifact_report_file_sha256": EXPECTED_REPORT_FILE_SHA,
        "report_sha256": EXPECTED_V7_REPORT,
        "dedup_report_sha256": EXPECTED_V7_DEDUP_REPORT,
        "source_authority_mode": "DIRECT_TERMINAL_MATERIALIZATION_AND_LIVE_AUTHORITY_BINDING",
        "materialized_twice_byte_identical": True,
        "public_evidence_text_free": True,
    }
    _require(authority == expected_authority, "terminal V7 authority drift")

    result = doc["dedup_result"]
    _require(result.get("source_object_count") == 35, "source object count drift")
    _require(result.get("source_family_counts") == EXPECTED_FAMILIES, "family vector drift")
    _require(result.get("pre_dedup_capacity_bytes") == EXPECTED_TOTAL, "pre-dedup capacity drift")
    _require(result.get("post_dedup_conservative_unique_capacity_bytes") == EXPECTED_TOTAL, "post-dedup capacity drift")
    _require(result.get("by_stratum_post_dedup_unique_capacity_bytes") == EXPECTED_BY_MODALITY, "post-dedup modality vector drift")
    _require(result.get("capacity_collapsing_duplicate_cluster_count") == 0, "duplicate cluster drift")
    _require(result.get("capacity_collapsing_duplicate_discount_bytes") == 0, "duplicate discount drift")
    _require(result.get("lineage_sibling_same_origin_edges") == 4, "lineage-edge count drift")
    _require(result.get("effective_independent_origin_count") == 15, "origin count drift")

    freeze = doc["candidate_freeze"]
    _require(freeze.get("frozen") is True, "candidate freeze disabled")
    _require(freeze.get("record_granularity") == "SOURCE_OBJECT_IDENTITY_NO_RAW_TEXT", "record granularity drift")
    _require(freeze.get("record_count") == 35, "record count drift")
    storage = freeze.get("record_inventory_storage")
    _require(isinstance(storage, dict), "record inventory storage missing")
    _require(storage.get("authority_artifact_id") == EXPECTED_V7_ARTIFACT, "inventory artifact drift")
    _require(storage.get("artifact_report_path") == EXPECTED_REPORT_PATH, "inventory report path drift")
    _require(storage.get("json_pointer") == "/dedup_v3/sources", "inventory pointer drift")
    _require(storage.get("record_inventory_digest_sha256") == EXPECTED_INVENTORY, "inventory storage digest drift")
    _require(freeze.get("record_inventory_digest_sha256") == EXPECTED_INVENTORY, "record inventory digest drift")
    _require(_candidate_identity(doc) == EXPECTED_CANDIDATE, "candidate identity recomputation drift")
    _require(freeze.get("candidate_identity_sha256") == EXPECTED_CANDIDATE, "candidate identity declaration drift")

    claims = doc["claim_boundary"]
    for key in ("predecontam_candidate_frozen", "global_dedup_terminal"):
        _require(claims.get(key) is True, f"required terminal claim lost: {key}")
    for key in (
        "source_bytes_are_training_tokens", "research_corpus_v1_terminal", "decontamination_executed",
        "quality_privacy_revalidated_post_composition", "balance_family_caps_passed", "cluster_safe_split_published",
        "two_clean_builds_passed", "postpack_unique_loss_ledger_complete", "final_test_payload_accessed",
        "final_test_outcomes_read", "tokenizer_fit_executed", "training_executed", "long_training_authorized",
        "paid_compute_used",
    ):
        _require(claims.get(key) is False, f"downstream claim prematurely enabled: {key}")
    _require(claims.get("authorized_unique_optimized_targets") == 0, "optimized-target capacity fabricated")
    _require(claims.get("optimizer_updates") == 0, "optimizer updates fabricated")

    gates = doc["downstream_gates"]
    _require(gates.get("source_authority") == "SATISFIED_BY_TERMINAL_V7_DIRECT_MATERIALIZATION", "source authority gate drift")
    _require(gates.get("global_exact_near_fragment_lineage_dedup") == "TERMINAL_SUCCESS", "global dedup gate drift")
    _require(gates.get("record_inventory_freeze") == "FROZEN_EXACT_IDENTITY", "freeze gate drift")
    _require(gates.get("reserved_evaluation_decontamination") == "PERMITTED_ONLY_AGAINST_EXACT_CANDIDATE_IDENTITY", "decontamination handoff drift")
    for key in ("cluster_safe_split_pack", "two_clean_build_proof", "postpack_unique_loss_ledger", "tokenizer_fit", "long_training", "paid_compute"):
        _require(gates.get(key) == "NOT_PERMITTED", f"downstream gate weakened: {key}")

    _require(doc.get("evidence_identity_sha256") == EXPECTED_EVIDENCE, "evidence identity drift")


def _github_request(path: str) -> bytes:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "data526-v4-validator"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/Oleksii-debug/12-6-ai./{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"GitHub live authority lookup failed: {path}") from exc


def _github_json(path: str) -> Any:
    try:
        return json.loads(_github_request(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"GitHub JSON authority decode failed: {path}") from exc


def validate_live() -> None:
    run = _github_json(f"actions/runs/{EXPECTED_V7_RUN}")
    _require(run.get("head_sha") == EXPECTED_V7_HEAD, "live V7 run head mismatch")
    _require(run.get("status") == "completed", "live V7 run nonterminal")
    _require(run.get("conclusion") == "success", "live V7 run not success")
    _require(run.get("name") == "NEXT100-065E Cross-Source Dedup V7", "live V7 workflow name drift")

    artifacts = _github_json(f"actions/runs/{EXPECTED_V7_RUN}/artifacts")
    matches = [item for item in artifacts.get("artifacts", []) if item.get("id") == EXPECTED_V7_ARTIFACT]
    _require(len(matches) == 1, "terminal V7 artifact unavailable or ambiguous")
    artifact = matches[0]
    _require(artifact.get("name") == EXPECTED_V7_ARTIFACT_NAME, "live artifact name drift")
    _require(artifact.get("expired") is False, "terminal V7 artifact expired")
    _require(artifact.get("digest") == EXPECTED_V7_ARTIFACT_DIGEST, "live artifact digest drift")

    archive = _github_request(f"actions/artifacts/{EXPECTED_V7_ARTIFACT}/zip")
    _require("sha256:" + hashlib.sha256(archive).hexdigest() == EXPECTED_V7_ARTIFACT_DIGEST, "downloaded artifact ZIP digest drift")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            report_bytes = package.read(EXPECTED_REPORT_PATH)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("terminal V7 artifact report missing or invalid") from exc
    _require(hashlib.sha256(report_bytes).hexdigest() == EXPECTED_REPORT_FILE_SHA, "artifact report file digest drift")
    report = json.loads(report_bytes.decode("utf-8"))
    _require(report.get("report_sha256") == EXPECTED_V7_REPORT, "artifact semantic report identity drift")
    _require(report.get("dedup_v3", {}).get("report_sha256") == EXPECTED_V7_DEDUP_REPORT, "artifact dedup report identity drift")
    sources = report.get("dedup_v3", {}).get("sources")
    _require(isinstance(sources, list) and len(sources) == 35, "artifact record inventory count drift")
    _require(_sha256(sources) == EXPECTED_INVENTORY, "artifact record inventory digest drift")
    _require(report.get("source_vector", {}).get("conservative_unique_capacity_bytes_after_global_dedup") == EXPECTED_TOTAL, "artifact post-dedup capacity drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_doc(doc)
    if args.github_live:
        validate_live()
    print(f"PASS {EXPECTED_STATUS} candidate={doc['candidate_freeze']['candidate_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
