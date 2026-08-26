#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = ROOT / "configs/data/next100_034_nist_terminal_authority_v2.json"


def fail(msg: str) -> None:
    raise SystemExit(f"NEXT100-034 terminal-seal failure: {msg}")


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> None:
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if seal["schema_version"] != "12-6.next100-034-nist-terminal-authority.v2":
        fail("schema drift")
    if seal["worker_id"] != "NEXT100-034-DATA-EN-NIST" or seal["terminal_status"] != "ADMIT":
        fail("terminal verdict drift")
    if seal["local_free_only"] is not True:
        fail("LOCAL_FREE weakened")

    expected_payload = seal.pop("terminal_payload_sha256")
    actual_payload = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_payload != expected_payload:
        fail("self-identity mismatch")
    seal["terminal_payload_sha256"] = expected_payload

    detail_path = ROOT / seal["detailed_authority"]["path"]
    detail_bytes = detail_path.read_bytes()
    if git_blob_sha1(detail_bytes) != seal["detailed_authority"]["git_blob_sha1"]:
        fail("supporting authority Git blob mismatch")
    detail = json.loads(detail_bytes)
    if detail["worker_id"] != seal["worker_id"]:
        fail("supporting authority worker mismatch")

    evidence_path = ROOT / seal["rights_evidence"]["path"]
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != seal["rights_evidence"]["sha256"]:
        fail("rights evidence hash mismatch")

    ids = [d["publication_id"] for d in seal["admit"]]
    if ids != ["NIST.SP.800-204", "NIST.SP.800-204C", "NIST.SP.800-215"]:
        fail("admitted publication order/set drift")
    if seal["family"]["independent_family_count"] != 1 or seal["family"]["publication_count"] != 3:
        fail("family accounting drift")
    if sum(d["normalized_utf8_bytes"] for d in seal["admit"]) != 59358:
        fail("normalized byte total drift")
    if len({d["raw_sha256"] for d in seal["admit"]}) != 3:
        fail("raw exact duplicate")
    if len({d["normalized_sha256"] for d in seal["admit"]}) != 3:
        fail("normalized exact duplicate")

    detail_by_id = {d["publication_id"]: d for d in detail["admit"]}
    for row in seal["admit"]:
        source = detail_by_id.get(row["publication_id"])
        if source is None:
            fail("terminal object missing from supporting authority")
        for key in ("raw_bytes", "raw_sha256", "normalized_utf8_bytes", "normalized_sha256"):
            if row[key] != source[key]:
                fail(f"{row['publication_id']}: terminal/supporting identity mismatch for {key}")

    rights = seal["rights"]
    if not rights["model_training"].startswith("ALLOWED"):
        fail("training admission missing")
    if not rights["redistribution"].startswith("ALLOWED"):
        fail("redistribution admission missing")
    if rights["evaluation"] != "NOT_SEPARATELY_ADMITTED":
        fail("evaluation boundary weakened")
    if rights["third_party_nist_publications"] != "RETEST_DOCUMENT_SPECIFICALLY":
        fail("third-party caveat weakened")
    if rights["standard_reference_data"] != "RETEST_LICENSE_SPECIFICALLY":
        fail("SRD caveat weakened")

    if set(seal["retest"]) != {"NIST.SP.800-204A", "NIST.SP.800-204B", "NIST.SP.800-204D", "NIST_STANDARD_REFERENCE_DATA"}:
        fail("RETEST set drift")
    if "ALL_NIST_MATERIAL_BLANKET_PUBLIC_DOMAIN_RULE" not in seal["reject"]:
        fail("blanket NIST public-domain rejection missing")

    if seal["dedup"]["exact_collision_against_data287_five_normalized_hashes"] is not False:
        fail("baseline exact-dedup status drift")
    if seal["dedup"]["canonical_cross_source_near_dedup"] != "REQUIRED_BEFORE_CORPUS_INTEGRATION":
        fail("future canonical dedup requirement weakened")

    reg = seal["registry_boundary"]
    if reg["latest_durable_external_registry_discovered"] != "DATA-287-EXTERNAL-SNAPSHOT-REGISTRY-V2":
        fail("registry baseline drift")
    if reg["head_sha"] != "b0523ccbc4b957615aac849d476cfa851be87578":
        fail("registry head drift")
    if reg["source_count"] != 5 or reg["independent_family_count"] != 4:
        fail("registry inventory drift")
    if reg["corpus_mutation"] != "NONE":
        fail("unexpected canonical registry mutation")

    concurrency = seal["concurrency_final"]
    if concurrency["conflicting_nist_pr_found"] is not False:
        fail("concurrency conflict unresolved")
    if concurrency["latest_durable_registry_changed_since_baseline"] is not False:
        fail("registry changed after baseline")
    if not concurrency["concurrent_english_candidate_prs_observed"]:
        fail("concurrent source scan not recorded")
    if "NOT_COMPOSED_OR_COUNTED" not in concurrency["candidate_treatment"]:
        fail("concurrent candidates accidentally composed")

    projection = seal["composition_projection"]
    if projection["data295_caps_pass_on_this_baseline"] is not True:
        fail("DATA-295 composition projection failed")
    if projection["projected_nist_global_share"] > 0.25:
        fail("global family cap exceeded")
    if projection["projected_nist_english_stratum_share"] > 0.60:
        fail("English stratum cap exceeded")
    if seal["corpus_integration"] != "NOT_INTEGRATED_REQUIRES_SUCCESSOR_CORPUS_CONTRACT_AND_CANONICAL_DEDUP":
        fail("corpus integration overclaim")

    print("NEXT100-034 TERMINAL SEAL PASS")
    print(f"terminal_payload_sha256={expected_payload}")
    print("admit=3 family=1 normalized_utf8_bytes=59358 evaluation_admitted=0 corpus_mutation=0")


if __name__ == "__main__":
    main()
