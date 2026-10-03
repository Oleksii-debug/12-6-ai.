#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_AUTHORITY = "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b"
EXPECTED_HEAD = "3f4ad26e1e8f3406a1274418cf5f485814ce3032"
EXPECTED_REPORT = "2d1e99f0cb41a1b90ce995076d88bffa7435b151f6036dea0a54552d89556cd0"
EXPECTED_ARTIFACT = "sha256:63fa5d9b403432074193e290beb0473b5a1f7b74de1ac30bad71b9ec8405e006"
EXPECTED_CORPUS_EVIDENCE = "e454d9b1a94497aec95776c8b9b6318b73647bc3248f62da70e45a7c369f637e"
EXPECTED_RECORDS = {
    "en.project-gutenberg.37177": (66612, "533f768fa90bd6fac90951760f12b1b70777ae3c147dbb053deb86ddf04860ba"),
    "en.project-gutenberg.37985": (1158509, "6c4cd7d1fa4ef2340fe94a3ec737c72a55658ea3fb2bb19e5f5859eb8d8e3810"),
    "en.project-gutenberg.40652": (446989, "e0e7095ba7ac0f1478d18feb4d87be1ed21be37e472fa4203bb92b43f9e593b4"),
}


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-107 validation failed: {message}")


def validate(path: Path) -> dict[str, object]:
    authority = json.loads(path.read_text(encoding="utf-8"))
    claimed = authority.pop("authority_identity_sha256", None)
    require(claimed == EXPECTED_AUTHORITY, "authority identity drift")
    require(hashlib.sha256(canonical(authority)).hexdigest() == claimed, "authority self-hash mismatch")
    require(authority["schema_version"] == "12-6.next100-107-gutenberg-terminal-seal.v1", "schema drift")
    require(authority["worker_id"] == "NEXT100-107-DATA-EN-GUTENBERG-TERMINAL-SEAL", "worker drift")
    require(authority["execution_profile"] == "LOCAL_FREE", "execution profile drift")
    require(authority["decision"] == "TERMINAL_SOURCE_ADMIT", "decision weakened")

    parent = authority["parent_authority"]
    require(parent["pr"] == 470 and parent["head_sha"] == EXPECTED_HEAD, "parent exact-head drift")

    execution = authority["exact_head_execution"]
    require(execution["workflow_run_id"] == 32998859164, "workflow run drift")
    require(execution["workflow_conclusion"] == "success", "workflow evidence is not success")
    require(execution["artifact_id"] == 9618402768 and execution["artifact_digest"] == EXPECTED_ARTIFACT, "artifact evidence drift")
    require(execution["report_sha256"] == EXPECTED_REPORT, "report identity drift")
    require(execution["report_terminal_decision"] == "ADMIT", "source report not ADMIT")
    require(execution["report_corpus_evidence_identity_sha256"] == EXPECTED_CORPUS_EVIDENCE, "evidence identity drift")

    family = authority["source_family"]
    require(family["family_id"] == "en.project-gutenberg.public-domain-books", "family drift")
    require(family["independent_family_credit"] == 1, "family credit inflation")
    require(family["record_count"] == 3 and family["normalized_utf8_bytes"] == 1672110, "family capacity drift")

    records = authority["records"]
    require(len(records) == 3, "record count drift")
    seen: set[str] = set()
    total = 0
    for record in records:
        source_id = record["source_id"]
        require(source_id in EXPECTED_RECORDS, "unexpected source")
        require(source_id not in seen, "duplicate source")
        seen.add(source_id)
        expected_bytes, expected_hash = EXPECTED_RECORDS[source_id]
        require(record["normalized_utf8_bytes"] == expected_bytes, "record byte count drift")
        require(record["normalized_sha256"] == expected_hash, "record normalized identity drift")
        total += expected_bytes
    require(total == 1672110, "record byte arithmetic drift")

    gates = authority["machine_gates"]
    require(all(str(value).startswith("PASS") for value in gates.values()), "machine gate not PASS")

    rights = authority["rights"]
    require(rights["model_training"] == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES", "training-purpose rights drift")
    require(rights["evaluation"] == "NOT_AUTHORIZED", "evaluation rights silently granted")
    require(rights["worldwide_public_domain_claim"] is False, "worldwide public-domain overclaim")

    capacity = authority["capacity"]
    require(capacity["exact_source_level_normalized_bytes"] == 1672110, "capacity drift")
    require(capacity["unique_loss_positions_authorized"] == 0, "source bytes promoted to loss budget")
    require(capacity["corpus_training_capacity_claimed"] is False, "source bytes promoted to corpus capacity")

    claim = authority["claim_boundary"]
    require(claim["source_terminal_admit"] is True, "source admission missing")
    for key in (
        "research_corpus_v1_frozen",
        "training_executed",
        "long_training_authorized",
        "paid_compute_authorized",
        "learned_20m_claim",
    ):
        require(claim[key] is False, f"overclaim: {key}")
    require(claim["corpus_identity"] is None and claim["shard_identity"] is None, "corpus/shard identity fabricated")
    require(claim["optimizer_updates"] == 0, "optimizer updates fabricated")

    return {
        "status": "PASS",
        "authority_identity_sha256": claimed,
        "family": family["family_id"],
        "records": 3,
        "normalized_utf8_bytes": 1672110,
        "authorized_unique_loss_positions": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="configs/data/next100_107_gutenberg_terminal_seal_v1.json")
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
