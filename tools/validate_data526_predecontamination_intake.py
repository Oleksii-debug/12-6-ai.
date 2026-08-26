#!/usr/bin/env python3
"""Validate DATA-526 Research Corpus V1 pre-decontamination intake.

Stdlib-only and network-free. This validator proves identity composition only.
It must never promote the candidate set to a training corpus or authorize
optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MANIFEST_ID = "8d56bf3884be4e9de3b0d024c48f436ee137da38ed2bd08c9a88e4228abe85e7"
EXPECTED_CANDIDATE_ID = "70b519d40ae921c7f8bee3e65e2047b26b266666c15622298c495d9924c647e8"
EXPECTED_DATA300_CONTRACT = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_DATA301_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
EXPECTED_CPYTHON_HEAD = "5a6a495a24bce449334cbc5126d0114f61a9f57c"
EXPECTED_CPYTHON_AUTHORITY = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
EXPECTED_WIKISOURCE_HEAD = "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2"
EXPECTED_WIKISOURCE_AUTHORITY = "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6"
EXPECTED_DECONTAM_HEAD = "80e8fc9828214ce86e16b5c7f2fdec9107b4df43"
EXPECTED_SOURCE_IDS = {
    "external-real:en.standardebooks.manual.8-typography",
    "external-real:en.standardebooks.manual.9-metadata",
    "external-real:ua.rada.open-data.laws-texts.d23314",
    "external-real:code.encode.httpx._content",
    "external-real:code.psf.requests._internal_utils",
    "en.python.docs.tutorial-introduction",
    "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13",
}
EXPECTED_FAMILIES = {
    "en.standardebooks.manual",
    "ua.rada.open-data.laws-texts",
    "github:encode/httpx",
    "github:psf/requests",
    "python.cpython.documentation",
    "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
}
EXPECTED_FAMILY_COUNTS = {"uk": 2, "en": 2, "code": 2}
EXPECTED_BOUND_BYTES = {"uk": 90044, "en": 102694, "code": 9703, "total": 202441}


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"DATA-526 validation failed: {message}")


def validate(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed_manifest_id = manifest.pop("manifest_identity_sha256", None)
    computed_manifest_id = hashlib.sha256(canonical(manifest)).hexdigest()
    require(claimed_manifest_id == EXPECTED_MANIFEST_ID, "unexpected manifest identity")
    require(computed_manifest_id == claimed_manifest_id, "manifest self-hash drift")

    require(
        manifest["schema_version"]
        == "12-6.data526-research-corpus-v1-predecontamination-intake.v1",
        "schema drift",
    )
    require(
        manifest["worker_id"]
        == "DATA-526-RESEARCH-CORPUS-V1-PREDECONTAMINATION-INTAKE",
        "worker drift",
    )
    require(manifest["execution_profile"] == "LOCAL_FREE", "LOCAL_FREE boundary weakened")
    require(manifest["issue"] == 526, "issue binding drift")

    authorities = manifest["authority_vector"]
    require(
        authorities["data300_contract"]["contract_identity_sha256"]
        == EXPECTED_DATA300_CONTRACT,
        "DATA-300 contract drift",
    )
    require(
        authorities["data301_terminal_blocker"]["head_sha"] == EXPECTED_DATA301_HEAD,
        "DATA-301 head drift",
    )
    require(
        authorities["data301_terminal_blocker"]["state"] == "TERMINAL_BLOCKED",
        "DATA-301 blocker state drift",
    )
    require(
        authorities["data301_terminal_blocker"]
        ["authorized_balanced_no_replay_capacity"]
        == 0,
        "stale DATA-301 capacity was silently promoted",
    )

    cp_authority = authorities["next100_037_cpython_docs"]
    require(
        cp_authority["head_sha"] == EXPECTED_CPYTHON_HEAD,
        "CPython authority head drift",
    )
    require(
        cp_authority["authority_identity_sha256"] == EXPECTED_CPYTHON_AUTHORITY,
        "CPython authority identity drift",
    )
    require(cp_authority["terminal_verdict"] == "ADMIT", "CPython is not terminal ADMIT")

    ua_authority = authorities["next100_022_ua_wikisource"]
    require(
        ua_authority["head_sha"] == EXPECTED_WIKISOURCE_HEAD,
        "UA Wikisource authority head drift",
    )
    require(
        ua_authority["authority_identity_sha256"] == EXPECTED_WIKISOURCE_AUTHORITY,
        "UA Wikisource authority identity drift",
    )
    require(
        ua_authority["authority_state"] == "TERMINAL_RIGHTS_AND_IMMUTABLE_SNAPSHOT",
        "UA Wikisource authority state drift",
    )

    decontam = authorities["next100_066_reserved_decontamination"]
    require(decontam["head_sha"] == EXPECTED_DECONTAM_HEAD, "decontamination head drift")
    require(
        decontam["state"] == "BLOCKED_NO_EXACT_CANDIDATE_CORPUS_IDENTITY",
        "predecessor decontamination state no longer matches this handoff",
    )

    candidate = manifest["candidate_set"]
    require(
        candidate["state"]
        == "FROZEN_PRE_DECONTAMINATION_CANDIDATE_SET_NOT_TRAINING_CORPUS",
        "candidate set promoted beyond intake scope",
    )
    require(candidate["identity_sha256"] == EXPECTED_CANDIDATE_ID, "candidate identity drift")
    candidate_identity_payload = {
        "base_contract_identity_sha256": authorities["data300_contract"]
        ["contract_identity_sha256"],
        "base_data301_head": authorities["data301_terminal_blocker"]["head_sha"],
        "sources": candidate["sources"],
    }
    computed_candidate_id = hashlib.sha256(canonical(candidate_identity_payload)).hexdigest()
    require(computed_candidate_id == candidate["identity_sha256"], "candidate payload hash drift")
    require(candidate["source_object_count"] == 7, "source object count drift")
    require(candidate["incumbent_source_object_count"] == 5, "incumbent count drift")
    require(candidate["successor_source_authority_count"] == 2, "successor count drift")

    sources = candidate["sources"]
    source_ids = [item["source_id"] for item in sources]
    require(len(source_ids) == len(set(source_ids)) == 7, "source IDs are not exact/unique")
    require(set(source_ids) == EXPECTED_SOURCE_IDS, "source ID set drift")
    require({item["family"] for item in sources} == EXPECTED_FAMILIES, "family set drift")

    family_counts: dict[str, int] = {}
    for stratum in ("uk", "en", "code"):
        family_counts[stratum] = len(
            {item["family"] for item in sources if item["stratum"] == stratum}
        )
    require(family_counts == EXPECTED_FAMILY_COUNTS, "source-authority family counts drift")
    require(
        candidate["independent_family_counts_at_source_authority_level"]
        == EXPECTED_FAMILY_COUNTS,
        "declared family counts drift",
    )
    require(
        candidate["source_authority_family_diversity"]
        == "PASS_TWO_FAMILIES_PER_STRATUM",
        "source-authority diversity result drift",
    )
    require(
        candidate["release_balance_diversity"].startswith("BLOCKED_"),
        "source authority diversity was misused as corpus release authority",
    )

    bound_bytes = candidate["authority_bound_normalized_source_bytes"]
    for key, expected in EXPECTED_BOUND_BYTES.items():
        require(bound_bytes[key] == expected, f"authority-bound bytes drift: {key}")
    require(
        sum(item["normalized_bytes"] for item in sources) == EXPECTED_BOUND_BYTES["total"],
        "source-level normalized byte total drift",
    )

    cp = next(
        item for item in sources if item["source_id"] == "en.python.docs.tutorial-introduction"
    )
    require(cp["authority_head"] == EXPECTED_CPYTHON_HEAD, "CPython source binding drift")
    require(
        cp["authority_identity_sha256"] == EXPECTED_CPYTHON_AUTHORITY,
        "CPython source authority drift",
    )
    require(cp["accepted_chunk_count"] == 14, "CPython accepted chunk count drift")
    require(cp["rejected_chunk_count"] == 2, "CPython rejected chunk count drift")
    require(
        len(cp["accepted_normalized_sha256"]) == 14
        and len(set(cp["accepted_normalized_sha256"])) == 14,
        "CPython accepted chunk hashes invalid",
    )
    require(
        cp["eligible_normalized_bytes"] is None,
        "unknown CPython eligible byte count fabricated",
    )
    require(
        cp["training_rights"] == "ALLOWED_ACCEPTED_CHUNKS_ONLY",
        "CPython rejected chunks leaked into training scope",
    )

    ua = next(
        item
        for item in sources
        if item["source_id"]
        == "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13"
    )
    require(ua["authority_head"] == EXPECTED_WIKISOURCE_HEAD, "UA source binding drift")
    require(
        ua["authority_identity_sha256"] == EXPECTED_WIKISOURCE_AUTHORITY,
        "UA source authority drift",
    )
    require(
        ua["near_match_decontamination"] == "REQUIRED_BEFORE_TRAINING_SELECTION",
        "UA near-match gate weakened",
    )

    for source in sources:
        require(
            source["evaluation_rights"] == "NOT_SEPARATELY_ADMITTED",
            "evaluation rights leaked",
        )

    eligibility = manifest["eligibility"]
    require(
        eligibility["authorized_unique_optimized_targets"] == 0,
        "training capacity fabricated",
    )
    require(eligibility["corpus_identity"] is None, "corpus identity fabricated")
    require(eligibility["shard_identity"] is None, "shard identity fabricated")
    require(eligibility["long_training_authorized"] is False, "long training authorized")
    require(eligibility["training_executed"] is False, "training execution claimed")
    require(eligibility["paid_compute_used"] is False, "paid compute claimed")
    require(
        eligibility["replay_or_replacement_sampling_allowed"] is False,
        "replay allowed",
    )
    require(
        eligibility["candidate_set_may_update_model"] is False,
        "candidate intake may update model",
    )
    require(
        eligibility["candidate_set_may_fit_tokenizer"] is False,
        "candidate intake may fit tokenizer",
    )

    gates = manifest["gates"]
    require(
        gates["G01_TERMINAL_SOURCE_AUTHORITY_COMPOSITION"] == "PASS",
        "composition gate drift",
    )
    require(
        gates["G02_EXACT_PREDECONTAMINATION_CANDIDATE_SET_IDENTITY"] == "PASS",
        "candidate identity gate drift",
    )
    require(
        gates["G05_RESERVED_EXACT_NEAR_MATCH_DECONTAMINATION"].startswith(
            "NEXT_ACTION_BLOCKED_"
        ),
        "decontamination was falsely claimed complete",
    )
    require(
        gates["G12_REAL_20M_CAMPAIGN"]
        == "BLOCKED_ZERO_AUTHORIZED_UNIQUE_OPTIMIZED_TARGETS",
        "20M campaign gate weakened",
    )

    handoff = manifest["handoff"]
    require(
        handoff["next_action"]
        == "RUN_STANDARD_EXACT_NEAR_MATCH_RESERVED_EVALUATION_DECONTAMINATION",
        "next-action ordering drift",
    )
    require(
        handoff["required_candidate_set_identity_sha256"] == EXPECTED_CANDIDATE_ID,
        "handoff candidate identity drift",
    )
    require(handoff["may_add_sources"] is False, "handoff permits silent source additions")

    boundary = manifest["truth_boundary"]
    for key in (
        "research_corpus_v1_terminal",
        "learned_20m_exists",
        "meaningful_20m_campaign_runnable",
        "primary_20m_mechanics_identity_changed",
        "this_change_modifies_model_or_trainer",
    ):
        require(boundary[key] is False, f"truth boundary weakened: {key}")

    return {
        "status": "PASS",
        "manifest_identity_sha256": claimed_manifest_id,
        "candidate_set_identity_sha256": candidate["identity_sha256"],
        "source_object_count": len(sources),
        "family_counts": family_counts,
        "authority_bound_normalized_source_bytes": EXPECTED_BOUND_BYTES["total"],
        "authorized_unique_optimized_targets": 0,
        "next_action": handoff["next_action"],
        "local_free_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=(
            "configs/data/"
            "data526_research_corpus_v1_predecontamination_intake_v1.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.path)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
