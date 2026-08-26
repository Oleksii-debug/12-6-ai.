#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SCHEMA = "12-6.next100-063-terminal-source-registry.v2"
EXPECTED_WORKER = "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE"
EXPECTED_DECISION = "CONVERGED_TERMINAL_SOURCE_AUTHORITY_VECTOR_REQUIRES_SUCCESSOR_GLOBAL_DEDUP_NOT_CORPUS_FREEZE"
EXPECTED_REGISTRY_IDENTITY = "56ab7c1e3ebf336c0ec9f51d8580f9aaad2890566d1bf7a5090e2b7f0102bfc5"
EXPECTED_BASE_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_BASE_IDENTITY = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_DEDUP_PARENT = {
    "head_sha": "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13",
    "config_blob_sha1": "c1e05f09490e25f6fed765dfb70d900717528f4d",
    "dedicated_workflow_run": 32999969398,
    "dedicated_workflow_conclusion": "success",
    "source_object_count": 11,
    "independent_family_count": 7,
    "numeric_training_capacity_bytes": 243970,
}

# Exact late authorities that may contribute source-family availability. Numeric
# capacity is separate: CPython's source authority is terminal, but only 14/16
# chunks are eligible and this authority deliberately gives them zero byte
# credit until an exact accepted-chunk byte ledger is terminal.
EXPECTED_LATE_AUTHORITIES = {
    449: (
        "40950a950b60921fd856af2719e1ae2486d9e892",
        "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        32997970539,
        "ua.kmu.portal.secretariat-news",
        9153,
        9153,
        "ALLOWED_PRETRAINING",
    ),
    462: (
        "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
        "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        32998503672,
        "ua.verba.public-domain.nomis1864",
        1659,
        1659,
        "ALLOWED",
    ),
    445: (
        "902eccc0b3efff09a38dc89cda789180b6c6e754",
        "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        32998544359,
        "en.mdn.webdocs.prose",
        6492,
        6492,
        "ALLOWED",
    ),
    472: (
        "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        32998703545,
        "en.usgov.nist.technical-series",
        59358,
        59358,
        "ALLOWED_WITH_NIST_SOURCE_PROVENANCE",
    ),
    467: (
        "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        32998356906,
        "python.cpython.documentation",
        0,
        17901,
        "ALLOWED_ACCEPTED_CHUNKS_ONLY",
    ),
}

EXPECTED_ZERO_CREDIT_FAILURES = {
    465: ("ca1755886f052d272029d6d68b2f1b7f02187936", "32999061340"),
    475: ("78cada1d69b3f0c438012c4e6cf79143aae2f603", "32999511493"),
}


def fail(msg: str) -> None:
    raise SystemExit(f"NEXT100-063 FAIL: {msg}")


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    expected_identity = data.pop("registry_identity_sha256", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    actual_identity = hashlib.sha256(canonical).hexdigest()
    if expected_identity != actual_identity:
        fail("registry identity mismatch")
    if expected_identity != EXPECTED_REGISTRY_IDENTITY:
        fail("unexpected terminal registry identity")
    if data.get("schema_version") != EXPECTED_SCHEMA:
        fail("unexpected registry schema")
    if data.get("worker_id") != EXPECTED_WORKER:
        fail("unexpected worker identity")
    if data.get("decision") != EXPECTED_DECISION:
        fail("decision boundary drift")

    base = data["base_registry"]
    if base.get("head_sha") != EXPECTED_BASE_HEAD:
        fail("unexpected DATA-287 ancestry head")
    if base.get("registry_identity_sha256") != EXPECTED_BASE_IDENTITY:
        fail("unexpected DATA-287 ancestry identity")

    parent = data["dedup_parent"]
    for key, value in EXPECTED_DEDUP_PARENT.items():
        if parent.get(key) != value:
            fail(f"NEXT100-065 parent drift: {key}")
    if parent.get("worker") != "NEXT100-065-CROSSSOURCE-DEDUP-V3":
        fail("unexpected dedup parent worker")
    if sum(v["numeric_training_capacity_bytes"] for v in parent["by_stratum"].values()) != parent["numeric_training_capacity_bytes"]:
        fail("dedup parent stratum byte arithmetic drift")
    if sum(v["family_count"] for v in parent["by_stratum"].values()) != parent["independent_family_count"]:
        fail("dedup parent family arithmetic drift")
    if len(set(parent["families"])) != parent["independent_family_count"]:
        fail("dedup parent family identity drift")

    rows = data["terminal_late_additions"]
    row_prs = {row.get("pr") for row in rows}
    if row_prs != set(EXPECTED_LATE_AUTHORITIES):
        fail("late terminal authority PR set drift")

    prs = set()
    heads = set()
    families = set(parent["families"])
    numeric_add = 0
    normalized_add = 0
    by = {
        key: {
            "family_count": value["family_count"],
            "numeric_training_capacity_bytes": value["numeric_training_capacity_bytes"],
            "source_normalized_envelope_bytes": value["numeric_training_capacity_bytes"],
        }
        for key, value in parent["by_stratum"].items()
    }

    for row in rows:
        pr = row["pr"]
        if pr in prs:
            fail("duplicate PR")
        if row["head"] in heads:
            fail("duplicate late source head")
        if row["family"] in families:
            fail(f"duplicate independent family: {row['family']}")
        if not HEX40.fullmatch(row["head"]):
            fail(f"invalid head SHA for PR {pr}")
        if not HEX64.fullmatch(row["authority_identity"]):
            fail(f"invalid authority identity for PR {pr}")

        expected = EXPECTED_LATE_AUTHORITIES[pr]
        observed = (
            row["head"],
            row["authority_identity"],
            row["dedicated_workflow_run"],
            row["family"],
            row["numeric_training_capacity_bytes"],
            row["source_normalized_bytes"],
            row["training"],
        )
        if observed != expected:
            fail(f"late terminal authority provenance drift for PR {pr}")
        if not row["verdict"].startswith("ADMIT"):
            fail(f"non-admit row counted: PR {pr}")
        if row["source_normalized_bytes"] <= 0:
            fail(f"non-positive normalized source envelope: PR {pr}")
        if row["numeric_training_capacity_bytes"] < 0:
            fail(f"negative numeric capacity: PR {pr}")
        if not row["training"].startswith("ALLOWED"):
            fail(f"training permission absent: PR {pr}")
        if "ALLOWED" in row["evaluation"] or row["evaluation"] == "AUTHORIZED":
            fail(f"evaluation permission leaked from training authority: PR {pr}")

        if pr == 467:
            if row.get("accepted_chunk_count") != 14 or row.get("rejected_chunk_count") != 2:
                fail("CPython accepted/rejected chunk boundary drift")
            if row["numeric_training_capacity_bytes"] != 0:
                fail("CPython source bytes promoted before terminal accepted-chunk ledger")
        elif row["numeric_training_capacity_bytes"] != row["source_normalized_bytes"]:
            fail(f"unexpected capacity/source-envelope mismatch: PR {pr}")

        prs.add(pr)
        heads.add(row["head"])
        families.add(row["family"])
        numeric_add += row["numeric_training_capacity_bytes"]
        normalized_add += row["source_normalized_bytes"]
        key = "code" if row["modality"] == "code" else row["language"]
        if key not in by:
            fail(f"unsupported stratum: {key}")
        by[key]["family_count"] += 1
        by[key]["numeric_training_capacity_bytes"] += row["numeric_training_capacity_bytes"]
        by[key]["source_normalized_envelope_bytes"] += row["source_normalized_bytes"]

    inv = data["pre_successor_global_dedup_inventory"]
    numeric_total = parent["numeric_training_capacity_bytes"] + numeric_add
    normalized_total = parent["numeric_training_capacity_bytes"] + normalized_add
    if inv["late_numeric_training_capacity_bytes"] != numeric_add:
        fail("late numeric-capacity arithmetic drift")
    if inv["candidate_numeric_training_capacity_bytes"] != numeric_total:
        fail("candidate numeric-capacity arithmetic drift")
    if inv["candidate_source_normalized_envelope_bytes"] != normalized_total:
        fail("source-normalized envelope arithmetic drift")
    if inv["uncredited_source_normalized_bytes"] != normalized_total - numeric_total:
        fail("uncredited normalized-byte arithmetic drift")
    if inv["candidate_independent_family_count"] != len(families):
        fail("family-count drift")
    if inv["by_stratum"] != by:
        fail("stratum accounting drift")
    if any(v["family_count"] < inv["minimum_independent_families_per_stratum"] for v in by.values()):
        fail("family-minimum gate cannot pass at authority layer")
    if inv["family_minimum_gate"] != "PASS_AUTHORITY_LAYER_PRE_SUCCESSOR_DEDUP":
        fail("family-minimum verdict drift")
    target = inv["research_corpus_v1_acquisition_planning_target_bytes"]
    if inv["target_gap_numeric_training_capacity_bytes"] != target - numeric_total:
        fail("acquisition-planning target-gap drift")

    zero = {row["pr"]: row for row in data["held_out_or_zero_credit"] if row.get("pr") in EXPECTED_ZERO_CREDIT_FAILURES}
    if set(zero) != set(EXPECTED_ZERO_CREDIT_FAILURES):
        fail("failed-source zero-credit set drift")
    for pr, (head, run) in EXPECTED_ZERO_CREDIT_FAILURES.items():
        row = zero[pr]
        if row.get("head") != head or run not in row.get("reason", "") or "COMPLETED_FAILURE" not in row.get("reason", ""):
            fail(f"failed-source boundary drift for PR {pr}")

    policy = data["composition_policy"]
    if policy["failed_queued_retest_or_pr_text_only_candidates_counted"] is not False:
        fail("nonterminal/failed sources may not gain credit")
    if policy["source_normalized_bytes_are_not_eligible_capacity_without_post_filter_materialization"] is not True:
        fail("source-byte/capacity firewall weakened")

    gates = data["downstream_gate_vector"]
    if gates["authorized_balanced_no_replay_loss_positions"] != 0:
        fail("training exposure must remain zero")
    if gates["long_training"] != "BLOCKED":
        fail("long training must remain blocked")
    if gates["paid_compute"] != "NOT_AUTHORIZED":
        fail("paid compute boundary weakened")
    if gates["successor_global_cross_source_exact_near_dedup"] != "REQUIRED_NEXT":
        fail("successor dedup gate weakened")
    if gates["evaluation_decontamination"] != "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY":
        fail("decontamination gate weakened")

    print(
        "NEXT100-063 PASS "
        f"identity={actual_identity} late_authorities={len(rows)} "
        f"numeric_capacity_bytes={numeric_total} normalized_envelope_bytes={normalized_total} "
        f"families={len(families)}"
    )


if __name__ == "__main__":
    main()
