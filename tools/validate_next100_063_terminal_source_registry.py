#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SCHEMA = "12-6.next100-063-terminal-source-registry.v1"
EXPECTED_WORKER = "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE"
EXPECTED_DECISION = "CONVERGED_TERMINAL_SOURCE_VECTOR_PRE_GLOBAL_DEDUP_NOT_CORPUS_FREEZE"
EXPECTED_REGISTRY_IDENTITY = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
EXPECTED_BASE_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_BASE_IDENTITY = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"

# Exact terminal authority bindings.  A source row is not trusted merely because
# it is syntactically well formed or because the registry self-hash was
# recomputed after metadata drift.  The provenance-bearing fields below are
# immutable inputs to this convergence authority.
EXPECTED_TERMINAL_AUTHORITIES = {
    449: (
        "40950a950b60921fd856af2719e1ae2486d9e892",
        "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        "ua.kmu.portal.secretariat-news",
        9153,
        "ALLOWED_PRETRAINING",
    ),
    455: (
        "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
        "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
        "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
        1479,
        "ALLOWED",
    ),
    462: (
        "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
        "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        "ua.verba.public-domain.nomis1864",
        1659,
        "ALLOWED",
    ),
    445: (
        "902eccc0b3efff09a38dc89cda789180b6c6e754",
        "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        "en.mdn.webdocs.prose",
        6492,
        "ALLOWED",
    ),
    472: (
        "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        "en.usgov.nist.technical-series",
        59358,
        "ALLOWED_WITH_NIST_SOURCE_PROVENANCE",
    ),
    467: (
        "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        "python.cpython.documentation",
        17901,
        "ALLOWED",
    ),
    458: (
        "c6756b5ebb6eb1d3bf3de2499167833d99d99a72",
        "c6b210c8977cce4441134ef048ed7dbea1a1e74b295ee96ce70ce5d612962722",
        "github:Kludex/starlette",
        5274,
        "ALLOWED",
    ),
    465: (
        "ca1755886f052d272029d6d68b2f1b7f02187936",
        "a25e618f4e26dd7c0df643768ab867a7ae080ca6ad2e5a88bda89bc757ae183a",
        "github:pydantic/pydantic",
        235204,
        "ALLOWED",
    ),
    475: (
        "78cada1d69b3f0c438012c4e6cf79143aae2f603",
        "15459ca82352e5cc7d9e76266ef48cbe49a831a01623d89cc18e67b131327249",
        "github:Textualize/rich",
        46162,
        "ALLOWED",
    ),
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
        fail("unexpected DATA-287 parent head")
    if base.get("registry_identity_sha256") != EXPECTED_BASE_IDENTITY:
        fail("unexpected DATA-287 parent identity")

    rows = data["terminal_additions"]
    row_prs = {row.get("pr") for row in rows}
    if row_prs != set(EXPECTED_TERMINAL_AUTHORITIES):
        fail("terminal authority PR set drift")

    prs, heads, families = set(), set(), set(base["families"])
    new_bytes = 0
    by = {
        "uk": dict(base["by_stratum"]["uk"]),
        "en": dict(base["by_stratum"]["en"]),
        "code": dict(base["by_stratum"]["code"]),
    }
    for row in rows:
        pr = row["pr"]
        if pr in prs:
            fail("duplicate PR")
        if row["head"] in heads:
            fail("duplicate source head")
        if row["family"] in families:
            fail(f"duplicate independent family: {row['family']}")
        if not HEX40.fullmatch(row["head"]):
            fail(f"invalid head SHA for PR {pr}")
        if not HEX64.fullmatch(row["authority_identity"]):
            fail(f"invalid authority identity for PR {pr}")

        expected = EXPECTED_TERMINAL_AUTHORITIES[pr]
        observed = (
            row["head"],
            row["authority_identity"],
            row["family"],
            row["normalized_bytes"],
            row["training"],
        )
        if observed != expected:
            fail(f"terminal authority provenance drift for PR {pr}")

        if not row["verdict"].startswith("ADMIT"):
            fail(f"non-terminal-admit row counted: PR {pr}")
        if row["normalized_bytes"] <= 0:
            fail(f"non-positive capacity: PR {pr}")
        if not row["training"].startswith("ALLOWED"):
            fail(f"training permission absent: PR {pr}")
        if "ALLOWED" in row["evaluation"] or row["evaluation"] == "AUTHORIZED":
            fail(f"evaluation permission leaked from training authority: PR {pr}")

        prs.add(pr)
        heads.add(row["head"])
        families.add(row["family"])
        new_bytes += row["normalized_bytes"]
        key = "code" if row["modality"] == "code" else row["language"]
        if key not in by:
            fail(f"unsupported stratum: {key}")
        by[key]["normalized_bytes"] += row["normalized_bytes"]
        by[key]["family_count"] += 1

    inv = data["pre_global_dedup_inventory"]
    total = base["unique_normalized_bytes"] + new_bytes
    if inv["new_terminal_normalized_bytes"] != new_bytes:
        fail("new-byte total drift")
    if inv["candidate_normalized_bytes"] != total:
        fail("candidate-byte total drift")
    if inv["candidate_independent_family_count"] != len(families):
        fail("family-count drift")
    if inv["by_stratum"] != by:
        fail("stratum accounting drift")
    if any(v["family_count"] < inv["minimum_independent_families_per_stratum"] for v in by.values()):
        fail("family-minimum gate cannot be PASS")
    if inv["family_minimum_gate"] != "PASS_PRE_GLOBAL_DEDUP":
        fail("family-minimum verdict drift")
    target = inv["research_corpus_v1_target_normalized_bytes"]
    if inv["target_gap_normalized_bytes"] != target - total:
        fail("target-gap drift")

    gates = data["downstream_gate_vector"]
    if gates["authorized_balanced_no_replay_loss_positions"] != 0:
        fail("training exposure must remain zero")
    if gates["long_training"] != "BLOCKED":
        fail("long training must remain blocked")
    if gates["paid_compute"] != "NOT_AUTHORIZED":
        fail("paid compute boundary weakened")
    if gates["global_cross_source_exact_near_dedup"] != "REQUIRED_NEXT":
        fail("dedup gate weakened")
    if gates["evaluation_decontamination"] != "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY":
        fail("decontamination gate weakened")

    print(
        "NEXT100-063 PASS "
        f"identity={actual_identity} terminal_additions={len(rows)} "
        f"candidate_bytes={total} families={len(families)}"
    )


if __name__ == "__main__":
    main()
