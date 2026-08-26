#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

def fail(msg: str) -> None:
    raise SystemExit(f"NEXT100-063 FAIL: {msg}")

def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    expected_identity = data.pop("registry_identity_sha256", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    actual_identity = hashlib.sha256(canonical).hexdigest()
    if expected_identity != actual_identity:
        fail("registry identity mismatch")

    base = data["base_registry"]
    if base["registry_identity_sha256"] != "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c":
        fail("unexpected DATA-287 parent identity")

    rows = data["terminal_additions"]
    prs, heads, families = set(), set(), set(base["families"])
    new_bytes = 0
    by = {
        "uk": dict(base["by_stratum"]["uk"]),
        "en": dict(base["by_stratum"]["en"]),
        "code": dict(base["by_stratum"]["code"]),
    }
    for row in rows:
        if row["pr"] in prs: fail("duplicate PR")
        if row["head"] in heads: fail("duplicate source head")
        if row["family"] in families: fail(f"duplicate independent family: {row['family']}")
        if not HEX40.fullmatch(row["head"]): fail(f"invalid head SHA for PR {row['pr']}")
        if not HEX64.fullmatch(row["authority_identity"]): fail(f"invalid authority identity for PR {row['pr']}")
        if not row["verdict"].startswith("ADMIT"): fail(f"non-terminal-admit row counted: PR {row['pr']}")
        if row["normalized_bytes"] <= 0: fail(f"non-positive capacity: PR {row['pr']}")
        if "ALLOWED" in row["evaluation"] or "AUTHORIZED" == row["evaluation"]:
            fail(f"evaluation permission leaked from training authority: PR {row['pr']}")
        prs.add(row["pr"]); heads.add(row["head"]); families.add(row["family"])
        new_bytes += row["normalized_bytes"]
        key = "code" if row["modality"] == "code" else row["language"]
        if key not in by: fail(f"unsupported stratum: {key}")
        by[key]["normalized_bytes"] += row["normalized_bytes"]
        by[key]["family_count"] += 1

    inv = data["pre_global_dedup_inventory"]
    total = base["unique_normalized_bytes"] + new_bytes
    if inv["new_terminal_normalized_bytes"] != new_bytes: fail("new-byte total drift")
    if inv["candidate_normalized_bytes"] != total: fail("candidate-byte total drift")
    if inv["candidate_independent_family_count"] != len(families): fail("family-count drift")
    if inv["by_stratum"] != by: fail("stratum accounting drift")
    if any(v["family_count"] < inv["minimum_independent_families_per_stratum"] for v in by.values()):
        fail("family-minimum gate cannot be PASS")
    if inv["family_minimum_gate"] != "PASS_PRE_GLOBAL_DEDUP": fail("family-minimum verdict drift")
    target = inv["research_corpus_v1_target_normalized_bytes"]
    if inv["target_gap_normalized_bytes"] != target - total: fail("target-gap drift")

    gates = data["downstream_gate_vector"]
    if gates["authorized_balanced_no_replay_loss_positions"] != 0: fail("training exposure must remain zero")
    if gates["long_training"] != "BLOCKED": fail("long training must remain blocked")
    if gates["paid_compute"] != "NOT_AUTHORIZED": fail("paid compute boundary weakened")
    if gates["global_cross_source_exact_near_dedup"] != "REQUIRED_NEXT": fail("dedup gate weakened")
    if gates["evaluation_decontamination"] != "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY": fail("decontamination gate weakened")

    print(f"NEXT100-063 PASS identity={actual_identity} terminal_additions={len(rows)} candidate_bytes={total} families={len(families)}")

if __name__ == "__main__":
    main()
