#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-063 terminal source registry V3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v3.json"
EXPECTED_IDENTITY = "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c"
EXPECTED_PARENT = {
    "worker": "NEXT100-065-CROSSSOURCE-DEDUP-V3",
    "head_sha": "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13",
    "config_blob_sha1": "c1e05f09490e25f6fed765dfb70d900717528f4d",
    "dedicated_workflow_run": 32999969398,
    "dedicated_workflow_conclusion": "success",
    "source_object_count": 11,
    "independent_family_count": 7,
    "numeric_training_capacity_bytes": 243970,
}
EXPECTED_LATE = {
    449: ("40950a950b60921fd856af2719e1ae2486d9e892", "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9", 32997970539, "NEXT100-026 KMu Source Rights Audit", "ua.kmu.portal.secretariat-news", 9153, 9153),
    462: ("d75edd497c7fb1054e86d892c9462f059c1f4aa9", "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7", 32998503672, "NEXT100-027 Ukrainian public-domain literature", "ua.verba.public-domain.nomis1864", 1659, 1659),
    445: ("902eccc0b3efff09a38dc89cda789180b6c6e754", "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47", 32998544359, "NEXT100-038 MDN Source Authority", "en.mdn.webdocs.prose", 6492, 6492),
    472: ("b7491745b34ac8679baaf69cb96cd609dcbe0a16", "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c", 32998703545, "NEXT100-034 NIST authority", "en.usgov.nist.technical-series", 59358, 59358),
    468: ("bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8", "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70", 32998548535, "NEXT100-049 NumPy Code Source Authority", "github:numpy/numpy", 36898, 36898),
    467: ("5a6a495a24bce449334cbc5126d0114f61a9f57c", "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d", 32998356906, "NEXT100-037 Python Docs Source Authority", "python.cpython.documentation", 0, 17901),
}
EXPECTED_FAILED = {
    465: ("ca1755886f052d272029d6d68b2f1b7f02187936", "32999061340"),
    475: ("78cada1d69b3f0c438012c4e6cf79143aae2f603", "32999511493"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-063 V3 FAIL: {message}")


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    claimed = data.get("registry_identity_sha256")
    body = dict(data)
    body.pop("registry_identity_sha256", None)
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(claimed == actual == EXPECTED_IDENTITY, "registry identity drift")
    require(data.get("schema_version") == "12-6.next100-063-terminal-source-registry.v3", "schema drift")
    require(data.get("worker_id") == "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V3", "worker drift")

    supersedes = data.get("supersedes", {})
    require(supersedes.get("v1_registry_identity_sha256") == "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526", "V1 binding drift")
    require(supersedes.get("v2_registry_identity_sha256") == "448dd61ed3e0d78d0bca9e202529a79c02811fd67beebe4833373d0c2ab0c0a7", "V2 binding drift")
    require(supersedes.get("v2_replaced_draft_registry_identity_sha256") == "934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d", "draft V2 binding drift")

    parent = data.get("dedup_parent", {})
    for key, value in EXPECTED_PARENT.items():
        require(parent.get(key) == value, f"dedup parent drift: {key}")
    require(sum(x["numeric_training_capacity_bytes"] for x in parent["by_stratum"].values()) == 243970, "parent byte arithmetic drift")
    require(sum(x["family_count"] for x in parent["by_stratum"].values()) == 7, "parent family arithmetic drift")
    require(len(set(parent["families"])) == 7, "parent family identity drift")

    rows = data.get("terminal_late_additions", [])
    require({row.get("pr") for row in rows} == set(EXPECTED_LATE), "late authority set drift")
    families = set(parent["families"])
    heads: set[str] = set()
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
        expected = EXPECTED_LATE[pr]
        observed = (
            row.get("head"), row.get("authority_identity"), row.get("dedicated_workflow_run"),
            row.get("dedicated_workflow_name"), row.get("family"),
            row.get("numeric_training_capacity_bytes"), row.get("source_normalized_bytes")
        )
        require(observed == expected, f"PR {pr} exact authority/capacity drift")
        require(row.get("dedicated_workflow_conclusion") == "success", f"PR {pr} dedicated workflow not successful")
        require(str(row.get("verdict", "")).startswith("ADMIT"), f"PR {pr} not admitted")
        require(str(row.get("training", "")).startswith("ALLOWED"), f"PR {pr} lacks training permission")
        require(str(row.get("evaluation", "")).startswith("NOT_"), f"PR {pr} evaluation permission leaked")
        require(row["head"] not in heads, f"duplicate late head: PR {pr}")
        require(row["family"] not in families, f"duplicate family: PR {pr}")
        heads.add(row["head"])
        families.add(row["family"])
        numeric = row["numeric_training_capacity_bytes"]
        normalized = row["source_normalized_bytes"]
        require(normalized > 0 and 0 <= numeric <= normalized, f"PR {pr} invalid capacity relationship")
        if pr == 467:
            require(numeric == 0, "CPython source bytes promoted before accepted-chunk ledger")
            require(row.get("accepted_chunk_count") == 14 and row.get("rejected_chunk_count") == 2, "CPython chunk boundary drift")
        else:
            require(numeric == normalized, f"PR {pr} unexplained capacity discount")
        if pr == 468:
            require(row.get("terminal_artifact_id") == 9618015895, "NumPy artifact id drift")
            require(row.get("terminal_artifact_zip_sha256") == "402016760c2ea5b341ed15537bb173e9bf10a938870313f00fd5e617ba20b020", "NumPy artifact digest drift")
        numeric_add += numeric
        normalized_add += normalized
        stratum = "code" if row.get("modality") == "code" else row.get("language")
        require(stratum in by, f"PR {pr} unsupported stratum")
        by[stratum]["family_count"] += 1
        by[stratum]["numeric_training_capacity_bytes"] += numeric
        by[stratum]["source_normalized_envelope_bytes"] += normalized

    inv = data.get("pre_successor_global_dedup_inventory", {})
    numeric_total = 243970 + numeric_add
    normalized_total = 243970 + normalized_add
    require(numeric_add == 113560, "late numeric byte arithmetic drift")
    require(numeric_total == 357530, "candidate numeric capacity drift")
    require(normalized_total == 375431, "normalized envelope drift")
    require(inv.get("candidate_numeric_training_capacity_bytes") == numeric_total, "inventory numeric capacity drift")
    require(inv.get("candidate_source_normalized_envelope_bytes") == normalized_total, "inventory normalized envelope drift")
    require(inv.get("uncredited_source_normalized_bytes") == normalized_total - numeric_total == 17901, "uncredited-byte drift")
    require(inv.get("candidate_independent_family_count") == len(families) == 13, "family count drift")
    require(inv.get("by_stratum") == by, "stratum vector drift")
    require(all(v["family_count"] >= 2 for v in by.values()), "family floor not met at authority layer")
    require(inv.get("family_minimum_gate") == "PASS_AUTHORITY_LAYER_PRE_SUCCESSOR_DEDUP", "family gate drift")
    target = inv.get("research_corpus_v1_acquisition_planning_target_bytes")
    require(target == 20_000_000, "acquisition target drift")
    require(inv.get("target_gap_numeric_training_capacity_bytes") == target - numeric_total == 19_642_470, "target gap drift")

    held = {row.get("pr"): row for row in data.get("held_out_or_zero_credit", []) if isinstance(row, dict)}
    require(468 not in held, "NumPy simultaneously credited and held out")
    for pr, (head, run) in EXPECTED_FAILED.items():
        row = held.get(pr, {})
        require(row.get("head") == head, f"failed PR {pr} head drift")
        require(run in row.get("reason", "") and "COMPLETED_FAILURE" in row.get("reason", ""), f"failed PR {pr} gained unsafe credit")

    policy = data.get("composition_policy", {})
    require(policy.get("dedup_parent_is_exact_green_authority") is True, "dedup-parent authority weakened")
    require(policy.get("only_exact_head_scoped_success_authorities_counted") is True, "scoped-success rule weakened")
    require(policy.get("source_normalized_bytes_are_not_eligible_capacity_without_post_filter_materialization") is True, "source/capacity firewall weakened")
    require(policy.get("failed_queued_retest_or_pr_text_only_candidates_counted") is False, "failed/nonterminal source credit enabled")

    gates = data.get("downstream_gate_vector", {})
    require(gates.get("authorized_balanced_no_replay_loss_positions") == 0, "training exposure must remain zero")
    require(gates.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "successor dedup gate weakened")
    require(gates.get("evaluation_decontamination") == "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY", "decontamination gate weakened")
    require(gates.get("long_training") == "BLOCKED", "long training promoted")
    require(gates.get("paid_compute") == "NOT_AUTHORIZED", "paid compute promoted")

    print(
        "NEXT100-063 V3 PASS "
        f"identity={actual} numeric_capacity_bytes={numeric_total} "
        f"normalized_envelope_bytes={normalized_total} families={len(families)}"
    )


if __name__ == "__main__":
    main()
