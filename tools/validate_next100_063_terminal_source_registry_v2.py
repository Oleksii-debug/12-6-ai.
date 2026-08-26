#!/usr/bin/env python3
"""Fail-closed validator for the NEXT100-063 terminal source registry V2."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v2.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SCHEMA = "12-6.next100-063-terminal-source-registry.v2"
EXPECTED_WORKER = "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V2"
EXPECTED_DECISION = "CONVERGED_FAIL_CLOSED_TERMINAL_SOURCE_VECTOR_PRE_GLOBAL_DEDUP_NOT_CORPUS_FREEZE"
EXPECTED_REGISTRY_IDENTITY = "934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d"
EXPECTED_V1_IDENTITY = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"
EXPECTED_BASE_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_BASE_IDENTITY = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"

EXPECTED_POLICY = {
    "only_terminal_admit_authorities_counted": True,
    "dedicated_exact_head_source_workflow_must_succeed": True,
    "generic_workflow_success_cannot_substitute": True,
    "quality_rejected_subrecords_must_not_receive_capacity_credit": True,
    "source_level_bytes_cannot_substitute_for_missing_eligible_subrecord_ledger": True,
    "evaluation_permission_never_inferred_from_training_permission": True,
    "one_independent_family_credit_per_canonical_lineage": True,
    "global_cross_source_dedup_required_before_corpus_identity": True,
    "decontamination_required_before_corpus_identity": True,
    "parallel_retest_or_queued_candidates_counted": False,
}

# head, authority identity, family, bytes, training, evaluation, workflow run,
# workflow name. These are exact inputs to this V2 registry, not discoverable
# aliases that may drift without a successor registry identity.
EXPECTED_TERMINAL_AUTHORITIES = {
    449: (
        "40950a950b60921fd856af2719e1ae2486d9e892",
        "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        "ua.kmu.portal.secretariat-news",
        9153,
        "ALLOWED_PRETRAINING",
        "NOT_SEPARATELY_ADMITTED",
        32997970539,
        "NEXT100-026 KMu Source Rights Audit",
    ),
    455: (
        "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
        "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
        "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
        1479,
        "ALLOWED",
        "NOT_SEPARATELY_ADMITTED",
        32998002424,
        "NEXT100-022 Ukrainian Wikisource Qualification",
    ),
    462: (
        "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
        "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        "ua.verba.public-domain.nomis1864",
        1659,
        "ALLOWED",
        "NOT_ADMITTED",
        32998503672,
        "NEXT100-027 Ukrainian public-domain literature",
    ),
    445: (
        "902eccc0b3efff09a38dc89cda789180b6c6e754",
        "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        "en.mdn.webdocs.prose",
        6492,
        "ALLOWED",
        "NOT_SEPARATELY_ADMITTED",
        32998544359,
        "NEXT100-038 MDN Source Authority",
    ),
    472: (
        "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        "en.usgov.nist.technical-series",
        59358,
        "ALLOWED_WITH_NIST_SOURCE_PROVENANCE",
        "NOT_SEPARATELY_ADMITTED",
        32998703545,
        "NEXT100-034 NIST authority",
    ),
    458: (
        "c6756b5ebb6eb1d3bf3de2499167833d99d99a72",
        "c6b210c8977cce4441134ef048ed7dbea1a1e74b295ee96ce70ce5d612962722",
        "github:Kludex/starlette",
        5274,
        "ALLOWED",
        "NOT_AUTHORIZED_BY_THIS_AUTHORITY",
        32998101312,
        "NEXT100-045 Starlette Code Source Admission",
    ),
}

EXPECTED_REQUIRED_HELD_OUT = {
    467: {
        "family": "python.cpython.documentation",
        "source_normalized_bytes_not_capacity_credit": 17901,
        "accepted_chunk_count": 14,
        "rejected_chunk_count": 2,
        "dedicated_workflow_run": 32998356906,
        "dedicated_workflow_conclusion": "success",
    },
    465: {
        "family": "github:pydantic/pydantic",
        "claimed_normalized_bytes_not_credited": 235204,
        "dedicated_workflow_run": 32999061340,
        "dedicated_workflow_conclusion": "failure",
    },
    475: {
        "family": "github:Textualize/rich",
        "claimed_normalized_bytes_not_credited": 46162,
        "dedicated_workflow_run": 32999511493,
        "dedicated_workflow_conclusion": "failure",
    },
}

EXPECTED_DOWNSTREAM = {
    "source_registry_convergence": "PASS_FAIL_CLOSED_CANDIDATE_AUTHORITY_VECTOR",
    "global_cross_source_exact_near_dedup": "REQUIRED_NEXT",
    "evaluation_decontamination": "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY",
    "quality_privacy_revalidation": "REQUIRED_AFTER_COMPOSITION",
    "split_shard_pack_materialization": "BLOCKED_UNTIL_DEDUP_DECONTAMINATION",
    "authorized_balanced_no_replay_loss_positions": 0,
    "tokenizer_fit": "BLOCKED",
    "long_training": "BLOCKED",
    "paid_compute": "NOT_AUTHORIZED",
}

EXPECTED_CLAIM_KEYS = {
    "research_corpus_v1_frozen",
    "corpus_identity_claimed",
    "post_dedup_capacity_claimed",
    "decontamination_pass_claimed",
    "tokenizer_fit_executed",
    "model_training_executed",
    "learned_20m_checkpoint_claimed",
    "learned_100m_checkpoint_claimed",
}


class RegistryValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryValidationError(message)


def _canonical_identity(data: Mapping[str, Any]) -> str:
    payload = dict(data)
    payload.pop("registry_identity_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_registry(
    data: Mapping[str, Any],
    *,
    expected_registry_identity: str = EXPECTED_REGISTRY_IDENTITY,
) -> dict[str, Any]:
    _require(data.get("schema_version") == EXPECTED_SCHEMA, "unexpected V2 schema")
    _require(data.get("worker_id") == EXPECTED_WORKER, "unexpected V2 worker")
    _require(data.get("decision") == EXPECTED_DECISION, "decision boundary drift")

    identity = data.get("registry_identity_sha256")
    _require(isinstance(identity, str) and HEX64.fullmatch(identity) is not None, "invalid registry identity")
    _require(_canonical_identity(data) == identity, "registry identity mismatch")
    _require(identity == expected_registry_identity, "unexpected V2 registry identity")

    supersedes = data.get("supersedes")
    _require(isinstance(supersedes, Mapping), "supersedes must be an object")
    _require(
        supersedes.get("registry_identity_sha256") == EXPECTED_V1_IDENTITY,
        "V2 must bind the superseded V1 identity",
    )
    _require(
        isinstance(supersedes.get("correction"), str) and bool(supersedes["correction"].strip()),
        "V2 correction rationale must be explicit",
    )

    _require(data.get("composition_policy") == EXPECTED_POLICY, "composition policy drift")

    base = data.get("base_registry")
    _require(isinstance(base, Mapping), "base_registry must be an object")
    _require(base.get("head_sha") == EXPECTED_BASE_HEAD, "unexpected DATA-287 base head")
    _require(
        base.get("registry_identity_sha256") == EXPECTED_BASE_IDENTITY,
        "unexpected DATA-287 base identity",
    )

    rows = data.get("terminal_additions")
    _require(isinstance(rows, list), "terminal_additions must be a list")
    _require({row.get("pr") for row in rows if isinstance(row, Mapping)} == set(EXPECTED_TERMINAL_AUTHORITIES), "terminal PR set drift")

    counted_prs: set[int] = set()
    counted_heads: set[str] = set()
    families: set[str] = set(base.get("families", []))
    by = {
        "uk": dict(base["by_stratum"]["uk"]),
        "en": dict(base["by_stratum"]["en"]),
        "code": dict(base["by_stratum"]["code"]),
    }
    new_bytes = 0

    for row in rows:
        _require(isinstance(row, Mapping), "terminal addition must be an object")
        pr = row.get("pr")
        _require(isinstance(pr, int) and not isinstance(pr, bool) and pr > 0, "invalid terminal PR")
        _require(pr not in counted_prs, f"duplicate terminal PR {pr}")
        head = row.get("head")
        authority_identity = row.get("authority_identity")
        family = row.get("family")
        _require(isinstance(head, str) and HEX40.fullmatch(head) is not None, f"invalid head for PR {pr}")
        _require(isinstance(authority_identity, str) and HEX64.fullmatch(authority_identity) is not None, f"invalid authority identity for PR {pr}")
        _require(isinstance(family, str) and bool(family), f"invalid family for PR {pr}")
        _require(head not in counted_heads, f"duplicate terminal head {head}")
        _require(family not in families, f"duplicate independent family {family}")

        normalized_bytes = row.get("normalized_bytes")
        _require(
            isinstance(normalized_bytes, int)
            and not isinstance(normalized_bytes, bool)
            and normalized_bytes > 0,
            f"invalid credited bytes for PR {pr}",
        )
        training = row.get("training")
        evaluation = row.get("evaluation")
        _require(isinstance(training, str) and training.startswith("ALLOWED"), f"training permission absent for PR {pr}")
        _require(isinstance(evaluation, str) and evaluation.startswith("NOT_"), f"evaluation permission must remain denied for PR {pr}")
        _require(isinstance(row.get("verdict"), str) and row["verdict"].startswith("ADMIT"), f"non-ADMIT verdict for PR {pr}")
        _require(row.get("dedicated_workflow_conclusion") == "success", f"source workflow not terminal-success for PR {pr}")

        expected = EXPECTED_TERMINAL_AUTHORITIES[pr]
        observed = (
            head,
            authority_identity,
            family,
            normalized_bytes,
            training,
            evaluation,
            row.get("dedicated_workflow_run"),
            row.get("dedicated_workflow_name"),
        )
        _require(observed == expected, f"exact source authority drift for PR {pr}")

        key = "code" if row.get("modality") == "code" else row.get("language")
        _require(key in by, f"unsupported stratum for PR {pr}")
        by[key]["normalized_bytes"] += normalized_bytes
        by[key]["family_count"] += 1
        new_bytes += normalized_bytes
        counted_prs.add(pr)
        counted_heads.add(head)
        families.add(family)

    held_out = data.get("held_out_or_noncomposable")
    _require(isinstance(held_out, list), "held_out_or_noncomposable must be a list")
    held_map: dict[int, Mapping[str, Any]] = {}
    for item in held_out:
        _require(isinstance(item, Mapping), "held-out row must be an object")
        pr = item.get("pr")
        _require(isinstance(pr, int) and not isinstance(pr, bool) and pr > 0, "invalid held-out PR")
        _require(pr not in held_map, f"duplicate held-out PR {pr}")
        _require(pr not in counted_prs, f"PR {pr} cannot be both counted and held out")
        held_map[pr] = item

    for pr, required in EXPECTED_REQUIRED_HELD_OUT.items():
        _require(pr in held_map, f"required held-out authority missing: PR {pr}")
        item = held_map[pr]
        for key, value in required.items():
            _require(item.get(key) == value, f"held-out boundary drift for PR {pr}: {key}")

    inv = data.get("pre_global_dedup_inventory")
    _require(isinstance(inv, Mapping), "pre_global_dedup_inventory must be an object")
    total = int(base["unique_normalized_bytes"]) + new_bytes
    _require(new_bytes == 83415, "V2 credited-byte total drift")
    _require(inv.get("new_terminal_normalized_bytes") == new_bytes, "new-byte accounting drift")
    _require(inv.get("candidate_normalized_bytes") == total, "candidate-byte accounting drift")
    _require(inv.get("candidate_source_authority_count") == int(base["source_count"]) + len(rows), "candidate source-count drift")
    _require(inv.get("candidate_independent_family_count") == len(families), "candidate family-count drift")
    _require(inv.get("by_stratum") == by, "stratum accounting drift")
    minimum = inv.get("minimum_independent_families_per_stratum")
    _require(isinstance(minimum, int) and not isinstance(minimum, bool) and minimum >= 2, "family minimum weakened")
    _require(all(bucket["family_count"] >= minimum for bucket in by.values()), "family minimum cannot pass")
    _require(inv.get("family_minimum_gate") == "PASS_PRE_GLOBAL_DEDUP", "family gate verdict drift")
    target = inv.get("research_corpus_v1_target_normalized_bytes")
    _require(isinstance(target, int) and not isinstance(target, bool) and target > total, "invalid acquisition target")
    _require(inv.get("target_gap_normalized_bytes") == target - total, "target-gap drift")
    _require(abs(float(inv.get("target_fraction")) - (total / target)) <= 1e-12, "target-fraction drift")

    _require(data.get("downstream_gate_vector") == EXPECTED_DOWNSTREAM, "downstream gate vector weakened or drifted")
    claims = data.get("claim_boundary")
    _require(isinstance(claims, Mapping), "claim_boundary must be an object")
    _require(set(claims) == EXPECTED_CLAIM_KEYS, "claim-boundary key set drift")
    _require(all(value is False for value in claims.values()), "premature corpus/training claim detected")

    return {
        "registry_identity_sha256": identity,
        "terminal_additions": len(rows),
        "credited_normalized_bytes": total,
        "new_credited_normalized_bytes": new_bytes,
        "independent_families": len(families),
        "held_out_authorities": len(held_out),
        "authorized_balanced_no_replay_loss_positions": 0,
        "long_training": "BLOCKED",
        "paid_compute": "NOT_AUTHORIZED",
    }


def load_registry(path: str | Path = PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RegistryValidationError("registry root must be an object")
    return data


def main() -> int:
    try:
        report = validate_registry(load_registry())
    except RegistryValidationError as exc:
        print(f"NEXT100-063-V2 FAIL: {exc}")
        return 1
    print("NEXT100-063-V2 PASS " + json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
