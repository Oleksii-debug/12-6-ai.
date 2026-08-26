#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-063 terminal source registry V2."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v2.json"
SCHEMA = "12-6.next100-063-terminal-source-registry.v2"
WORKER = "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V2"
BASE_IDENTITY = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
V1_IDENTITY = "77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526"

EXPECTED_ROWS = {
    449: {
        "worker": "NEXT100-026-DATA-UA-CABINET-MINISTRY",
        "head": "40950a950b60921fd856af2719e1ae2486d9e892",
        "authority_identity": "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        "family": "ua.kmu.portal.secretariat-news",
        "normalized_bytes": 9153,
        "dedicated_workflow_run": 32997970539,
        "dedicated_workflow_name": "NEXT100-026 KMu Source Rights Audit",
    },
    455: {
        "worker": "NEXT100-022-DATA-UA-WIKISOURCE",
        "head": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
        "authority_identity": "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
        "family": "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
        "normalized_bytes": 1479,
        "dedicated_workflow_run": 32998002424,
        "dedicated_workflow_name": "NEXT100-022 Ukrainian Wikisource Qualification",
    },
    462: {
        "worker": "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT",
        "head": "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
        "authority_identity": "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        "family": "ua.verba.public-domain.nomis1864",
        "normalized_bytes": 1659,
        "dedicated_workflow_run": 32998503672,
        "dedicated_workflow_name": "NEXT100-027 Ukrainian public-domain literature",
    },
    445: {
        "worker": "NEXT100-038-DATA-EN-MDN",
        "head": "902eccc0b3efff09a38dc89cda789180b6c6e754",
        "authority_identity": "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        "family": "en.mdn.webdocs.prose",
        "normalized_bytes": 6492,
        "dedicated_workflow_run": 32998544359,
        "dedicated_workflow_name": "NEXT100-038 MDN Source Authority",
    },
    472: {
        "worker": "NEXT100-034-DATA-EN-NIST",
        "head": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "authority_identity": "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        "family": "en.usgov.nist.technical-series",
        "normalized_bytes": 59358,
        "dedicated_workflow_run": 32998703545,
        "dedicated_workflow_name": "NEXT100-034 NIST authority",
    },
    458: {
        "worker": "NEXT100-045-CODE-STARLETTE",
        "head": "c6756b5ebb6eb1d3bf3de2499167833d99d99a72",
        "authority_identity": "c6b210c8977cce4441134ef048ed7dbea1a1e74b295ee96ce70ce5d612962722",
        "family": "github:Kludex/starlette",
        "normalized_bytes": 5274,
        "dedicated_workflow_run": 32998101312,
        "dedicated_workflow_name": "NEXT100-045 Starlette Code Source Admission",
    },
    468: {
        "worker": "NEXT100-049-CODE-NUMPY",
        "head": "bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8",
        "authority_identity": "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70",
        "family": "github:numpy/numpy",
        "normalized_bytes": 36898,
        "dedicated_workflow_run": 32998548535,
        "dedicated_workflow_name": "NEXT100-049 NumPy Code Source Authority",
    },
}

BLOCKED = {
    467: {
        "worker": "NEXT100-037-DATA-EN-PYTHON-DOCS",
        "head": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "family": "python.cpython.documentation",
        "dedicated_workflow_run": 32998356906,
        "dedicated_workflow_name": "NEXT100-037 Python Docs Source Authority",
        "dedicated_workflow_conclusion": "success",
    },
    465: {
        "worker": "NEXT100-048-CODE-PYDANTIC",
        "head": "ca1755886f052d272029d6d68b2f1b7f02187936",
        "family": "github:pydantic/pydantic",
        "dedicated_workflow_run": 32999061340,
        "dedicated_workflow_name": "NEXT100-048 Pydantic Source Admission",
        "dedicated_workflow_conclusion": "failure",
    },
    475: {
        "worker": "NEXT100-051-CODE-RICH",
        "head": "78cada1d69b3f0c438012c4e6cf79143aae2f603",
        "family": "github:Textualize/rich",
        "dedicated_workflow_run": 32999511493,
        "dedicated_workflow_name": "NEXT100-051 Rich Source Admission",
        "dedicated_workflow_conclusion": "failure",
    },
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_identity(data: dict[str, Any]) -> str:
    body = dict(data)
    body.pop("registry_identity_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(data: dict[str, Any]) -> dict[str, Any]:
    require(data.get("schema_version") == SCHEMA, "schema mismatch")
    require(data.get("worker_id") == WORKER, "worker mismatch")
    require(data.get("registry_identity_sha256") == canonical_identity(data), "registry identity mismatch")

    supersedes = data.get("supersedes")
    require(isinstance(supersedes, dict), "supersedes missing")
    require(supersedes.get("registry_identity_sha256") == V1_IDENTITY, "V1 binding drift")

    policy = data.get("composition_policy")
    require(isinstance(policy, dict), "composition policy missing")
    for key in (
        "only_terminal_admit_authorities_counted",
        "dedicated_exact_head_source_workflow_must_succeed",
        "generic_workflow_success_cannot_substitute",
        "quality_rejected_subrecords_must_not_receive_capacity_credit",
        "source_level_bytes_cannot_substitute_for_missing_eligible_subrecord_ledger",
        "evaluation_permission_never_inferred_from_training_permission",
        "one_independent_family_credit_per_canonical_lineage",
        "global_cross_source_dedup_required_before_corpus_identity",
        "decontamination_required_before_corpus_identity",
    ):
        require(policy.get(key) is True, f"policy weakened: {key}")
    require(policy.get("parallel_retest_queued_or_failed_candidates_counted") is False, "nonterminal candidates counted")
    require(policy.get("replay_or_duplication_may_repair_capacity") is False, "replay/duplication capacity repair enabled")

    base = data.get("base_registry")
    require(isinstance(base, dict), "base registry missing")
    require(base.get("registry_identity_sha256") == BASE_IDENTITY, "DATA-287 identity drift")
    require(base.get("head_sha") == "b0523ccbc4b957615aac849d476cfa851be87578", "DATA-287 head drift")
    require(base.get("unique_normalized_bytes") == 183061, "base byte drift")
    require(base.get("source_count") == 5, "base source-count drift")
    require(base.get("independent_family_count") == 4, "base family-count drift")
    require(
        base.get("by_stratum")
        == {
            "code": {"family_count": 2, "normalized_bytes": 9703},
            "en": {"family_count": 1, "normalized_bytes": 84793},
            "uk": {"family_count": 1, "normalized_bytes": 88565},
        },
        "base stratum vector drift",
    )

    rows = data.get("terminal_additions")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_ROWS), "credited row set changed")
    by_pr: dict[int, dict[str, Any]] = {}
    seen_heads: set[str] = set()
    seen_families = set(base.get("families", []))
    by_stratum = {key: dict(value) for key, value in base["by_stratum"].items()}
    new_bytes = 0

    for raw in rows:
        require(isinstance(raw, dict), "credited row must be an object")
        pr = raw.get("pr")
        require(isinstance(pr, int) and pr not in by_pr, "credited PRs must be unique")
        require(pr in EXPECTED_ROWS, f"unexpected credited PR: {pr}")
        by_pr[pr] = raw
        expected = EXPECTED_ROWS[pr]
        for key, value in expected.items():
            require(raw.get(key) == value, f"PR {pr}: binding drift: {key}")
        require(raw.get("dedicated_workflow_conclusion") == "success", f"PR {pr}: dedicated workflow not success")
        require(str(raw.get("verdict", "")).startswith("ADMIT"), f"PR {pr}: not terminal ADMIT")
        require(str(raw.get("training", "")).startswith("ALLOWED"), f"PR {pr}: training not authorized")
        require(str(raw.get("evaluation", "")).startswith("NOT_"), f"PR {pr}: evaluation permission leaked")
        head = raw["head"]
        family = raw["family"]
        require(head not in seen_heads, f"duplicate head: {head}")
        require(family not in seen_families, f"duplicate family: {family}")
        seen_heads.add(head)
        seen_families.add(family)
        capacity = raw["normalized_bytes"]
        require(isinstance(capacity, int) and capacity > 0, f"PR {pr}: invalid capacity")
        stratum = "code" if raw.get("modality") == "code" else raw.get("language")
        require(stratum in by_stratum, f"PR {pr}: unsupported stratum")
        by_stratum[stratum]["normalized_bytes"] += capacity
        by_stratum[stratum]["family_count"] += 1
        new_bytes += capacity

    require(set(by_pr) == set(EXPECTED_ROWS), "credited source set incomplete")
    require(not (set(BLOCKED) & set(by_pr)), "blocked source received credit")

    numpy_row = by_pr[468]
    require(numpy_row.get("terminal_report_sha256") == "e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70", "NumPy terminal report drift")
    require(numpy_row.get("terminal_artifact_id") == 9618015895, "NumPy terminal artifact drift")
    require(numpy_row.get("terminal_artifact_zip_sha256") == "402016760c2ea5b341ed15537bb173e9bf10a938870313f00fd5e617ba20b020", "NumPy artifact identity drift")

    held = data.get("held_out_or_noncomposable")
    require(isinstance(held, list), "held-out vector missing")
    held_by_pr = {row.get("pr"): row for row in held if isinstance(row, dict) and isinstance(row.get("pr"), int)}
    for pr, expected in BLOCKED.items():
        require(pr in held_by_pr, f"blocked PR {pr} missing")
        row = held_by_pr[pr]
        for key, value in expected.items():
            require(row.get(key) == value, f"blocked PR {pr}: binding drift: {key}")

    cpython = held_by_pr[467]
    require(cpython.get("source_normalized_bytes_not_capacity_credit") == 17901, "CPython source bytes drift")
    require(cpython.get("accepted_chunk_count") == 14, "CPython accepted chunk count drift")
    require(cpython.get("rejected_chunk_count") == 2, "CPython rejected chunk count drift")
    require("exact accepted-chunk byte ledger" in cpython.get("reason", ""), "CPython fail-closed reason missing")

    for pr in (465, 475):
        row = held_by_pr[pr]
        require(row.get("dedicated_workflow_conclusion") == "failure", f"PR {pr}: failure evidence lost")
        require("generic DATA-227 success cannot substitute" in row.get("reason", ""), f"PR {pr}: generic-CI substitution boundary missing")

    inv = data.get("pre_global_dedup_inventory")
    require(isinstance(inv, dict), "inventory missing")
    total = int(base["unique_normalized_bytes"]) + new_bytes
    require(new_bytes == 120313, "new terminal byte total drift")
    require(total == 303374, "candidate byte total drift")
    require(inv.get("base_unique_normalized_bytes") == 183061, "inventory base-byte drift")
    require(inv.get("new_terminal_normalized_bytes") == new_bytes, "inventory new-byte drift")
    require(inv.get("terminal_addition_authority_count") == len(rows), "terminal addition count drift")
    require(inv.get("candidate_normalized_bytes") == total, "inventory candidate-byte drift")
    require(inv.get("candidate_independent_family_count") == len(seen_families), "candidate family-count drift")
    require(inv.get("by_stratum") == by_stratum, "stratum accounting drift")
    minimum = inv.get("minimum_independent_families_per_stratum")
    require(minimum == 2, "family minimum policy drift")
    require(all(v["family_count"] >= minimum for v in by_stratum.values()), "family minimum does not pass")
    require(inv.get("family_minimum_gate") == "PASS_PRE_GLOBAL_DEDUP", "family gate promoted incorrectly")
    target = inv.get("research_corpus_v1_target_normalized_bytes")
    require(target == 20_000_000, "Research Corpus V1 target drift")
    require(inv.get("target_gap_normalized_bytes") == target - total, "target gap drift")
    require(math.isclose(float(inv.get("target_fraction")), total / target, rel_tol=0.0, abs_tol=1e-12), "target fraction drift")
    require(inv.get("required_stratum_fraction") == {"uk": 0.45, "en": 0.35, "code": 0.2}, "required stratum mix drift")
    expected_ceilings = {
        key: math.floor(by_stratum[key]["normalized_bytes"] / frac)
        for key, frac in inv["required_stratum_fraction"].items()
    }
    require(inv.get("stratum_only_ceiling_by_stratum") == expected_ceilings, "stratum no-replay ceilings drift")
    require(inv.get("stratum_only_no_replay_ceiling_normalized_bytes") == min(expected_ceilings.values()), "global no-replay ceiling drift")
    require(inv.get("stratum_only_no_replay_ceiling_limiter") == "uk", "no-replay limiter drift")

    gates = data.get("downstream_gate_vector")
    require(isinstance(gates, dict), "downstream gate vector missing")
    require(gates.get("source_registry_convergence") == "PASS_FAIL_CLOSED_CANDIDATE_AUTHORITY_VECTOR", "source convergence state drift")
    require(gates.get("global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "dedup gate weakened")
    require(gates.get("evaluation_decontamination") == "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY", "decontamination gate weakened")
    require(gates.get("authorized_balanced_no_replay_loss_positions") == 0, "training exposure must remain zero")
    require(gates.get("tokenizer_fit") == "BLOCKED", "tokenizer fit promoted")
    require(gates.get("long_training") == "BLOCKED", "long training promoted")
    require(gates.get("paid_compute") == "NOT_AUTHORIZED", "paid compute authorized")

    boundary = data.get("claim_boundary")
    require(isinstance(boundary, dict) and boundary, "claim boundary missing")
    require(all(value is False for value in boundary.values()), "premature downstream claim")

    return {
        "status": "PASS",
        "registry_identity_sha256": data["registry_identity_sha256"],
        "candidate_normalized_bytes": total,
        "candidate_independent_family_count": len(seen_families),
        "by_stratum": by_stratum,
        "held_fail_closed_prs": [467, 465, 475],
        "next_gate": "GLOBAL_CROSS_SOURCE_EXACT_NEAR_DEDUP",
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report = validate(data)
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as exc:
        print(f"NEXT100-063 V2 FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
