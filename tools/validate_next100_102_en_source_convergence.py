"""Validate the NEXT100-102 terminal-English source convergence intake.

This is deliberately a source-authority intake gate, not a corpus-release gate.
It fails closed if the frozen DATA-287 baseline, authority identities, independent
families, normalized object identities, arithmetic, or downstream truth boundary
drift from the committed contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/next100_102_en_source_convergence_v1.json"
BASE_REGISTRY = ROOT / "data/registry/external_snapshots.v2.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ConvergenceError(RuntimeError):
    """Raised when source convergence cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConvergenceError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def validate(payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == "12-6.next100-102-en-source-convergence.v1",
        "unexpected convergence schema",
    )
    _require(payload.get("local_free_only") is True, "LOCAL_FREE boundary missing")

    bound = payload.get("base_registry")
    _require(isinstance(bound, dict), "base_registry must be a mapping")
    expected_registry_identity = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
    _require(
        base.get("registry_identity_sha256") == expected_registry_identity,
        "live checkout DATA-287 registry identity differs from frozen baseline",
    )
    _require(
        bound.get("registry_identity_sha256") == expected_registry_identity,
        "manifest baseline registry identity drift",
    )
    _require(
        bound.get("head_sha") == "b0523ccbc4b957615aac849d476cfa851be87578",
        "DATA-287 head binding drift",
    )
    _require(HEX40.fullmatch(bound.get("head_sha", "")) is not None, "invalid base head SHA")

    byte_report = base.get("byte_report")
    _require(isinstance(byte_report, dict), "DATA-287 byte_report missing")
    _require(byte_report.get("unique_normalized_bytes") == 183061, "DATA-287 byte total drift")
    _require(base.get("source_count") == 5, "DATA-287 source count drift")
    _require(base.get("independent_source_family_count") == 4, "DATA-287 family count drift")

    by_language = {
        row["key"]: row
        for row in byte_report.get("by_language", [])
        if isinstance(row, dict) and isinstance(row.get("key"), str)
    }
    _require(by_language.get("en", {}).get("unique_normalized_bytes") == 84793, "DATA-287 EN bytes drift")

    base_families = {
        row["key"]
        for row in base.get("family_deduplication", {}).get("family_rows", [])
        if isinstance(row, dict) and isinstance(row.get("key"), str)
    }
    base_hashes = {
        source.get("snapshot", {}).get("normalized_sha256")
        for source in base.get("sources", [])
        if isinstance(source, dict)
    }
    base_hashes.discard(None)

    additions = payload.get("terminal_additions")
    _require(isinstance(additions, list) and len(additions) == 2, "exactly two terminal additions required")

    added_families: set[str] = set()
    added_hashes: set[str] = set()
    added_bytes = 0
    added_objects = 0
    for source in additions:
        _require(isinstance(source, dict), "terminal addition must be a mapping")
        _require(source.get("language") == "en", "NEXT100-102 additions must be English")
        _require(source.get("modality") == "text", "NEXT100-102 additions must be text")
        _require(source.get("scoped_workflow_conclusion") == "success", "scoped authority is not terminal green")
        _require(source.get("generic_ci_release_green") is False, "generic CI must not be relabeled release-green")
        _require(source.get("evaluation_rights") == "NOT_SEPARATELY_ADMITTED", "evaluation-purpose firewall drift")
        _require(source.get("verdict") in {"ADMIT", "ADMIT_PROSE_ONLY"}, "non-admitted source consumed")
        _require(HEX40.fullmatch(source.get("head_sha", "")) is not None, "invalid source head SHA")
        _require(HEX40.fullmatch(source.get("authority_blob_sha1", "")) is not None, "invalid authority blob SHA-1")
        _require(HEX64.fullmatch(source.get("authority_identity", "")) is not None, "invalid authority identity")
        _require(isinstance(source.get("scoped_workflow_run"), int) and source["scoped_workflow_run"] > 0, "invalid scoped run")

        family = source.get("family_id")
        _require(isinstance(family, str) and family, "missing family id")
        _require(family not in base_families, f"added family already exists in baseline: {family}")
        _require(family not in added_families, f"duplicate added family: {family}")
        added_families.add(family)

        hashes = source.get("normalized_sha256")
        object_count = source.get("snapshot_object_count")
        _require(isinstance(hashes, list) and hashes, "normalized hash inventory missing")
        _require(object_count == len(hashes), "snapshot object count does not match hash inventory")
        for digest in hashes:
            _require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, "invalid normalized SHA-256")
            _require(digest not in base_hashes, "exact normalized collision with DATA-287")
            _require(digest not in added_hashes, "exact normalized collision across additions")
            added_hashes.add(digest)

        normalized_bytes = source.get("normalized_bytes")
        _require(isinstance(normalized_bytes, int) and normalized_bytes > 0, "invalid normalized byte count")
        added_bytes += normalized_bytes
        added_objects += object_count

    converged = payload.get("converged_intake")
    _require(isinstance(converged, dict), "converged_intake missing")
    expected_bytes = byte_report["unique_normalized_bytes"] + added_bytes
    expected_objects = base["source_count"] + added_objects
    expected_families = base["independent_source_family_count"] + len(added_families)
    expected_en_bytes = by_language["en"]["unique_normalized_bytes"] + added_bytes
    expected_en_families = 1 + len(added_families)

    _require(converged.get("unique_normalized_bytes_before_cross_source_near_dedup") == expected_bytes, "converged byte arithmetic drift")
    _require(converged.get("snapshot_object_count") == expected_objects, "converged object arithmetic drift")
    _require(converged.get("independent_family_count") == expected_families, "converged family arithmetic drift")
    _require(converged.get("english_normalized_bytes_before_cross_source_near_dedup") == expected_en_bytes, "converged EN byte arithmetic drift")
    _require(converged.get("english_independent_family_count") == expected_en_families, "converged EN family arithmetic drift")
    _require(converged.get("english_family_minimum_required") == 2, "family minimum policy drift")
    _require(expected_en_families >= 2, "English independent-family minimum still fails")
    _require(converged.get("replay_used") is False, "replay cannot repair diversity")
    _require(converged.get("exact_normalized_hash_collision_within_added_set") is False, "collision truth drift")

    gates = payload.get("downstream_gates")
    truth = payload.get("truth_boundary")
    _require(isinstance(gates, dict) and isinstance(truth, dict), "downstream truth boundary missing")
    _require(gates.get("canonical_cross_source_exact_dedup") == "REQUIRED", "exact dedup falsely bypassed")
    _require(gates.get("canonical_cross_source_near_dedup") == "REQUIRED", "near dedup falsely bypassed")
    _require(gates.get("evaluation_decontamination") == "REQUIRED", "decontamination falsely bypassed")
    _require(gates.get("research_corpus_v1") == "BLOCKED_NOT_MATERIALIZED", "Research Corpus V1 falsely promoted")
    _require(gates.get("long_20m_training") == "BLOCKED", "20M long training falsely authorized")
    _require(truth.get("canonical_registry_mutated") is False, "candidate intake mislabeled canonical registry")
    _require(truth.get("candidate_corpus_identity_created") is False, "candidate corpus identity fabricated")
    _require(truth.get("training_shards_created") is False, "training shards fabricated")
    _require(truth.get("training_authorized_exposure") == 0, "non-zero exposure fabricated")
    _require(truth.get("learned_20m_claimed") is False, "learned 20M falsely claimed")

    return {
        "status": "PASS_SOURCE_AUTHORITY_CONVERGENCE_ONLY",
        "base_registry_identity": expected_registry_identity,
        "added_families": sorted(added_families),
        "added_normalized_bytes": added_bytes,
        "converged_normalized_bytes_before_near_dedup": expected_bytes,
        "english_family_count": expected_en_families,
        "research_corpus_v1": "BLOCKED_NOT_MATERIALIZED",
        "training_authorized_exposure": 0,
    }


def self_test(payload: dict[str, Any], base: dict[str, Any]) -> None:
    mutations: list[tuple[str, Any]] = []

    collision = copy.deepcopy(payload)
    collision["terminal_additions"][1]["normalized_sha256"][0] = collision["terminal_additions"][0]["normalized_sha256"][0]
    mutations.append(("duplicate-normalized-hash", collision))

    replay = copy.deepcopy(payload)
    replay["converged_intake"]["replay_used"] = True
    mutations.append(("replay", replay))

    fake_release = copy.deepcopy(payload)
    fake_release["downstream_gates"]["research_corpus_v1"] = "PASS"
    mutations.append(("fake-corpus-release", fake_release))

    fake_exposure = copy.deepcopy(payload)
    fake_exposure["truth_boundary"]["training_authorized_exposure"] = 1
    mutations.append(("fake-training-exposure", fake_exposure))

    wrong_total = copy.deepcopy(payload)
    wrong_total["converged_intake"]["unique_normalized_bytes_before_cross_source_near_dedup"] += 1
    mutations.append(("wrong-arithmetic", wrong_total))

    for name, mutated in mutations:
        try:
            validate(mutated, base)
        except ConvergenceError:
            continue
        raise ConvergenceError(f"self-test mutation did not fail closed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--base-registry", type=Path, default=BASE_REGISTRY)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    payload = _load(args.manifest)
    base = _load(args.base_registry)
    result = validate(payload, base)
    if args.self_test:
        self_test(payload, base)
        result["adversarial_self_test"] = "PASS_5_OF_5_FAIL_CLOSED"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
