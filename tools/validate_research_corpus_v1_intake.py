#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

EXPECTED_SCHEMA = "12-6.research-corpus-v1-intake.v1"
EXPECTED_DATA287_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_DATA287_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_DATA301_HEAD = "8820ba1b255f6bb95c7db0531fd846078a1aae01"
REQUIRED_GATES = {
    "materialize_exact_candidate_record_payloads",
    "purpose_specific_rights_firewall",
    "global_exact_and_near_dedup_with_lineage_collapse",
    "selection_and_final_test_decontamination",
    "quality_and_privacy_rescan",
    "cluster_safe_train_validation_split",
    "deterministic_tokenization_and_packing",
    "unique_nonignored_causal_loss_ledger",
    "two_clean_byte_identical_builds",
    "checkpoint_and_trainer_requalification_on_model341",
    "fresh_learn345_campaign_gate",
    "explicit_compute_authorization_if_materially_paid",
}


def canonical_identity(payload: dict) -> str:
    clean = copy.deepcopy(payload)
    clean.pop("intake_identity_sha256", None)
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(payload: dict) -> list[str]:
    errors: list[str] = []

    if payload.get("schema_version") != EXPECTED_SCHEMA:
        errors.append("unexpected schema_version")
    if payload.get("state") != "CANDIDATE_INTAKE_NOT_CORPUS_RELEASE":
        errors.append("intake state must remain non-release")
    if payload.get("execution_class") != "LOCAL_FREE":
        errors.append("execution_class must remain LOCAL_FREE")

    expected_identity = canonical_identity(payload)
    if payload.get("intake_identity_sha256") != expected_identity:
        errors.append("intake_identity_sha256 mismatch")

    base = payload.get("base_authorities", {})
    data287 = base.get("data287", {})
    if data287.get("head_sha") != EXPECTED_DATA287_HEAD:
        errors.append("DATA-287 head drift")
    if data287.get("registry_identity_sha256") != EXPECTED_DATA287_ID:
        errors.append("DATA-287 registry identity drift")
    data301 = base.get("data301", {})
    if data301.get("head_sha") != EXPECTED_DATA301_HEAD:
        errors.append("DATA-301 head drift")
    if data301.get("authorized_balanced_no_replay_capacity") != 0:
        errors.append("DATA-301 zero-capacity blocker must remain explicit")

    inventory = payload.get("candidate_inventory_contract", {})
    if inventory.get("candidate_corpus_identity") is not None:
        errors.append("candidate corpus identity must remain null before materialization/decontamination")
    if inventory.get("shard_identity") is not None:
        errors.append("shard identity must remain null before deterministic build")
    if inventory.get("no_replay") is not True or inventory.get("no_replacement_sampling") is not True:
        errors.append("no-replay/no-replacement invariants must remain true")
    if inventory.get("no_benchmark_or_final_test_training") is not True:
        errors.append("evaluation/final-test training firewall must remain true")

    sources = payload.get("additive_terminal_sources", [])
    if len(sources) != 3:
        errors.append("exactly three additive terminal source authorities are expected in v1")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    additive_records = 0
    for source in sources:
        sid = source.get("source_id")
        if not sid or sid in seen_ids:
            errors.append(f"duplicate or missing source_id: {sid!r}")
        else:
            seen_ids.add(sid)
        if source.get("scoped_workflow", {}).get("conclusion") != "success":
            errors.append(f"source {sid}: scoped workflow is not terminal success")
        if not str(source.get("training_use", "")).startswith("ALLOWED"):
            errors.append(f"source {sid}: training purpose is not explicitly allowed")
        if source.get("evaluation_use") != "NOT_SEPARATELY_ADMITTED":
            errors.append(f"source {sid}: evaluation authority must not be inferred")
        records = source.get("records", [])
        if not records:
            errors.append(f"source {sid}: no exact record identities")
        additive_records += len(records)
        for record in records:
            digest = record.get("normalized_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"source {sid}: malformed normalized sha256")
            elif digest in seen_hashes:
                errors.append(f"duplicate normalized record hash: {digest}")
            else:
                seen_hashes.add(digest)

    if inventory.get("additive_record_count") != additive_records:
        errors.append("additive_record_count does not match record inventory")

    projection = payload.get("intake_projection", {})
    family_counts = projection.get("declared_stratum_family_counts_before_cross_source_collapse", {})
    minimum = projection.get("data295_minimum_families_per_stratum")
    if minimum != 2:
        errors.append("DATA-295 family minimum drift")
    for stratum in ("uk", "en", "code"):
        value = family_counts.get(stratum)
        if not isinstance(value, int) or value < 2:
            errors.append(f"declared pre-decontamination family count below minimum for {stratum}")
    if projection.get("authorized_training_exposure") != 0:
        errors.append("intake must not authorize training exposure")
    if projection.get("optimized_causal_positions") is not None:
        errors.append("optimized causal positions must remain unknown before packing/loss ledger")

    missing_gates = REQUIRED_GATES.difference(payload.get("next_required_gates", []))
    if missing_gates:
        errors.append("missing fail-closed next gates: " + ", ".join(sorted(missing_gates)))

    boundary = payload.get("claim_boundary", {})
    must_be_false = (
        "training_executed",
        "paid_compute_used",
        "long_training_authorized",
        "corpus_frozen",
        "corpus_release_claimed",
        "model_quality_claimed",
        "family_diversity_final_pass_claimed",
    )
    for key in must_be_false:
        if boundary.get(key) is not False:
            errors.append(f"claim boundary {key} must remain false")
    if boundary.get("optimizer_updates") != 0:
        errors.append("optimizer_updates must remain zero")

    return errors


def run_self_tests(payload: dict) -> list[str]:
    failures: list[str] = []

    def reseal(candidate: dict) -> dict:
        candidate["intake_identity_sha256"] = canonical_identity(candidate)
        return candidate

    if validate(copy.deepcopy(payload)):
        failures.append("frozen intake does not validate")

    tampered = copy.deepcopy(payload)
    tampered["purpose"] += " tampered"
    if "intake_identity_sha256 mismatch" not in validate(tampered):
        failures.append("hash tamper was not rejected")

    training = reseal(copy.deepcopy(payload))
    training["intake_projection"]["authorized_training_exposure"] = 1
    training = reseal(training)
    if "intake must not authorize training exposure" not in validate(training):
        failures.append("training authorization mutation was not rejected")

    duplicate = copy.deepcopy(payload)
    duplicate["additive_terminal_sources"][1]["records"][0]["normalized_sha256"] = duplicate["additive_terminal_sources"][0]["records"][0]["normalized_sha256"]
    duplicate = reseal(duplicate)
    if not any("duplicate normalized record hash" in e for e in validate(duplicate)):
        failures.append("duplicate record hash was not rejected")

    family = copy.deepcopy(payload)
    family["intake_projection"]["declared_stratum_family_counts_before_cross_source_collapse"]["uk"] = 1
    family = reseal(family)
    if not any("below minimum for uk" in e for e in validate(family)):
        failures.append("family minimum mutation was not rejected")

    corpus = copy.deepcopy(payload)
    corpus["candidate_inventory_contract"]["candidate_corpus_identity"] = "0" * 64
    corpus = reseal(corpus)
    if "candidate corpus identity must remain null before materialization/decontamination" not in validate(corpus):
        failures.append("premature corpus identity was not rejected")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Research Corpus V1 pre-decontamination intake.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Run fail-closed adversarial mutation checks.")
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS intake_identity_sha256={payload['intake_identity_sha256']}")
    print("PASS declared_family_counts=uk:2,en:3,code:2 pending global lineage/dedup")
    print("PASS long_training_authorized=false authorized_training_exposure=0")
    if args.self_test:
        failures = run_self_tests(payload)
        if failures:
            for failure in failures:
                print(f"SELFTEST FAIL: {failure}")
            return 1
        print("SELFTEST PASS 6 fail-closed mutation checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
