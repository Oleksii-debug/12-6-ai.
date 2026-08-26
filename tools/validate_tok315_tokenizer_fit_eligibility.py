#!/usr/bin/env python3
"""Validate TOK-315 tokenizer-fit eligibility evidence without fitting a tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.tok315-tokenizer-fit-eligibility.v1"
EXPECTED_WORKER = "TOK-315-TOKENIZER-FIT-ELIGIBILITY"
EXPECTED_STATUS = "BLOCKED_PENDING_MATERIALIZED_RESERVED_DECONTAMINATION"
EXPECTED_CONTRACT_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_INVENTORY_SHA = "945afd3dbd144f81c8441adf92e7784259de3f21a4dd547e95893243dec6e90d"
EXPECTED_SOURCES_SHA = "6da9104da534b7f3a266926e1285d0e0519b893f23152d2b30f52260c4506ada"


class Tok315EligibilityError(RuntimeError):
    """Raised when tokenizer-fit eligibility evidence is weakened or mismatched."""


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_projection(source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_id",
        "family",
        "language",
        "modality",
        "normalized_bytes",
        "normalized_sha256",
        "raw_sha256",
        "git_blob_sha1",
        "path",
        "upstream_revision",
    )
    return {key: source[key] for key in keys if key in source}


def validate(evidence: dict[str, Any], contract: dict[str, Any]) -> None:
    if evidence.get("schema_version") != EXPECTED_SCHEMA:
        raise Tok315EligibilityError("schema drift")
    if evidence.get("worker_id") != EXPECTED_WORKER:
        raise Tok315EligibilityError("worker identity drift")
    if evidence.get("repository") != "Oleksii-debug/12-6-ai.":
        raise Tok315EligibilityError("repository identity drift")
    if evidence.get("execution_profile") != "LOCAL_FREE":
        raise Tok315EligibilityError("LOCAL_FREE boundary weakened")
    if evidence.get("status") != EXPECTED_STATUS:
        raise Tok315EligibilityError("fail-closed status drift")

    claimed_self_hash = str(evidence.get("evidence_sha256", ""))
    core = dict(evidence)
    core.pop("evidence_sha256", None)
    if canonical_sha(core) != claimed_self_hash:
        raise Tok315EligibilityError("evidence self-hash mismatch")

    binding = evidence.get("corpus_binding", {})
    if contract.get("contract_identity_sha256") != EXPECTED_CONTRACT_ID:
        raise Tok315EligibilityError("DATA-300 contract identity mismatch")
    if binding.get("contract_identity_sha256") != EXPECTED_CONTRACT_ID:
        raise Tok315EligibilityError("evidence DATA-300 identity mismatch")
    if contract.get("contract_state") != "FROZEN_EXECUTABLE_CONTRACT":
        raise Tok315EligibilityError("DATA-300 contract is not frozen")
    if contract.get("corpus_state") != "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL":
        raise Tok315EligibilityError("TOK-315 v1 must be rebound after corpus-state change")
    if binding.get("corpus_state") != contract.get("corpus_state"):
        raise Tok315EligibilityError("corpus-state binding mismatch")
    if contract.get("execution_profile") != "LOCAL_FREE":
        raise Tok315EligibilityError("DATA-300 is not LOCAL_FREE")

    split = contract.get("split_contract", {})
    if split.get("train", {}).get("may_fit_tokenizer") is not True:
        raise Tok315EligibilityError("train tokenizer-fit permission missing")
    if split.get("selection_validation", {}).get("may_fit_tokenizer") is not False:
        raise Tok315EligibilityError("selection-validation may not fit tokenizer")
    if split.get("final_test", {}).get("may_fit_tokenizer") is not False:
        raise Tok315EligibilityError("final-test may not fit tokenizer")
    global_split = split.get("global", {})
    if global_split.get("exact_content_overlap_allowed") is not False:
        raise Tok315EligibilityError("exact content overlap prohibition weakened")
    if global_split.get("record_identity_reuse_allowed") is not False:
        raise Tok315EligibilityError("record identity reuse prohibition weakened")

    inventory = contract.get("exact_training_candidate_inventory")
    if not isinstance(inventory, dict):
        raise Tok315EligibilityError("DATA-300 exact training inventory missing")
    sources = inventory.get("sources")
    if not isinstance(sources, list):
        raise Tok315EligibilityError("DATA-300 source list missing")
    if canonical_sha(inventory) != EXPECTED_INVENTORY_SHA:
        raise Tok315EligibilityError("exact DATA-300 training inventory hash mismatch")
    if canonical_sha(sources) != EXPECTED_SOURCES_SHA:
        raise Tok315EligibilityError("exact DATA-300 source-list hash mismatch")
    if inventory.get("source_count") != 5 or inventory.get("admitted_source_bytes") != 183061:
        raise Tok315EligibilityError("source count or admitted-byte count drift")
    if any(source.get("training_rights") != "ALLOWED" for source in sources):
        raise Tok315EligibilityError("non-training-authorized source in fit inventory")

    fit_inventory = evidence.get("tokenizer_training_inventory", {})
    if fit_inventory.get("resolution") != "EXACT_DATA300_TRAINING_INVENTORY_ALLOWLIST_ONLY":
        raise Tok315EligibilityError("fit resolution must remain exact allowlist only")
    if fit_inventory.get("inventory_sha256") != EXPECTED_INVENTORY_SHA:
        raise Tok315EligibilityError("evidence inventory hash drift")
    if fit_inventory.get("sources_sha256") != EXPECTED_SOURCES_SHA:
        raise Tok315EligibilityError("evidence source-list hash drift")
    if fit_inventory.get("source_count") != inventory.get("source_count"):
        raise Tok315EligibilityError("fit source count mismatch")
    if fit_inventory.get("admitted_source_bytes") != inventory.get("admitted_source_bytes"):
        raise Tok315EligibilityError("fit byte count mismatch")
    if fit_inventory.get("sources") != [_source_projection(source) for source in sources]:
        raise Tok315EligibilityError("fit source allowlist does not exactly project DATA-300")

    separation = evidence.get("separation", {})
    required_false = (
        "filesystem_glob_or_fallback_allowed",
        "unlisted_source_allowed",
        "selection_validation_may_fit_tokenizer",
        "final_test_may_fit_tokenizer",
    )
    if any(separation.get(key) is not False for key in required_false):
        raise Tok315EligibilityError("evaluation-ingress or fallback boundary weakened")
    if separation.get("selection_validation_ingress_count") != 0:
        raise Tok315EligibilityError("selection-validation entered tokenizer fit")
    if separation.get("final_test_ingress_count") != 0:
        raise Tok315EligibilityError("final-test entered tokenizer fit")
    if separation.get("eval291", {}).get("records") != 0:
        raise Tok315EligibilityError("bound EVAL-291 is not empty")
    if separation.get("eval292", {}).get("records") != 0:
        raise Tok315EligibilityError("bound EVAL-292 is not empty")
    if separation.get("eval233", {}).get("tokenizer_fit_eligible") is not False:
        raise Tok315EligibilityError("EVAL-233 final-test fit prohibition weakened")

    proof = evidence.get("proof", {})
    if proof.get("source_membership_ingress") != "PASS_EXACT_ALLOWLIST":
        raise Tok315EligibilityError("source-membership proof missing")
    if proof.get("reserved_byte_overlap") != "BLOCKED_PENDING_DATA300_G08_ON_EXACT_MATERIALIZATION":
        raise Tok315EligibilityError("reserved-byte proof overclaimed")
    if proof.get("bpe_fit_execution_permitted") is not False:
        raise Tok315EligibilityError("BPE fit cannot start before materialized G08 proof")
    if not any(gate.get("id") == "G08_RESERVED_DECONTAMINATION" for gate in contract.get("release_gates", [])):
        raise Tok315EligibilityError("DATA-300 G08 gate missing")

    baseline = evidence.get("canonical_byte_baseline", {})
    if baseline.get("tokenizer_version") != "s0-byte-v1" or baseline.get("fit_required") is not False:
        raise Tok315EligibilityError("canonical byte baseline drift")
    if baseline.get("winner") is not False:
        raise Tok315EligibilityError("byte baseline must not be declared winner")

    bpe = evidence.get("future_bpe", {})
    if bpe.get("supported") is not True or bpe.get("fit_may_start_now") is not False:
        raise Tok315EligibilityError("future BPE gate drift")
    if bpe.get("required_inventory_sha256") != EXPECTED_INVENTORY_SHA:
        raise Tok315EligibilityError("future BPE inventory binding drift")
    if bpe.get("requires_data300_g08_pass") is not True:
        raise Tok315EligibilityError("future BPE must require G08")
    if bpe.get("requires_materialized_training_bytes_identity") is not True:
        raise Tok315EligibilityError("future BPE must bind materialized bytes")
    if bpe.get("may_read_selection_validation_for_fit") is not False:
        raise Tok315EligibilityError("BPE may not fit on selection-validation")
    if bpe.get("may_read_final_test_for_fit") is not False:
        raise Tok315EligibilityError("BPE may not fit on final-test")
    if bpe.get("winner") is not False or evidence.get("tokenizer_winner") != "UNSELECTED":
        raise Tok315EligibilityError("TOK-315 may not choose a tokenizer winner")


def load_and_validate(evidence_path: Path, contract_path: Path) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate(evidence, contract)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/tok315/tokenizer-fit-eligibility-v1.json"))
    parser.add_argument("--contract", type=Path, default=Path("configs/data/data300_corpus_v03_frozen_build_contract_v2.json"))
    args = parser.parse_args(argv)
    evidence = load_and_validate(args.evidence, args.contract)
    print(json.dumps({
        "status": evidence["status"],
        "inventory_sha256": evidence["tokenizer_training_inventory"]["inventory_sha256"],
        "tokenizer_winner": evidence["tokenizer_winner"],
        "bpe_fit_execution_permitted": evidence["proof"]["bpe_fit_execution_permitted"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
