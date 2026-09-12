from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from copy import deepcopy

from twelve_six.data.deterministic_double_pack import verify_deterministic_double_pack
from twelve_six.data.unique_loss_ledger_v2 import LedgerError


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_no_lf(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_with_lf(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_with_lf(payload)).hexdigest()


def _identity_no_lf(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_no_lf(payload)).hexdigest()


def _terminal_record_inventory(*, include_en_train: bool = False) -> dict:
    records = [
        {
            "record_id": "uk-doc",
            "source_id": "source.uk.synthetic",
            "family": "family.uk",
            "modality": "text",
            "payload_sha256": _sha("uk-payload"),
            "payload_bytes": 12,
        }
    ]
    if include_en_train:
        records.append(
            {
                "record_id": "en-doc",
                "source_id": "source.en.synthetic",
                "family": "family.en",
                "modality": "text",
                "payload_sha256": _sha("en-payload"),
                "payload_bytes": 11,
            }
        )
    records.sort(key=lambda item: item["record_id"])
    payload_projection = [
        {
            "record_id": item["record_id"],
            "payload_sha256": item["payload_sha256"],
            "payload_bytes": item["payload_bytes"],
        }
        for item in records
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(records),
        "total_payload_bytes": sum(item["payload_bytes"] for item in records),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical_no_lf(records)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical_no_lf(payload_projection)
        ).hexdigest(),
        "records": records,
    }


def _expected_train_record_ids() -> list[str]:
    return ["uk-doc"]


def _train_membership_digest(record_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_with_lf(list(record_ids))).hexdigest()


def _split_variant(
    *,
    variant_id: str,
    seed: str,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> dict:
    core = {
        "schema_version": "12-6.validation-split.v1",
        "variant_id": variant_id,
        "seed": seed,
        "algorithm": "cluster-hash-ranked-greedy-v1",
        "eligible_corpus_sha256": _sha("eligible-corpus"),
        "dedup_relations_sha256": _sha("dedup-relations"),
        "validation_fraction_requested": 0.25,
        "validation_clusters": ["cluster-selection"],
        "train_record_ids": list(train_ids),
        "validation_record_ids": list(validation_ids),
        "train_documents": len(train_ids),
        "validation_documents": len(validation_ids),
        "cluster_straddles": [],
    }
    return {**core, "split_identity_sha256": hashlib.sha256(_canonical_with_lf(core)).hexdigest()}


def _terminal_split_application(
    shared_train_record_ids: Sequence[str] | None = None,
) -> dict:
    train_ids = sorted(
        list(shared_train_record_ids)
        if shared_train_record_ids is not None
        else _expected_train_record_ids()
    )
    validation_ids = ["selection-doc"]
    variants = [
        _split_variant(
            variant_id="v01",
            seed="variant-a",
            train_ids=train_ids,
            validation_ids=validation_ids,
        ),
        _split_variant(
            variant_id="v02",
            seed="variant-b",
            train_ids=train_ids,
            validation_ids=validation_ids,
        ),
    ]
    family_core = {
        "schema_version": "12-6.validation-split-family.v1",
        "eligible_corpus_sha256": _sha("eligible-corpus"),
        "dedup_relations_sha256": _sha("dedup-relations"),
        "algorithm": "cluster-hash-ranked-greedy-v1",
        "validation_fraction_requested": 0.25,
        "variant_split_identities": [item["split_identity_sha256"] for item in variants],
        "variants": variants,
        "validation_union_record_ids": validation_ids,
        "shared_train_record_ids": train_ids,
        "shared_train_documents": len(train_ids),
        "validation_union_documents": len(validation_ids),
        "cluster_straddles_across_variants": 0,
        "legacy_record_hash_risk_audit": [
            {
                "seed": "variant-a",
                "near_duplicate_cluster_straddles": 0,
                "straddled_cluster_ids": [],
            },
            {
                "seed": "variant-b",
                "near_duplicate_cluster_straddles": 0,
                "straddled_cluster_ids": [],
            },
        ],
        "training_policy": "optimize_shared_train_core_only_excluding_validation_union",
    }
    split_family = {
        **family_core,
        "split_family_identity_sha256": hashlib.sha256(
            _canonical_with_lf(family_core)
        ).hexdigest(),
    }
    application_core = {
        "schema": "12-6.d03-balanced-split-application.v1",
        "status": "PASS_ZERO_CREDIT",
        "balanced_selection_identity_sha256": _sha("balanced-selection"),
        "retained_inventory_identity_sha256": _sha("retained-inventory"),
        "decontamination_authority_sha256": _sha("decontamination"),
        "dedup_authority_sha256": _sha("dedup-authority"),
        "balance_policy_identity_sha256": _sha("balance-policy"),
        "balance_result_identity_sha256": _sha("balance-result"),
        "canonical_split_git_blob_sha1": hashlib.sha1(b"canonical-split").hexdigest(),
        "selected_record_count": len(train_ids) + len(validation_ids),
        "selected_source_bytes": 1,
        "selected_family_source_bytes": {"family.synthetic": 1},
        "selected_stratum_source_bytes": {"uk": 1},
        "split_family": split_family,
        "claim_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
        },
    }
    return {
        **application_core,
        "application_identity_sha256": hashlib.sha256(
            _canonical_no_lf(application_core)
        ).hexdigest(),
    }


def _materialization() -> dict:
    value = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "terminal_corpus_authority_identity_sha256": _sha(
            "terminal-corpus-authority"
        ),
        "stage_bindings": {
            "normalization": _sha("normalization"),
            "evaluation_reservations": _sha("reservations"),
            "dedup": _sha("dedup"),
            "split": _sha("split"),
            "packing": _sha("packing-stage"),
        },
        "tokenizer": {
            "name": "s0-byte-v1",
            "identity_sha256": _sha("tokenizer"),
            "source_bytes_are_loss_positions": False,
        },
        "documents": [
            {
                "document_id": "uk-doc",
                "source_id": "source.uk.synthetic",
                "language": "uk",
                "modality": "text",
                "family_id": "family.uk",
                "normalized_payload_sha256": _sha("uk-payload"),
                "source_bytes": 12,
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-uk",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            },
            {
                "document_id": "selection-doc",
                "language": "en",
                "modality": "text",
                "family_id": "family.selection",
                "normalized_payload_sha256": _sha("selection-payload"),
                "source_bytes": 7,
                "token_count": 3,
                "split": "selection",
                "dedup_cluster_id": "cluster-selection",
                "retained_after_dedup": True,
                "evaluation_reserved": True,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [],
            },
        ],
        "packing": {
            "identity_sha256": _sha("packing-materialization"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 5,
                    "loss_spans": [
                        {
                            "document_id": "uk-doc",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    value["materialization_identity_sha256"] = _identity(
        value, "materialization_identity_sha256"
    )
    return value


def _proof(
    build_a: dict,
    build_b: dict,
    *,
    inventory: dict | None = None,
    expected_inventory: dict | None = None,
    split_application: dict | None = None,
    expected_split_application: dict | None = None,
) -> dict:
    record_inventory = inventory or _terminal_record_inventory()
    expected_record_inventory = expected_inventory or _terminal_record_inventory()
    split_authority = split_application or _terminal_split_application()
    trusted_split_authority = expected_split_application or _terminal_split_application()
    return verify_deterministic_double_pack(
        build_a,
        build_b,
        terminal_corpus_authority_identity_sha256=_sha("terminal-corpus-authority"),
        terminal_record_inventory=record_inventory,
        expected_record_inventory_digest_sha256=(
            expected_record_inventory["record_inventory_digest_sha256"]
        ),
        expected_payload_inventory_digest_sha256=(
            expected_record_inventory["payload_inventory_digest_sha256"]
        ),
        terminal_split_application=split_authority,
        expected_terminal_split_application_identity_sha256=(
            trusted_split_authority["application_identity_sha256"]
        ),
        expected_stage_bindings=_materialization()["stage_bindings"],
        expected_tokenizer_identity_sha256=_sha("tokenizer"),
    )


def _rehash_pair(build_a: dict, build_b: dict) -> None:
    for materialization in (build_a, build_b):
        materialization["materialization_identity_sha256"] = _identity(
            materialization, "materialization_identity_sha256"
        )


def _rehash_split_application(application: dict) -> None:
    split_family = application["split_family"]
    split_family["split_family_identity_sha256"] = _identity(
        split_family, "split_family_identity_sha256"
    )
    application["application_identity_sha256"] = _identity_no_lf(
        application, "application_identity_sha256"
    )


def _expect_failure(action: Callable[[], object], message: str) -> None:
    try:
        action()
    except LedgerError as exc:
        if message not in str(exc):
            raise SystemExit(
                f"expected failure containing {message!r}, got {str(exc)!r}"
            ) from exc
    else:
        raise SystemExit(f"expected fail-closed rejection containing {message!r}")


def main() -> None:
    build_a = _materialization()
    build_b = _materialization()
    proof = _proof(build_a, build_b)
    if proof["independent_builds_byte_identical"] is not True:
        raise SystemExit("double-pack proof did not establish byte identity")
    if proof["one_pass_unique_nonignored_causal_loss_positions"] != 4:
        raise SystemExit("double-pack proof unique loss count mismatch")
    if proof["retained_train_record_membership_verified"] is not True:
        raise SystemExit("terminal train-record membership was not proven")
    if proof["retained_train_records_matched_to_terminal_inventory"] != 1:
        raise SystemExit("terminal train-record membership count mismatch")
    expected_membership_digest = _train_membership_digest(_expected_train_record_ids())
    if (
        proof["terminal_split_train_record_membership_sha256"]
        != expected_membership_digest
    ):
        raise SystemExit("terminal split train-membership digest mismatch")
    if proof["terminal_split_application_identity_sha256"] != (
        _terminal_split_application()["application_identity_sha256"]
    ):
        raise SystemExit("terminal split application identity mismatch")
    if proof["retained_document_isolation_verified"] is not True:
        raise SystemExit("retained-document isolation was not proven")
    if proof["heldout_reservation_verified"] is not True:
        raise SystemExit("held-out reservation was not proven")
    if proof["training_authorized_by_this_proof"] is not False:
        raise SystemExit("double-pack proof must never self-authorize training")
    if proof["build_a_canonical_sha256"] != proof["build_b_canonical_sha256"]:
        raise SystemExit("double-pack canonical hashes differ")

    corpus_drift_a = _materialization()
    corpus_drift_b = _materialization()
    for materialization in (corpus_drift_a, corpus_drift_b):
        materialization["terminal_corpus_authority_identity_sha256"] = _sha(
            "other-terminal-corpus"
        )
    _rehash_pair(corpus_drift_a, corpus_drift_b)
    _expect_failure(
        lambda: _proof(corpus_drift_a, corpus_drift_b),
        "corpus identity does not match terminal handoff",
    )

    forged_payload_a = _materialization()
    forged_payload_b = _materialization()
    for materialization in (forged_payload_a, forged_payload_b):
        materialization["documents"][0]["normalized_payload_sha256"] = _sha(
            "forged-train-payload"
        )
    _rehash_pair(forged_payload_a, forged_payload_b)
    _expect_failure(
        lambda: _proof(forged_payload_a, forged_payload_b),
        "train document payload does not match terminal D03 inventory",
    )

    invented_record_a = _materialization()
    invented_record_b = _materialization()
    for materialization in (invented_record_a, invented_record_b):
        materialization["documents"][0]["document_id"] = "invented-train-record"
    _rehash_pair(invented_record_a, invented_record_b)
    _expect_failure(
        lambda: _proof(invented_record_a, invented_record_b),
        "retained train document is absent from terminal D03 inventory",
    )

    source_drift_a = _materialization()
    source_drift_b = _materialization()
    for materialization in (source_drift_a, source_drift_b):
        materialization["documents"][0]["source_id"] = "other.source"
    _rehash_pair(source_drift_a, source_drift_b)
    _expect_failure(
        lambda: _proof(source_drift_a, source_drift_b),
        "train document source_id does not match terminal D03 inventory",
    )

    inventory_drift = _terminal_record_inventory()
    inventory_drift["records"][0]["family"] = "forged.family"
    inventory_drift["record_inventory_digest_sha256"] = hashlib.sha256(
        _canonical_no_lf(inventory_drift["records"])
    ).hexdigest()
    _expect_failure(
        lambda: _proof(build_a, build_b, inventory=inventory_drift),
        "terminal record inventory does not match expected D03 handoff",
    )

    closed_world_inventory = _terminal_record_inventory(include_en_train=True)
    authoritative_split = _terminal_split_application(["en-doc", "uk-doc"])
    _expect_failure(
        lambda: _proof(
            _materialization(),
            _materialization(),
            inventory=closed_world_inventory,
            expected_inventory=closed_world_inventory,
            split_application=authoritative_split,
            expected_split_application=authoritative_split,
        ),
        "retained train record membership does not match authenticated terminal split authority",
    )

    resealed_smaller_split = deepcopy(authoritative_split)
    resealed_smaller_split["split_family"]["shared_train_record_ids"] = ["uk-doc"]
    resealed_smaller_split["split_family"]["shared_train_documents"] = 1
    _rehash_split_application(resealed_smaller_split)
    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            inventory=closed_world_inventory,
            expected_inventory=closed_world_inventory,
            split_application=resealed_smaller_split,
            expected_split_application=authoritative_split,
        ),
        "terminal split application does not match expected identity",
    )

    duplicate_membership = _terminal_split_application()
    duplicate_membership["split_family"]["shared_train_record_ids"] = [
        "uk-doc",
        "uk-doc",
    ]
    duplicate_membership["split_family"]["shared_train_documents"] = 2
    _rehash_split_application(duplicate_membership)
    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            split_application=duplicate_membership,
            expected_split_application=duplicate_membership,
        ),
        "shared_train_record_ids contains duplicate record_id",
    )

    noncanonical_membership = _terminal_split_application(["en-doc", "uk-doc"])
    noncanonical_membership["split_family"]["shared_train_record_ids"] = [
        "uk-doc",
        "en-doc",
    ]
    _rehash_split_application(noncanonical_membership)
    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            split_application=noncanonical_membership,
            expected_split_application=noncanonical_membership,
        ),
        "shared_train_record_ids must be in canonical record_id order",
    )

    missing_inventory_member = _terminal_split_application(["en-doc", "uk-doc"])
    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            split_application=missing_inventory_member,
            expected_split_application=missing_inventory_member,
        ),
        "authenticated terminal split train record is absent from terminal D03 inventory",
    )

    cluster_leak_a = _materialization()
    cluster_leak_b = _materialization()
    for materialization in (cluster_leak_a, cluster_leak_b):
        materialization["documents"][1]["dedup_cluster_id"] = "cluster-uk"
    _rehash_pair(cluster_leak_a, cluster_leak_b)
    _expect_failure(
        lambda: _proof(cluster_leak_a, cluster_leak_b),
        "retained dedup cluster is shared across documents/splits",
    )

    payload_leak_a = _materialization()
    payload_leak_b = _materialization()
    for materialization in (payload_leak_a, payload_leak_b):
        materialization["documents"][1]["normalized_payload_sha256"] = _sha(
            "uk-payload"
        )
    _rehash_pair(payload_leak_a, payload_leak_b)
    _expect_failure(
        lambda: _proof(payload_leak_a, payload_leak_b),
        "retained normalized payload is duplicated across documents/splits",
    )

    unreserved_a = _materialization()
    unreserved_b = _materialization()
    for materialization in (unreserved_a, unreserved_b):
        materialization["documents"][1]["evaluation_reserved"] = False
    _rehash_pair(unreserved_a, unreserved_b)
    _expect_failure(
        lambda: _proof(unreserved_a, unreserved_b),
        "held-out retained document must be evaluation_reserved",
    )

    reordered = _materialization()
    reordered["packing"]["packs"][0]["loss_spans"][0]["pack_target_start"] = 0
    reordered["materialization_identity_sha256"] = _identity(
        reordered, "materialization_identity_sha256"
    )
    _expect_failure(
        lambda: _proof(build_a, reordered),
        "loss span exceeds pack target slots",
    )

    tokenizer_drift_a = _materialization()
    tokenizer_drift_b = _materialization()
    for materialization in (tokenizer_drift_a, tokenizer_drift_b):
        materialization["tokenizer"]["identity_sha256"] = _sha("other-tokenizer")
    _rehash_pair(tokenizer_drift_a, tokenizer_drift_b)
    _expect_failure(
        lambda: _proof(tokenizer_drift_a, tokenizer_drift_b),
        "tokenizer identity does not match terminal handoff",
    )

    split_drift_a = _materialization()
    split_drift_b = _materialization()
    for materialization in (split_drift_a, split_drift_b):
        materialization["stage_bindings"]["split"] = _sha("other-split")
    _rehash_pair(split_drift_a, split_drift_b)
    _expect_failure(
        lambda: _proof(split_drift_a, split_drift_b),
        "stage bindings do not match terminal handoff",
    )

    build_drift = _materialization()
    build_drift["documents"][1]["source_bytes"] += 1
    build_drift["materialization_identity_sha256"] = _identity(
        build_drift, "materialization_identity_sha256"
    )
    _expect_failure(
        lambda: _proof(build_a, build_drift),
        "independent post-pack materializations are not byte-identical",
    )

    _expect_failure(
        lambda: verify_deterministic_double_pack(
            build_a,
            build_b,
            terminal_corpus_authority_identity_sha256="not-a-hash",
            terminal_record_inventory=_terminal_record_inventory(),
            expected_record_inventory_digest_sha256=(
                _terminal_record_inventory()["record_inventory_digest_sha256"]
            ),
            expected_payload_inventory_digest_sha256=(
                _terminal_record_inventory()["payload_inventory_digest_sha256"]
            ),
            terminal_split_application=_terminal_split_application(),
            expected_terminal_split_application_identity_sha256=(
                _terminal_split_application()["application_identity_sha256"]
            ),
            expected_stage_bindings=build_a["stage_bindings"],
            expected_tokenizer_identity_sha256=_sha("tokenizer"),
        ),
        "terminal_corpus_authority_identity_sha256 must be a 64-hex",
    )

    print("D04 DETERMINISTIC DOUBLE PACK BINDING: PASS")
    print(f"proof_identity_sha256={proof['proof_identity_sha256']}")
    print(
        "one_pass_unique_nonignored_causal_loss_positions="
        f"{proof['one_pass_unique_nonignored_causal_loss_positions']}"
    )
    print(
        "terminal_split_application_identity_sha256="
        f"{proof['terminal_split_application_identity_sha256']}"
    )
    print(
        "terminal_split_train_record_membership_sha256="
        f"{proof['terminal_split_train_record_membership_sha256']}"
    )
    print("retained_train_record_membership_verified=true")
    print("retained_document_isolation_verified=true")
    print("heldout_reservation_verified=true")
    print("training_authorized_by_this_proof=false")


if __name__ == "__main__":
    main()
