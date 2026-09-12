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


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


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
    return hashlib.sha256(
        (
            json.dumps(
                list(record_ids),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


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
    expected_train_record_ids: Sequence[str] | None = None,
    expected_train_record_membership_digest_sha256: str | None = None,
) -> dict:
    record_inventory = inventory or _terminal_record_inventory()
    expected_record_inventory = expected_inventory or _terminal_record_inventory()
    train_record_ids = (
        list(expected_train_record_ids)
        if expected_train_record_ids is not None
        else _expected_train_record_ids()
    )
    train_membership_digest = (
        expected_train_record_membership_digest_sha256
        if expected_train_record_membership_digest_sha256 is not None
        else _train_membership_digest(_expected_train_record_ids())
    )
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
        expected_stage_bindings=_materialization()["stage_bindings"],
        expected_tokenizer_identity_sha256=_sha("tokenizer"),
        expected_train_record_ids=train_record_ids,
        expected_train_record_membership_digest_sha256=train_membership_digest,
    )


def _rehash_pair(build_a: dict, build_b: dict) -> None:
    for materialization in (build_a, build_b):
        materialization["materialization_identity_sha256"] = _identity(
            materialization, "materialization_identity_sha256"
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
    omitted_authoritative_train_a = _materialization()
    omitted_authoritative_train_b = _materialization()
    _expect_failure(
        lambda: _proof(
            omitted_authoritative_train_a,
            omitted_authoritative_train_b,
            inventory=closed_world_inventory,
            expected_inventory=closed_world_inventory,
            expected_train_record_ids=["en-doc", "uk-doc"],
            expected_train_record_membership_digest_sha256=_train_membership_digest(
                ["en-doc", "uk-doc"]
            ),
        ),
        "retained train record membership does not match expected terminal split authority",
    )

    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            expected_train_record_ids=[],
        ),
        "expected_train_record_ids do not match expected terminal split membership digest",
    )

    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            expected_train_record_ids=["uk-doc", "uk-doc"],
        ),
        "expected_train_record_ids contains duplicate record_id",
    )
    _expect_failure(
        lambda: _proof(
            build_a,
            build_b,
            expected_train_record_ids=["uk-doc", "en-doc"],
        ),
        "expected_train_record_ids must be in canonical record_id order",
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
            expected_stage_bindings=build_a["stage_bindings"],
            expected_tokenizer_identity_sha256=_sha("tokenizer"),
            expected_train_record_ids=_expected_train_record_ids(),
            expected_train_record_membership_digest_sha256=(
                _train_membership_digest(_expected_train_record_ids())
            ),
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
        "terminal_split_train_record_membership_sha256="
        f"{proof['terminal_split_train_record_membership_sha256']}"
    )
    print("retained_train_record_membership_verified=true")
    print("retained_document_isolation_verified=true")
    print("heldout_reservation_verified=true")
    print("training_authorized_by_this_proof=false")


if __name__ == "__main__":
    main()
