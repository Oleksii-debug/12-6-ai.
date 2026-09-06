from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Callable

from twelve_six.data.deterministic_double_pack import verify_deterministic_double_pack
from twelve_six.data.unique_loss_ledger_v2 import LedgerError


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


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


def _proof(build_a: dict, build_b: dict) -> dict:
    return verify_deterministic_double_pack(
        build_a,
        build_b,
        terminal_corpus_authority_identity_sha256=_sha("terminal-corpus-authority"),
        expected_stage_bindings=_materialization()["stage_bindings"],
        expected_tokenizer_identity_sha256=_sha("tokenizer"),
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
    _expect_failure(lambda: _proof(build_a, reordered), "loss span exceeds pack target slots")

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
    build_drift["documents"][0]["source_bytes"] += 1
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
    print("retained_document_isolation_verified=true")
    print("heldout_reservation_verified=true")
    print("training_authorized_by_this_proof=false")


if __name__ == "__main__":
    main()
