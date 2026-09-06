from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger, verify_ledger

DOUBLE_PACK_PROOF_SCHEMA = "12-6.d04-deterministic-double-pack-proof.v1"
_REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_obj(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{label} must be a non-empty string")
    return value


def _normalize_expected_stage_bindings(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != set(_REQUIRED_STAGE_BINDINGS):
        raise LedgerError(
            "expected_stage_bindings must contain exactly normalization, "
            "evaluation_reservations, dedup, split and packing"
        )
    return {
        name: _require_sha256(value[name], f"expected_stage_bindings.{name}")
        for name in _REQUIRED_STAGE_BINDINGS
    }


def _validate_retained_document_isolation(
    materialization: Mapping[str, Any], *, label: str
) -> None:
    """Cross-validate upstream dedup/split isolation before terminal packing proof."""
    documents = materialization.get("documents")
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise LedgerError(f"{label}.documents must be a sequence")

    retained_clusters: dict[str, tuple[str, str]] = {}
    retained_payloads: dict[str, tuple[str, str]] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise LedgerError(f"{label}.documents[{index}] must be an object")
        retained = document.get("retained_after_dedup")
        if not isinstance(retained, bool):
            raise LedgerError(
                f"{label}.documents[{index}].retained_after_dedup must be boolean"
            )
        if not retained:
            continue

        document_id = _require_nonempty_string(
            document.get("document_id"), f"{label}.documents[{index}].document_id"
        )
        split = _require_nonempty_string(
            document.get("split"), f"{label}.documents[{index}].split"
        )
        cluster_id = _require_nonempty_string(
            document.get("dedup_cluster_id"),
            f"{label}.documents[{index}].dedup_cluster_id",
        )
        payload_sha256 = _require_sha256(
            document.get("normalized_payload_sha256"),
            f"{label}.documents[{index}].normalized_payload_sha256",
        )

        previous_cluster_owner = retained_clusters.get(cluster_id)
        if previous_cluster_owner is not None:
            previous_document, previous_split = previous_cluster_owner
            raise LedgerError(
                "retained dedup cluster is shared across documents/splits: "
                f"{previous_document}:{previous_split} and {document_id}:{split}"
            )
        retained_clusters[cluster_id] = (document_id, split)

        previous_payload_owner = retained_payloads.get(payload_sha256)
        if previous_payload_owner is not None:
            previous_document, previous_split = previous_payload_owner
            raise LedgerError(
                "retained normalized payload is duplicated across documents/splits: "
                f"{previous_document}:{previous_split} and {document_id}:{split}"
            )
        retained_payloads[payload_sha256] = (document_id, split)

        if split != "train" and document.get("evaluation_reserved") is not True:
            raise LedgerError(
                "held-out retained document must be evaluation_reserved before packing"
            )


def _validate_build(
    materialization: Mapping[str, Any],
    *,
    label: str,
    expected_terminal_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    observed_terminal_corpus_identity = _require_sha256(
        materialization.get("terminal_corpus_authority_identity_sha256"),
        f"{label}.terminal_corpus_authority_identity_sha256",
    )
    if observed_terminal_corpus_identity != expected_terminal_corpus_identity_sha256:
        raise LedgerError(f"{label} corpus identity does not match terminal handoff")

    observed_stage_bindings = materialization.get("stage_bindings")
    if not isinstance(observed_stage_bindings, Mapping):
        raise LedgerError(f"{label}.stage_bindings must be an object")
    normalized_observed = _normalize_expected_stage_bindings(observed_stage_bindings)
    if normalized_observed != dict(expected_stage_bindings):
        raise LedgerError(f"{label} stage bindings do not match terminal handoff")

    tokenizer = materialization.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise LedgerError(f"{label}.tokenizer must be an object")
    observed_tokenizer_identity = _require_sha256(
        tokenizer.get("identity_sha256"), f"{label}.tokenizer.identity_sha256"
    )
    if observed_tokenizer_identity != expected_tokenizer_identity_sha256:
        raise LedgerError(f"{label} tokenizer identity does not match terminal handoff")

    _validate_retained_document_isolation(materialization, label=label)
    ledger = build_ledger(materialization)
    verify_ledger(materialization, ledger)
    return ledger, _canonical_json_bytes(materialization)


def verify_deterministic_double_pack(
    build_a: Mapping[str, Any],
    build_b: Mapping[str, Any],
    *,
    terminal_corpus_authority_identity_sha256: str,
    expected_stage_bindings: Mapping[str, Any],
    expected_tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    """Bind two independent post-pack builds to one immutable terminal handoff.

    This function does not create corpus authority and does not authorize training.
    It consumes an externally terminal corpus authority identity plus exact D04 stage
    and tokenizer bindings, requires both builds to carry that same corpus authority,
    validates retained-document isolation and the existing unique-loss ledger, and
    requires canonical byte identity across the two builds.
    """
    terminal_corpus_identity = _require_sha256(
        terminal_corpus_authority_identity_sha256,
        "terminal_corpus_authority_identity_sha256",
    )
    tokenizer_identity = _require_sha256(
        expected_tokenizer_identity_sha256,
        "expected_tokenizer_identity_sha256",
    )
    if not isinstance(expected_stage_bindings, Mapping):
        raise LedgerError("expected_stage_bindings must be an object")
    stage_bindings = _normalize_expected_stage_bindings(expected_stage_bindings)

    if not isinstance(build_a, Mapping) or not isinstance(build_b, Mapping):
        raise LedgerError("independent builds must be mapping materializations")

    ledger_a, bytes_a = _validate_build(
        build_a,
        label="build_a",
        expected_terminal_corpus_identity_sha256=terminal_corpus_identity,
        expected_stage_bindings=stage_bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
    )
    ledger_b, bytes_b = _validate_build(
        build_b,
        label="build_b",
        expected_terminal_corpus_identity_sha256=terminal_corpus_identity,
        expected_stage_bindings=stage_bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
    )

    if bytes_a != bytes_b:
        raise LedgerError("independent post-pack materializations are not byte-identical")
    if ledger_a != ledger_b:
        raise LedgerError("independent post-pack ledgers are not identical")

    materialization_identity = _require_sha256(
        build_a.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    packing = build_a.get("packing")
    if not isinstance(packing, Mapping):
        raise LedgerError("packing must be an object")
    packing_identity = _require_sha256(
        packing.get("identity_sha256"), "packing.identity_sha256"
    )
    ledger_identity = _require_sha256(
        ledger_a.get("ledger_identity_sha256"), "ledger_identity_sha256"
    )
    unique_positions = ledger_a.get(
        "one_pass_unique_nonignored_causal_loss_positions"
    )
    if isinstance(unique_positions, bool) or not isinstance(unique_positions, int):
        raise LedgerError("ledger unique loss position count must be an integer")
    if unique_positions < 0:
        raise LedgerError("ledger unique loss position count must be non-negative")

    canonical_build_sha256 = _sha256_bytes(bytes_a)
    proof: dict[str, Any] = {
        "schema_version": DOUBLE_PACK_PROOF_SCHEMA,
        "terminal_corpus_authority_identity_sha256": terminal_corpus_identity,
        "stage_bindings": stage_bindings,
        "tokenizer_identity_sha256": tokenizer_identity,
        "materialization_identity_sha256": materialization_identity,
        "packing_identity_sha256": packing_identity,
        "ledger_identity_sha256": ledger_identity,
        "canonical_build_sha256": canonical_build_sha256,
        "build_a_canonical_sha256": canonical_build_sha256,
        "build_b_canonical_sha256": _sha256_bytes(bytes_b),
        "one_pass_unique_nonignored_causal_loss_positions": unique_positions,
        "retained_document_isolation_verified": True,
        "heldout_reservation_verified": True,
        "independent_builds_byte_identical": True,
        "training_authorized_by_this_proof": False,
    }
    proof["proof_identity_sha256"] = _sha256_obj(proof)
    return proof
