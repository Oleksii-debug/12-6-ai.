from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger, verify_ledger

DOUBLE_PACK_PROOF_SCHEMA = "12-6.d04-deterministic-double-pack-proof.v1"
_TERMINAL_RECORD_INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
_REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_RECORD_KEYS = frozenset(
    {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes_no_lf(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
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


def _validate_terminal_record_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate the exact text-free DATA-526 record authority.

    DATA-526 deliberately publishes payload hashes/byte counts rather than payload
    text. D04 consumes that existing projection directly so a materialization cannot
    invent or mutate a training record while retaining an opaque corpus label.
    """
    if not isinstance(inventory, Mapping):
        raise LedgerError("terminal_record_inventory must be an object")
    if inventory.get("schema_version") != _TERMINAL_RECORD_INVENTORY_SCHEMA:
        raise LedgerError("unsupported terminal record inventory schema")

    records = inventory.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise LedgerError("terminal record inventory records must be a sequence")

    normalized_records: list[dict[str, Any]] = []
    record_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
            raise LedgerError(
                f"terminal record inventory records[{index}] fields do not match schema"
            )
        record_id = _require_nonempty_string(
            record["record_id"], f"terminal records[{index}].record_id"
        )
        if record_id in record_by_id:
            raise LedgerError("duplicate terminal record_id")
        normalized = {
            "record_id": record_id,
            "source_id": _require_nonempty_string(
                record["source_id"], f"terminal records[{index}].source_id"
            ),
            "family": _require_nonempty_string(
                record["family"], f"terminal records[{index}].family"
            ),
            "modality": _require_nonempty_string(
                record["modality"], f"terminal records[{index}].modality"
            ),
            "payload_sha256": _require_sha256(
                record["payload_sha256"],
                f"terminal records[{index}].payload_sha256",
            ),
            "payload_bytes": _require_nonnegative_int(
                record["payload_bytes"], f"terminal records[{index}].payload_bytes"
            ),
        }
        normalized_records.append(normalized)
        record_by_id[record_id] = normalized

    normalized_records.sort(key=lambda item: item["record_id"])
    if list(records) != normalized_records:
        raise LedgerError("terminal record inventory must be canonical record_id order")

    declared_count = _require_nonnegative_int(
        inventory.get("record_count"), "terminal_record_inventory.record_count"
    )
    if declared_count != len(normalized_records):
        raise LedgerError("terminal record inventory record_count mismatch")
    declared_bytes = _require_nonnegative_int(
        inventory.get("total_payload_bytes"),
        "terminal_record_inventory.total_payload_bytes",
    )
    if declared_bytes != sum(item["payload_bytes"] for item in normalized_records):
        raise LedgerError("terminal record inventory total_payload_bytes mismatch")

    expected_record_digest = _require_sha256(
        expected_record_inventory_digest_sha256,
        "expected_record_inventory_digest_sha256",
    )
    observed_record_digest = _require_sha256(
        inventory.get("record_inventory_digest_sha256"),
        "terminal_record_inventory.record_inventory_digest_sha256",
    )
    computed_record_digest = _sha256_bytes(
        _canonical_json_bytes_no_lf(normalized_records)
    )
    if observed_record_digest != computed_record_digest:
        raise LedgerError("terminal record inventory self-digest mismatch")
    if observed_record_digest != expected_record_digest:
        raise LedgerError("terminal record inventory does not match expected D03 handoff")

    payload_projection = [
        {
            "record_id": item["record_id"],
            "payload_sha256": item["payload_sha256"],
            "payload_bytes": item["payload_bytes"],
        }
        for item in normalized_records
    ]
    expected_payload_digest = _require_sha256(
        expected_payload_inventory_digest_sha256,
        "expected_payload_inventory_digest_sha256",
    )
    observed_payload_digest = _require_sha256(
        inventory.get("payload_inventory_digest_sha256"),
        "terminal_record_inventory.payload_inventory_digest_sha256",
    )
    computed_payload_digest = _sha256_bytes(
        _canonical_json_bytes_no_lf(payload_projection)
    )
    if observed_payload_digest != computed_payload_digest:
        raise LedgerError("terminal payload inventory self-digest mismatch")
    if observed_payload_digest != expected_payload_digest:
        raise LedgerError("terminal payload inventory does not match expected D03 handoff")

    return record_by_id


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


def _validate_train_record_membership(
    materialization: Mapping[str, Any],
    record_by_id: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> int:
    """Require every retained train document to derive from the terminal D03 inventory."""
    documents = materialization.get("documents")
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise LedgerError(f"{label}.documents must be a sequence")

    matched = 0
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise LedgerError(f"{label}.documents[{index}] must be an object")
        if document.get("retained_after_dedup") is not True or document.get("split") != "train":
            continue

        record_id = _require_nonempty_string(
            document.get("document_id"), f"{label}.documents[{index}].document_id"
        )
        authority = record_by_id.get(record_id)
        if authority is None:
            raise LedgerError("retained train document is absent from terminal D03 inventory")

        source_id = _require_nonempty_string(
            document.get("source_id"), f"{label}.documents[{index}].source_id"
        )
        family = _require_nonempty_string(
            document.get("family_id"), f"{label}.documents[{index}].family_id"
        )
        modality = _require_nonempty_string(
            document.get("modality"), f"{label}.documents[{index}].modality"
        )
        payload_sha256 = _require_sha256(
            document.get("normalized_payload_sha256"),
            f"{label}.documents[{index}].normalized_payload_sha256",
        )
        source_bytes = _require_nonnegative_int(
            document.get("source_bytes"), f"{label}.documents[{index}].source_bytes"
        )

        if source_id != authority["source_id"]:
            raise LedgerError("train document source_id does not match terminal D03 inventory")
        if family != authority["family"]:
            raise LedgerError("train document family does not match terminal D03 inventory")
        if modality != authority["modality"]:
            raise LedgerError("train document modality does not match terminal D03 inventory")
        if payload_sha256 != authority["payload_sha256"]:
            raise LedgerError("train document payload does not match terminal D03 inventory")
        if source_bytes != authority["payload_bytes"]:
            raise LedgerError("train document byte count does not match terminal D03 inventory")
        matched += 1

    return matched


def _validate_build(
    materialization: Mapping[str, Any],
    *,
    label: str,
    expected_terminal_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    terminal_record_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes, int]:
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
    matched_train_records = _validate_train_record_membership(
        materialization,
        terminal_record_by_id,
        label=label,
    )
    ledger = build_ledger(materialization)
    verify_ledger(materialization, ledger)
    return ledger, _canonical_json_bytes(materialization), matched_train_records


def verify_deterministic_double_pack(
    build_a: Mapping[str, Any],
    build_b: Mapping[str, Any],
    *,
    terminal_corpus_authority_identity_sha256: str,
    terminal_record_inventory: Mapping[str, Any],
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    expected_stage_bindings: Mapping[str, Any],
    expected_tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    """Bind two independent post-pack builds to one immutable terminal handoff.

    The proof consumes the existing text-free DATA-526 record inventory and requires
    every retained train document to be a payload/source/family/modality/byte-exact
    member of it. This prevents two mutually identical, self-rehashed builds from
    manufacturing a different training corpus behind a copied corpus identity.
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

    record_by_id = _validate_terminal_record_inventory(
        terminal_record_inventory,
        expected_record_inventory_digest_sha256=expected_record_inventory_digest_sha256,
        expected_payload_inventory_digest_sha256=expected_payload_inventory_digest_sha256,
    )
    record_inventory_digest = _require_sha256(
        terminal_record_inventory.get("record_inventory_digest_sha256"),
        "terminal_record_inventory.record_inventory_digest_sha256",
    )
    payload_inventory_digest = _require_sha256(
        terminal_record_inventory.get("payload_inventory_digest_sha256"),
        "terminal_record_inventory.payload_inventory_digest_sha256",
    )

    ledger_a, bytes_a, matched_a = _validate_build(
        build_a,
        label="build_a",
        expected_terminal_corpus_identity_sha256=terminal_corpus_identity,
        expected_stage_bindings=stage_bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
        terminal_record_by_id=record_by_id,
    )
    ledger_b, bytes_b, matched_b = _validate_build(
        build_b,
        label="build_b",
        expected_terminal_corpus_identity_sha256=terminal_corpus_identity,
        expected_stage_bindings=stage_bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
        terminal_record_by_id=record_by_id,
    )

    if bytes_a != bytes_b:
        raise LedgerError("independent post-pack materializations are not byte-identical")
    if ledger_a != ledger_b:
        raise LedgerError("independent post-pack ledgers are not identical")
    if matched_a != matched_b:
        raise LedgerError("independent builds disagree on terminal train-record membership")

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
        "terminal_record_inventory_digest_sha256": record_inventory_digest,
        "terminal_payload_inventory_digest_sha256": payload_inventory_digest,
        "stage_bindings": stage_bindings,
        "tokenizer_identity_sha256": tokenizer_identity,
        "materialization_identity_sha256": materialization_identity,
        "packing_identity_sha256": packing_identity,
        "ledger_identity_sha256": ledger_identity,
        "canonical_build_sha256": canonical_build_sha256,
        "build_a_canonical_sha256": canonical_build_sha256,
        "build_b_canonical_sha256": _sha256_bytes(bytes_b),
        "one_pass_unique_nonignored_causal_loss_positions": unique_positions,
        "retained_train_records_matched_to_terminal_inventory": matched_a,
        "retained_train_record_membership_verified": True,
        "retained_document_isolation_verified": True,
        "heldout_reservation_verified": True,
        "independent_builds_byte_identical": True,
        "training_authorized_by_this_proof": False,
    }
    proof["proof_identity_sha256"] = _sha256_obj(proof)
    return proof
