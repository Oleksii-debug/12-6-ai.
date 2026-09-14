from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger, verify_ledger

DOUBLE_PACK_PROOF_SCHEMA = "12-6.d04-deterministic-double-pack-proof.v1"
_TERMINAL_RECORD_INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
_TERMINAL_SPLIT_APPLICATION_SCHEMA = "12-6.d03-balanced-split-application.v1"
_TERMINAL_SPLIT_FAMILY_SCHEMA = "12-6.validation-split-family.v1"
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
_SPLIT_APPLICATION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "balanced_selection_identity_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "dedup_authority_sha256",
        "balance_policy_identity_sha256",
        "balance_result_identity_sha256",
        "canonical_split_git_blob_sha1",
        "split_spec_identity_sha256",
        "selected_record_count",
        "selected_source_bytes",
        "selected_family_source_bytes",
        "selected_stratum_source_bytes",
        "split_family",
        "claim_boundary",
        "application_identity_sha256",
    }
)
_ZERO_CREDIT_SPLIT_CLAIM_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}


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


def _normalize_record_ids(value: Sequence[Any], *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LedgerError(f"{label} must be a sequence")

    normalized = tuple(
        _require_nonempty_string(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if not normalized:
        raise LedgerError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise LedgerError(f"{label} contains duplicate record_id")
    if normalized != tuple(sorted(normalized)):
        raise LedgerError(f"{label} must be in canonical record_id order")
    return normalized


def _validate_terminal_split_application(
    application: Mapping[str, Any],
    *,
    expected_terminal_split_application_identity_sha256: str,
) -> tuple[str, str, tuple[str, ...]]:
    """Authenticate a text-free terminal split application and derive train IDs.

    The expected application identity is an independent trust root. The application
    may self-hash its own bytes, but a caller cannot replace the canonical shared
    train core and authorize that replacement by merely recomputing the self-hash.
    Raw records are intentionally not consumed at D04: the already-reviewed D03
    split application is the sealed authority passed across this stage boundary.
    """
    if not isinstance(application, Mapping):
        raise LedgerError("terminal_split_application must be an object")
    if set(application) != set(_SPLIT_APPLICATION_FIELDS):
        raise LedgerError("terminal split application fields are not closed-world")
    if application.get("schema") != _TERMINAL_SPLIT_APPLICATION_SCHEMA:
        raise LedgerError("unsupported terminal split application schema")
    if application.get("status") != "PASS_ZERO_CREDIT":
        raise LedgerError("terminal split application is not PASS_ZERO_CREDIT")

    expected_identity = _require_sha256(
        expected_terminal_split_application_identity_sha256,
        "expected_terminal_split_application_identity_sha256",
    )
    observed_identity = _require_sha256(
        application.get("application_identity_sha256"),
        "terminal_split_application.application_identity_sha256",
    )
    application_core = dict(application)
    application_core.pop("application_identity_sha256", None)
    computed_identity = _sha256_bytes(_canonical_json_bytes_no_lf(application_core))
    if observed_identity != computed_identity:
        raise LedgerError("terminal split application self-hash mismatch")
    if observed_identity != expected_identity:
        raise LedgerError("terminal split application does not match expected identity")
    if application.get("claim_boundary") != _ZERO_CREDIT_SPLIT_CLAIM_BOUNDARY:
        raise LedgerError("terminal split application claim boundary widened")

    split_spec_identity = _require_sha256(
        application.get("split_spec_identity_sha256"),
        "terminal_split_application.split_spec_identity_sha256",
    )
    split_family = application.get("split_family")
    if not isinstance(split_family, Mapping):
        raise LedgerError("terminal split application split_family must be an object")
    if split_family.get("schema_version") != _TERMINAL_SPLIT_FAMILY_SCHEMA:
        raise LedgerError("unsupported terminal split-family schema")

    split_family_core = dict(split_family)
    claimed_family_identity = _require_sha256(
        split_family_core.pop("split_family_identity_sha256", None),
        "terminal_split_application.split_family.split_family_identity_sha256",
    )
    if claimed_family_identity != _sha256_obj(split_family_core):
        raise LedgerError("terminal split-family identity/content mismatch")
    if split_family.get("training_policy") != (
        "optimize_shared_train_core_only_excluding_validation_union"
    ):
        raise LedgerError("terminal split-family training policy drift")
    if split_family.get("cluster_straddles_across_variants") != 0:
        raise LedgerError("terminal split-family reports cluster straddles")

    train_record_ids = _normalize_record_ids(
        split_family.get("shared_train_record_ids"),
        label="terminal_split_application.split_family.shared_train_record_ids",
    )
    shared_train_documents = _require_nonnegative_int(
        split_family.get("shared_train_documents"),
        "terminal_split_application.split_family.shared_train_documents",
    )
    if shared_train_documents != len(train_record_ids):
        raise LedgerError("terminal split-family shared train count mismatch")

    validation_union = _normalize_record_ids(
        split_family.get("validation_union_record_ids"),
        label="terminal_split_application.split_family.validation_union_record_ids",
    )
    validation_union_documents = _require_nonnegative_int(
        split_family.get("validation_union_documents"),
        "terminal_split_application.split_family.validation_union_documents",
    )
    if validation_union_documents != len(validation_union):
        raise LedgerError("terminal split-family validation union count mismatch")
    if set(train_record_ids) & set(validation_union):
        raise LedgerError("terminal split-family shared train overlaps validation union")

    return observed_identity, split_spec_identity, train_record_ids


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
) -> tuple[str, ...]:
    """Validate present train records against D03 and return their canonical IDs."""
    documents = materialization.get("documents")
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise LedgerError(f"{label}.documents must be a sequence")

    matched_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise LedgerError(f"{label}.documents[{index}] must be an object")
        if document.get("retained_after_dedup") is not True or document.get("split") != "train":
            continue

        record_id = _require_nonempty_string(
            document.get("document_id"), f"{label}.documents[{index}].document_id"
        )
        if record_id in seen_ids:
            raise LedgerError("duplicate retained train document_id")
        seen_ids.add(record_id)

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
        matched_ids.append(record_id)

    return tuple(sorted(matched_ids))


def _validate_build(
    materialization: Mapping[str, Any],
    *,
    label: str,
    expected_terminal_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    terminal_record_by_id: Mapping[str, Mapping[str, Any]],
    expected_train_record_ids: tuple[str, ...],
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
    matched_train_record_ids = _validate_train_record_membership(
        materialization,
        terminal_record_by_id,
        label=label,
    )
    if matched_train_record_ids != expected_train_record_ids:
        raise LedgerError(
            f"{label} retained train record membership does not match "
            "authenticated terminal split authority"
        )

    ledger = build_ledger(materialization)
    verify_ledger(materialization, ledger)
    return ledger, _canonical_json_bytes(materialization), len(matched_train_record_ids)


def verify_deterministic_double_pack(
    build_a: Mapping[str, Any],
    build_b: Mapping[str, Any],
    *,
    terminal_corpus_authority_identity_sha256: str,
    terminal_record_inventory: Mapping[str, Any],
    expected_record_inventory_digest_sha256: str,
    expected_payload_inventory_digest_sha256: str,
    terminal_split_application: Mapping[str, Any],
    expected_terminal_split_application_identity_sha256: str,
    expected_stage_bindings: Mapping[str, Any],
    expected_tokenizer_identity_sha256: str,
) -> dict[str, Any]:
    """Bind two independent post-pack builds to one immutable terminal handoff.

    D04 consumes the text-free DATA-526 record inventory plus an independently
    identity-pinned D03 split application. The exact shared train record set is
    derived internally from that authenticated split authority, so a caller cannot
    omit a record from both builds and authorize the smaller universe by resealing
    a candidate membership list/digest.
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
    (
        split_application_identity,
        split_spec_identity,
        train_record_ids,
    ) = _validate_terminal_split_application(
        terminal_split_application,
        expected_terminal_split_application_identity_sha256=(
            expected_terminal_split_application_identity_sha256
        ),
    )
    train_membership_digest = _sha256_obj(list(train_record_ids))

    if not isinstance(build_a, Mapping) or not isinstance(build_b, Mapping):
        raise LedgerError("independent builds must be mapping materializations")

    record_by_id = _validate_terminal_record_inventory(
        terminal_record_inventory,
        expected_record_inventory_digest_sha256=expected_record_inventory_digest_sha256,
        expected_payload_inventory_digest_sha256=expected_payload_inventory_digest_sha256,
    )
    for record_id in train_record_ids:
        if record_id not in record_by_id:
            raise LedgerError(
                "authenticated terminal split train record is absent from terminal D03 inventory"
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
        expected_train_record_ids=train_record_ids,
    )
    ledger_b, bytes_b, matched_b = _validate_build(
        build_b,
        label="build_b",
        expected_terminal_corpus_identity_sha256=terminal_corpus_identity,
        expected_stage_bindings=stage_bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
        terminal_record_by_id=record_by_id,
        expected_train_record_ids=train_record_ids,
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
        "terminal_split_application_identity_sha256": split_application_identity,
        "terminal_split_spec_identity_sha256": split_spec_identity,
        "terminal_split_train_record_membership_sha256": train_membership_digest,
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
