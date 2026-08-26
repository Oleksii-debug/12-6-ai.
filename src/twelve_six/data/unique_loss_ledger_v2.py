from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

LEDGER_SCHEMA = "12-6.unique-loss-position-ledger.v2"
MATERIALIZATION_SCHEMA = "12-6.postpack-loss-materialization.v2"
EXPOSURE_STATE_SCHEMA = "12-6.unique-loss-exposure-state.v2"
POSITION_POLICY = "logical-causal-token-target-postpack-v2"
REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)


class LedgerError(ValueError):
    pass


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


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
    return value


def _normalize_ranges(
    ranges: Any,
    *,
    lower: int,
    upper: int,
    label: str,
) -> list[tuple[int, int]]:
    if not isinstance(ranges, list):
        raise LedgerError(f"{label} must be a list")
    normalized: list[tuple[int, int]] = []
    previous_end = lower
    for index, item in enumerate(ranges):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
        ):
            raise LedgerError(f"{label}[{index}] must be [start,end]")
        start, end = item
        if start < lower or end > upper or start >= end:
            raise LedgerError(
                f"{label}[{index}] is outside [{lower},{upper}) or empty"
            )
        if normalized and start < previous_end:
            raise LedgerError(f"{label} must be sorted and non-overlapping")
        normalized.append((start, end))
        previous_end = end
    return normalized


def _subtract_ranges(
    base_start: int,
    base_end: int,
    removed: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    cursor = base_start
    kept: list[tuple[int, int]] = []
    for start, end in removed:
        if cursor < start:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < base_end:
        kept.append((cursor, base_end))
    return kept


def _range_length(ranges: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _contains_range(
    ranges: Sequence[tuple[int, int]], start: int, end: int
) -> bool:
    return any(
        parent_start <= start and end <= parent_end
        for parent_start, parent_end in ranges
    )


def _materialization_identity(materialization: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(materialization))
    payload.pop("materialization_identity_sha256", None)
    return _sha256_obj(payload)


def _validate_stage_bindings(materialization: Mapping[str, Any]) -> dict[str, str]:
    bindings = materialization.get("stage_bindings")
    if not isinstance(bindings, dict):
        raise LedgerError("stage_bindings must be an object")
    if set(bindings) != set(REQUIRED_STAGE_BINDINGS):
        raise LedgerError(
            "stage_bindings must contain exactly normalization, "
            "evaluation_reservations, dedup, split and packing"
        )
    return {
        name: _require_sha256(bindings[name], f"stage_bindings.{name}")
        for name in REQUIRED_STAGE_BINDINGS
    }


def build_ledger(
    materialization: Mapping[str, Any],
    *,
    require_complete_one_pass: bool = True,
) -> dict[str, Any]:
    if materialization.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise LedgerError(f"schema_version must be {MATERIALIZATION_SCHEMA}")

    expected_materialization_id = _materialization_identity(materialization)
    observed_materialization_id = _require_sha256(
        materialization.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    if observed_materialization_id != expected_materialization_id:
        raise LedgerError("materialization_identity_sha256 mismatch")

    stage_bindings = _validate_stage_bindings(materialization)
    tokenizer = materialization.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise LedgerError("tokenizer must be an object")
    tokenizer_name = _require_nonempty_string(tokenizer.get("name"), "tokenizer.name")
    tokenizer_identity = _require_sha256(
        tokenizer.get("identity_sha256"), "tokenizer.identity_sha256"
    )
    if tokenizer.get("source_bytes_are_loss_positions") is not False:
        raise LedgerError("tokenizer.source_bytes_are_loss_positions must be false")

    documents = materialization.get("documents")
    if not isinstance(documents, list) or not documents:
        raise LedgerError("documents must be a non-empty list")

    doc_by_id: dict[str, dict[str, Any]] = {}
    eligible_by_doc: dict[str, list[tuple[int, int]]] = {}
    retained_cluster_owner: dict[str, str] = {}
    eligible_total = 0

    for index, raw_document in enumerate(documents):
        if not isinstance(raw_document, dict):
            raise LedgerError(f"documents[{index}] must be an object")
        doc = dict(raw_document)
        document_id = _require_nonempty_string(
            doc.get("document_id"), f"documents[{index}].document_id"
        )
        if document_id in doc_by_id:
            raise LedgerError(f"duplicate document_id: {document_id}")
        language = _require_nonempty_string(
            doc.get("language"), f"documents[{index}].language"
        )
        modality = _require_nonempty_string(
            doc.get("modality"), f"documents[{index}].modality"
        )
        if modality not in {"text", "code"}:
            raise LedgerError(f"unsupported modality for {document_id}: {modality}")
        family_id = _require_nonempty_string(
            doc.get("family_id"), f"documents[{index}].family_id"
        )
        _require_sha256(
            doc.get("normalized_payload_sha256"),
            f"documents[{index}].normalized_payload_sha256",
        )
        token_count = _require_nonnegative_int(
            doc.get("token_count"), f"documents[{index}].token_count"
        )
        split = _require_nonempty_string(doc.get("split"), f"documents[{index}].split")
        dedup_cluster_id = _require_nonempty_string(
            doc.get("dedup_cluster_id"),
            f"documents[{index}].dedup_cluster_id",
        )
        retained_after_dedup = doc.get("retained_after_dedup")
        if not isinstance(retained_after_dedup, bool):
            raise LedgerError(
                f"documents[{index}].retained_after_dedup must be boolean"
            )
        evaluation_reserved = doc.get("evaluation_reserved")
        if not isinstance(evaluation_reserved, bool):
            raise LedgerError(
                f"documents[{index}].evaluation_reserved must be boolean"
            )

        reservations = _normalize_ranges(
            doc.get("reserved_target_ranges"),
            lower=1,
            upper=max(token_count, 1),
            label=f"documents[{index}].reserved_target_ranges",
        )
        if any(end > token_count for _, end in reservations):
            raise LedgerError(f"reservation exceeds token_count for {document_id}")

        if (
            split == "train"
            and retained_after_dedup
            and not evaluation_reserved
            and token_count > 1
        ):
            eligible = _subtract_ranges(1, token_count, reservations)
            previous_owner = retained_cluster_owner.get(dedup_cluster_id)
            if previous_owner is not None and previous_owner != document_id:
                raise LedgerError(
                    "multiple retained train documents share dedup cluster "
                    f"{dedup_cluster_id}: {previous_owner}, {document_id}"
                )
            retained_cluster_owner[dedup_cluster_id] = document_id
        else:
            eligible = []

        declared_eligible = _normalize_ranges(
            doc.get("eligible_target_ranges"),
            lower=1,
            upper=max(token_count, 1),
            label=f"documents[{index}].eligible_target_ranges",
        )
        if declared_eligible != eligible:
            raise LedgerError(
                f"eligible_target_ranges mismatch for {document_id}; "
                "document boundary, reservation, dedup and split semantics are authoritative"
            )

        doc["_validated_language"] = language
        doc["_validated_modality"] = modality
        doc["_validated_family_id"] = family_id
        doc_by_id[document_id] = doc
        eligible_by_doc[document_id] = eligible
        eligible_total += _range_length(eligible)

    packing = materialization.get("packing")
    if not isinstance(packing, dict):
        raise LedgerError("packing must be an object")
    complete_one_pass = packing.get("complete_one_pass")
    if not isinstance(complete_one_pass, bool):
        raise LedgerError("packing.complete_one_pass must be boolean")
    packing_identity = _require_sha256(
        packing.get("identity_sha256"), "packing.identity_sha256"
    )
    packs = packing.get("packs")
    if not isinstance(packs, list):
        raise LedgerError("packing.packs must be a list")

    logical_targets_seen: set[tuple[str, int]] = set()
    segments: list[dict[str, Any]] = []
    pack_ids: set[str] = set()
    total_count = 0
    by_language: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    by_family: dict[str, int] = {}

    for pack_index, raw_pack in enumerate(packs):
        if not isinstance(raw_pack, dict):
            raise LedgerError(f"packing.packs[{pack_index}] must be an object")
        pack_id = _require_nonempty_string(
            raw_pack.get("pack_id"), f"packing.packs[{pack_index}].pack_id"
        )
        if pack_id in pack_ids:
            raise LedgerError(f"duplicate pack_id: {pack_id}")
        pack_ids.add(pack_id)
        pack_token_count = _require_nonnegative_int(
            raw_pack.get("token_count"),
            f"packing.packs[{pack_index}].token_count",
        )
        spans = raw_pack.get("loss_spans")
        if not isinstance(spans, list):
            raise LedgerError(f"packing.packs[{pack_index}].loss_spans must be a list")

        occupied_pack_slots: list[tuple[int, int]] = []
        for span_index, raw_span in enumerate(spans):
            if not isinstance(raw_span, dict):
                raise LedgerError(
                    f"packing.packs[{pack_index}].loss_spans[{span_index}] "
                    "must be an object"
                )
            document_id = _require_nonempty_string(
                raw_span.get("document_id"),
                f"packing.packs[{pack_index}].loss_spans[{span_index}].document_id",
            )
            if document_id not in doc_by_id:
                raise LedgerError(f"pack references unknown document {document_id}")
            target_start = _require_nonnegative_int(
                raw_span.get("target_start"),
                f"packing.packs[{pack_index}].loss_spans[{span_index}].target_start",
            )
            target_end = _require_nonnegative_int(
                raw_span.get("target_end"),
                f"packing.packs[{pack_index}].loss_spans[{span_index}].target_end",
            )
            pack_target_start = _require_nonnegative_int(
                raw_span.get("pack_target_start"),
                f"packing.packs[{pack_index}].loss_spans[{span_index}].pack_target_start",
            )
            if target_start < 1 or target_start >= target_end:
                raise LedgerError("loss span target range must be non-empty and start >= 1")
            length = target_end - target_start
            pack_target_end = pack_target_start + length
            if pack_target_start < 1 or pack_target_end > pack_token_count:
                raise LedgerError(
                    f"loss span exceeds pack target slots for pack {pack_id}"
                )
            if not _contains_range(
                eligible_by_doc[document_id], target_start, target_end
            ):
                raise LedgerError(
                    f"loss span includes non-eligible/reserved target(s) for {document_id}"
                )
            for occupied_start, occupied_end in occupied_pack_slots:
                if pack_target_start < occupied_end and occupied_start < pack_target_end:
                    raise LedgerError(f"overlapping loss slots in pack {pack_id}")
            occupied_pack_slots.append((pack_target_start, pack_target_end))

            for logical_target in range(target_start, target_end):
                key = (document_id, logical_target)
                if key in logical_targets_seen:
                    raise LedgerError(
                        "logical causal target replayed during packing: "
                        f"{document_id}:{logical_target}"
                    )
                logical_targets_seen.add(key)

            doc = doc_by_id[document_id]
            segment_core = {
                "document_id": document_id,
                "target_start": target_start,
                "target_end": target_end,
                "pack_id": pack_id,
                "pack_target_start": pack_target_start,
                "pack_target_end": pack_target_end,
                "normalized_payload_sha256": doc["normalized_payload_sha256"],
            }
            segment_id = _sha256_obj(
                {
                    "position_policy": POSITION_POLICY,
                    "materialization_identity_sha256": observed_materialization_id,
                    **segment_core,
                }
            )
            segment = {
                "segment_identity_sha256": segment_id,
                **segment_core,
                "language": doc["_validated_language"],
                "modality": doc["_validated_modality"],
                "family_id": doc["_validated_family_id"],
                "loss_position_count": length,
            }
            segments.append(segment)
            total_count += length
            by_language[segment["language"]] = (
                by_language.get(segment["language"], 0) + length
            )
            by_modality[segment["modality"]] = (
                by_modality.get(segment["modality"], 0) + length
            )
            by_family[segment["family_id"]] = (
                by_family.get(segment["family_id"], 0) + length
            )

    packed_target_count = len(logical_targets_seen)
    if packed_target_count != total_count:
        raise LedgerError("internal packed target counting mismatch")
    eligible_not_packed = eligible_total - packed_target_count
    if eligible_not_packed < 0:
        raise LedgerError("packed target count exceeds eligible target count")
    if complete_one_pass and eligible_not_packed != 0:
        raise LedgerError(
            "packing.complete_one_pass=true but eligible causal targets are missing"
        )
    if require_complete_one_pass and not complete_one_pass:
        raise LedgerError("terminal ledger requires packing.complete_one_pass=true")

    segments.sort(
        key=lambda item: (
            item["pack_id"],
            item["pack_target_start"],
            item["document_id"],
            item["target_start"],
        )
    )

    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA,
        "position_policy": POSITION_POLICY,
        "materialization_identity_sha256": observed_materialization_id,
        "stage_bindings": stage_bindings,
        "tokenizer": {
            "name": tokenizer_name,
            "identity_sha256": tokenizer_identity,
            "source_bytes_are_loss_positions": False,
        },
        "packing_identity_sha256": packing_identity,
        "complete_one_pass": complete_one_pass,
        "eligible_causal_targets_before_packing": eligible_total,
        "one_pass_unique_nonignored_causal_loss_positions": total_count,
        "eligible_targets_not_packed": eligible_not_packed,
        "by_language": dict(sorted(by_language.items())),
        "by_modality": dict(sorted(by_modality.items())),
        "by_family": dict(sorted(by_family.items())),
        "segments": segments,
        "padding_loss_positions": 0,
        "cross_document_loss_positions": 0,
        "source_bytes_relabelled_as_loss_positions": False,
    }
    ledger["ledger_identity_sha256"] = _sha256_obj(ledger)
    return ledger


def verify_ledger(
    materialization: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    require_complete_one_pass: bool = True,
) -> None:
    rebuilt = build_ledger(
        materialization, require_complete_one_pass=require_complete_one_pass
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(dict(ledger)):
        raise LedgerError("ledger does not match deterministic rebuild")


def count_nonignored_targets(loss_mask: Any) -> int:
    if hasattr(loss_mask, "tolist"):
        loss_mask = loss_mask.tolist()

    def walk(value: Any) -> int:
        if isinstance(value, (list, tuple)):
            return sum(walk(item) for item in value)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            if value not in (0, 1):
                raise LedgerError("integer loss mask values must be 0 or 1")
            return value
        if isinstance(value, float):
            if value not in (0.0, 1.0):
                raise LedgerError("float loss mask values must be 0.0 or 1.0")
            return int(value)
        raise LedgerError("unsupported loss_mask value")

    return walk(loss_mask)


class ExposureReplayGuard:
    def __init__(
        self,
        ledger: Mapping[str, Any],
        *,
        authorized_budget: int,
        trainer_state_binding: Mapping[str, Any],
    ) -> None:
        if ledger.get("schema_version") != LEDGER_SCHEMA:
            raise LedgerError("guard requires a V2 ledger")
        self.ledger_identity_sha256 = _require_sha256(
            ledger.get("ledger_identity_sha256"), "ledger_identity_sha256"
        )
        self.materialization_identity_sha256 = _require_sha256(
            ledger.get("materialization_identity_sha256"),
            "materialization_identity_sha256",
        )
        self.packing_identity_sha256 = _require_sha256(
            ledger.get("packing_identity_sha256"), "packing_identity_sha256"
        )
        self.one_pass_maximum = _require_nonnegative_int(
            ledger.get("one_pass_unique_nonignored_causal_loss_positions"),
            "one_pass_unique_nonignored_causal_loss_positions",
        )
        self.authorized_budget = _require_nonnegative_int(
            authorized_budget, "authorized_budget"
        )
        if self.authorized_budget > self.one_pass_maximum:
            raise LedgerError("authorized_budget exceeds ledger one-pass maximum")
        self._segments: dict[str, int] = {}
        for segment in ledger.get("segments", []):
            segment_id = _require_sha256(
                segment.get("segment_identity_sha256"), "segment_identity_sha256"
            )
            length = _require_nonnegative_int(
                segment.get("loss_position_count"), "loss_position_count"
            )
            if segment_id in self._segments:
                raise LedgerError("duplicate ledger segment identity")
            self._segments[segment_id] = length
        self._claims: dict[str, list[tuple[int, int]]] = {
            segment_id: [] for segment_id in self._segments
        }
        self.consumed_loss_positions = 0
        self.claim_sequence = 0
        self.trainer_state_binding = self._validate_trainer_binding(
            trainer_state_binding
        )

    @staticmethod
    def _validate_trainer_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(binding, Mapping):
            raise LedgerError("trainer_state_binding must be an object")
        required = {
            "checkpoint_generation",
            "checkpoint_manifest_sha256",
            "optimizer_step",
            "trainer_nonignored_target_count",
        }
        if set(binding) != required:
            raise LedgerError(
                "trainer_state_binding must contain exactly checkpoint_generation, "
                "checkpoint_manifest_sha256, optimizer_step and "
                "trainer_nonignored_target_count"
            )
        return {
            "checkpoint_generation": _require_nonempty_string(
                binding["checkpoint_generation"], "checkpoint_generation"
            ),
            "checkpoint_manifest_sha256": _require_sha256(
                binding["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256"
            ),
            "optimizer_step": _require_nonnegative_int(
                binding["optimizer_step"], "optimizer_step"
            ),
            "trainer_nonignored_target_count": _require_nonnegative_int(
                binding["trainer_nonignored_target_count"],
                "trainer_nonignored_target_count",
            ),
        }

    @staticmethod
    def _insert_interval(
        existing: list[tuple[int, int]], start: int, end: int
    ) -> list[tuple[int, int]]:
        for old_start, old_end in existing:
            if start < old_end and old_start < end:
                raise LedgerError("replay/overlapping loss-position claim")
        return sorted([*existing, (start, end)])

    def authorize_batch(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        actual_nonignored_targets: int,
    ) -> None:
        actual = _require_nonnegative_int(
            actual_nonignored_targets, "actual_nonignored_targets"
        )
        tentative = {key: list(value) for key, value in self._claims.items()}
        claimed_count = 0
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise LedgerError(f"claims[{index}] must be an object")
            segment_id = _require_sha256(
                claim.get("segment_identity_sha256"),
                f"claims[{index}].segment_identity_sha256",
            )
            if segment_id not in self._segments:
                raise LedgerError("claim references unknown ledger segment")
            start = _require_nonnegative_int(
                claim.get("offset_start"), f"claims[{index}].offset_start"
            )
            end = _require_nonnegative_int(
                claim.get("offset_end"), f"claims[{index}].offset_end"
            )
            if start >= end or end > self._segments[segment_id]:
                raise LedgerError("claim interval is outside ledger segment")
            tentative[segment_id] = self._insert_interval(
                tentative[segment_id], start, end
            )
            claimed_count += end - start

        if claimed_count != actual:
            raise LedgerError(
                "claimed loss-position count does not match actual nonignored target count"
            )
        if self.consumed_loss_positions + claimed_count > self.authorized_budget:
            raise LedgerError("batch would exceed authorized exposure budget")

        self._claims = tentative
        self.consumed_loss_positions += claimed_count
        self.claim_sequence += 1

    def authorize_loss_mask(
        self,
        claims: Sequence[Mapping[str, Any]],
        *,
        loss_mask: Any,
    ) -> None:
        self.authorize_batch(
            claims,
            actual_nonignored_targets=count_nonignored_targets(loss_mask),
        )

    def bind_checkpoint_state(
        self, trainer_state_binding: Mapping[str, Any]
    ) -> None:
        binding = self._validate_trainer_binding(trainer_state_binding)
        if binding["trainer_nonignored_target_count"] != self.consumed_loss_positions:
            raise LedgerError(
                "trainer target counter must equal consumed unique loss positions"
            )
        self.trainer_state_binding = binding

    def _state_without_hash(self) -> dict[str, Any]:
        claims = {
            segment_id: [[start, end] for start, end in intervals]
            for segment_id, intervals in sorted(self._claims.items())
            if intervals
        }
        return {
            "schema_version": EXPOSURE_STATE_SCHEMA,
            "ledger_identity_sha256": self.ledger_identity_sha256,
            "materialization_identity_sha256": self.materialization_identity_sha256,
            "packing_identity_sha256": self.packing_identity_sha256,
            "authorized_budget": self.authorized_budget,
            "one_pass_maximum": self.one_pass_maximum,
            "consumed_loss_positions": self.consumed_loss_positions,
            "claim_sequence": self.claim_sequence,
            "claims": claims,
            "trainer_state_binding": self.trainer_state_binding,
        }

    def state_dict(self) -> dict[str, Any]:
        state = self._state_without_hash()
        state["state_identity_sha256"] = _sha256_obj(state)
        return state

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_trainer_state_binding: Mapping[str, Any],
    ) -> None:
        if state.get("schema_version") != EXPOSURE_STATE_SCHEMA:
            raise LedgerError("exposure state schema mismatch")
        state_copy = deepcopy(dict(state))
        observed_hash = _require_sha256(
            state_copy.pop("state_identity_sha256", None),
            "state_identity_sha256",
        )
        if _sha256_obj(state_copy) != observed_hash:
            raise LedgerError("exposure state self-hash mismatch")
        if state_copy.get("ledger_identity_sha256") != self.ledger_identity_sha256:
            raise LedgerError("resume ledger identity mismatch")
        if (
            state_copy.get("materialization_identity_sha256")
            != self.materialization_identity_sha256
        ):
            raise LedgerError("resume materialization identity mismatch")
        if state_copy.get("packing_identity_sha256") != self.packing_identity_sha256:
            raise LedgerError("resume packing identity mismatch")
        if state_copy.get("authorized_budget") != self.authorized_budget:
            raise LedgerError("resume authorized budget mismatch")
        if state_copy.get("one_pass_maximum") != self.one_pass_maximum:
            raise LedgerError("resume one-pass maximum mismatch")

        expected_binding = self._validate_trainer_binding(
            expected_trainer_state_binding
        )
        saved_binding = self._validate_trainer_binding(
            state_copy.get("trainer_state_binding")
        )
        if saved_binding != expected_binding:
            raise LedgerError("resume trainer/checkpoint state binding mismatch")

        claims = state_copy.get("claims")
        if not isinstance(claims, dict):
            raise LedgerError("resume claims must be an object")
        tentative: dict[str, list[tuple[int, int]]] = {
            key: [] for key in self._segments
        }
        recomputed_count = 0
        for segment_id, intervals in claims.items():
            if segment_id not in self._segments:
                raise LedgerError("resume state references unknown segment")
            normalized = _normalize_ranges(
                intervals,
                lower=0,
                upper=self._segments[segment_id],
                label=f"resume.claims.{segment_id}",
            )
            tentative[segment_id] = normalized
            recomputed_count += _range_length(normalized)
        consumed = _require_nonnegative_int(
            state_copy.get("consumed_loss_positions"),
            "consumed_loss_positions",
        )
        if recomputed_count != consumed:
            raise LedgerError("resume consumed count does not match claimed intervals")
        if consumed > self.authorized_budget:
            raise LedgerError("resume consumed count exceeds authorized budget")
        if saved_binding["trainer_nonignored_target_count"] != consumed:
            raise LedgerError(
                "resume trainer target counter differs from consumed unique positions"
            )
        claim_sequence = _require_nonnegative_int(
            state_copy.get("claim_sequence"), "claim_sequence"
        )

        self._claims = tentative
        self.consumed_loss_positions = consumed
        self.claim_sequence = claim_sequence
        self.trainer_state_binding = saved_binding
