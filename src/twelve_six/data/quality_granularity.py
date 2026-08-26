"""Frozen quality-filter granularity policy for NEXT100-067.

This module changes only the authority boundary around the existing deterministic
document-quality metrics. It never reads model results, evaluation outcomes,
rights state, privacy state, or final-test payloads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from twelve_six.data.document_quality import (
    QualityDecision,
    QualityPolicy,
    assess_document,
    default_quality_policy,
)

Mode = Literal["uk", "en", "code"]
GRANULARITY_POLICY_SCHEMA = "12-6.quality-filter-granularity-policy.v1"
FROZEN_POLICY_ID = "next100-067-document-first-local-window-salvage-v1"
FROZEN_POLICY_SHA256 = "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"


class QualityGranularityError(ValueError):
    """Raised when the frozen granularity contract is malformed."""


def _cjson(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class QualityWindow:
    index: int
    start_char: int
    end_char: int
    text: str

    @property
    def chars(self) -> int:
        return self.end_char - self.start_char

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass(frozen=True)
class FrozenGranularityPolicy:
    policy_id: str = FROZEN_POLICY_ID
    quality_threshold_policy_id: str = "d03-lightweight-uk-en-code-v1"
    quality_threshold_policy_sha256: str = (
        "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
    )
    source_soft_rejection_authority: bool = False
    source_family_eviction_authority: bool = False
    document_primary_quality_unit: bool = True
    document_require_source_native_boundary: bool = True
    document_soft_rejection_authority: bool = True
    document_family_eviction_authority: bool = False
    natural_window_trigger_chars: int = 16_384
    natural_window_target_chars: int = 8_192
    natural_window_max_chars: int = 12_288
    natural_window_min_tail_chars: int = 2_048
    natural_window_partition: str = (
        "ORDERED_LINE_PREFERRED_NO_OVERLAP_EXACT_RECONSTRUCTION_V1"
    )
    natural_accepted_windows_retained_independently: bool = True
    natural_rejected_window_rejects_siblings: bool = False
    code_atomic_unit: str = "SOURCE_NATIVE_FILE_OR_SYNTAX_AWARE_UPSTREAM_UNIT"
    code_arbitrary_pack_rejection_authority: bool = False
    code_oversized_without_syntax_split: str = "QUARANTINE_DOCUMENT_NOT_FAMILY"
    post_packing_quality_rejection_authority: bool = False
    schema_version: str = GRANULARITY_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GRANULARITY_POLICY_SCHEMA:
            raise QualityGranularityError("unsupported granularity policy schema")
        if self.policy_id != FROZEN_POLICY_ID:
            raise QualityGranularityError("frozen policy id drift")
        if self.source_soft_rejection_authority:
            raise QualityGranularityError("source-level soft quality rejection is forbidden")
        if self.source_family_eviction_authority or self.document_family_eviction_authority:
            raise QualityGranularityError("quality filtering may not evict a family")
        if self.code_arbitrary_pack_rejection_authority:
            raise QualityGranularityError("arbitrary code packs may not reject code")
        if self.post_packing_quality_rejection_authority:
            raise QualityGranularityError("post-packing quality filtering is forbidden")
        if not (
            0 < self.natural_window_min_tail_chars
            < self.natural_window_target_chars
            <= self.natural_window_max_chars
            < self.natural_window_trigger_chars
        ):
            raise QualityGranularityError("invalid natural-language window bounds")

    def manifest_core(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "quality_threshold_policy_id": self.quality_threshold_policy_id,
            "quality_threshold_policy_sha256": self.quality_threshold_policy_sha256,
            "whole_source": {
                "soft_quality_rejection_authority": self.source_soft_rejection_authority,
                "purpose": "DIAGNOSTIC_ONLY",
                "family_eviction_authority": self.source_family_eviction_authority,
            },
            "document": {
                "primary_quality_unit": self.document_primary_quality_unit,
                "require_source_native_boundary": self.document_require_source_native_boundary,
                "soft_quality_rejection_authority": self.document_soft_rejection_authority,
                "family_eviction_authority": self.document_family_eviction_authority,
            },
            "natural_language_large_document": {
                "modes": ["uk", "en"],
                "window_trigger_chars": self.natural_window_trigger_chars,
                "window_target_chars": self.natural_window_target_chars,
                "window_max_chars": self.natural_window_max_chars,
                "window_min_tail_chars": self.natural_window_min_tail_chars,
                "partition": self.natural_window_partition,
                "accepted_windows_are_retained_independently": (
                    self.natural_accepted_windows_retained_independently
                ),
                "rejected_windows_do_not_reject_siblings": (
                    not self.natural_rejected_window_rejects_siblings
                ),
                "family_eviction_authority": False,
            },
            "code": {
                "atomic_unit": self.code_atomic_unit,
                "arbitrary_bounded_pack_soft_rejection_authority": (
                    self.code_arbitrary_pack_rejection_authority
                ),
                "arbitrary_bounded_pack_purpose": "DIAGNOSTIC_ONLY",
                "oversized_without_syntax_aware_split": (
                    self.code_oversized_without_syntax_split
                ),
                "family_eviction_authority": False,
            },
            "post_packing": {
                "quality_rejection_authority": self.post_packing_quality_rejection_authority,
                "reason": "TRAINING_PACK_BOUNDARIES_ARE_NOT_QUALITY_OR_SEMANTIC_BOUNDARIES",
            },
        }

    def manifest(self) -> dict[str, Any]:
        core = self.manifest_core()
        return {**core, "policy_sha256": _sha256(_cjson(core))}

    def verify_frozen_identity(self) -> None:
        observed = self.manifest()["policy_sha256"]
        if observed != FROZEN_POLICY_SHA256:
            raise QualityGranularityError(
                f"frozen granularity identity drift: expected {FROZEN_POLICY_SHA256}, got {observed}"
            )


def frozen_granularity_policy() -> FrozenGranularityPolicy:
    policy = FrozenGranularityPolicy()
    policy.verify_frozen_identity()
    return policy


def _preferred_cut(text: str, start: int, target_end: int, max_end: int) -> int:
    """Prefer a newline near/after the target; fall back to an exact char boundary."""
    if target_end >= len(text):
        return len(text)
    forward = text.find("\n", target_end, max_end)
    if forward >= 0:
        return forward + 1
    lower = start + max((target_end - start) // 2, 1)
    backward = text.rfind("\n", lower, target_end)
    if backward >= 0:
        return backward + 1
    return target_end


def natural_quality_windows(
    text: str,
    *,
    policy: FrozenGranularityPolicy | None = None,
) -> list[QualityWindow]:
    """Return deterministic local windows with exact no-overlap reconstruction."""
    granularity = policy or frozen_granularity_policy()
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if not text:
        return []

    target = granularity.natural_window_target_chars
    maximum = granularity.natural_window_max_chars
    min_tail = granularity.natural_window_min_tail_chars
    spans: list[tuple[int, int]] = []
    start = 0
    while len(text) - start > maximum:
        target_end = min(start + target, len(text))
        max_end = min(start + maximum, len(text))
        end = _preferred_cut(text, start, target_end, max_end)
        if end <= start or end > max_end:
            raise QualityGranularityError("invalid quality-window cut")
        spans.append((start, end))
        start = end
    spans.append((start, len(text)))

    if len(spans) > 1:
        tail_start, tail_end = spans[-1]
        if tail_end - tail_start < min_tail:
            prev_start, prev_end = spans[-2]
            if tail_end - prev_start <= maximum:
                spans[-2:] = [(prev_start, tail_end)]
            else:
                needed = min_tail - (tail_end - tail_start)
                new_boundary = prev_end - needed
                if new_boundary <= prev_start:
                    raise QualityGranularityError("cannot rebalance short tail")
                spans[-2:] = [(prev_start, new_boundary), (new_boundary, tail_end)]

    windows = [
        QualityWindow(index=i, start_char=start, end_char=end, text=text[start:end])
        for i, (start, end) in enumerate(spans)
    ]
    if "".join(window.text for window in windows) != text:
        raise QualityGranularityError("quality windows do not reconstruct the document")
    if any(window.end_char <= window.start_char for window in windows):
        raise QualityGranularityError("empty quality window")
    if any(window.chars > maximum for window in windows):
        raise QualityGranularityError("quality window exceeded maximum bound")
    return windows


def _decision_dict(decision: QualityDecision) -> dict[str, Any]:
    return {
        "accepted": decision.accepted,
        "score": decision.score,
        "reasons": list(decision.reasons),
        "warnings": list(decision.warnings),
        "edge_margin": decision.edge_margin,
        "features": asdict(decision.features),
    }


def apply_frozen_granularity(
    record_id: str,
    text: str,
    mode: Mode,
    *,
    quality_policy: QualityPolicy | None = None,
    granularity_policy: FrozenGranularityPolicy | None = None,
) -> dict[str, Any]:
    """Apply the frozen deletion-authority policy to one source-native document."""
    if mode not in {"uk", "en", "code"}:
        raise QualityGranularityError(f"unsupported mode: {mode!r}")
    granularity = granularity_policy or frozen_granularity_policy()
    quality = quality_policy or default_quality_policy()
    if quality.manifest()["policy_sha256"] != granularity.quality_threshold_policy_sha256:
        raise QualityGranularityError("quality threshold policy drift is forbidden")

    full = assess_document(record_id, text, mode, policy=quality)
    total_bytes = len(text.encode("utf-8"))

    if mode == "code":
        status = "RETAIN_ALL" if full.accepted else "REJECT_DOCUMENT"
        return {
            "record_id": record_id,
            "mode": mode,
            "authoritative_unit": "DOCUMENT",
            "status": status,
            "family_eviction_authority": False,
            "retained_utf8_bytes": total_bytes if full.accepted else 0,
            "rejected_utf8_bytes": 0 if full.accepted else total_bytes,
            "full_document_diagnostic": _decision_dict(full),
            "windows": [],
        }

    if len(text) <= granularity.natural_window_trigger_chars:
        status = "RETAIN_ALL" if full.accepted else "REJECT_DOCUMENT"
        return {
            "record_id": record_id,
            "mode": mode,
            "authoritative_unit": "DOCUMENT",
            "status": status,
            "family_eviction_authority": False,
            "retained_utf8_bytes": total_bytes if full.accepted else 0,
            "rejected_utf8_bytes": 0 if full.accepted else total_bytes,
            "full_document_diagnostic": _decision_dict(full),
            "windows": [],
        }

    windows = natural_quality_windows(text, policy=granularity)
    rows: list[dict[str, Any]] = []
    retained = 0
    rejected = 0
    accepted_count = 0
    for window in windows:
        decision = assess_document(
            f"{record_id}#quality-window-{window.index:04d}",
            window.text,
            mode,
            policy=quality,
        )
        window_bytes = window.utf8_bytes
        if decision.accepted:
            retained += window_bytes
            accepted_count += 1
        else:
            rejected += window_bytes
        rows.append(
            {
                "index": window.index,
                "start_char": window.start_char,
                "end_char": window.end_char,
                "utf8_bytes": window_bytes,
                "authoritative": True,
                "decision": _decision_dict(decision),
            }
        )
    if retained + rejected != total_bytes:
        raise QualityGranularityError("window byte accounting mismatch")
    if accepted_count == len(rows):
        status = "RETAIN_ALL"
    elif accepted_count:
        status = "RETAIN_PARTIAL"
    else:
        status = "REJECT_DOCUMENT"

    return {
        "record_id": record_id,
        "mode": mode,
        "authoritative_unit": "BOUNDED_NATURAL_LANGUAGE_WINDOW",
        "status": status,
        "family_eviction_authority": False,
        "retained_utf8_bytes": retained,
        "rejected_utf8_bytes": rejected,
        "full_document_diagnostic": _decision_dict(full),
        "windows": rows,
    }


def fixed_diagnostic_packs(text: str, target_chars: int) -> list[str]:
    """DATA-296-style exact bounded packs for non-authoritative comparison only."""
    if target_chars <= 0:
        raise QualityGranularityError("target_chars must be positive")
    return [text[start : start + target_chars] for start in range(0, len(text), target_chars)]


def compare_three_granularities(
    documents: list[dict[str, str]],
    *,
    quality_policy: QualityPolicy | None = None,
    granularity_policy: FrozenGranularityPolicy | None = None,
    diagnostic_pack_chars: int = 2048,
) -> dict[str, Any]:
    """Compare whole-source, source-native-document, and arbitrary-pack decisions."""
    if not documents:
        raise QualityGranularityError("documents must not be empty")
    granularity = granularity_policy or frozen_granularity_policy()
    quality = quality_policy or default_quality_policy()
    modes = {row.get("mode") for row in documents}
    if len(modes) != 1 or next(iter(modes)) not in {"uk", "en", "code"}:
        raise QualityGranularityError("comparison requires one common mode")
    mode = next(iter(modes))
    assert mode in {"uk", "en", "code"}

    whole_text = "".join(row["text"] for row in documents)
    whole = assess_document("whole-source#diagnostic", whole_text, mode, policy=quality)

    doc_rows = []
    pack_rows = []
    frozen_rows = []
    for row in documents:
        document_id = row["id"]
        text = row["text"]
        decision = assess_document(document_id, text, mode, policy=quality)
        doc_rows.append({"id": document_id, "utf8_bytes": len(text.encode("utf-8")), **_decision_dict(decision)})
        for index, pack in enumerate(fixed_diagnostic_packs(text, diagnostic_pack_chars)):
            decision = assess_document(
                f"{document_id}#diagnostic-pack-{index:04d}",
                pack,
                mode,
                policy=quality,
            )
            pack_rows.append(
                {
                    "id": f"{document_id}#diagnostic-pack-{index:04d}",
                    "utf8_bytes": len(pack.encode("utf-8")),
                    **_decision_dict(decision),
                }
            )
        frozen_rows.append(
            apply_frozen_granularity(
                document_id,
                text,
                mode,
                quality_policy=quality,
                granularity_policy=granularity,
            )
        )

    retained = sum(row["retained_utf8_bytes"] for row in frozen_rows)
    rejected = sum(row["rejected_utf8_bytes"] for row in frozen_rows)
    total = len(whole_text.encode("utf-8"))
    if retained + rejected != total:
        raise QualityGranularityError("comparison byte accounting mismatch")

    document_accepted = [row["accepted"] for row in doc_rows]
    pack_accepted = [row["accepted"] for row in pack_rows]
    frozen_has_retention = retained > 0
    disagreement = int(whole.accepted != all(document_accepted))
    disagreement += int(bool(pack_accepted) and (whole.accepted != all(pack_accepted)))
    disagreement += int(whole.accepted != frozen_has_retention)

    return {
        "mode": mode,
        "whole_source": {
            "authoritative": False,
            "utf8_bytes": total,
            **_decision_dict(whole),
        },
        "documents": {
            "authoritative": True,
            "items": doc_rows,
        },
        "bounded_packs": {
            "authoritative": False if mode == "code" else "SALVAGE_ONLY_VIA_FROZEN_POLICY",
            "items": pack_rows,
        },
        "frozen_policy": {
            "policy_id": granularity.policy_id,
            "policy_sha256": granularity.manifest()["policy_sha256"],
            "items": frozen_rows,
        },
        "predeclared_metrics": {
            "retained_unique_utf8_bytes": retained,
            "rejected_unique_utf8_bytes": rejected,
            "document_acceptance_rate": round(sum(document_accepted) / len(document_accepted), 6),
            "bounded_window_acceptance_rate": (
                round(sum(pack_accepted) / len(pack_accepted), 6) if pack_accepted else 0.0
            ),
            "granularity_disagreement_count": disagreement,
            "partition_exact_reconstruction_failures": 0,
            "family_soft_quality_eviction_count": 0,
            "source_total_loss_from_soft_quality_count": int(retained == 0),
            "maximum_soft_quality_blast_radius_utf8_bytes": max(
                (
                    window["utf8_bytes"]
                    for item in frozen_rows
                    for window in item.get("windows", [])
                    if not window["decision"]["accepted"]
                ),
                default=max(
                    (
                        item["rejected_utf8_bytes"]
                        for item in frozen_rows
                        if item["rejected_utf8_bytes"]
                    ),
                    default=0,
                ),
            ),
            "code_authoritative_parse_preservation": "SOURCE_NATIVE_UNIT_ONLY"
            if mode == "code"
            else "NOT_APPLICABLE",
        },
        "model_results_read": False,
        "final_test_outcomes_read": False,
    }
