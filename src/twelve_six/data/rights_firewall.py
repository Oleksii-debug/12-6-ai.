"""Purpose-specific source-object rights firewall.

This module is intentionally stricter than the legacy training eligibility layer.
A public or broadly licensed object receives no project-purpose permission unless
all six project purposes are decided explicitly by an immutable authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

FIREWALL_SCHEMA = "12-6.purpose-rights-firewall.v1"

PURPOSE_TRAINING = "training"
PURPOSE_TOKENIZER_FITTING = "tokenizer-fitting"
PURPOSE_SELECTION_VALIDATION = "selection-validation"
PURPOSE_FINAL_TEST = "final-test"
PURPOSE_REDISTRIBUTION = "redistribution"
PURPOSE_ANALYSIS = "analysis"

PURPOSES = (
    PURPOSE_TRAINING,
    PURPOSE_TOKENIZER_FITTING,
    PURPOSE_SELECTION_VALIDATION,
    PURPOSE_FINAL_TEST,
    PURPOSE_REDISTRIBUTION,
    PURPOSE_ANALYSIS,
)
_EVALUATION_PURPOSES = frozenset({PURPOSE_SELECTION_VALIDATION, PURPOSE_FINAL_TEST})
_ADAPTIVE_PURPOSES = frozenset(
    {PURPOSE_TRAINING, PURPOSE_TOKENIZER_FITTING, PURPOSE_SELECTION_VALIDATION}
)

ALLOW = "ALLOW"
DENY = "DENY"
_DECISIONS = frozenset({ALLOW, DENY})


class PurposeRightsError(ValueError):
    """Raised when a purpose-specific rights or reservation invariant is violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PurposeRightsError(f"{field} must be a non-empty string")
    return value.strip()


def _require_hex(value: Any, field: str, length: int) -> str:
    text = _require_text(value, field)
    if (
        len(text) != length
        or text != text.lower()
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise PurposeRightsError(f"{field} must be lowercase {length}-hex")
    return text


def _parse_utc(value: Any, field: str) -> datetime:
    text = _require_text(value, field)
    if not text.endswith("Z"):
        raise PurposeRightsError(f"{field} must be an explicit UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise PurposeRightsError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PurposeRightsError(f"{field} must be UTC")
    return parsed


@dataclass(frozen=True)
class SourceObjectIdentity:
    """Exact immutable identity of one source object, not a whole licensed corpus."""

    source_id: str
    source_family: str
    upstream_revision: str
    object_path: str
    content_sha256: str
    git_blob_sha1: str | None = None

    def __post_init__(self) -> None:
        for field in ("source_id", "source_family", "upstream_revision", "object_path"):
            _require_text(getattr(self, field), field)
        _require_hex(self.content_sha256, "content_sha256", 64)
        if self.git_blob_sha1 is not None:
            _require_hex(self.git_blob_sha1, "git_blob_sha1", 40)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class PurposeDecisions:
    """Complete project-purpose decision vector. Omission is structurally impossible."""

    training: str
    tokenizer_fitting: str
    selection_validation: str
    final_test: str
    redistribution: str
    analysis: str

    def __post_init__(self) -> None:
        for purpose, value in self.to_dict().items():
            if value not in _DECISIONS:
                raise PurposeRightsError(
                    f"purpose decision {purpose!r} must be explicit ALLOW or DENY"
                )
        if self.selection_validation == ALLOW:
            if self.training != DENY or self.tokenizer_fitting != DENY:
                raise PurposeRightsError(
                    "selection-validation object must deny training and tokenizer fitting"
                )
            if self.final_test != DENY:
                raise PurposeRightsError(
                    "selection-validation and final-test may not share one source object"
                )
        if self.final_test == ALLOW:
            if (
                self.training != DENY
                or self.tokenizer_fitting != DENY
                or self.selection_validation != DENY
            ):
                raise PurposeRightsError(
                    "final-test object must deny training, tokenizer fitting, and selection"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            PURPOSE_TRAINING: self.training,
            PURPOSE_TOKENIZER_FITTING: self.tokenizer_fitting,
            PURPOSE_SELECTION_VALIDATION: self.selection_validation,
            PURPOSE_FINAL_TEST: self.final_test,
            PURPOSE_REDISTRIBUTION: self.redistribution,
            PURPOSE_ANALYSIS: self.analysis,
        }

    def for_purpose(self, purpose: str) -> str:
        requested = _require_text(purpose, "purpose")
        if requested not in PURPOSES:
            raise PurposeRightsError(f"unknown project purpose: {requested!r}")
        return self.to_dict()[requested]


@dataclass(frozen=True)
class EvaluationReservation:
    """Immutable reservation made before an object can serve an evaluation purpose."""

    reserved_purposes: tuple[str, ...]
    reserved_at_utc: str
    reservation_commit_sha: str

    def __post_init__(self) -> None:
        if not isinstance(self.reserved_purposes, tuple) or not self.reserved_purposes:
            raise PurposeRightsError("reserved_purposes must be a non-empty tuple")
        if any(purpose not in _EVALUATION_PURPOSES for purpose in self.reserved_purposes):
            raise PurposeRightsError(
                "only selection-validation and final-test may be evaluation reservations"
            )
        if len(set(self.reserved_purposes)) != len(self.reserved_purposes):
            raise PurposeRightsError("reserved_purposes must not contain duplicates")
        if self.reserved_purposes != tuple(sorted(self.reserved_purposes)):
            raise PurposeRightsError("reserved_purposes must be canonical sorted order")
        _parse_utc(self.reserved_at_utc, "reserved_at_utc")
        _require_hex(self.reservation_commit_sha, "reservation_commit_sha", 40)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reserved_purposes"] = list(self.reserved_purposes)
        return value


@dataclass(frozen=True)
class PurposeRightsAuthority:
    """One immutable rights decision for one exact source object."""

    authority_id: str
    authority_commit_sha: str
    issued_at_utc: str
    source_object: SourceObjectIdentity
    upstream_license_id: str
    rights_evidence_sha256: str
    project_decision_ref: str
    decisions: PurposeDecisions
    reservation: EvaluationReservation | None = None
    successor_of_authority_id: str | None = None

    def __post_init__(self) -> None:
        for field in ("authority_id", "upstream_license_id", "project_decision_ref"):
            _require_text(getattr(self, field), field)
        _require_hex(self.authority_commit_sha, "authority_commit_sha", 40)
        _parse_utc(self.issued_at_utc, "issued_at_utc")
        _require_hex(self.rights_evidence_sha256, "rights_evidence_sha256", 64)
        if not isinstance(self.source_object, SourceObjectIdentity):
            raise PurposeRightsError("source_object must be SourceObjectIdentity")
        if not isinstance(self.decisions, PurposeDecisions):
            raise PurposeRightsError("decisions must be PurposeDecisions")
        if self.reservation is not None and not isinstance(
            self.reservation, EvaluationReservation
        ):
            raise PurposeRightsError("reservation must be EvaluationReservation or null")
        if self.successor_of_authority_id is not None:
            _require_text(self.successor_of_authority_id, "successor_of_authority_id")

        allowed_evaluation = tuple(
            sorted(
                purpose
                for purpose in _EVALUATION_PURPOSES
                if self.decisions.for_purpose(purpose) == ALLOW
            )
        )
        if allowed_evaluation:
            if self.reservation is None:
                raise PurposeRightsError(
                    "evaluation ALLOW requires an exact pre-use reservation timestamp and commit"
                )
            if self.reservation.reserved_purposes != allowed_evaluation:
                raise PurposeRightsError(
                    "reservation purposes must exactly equal evaluation purposes allowed"
                )
            if _parse_utc(
                self.reservation.reserved_at_utc, "reserved_at_utc"
            ) > _parse_utc(self.issued_at_utc, "issued_at_utc"):
                raise PurposeRightsError("reservation cannot postdate its rights authority")
        elif self.reservation is not None:
            raise PurposeRightsError(
                "evaluation reservation is invalid when no evaluation purpose is allowed"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "authority_commit_sha": self.authority_commit_sha,
            "issued_at_utc": self.issued_at_utc,
            "source_object": self.source_object.to_dict(),
            "source_object_identity_sha256": self.source_object.identity_sha256,
            "upstream_license_id": self.upstream_license_id,
            "rights_evidence_sha256": self.rights_evidence_sha256,
            "project_decision_ref": self.project_decision_ref,
            "decisions": self.decisions.to_dict(),
            "reservation": None if self.reservation is None else self.reservation.to_dict(),
            "successor_of_authority_id": self.successor_of_authority_id,
        }

    @property
    def rights_state_sha256(self) -> str:
        state = {
            "source_object_identity_sha256": self.source_object.identity_sha256,
            "upstream_license_id": self.upstream_license_id,
            "rights_evidence_sha256": self.rights_evidence_sha256,
            "project_decision_ref": self.project_decision_ref,
            "decisions": self.decisions.to_dict(),
            "reservation": None if self.reservation is None else self.reservation.to_dict(),
        }
        return _sha256(state)

    @property
    def authority_identity_sha256(self) -> str:
        return _sha256(self.to_dict())


class PurposeRightsFirewall:
    """Append-only resolver enforcing purpose separation and irreversible reservations."""

    def __init__(self) -> None:
        self._by_authority_id: dict[str, PurposeRightsAuthority] = {}
        self._history: dict[str, list[PurposeRightsAuthority]] = {}

    def register(self, authority: PurposeRightsAuthority) -> None:
        if not isinstance(authority, PurposeRightsAuthority):
            raise PurposeRightsError("authority must be PurposeRightsAuthority")
        if authority.authority_id in self._by_authority_id:
            raise PurposeRightsError(f"duplicate authority_id: {authority.authority_id}")

        object_id = authority.source_object.identity_sha256
        history = self._history.setdefault(object_id, [])
        previous = history[-1] if history else None

        if previous is None:
            if authority.successor_of_authority_id is not None:
                raise PurposeRightsError(
                    "first authority for an exact source object cannot name a successor parent"
                )
        else:
            rights_changed = authority.rights_state_sha256 != previous.rights_state_sha256
            if rights_changed:
                if authority.successor_of_authority_id != previous.authority_id:
                    raise PurposeRightsError(
                        "rights changes require a successor authority naming the current authority"
                    )
                if _parse_utc(
                    authority.issued_at_utc, "issued_at_utc"
                ) <= _parse_utc(previous.issued_at_utc, "previous.issued_at_utc"):
                    raise PurposeRightsError(
                        "successor authority must be issued after its predecessor"
                    )
                if authority.authority_commit_sha == previous.authority_commit_sha:
                    raise PurposeRightsError(
                        "successor authority must have a distinct immutable commit"
                    )

            historically_reserved = {
                purpose
                for item in history
                if item.reservation is not None
                for purpose in item.reservation.reserved_purposes
            }
            if PURPOSE_SELECTION_VALIDATION in historically_reserved:
                if (
                    authority.decisions.training == ALLOW
                    or authority.decisions.tokenizer_fitting == ALLOW
                ):
                    raise PurposeRightsError(
                        "selection-reserved object can never later enter training/tokenizer fitting"
                    )
            if PURPOSE_FINAL_TEST in historically_reserved:
                if (
                    authority.decisions.training == ALLOW
                    or authority.decisions.tokenizer_fitting == ALLOW
                    or authority.decisions.selection_validation == ALLOW
                ):
                    raise PurposeRightsError(
                        "final-test-reserved object can never enter adaptive purposes"
                    )

        history.append(authority)
        self._by_authority_id[authority.authority_id] = authority

    def current(self, source_object: SourceObjectIdentity) -> PurposeRightsAuthority:
        object_id = source_object.identity_sha256
        history = self._history.get(object_id)
        if not history:
            raise PurposeRightsError(
                f"no purpose-specific authority for source object {object_id}"
            )
        return history[-1]

    def require_use(
        self, source_object: SourceObjectIdentity, purpose: str
    ) -> PurposeRightsAuthority:
        authority = self.current(source_object)
        requested = _require_text(purpose, "purpose")
        if authority.decisions.for_purpose(requested) != ALLOW:
            raise PurposeRightsError(
                f"{authority.authority_id}: {requested} denied for exact source object"
            )
        return authority

    def assert_observation_may_influence(
        self,
        source_object: SourceObjectIdentity,
        *,
        observed_under: str,
        target_purpose: str,
    ) -> None:
        self.require_use(source_object, observed_under)
        target = _require_text(target_purpose, "target_purpose")
        if target not in PURPOSES:
            raise PurposeRightsError(f"unknown target purpose: {target!r}")
        if observed_under == PURPOSE_FINAL_TEST and target in _ADAPTIVE_PURPOSES:
            raise PurposeRightsError(
                "final-test payloads, outcomes, or observations may not influence "
                "training, tokenizer fitting, or selection-validation"
            )

    def history(self, source_object: SourceObjectIdentity) -> tuple[PurposeRightsAuthority, ...]:
        return tuple(self._history.get(source_object.identity_sha256, ()))


def purpose_decisions_from_mapping(data: Any) -> PurposeDecisions:
    if not isinstance(data, dict):
        raise PurposeRightsError("decisions must be an object")
    if set(data) != set(PURPOSES):
        missing = sorted(set(PURPOSES) - set(data))
        extra = sorted(set(data) - set(PURPOSES))
        raise PurposeRightsError(
            f"decisions must contain exactly all six project purposes; "
            f"missing={missing}, extra={extra}"
        )
    return PurposeDecisions(
        training=data[PURPOSE_TRAINING],
        tokenizer_fitting=data[PURPOSE_TOKENIZER_FITTING],
        selection_validation=data[PURPOSE_SELECTION_VALIDATION],
        final_test=data[PURPOSE_FINAL_TEST],
        redistribution=data[PURPOSE_REDISTRIBUTION],
        analysis=data[PURPOSE_ANALYSIS],
    )
